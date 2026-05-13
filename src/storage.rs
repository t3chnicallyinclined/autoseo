use anyhow::Context;
use rusqlite::Connection;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Storage {
    conn: Arc<Mutex<Connection>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Transcribed,
    Ranked,
    Rendered,
    Posted,
    Done,
    Failed,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Pending => "pending",
            JobStatus::Transcribed => "transcribed",
            JobStatus::Ranked => "ranked",
            JobStatus::Rendered => "rendered",
            JobStatus::Posted => "posted",
            JobStatus::Done => "done",
            JobStatus::Failed => "failed",
        }
    }
}

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    show_slug       TEXT,
    media_name      TEXT,
    drive_file_id   TEXT,
    status          TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    cost_cents      INTEGER NOT NULL DEFAULT 0,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_show_slug ON jobs(show_slug);

CREATE TABLE IF NOT EXISTS clips (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    rank            INTEGER,
    score           REAL,
    hook            TEXT,
    reasoning_json  TEXT,
    trend_match     TEXT
);

CREATE INDEX IF NOT EXISTS idx_clips_job_id ON clips(job_id);
CREATE INDEX IF NOT EXISTS idx_clips_rank ON clips(job_id, rank);

CREATE TABLE IF NOT EXISTS clip_renders (
    clip_id         TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    variant         TEXT NOT NULL,
    path            TEXT NOT NULL,
    bytes           INTEGER,
    duration_ms     INTEGER,
    PRIMARY KEY (clip_id, variant)
);

CREATE TABLE IF NOT EXISTS posts (
    clip_id         TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    platform        TEXT NOT NULL,
    status          TEXT NOT NULL,
    external_id     TEXT,
    external_url    TEXT,
    posted_at       INTEGER,
    error           TEXT,
    PRIMARY KEY (clip_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status);

CREATE TABLE IF NOT EXISTS analytics (
    clip_id         TEXT NOT NULL,
    platform        TEXT NOT NULL,
    fetched_at      INTEGER NOT NULL,
    views           INTEGER,
    ctr             REAL,
    watch_pct       REAL,
    PRIMARY KEY (clip_id, platform, fetched_at)
);

CREATE TABLE IF NOT EXISTS trends (
    source          TEXT NOT NULL,
    topic_id        TEXT NOT NULL,
    label           TEXT,
    score           REAL,
    fetched_at      INTEGER NOT NULL,
    PRIMARY KEY (source, topic_id, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_trends_recent ON trends(source, fetched_at DESC);
"#;

impl Storage {
    pub async fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let conn = tokio::task::spawn_blocking(move || -> anyhow::Result<Connection> {
            let conn = Connection::open(&path)
                .with_context(|| format!("open sqlite at {}", path.display()))?;
            conn.pragma_update(None, "journal_mode", "WAL")
                .context("set journal_mode=WAL")?;
            conn.pragma_update(None, "foreign_keys", "ON")
                .context("enable foreign keys")?;
            migrate(&conn)?;
            Ok(conn)
        })
        .await
        .context("join sqlite open")??;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Compatibility: import a flat newline-delimited dedupe file as completed jobs.
    /// Returns the count of newly-imported message IDs (existing rows are left alone).
    pub async fn import_legacy_dedupe(&self, path: impl AsRef<Path>) -> anyhow::Result<usize> {
        let path = path.as_ref().to_path_buf();
        let contents = match tokio::fs::read_to_string(&path).await {
            Ok(s) => s,
            Err(_) => return Ok(0),
        };
        let ids: Vec<String> = contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        if ids.is_empty() {
            return Ok(0);
        }

        let conn = self.conn.clone();
        let imported = tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let mut conn = conn.blocking_lock();
            let tx = conn.transaction().context("begin tx")?;
            let now = unix_now();
            let mut count = 0usize;
            {
                let mut stmt = tx
                    .prepare(
                        "INSERT OR IGNORE INTO jobs \
                         (id, status, created_at, updated_at) \
                         VALUES (?1, 'done', ?2, ?2)",
                    )
                    .context("prepare legacy import")?;
                for id in &ids {
                    let changed = stmt.execute((id, now)).context("insert legacy job")?;
                    count += changed;
                }
            }
            tx.commit().context("commit legacy import")?;
            Ok(count)
        })
        .await
        .context("join legacy import")??;

        Ok(imported)
    }

    /// Mark a Gmail message as processed (status='done'). Idempotent.
    /// Drop-in replacement for the legacy `FileBackedDedupe::insert`.
    pub async fn mark_processed(&self, message_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let message_id = message_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            conn.execute(
                "INSERT INTO jobs (id, status, created_at, updated_at) \
                 VALUES (?1, 'done', ?2, ?2) \
                 ON CONFLICT(id) DO UPDATE SET status = 'done', updated_at = excluded.updated_at",
                (message_id, now),
            )
            .context("mark_processed upsert")?;
            Ok(())
        })
        .await
        .context("join mark_processed")??;
        Ok(())
    }

    /// Clone the inner `Arc<Mutex<Connection>>` for use in
    /// `spawn_blocking` calls from the dashboard repo layer. All DB
    /// access must still go through `blocking_lock()` inside a blocking
    /// task — this is just a convenience to keep call sites tight.
    pub(crate) fn conn(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    /// Has this gmail message_id been seen before (any status)?
    pub async fn job_exists(&self, message_id: &str) -> anyhow::Result<bool> {
        let conn = self.conn.clone();
        let message_id = message_id.to_string();
        let exists = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            let conn = conn.blocking_lock();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(1) FROM jobs WHERE id = ?1",
                    [message_id],
                    |r| r.get(0),
                )
                .context("query job_exists")?;
            Ok(n > 0)
        })
        .await
        .context("join job_exists")??;

        Ok(exists)
    }
}

fn migrate(conn: &Connection) -> anyhow::Result<()> {
    let version: u32 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .context("read user_version")?;
    if version < 1 {
        conn.execute_batch(SCHEMA_V1).context("apply schema v1")?;
        conn.pragma_update(None, "user_version", 1)
            .context("set user_version=1")?;
    }
    if version < 2 {
        apply_v2(conn).context("apply schema v2")?;
        conn.pragma_update(None, "user_version", 2)
            .context("set user_version=2")?;
    }
    Ok(())
}

/// Schema v2 — dashboard tables + workspace_id extensions to v1 tables.
/// See [/home/tris/.claude/plans/ok-great-lets-write-zazzy-hellman.md] for design.
fn apply_v2(conn: &Connection) -> anyhow::Result<()> {
    // ALTER TABLE … ADD COLUMN is per-statement and not idempotent on older
    // SQLite. apply_alter_safe swallows "duplicate column name" so re-runs
    // are no-ops.
    for stmt in SCHEMA_V2_ALTERS {
        apply_alter_safe(conn, stmt)?;
    }
    conn.execute_batch(SCHEMA_V2_NEW_TABLES)
        .context("apply v2 new tables")?;

    // Seed the default workspace if no rows exist. v1 always uses 'ws_default';
    // v2 (SaaS) will create real workspaces on signup.
    let count: i64 = conn
        .query_row("SELECT COUNT(1) FROM workspaces", [], |r| r.get(0))
        .context("count workspaces")?;
    if count == 0 {
        let now = unix_now();
        conn.execute(
            "INSERT INTO workspaces (id, slug, name, created_at, updated_at) \
             VALUES ('ws_default', 'default', 'Default Workspace', ?1, ?1)",
            [now],
        )
        .context("seed ws_default workspace")?;
    }
    Ok(())
}

/// Run an `ALTER TABLE … ADD COLUMN` statement. SQLite ≤ 3.45 doesn't support
/// `IF NOT EXISTS` on ADD COLUMN, so we swallow the specific "duplicate column
/// name" error to make migrations idempotent across re-runs.
fn apply_alter_safe(conn: &Connection, sql: &str) -> anyhow::Result<()> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg)))
            if msg.contains("duplicate column name") =>
        {
            Ok(())
        }
        Err(e) => Err(e).with_context(|| format!("apply_alter_safe: {sql}")),
    }
}

const SCHEMA_V2_ALTERS: &[&str] = &[
    // Existing v1 tables get workspace_id + dashboard-aware columns.
    "ALTER TABLE jobs ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'ws_default'",
    "ALTER TABLE jobs ADD COLUMN show_id TEXT",
    "ALTER TABLE jobs ADD COLUMN source_kind TEXT",
    "ALTER TABLE jobs ADD COLUMN source_ref TEXT",
    "ALTER TABLE jobs ADD COLUMN clips_dir TEXT",
    "ALTER TABLE jobs ADD COLUMN duration_secs REAL",
    "ALTER TABLE jobs ADD COLUMN manifest_json TEXT",
    "ALTER TABLE clips ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'ws_default'",
    "ALTER TABLE clips ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'pending'",
    "ALTER TABLE clips ADD COLUMN overlay_hook TEXT",
    "ALTER TABLE clips ADD COLUMN social_json TEXT",
    "ALTER TABLE clips ADD COLUMN edit_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE clips ADD COLUMN last_edited_at INTEGER",
    "ALTER TABLE clips ADD COLUMN last_edited_by TEXT",
    "ALTER TABLE posts ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'ws_default'",
    "ALTER TABLE posts ADD COLUMN skipped_by_user INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE posts ADD COLUMN schedule_id TEXT",
    "ALTER TABLE posts ADD COLUMN updated_at INTEGER",
];

const SCHEMA_V2_NEW_TABLES: &str = r#"
CREATE TABLE IF NOT EXISTS workspaces (
    id              TEXT PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_workspace ON jobs(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_clips_workspace ON clips(workspace_id, job_id);
CREATE INDEX IF NOT EXISTS idx_clips_approval  ON clips(workspace_id, approval_status);
CREATE INDEX IF NOT EXISTS idx_posts_workspace ON posts(workspace_id, status);

CREATE TABLE IF NOT EXISTS shows (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    slug            TEXT NOT NULL,
    name            TEXT NOT NULL,
    clip_top_k      INTEGER NOT NULL DEFAULT 10,
    render_formats  TEXT    NOT NULL DEFAULT '9x16,1x1,16x9',
    vlm_rerank      INTEGER NOT NULL DEFAULT 0,
    youtube_privacy TEXT    NOT NULL DEFAULT 'unlisted',
    prompt_overrides_json   TEXT,
    default_post_platforms  TEXT NOT NULL DEFAULT '',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE(workspace_id, slug)
);
CREATE INDEX IF NOT EXISTS idx_shows_workspace ON shows(workspace_id);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    email           TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    display_name    TEXT,
    role            TEXT NOT NULL DEFAULT 'admin',
    created_at      INTEGER NOT NULL,
    last_login_at   INTEGER,
    UNIQUE(workspace_id, email)
);
CREATE INDEX IF NOT EXISTS idx_users_workspace ON users(workspace_id);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      INTEGER NOT NULL,
    expires_at      INTEGER NOT NULL,
    last_seen_at    INTEGER NOT NULL,
    user_agent      TEXT,
    ip              TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS credentials (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    platform        TEXT NOT NULL,
    profile_name    TEXT NOT NULL DEFAULT 'default',
    ciphertext      TEXT NOT NULL,
    last_test_at    INTEGER,
    last_test_ok    INTEGER NOT NULL DEFAULT 0,
    last_test_msg   TEXT,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE(workspace_id, platform, profile_name)
);

CREATE TABLE IF NOT EXISTS clip_edits (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    clip_id         TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    user_id         TEXT REFERENCES users(id),
    patch_json      TEXT NOT NULL,
    resolved_json   TEXT NOT NULL,
    needs_rerender  INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clip_edits_clip ON clip_edits(clip_id, created_at DESC);

CREATE TABLE IF NOT EXISTS clip_history (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    clip_id         TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    edit_id         TEXT REFERENCES clip_edits(id) ON DELETE SET NULL,
    field_path      TEXT NOT NULL,
    before_value    TEXT,
    after_value     TEXT,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clip_history_clip ON clip_history(clip_id, created_at DESC);

CREATE TABLE IF NOT EXISTS schedules (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    clip_id         TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    platform        TEXT NOT NULL,
    scheduled_at    INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempt         INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    last_attempt_at INTEGER,
    external_url    TEXT,
    created_by      TEXT REFERENCES users(id),
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_schedules_due  ON schedules(workspace_id, status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_schedules_clip ON schedules(clip_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    user_id         TEXT REFERENCES users(id),
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    target_kind     TEXT,
    target_id       TEXT,
    metadata_json   TEXT,
    ip              TEXT,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(workspace_id, target_kind, target_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_recent ON audit_log(workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS clip_embeddings (
    clip_id         TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    workspace_id    TEXT NOT NULL,
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    embedding       BLOB NOT NULL,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (clip_id, model)
);
CREATE INDEX IF NOT EXISTS idx_clip_embeddings_workspace ON clip_embeddings(workspace_id);

CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    topic           TEXT NOT NULL,
    target_kind     TEXT,
    target_id       TEXT,
    payload_json    TEXT,
    created_at      INTEGER NOT NULL,
    consumed_at     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_workspace_topic ON events(workspace_id, topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_unconsumed
    ON events(workspace_id, created_at) WHERE consumed_at IS NULL;

CREATE VIEW IF NOT EXISTS analytics_latest AS
SELECT clip_id, platform, MAX(fetched_at) AS fetched_at, views, ctr, watch_pct
FROM analytics
GROUP BY clip_id, platform;
"#;

fn unix_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn migration_applies_v1() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let _storage = Storage::open(&db_path).await?;

        // Reopen — migration should be idempotent.
        let _storage2 = Storage::open(&db_path).await?;
        Ok(())
    }

    #[tokio::test]
    async fn legacy_dedupe_imports() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let dedupe_path = dir.path().join("dedupe.txt");
        tokio::fs::write(&dedupe_path, "msg_a\nmsg_b\n\nmsg_c\n").await?;

        let storage = Storage::open(&db_path).await?;
        let imported = storage.import_legacy_dedupe(&dedupe_path).await?;
        assert_eq!(imported, 3, "expected 3 legacy IDs imported");

        assert!(storage.job_exists("msg_a").await?);
        assert!(storage.job_exists("msg_b").await?);
        assert!(storage.job_exists("msg_c").await?);
        assert!(!storage.job_exists("msg_d").await?);

        // Idempotent re-import.
        let imported2 = storage.import_legacy_dedupe(&dedupe_path).await?;
        assert_eq!(imported2, 0, "second import should be a no-op");

        Ok(())
    }

    #[tokio::test]
    async fn mark_processed_is_idempotent() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let storage = Storage::open(&db_path).await?;

        assert!(!storage.job_exists("msg_x").await?);
        storage.mark_processed("msg_x").await?;
        assert!(storage.job_exists("msg_x").await?);
        // Second call must not error.
        storage.mark_processed("msg_x").await?;
        assert!(storage.job_exists("msg_x").await?);
        Ok(())
    }

    #[tokio::test]
    async fn migration_applies_v2() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let storage = Storage::open(&db_path).await?;

        // user_version must be at 2 after migration.
        let conn = storage.conn();
        let (version, ws_count, has_audit, has_events, jobs_has_workspace) =
            tokio::task::spawn_blocking(move || -> anyhow::Result<(u32, i64, bool, bool, bool)> {
                let conn = conn.blocking_lock();
                let v: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
                let n: i64 = conn.query_row("SELECT COUNT(1) FROM workspaces", [], |r| r.get(0))?;
                let audit: i64 = conn.query_row(
                    "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='audit_log'",
                    [],
                    |r| r.get(0),
                )?;
                let events: i64 = conn.query_row(
                    "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='events'",
                    [],
                    |r| r.get(0),
                )?;
                let cols: i64 = conn.query_row(
                    "SELECT COUNT(1) FROM pragma_table_info('jobs') WHERE name='workspace_id'",
                    [],
                    |r| r.get(0),
                )?;
                Ok((v, n, audit > 0, events > 0, cols > 0))
            })
            .await??;

        assert_eq!(version, 2, "schema should be at v2");
        assert_eq!(ws_count, 1, "ws_default should be seeded once");
        assert!(has_audit, "audit_log table should exist");
        assert!(has_events, "events table should exist");
        assert!(jobs_has_workspace, "jobs.workspace_id column should exist");

        // Idempotent re-open.
        let _storage2 = Storage::open(&db_path).await?;
        Ok(())
    }

    #[tokio::test]
    async fn import_missing_file_is_noop() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let storage = Storage::open(&db_path).await?;
        let imported = storage
            .import_legacy_dedupe(dir.path().join("missing.txt"))
            .await?;
        assert_eq!(imported, 0);
        Ok(())
    }
}
