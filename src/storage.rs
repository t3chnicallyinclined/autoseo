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
    Ok(())
}

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
