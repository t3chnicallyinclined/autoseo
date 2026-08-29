use anyhow::Context;
use rusqlite::{Connection, OptionalExtension};
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
    /// User-cancelled before the worker reached the job. Terminal.
    /// Mid-flight cancellation is not yet implemented; only `pending` jobs
    /// can be cancelled today.
    Cancelled,
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
            JobStatus::Cancelled => "cancelled",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(JobStatus::Pending),
            "transcribed" => Some(JobStatus::Transcribed),
            "ranked" => Some(JobStatus::Ranked),
            "rendered" => Some(JobStatus::Rendered),
            "posted" => Some(JobStatus::Posted),
            "done" => Some(JobStatus::Done),
            "failed" => Some(JobStatus::Failed),
            "cancelled" => Some(JobStatus::Cancelled),
            _ => None,
        }
    }
}

/// A row from the clips + analytics join — used to feed historical
/// performance data back into the ranker prompt.
#[derive(Debug, Clone)]
pub struct ClipPerformanceRow {
    pub clip_id: String,
    pub hook: Option<String>,
    pub score: Option<f64>,
    pub rank: Option<i64>,
    pub views: Option<i64>,
    pub ctr: Option<f64>,
    pub watch_pct: Option<f64>,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// A row from the `jobs` table.
#[derive(Debug, Clone)]
pub struct JobRow {
    pub id: String,
    pub show_slug: Option<String>,
    pub media_name: Option<String>,
    pub drive_file_id: Option<String>,
    pub status: JobStatus,
    pub retry_count: i64,
    pub cost_cents: i64,
    pub error: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    /// Path to the source video on the operator's filesystem (set when the
    /// job was enqueued from the dashboard's New Job UI).
    pub local_path: Option<String>,
    /// Original URL the source was fetched from (Drive share link, direct
    /// HTTPS) — for audit / re-download.
    pub source_url: Option<String>,
    /// Per-job JSON config overrides: clip_top_k, render_formats, mode_tag,
    /// skip_ranges, etc. Parsed by the worker before running the pipeline.
    pub config_json: Option<String>,
}

/// A row from the `trends` table.
#[derive(Debug, Clone)]
pub struct TrendRow {
    pub source: String,
    pub topic_id: String,
    pub label: Option<String>,
    pub score: Option<f64>,
    pub fetched_at: i64,
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

const SCHEMA_V3: &str = r#"
CREATE TABLE IF NOT EXISTS show_loudness (
    show_slug       TEXT PRIMARY KEY,
    integrated_lufs REAL NOT NULL,
    measured_at     INTEGER NOT NULL,
    episode_count   INTEGER NOT NULL DEFAULT 1
);
"#;

const SCHEMA_V4: &str = r#"
ALTER TABLE clips ADD COLUMN status TEXT NOT NULL DEFAULT 'generated';
ALTER TABLE clips ADD COLUMN social_copy_json TEXT;
"#;

/// A clip row with its renders and posts joined.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClipDetail {
    pub id: String,
    pub job_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub rank: Option<i64>,
    pub score: Option<f64>,
    pub hook: Option<String>,
    pub reasoning_json: Option<String>,
    pub cover_path: Option<String>,
    pub cover_url: Option<String>,
    pub trend_match: Option<String>,
    /// Agency-style hook formula classification set by the ranker LLM.
    /// One of: `contrarian`, `open_loop`, `specific_number`, `pov`,
    /// `confession`, `pattern_interrupt`, `list_teaser`, `question`,
    /// `literal_reaction`, `story`, `other`. `None` for older rows
    /// written before SCHEMA_V8 or by an LLM that omitted the field.
    pub hook_type: Option<String>,
    pub status: String,
    pub social_copy_json: Option<String>,
    pub renders: Vec<ClipRenderRow>,
    pub posts: Vec<PostRow>,
    pub job: Option<JobSummary>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ClipRenderRow {
    pub clip_id: String,
    pub variant: String,
    pub path: String,
    pub bytes: Option<i64>,
    pub duration_ms: Option<i64>,
    /// Public URL when the render has been uploaded to object storage
    /// (R2 / S3-compatible). `None` means the file lives only on local disk
    /// at `path` and must be served via the `/media/clipper/*` proxy.
    pub url: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PostRow {
    pub clip_id: String,
    pub platform: String,
    pub status: String,
    pub external_id: Option<String>,
    pub external_url: Option<String>,
    pub posted_at: Option<i64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JobSummary {
    pub id: String,
    pub show_slug: Option<String>,
    pub media_name: Option<String>,
    pub status: String,
}

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

    /// Open an in-memory SQLite database (for tests).
    #[cfg(test)]
    pub fn open_in_memory_sync() -> Self {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        conn.pragma_update(None, "foreign_keys", "ON").ok();
        migrate(&conn).expect("migrate in-memory db");
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// Expose the raw connection for cross-module access (used by the
    /// dashboard stub endpoints to run ad-hoc queries that don't have a
    /// dedicated typed method yet).
    pub fn conn(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    /// Alias preserved for older test code that used the explicit name.
    #[cfg(test)]
    pub fn conn_for_test(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
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

    /// Get the stored integrated loudness (LUFS) for a show slug.
    /// Returns `None` if no loudness has been stored yet.
    pub async fn get_show_loudness(&self, show_slug: &str) -> anyhow::Result<Option<f64>> {
        let conn = self.conn.clone();
        let slug = show_slug.to_string();
        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Option<f64>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare("SELECT integrated_lufs FROM show_loudness WHERE show_slug = ?1")
                .context("prepare get_show_loudness")?;
            let mut rows = stmt
                .query_map([&slug], |r| r.get::<_, f64>(0))
                .context("query get_show_loudness")?;
            match rows.next() {
                Some(Ok(lufs)) => Ok(Some(lufs)),
                Some(Err(e)) => Err(e.into()),
                None => Ok(None),
            }
        })
        .await
        .context("join get_show_loudness")??;
        Ok(result)
    }

    /// Store or update the integrated loudness (LUFS) for a show slug.
    /// On first call, inserts with episode_count=1. On subsequent calls,
    /// updates with a running average and increments episode_count.
    pub async fn set_show_loudness(
        &self,
        show_slug: &str,
        integrated_lufs: f64,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let slug = show_slug.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            conn.execute(
                "INSERT INTO show_loudness (show_slug, integrated_lufs, measured_at, episode_count) \
                 VALUES (?1, ?2, ?3, 1) \
                 ON CONFLICT(show_slug) DO UPDATE SET \
                   integrated_lufs = (show_loudness.integrated_lufs * show_loudness.episode_count + excluded.integrated_lufs) \
                     / (show_loudness.episode_count + 1), \
                   episode_count = show_loudness.episode_count + 1, \
                   measured_at = excluded.measured_at",
                rusqlite::params![slug, integrated_lufs, now],
            )
            .context("upsert show_loudness")?;
            Ok(())
        })
        .await
        .context("join set_show_loudness")??;
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

    /// Create a new job row with status=pending. If the job already exists, this is a no-op.
    pub async fn create_job(
        &self,
        job_id: &str,
        show_slug: Option<&str>,
        media_name: Option<&str>,
        drive_file_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let show_slug = show_slug.map(str::to_string);
        let media_name = media_name.map(str::to_string);
        let drive_file_id = drive_file_id.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            conn.execute(
                "INSERT OR IGNORE INTO jobs \
                 (id, show_slug, media_name, drive_file_id, status, retry_count, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, 'pending', 0, ?5, ?5)",
                rusqlite::params![job_id, show_slug, media_name, drive_file_id, now],
            )
            .context("create_job insert")?;
            Ok(())
        })
        .await
        .context("join create_job")??;
        Ok(())
    }

    /// Enqueue a new job from the dashboard, recording its source (local
    /// file path and/or URL) and any per-job overrides as JSON. Inserts with
    /// status='pending' so the background worker picks it up.
    pub async fn enqueue_job(
        &self,
        job_id: &str,
        show_slug: Option<&str>,
        media_name: Option<&str>,
        local_path: Option<&str>,
        source_url: Option<&str>,
        config_json: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let show_slug = show_slug.map(str::to_string);
        let media_name = media_name.map(str::to_string);
        let local_path = local_path.map(str::to_string);
        let source_url = source_url.map(str::to_string);
        let config_json = config_json.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            conn.execute(
                "INSERT INTO jobs \
                 (id, show_slug, media_name, drive_file_id, status, retry_count, \
                  created_at, updated_at, local_path, source_url, config_json) \
                 VALUES (?1, ?2, ?3, NULL, 'pending', 0, ?4, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    job_id,
                    show_slug,
                    media_name,
                    now,
                    local_path,
                    source_url,
                    config_json
                ],
            )
            .context("enqueue_job insert")?;
            Ok(())
        })
        .await
        .context("join enqueue_job")??;
        Ok(())
    }

    /// Update `local_path` on a job — used by the worker after downloading a
    /// URL-only source so the path is recorded for retries / debugging.
    pub async fn update_job_local_path(
        &self,
        job_id: &str,
        local_path: &str,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let local_path = local_path.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "UPDATE jobs SET local_path = ?1, updated_at = ?2 WHERE id = ?3",
                rusqlite::params![local_path, unix_now(), job_id],
            )
            .context("update_job_local_path")?;
            Ok(())
        })
        .await
        .context("join update_job_local_path")??;
        Ok(())
    }

    /// Atomically claim the oldest pending job and mark it as running.
    /// Returns `None` when no job is pending. Designed for a single worker;
    /// runs the SELECT + UPDATE inside the same blocking section so two
    /// concurrent callers can't pick the same job.
    pub async fn claim_next_pending_job(&self) -> anyhow::Result<Option<JobRow>> {
        let conn = self.conn.clone();
        let row = tokio::task::spawn_blocking(move || -> anyhow::Result<Option<JobRow>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn.prepare(
                "SELECT id, show_slug, media_name, drive_file_id, status, \
                 retry_count, cost_cents, error, created_at, updated_at, \
                 local_path, source_url, config_json \
                 FROM jobs WHERE status = 'pending' \
                 ORDER BY created_at ASC LIMIT 1",
            )?;
            let row: Option<JobRow> = stmt
                .query_row([], |r| {
                    Ok(JobRow {
                        id: r.get(0)?,
                        show_slug: r.get(1)?,
                        media_name: r.get(2)?,
                        drive_file_id: r.get(3)?,
                        status: JobStatus::from_str(&r.get::<_, String>(4)?)
                            .unwrap_or(JobStatus::Pending),
                        retry_count: r.get(5)?,
                        cost_cents: r.get(6)?,
                        error: r.get(7)?,
                        created_at: r.get(8)?,
                        updated_at: r.get(9)?,
                        local_path: r.get(10)?,
                        source_url: r.get(11)?,
                        config_json: r.get(12)?,
                    })
                })
                .optional()?;
            if let Some(ref r) = row {
                conn.execute(
                    "UPDATE jobs SET status = 'transcribed', updated_at = ?1 WHERE id = ?2 AND status = 'pending'",
                    rusqlite::params![unix_now(), &r.id],
                )?;
            }
            Ok(row)
        })
        .await
        .context("join claim_next_pending_job")??;
        Ok(row)
    }

    /// Transition a job to a new status, updating `updated_at`. On failure status,
    /// also stores the error message.
    pub async fn update_job_status(
        &self,
        job_id: &str,
        status: JobStatus,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let error = error.map(str::to_string);
        let status_str = status.as_str().to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            conn.execute(
                "UPDATE jobs SET status = ?1, error = ?2, updated_at = ?3 WHERE id = ?4",
                rusqlite::params![status_str, error, now, job_id],
            )
            .context("update_job_status")?;
            Ok(())
        })
        .await
        .context("join update_job_status")??;
        Ok(())
    }

    /// Get a job row by ID.
    pub async fn get_job(&self, job_id: &str) -> anyhow::Result<Option<JobRow>> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let row = tokio::task::spawn_blocking(move || -> anyhow::Result<Option<JobRow>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT id, show_slug, media_name, drive_file_id, status, \
                     retry_count, cost_cents, error, created_at, updated_at, \
                     local_path, source_url, config_json \
                     FROM jobs WHERE id = ?1",
                )
                .context("prepare get_job")?;
            let row = stmt
                .query_row([&job_id], |r| {
                    Ok(JobRow {
                        id: r.get(0)?,
                        show_slug: r.get(1)?,
                        media_name: r.get(2)?,
                        drive_file_id: r.get(3)?,
                        status: JobStatus::from_str(&r.get::<_, String>(4)?)
                            .unwrap_or(JobStatus::Pending),
                        retry_count: r.get(5)?,
                        cost_cents: r.get(6)?,
                        error: r.get(7)?,
                        created_at: r.get(8)?,
                        updated_at: r.get(9)?,
                        local_path: r.get(10)?,
                        source_url: r.get(11)?,
                        config_json: r.get(12)?,
                    })
                })
                .optional()
                .context("query get_job")?;
            Ok(row)
        })
        .await
        .context("join get_job")??;
        Ok(row)
    }

    /// Update the accumulated cost_cents for a job.
    /// Count clips associated with this job. Used by the WS emitter to fill
    /// `JobComplete.clipsGenerated`. Returns 0 on DB error rather than failing
    /// so a completed job isn't marked as broken just because the count query
    /// hiccupped.
    pub async fn count_clips_for_job(&self, job_id: &str) -> anyhow::Result<i64> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let n = tokio::task::spawn_blocking(move || -> anyhow::Result<i64> {
            let conn = conn.blocking_lock();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM clips WHERE job_id = ?1",
                    rusqlite::params![job_id],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            Ok(n)
        })
        .await
        .context("join count_clips_for_job")??;
        Ok(n)
    }

    pub async fn update_job_cost(&self, job_id: &str, cost_cents: i64) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            conn.execute(
                "UPDATE jobs SET cost_cents = ?1, updated_at = ?2 WHERE id = ?3",
                rusqlite::params![cost_cents, now, job_id],
            )
            .context("update_job_cost")?;
            Ok(())
        })
        .await
        .context("join update_job_cost")??;
        Ok(())
    }

    /// List all jobs with status='failed'.
    pub async fn get_failed_jobs(&self) -> anyhow::Result<Vec<JobRow>> {
        let conn = self.conn.clone();
        let rows = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<JobRow>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT id, show_slug, media_name, drive_file_id, status, \
                     retry_count, cost_cents, error, created_at, updated_at, \
                     local_path, source_url, config_json \
                     FROM jobs WHERE status = 'failed' ORDER BY updated_at DESC",
                )
                .context("prepare get_failed_jobs")?;
            let rows = stmt
                .query_map([], |r| {
                    Ok(JobRow {
                        id: r.get(0)?,
                        show_slug: r.get(1)?,
                        media_name: r.get(2)?,
                        drive_file_id: r.get(3)?,
                        status: JobStatus::Failed,
                        retry_count: r.get(5)?,
                        cost_cents: r.get(6)?,
                        error: r.get(7)?,
                        created_at: r.get(8)?,
                        updated_at: r.get(9)?,
                        local_path: r.get(10)?,
                        source_url: r.get(11)?,
                        config_json: r.get(12)?,
                    })
                })
                .context("query get_failed_jobs")?
                .collect::<Result<Vec<_>, _>>()
                .context("collect get_failed_jobs")?;
            Ok(rows)
        })
        .await
        .context("join get_failed_jobs")??;
        Ok(rows)
    }

    /// Reset a failed job back to pending for retry. Increments `retry_count`,
    /// clears the error, and sets status to `pending`.
    pub async fn retry_job(&self, job_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            let changed = conn
                .execute(
                    "UPDATE jobs SET status = 'pending', error = NULL, \
                     retry_count = retry_count + 1, updated_at = ?1 \
                     WHERE id = ?2 AND status = 'failed'",
                    rusqlite::params![now, job_id],
                )
                .context("retry_job update")?;
            if changed == 0 {
                anyhow::bail!("job {job_id} is not in 'failed' status (or does not exist)");
            }
            Ok(())
        })
        .await
        .context("join retry_job")??;
        Ok(())
    }

    /// Mark a pending job as cancelled so the worker skips it.
    ///
    /// Only `pending` is supported today — once a job has been claimed by the
    /// worker (any status from `transcribed` onward), cooperative
    /// cancellation hasn't been wired through the clipper yet. Returns an
    /// error if the job is not in `pending` status.
    pub async fn cancel_pending_job(&self, job_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            let now = unix_now();
            let changed = conn
                .execute(
                    "UPDATE jobs SET status = 'cancelled', updated_at = ?1 \
                     WHERE id = ?2 AND status = 'pending'",
                    rusqlite::params![now, job_id],
                )
                .context("cancel_pending_job update")?;
            if changed == 0 {
                anyhow::bail!(
                    "job {job_id} is not in 'pending' status (mid-flight cancel not yet supported)"
                );
            }
            Ok(())
        })
        .await
        .context("join cancel_pending_job")??;
        Ok(())
    }

    /// Delete a job row. The schema has `ON DELETE CASCADE` on `clips`,
    /// `clip_renders`, and `posts`, so removing the job row removes all
    /// downstream rows too. Returns the number of rows removed (0 if the job
    /// didn't exist).
    pub async fn delete_job(&self, job_id: &str) -> anyhow::Result<usize> {
        let conn = self.conn.clone();
        let job_id = job_id.to_string();
        let n = tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let conn = conn.blocking_lock();
            // SQLite enforces foreign keys per-connection — `PRAGMA
            // foreign_keys = ON` is set at open time in `Storage::open`. The
            // CASCADE will fire on this DELETE.
            let n = conn
                .execute("DELETE FROM jobs WHERE id = ?1", rusqlite::params![job_id])
                .context("delete_job DELETE")?;
            Ok(n)
        })
        .await
        .context("join delete_job")??;
        Ok(n)
    }

    /// Clone a job for a re-run: copies show_slug, media_name, source_url,
    /// local_path, and config_json into a brand-new row with a generated id
    /// and `status = 'pending'`. Returns the new job id.
    ///
    /// Use case: you finished a run, tweaked prompts/settings, and want to
    /// reprocess the same source without losing the original outputs.
    pub async fn clone_job_for_rerun(&self, original_id: &str) -> anyhow::Result<String> {
        let conn = self.conn.clone();
        let original_id = original_id.to_string();
        let new_id = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let conn = conn.blocking_lock();
            let row: Option<(
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
            )> = conn
                .query_row(
                    "SELECT show_slug, media_name, drive_file_id, local_path, source_url, \
                            config_json FROM jobs WHERE id = ?1",
                    rusqlite::params![original_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
                )
                .ok();
            let row = row.ok_or_else(|| anyhow::anyhow!("job {original_id} not found"))?;
            // Generate a new id with the same `dashboard_` prefix the
            // dashboard create-job path uses, so the worker treats it the
            // same way.
            let now = unix_now();
            use rand::Rng;
            let rand: u32 = rand::thread_rng().r#gen::<u32>() & 0xFFFFFF;
            let new_id = format!("dashboard_{now}_{rand:06x}");
            conn.execute(
                "INSERT INTO jobs (id, show_slug, media_name, drive_file_id, local_path, \
                                   source_url, config_json, status, retry_count, cost_cents, \
                                   created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', 0, 0, ?8, ?8)",
                rusqlite::params![
                    new_id, row.0, row.1, row.2, row.3, row.4, row.5, now
                ],
            )
            .context("clone_job_for_rerun INSERT")?;
            Ok(new_id)
        })
        .await
        .context("join clone_job_for_rerun")??;
        Ok(new_id)
    }

    /// Insert a clip row. Idempotent (INSERT OR REPLACE).
    ///
    /// `hook_type` is the agency-style classification produced by the
    /// ranker LLM (contrarian / open_loop / etc.). Pass `None` when the
    /// LLM omitted the field — the column is nullable.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_clip(
        &self,
        clip_id: &str,
        job_id: &str,
        start_ms: i64,
        end_ms: i64,
        rank: Option<i64>,
        score: Option<f64>,
        hook: Option<&str>,
        reasoning_json: Option<&str>,
        trend_match: Option<&str>,
        hook_type: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let job_id = job_id.to_string();
        let hook = hook.map(str::to_string);
        let reasoning_json = reasoning_json.map(str::to_string);
        let trend_match = trend_match.map(str::to_string);
        let hook_type = hook_type.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT OR REPLACE INTO clips \
                 (id, job_id, start_ms, end_ms, rank, score, hook, reasoning_json, trend_match, hook_type) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    clip_id,
                    job_id,
                    start_ms,
                    end_ms,
                    rank,
                    score,
                    hook,
                    reasoning_json,
                    trend_match,
                    hook_type
                ],
            )
            .context("insert_clip")?;
            Ok(())
        })
        .await
        .context("join insert_clip")??;
        Ok(())
    }

    /// Update a clip's time bounds. Used by the dashboard's recut/trim
    /// flow when the operator nudges start/end by a few seconds and
    /// re-renders the variant. The corresponding `clip_renders` row(s)
    /// must be replaced separately via [`insert_clip_render`] — this
    /// only touches the `clips` table.
    pub async fn update_clip_bounds(
        &self,
        clip_id: &str,
        start_ms: i64,
        end_ms: i64,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "UPDATE clips SET start_ms = ?1, end_ms = ?2 WHERE id = ?3",
                rusqlite::params![start_ms, end_ms, clip_id],
            )
            .context("update_clip_bounds")?;
            Ok(())
        })
        .await
        .context("join update_clip_bounds")??;
        Ok(())
    }

    /// Insert a clip render variant row. Idempotent (INSERT OR REPLACE).
    pub async fn insert_clip_render(
        &self,
        clip_id: &str,
        variant: &str,
        path: &str,
        bytes: Option<i64>,
        duration_ms: Option<i64>,
        url: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let variant = variant.to_string();
        let path = path.to_string();
        let url = url.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT OR REPLACE INTO clip_renders \
                 (clip_id, variant, path, bytes, duration_ms, url) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![clip_id, variant, path, bytes, duration_ms, url],
            )
            .context("insert_clip_render")?;
            Ok(())
        })
        .await
        .context("join insert_clip_render")??;
        Ok(())
    }

    /// Persist a clip's cover frame path + optional remote URL.
    pub async fn set_clip_cover(
        &self,
        clip_id: &str,
        local_path: &str,
        url: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let local_path = local_path.to_string();
        let url = url.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "UPDATE clips SET cover_path = ?1, cover_url = ?2 WHERE id = ?3",
                rusqlite::params![local_path, url, clip_id],
            )
            .context("set_clip_cover")?;
            Ok(())
        })
        .await
        .context("join set_clip_cover")??;
        Ok(())
    }

    /// Fetch historical clip performance data by joining clips → analytics.
    /// Returns up to `limit` rows ordered by CTR descending (best first).
    /// Gracefully returns an empty vec if the analytics table has no data.
    pub async fn get_clip_performance_history(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<ClipPerformanceRow>> {
        let conn = self.conn.clone();
        let rows =
            tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<ClipPerformanceRow>> {
                let conn = conn.blocking_lock();
                let mut stmt = conn
                    .prepare(
                        "SELECT c.id, c.hook, c.score, c.rank,
                            a.views, a.ctr, a.watch_pct,
                            c.start_ms, c.end_ms
                     FROM clips c
                     INNER JOIN analytics a ON a.clip_id = c.id
                     WHERE a.ctr IS NOT NULL
                     ORDER BY a.ctr DESC
                     LIMIT ?1",
                    )
                    .context("prepare get_clip_performance_history")?;
                let rows = stmt
                    .query_map([limit as i64], |r| {
                        Ok(ClipPerformanceRow {
                            clip_id: r.get(0)?,
                            hook: r.get(1)?,
                            score: r.get(2)?,
                            rank: r.get(3)?,
                            views: r.get(4)?,
                            ctr: r.get(5)?,
                            watch_pct: r.get(6)?,
                            start_ms: r.get(7)?,
                            end_ms: r.get(8)?,
                        })
                    })
                    .context("query get_clip_performance_history")?
                    .collect::<Result<Vec<_>, _>>()
                    .context("collect get_clip_performance_history")?;
                Ok(rows)
            })
            .await
            .context("join get_clip_performance_history")??;
        Ok(rows)
    }

    /// List all clips with optional status filter, including renders and posts.
    pub async fn list_clips(&self, status_filter: Option<&str>) -> anyhow::Result<Vec<ClipDetail>> {
        let conn = self.conn.clone();
        let status_filter = status_filter.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<ClipDetail>> {
            let conn = conn.blocking_lock();

            // Helper to keep the row-mapping closure single-source.
            const COLS: &str = "c.id, c.job_id, c.start_ms, c.end_ms, c.rank, c.score, c.hook, \
                 c.reasoning_json, c.trend_match, c.status, c.social_copy_json, \
                 c.cover_path, c.cover_url, c.hook_type, \
                 j.show_slug, j.media_name, j.status as job_status";

            let clip_sql = if status_filter.is_some() {
                format!(
                    "SELECT {COLS} FROM clips c LEFT JOIN jobs j ON c.job_id = j.id \
                     WHERE c.status = ?1 ORDER BY c.rank ASC NULLS LAST"
                )
            } else {
                format!(
                    "SELECT {COLS} FROM clips c LEFT JOIN jobs j ON c.job_id = j.id \
                     ORDER BY c.rank ASC NULLS LAST"
                )
            };

            let mut stmt = conn.prepare(&clip_sql).context("prepare list_clips")?;
            let map_row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<ClipDetail> {
                Ok(ClipDetail {
                    id: r.get(0)?,
                    job_id: r.get(1)?,
                    start_ms: r.get(2)?,
                    end_ms: r.get(3)?,
                    rank: r.get(4)?,
                    score: r.get(5)?,
                    hook: r.get(6)?,
                    reasoning_json: r.get(7)?,
                    trend_match: r.get(8)?,
                    status: r.get(9)?,
                    social_copy_json: r.get(10)?,
                    cover_path: r.get(11)?,
                    cover_url: r.get(12)?,
                    hook_type: r.get(13)?,
                    renders: Vec::new(),
                    posts: Vec::new(),
                    job: Some(JobSummary {
                        id: r.get::<_, String>(1)?,
                        show_slug: r.get(14)?,
                        media_name: r.get(15)?,
                        status: r.get::<_, Option<String>>(16)?.unwrap_or_default(),
                    }),
                })
            };
            let rows: Vec<ClipDetail> = if let Some(ref sf) = status_filter {
                stmt.query_map([sf], map_row)
                    .context("query list_clips")?
                    .collect::<Result<Vec<_>, _>>()
                    .context("collect list_clips")?
            } else {
                stmt.query_map([], map_row)
                    .context("query list_clips")?
                    .collect::<Result<Vec<_>, _>>()
                    .context("collect list_clips")?
            };

            // Fetch renders and posts for all clips
            let mut result = rows;
            for clip in &mut result {
                let mut render_stmt = conn
                    .prepare(
                        "SELECT clip_id, variant, path, bytes, duration_ms, url \
                         FROM clip_renders WHERE clip_id = ?1",
                    )
                    .context("prepare renders")?;
                clip.renders = render_stmt
                    .query_map([&clip.id], |r| {
                        Ok(ClipRenderRow {
                            clip_id: r.get(0)?,
                            variant: r.get(1)?,
                            path: r.get(2)?,
                            bytes: r.get(3)?,
                            duration_ms: r.get(4)?,
                            url: r.get(5)?,
                        })
                    })
                    .context("query renders")?
                    .collect::<Result<Vec<_>, _>>()
                    .context("collect renders")?;

                let mut post_stmt = conn
                    .prepare(
                        "SELECT clip_id, platform, status, external_id, external_url, posted_at, error \
                         FROM posts WHERE clip_id = ?1",
                    )
                    .context("prepare posts")?;
                clip.posts = post_stmt
                    .query_map([&clip.id], |r| {
                        Ok(PostRow {
                            clip_id: r.get(0)?,
                            platform: r.get(1)?,
                            status: r.get(2)?,
                            external_id: r.get(3)?,
                            external_url: r.get(4)?,
                            posted_at: r.get(5)?,
                            error: r.get(6)?,
                        })
                    })
                    .context("query posts")?
                    .collect::<Result<Vec<_>, _>>()
                    .context("collect posts")?;
            }
            Ok(result)
        })
        .await
        .context("join list_clips")?
    }

    /// Get a single clip by ID with renders and posts.
    pub async fn get_clip(&self, clip_id: &str) -> anyhow::Result<Option<ClipDetail>> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Option<ClipDetail>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT c.id, c.job_id, c.start_ms, c.end_ms, c.rank, c.score, c.hook, \
                     c.reasoning_json, c.trend_match, c.status, c.social_copy_json, \
                     c.cover_path, c.cover_url, c.hook_type, \
                     j.show_slug, j.media_name, j.status as job_status \
                     FROM clips c LEFT JOIN jobs j ON c.job_id = j.id \
                     WHERE c.id = ?1",
                )
                .context("prepare get_clip")?;
            let clip = stmt
                .query_row([&clip_id], |r| {
                    Ok(ClipDetail {
                        id: r.get(0)?,
                        job_id: r.get(1)?,
                        start_ms: r.get(2)?,
                        end_ms: r.get(3)?,
                        rank: r.get(4)?,
                        score: r.get(5)?,
                        hook: r.get(6)?,
                        reasoning_json: r.get(7)?,
                        trend_match: r.get(8)?,
                        status: r.get(9)?,
                        social_copy_json: r.get(10)?,
                        cover_path: r.get(11)?,
                        cover_url: r.get(12)?,
                        hook_type: r.get(13)?,
                        renders: Vec::new(),
                        posts: Vec::new(),
                        job: Some(JobSummary {
                            id: r.get::<_, String>(1)?,
                            show_slug: r.get(14)?,
                            media_name: r.get(15)?,
                            status: r.get::<_, Option<String>>(16)?.unwrap_or_default(),
                        }),
                    })
                })
                .optional()
                .context("query get_clip")?;

            let Some(mut clip) = clip else {
                return Ok(None);
            };

            // Fetch renders
            let mut render_stmt = conn
                .prepare("SELECT clip_id, variant, path, bytes, duration_ms, url FROM clip_renders WHERE clip_id = ?1")
                .context("prepare clip renders")?;
            clip.renders = render_stmt
                .query_map([&clip.id], |r| {
                    Ok(ClipRenderRow {
                        clip_id: r.get(0)?,
                        variant: r.get(1)?,
                        path: r.get(2)?,
                        bytes: r.get(3)?,
                        duration_ms: r.get(4)?,
                        url: r.get(5)?,
                    })
                })
                .context("query clip renders")?
                .collect::<Result<Vec<_>, _>>()
                .context("collect clip renders")?;

            // Fetch posts
            let mut post_stmt = conn
                .prepare("SELECT clip_id, platform, status, external_id, external_url, posted_at, error FROM posts WHERE clip_id = ?1")
                .context("prepare clip posts")?;
            clip.posts = post_stmt
                .query_map([&clip.id], |r| {
                    Ok(PostRow {
                        clip_id: r.get(0)?,
                        platform: r.get(1)?,
                        status: r.get(2)?,
                        external_id: r.get(3)?,
                        external_url: r.get(4)?,
                        posted_at: r.get(5)?,
                        error: r.get(6)?,
                    })
                })
                .context("query clip posts")?
                .collect::<Result<Vec<_>, _>>()
                .context("collect clip posts")?;

            Ok(Some(clip))
        })
        .await
        .context("join get_clip")?
    }

    /// Update the status of a clip (generated, approved, vetoed, posted).
    pub async fn update_clip_status(&self, clip_id: &str, status: &str) -> anyhow::Result<bool> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let status = status.to_string();
        let changed = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            let conn = conn.blocking_lock();
            let n = conn
                .execute(
                    "UPDATE clips SET status = ?1 WHERE id = ?2",
                    rusqlite::params![status, clip_id],
                )
                .context("update_clip_status")?;
            Ok(n > 0)
        })
        .await
        .context("join update_clip_status")??;
        Ok(changed)
    }

    /// Update the hook text of a clip.
    pub async fn update_clip_hook(&self, clip_id: &str, hook: &str) -> anyhow::Result<bool> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let hook = hook.to_string();
        let changed = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            let conn = conn.blocking_lock();
            let n = conn
                .execute(
                    "UPDATE clips SET hook = ?1 WHERE id = ?2",
                    rusqlite::params![hook, clip_id],
                )
                .context("update_clip_hook")?;
            Ok(n > 0)
        })
        .await
        .context("join update_clip_hook")??;
        Ok(changed)
    }

    /// Update social copy JSON for a clip.
    pub async fn update_clip_social_copy(
        &self,
        clip_id: &str,
        social_copy_json: &str,
    ) -> anyhow::Result<bool> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let json = social_copy_json.to_string();
        let changed = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
            let conn = conn.blocking_lock();
            let n = conn
                .execute(
                    "UPDATE clips SET social_copy_json = ?1 WHERE id = ?2",
                    rusqlite::params![json, clip_id],
                )
                .context("update_clip_social_copy")?;
            Ok(n > 0)
        })
        .await
        .context("join update_clip_social_copy")??;
        Ok(changed)
    }

    /// Bulk update clip statuses.
    pub async fn bulk_update_clip_status(
        &self,
        clip_ids: &[String],
        status: &str,
    ) -> anyhow::Result<usize> {
        let conn = self.conn.clone();
        let ids = clip_ids.to_vec();
        let status = status.to_string();
        let count = tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let conn = conn.blocking_lock();
            let mut total = 0usize;
            for id in &ids {
                let n = conn
                    .execute(
                        "UPDATE clips SET status = ?1 WHERE id = ?2",
                        rusqlite::params![status, id],
                    )
                    .context("bulk_update_clip_status")?;
                total += n;
            }
            Ok(total)
        })
        .await
        .context("join bulk_update_clip_status")??;
        Ok(count)
    }

    /// Fetch the top-N trending topics, ordered by score descending.
    /// Only returns the most recent fetch per (source, topic_id) pair.
    /// Returns an empty vec if the trends table is empty.
    pub async fn get_recent_trends(&self, top_n: usize) -> anyhow::Result<Vec<TrendRow>> {
        let conn = self.conn.clone();
        let rows = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<TrendRow>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT source, topic_id, label, score, fetched_at \
                     FROM trends t1 \
                     WHERE fetched_at = (SELECT MAX(t2.fetched_at) FROM trends t2 \
                                         WHERE t2.source = t1.source AND t2.topic_id = t1.topic_id) \
                     ORDER BY score DESC \
                     LIMIT ?1",
                )
                .context("prepare get_recent_trends")?;
            let rows = stmt
                .query_map([top_n as i64], |r| {
                    Ok(TrendRow {
                        source: r.get(0)?,
                        topic_id: r.get(1)?,
                        label: r.get(2)?,
                        score: r.get(3)?,
                        fetched_at: r.get(4)?,
                    })
                })
                .context("query get_recent_trends")?
                .collect::<Result<Vec<_>, _>>()
                .context("collect get_recent_trends")?;
            Ok(rows)
        })
        .await
        .context("join get_recent_trends")??;
        Ok(rows)
    }

    /// Insert a post row. Idempotent (INSERT OR REPLACE).
    pub async fn insert_post(
        &self,
        clip_id: &str,
        platform: &str,
        status: &str,
        external_id: Option<&str>,
        external_url: Option<&str>,
        posted_at: Option<i64>,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let platform = platform.to_string();
        let status = status.to_string();
        let external_id = external_id.map(str::to_string);
        let external_url = external_url.map(str::to_string);
        let error = error.map(str::to_string);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT OR REPLACE INTO posts \
                 (clip_id, platform, status, external_id, external_url, posted_at, error) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    clip_id,
                    platform,
                    status,
                    external_id,
                    external_url,
                    posted_at,
                    error
                ],
            )
            .context("insert_post")?;
            Ok(())
        })
        .await
        .context("join insert_post")??;
        Ok(())
    }

    /// Insert an analytics row. Uses INSERT OR REPLACE keyed on (clip_id, platform, fetched_at).
    pub async fn insert_analytics(&self, row: &AnalyticsRow) -> anyhow::Result<()> {
        let conn = self.conn.clone();
        let row = row.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT OR REPLACE INTO analytics \
                 (clip_id, platform, fetched_at, views, ctr, watch_pct, likes, reposts, replies) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    row.clip_id,
                    row.platform,
                    row.fetched_at,
                    row.views,
                    row.ctr,
                    row.watch_pct,
                    row.likes,
                    row.reposts,
                    row.replies,
                ],
            )
            .context("insert_analytics")?;
            Ok(())
        })
        .await
        .context("join insert_analytics")??;
        Ok(())
    }

    /// Find posts that are due for analytics fetching.
    /// Returns posts where `posted_at` is approximately `target_age_secs` ago
    /// (within a ±window), and no analytics row exists yet for that age bucket.
    pub async fn posts_due_for_analytics(
        &self,
        target_age_secs: i64,
        window_secs: i64,
    ) -> anyhow::Result<Vec<PostRow>> {
        let conn = self.conn.clone();
        let now = unix_now();
        let rows = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<PostRow>> {
            let conn = conn.blocking_lock();
            let min_posted = now - target_age_secs - window_secs;
            let max_posted = now - target_age_secs + window_secs;
            // Select posts that were posted in the target window and don't yet
            // have an analytics row fetched within ±window of the target age.
            let mut stmt = conn.prepare(
                "SELECT p.clip_id, p.platform, p.status, p.external_id, p.external_url, p.posted_at, p.error \
                 FROM posts p \
                 WHERE p.status = 'posted' \
                   AND p.posted_at IS NOT NULL \
                   AND p.posted_at BETWEEN ?1 AND ?2 \
                   AND NOT EXISTS ( \
                       SELECT 1 FROM analytics a \
                       WHERE a.clip_id = p.clip_id \
                         AND a.platform = p.platform \
                         AND a.fetched_at BETWEEN (p.posted_at + ?3 - ?4) AND (p.posted_at + ?3 + ?4) \
                   )"
            ).context("prepare posts_due_for_analytics")?;
            let rows = stmt.query_map(
                rusqlite::params![min_posted, max_posted, target_age_secs, window_secs],
                |r| {
                    Ok(PostRow {
                        clip_id: r.get(0)?,
                        platform: r.get(1)?,
                        status: r.get(2)?,
                        external_id: r.get(3)?,
                        external_url: r.get(4)?,
                        posted_at: r.get(5)?,
                        error: r.get(6)?,
                    })
                },
            )
            .context("query posts_due_for_analytics")?
            .collect::<Result<Vec<_>, _>>()
            .context("collect posts_due_for_analytics")?;
            Ok(rows)
        })
        .await
        .context("join posts_due_for_analytics")??;
        Ok(rows)
    }

    /// Get all analytics rows for a clip, ordered by fetched_at.
    pub async fn get_analytics_for_clip(&self, clip_id: &str) -> anyhow::Result<Vec<AnalyticsRow>> {
        let conn = self.conn.clone();
        let clip_id = clip_id.to_string();
        let rows = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<AnalyticsRow>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn.prepare(
                "SELECT clip_id, platform, fetched_at, views, ctr, watch_pct, likes, reposts, replies \
                 FROM analytics WHERE clip_id = ?1 ORDER BY fetched_at"
            ).context("prepare get_analytics_for_clip")?;
            let rows = stmt.query_map([&clip_id], |r| {
                Ok(AnalyticsRow {
                    clip_id: r.get(0)?,
                    platform: r.get(1)?,
                    fetched_at: r.get(2)?,
                    views: r.get(3)?,
                    ctr: r.get(4)?,
                    watch_pct: r.get(5)?,
                    likes: r.get(6)?,
                    reposts: r.get(7)?,
                    replies: r.get(8)?,
                })
            })
            .context("query get_analytics_for_clip")?
            .collect::<Result<Vec<_>, _>>()
            .context("collect get_analytics_for_clip")?;
            Ok(rows)
        })
        .await
        .context("join get_analytics_for_clip")??;
        Ok(rows)
    }
}

/// A row from the `analytics` table.
#[derive(Debug, Clone)]
pub struct AnalyticsRow {
    pub clip_id: String,
    pub platform: String,
    pub fetched_at: i64,
    pub views: Option<i64>,
    pub ctr: Option<f64>,
    pub watch_pct: Option<f64>,
    pub likes: Option<i64>,
    pub reposts: Option<i64>,
    pub replies: Option<i64>,
}

const SCHEMA_V2: &str = r#"
ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0;
"#;

const SCHEMA_V5: &str = r#"
ALTER TABLE analytics ADD COLUMN likes INTEGER;
ALTER TABLE analytics ADD COLUMN reposts INTEGER;
ALTER TABLE analytics ADD COLUMN replies INTEGER;
"#;

/// V6 adds object-storage URL columns so renders + covers can live on R2
/// (or any S3-compatible bucket) instead of the operator's local disk.
const SCHEMA_V6: &str = r#"
ALTER TABLE clip_renders ADD COLUMN url TEXT;
ALTER TABLE clips ADD COLUMN cover_path TEXT;
ALTER TABLE clips ADD COLUMN cover_url TEXT;
"#;

/// V7 extends `jobs` with the inputs needed to drive the pipeline from the
/// dashboard's New Job UI: a local file path (uploaded mp4) or a source URL
/// (Drive / direct HTTPS) plus a per-job config-overrides JSON blob.
const SCHEMA_V7: &str = r#"
ALTER TABLE jobs ADD COLUMN local_path TEXT;
ALTER TABLE jobs ADD COLUMN source_url TEXT;
ALTER TABLE jobs ADD COLUMN config_json TEXT;
"#;

/// V8 captures the ranker LLM's hook-formula classification per clip
/// (contrarian / open_loop / specific_number / pov / confession / ...).
/// Used by the dashboard to show which hook types are landing for a show.
const SCHEMA_V8: &str = r#"
ALTER TABLE clips ADD COLUMN hook_type TEXT;
"#;

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
        // V1 already includes `error` in the CREATE TABLE; V2 adds `retry_count`.
        // If version is 0 we just created the table fresh with V1 which doesn't
        // have retry_count yet, so always run V2 when version < 2.
        conn.execute_batch(SCHEMA_V2).context("apply schema v2")?;
        conn.pragma_update(None, "user_version", 2)
            .context("set user_version=2")?;
    }
    if version < 3 {
        conn.execute_batch(SCHEMA_V3).context("apply schema v3")?;
        conn.pragma_update(None, "user_version", 3)
            .context("set user_version=3")?;
    }
    if version < 4 {
        conn.execute_batch(SCHEMA_V4).context("apply schema v4")?;
        conn.pragma_update(None, "user_version", 4)
            .context("set user_version=4")?;
    }
    if version < 5 {
        conn.execute_batch(SCHEMA_V5).context("apply schema v5")?;
        conn.pragma_update(None, "user_version", 5)
            .context("set user_version=5")?;
    }
    if version < 6 {
        conn.execute_batch(SCHEMA_V6).context("apply schema v6")?;
        conn.pragma_update(None, "user_version", 6)
            .context("set user_version=6")?;
    }
    if version < 7 {
        conn.execute_batch(SCHEMA_V7).context("apply schema v7")?;
        conn.pragma_update(None, "user_version", 7)
            .context("set user_version=7")?;
    }
    if version < 8 {
        conn.execute_batch(SCHEMA_V8).context("apply schema v8")?;
        conn.pragma_update(None, "user_version", 8)
            .context("set user_version=8")?;
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

    #[tokio::test]
    async fn show_loudness_roundtrip() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");
        let storage = Storage::open(&db_path).await?;

        // No loudness stored yet.
        assert_eq!(storage.get_show_loudness("tfatk").await?, None);

        // Store first measurement.
        storage.set_show_loudness("tfatk", -16.5).await?;
        let lufs = storage.get_show_loudness("tfatk").await?;
        assert!((lufs.unwrap() - -16.5).abs() < 0.01, "got {:?}", lufs);

        // Second measurement updates with running average: (-16.5 + -14.5) / 2 = -15.5
        storage.set_show_loudness("tfatk", -14.5).await?;
        let lufs = storage.get_show_loudness("tfatk").await?;
        assert!((lufs.unwrap() - -15.5).abs() < 0.01, "got {:?}", lufs);

        // Different show is independent.
        assert_eq!(storage.get_show_loudness("other_show").await?, None);

        Ok(())
    }

    #[tokio::test]
    async fn job_status_transitions() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;

        // Create a job.
        storage
            .create_job("job1", Some("show"), Some("ep.mp4"), None)
            .await?;
        let job = storage.get_job("job1").await?.expect("job should exist");
        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.retry_count, 0);
        assert!(job.error.is_none());

        // Walk through the pipeline stages.
        storage
            .update_job_status("job1", JobStatus::Transcribed, None)
            .await?;
        let job = storage.get_job("job1").await?.unwrap();
        assert_eq!(job.status, JobStatus::Transcribed);

        storage
            .update_job_status("job1", JobStatus::Ranked, None)
            .await?;
        let job = storage.get_job("job1").await?.unwrap();
        assert_eq!(job.status, JobStatus::Ranked);

        storage
            .update_job_status("job1", JobStatus::Rendered, None)
            .await?;
        storage
            .update_job_status("job1", JobStatus::Posted, None)
            .await?;
        storage
            .update_job_status("job1", JobStatus::Done, None)
            .await?;
        let job = storage.get_job("job1").await?.unwrap();
        assert_eq!(job.status, JobStatus::Done);

        Ok(())
    }

    #[tokio::test]
    async fn job_failure_and_retry() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;

        storage.create_job("job2", None, None, None).await?;
        storage
            .update_job_status("job2", JobStatus::Transcribed, None)
            .await?;

        // Fail with an error message.
        storage
            .update_job_status("job2", JobStatus::Failed, Some("LLM timeout"))
            .await?;
        let job = storage.get_job("job2").await?.unwrap();
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error.as_deref(), Some("LLM timeout"));

        // Shows up in failed jobs list.
        let failed = storage.get_failed_jobs().await?;
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, "job2");

        // Retry resets to pending and increments retry_count.
        storage.retry_job("job2").await?;
        let job = storage.get_job("job2").await?.unwrap();
        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.retry_count, 1);
        assert!(job.error.is_none());

        // Failed list is now empty.
        let failed = storage.get_failed_jobs().await?;
        assert!(failed.is_empty());

        // Retry on non-failed job should error.
        let err = storage.retry_job("job2").await;
        assert!(err.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn create_job_is_idempotent() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;

        storage.create_job("j1", None, Some("a.mp4"), None).await?;
        storage
            .update_job_status("j1", JobStatus::Transcribed, None)
            .await?;

        // Second create should be a no-op (INSERT OR IGNORE).
        storage.create_job("j1", None, Some("b.mp4"), None).await?;
        let job = storage.get_job("j1").await?.unwrap();
        assert_eq!(
            job.status,
            JobStatus::Transcribed,
            "status should not have been reset"
        );
        assert_eq!(
            job.media_name.as_deref(),
            Some("a.mp4"),
            "media_name should not change"
        );

        Ok(())
    }

    #[tokio::test]
    async fn insert_clip_and_render_and_post() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;

        storage.create_job("j1", None, None, None).await?;

        // Insert a clip.
        storage
            .insert_clip(
                "c1",
                "j1",
                10000,
                30000,
                Some(1),
                Some(85.0),
                Some("hook"),
                Some("{}"),
                None,
                None,
            )
            .await?;

        // Insert a render variant.
        storage
            .insert_clip_render(
                "c1",
                "9x16",
                "/tmp/clip.mp4",
                Some(2500000),
                Some(20000),
                None,
            )
            .await?;

        // Insert a post.
        storage
            .insert_post(
                "c1",
                "youtube_shorts",
                "posted",
                Some("abc123"),
                Some("https://yt.be/abc"),
                Some(1700000000),
                None,
            )
            .await?;

        // Idempotent re-insert should not error.
        storage
            .insert_clip(
                "c1",
                "j1",
                10000,
                30000,
                Some(1),
                Some(85.0),
                Some("hook"),
                Some("{}"),
                None,
                None,
            )
            .await?;
        storage
            .insert_clip_render(
                "c1",
                "9x16",
                "/tmp/clip.mp4",
                Some(2500000),
                Some(20000),
                None,
            )
            .await?;
        storage
            .insert_post(
                "c1",
                "youtube_shorts",
                "posted",
                Some("abc123"),
                None,
                None,
                None,
            )
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn migration_v2_on_existing_v1_db() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");

        // Simulate a V1-only database.
        {
            let conn = Connection::open(&db_path)?;
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.execute_batch(SCHEMA_V1)?;
            conn.pragma_update(None, "user_version", 1)?;
            conn.execute(
                "INSERT INTO jobs (id, status, created_at, updated_at) VALUES ('old_job', 'done', 0, 0)",
                [],
            )?;
        }

        // Open with new code — should apply V2 migration.
        let storage = Storage::open(&db_path).await?;
        let job = storage
            .get_job("old_job")
            .await?
            .expect("old_job should exist");
        assert_eq!(job.status, JobStatus::Done);
        assert_eq!(job.retry_count, 0, "retry_count should default to 0");

        Ok(())
    }

    #[tokio::test]
    async fn update_job_cost_persists() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;

        storage.create_job("j1", None, None, None).await?;
        let job = storage.get_job("j1").await?.unwrap();
        assert_eq!(job.cost_cents, 0, "initial cost should be 0");

        storage.update_job_cost("j1", 42).await?;
        let job = storage.get_job("j1").await?.unwrap();
        assert_eq!(job.cost_cents, 42, "cost should be updated to 42");

        // Update again — overwrites, not accumulates.
        storage.update_job_cost("j1", 100).await?;
        let job = storage.get_job("j1").await?.unwrap();
        assert_eq!(job.cost_cents, 100);

        Ok(())
    }

    #[tokio::test]
    async fn get_nonexistent_job_returns_none() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;
        let job = storage.get_job("nope").await?;
        assert!(job.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn show_loudness_survives_reopen() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("test.db");

        {
            let storage = Storage::open(&db_path).await?;
            storage.set_show_loudness("myshow", -18.0).await?;
        }

        // Reopen and verify persistence.
        let storage2 = Storage::open(&db_path).await?;
        let lufs = storage2.get_show_loudness("myshow").await?;
        assert!((lufs.unwrap() - -18.0).abs() < 0.01);
        Ok(())
    }

    #[tokio::test]
    async fn get_recent_trends_returns_empty_when_no_trends() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;
        let trends = storage.get_recent_trends(10).await?;
        assert!(trends.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn get_recent_trends_returns_top_n_by_score() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;

        // Insert some trends directly.
        {
            let conn = storage.conn.lock().await;
            conn.execute(
                "INSERT INTO trends (source, topic_id, label, score, fetched_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["google", "t1", "AI Safety", 90.0, 1000],
            )?;
            conn.execute(
                "INSERT INTO trends (source, topic_id, label, score, fetched_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["google", "t2", "Rust Language", 70.0, 1000],
            )?;
            conn.execute(
                "INSERT INTO trends (source, topic_id, label, score, fetched_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["google", "t3", "Climate Change", 50.0, 1000],
            )?;
        }

        let trends = storage.get_recent_trends(2).await?;
        assert_eq!(trends.len(), 2);
        assert_eq!(trends[0].label.as_deref(), Some("AI Safety"));
        assert_eq!(trends[1].label.as_deref(), Some("Rust Language"));

        // top_n=0 returns nothing.
        let trends_zero = storage.get_recent_trends(0).await?;
        assert!(trends_zero.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn insert_clip_with_trend_match() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let storage = Storage::open(dir.path().join("test.db")).await?;
        storage.create_job("j1", None, None, None).await?;

        // Insert a clip with trend_match.
        storage
            .insert_clip(
                "c1",
                "j1",
                10000,
                30000,
                Some(1),
                Some(85.0),
                Some("hook"),
                Some("{}"),
                Some("AI Safety"),
                Some("contrarian"),
            )
            .await?;

        // Verify trend_match was stored.
        let conn = storage.conn.lock().await;
        let tm: Option<String> =
            conn.query_row("SELECT trend_match FROM clips WHERE id = 'c1'", [], |r| {
                r.get(0)
            })?;
        assert_eq!(tm.as_deref(), Some("AI Safety"));

        Ok(())
    }
}
