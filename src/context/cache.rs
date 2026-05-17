use anyhow::Context as _;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::{ContextFetcher, TrendEntry};

/// How long cached trend data is considered fresh (24 hours).
const FRESHNESS_SECS: i64 = 24 * 60 * 60;

/// SQLite-backed trend cache.  Wraps a shared connection (the same one `Storage`
/// owns) and provides read/write helpers for the `trends` table.
#[derive(Clone)]
pub struct TrendCache {
    conn: Arc<Mutex<Connection>>,
}

impl TrendCache {
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Return the most recent `fetched_at` epoch for a given source, or 0 if none.
    pub async fn last_fetched(&self, source: &str) -> anyhow::Result<i64> {
        let conn = self.conn.clone();
        let source = source.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<i64> {
            let conn = conn.blocking_lock();
            let ts: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(fetched_at), 0) FROM trends WHERE source = ?1",
                    [&source],
                    |r| r.get(0),
                )
                .context("query last_fetched")?;
            Ok(ts)
        })
        .await
        .context("join last_fetched")?
    }

    /// Returns `true` if data for `source` is less than 24 hours old.
    pub async fn is_fresh(&self, source: &str) -> anyhow::Result<bool> {
        let last = self.last_fetched(source).await?;
        Ok(unix_now() - last < FRESHNESS_SECS)
    }

    /// Insert a batch of trend entries into the `trends` table.
    pub async fn store(&self, entries: &[TrendEntry]) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let entries = entries.to_vec();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mut conn = conn.blocking_lock();
            let now = unix_now();
            let tx = conn.transaction().context("begin trends tx")?;
            {
                let mut stmt = tx
                    .prepare(
                        "INSERT OR REPLACE INTO trends (source, topic_id, label, score, fetched_at) \
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                    )
                    .context("prepare trends insert")?;
                for e in &entries {
                    stmt.execute((&e.source, &e.topic_id, &e.label, e.score, now))
                        .context("insert trend")?;
                }
            }
            tx.commit().context("commit trends")?;
            Ok(())
        })
        .await
        .context("join store")?
    }

    /// Read the latest trends for a given source (most recent batch).
    pub async fn read_latest(&self, source: &str) -> anyhow::Result<Vec<TrendEntry>> {
        let conn = self.conn.clone();
        let source = source.to_string();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<TrendEntry>> {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT source, topic_id, label, score FROM trends \
                     WHERE source = ?1 AND fetched_at = \
                       (SELECT MAX(fetched_at) FROM trends WHERE source = ?1)",
                )
                .context("prepare read_latest")?;
            let rows = stmt
                .query_map([&source], |r| {
                    Ok(TrendEntry {
                        source: r.get(0)?,
                        topic_id: r.get(1)?,
                        label: r.get(2)?,
                        score: r.get(3)?,
                    })
                })
                .context("query read_latest")?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.context("read trend row")?);
            }
            Ok(out)
        })
        .await
        .context("join read_latest")?
    }

    /// Fetch-or-cache for a single source: return cached data if fresh, otherwise
    /// call the fetcher, store results, and return them.
    pub async fn get_or_fetch(
        &self,
        fetcher: &impl ContextFetcher,
    ) -> anyhow::Result<Vec<TrendEntry>> {
        let source = fetcher.source_name();
        if self.is_fresh(source).await? {
            info!(source, "trend cache fresh, using cached data");
            return self.read_latest(source).await;
        }

        info!(source, "trend cache stale, fetching new data");
        match fetcher.fetch().await {
            Ok(entries) => {
                self.store(&entries).await?;
                Ok(entries)
            }
            Err(e) => {
                warn!(source, error = %e, "trend fetch failed, falling back to stale cache");
                self.read_latest(source).await
            }
        }
    }

    /// Merge trends from all configured sources, using the cache for freshness.
    /// Each source is fetched independently; failures are logged and skipped.
    /// Returns a combined, score-descending list.
    pub async fn get_current_trends(
        &self,
        gdelt: &impl ContextFetcher,
        reddit: &impl ContextFetcher,
        google: &impl ContextFetcher,
    ) -> anyhow::Result<Vec<TrendEntry>> {
        let mut all = Vec::new();
        for name in ["gdelt", "reddit", "google"] {
            let result = match name {
                "gdelt" => self.get_or_fetch(gdelt).await,
                "reddit" => self.get_or_fetch(reddit).await,
                "google" => self.get_or_fetch(google).await,
                _ => unreachable!(),
            };
            match result {
                Ok(entries) => all.extend(entries),
                Err(e) => warn!(source = name, error = %e, "skipping source"),
            }
        }
        all.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(all)
    }

    /// Prune entries older than `max_age_secs` to keep the DB tidy.
    pub async fn prune(&self, max_age_secs: i64) -> anyhow::Result<usize> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<usize> {
            let conn = conn.blocking_lock();
            let cutoff = unix_now() - max_age_secs;
            let deleted = conn
                .execute("DELETE FROM trends WHERE fetched_at < ?1", [cutoff])
                .context("prune trends")?;
            Ok(deleted)
        })
        .await
        .context("join prune")?
    }
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

    async fn open_test_db() -> (TrendCache, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS trends (
                source     TEXT NOT NULL,
                topic_id   TEXT NOT NULL,
                label      TEXT,
                score      REAL,
                fetched_at INTEGER NOT NULL,
                PRIMARY KEY (source, topic_id, fetched_at)
            );",
        )
        .unwrap();
        let cache = TrendCache::new(Arc::new(Mutex::new(conn)));
        (cache, dir)
    }

    #[tokio::test]
    async fn empty_cache_is_not_fresh() {
        let (cache, _dir) = open_test_db().await;
        assert!(!cache.is_fresh("gdelt").await.unwrap());
    }

    #[tokio::test]
    async fn store_and_read_latest() {
        let (cache, _dir) = open_test_db().await;
        let entries = vec![
            TrendEntry {
                source: "gdelt".into(),
                topic_id: "t1".into(),
                label: "Topic One".into(),
                score: 0.9,
            },
            TrendEntry {
                source: "gdelt".into(),
                topic_id: "t2".into(),
                label: "Topic Two".into(),
                score: 0.7,
            },
        ];
        cache.store(&entries).await.unwrap();

        let latest = cache.read_latest("gdelt").await.unwrap();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[0].topic_id, "t1");
    }

    #[tokio::test]
    async fn freshness_after_store() {
        let (cache, _dir) = open_test_db().await;
        assert!(!cache.is_fresh("reddit").await.unwrap());

        cache
            .store(&[TrendEntry {
                source: "reddit".into(),
                topic_id: "r1".into(),
                label: "Hot post".into(),
                score: 0.5,
            }])
            .await
            .unwrap();

        assert!(cache.is_fresh("reddit").await.unwrap());
    }

    #[tokio::test]
    async fn prune_old_entries() {
        let (cache, _dir) = open_test_db().await;
        // Insert entry with old timestamp directly.
        {
            let conn = cache.conn.lock().await;
            conn.execute(
                "INSERT INTO trends (source, topic_id, label, score, fetched_at) \
                 VALUES ('old', 'o1', 'Old', 0.1, 1000)",
                [],
            )
            .unwrap();
        }
        cache
            .store(&[TrendEntry {
                source: "new".into(),
                topic_id: "n1".into(),
                label: "New".into(),
                score: 0.8,
            }])
            .await
            .unwrap();

        let pruned = cache.prune(60).await.unwrap();
        assert_eq!(pruned, 1);

        let old = cache.read_latest("old").await.unwrap();
        assert!(old.is_empty());
        let new = cache.read_latest("new").await.unwrap();
        assert_eq!(new.len(), 1);
    }

    #[tokio::test]
    async fn read_latest_empty_source() {
        let (cache, _dir) = open_test_db().await;
        let entries = cache.read_latest("nonexistent").await.unwrap();
        assert!(entries.is_empty());
    }
}
