//! Post-publish analytics fetching for YouTube and Bluesky.
//!
//! Checks the `posts` table for clips posted ~24h and ~72h ago, fetches
//! platform-specific metrics, and stores them in the `analytics` table.
//!
//! - **YouTube**: uses YouTube Data API v3 `videos.list` with `part=statistics`
//!   to get view count. CTR and watch percentage require YouTube Analytics API
//!   (owner-only, needs `yt-analytics.readonly` scope) — gracefully skipped when
//!   unavailable.
//! - **Bluesky**: uses `app.bsky.feed.getPostThread` to read like, repost, and
//!   reply counts from the thread view.

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::config::Config;
use crate::google_auth::GoogleAuth;
use crate::storage::{AnalyticsRow, Storage};

/// Age targets for analytics pulls, in seconds.
const ANALYTICS_24H: i64 = 24 * 3600;
const ANALYTICS_72H: i64 = 72 * 3600;
/// Window around the target age (±3 hours) to account for polling frequency.
const ANALYTICS_WINDOW: i64 = 3 * 3600;

/// Run one analytics-pull cycle: check for posts due at 24h and 72h, fetch
/// metrics, and store them. Errors on individual posts are logged but don't
/// abort the whole cycle.
pub async fn pull_analytics(
    cfg: &Config,
    google: Option<&GoogleAuth>,
    storage: &Storage,
) -> Result<usize> {
    let mut fetched = 0usize;

    for target_age in [ANALYTICS_24H, ANALYTICS_72H] {
        let due = storage
            .posts_due_for_analytics(target_age, ANALYTICS_WINDOW)
            .await
            .context("query posts due for analytics")?;

        if due.is_empty() {
            continue;
        }

        let label = if target_age == ANALYTICS_24H {
            "24h"
        } else {
            "72h"
        };
        tracing::info!(count = due.len(), window = label, "analytics: posts due");

        for post in &due {
            let external_id = match &post.external_id {
                Some(id) if !id.is_empty() => id.as_str(),
                _ => {
                    tracing::debug!(
                        clip_id = %post.clip_id,
                        platform = %post.platform,
                        "analytics: no external_id, skipping"
                    );
                    continue;
                }
            };

            let result = match post.platform.as_str() {
                "youtube" => fetch_youtube_analytics(google, external_id).await,
                "bluesky" => fetch_bluesky_analytics(cfg, external_id).await,
                other => {
                    tracing::debug!(
                        platform = other,
                        "analytics: unsupported platform, skipping"
                    );
                    continue;
                }
            };

            match result {
                Ok(mut row) => {
                    row.clip_id = post.clip_id.clone();
                    row.platform = post.platform.clone();
                    if let Err(e) = storage.insert_analytics(&row).await {
                        tracing::warn!(
                            clip_id = %post.clip_id,
                            platform = %post.platform,
                            error = %e,
                            "analytics: failed to store"
                        );
                    } else {
                        fetched += 1;
                        tracing::info!(
                            clip_id = %post.clip_id,
                            platform = %post.platform,
                            window = label,
                            views = ?row.views,
                            likes = ?row.likes,
                            "analytics: stored"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        clip_id = %post.clip_id,
                        platform = %post.platform,
                        error = %e,
                        "analytics: fetch failed"
                    );
                }
            }
        }
    }

    Ok(fetched)
}

// ---------------------------------------------------------------------------
// YouTube
// ---------------------------------------------------------------------------

/// YouTube Data API v3 `videos.list` response (part=statistics).
#[derive(Debug, Deserialize)]
struct YtVideoListResponse {
    #[serde(default)]
    items: Vec<YtVideoItem>,
}

#[derive(Debug, Deserialize)]
struct YtVideoItem {
    #[serde(default)]
    statistics: Option<YtStatistics>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YtStatistics {
    #[serde(default)]
    view_count: Option<String>,
    #[serde(default)]
    like_count: Option<String>,
    #[serde(default)]
    comment_count: Option<String>,
}

async fn fetch_youtube_analytics(
    google: Option<&GoogleAuth>,
    video_id: &str,
) -> Result<AnalyticsRow> {
    let google = google.context("YouTube analytics requires Google credentials")?;
    let token = google
        .access_token()
        .await
        .context("refresh token for YouTube analytics")?;

    let http = reqwest::Client::new();
    let url = format!("https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_id}");
    let res = http
        .get(&url)
        .bearer_auth(&token)
        .send()
        .await
        .context("GET youtube videos.list")?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("youtube videos.list failed: {status} {body}");
    }

    let parsed: YtVideoListResponse = res.json().await.context("parse youtube videos.list")?;
    let stats = parsed.items.first().and_then(|i| i.statistics.as_ref());

    let now = unix_now();
    match stats {
        Some(s) => Ok(AnalyticsRow {
            clip_id: String::new(), // filled by caller
            platform: String::new(),
            fetched_at: now,
            views: s.view_count.as_deref().and_then(|v| v.parse().ok()),
            ctr: None,       // requires YouTube Analytics API (yt-analytics.readonly scope)
            watch_pct: None, // requires YouTube Analytics API
            likes: s.like_count.as_deref().and_then(|v| v.parse().ok()),
            reposts: None,
            replies: s.comment_count.as_deref().and_then(|v| v.parse().ok()),
        }),
        None => Ok(AnalyticsRow {
            clip_id: String::new(),
            platform: String::new(),
            fetched_at: now,
            views: None,
            ctr: None,
            watch_pct: None,
            likes: None,
            reposts: None,
            replies: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// Bluesky
// ---------------------------------------------------------------------------

/// Subset of `app.bsky.feed.defs#threadViewPost`.
#[derive(Debug, Deserialize)]
struct BskyThreadResponse {
    #[serde(default)]
    thread: Option<BskyThreadViewPost>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BskyThreadViewPost {
    #[serde(default)]
    post: Option<BskyPostView>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BskyPostView {
    #[serde(default)]
    like_count: Option<i64>,
    #[serde(default)]
    repost_count: Option<i64>,
    #[serde(default)]
    reply_count: Option<i64>,
}

async fn fetch_bluesky_analytics(cfg: &Config, at_uri: &str) -> Result<AnalyticsRow> {
    let handle = cfg
        .bluesky_handle
        .as_deref()
        .context("BLUESKY_HANDLE required for analytics")?;
    let app_password = cfg
        .bluesky_app_password
        .as_deref()
        .context("BLUESKY_APP_PASSWORD required for analytics")?;

    let http = reqwest::Client::new();
    let pds_url = cfg.bluesky_pds_url.trim_end_matches('/');

    // Authenticate to get an access token.
    let session_url = format!("{pds_url}/xrpc/com.atproto.server.createSession");
    let session_body = serde_json::json!({
        "identifier": handle,
        "password": app_password,
    });
    let res = http
        .post(&session_url)
        .json(&session_body)
        .send()
        .await
        .context("bluesky createSession for analytics")?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("bluesky createSession failed: {status} {body}");
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Session {
        access_jwt: String,
    }
    let session: Session = res.json().await.context("parse bluesky session")?;

    // Fetch the thread to get engagement counts.
    let encoded_uri =
        percent_encoding::utf8_percent_encode(at_uri, percent_encoding::NON_ALPHANUMERIC);
    let thread_url =
        format!("{pds_url}/xrpc/app.bsky.feed.getPostThread?uri={encoded_uri}&depth=0");
    let res = http
        .get(&thread_url)
        .bearer_auth(&session.access_jwt)
        .send()
        .await
        .context("bluesky getPostThread for analytics")?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        anyhow::bail!("bluesky getPostThread failed: {status} {body}");
    }

    let parsed: BskyThreadResponse = res.json().await.context("parse bluesky thread")?;
    let post = parsed.thread.and_then(|t| t.post);

    let now = unix_now();
    match post {
        Some(p) => Ok(AnalyticsRow {
            clip_id: String::new(),
            platform: String::new(),
            fetched_at: now,
            views: None, // Bluesky doesn't expose view counts
            ctr: None,
            watch_pct: None,
            likes: p.like_count,
            reposts: p.repost_count,
            replies: p.reply_count,
        }),
        None => Ok(AnalyticsRow {
            clip_id: String::new(),
            platform: String::new(),
            fetched_at: now,
            views: None,
            ctr: None,
            watch_pct: None,
            likes: None,
            reposts: None,
            replies: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// Digest summary
// ---------------------------------------------------------------------------

/// Build a human-readable analytics summary for inclusion in digest emails.
/// Returns `None` if no analytics data exists for any of the given clip IDs.
pub async fn build_analytics_summary(
    storage: &Storage,
    clip_ids: &[String],
) -> Result<Option<String>> {
    let mut lines = Vec::new();

    for clip_id in clip_ids {
        let rows = storage.get_analytics_for_clip(clip_id).await?;
        if rows.is_empty() {
            continue;
        }
        for row in &rows {
            let age_label = age_bucket_label(row.fetched_at, clip_id, storage).await;
            let mut parts = Vec::new();

            if let Some(v) = row.views {
                parts.push(format!("{v} views"));
            }
            if let Some(l) = row.likes {
                parts.push(format!("{l} likes"));
            }
            if let Some(r) = row.reposts {
                parts.push(format!("{r} reposts"));
            }
            if let Some(r) = row.replies {
                parts.push(format!("{r} replies"));
            }
            if let Some(ctr) = row.ctr {
                parts.push(format!("{ctr:.1}% CTR"));
            }
            if let Some(wp) = row.watch_pct {
                parts.push(format!("{wp:.0}% watch"));
            }

            if !parts.is_empty() {
                lines.push(format!(
                    "  {} ({age_label}): {}",
                    row.platform,
                    parts.join(", ")
                ));
            }
        }
    }

    if lines.is_empty() {
        return Ok(None);
    }

    let mut out = String::from("## Analytics\n\n");
    out.push_str(&lines.join("\n"));
    out.push('\n');
    Ok(Some(out))
}

async fn age_bucket_label(fetched_at: i64, _clip_id: &str, _storage: &Storage) -> String {
    // Approximate: just report the fetched_at relative to now.
    let now = unix_now();
    let age_h = (now - fetched_at).max(0) / 3600;
    if age_h < 1 {
        "just now".to_string()
    } else {
        format!("fetched {age_h}h ago")
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
    use crate::storage::Storage;

    #[tokio::test]
    async fn insert_and_query_analytics() {
        let storage = Storage::open_in_memory_sync();

        // Create job + clip + post.
        storage.create_job("j1", None, None, None).await.unwrap();
        storage
            .insert_clip("c1", "j1", 0, 30000, Some(1), Some(90.0), None, None, None)
            .await
            .unwrap();

        let now = unix_now();
        let posted_24h_ago = now - 24 * 3600;
        storage
            .insert_post(
                "c1",
                "youtube",
                "posted",
                Some("vid123"),
                Some("https://youtube.com/shorts/vid123"),
                Some(posted_24h_ago),
                None,
            )
            .await
            .unwrap();

        // Should be due for 24h analytics.
        let due = storage
            .posts_due_for_analytics(24 * 3600, 3 * 3600)
            .await
            .unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].clip_id, "c1");
        assert_eq!(due[0].platform, "youtube");

        // Insert analytics row.
        let row = AnalyticsRow {
            clip_id: "c1".into(),
            platform: "youtube".into(),
            fetched_at: now,
            views: Some(150),
            ctr: None,
            watch_pct: None,
            likes: Some(10),
            reposts: None,
            replies: Some(3),
        };
        storage.insert_analytics(&row).await.unwrap();

        // Should no longer be due.
        let due = storage
            .posts_due_for_analytics(24 * 3600, 3 * 3600)
            .await
            .unwrap();
        assert!(due.is_empty());

        // Query analytics.
        let rows = storage.get_analytics_for_clip("c1").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].views, Some(150));
        assert_eq!(rows[0].likes, Some(10));
    }

    #[tokio::test]
    async fn posts_not_yet_due_are_excluded() {
        let storage = Storage::open_in_memory_sync();
        storage.create_job("j1", None, None, None).await.unwrap();
        storage
            .insert_clip("c1", "j1", 0, 30000, Some(1), None, None, None, None)
            .await
            .unwrap();

        // Posted 1 hour ago — not yet due for 24h window.
        let now = unix_now();
        storage
            .insert_post(
                "c1",
                "bluesky",
                "posted",
                Some("at://x"),
                None,
                Some(now - 3600),
                None,
            )
            .await
            .unwrap();

        let due = storage
            .posts_due_for_analytics(24 * 3600, 3 * 3600)
            .await
            .unwrap();
        assert!(due.is_empty());
    }

    #[tokio::test]
    async fn analytics_summary_renders() {
        let storage = Storage::open_in_memory_sync();
        storage.create_job("j1", None, None, None).await.unwrap();
        storage
            .insert_clip("c1", "j1", 0, 30000, Some(1), None, None, None, None)
            .await
            .unwrap();

        let now = unix_now();
        let row = AnalyticsRow {
            clip_id: "c1".into(),
            platform: "youtube".into(),
            fetched_at: now,
            views: Some(500),
            ctr: Some(4.2),
            watch_pct: Some(65.0),
            likes: Some(25),
            reposts: None,
            replies: Some(8),
        };
        storage.insert_analytics(&row).await.unwrap();

        let summary = build_analytics_summary(&storage, &["c1".into()])
            .await
            .unwrap();
        assert!(summary.is_some());
        let text = summary.unwrap();
        assert!(text.contains("500 views"));
        assert!(text.contains("25 likes"));
        assert!(text.contains("4.2% CTR"));
        assert!(text.contains("65% watch"));
    }

    #[tokio::test]
    async fn analytics_summary_empty_when_no_data() {
        let storage = Storage::open_in_memory_sync();
        let summary = build_analytics_summary(&storage, &["nonexistent".into()])
            .await
            .unwrap();
        assert!(summary.is_none());
    }
}
