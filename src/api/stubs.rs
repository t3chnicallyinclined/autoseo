//! Stub endpoints for routes the dashboard expects but which the Rust
//! backend doesn't yet implement with real data.
//!
//! Each handler returns the minimal valid JSON shape so dashboard pages
//! render their empty-state UI instead of erroring with `404`. As real
//! endpoints land they should replace these.

use std::sync::Arc;

use axum::{Json, Router, extract::State, routing::get};
use serde_json::{Value, json};

use super::AppState;

async fn shows() -> Json<Vec<Value>> {
    Json(vec![])
}

async fn episodes() -> Json<Vec<Value>> {
    Json(vec![])
}

/// Synthesize one "agent" per pipeline stage so the Agents page has something
/// to render. There's no first-class agent concept in the backend today; this
/// gives the dashboard a stable list driven by the actual stage taxonomy used
/// elsewhere (must stay in lockstep with `pipeline_status`).
async fn agents() -> Json<Vec<Value>> {
    const STAGES: &[(&str, &str, &str, &str, &[&str])] = &[
        ("ingest", "Ingest", "Gmail/Drive poller", "#3b82f6",
         &["Gmail API", "Drive API"]),
        ("download", "Download", "yt-dlp / Drive download", "#06b6d4",
         &["yt-dlp", "HTTP"]),
        ("audio", "Audio", "ffmpeg audio extract + loudnorm", "#22c55e",
         &["ffmpeg", "loudnorm"]),
        ("transcribe", "Transcribe", "Whisper STT (Groq/local)", "#f59e0b",
         &["Whisper", "Groq", "VAD"]),
        ("features", "Features", "VAD / prosody / embedding / AST", "#8b5cf6",
         &["Silero VAD", "fastembed", "AST"]),
        ("rank", "Rank", "LLM + VLM re-rank", "#ec4899",
         &["LLM ranker", "VLM Qwen3", "VLM premium"]),
        ("render", "Render", "Crop / caption / overlay / encode", "#ef4444",
         &["SCRFD", "ASS subtitles", "ffmpeg"]),
        ("post", "Post", "YouTube / Bluesky / Ayrshare", "#10b981",
         &["YouTube Data", "Bluesky", "Ayrshare"]),
    ];
    let out: Vec<Value> = STAGES
        .iter()
        .map(|(id, name, role, color, skills)| {
            json!({
                "id": id,
                "name": name,
                "role": role,
                "color": color,
                "status": "idle",
                "currentTask": null,
                "elapsed": null,
                "skills": skills,
                "tasksCompleted": 0,
                "avgDuration": "—",
                "successRate": 0,
            })
        })
        .collect();
    Json(out)
}

/// Project rows from the `trends` table (last 24h) onto the three sources
/// the dashboard renders. Fields not stored per-source (sources / tone /
/// comments / subreddit / volume / related) get sensible neutral defaults
/// until extractors persist them.
async fn trends(State(state): State<Arc<AppState>>) -> Json<Value> {
    let conn = state.storage.conn();
    let body = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let conn = conn.blocking_lock();
        let cutoff = chrono::Utc::now().timestamp() - 24 * 3600;
        let mut stmt = conn.prepare(
            "SELECT source, topic_id, label, score \
             FROM trends \
             WHERE fetched_at >= ?1 \
             ORDER BY fetched_at DESC, score DESC \
             LIMIT 300",
        )?;
        let mut iter = stmt.query(rusqlite::params![cutoff])?;
        let mut gdelt = Vec::new();
        let mut reddit = Vec::new();
        let mut google = Vec::new();
        while let Some(row) = iter.next()? {
            let source: String = row.get(0)?;
            let topic_id: String = row.get(1).unwrap_or_default();
            let label: Option<String> = row.get(2).ok();
            let score: f64 = row.get(3).unwrap_or(0.0);
            let topic = label.clone().unwrap_or_else(|| topic_id.clone());
            match source.as_str() {
                "gdelt" => gdelt.push(json!({
                    "topic": topic,
                    "score": score,
                    "sources": 0,
                    "tone": 0,
                    "matched": 0,
                })),
                "reddit" => reddit.push(json!({
                    "title": topic,
                    // topic_id often encodes the subreddit when present; fall
                    // back to "" when the poller hasn't filled it.
                    "subreddit": topic_id,
                    "score": (score * 1000.0).round() as i64,
                    "comments": 0,
                })),
                "google" => google.push(json!({
                    "term": topic,
                    // We store a 0..1 normalized score; render as a faux
                    // volume by scaling so the dashboard's progress bar
                    // (which divides by 1.3M) reads roughly score%.
                    "volume": (score * 1_000_000.0).round() as i64,
                    "related": Value::Array(vec![]),
                })),
                _ => { /* unknown source — skip */ }
            }
        }
        // Cap each section so a runaway poller can't blow up the UI.
        gdelt.truncate(15);
        reddit.truncate(15);
        google.truncate(15);
        Ok(json!({ "gdelt": gdelt, "reddit": reddit, "google": google }))
    })
    .await
    .unwrap_or(Ok(json!({ "gdelt": [], "reddit": [], "google": [] })))
    .unwrap_or(json!({ "gdelt": [], "reddit": [], "google": [] }));
    Json(body)
}

/// Aggregate the `analytics` table into the dashboard's two charts:
///   - `views`: daily totals, split by platform — youtube / bluesky / linkedin / threads
///   - `topClips`: 10 best by lifetime views, joined to clips for hook + episode
async fn analytics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let conn = state.storage.conn();
    let body = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        let conn = conn.blocking_lock();

        // Daily views grouped by platform. Using SQLite's `date(...)` to bucket
        // unix-timestamps into YYYY-MM-DD; widening to 14 days keeps the chart
        // useful while a single episode is in flight.
        let cutoff = chrono::Utc::now().timestamp() - 14 * 86_400;
        let mut by_day: std::collections::BTreeMap<String, [i64; 4]> =
            std::collections::BTreeMap::new();
        let mut stmt = conn.prepare(
            "SELECT date(fetched_at, 'unixepoch') AS d, platform, SUM(views) \
             FROM analytics \
             WHERE fetched_at >= ?1 \
             GROUP BY d, platform",
        )?;
        let mut iter = stmt.query(rusqlite::params![cutoff])?;
        while let Some(row) = iter.next()? {
            let day: String = row.get(0)?;
            let platform: String = row.get(1)?;
            let views: i64 = row.get(2).unwrap_or(0);
            let entry = by_day.entry(day).or_insert([0, 0, 0, 0]);
            match platform.as_str() {
                "youtube" => entry[0] += views,
                "bluesky" => entry[1] += views,
                "linkedin" => entry[2] += views,
                "threads" => entry[3] += views,
                _ => { /* unknown platform — ignore */ }
            }
        }
        let views: Vec<Value> = by_day
            .iter()
            .map(|(d, v)| {
                json!({
                    "date": d,
                    "youtube": v[0],
                    "bluesky": v[1],
                    "linkedin": v[2],
                    "threads": v[3],
                })
            })
            .collect();

        // Top 10 clips by latest views per (clip_id, platform). Using a
        // single LEFT JOIN against clips so we can carry hook/job_id even
        // when analytics rows outlive a clip's manifest.
        let mut stmt2 = conn.prepare(
            "SELECT a.clip_id, a.platform, \
                    SUM(a.views) AS total_views, \
                    AVG(a.ctr) AS ctr, \
                    AVG(a.watch_pct) AS watch, \
                    c.hook, c.job_id, c.score, c.rank \
             FROM analytics a \
             LEFT JOIN clips c ON c.id = a.clip_id \
             GROUP BY a.clip_id, a.platform \
             ORDER BY total_views DESC \
             LIMIT 10",
        )?;
        let mut iter2 = stmt2.query([])?;
        let mut top_clips: Vec<Value> = Vec::new();
        let mut rank = 1i64;
        while let Some(row) = iter2.next()? {
            let _clip_id: String = row.get(0)?;
            let platform: String = row.get(1).unwrap_or_default();
            let total_views: i64 = row.get(2).unwrap_or(0);
            let ctr: f64 = row.get(3).unwrap_or(0.0);
            let watch: f64 = row.get(4).unwrap_or(0.0);
            let hook: Option<String> = row.get(5).ok();
            let job_id: Option<String> = row.get(6).ok();
            let score: f64 = row.get(7).unwrap_or(0.0);
            top_clips.push(json!({
                "rank": rank,
                "hook": hook.unwrap_or_default(),
                "episode": job_id.unwrap_or_default(),
                "platform": platform,
                "views": total_views,
                "ctr": (ctr * 100.0).round() / 100.0,
                "watchPct": (watch * 100.0).round() / 100.0,
                "score": score.round() as i64,
            }));
            rank += 1;
        }
        Ok(json!({ "views": views, "topClips": top_clips }))
    })
    .await
    .unwrap_or(Ok(json!({ "views": [], "topClips": [] })))
    .unwrap_or(json!({ "views": [], "topClips": [] }));
    Json(body)
}

/// Project the most-recent job's status onto the dashboard's 8 pipeline
/// stages so the Pipeline Architecture card lights up live.
///
/// Mapping (internal FSM → which stage is currently "active"):
///   pending     → ingest
///   transcribed → features  (audio + transcribe are done; features is what
///                            runs next inside the pipeline)
///   ranked      → render
///   rendered    → post
///   posted      → post (active until done)
///   done        → all done
///   failed      → the active stage of the *previous* status becomes "error"
///
/// "Most recent" = highest `updated_at`, regardless of status. If there are
/// no jobs in the DB yet, every stage is `idle`.
async fn pipeline_status(State(state): State<Arc<AppState>>) -> Json<Vec<Value>> {
    const STAGES: &[(&str, &str, &str)] = &[
        ("ingest", "Ingest", "Gmail/Drive"),
        ("download", "Download", "Drive API / yt-dlp"),
        ("audio", "Audio Extract", "ffmpeg"),
        ("transcribe", "Transcribe", "Whisper"),
        ("features", "Feature Extract", "VAD/Prosody/Embed"),
        ("rank", "Rank", "LLM + VLM"),
        ("render", "Render", "ffmpeg"),
        ("post", "Post", "Platforms"),
    ];

    let conn = state.storage.conn();
    let (status, error_present): (String, bool) =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(String, bool)> {
            let conn = conn.blocking_lock();
            conn.query_row(
                "SELECT status, COALESCE(error, '') FROM jobs ORDER BY updated_at DESC LIMIT 1",
                [],
                |r| {
                    let s: String = r.get(0)?;
                    let e: String = r.get(1)?;
                    Ok((s, !e.is_empty()))
                },
            )
            .or_else(|_| Ok::<_, anyhow::Error>((String::new(), false)))
        })
        .await
        .unwrap_or(Ok((String::new(), false)))
        .unwrap_or((String::new(), false));

    // Map status string → index of the currently-active stage (0-based).
    // None = no active stage (every stage idle, or all done).
    let active_idx: Option<usize> = match status.as_str() {
        "pending" => Some(0),     // ingest
        "transcribed" => Some(4), // features (audio+transcribe done)
        "ranked" => Some(6),      // render
        "rendered" => Some(7),    // post
        "posted" => Some(7),      // still post until done
        "done" => None,           // all stages done
        "failed" => Some(0),      // best-effort; we don't know which gate failed
        _ => None,                // no jobs at all
    };

    let is_failed = status == "failed" || error_present;
    let all_done = status == "done";

    let stages: Vec<Value> = STAGES
        .iter()
        .enumerate()
        .map(|(i, (id, label, sublabel))| {
            let status_str = if all_done {
                "done"
            } else if is_failed && active_idx == Some(i) {
                "error"
            } else if let Some(active) = active_idx {
                if i < active {
                    "done"
                } else if i == active {
                    "active"
                } else {
                    "idle"
                }
            } else {
                "idle"
            };
            json!({
                "id": id,
                "label": label,
                "sublabel": sublabel,
                "status": status_str,
            })
        })
        .collect();

    Json(stages)
}

/// Aggregate cost across all jobs in the DB.
async fn cost(State(state): State<Arc<AppState>>) -> Json<Value> {
    let conn = state.storage.conn();
    let total_cents: i64 = tokio::task::spawn_blocking(move || -> anyhow::Result<i64> {
        let conn = conn.blocking_lock();
        let total: i64 = conn
            .query_row("SELECT COALESCE(SUM(cost_cents), 0) FROM jobs", [], |r| {
                r.get(0)
            })
            .unwrap_or(0);
        Ok(total)
    })
    .await
    .unwrap_or(Ok(0))
    .unwrap_or(0);

    Json(json!({
        "total": (total_cents as f64) / 100.0,
        "breakdown": { "stt": 0.0, "chat": 0.0, "embeddings": 0.0, "vlm": 0.0 },
        "budget": 50.0,
        "dailyBurn": 0.0,
    }))
}

/// Inspect env vars to report which platform integrations are configured.
async fn platforms() -> Json<Vec<Value>> {
    fn env_set(key: &str) -> bool {
        std::env::var(key).map(|v| !v.is_empty()).unwrap_or(false)
    }

    let youtube_status = if env_set("GOOGLE_REFRESH_TOKEN") {
        "connected"
    } else {
        "not_configured"
    };
    let bluesky_status = if env_set("BLUESKY_HANDLE") && env_set("BLUESKY_APP_PASSWORD") {
        "connected"
    } else {
        "not_configured"
    };
    let ayrshare_status = if env_set("AYRSHARE_API_KEY") {
        "connected"
    } else {
        "not_configured"
    };

    Json(vec![
        json!({
            "id": "youtube",
            "name": "YouTube Shorts",
            "icon": "YT",
            "status": youtube_status,
            "handle": std::env::var("YOUTUBE_CHANNEL_HANDLE").ok(),
            "color": "#ef4444",
            "totalPosts": 0,
            "totalViews": 0,
            "avgCtr": 0,
            "avgWatch": 0,
            "lastPost": "Never",
        }),
        json!({
            "id": "bluesky",
            "name": "Bluesky",
            "icon": "BS",
            "status": bluesky_status,
            "handle": std::env::var("BLUESKY_HANDLE").ok(),
            "color": "#0085ff",
            "totalPosts": 0,
            "totalViews": 0,
            "avgCtr": 0,
            "avgWatch": 0,
            "lastPost": "Never",
        }),
        json!({
            "id": "tiktok",
            "name": "TikTok",
            "icon": "TT",
            "status": ayrshare_status,
            "handle": null,
            "color": "#ff0050",
            "totalPosts": 0,
            "totalViews": 0,
            "avgCtr": 0,
            "avgWatch": 0,
            "lastPost": "Never",
        }),
        json!({
            "id": "instagram",
            "name": "Instagram Reels",
            "icon": "IG",
            "status": ayrshare_status,
            "handle": null,
            "color": "#e1306c",
            "totalPosts": 0,
            "totalViews": 0,
            "avgCtr": 0,
            "avgWatch": 0,
            "lastPost": "Never",
        }),
        json!({
            "id": "linkedin",
            "name": "LinkedIn",
            "icon": "LI",
            "status": "not_configured",
            "handle": null,
            "color": "#0077b5",
            "totalPosts": 0,
            "totalViews": 0,
            "avgCtr": 0,
            "avgWatch": 0,
            "lastPost": "Never",
        }),
        json!({
            "id": "threads",
            "name": "Threads",
            "icon": "TH",
            "status": "not_configured",
            "handle": null,
            "color": "#101010",
            "totalPosts": 0,
            "totalViews": 0,
            "avgCtr": 0,
            "avgWatch": 0,
            "lastPost": "Never",
        }),
    ])
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/shows", get(shows))
        .route("/episodes", get(episodes))
        .route("/platforms", get(platforms))
        .route("/trends", get(trends))
        .route("/agents", get(agents))
        .route("/cost", get(cost))
        .route("/analytics", get(analytics))
        .route("/pipeline/status", get(pipeline_status))
}
