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

async fn agents() -> Json<Vec<Value>> {
    Json(vec![])
}

async fn trends() -> Json<Value> {
    Json(json!({ "gdelt": [], "reddit": [], "google": [] }))
}

async fn analytics() -> Json<Value> {
    Json(json!({ "views": [], "topClips": [] }))
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
