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

async fn pipeline_status() -> Json<Vec<Value>> {
    Json(vec![])
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

/// List jobs from the DB in the shape the dashboard's Job type expects.
async fn jobs(State(state): State<Arc<AppState>>) -> Json<Vec<Value>> {
    let conn = state.storage.conn();
    let rows: Vec<Value> = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Value>> {
        let conn = conn.blocking_lock();
        let mut stmt = conn.prepare(
            "SELECT id, show_slug, media_name, status, cost_cents, created_at, updated_at, error \
             FROM jobs ORDER BY created_at DESC LIMIT 200",
        )?;
        let mut rows = Vec::new();
        let mut iter = stmt.query([])?;
        while let Some(row) = iter.next()? {
            let id: String = row.get(0)?;
            let show_slug: Option<String> = row.get(1)?;
            let media_name: Option<String> = row.get(2)?;
            let status: String = row.get(3)?;
            let cost_cents: i64 = row.get(4).unwrap_or(0);
            let created_at: i64 = row.get(5).unwrap_or(0);
            let updated_at: i64 = row.get(6).unwrap_or(0);
            let error: Option<String> = row.get(7).ok();

            let created_iso = chrono::DateTime::from_timestamp(created_at, 0)
                .map(|d| d.to_rfc3339())
                .unwrap_or_default();
            let duration_secs = (updated_at - created_at).max(0);
            let duration = format!("{}m {}s", duration_secs / 60, duration_secs % 60);

            // Translate the DB's gate-by-gate status into a friendly
            // dashboard status + progress %. The clipper writes:
            //   pending → transcribed → ranked → rendered → posted → done
            // The dashboard's existing Job UI knows about
            //   pending / transcribing / rendering / done / failed,
            // so we collapse closely-related stages and surface a stage
            // string + percentage that's safe to render either way.
            let (dash_status, stage_label, progress) = match status.as_str() {
                "pending" => ("pending", "Queued", 5),
                "transcribed" => ("transcribing", "Transcribed", 30),
                "ranked" => ("rendering", "Ranked — generating clips", 55),
                "rendered" => ("rendering", "Rendered — uploading", 80),
                "posted" => ("rendering", "Posted to platforms", 95),
                "done" => ("done", "Complete", 100),
                "failed" => ("failed", "Failed", 0),
                other => ("pending", other, 5),
            };

            // Get clip count for this job (cheap join — small table).
            let clips_generated: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM clips WHERE job_id = ?1",
                    rusqlite::params![&id],
                    |r| r.get(0),
                )
                .unwrap_or(0);

            rows.push(json!({
                "id": id,
                "episodeId": null,
                "showId": show_slug,
                "media": media_name,
                "status": dash_status,
                "stage": stage_label,
                "progress": progress,
                "clipsGenerated": clips_generated,
                "postsSuccess": 0,
                "postsTotal": 0,
                "cost": (cost_cents as f64) / 100.0,
                "duration": duration,
                "created": created_iso,
                "error": error,
            }));
        }
        Ok(rows)
    })
    .await
    .unwrap_or(Ok(vec![]))
    .unwrap_or_default();

    Json(rows)
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
        .route("/jobs", get(jobs))
        .route("/platforms", get(platforms))
        .route("/trends", get(trends))
        .route("/agents", get(agents))
        .route("/cost", get(cost))
        .route("/analytics", get(analytics))
        .route("/pipeline/status", get(pipeline_status))
}
