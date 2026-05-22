//! Job-management API endpoints.
//!
//! - `POST   /api/jobs` — create a new clipper job from either a multipart file
//!   upload or a JSON body with a `video_url`. The video lands at
//!   `WORK_DIR/uploads/<job_id>/<filename>` and a row is inserted into `jobs`
//!   with `status='pending'` so the background worker picks it up.
//! - `GET    /api/jobs` — list all jobs (most-recent first). Returns the
//!   dashboard's `Job[]` shape so `useJobs()` consumes it directly.
//! - `GET    /api/jobs/:id` — one job's `Job` payload, with `clips` summary.
//! - `POST   /api/jobs/:id/retry` — flip a `failed` job back to `pending` so
//!   the worker re-claims it.

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

use super::AppState;
use crate::storage::JobRow;

/// JSON body for URL-based job creation.
#[derive(Deserialize)]
pub struct CreateJobBody {
    /// HTTPS URL (Drive share link, direct mp4, etc.) — required when not
    /// uploading a file via multipart.
    pub video_url: Option<String>,
    /// Human-readable name for the source. Defaults to the URL's filename
    /// segment or the uploaded file's name.
    pub media_name: Option<String>,
    /// Optional show slug. When set, the clipper loads per-show prompt
    /// overrides from `SHOWS_DIR/<slug>/`.
    pub show_slug: Option<String>,
    /// Per-job overrides (clip_top_k, render_formats, skip_ranges, …).
    /// Worker reads this before invoking the pipeline.
    pub config: Option<serde_json::Value>,
}

fn err_json(status: StatusCode, msg: impl Into<String>) -> impl IntoResponse {
    (status, Json(json!({ "error": msg.into() })))
}

/// Generate a short-ish job id: `dashboard_<unix_secs>_<rand6>`. Stable and
/// safe for use as a directory name.
fn new_job_id() -> String {
    use rand::Rng;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let rand: u32 = rand::thread_rng().r#gen::<u32>() & 0xFFFFFF;
    format!("dashboard_{now}_{rand:06x}")
}

/// POST /api/jobs — accept either a multipart file upload or a JSON body
/// describing a URL to fetch. Returns 201 with the created job id.
///
/// Content negotiation is by Content-Type:
///   - `multipart/form-data` → expects a `file` part (binary mp4/etc.)
///     and optional text parts `media_name`, `show_slug`, `config_json`.
///   - anything else → parse as JSON `CreateJobBody`.
pub async fn create_job(
    State(state): State<Arc<AppState>>,
    request: Request,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let content_type = request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    if content_type.starts_with("multipart/form-data") {
        create_from_multipart(state, request).await
    } else {
        create_from_json(state, request.into_body()).await
    }
}

async fn create_from_multipart(
    state: Arc<AppState>,
    request: Request,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    let mut multipart = Multipart::from_request(request, &()).await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("multipart parse: {e}")})),
        )
    })?;

    let job_id = new_job_id();
    let uploads_root = PathBuf::from(&state.work_dir).join("uploads").join(&job_id);
    if let Err(e) = tokio::fs::create_dir_all(&uploads_root).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("mkdir {}: {e}", uploads_root.display())})),
        ));
    }

    let mut media_name: Option<String> = None;
    let mut show_slug: Option<String> = None;
    let mut config_json: Option<String> = None;
    let mut local_path: Option<PathBuf> = None;
    let mut original_filename: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("multipart field: {e}")})),
        )
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let fname = field
                    .file_name()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{job_id}.mp4"));
                let safe = sanitize_filename(&fname);
                original_filename = Some(safe.clone());
                let path = uploads_root.join(&safe);
                // axum 0.8's Field doesn't expose a streaming API in all
                // builds; load into memory and write once. For very large
                // uploads consider switching to a stream extractor.
                let bytes = field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": format!("read upload body: {e}")})),
                    )
                })?;
                let mut file = tokio::fs::File::create(&path).await.map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": format!("create file {}: {e}", path.display())})),
                    )
                })?;
                file.write_all(&bytes).await.map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": format!("write file: {e}")})),
                    )
                })?;
                file.flush().await.ok();
                local_path = Some(path);
            }
            "media_name" => media_name = Some(field.text().await.unwrap_or_default()),
            "show_slug" => show_slug = Some(field.text().await.unwrap_or_default()),
            "config_json" => config_json = Some(field.text().await.unwrap_or_default()),
            _ => { /* ignore unknown */ }
        }
    }

    let local_path = local_path.ok_or((
        StatusCode::BAD_REQUEST,
        Json(json!({"error": "multipart upload missing required `file` part"})),
    ))?;
    let local_path_str = local_path.display().to_string();
    let media_name = media_name
        .filter(|s| !s.is_empty())
        .or(original_filename.clone())
        .unwrap_or_else(|| format!("{job_id}.mp4"));

    state
        .storage
        .enqueue_job(
            &job_id,
            show_slug.as_deref().filter(|s| !s.is_empty()),
            Some(&media_name),
            Some(&local_path_str),
            None,
            config_json.as_deref().filter(|s| !s.is_empty()),
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("enqueue: {e:#}")})),
            )
        })?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": job_id,
            "status": "pending",
            "media": media_name,
            "local_path": local_path_str,
        })),
    ))
}

async fn create_from_json(
    state: Arc<AppState>,
    body: axum::body::Body,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    let bytes = axum::body::to_bytes(body, 1024 * 1024).await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("read body: {e}")})),
        )
    })?;
    let parsed: CreateJobBody = serde_json::from_slice(&bytes).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("parse JSON: {e}")})),
        )
    })?;

    let url = parsed.video_url.unwrap_or_default();
    if url.is_empty() {
        return Err(err_response(
            StatusCode::BAD_REQUEST,
            "video_url is required when not uploading a file via multipart",
        ));
    }

    let job_id = new_job_id();
    let media_name = parsed
        .media_name
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| filename_from_url(&url).unwrap_or_else(|| format!("{job_id}.mp4")));
    let config_json = parsed
        .config
        .as_ref()
        .map(|v| v.to_string())
        .filter(|s| !s.is_empty() && s != "null");

    state
        .storage
        .enqueue_job(
            &job_id,
            parsed.show_slug.as_deref().filter(|s| !s.is_empty()),
            Some(&media_name),
            None,
            Some(&url),
            config_json.as_deref(),
        )
        .await
        .map_err(|e| err_response(StatusCode::INTERNAL_SERVER_ERROR, format!("enqueue: {e:#}")))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": job_id,
            "status": "pending",
            "media": media_name,
            "source_url": url,
        })),
    ))
}

fn err_response(
    status: StatusCode,
    msg: impl Into<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(json!({ "error": msg.into() })))
}

fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "upload.mp4".to_string()
    } else {
        cleaned
    }
}

fn filename_from_url(url: &str) -> Option<String> {
    let no_query = url.split('?').next()?;
    let last = no_query.rsplit('/').next()?;
    if last.is_empty() {
        None
    } else {
        Some(sanitize_filename(last))
    }
}

// ── List / detail / retry ───────────────────────────────────────────────────

/// GET /api/jobs — full list (newest first), in the dashboard's `Job` shape.
async fn list_jobs(State(state): State<Arc<AppState>>) -> Json<Vec<Value>> {
    let conn = state.storage.conn();
    let rows: Vec<Value> = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Value>> {
        let conn = conn.blocking_lock();
        let mut stmt = conn.prepare(
            "SELECT id, show_slug, media_name, status, cost_cents, created_at, updated_at, error \
             FROM jobs ORDER BY created_at DESC LIMIT 200",
        )?;
        let mut iter = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = iter.next()? {
            let id: String = row.get(0)?;
            let show_slug: Option<String> = row.get(1)?;
            let media_name: Option<String> = row.get(2)?;
            let status: String = row.get(3)?;
            let cost_cents: i64 = row.get(4).unwrap_or(0);
            let created_at: i64 = row.get(5).unwrap_or(0);
            let updated_at: i64 = row.get(6).unwrap_or(0);
            let error: Option<String> = row.get(7).ok();

            let clips_generated: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM clips WHERE job_id = ?1",
                    rusqlite::params![&id],
                    |r| r.get(0),
                )
                .unwrap_or(0);

            out.push(map_job_json(
                &id,
                show_slug.as_deref(),
                media_name.as_deref(),
                &status,
                cost_cents,
                created_at,
                updated_at,
                error.as_deref(),
                clips_generated,
            ));
        }
        Ok(out)
    })
    .await
    .unwrap_or(Ok(vec![]))
    .unwrap_or_default();

    Json(rows)
}

/// GET /api/jobs/:id — single job with a small `clips` summary.
async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let row = state.storage.get_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("storage: {e:#}")})),
        )
    })?;
    let Some(row) = row else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Job not found"})),
        ));
    };

    let conn = state.storage.conn();
    let job_id = id.clone();
    let (clips_generated, clips_summary) = tokio::task::spawn_blocking(
        move || -> anyhow::Result<(i64, Vec<Value>)> {
            let conn = conn.blocking_lock();
            let mut stmt = conn.prepare(
                "SELECT id, start_ms, end_ms, rank, score, hook \
                 FROM clips WHERE job_id = ?1 ORDER BY rank ASC NULLS LAST",
            )?;
            let mut iter = stmt.query(rusqlite::params![&job_id])?;
            let mut summary = Vec::new();
            while let Some(c) = iter.next()? {
                let cid: String = c.get(0)?;
                let start_ms: i64 = c.get(1).unwrap_or(0);
                let end_ms: i64 = c.get(2).unwrap_or(0);
                let rank: Option<i64> = c.get(3).ok();
                let score: Option<f64> = c.get(4).ok();
                let hook: Option<String> = c.get(5).ok();
                summary.push(json!({
                    "id": cid,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "rank": rank,
                    "score": score,
                    "hook": hook,
                }));
            }
            let count = summary.len() as i64;
            Ok((count, summary))
        },
    )
    .await
    .unwrap_or(Ok((0, vec![])))
    .unwrap_or((0, vec![]));

    let mut body = map_job_json(
        &row.id,
        row.show_slug.as_deref(),
        row.media_name.as_deref(),
        row.status.as_str(),
        row.cost_cents,
        row.created_at,
        row.updated_at,
        row.error.as_deref(),
        clips_generated,
    );
    if let Some(obj) = body.as_object_mut() {
        obj.insert("clips".into(), Value::Array(clips_summary));
    }
    Ok(Json(body))
}

/// POST /api/jobs/:id/retry — flip a failed job back to `pending` so the
/// worker re-claims it on its next poll. Returns the updated `Job` payload.
async fn retry_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Check the row exists + is in 'failed' before we touch it so we can
    // return the right status code (404 vs 409).
    let existing = state.storage.get_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("storage: {e:#}")})),
        )
    })?;
    let Some(row) = existing else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Job not found"})),
        ));
    };
    if row.status.as_str() != "failed" {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({"error": "Only failed jobs can be retried"})),
        ));
    }

    state.storage.retry_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("retry: {e:#}")})),
        )
    })?;

    // Return the refreshed row.
    let refreshed = state
        .storage
        .get_job(&id)
        .await
        .ok()
        .flatten()
        .unwrap_or(row);
    let body = map_job_json_from_row(&refreshed, 0);
    Ok(Json(body))
}

fn map_job_json_from_row(row: &JobRow, clips_generated: i64) -> Value {
    map_job_json(
        &row.id,
        row.show_slug.as_deref(),
        row.media_name.as_deref(),
        row.status.as_str(),
        row.cost_cents,
        row.created_at,
        row.updated_at,
        row.error.as_deref(),
        clips_generated,
    )
}

/// Shape one job row in the dashboard `Job` interface (camelCase, with
/// dashboard-friendly status / stage / progress derived from the DB FSM via
/// the shared [`crate::events::dashboard_view`]).
#[allow(clippy::too_many_arguments)]
fn map_job_json(
    id: &str,
    show_slug: Option<&str>,
    media_name: Option<&str>,
    status: &str,
    cost_cents: i64,
    created_at: i64,
    updated_at: i64,
    error: Option<&str>,
    clips_generated: i64,
) -> Value {
    let created_iso = chrono::DateTime::from_timestamp(created_at, 0)
        .map(|d| d.to_rfc3339())
        .unwrap_or_default();
    let duration_secs = (updated_at - created_at).max(0);
    let duration = format!("{}m {}s", duration_secs / 60, duration_secs % 60);

    // Shared FSM → (status, stage, progress) mapping lives in events.rs so
    // the Jobs card, the Pipeline card, and WS JobUpdate messages all stay
    // in lockstep.
    let (dash_status, stage_label, progress) = crate::events::dashboard_view(status);

    json!({
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
    })
}

/// Query params for `DELETE /api/jobs/:id`.
#[derive(Deserialize, Default)]
pub struct DeleteJobQuery {
    /// When `true`, also remove the on-disk render dir + the uploaded source
    /// file. Default `false` keeps files (DB-only delete) so accidental clicks
    /// are recoverable.
    #[serde(default)]
    pub purge: bool,
}

/// DELETE /api/jobs/:id[?purge=true]
async fn delete_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<DeleteJobQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Look up the row first so we can find on-disk artifacts to purge before
    // the row goes away.
    let row = state.storage.get_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("storage: {e:#}")})),
        )
    })?;
    let Some(row) = row else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Job not found"})),
        ));
    };

    let n = state.storage.delete_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("delete: {e:#}")})),
        )
    })?;

    let mut purged_paths: Vec<String> = Vec::new();
    if q.purge {
        // Render output dir: `${WORK_DIR}/clipper/<sanitized media>/`. We
        // don't know the exact timestamp subdir without scanning, so we
        // remove the whole media folder. The dashboard surfaces this as
        // irreversible.
        if let Some(media) = row.media_name.as_deref() {
            let sanitized = sanitize_filename(media);
            let dir = PathBuf::from(&state.work_dir).join("clipper").join(&sanitized);
            if tokio::fs::metadata(&dir).await.is_ok() {
                if let Err(e) = tokio::fs::remove_dir_all(&dir).await {
                    tracing::warn!(path = %dir.display(), error = ?e, "delete_job purge: render dir removal failed");
                } else {
                    purged_paths.push(dir.display().to_string());
                }
            }
        }
        // Upload dir (for jobs created by file upload via the dashboard):
        // `${WORK_DIR}/uploads/<job_id>/`.
        let uploads = PathBuf::from(&state.work_dir).join("uploads").join(&id);
        if tokio::fs::metadata(&uploads).await.is_ok() {
            if let Err(e) = tokio::fs::remove_dir_all(&uploads).await {
                tracing::warn!(path = %uploads.display(), error = ?e, "delete_job purge: uploads dir removal failed");
            } else {
                purged_paths.push(uploads.display().to_string());
            }
        }
        // R2 object cleanup is out of scope here — we don't currently track
        // the uploaded keys per job in the DB. Documented as a follow-up.
    }

    Ok(Json(json!({
        "deleted": n,
        "id": id,
        "purged": q.purge,
        "purged_paths": purged_paths,
    })))
}

/// POST /api/jobs/:id/cancel — only valid while the job is still `pending`.
async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let existing = state.storage.get_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("storage: {e:#}")})),
        )
    })?;
    let Some(row) = existing else {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Job not found"})),
        ));
    };
    if row.status.as_str() != "pending" {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({
                "error": "Only pending jobs can be cancelled (mid-flight cancel not yet supported)",
                "current_status": row.status.as_str(),
            })),
        ));
    }
    state.storage.cancel_pending_job(&id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("cancel: {e:#}")})),
        )
    })?;
    let refreshed = state
        .storage
        .get_job(&id)
        .await
        .ok()
        .flatten()
        .unwrap_or(row);
    Ok(Json(map_job_json_from_row(&refreshed, 0)))
}

/// POST /api/jobs/:id/rerun — clones the job to a new id pointing at the same
/// source, status=pending. Returns the new job's payload.
async fn rerun_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<Value>)> {
    let new_id = state
        .storage
        .clone_job_for_rerun(&id)
        .await
        .map_err(|e| {
            // Surface "not found" distinctly so the dashboard can show the
            // right error.
            let msg = format!("{e:#}");
            let code = if msg.contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (code, Json(json!({"error": msg})))
        })?;
    let refreshed = state.storage.get_job(&new_id).await.ok().flatten();
    let body = match refreshed {
        Some(row) => map_job_json_from_row(&row, 0),
        // Defensive — the row was just inserted; if it's already gone,
        // surface the new id at least so the dashboard can navigate.
        None => json!({"id": new_id, "status": "pending"}),
    };
    Ok((StatusCode::CREATED, Json(body)))
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        // 8 GiB upload ceiling — practical max for raw 4K episodes.
        .route(
            "/jobs",
            post(create_job)
                .get(list_jobs)
                .layer(DefaultBodyLimit::max(8 * 1024 * 1024 * 1024)),
        )
        .route("/jobs/{id}", get(get_job).delete(delete_job))
        .route("/jobs/{id}/retry", post(retry_job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/jobs/{id}/rerun", post(rerun_job))
}
