//! Job-management API endpoints.
//!
//! - `POST /api/jobs` — create a new clipper job from either a multipart file
//!   upload or a JSON body with a `video_url`. The video lands at
//!   `WORK_DIR/uploads/<job_id>/<filename>` and a row is inserted into `jobs`
//!   with `status='pending'` so the background worker picks it up.
//!
//! The matching `GET /api/jobs` (list) and the worker loop that actually runs
//! the clipper pipeline live in `worker.rs` and `stubs.rs` respectively for
//! now — kept that way to minimize blast radius. Eventually this file can
//! absorb the list-jobs query too.

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
};
use serde::Deserialize;
use serde_json::json;
use tokio::io::AsyncWriteExt;

use super::AppState;

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

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        // 8 GiB upload ceiling — practical max for raw 4K episodes.
        .route(
            "/jobs",
            post(create_job).layer(DefaultBodyLimit::max(8 * 1024 * 1024 * 1024)),
        )
}
