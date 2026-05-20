//! Clip management API endpoints.
//!
//! Provides CRUD operations for clips, status transitions (approve/veto),
//! posting to platforms, and bulk actions.

use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::AppState;
use crate::storage::ClipDetail;

#[derive(Deserialize)]
pub struct ListClipsQuery {
    pub status: Option<String>,
}

#[derive(Deserialize)]
pub struct PatchClipBody {
    pub hook: Option<String>,
    pub status: Option<String>,
    pub social_copy_json: Option<String>,
}

#[derive(Deserialize)]
pub struct PostClipBody {
    pub platforms: Vec<String>,
}

#[derive(Deserialize)]
pub struct BulkActionBody {
    pub clip_ids: Vec<String>,
    pub action: String, // "approve" | "veto" | "post"
    pub platforms: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn err_json(status: StatusCode, msg: impl Into<String>) -> impl IntoResponse {
    (status, Json(ErrorResponse { error: msg.into() }))
}

/// Convert an absolute file path under `{work_dir}/clipper/...` into a URL
/// served by the `/media/clipper/*` static-file mount. Returns `None` if the
/// path doesn't live under the work dir (defensive — DB shouldn't contain
/// such rows but we don't trust them).
fn media_url_from_path(work_dir: &str, abs_path: &str) -> Option<String> {
    let work_clipper = StdPath::new(work_dir).join("clipper");
    let p = StdPath::new(abs_path);
    p.strip_prefix(&work_clipper)
        .ok()
        .map(|rel| format!("/media/clipper/{}", rel.to_string_lossy()))
}

/// Find and parse the manifest.json next to any rendered variant for a clip.
/// Returns the entire manifest plus the index of this clip within it.
fn load_manifest_for_clip(clip: &ClipDetail) -> Option<(Value, usize)> {
    let render = clip.renders.first()?;
    let render_path = PathBuf::from(&render.path);
    let manifest_path = render_path.parent()?.join("manifest.json");
    let bytes = std::fs::read(&manifest_path).ok()?;
    let manifest: Value = serde_json::from_slice(&bytes).ok()?;
    let clips_arr = manifest.get("clips")?.as_array()?;
    let rank = clip.rank?;
    let idx = clips_arr
        .iter()
        .position(|c| c.get("rank").and_then(|v| v.as_i64()) == Some(rank))?;
    Some((manifest, idx))
}

/// Enrich a single ClipDetail into a JSON object that includes manifest data
/// the dashboard needs (variants with media URLs, cover thumbnail URL, social
/// copy per platform, overlay hook). Falls back gracefully when manifest is
/// missing — the dashboard still gets all the DB fields.
fn enrich_clip(clip: &ClipDetail, work_dir: &str) -> Value {
    // Variants from clip_renders rows. Prefer the DB-stored object-storage URL
    // (set when R2_* envs are configured and upload succeeded); fall back to
    // the local `/media/clipper/*` proxy path derived from the absolute path.
    let variants: Vec<Value> = clip
        .renders
        .iter()
        .map(|r| {
            let url = r
                .url
                .clone()
                .or_else(|| media_url_from_path(work_dir, &r.path));
            serde_json::json!({
                "variant": r.variant,
                "path": r.path,
                "url": url,
                "bytes": r.bytes,
                "duration_ms": r.duration_ms,
            })
        })
        .collect();

    // Prefer the DB-stored cover URL; fall back to deriving one from the
    // stored local path. Manifest scan further down can also fill it.
    let cover_url_initial = clip.cover_url.clone().or_else(|| {
        clip.cover_path
            .as_deref()
            .and_then(|p| media_url_from_path(work_dir, p))
    });

    let mut extra = serde_json::json!({
        "variants": variants,
        "cover_url": cover_url_initial,
        "social": Value::Null,
        "overlay_hook": Value::Null,
        "hook_source": Value::Null,
    });

    if let Some((manifest, idx)) = load_manifest_for_clip(clip) {
        if let Some(m_clip) = manifest.get("clips").and_then(|c| c.get(idx)) {
            // Only fill cover from the manifest if the DB didn't already provide one.
            if extra["cover_url"].is_null() {
                if let Some(cover) = m_clip.get("cover_frame") {
                    if let Some(cover_abs) = cover.get("abs_path").and_then(|v| v.as_str()) {
                        if let Some(url) = media_url_from_path(work_dir, cover_abs) {
                            extra["cover_url"] = Value::String(url);
                        }
                    }
                }
            }
            // Social copy + overlay hook (per-clip from manifest, not DB)
            if let Some(social) = m_clip.get("social") {
                extra["social"] = social.clone();
            }
            if let Some(overlay) = m_clip.get("overlay_hook") {
                extra["overlay_hook"] = overlay.clone();
            }
            if let Some(src) = m_clip.get("hook_source") {
                extra["hook_source"] = src.clone();
            }
        }
    }

    // Base clip JSON + merge extras
    let mut base = serde_json::to_value(clip).unwrap_or_else(|_| serde_json::json!({}));
    if let Value::Object(ref mut map) = base {
        if let Value::Object(extra_map) = extra {
            for (k, v) in extra_map {
                map.insert(k, v);
            }
        }
    }
    base
}

/// GET /clips
async fn list_clips(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListClipsQuery>,
) -> impl IntoResponse {
    match state.storage.list_clips(query.status.as_deref()).await {
        Ok(clips) => {
            let work_dir = state.work_dir.clone();
            let enriched: Vec<Value> = clips.iter().map(|c| enrich_clip(c, &work_dir)).collect();
            (
                StatusCode::OK,
                Json(serde_json::json!({ "clips": enriched })),
            )
                .into_response()
        }
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// GET /clips/:id
async fn get_clip(
    State(state): State<Arc<AppState>>,
    Path(clip_id): Path<String>,
) -> impl IntoResponse {
    match state.storage.get_clip(&clip_id).await {
        Ok(Some(clip)) => {
            let enriched = enrich_clip(&clip, &state.work_dir);
            (
                StatusCode::OK,
                Json(serde_json::json!({ "clip": enriched })),
            )
                .into_response()
        }
        Ok(None) => err_json(StatusCode::NOT_FOUND, "clip not found").into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// PATCH /clips/:id
async fn patch_clip(
    State(state): State<Arc<AppState>>,
    Path(clip_id): Path<String>,
    Json(body): Json<PatchClipBody>,
) -> impl IntoResponse {
    if let Some(ref hook) = body.hook {
        if let Err(e) = state.storage.update_clip_hook(&clip_id, hook).await {
            return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
        }
    }
    if let Some(ref status) = body.status {
        let valid = ["generated", "approved", "vetoed", "posted"];
        if !valid.contains(&status.as_str()) {
            return err_json(StatusCode::BAD_REQUEST, format!("invalid status: {status}"))
                .into_response();
        }
        if let Err(e) = state.storage.update_clip_status(&clip_id, status).await {
            return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
        }
    }
    if let Some(ref social_copy) = body.social_copy_json {
        if let Err(e) = state
            .storage
            .update_clip_social_copy(&clip_id, social_copy)
            .await
        {
            return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
        }
    }

    // Return the updated clip
    match state.storage.get_clip(&clip_id).await {
        Ok(Some(clip)) => {
            (StatusCode::OK, Json(serde_json::json!({ "clip": clip }))).into_response()
        }
        Ok(None) => err_json(StatusCode::NOT_FOUND, "clip not found").into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// POST /clips/:id/approve
async fn approve_clip(
    State(state): State<Arc<AppState>>,
    Path(clip_id): Path<String>,
) -> impl IntoResponse {
    match state.storage.update_clip_status(&clip_id, "approved").await {
        Ok(true) => match state.storage.get_clip(&clip_id).await {
            Ok(Some(clip)) => {
                (StatusCode::OK, Json(serde_json::json!({ "clip": clip }))).into_response()
            }
            _ => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response(),
        },
        Ok(false) => err_json(StatusCode::NOT_FOUND, "clip not found").into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// POST /clips/:id/veto
async fn veto_clip(
    State(state): State<Arc<AppState>>,
    Path(clip_id): Path<String>,
) -> impl IntoResponse {
    match state.storage.update_clip_status(&clip_id, "vetoed").await {
        Ok(true) => match state.storage.get_clip(&clip_id).await {
            Ok(Some(clip)) => {
                (StatusCode::OK, Json(serde_json::json!({ "clip": clip }))).into_response()
            }
            _ => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response(),
        },
        Ok(false) => err_json(StatusCode::NOT_FOUND, "clip not found").into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// POST /clips/:id/post
///
/// Marks the clip as posted and records the platform request.
/// Actual platform API calls require the full pipeline config (OAuth tokens, etc.)
/// which is only available in clipper mode. This endpoint updates the database
/// to reflect the user's intent; the posting worker picks up approved/queued clips.
async fn post_clip(
    State(state): State<Arc<AppState>>,
    Path(clip_id): Path<String>,
    Json(body): Json<PostClipBody>,
) -> impl IntoResponse {
    if body.platforms.is_empty() {
        return err_json(StatusCode::BAD_REQUEST, "platforms list cannot be empty").into_response();
    }

    // Update clip status to posted
    if let Err(e) = state.storage.update_clip_status(&clip_id, "posted").await {
        return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
    }

    // Record pending post entries for each requested platform
    for platform in &body.platforms {
        if let Err(e) = state
            .storage
            .insert_post(&clip_id, platform, "pending", None, None, None, None)
            .await
        {
            tracing::warn!(clip_id, platform, error = %e, "failed to insert pending post");
        }
    }

    match state.storage.get_clip(&clip_id).await {
        Ok(Some(clip)) => {
            (StatusCode::OK, Json(serde_json::json!({ "clip": clip }))).into_response()
        }
        Ok(None) => err_json(StatusCode::NOT_FOUND, "clip not found").into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// POST /clips/bulk
async fn bulk_action(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BulkActionBody>,
) -> impl IntoResponse {
    if body.clip_ids.is_empty() {
        return err_json(StatusCode::BAD_REQUEST, "clip_ids cannot be empty").into_response();
    }

    let status = match body.action.as_str() {
        "approve" => "approved",
        "veto" => "vetoed",
        "post" => "posted",
        other => {
            return err_json(StatusCode::BAD_REQUEST, format!("unknown action: {other}"))
                .into_response();
        }
    };

    match state
        .storage
        .bulk_update_clip_status(&body.clip_ids, status)
        .await
    {
        Ok(count) => {
            // If posting, also insert pending post rows
            if body.action == "post" {
                if let Some(ref platforms) = body.platforms {
                    for clip_id in &body.clip_ids {
                        for platform in platforms {
                            state
                                .storage
                                .insert_post(clip_id, platform, "pending", None, None, None, None)
                                .await
                                .ok();
                        }
                    }
                }
            }
            (
                StatusCode::OK,
                Json(serde_json::json!({ "updated": count })),
            )
                .into_response()
        }
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// Build the clips sub-router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/clips", get(list_clips))
        .route("/clips/bulk", post(bulk_action))
        .route("/clips/{id}", get(get_clip).patch(patch_clip))
        .route("/clips/{id}/approve", post(approve_clip))
        .route("/clips/{id}/veto", post(veto_clip))
        .route("/clips/{id}/post", post(post_clip))
}
