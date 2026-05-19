//! Clip management API endpoints.
//!
//! Provides CRUD operations for clips, status transitions (approve/veto),
//! posting to platforms, and bulk actions.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use super::AppState;

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

/// GET /clips
async fn list_clips(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListClipsQuery>,
) -> impl IntoResponse {
    match state.storage.list_clips(query.status.as_deref()).await {
        Ok(clips) => (StatusCode::OK, Json(serde_json::json!({ "clips": clips }))).into_response(),
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
            (StatusCode::OK, Json(serde_json::json!({ "clip": clip }))).into_response()
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
