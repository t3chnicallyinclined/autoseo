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

/// A single posting target. `account_id` is required for browser-backed
/// platforms that have multiple connected accounts; absent for API-backed
/// platforms (YouTube/Bluesky/IG-Graph/Ayrshare) which are single-account.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TargetSpec {
    pub platform: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
}

impl TargetSpec {
    /// Storage encoding: API-backed → `"youtube"`; browser-backed with account
    /// → `"x:tris_main"`. The posts table primary-keys on (clip_id, platform)
    /// so the encoding gives multi-account rows distinct keys without a
    /// schema migration.
    pub fn encoded_platform(&self) -> String {
        match self.account_id.as_deref() {
            Some(a) if !a.is_empty() => format!("{}:{a}", self.platform),
            _ => self.platform.clone(),
        }
    }
}

#[derive(Deserialize)]
pub struct PostClipBody {
    /// Legacy single-account form: list of platform names. Each becomes a
    /// `TargetSpec` with `account_id=None`. Use `targets` for the modern form.
    #[serde(default)]
    pub platforms: Vec<String>,
    /// Account-aware target list. Takes precedence over `platforms` when present.
    #[serde(default)]
    pub targets: Vec<TargetSpec>,
}

impl PostClipBody {
    pub fn resolved_targets(&self) -> Vec<TargetSpec> {
        if !self.targets.is_empty() {
            return self.targets.clone();
        }
        self.platforms
            .iter()
            .map(|p| TargetSpec {
                platform: p.clone(),
                account_id: None,
            })
            .collect()
    }
}

#[derive(Deserialize)]
pub struct BulkActionBody {
    pub clip_ids: Vec<String>,
    pub action: String, // "approve" | "veto" | "post"
    #[serde(default)]
    pub platforms: Option<Vec<String>>,
    #[serde(default)]
    pub targets: Option<Vec<TargetSpec>>,
}

impl BulkActionBody {
    pub fn resolved_targets(&self) -> Vec<TargetSpec> {
        if let Some(t) = &self.targets {
            return t.clone();
        }
        self.platforms
            .as_ref()
            .map(|ps| {
                ps.iter()
                    .map(|p| TargetSpec {
                        platform: p.clone(),
                        account_id: None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
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
        "scores": Value::Null,
        // Whether the per-clip manifest has the word-level transcript
        // needed by the dashboard's caption editor. False for clips
        // rendered before SCHEMA_VERSION 4 — the editor disables its
        // caption controls so the producer doesn't waste a click.
        "has_captionable_words": false,
    });

    if let Some((manifest, idx)) = load_manifest_for_clip(clip) {
        if let Some(m_clip) = manifest.get("clips").and_then(|c| c.get(idx)) {
            // Pre-flight signal: do we have words[] for this clip in the
            // manifest? Drives the dashboard's caption-editor enable state.
            let has_words = m_clip
                .get("words")
                .and_then(|v| v.as_array())
                .is_some_and(|arr| !arr.is_empty());
            extra["has_captionable_words"] = Value::Bool(has_words);
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
            // A/B score lineage (blended / llm / vlm / vlm_premium) per
            // manifest v3. DB only persists the blended score; the dashboard
            // needs the lane-level scores to surface why a clip was ranked.
            if let Some(scores) = m_clip.get("scores") {
                extra["scores"] = scores.clone();
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
    let targets = body.resolved_targets();
    if targets.is_empty() {
        return err_json(
            StatusCode::BAD_REQUEST,
            "must specify either `platforms` or `targets`",
        )
        .into_response();
    }

    if let Err(e) = state.storage.update_clip_status(&clip_id, "posted").await {
        return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
    }

    // Record a pending row per target. Encoded platform = `"x:tris_main"` for
    // browser-backed multi-account, plain `"youtube"` for API-backed.
    for target in &targets {
        let encoded = target.encoded_platform();
        if let Err(e) = state
            .storage
            .insert_post(&clip_id, &encoded, "pending", None, None, None, None)
            .await
        {
            tracing::warn!(clip_id, platform = encoded, error = %e, "failed to insert pending post");
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
            // If posting, also insert pending post rows. Honors the modern
            // `targets` field; falls back to legacy `platforms` for compat.
            if body.action == "post" {
                let targets = body.resolved_targets();
                if !targets.is_empty() {
                    for clip_id in &body.clip_ids {
                        for target in &targets {
                            let encoded = target.encoded_platform();
                            state
                                .storage
                                .insert_post(clip_id, &encoded, "pending", None, None, None, None)
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

/// Body for `POST /api/clips/{id}/recut` — operator-driven trim + optional
/// caption restyle.
///
/// Re-renders **every** variant that already exists for this clip so the
/// 9x16 / 1x1 / 16x9 renders stay consistent. Captions are re-generated
/// from the per-clip word timestamps stored in `manifest.json` (added by
/// SCHEMA_VERSION 4); when the manifest predates that or `burn_captions`
/// is false, the recut renders without subtitles.
#[derive(Deserialize, Default)]
pub struct RecutBody {
    pub start_secs: f64,
    pub end_secs: f64,
    /// Burn karaoke captions into the re-render. Defaults to `true` when
    /// the manifest has per-clip words available, `false` otherwise. The
    /// dashboard sends `false` only when the producer toggles captions
    /// off explicitly.
    #[serde(default)]
    pub burn_captions: Option<bool>,
    /// Per-clip caption style overrides. Layers on top of the global
    /// env-driven [`crate::captions::CaptionOverrides`] so the producer
    /// can pick a font / size / color for this one clip without changing
    /// global settings.
    #[serde(default)]
    pub caption_overrides: Option<crate::captions::CaptionOverrides>,
}

/// `POST /api/clips/{id}/recut` — re-render every existing variant of a
/// clip with new time bounds. Skips the LLM + ranker + social copy
/// stages; just runs ffmpeg against the original source video with the
/// new in/out points. All variants render in parallel up to
/// `RENDER_CONCURRENCY` so the operator's wait is dominated by the
/// slowest single variant, not the sum.
///
/// Captions are intentionally dropped on recut — the per-word ASS
/// timings were authored against the original window and can't be
/// shifted faithfully without re-deriving from the transcript. Operator
/// can re-run the full pipeline if they need captions to match the
/// trimmed boundaries.
async fn recut_clip(
    State(state): State<Arc<AppState>>,
    Path(clip_id): Path<String>,
    Json(body): Json<RecutBody>,
) -> impl IntoResponse {
    // ── Validate bounds ────────────────────────────────────────────
    if !body.start_secs.is_finite() || !body.end_secs.is_finite() {
        return err_json(StatusCode::BAD_REQUEST, "start/end must be finite")
            .into_response();
    }
    let new_start = body.start_secs.max(0.0);
    let new_end = body.end_secs;
    let duration = new_end - new_start;
    if duration < 3.0 {
        return err_json(
            StatusCode::BAD_REQUEST,
            format!("duration {duration:.1}s below 3s minimum"),
        )
        .into_response();
    }
    if duration > 300.0 {
        return err_json(
            StatusCode::BAD_REQUEST,
            format!("duration {duration:.1}s above 300s maximum"),
        )
        .into_response();
    }

    // ── Resolve clip + job + source video ──────────────────────────
    let clip = match state.storage.get_clip(&clip_id).await {
        Ok(Some(c)) => c,
        Ok(None) => return err_json(StatusCode::NOT_FOUND, "clip not found").into_response(),
        Err(e) => {
            return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
        }
    };
    let job = match state.storage.get_job(&clip.job_id).await {
        Ok(Some(j)) => j,
        Ok(None) => {
            return err_json(StatusCode::NOT_FOUND, "owning job not found").into_response();
        }
        Err(e) => {
            return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response();
        }
    };

    if clip.renders.is_empty() {
        return err_json(
            StatusCode::PRECONDITION_FAILED,
            "clip has no rendered variants to recut — re-run the full pipeline",
        )
        .into_response();
    }

    let source_path = match resolve_source_video(&job, &state.work_dir).await {
        Some(p) => p,
        None => {
            return err_json(
                StatusCode::PRECONDITION_FAILED,
                "original source video is no longer on disk \
                 (work/uploads/ cleanup or remote-only ingestion). \
                 Re-run the full pipeline instead of trimming.",
            )
            .into_response();
        }
    };

    // ── Build a render profile from current env (CRF/preset/audio) ──
    let cfg = {
        use clap::Parser as _;
        match crate::config::Config::try_parse_from(["autoseo"]) {
            Ok(c) => c,
            Err(e) => {
                return err_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("config snapshot failed: {e}"),
                )
                .into_response();
            }
        }
    };

    // ── Pull per-clip words from the manifest for caption regen ────
    // SCHEMA_VERSION 4 added a `words` array per clip with clip-relative
    // timestamps. Older manifests omit it; in that case captions are
    // silently dropped on recut (logged to stderr for debug). The
    // dashboard's caption editor surfaces this state via the
    // `has_captionable_words` flag in the response.
    let manifest_words: Vec<crate::align::AlignedWord> = load_manifest_for_clip(&clip)
        .as_ref()
        .and_then(|(m, idx)| {
            let arr = m.get("clips")?.get(*idx)?.get("words")?.as_array()?;
            Some(
                arr.iter()
                    .filter_map(|w| {
                        let text = w.get("text")?.as_str()?.to_string();
                        let start_secs = w.get("start_secs")?.as_f64()?;
                        let end_secs = w.get("end_secs")?.as_f64()?;
                        Some(crate::align::AlignedWord {
                            text,
                            start_secs,
                            end_secs,
                        })
                    })
                    .collect(),
            )
        })
        .unwrap_or_default();

    // Filter to the new bounds and re-anchor to the NEW clip start so the
    // ASS karaoke times line up with the trimmed video.
    let clip_orig_start_secs = clip.start_ms as f64 / 1000.0;
    let new_start_offset = (new_start - clip_orig_start_secs).max(0.0);
    let filtered_words: Vec<crate::align::AlignedWord> = manifest_words
        .iter()
        .filter(|w| {
            w.start_secs >= new_start_offset - 0.001
                && w.end_secs <= new_start_offset + duration + 0.5
        })
        .map(|w| crate::align::AlignedWord {
            text: w.text.clone(),
            start_secs: (w.start_secs - new_start_offset).max(0.0),
            end_secs: (w.end_secs - new_start_offset).max(0.0),
        })
        .collect();

    let burn = match body.burn_captions {
        Some(b) => b && !filtered_words.is_empty(),
        None => !filtered_words.is_empty(),
    };

    // ── Build per-clip caption overrides ────────────────────────────
    // Layer order: env defaults first, request overrides on top. The
    // per-show JSON path (used by the full pipeline) is skipped here —
    // dashboard edits are explicit operator overrides, not show-level.
    let mut caption_overrides = crate::captions::CaptionOverrides::default();
    caption_overrides.apply_env();
    if let Some(req) = body.caption_overrides {
        caption_overrides.merge_from(req);
    }

    // Emit a recut-start event so the dashboard's Activity Log + the
    // trim panel's in-place progress indicator both light up. We piggy-
    // back on the clip's owning job_id so the existing event subscriber
    // routes it correctly.
    let total_variants = clip.renders.len();
    let captions_will_apply = burn && !filtered_words.is_empty();
    let recut_msg = if captions_will_apply {
        format!(
            "recut: rendering {} variant(s) with captions ({} words)",
            total_variants,
            filtered_words.len()
        )
    } else if burn {
        format!(
            "recut: rendering {} variant(s) — manifest has no transcript words, captions skipped",
            total_variants
        )
    } else {
        format!(
            "recut: rendering {} variant(s) — captions off",
            total_variants
        )
    };
    state
        .event_bus
        .emit(crate::events::PipelineEvent::PipelineStage {
            job_id: clip.job_id.clone(),
            stage_id: "recut".into(),
            status: "active".into(),
            progress: Some(0),
            message: Some(recut_msg.clone()),
        });

    // ── Render every variant in parallel ───────────────────────────
    // `clip.renders` is the source of truth for which aspects already
    // exist for this clip. We don't add variants the original pipeline
    // didn't render — that would change the manifest's variant set.
    use futures_util::{StreamExt, stream};
    let concurrency = cfg.effective_render_concurrency();
    let ffmpeg = cfg.ffmpeg.clone();
    let crf = cfg.clip_video_crf;
    let preset = cfg.clip_video_preset.clone();
    let audio_kbps = cfg.clip_audio_bitrate_kbps;
    // Shared counter so each variant's task can announce its completion
    // back to the dashboard with `done/total` progress.
    let done_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let bus = state.event_bus.clone();
    let job_id_for_events = clip.job_id.clone();
    let results: Vec<(String, PathBuf, Result<(), anyhow::Error>)> =
        stream::iter(clip.renders.iter().cloned())
            .map(|render| {
                let ffmpeg = ffmpeg.clone();
                let source_path = source_path.clone();
                let preset = preset.clone();
                let words = filtered_words.clone();
                let overrides = caption_overrides.clone();
                let bus = bus.clone();
                let job_id_for_events = job_id_for_events.clone();
                let done_counter = done_counter.clone();
                async move {
                    let out_path = PathBuf::from(&render.path);
                    let (profile, aspect) = match render.variant.as_str() {
                        "9x16" | "vertical" | "shorts" => (
                            Some(crate::render::RenderProfile::shorts_vertical()),
                            Some("vertical"),
                        ),
                        "1x1" | "square" => (
                            Some(crate::render::RenderProfile::linkedin_square()),
                            Some("square"),
                        ),
                        "16x9" | "landscape" | "horizontal" => (
                            Some(crate::render::RenderProfile::bluesky_landscape()),
                            Some("landscape"),
                        ),
                        _ => (None, None),
                    };
                    let (Some(profile), Some(aspect)) = (profile, aspect) else {
                        return (
                            render.variant.clone(),
                            out_path,
                            Err(anyhow::anyhow!(
                                "unknown variant {:?} on existing render row",
                                render.variant
                            )),
                        );
                    };
                    let profile = profile.with_quality(crf, &preset, audio_kbps);

                    // Write a fresh ASS file per variant when captions are
                    // wanted AND we have words for this clip. Drops in
                    // alongside the existing render, distinct name so we
                    // don't clobber the original.
                    let ass_path = if burn && !words.is_empty() {
                        let parent = out_path
                            .parent()
                            .map(|p| p.to_path_buf())
                            .unwrap_or_else(|| PathBuf::from("."));
                        let stem = out_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("clip");
                        let p = parent.join(format!("{stem}_recut.ass"));
                        let style = match aspect {
                            "vertical" => {
                                crate::captions::CaptionStyle::for_vertical_with(&overrides)
                            }
                            "square" => {
                                crate::captions::CaptionStyle::for_square_with(&overrides)
                            }
                            _ => crate::captions::CaptionStyle::for_landscape_with(&overrides),
                        };
                        match crate::captions::write_ass(
                            &p,
                            &words,
                            profile.width,
                            profile.height,
                            &style,
                        )
                        .await
                        {
                            Ok(()) => Some(p),
                            Err(e) => {
                                tracing::warn!(
                                    variant = %render.variant,
                                    error = ?e,
                                    "recut: ASS write failed; rendering without captions"
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let subtitle_paths: Vec<&std::path::Path> = ass_path
                        .as_ref()
                        .map(|p| vec![p.as_path()])
                        .unwrap_or_default();

                    let res = crate::render::render_clip(
                        &ffmpeg,
                        &source_path,
                        new_start,
                        new_end,
                        &out_path,
                        &profile,
                        &subtitle_paths,
                    )
                    .await;

                    // Announce this variant's completion so the dashboard
                    // can update the progress bar incrementally instead of
                    // waiting for ALL variants to finish.
                    let done = done_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let progress = ((done as f64 / total_variants as f64) * 100.0).round() as u8;
                    let status_label = if res.is_ok() { "done" } else { "failed" };
                    let msg = if res.is_ok() {
                        format!("recut: {} {} ({}/{})", render.variant, status_label, done, total_variants)
                    } else {
                        format!(
                            "recut: {} FAILED ({}/{}): {}",
                            render.variant,
                            done,
                            total_variants,
                            res.as_ref()
                                .err()
                                .map(|e| format!("{e:#}"))
                                .unwrap_or_default()
                        )
                    };
                    bus.emit(crate::events::PipelineEvent::PipelineStage {
                        job_id: job_id_for_events.clone(),
                        stage_id: "recut".into(),
                        status: "active".into(),
                        progress: Some(progress),
                        message: Some(msg),
                    });

                    (render.variant.clone(), out_path, res)
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

    let mut errors: Vec<String> = Vec::new();
    let mut succeeded: Vec<(String, PathBuf)> = Vec::new();
    for (variant, out_path, res) in results {
        match res {
            Ok(()) => succeeded.push((variant, out_path)),
            Err(e) => errors.push(format!("{variant}: {e:#}")),
        }
    }
    if succeeded.is_empty() {
        return err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("all variants failed to re-render: {}", errors.join(" | ")),
        )
        .into_response();
    }

    // ── Update DB ──────────────────────────────────────────────────
    let new_start_ms = (new_start * 1000.0) as i64;
    let new_end_ms = (new_end * 1000.0) as i64;
    if let Err(e) = state
        .storage
        .update_clip_bounds(&clip.id, new_start_ms, new_end_ms)
        .await
    {
        return err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("DB bounds update failed: {e:#}"),
        )
        .into_response();
    }
    // Replace each successful variant's render row. URL=None clears any
    // stale R2 URL — the /media/clipper/* proxy serves the freshly-rendered
    // local file. Variants that failed keep their existing row (the user
    // sees them as stale until they trim again or re-run the pipeline).
    for (variant, out_path) in &succeeded {
        let bytes = tokio::fs::metadata(out_path)
            .await
            .map(|m| m.len() as i64)
            .ok();
        if let Err(e) = state
            .storage
            .insert_clip_render(
                &clip.id,
                variant,
                &out_path.to_string_lossy(),
                bytes,
                Some((duration * 1000.0) as i64),
                None,
            )
            .await
        {
            tracing::warn!(
                clip_id = %clip.id,
                variant = %variant,
                error = ?e,
                "DB render row update failed; render file is on disk but metadata is stale"
            );
        }
    }

    tracing::info!(
        clip_id = %clip.id,
        variants_succeeded = succeeded.len(),
        variants_failed = errors.len(),
        new_start_secs = new_start,
        new_end_secs = new_end,
        captions_applied = captions_will_apply,
        "clip recut complete"
    );
    if !errors.is_empty() {
        tracing::warn!(
            clip_id = %clip.id,
            errors = %errors.join(" | "),
            "some variants failed to re-render"
        );
    }

    // Final "done" event so the dashboard's progress strip clears + the
    // Activity Log gets a clean closing line.
    let final_msg = match (errors.is_empty(), captions_will_apply) {
        (true, true) => format!(
            "recut complete: {} variants re-rendered with captions",
            succeeded.len()
        ),
        (true, false) if burn => format!(
            "recut complete: {} variants re-rendered (captions skipped — no transcript words in manifest; re-run pipeline to enable caption regen)",
            succeeded.len()
        ),
        (true, false) => format!(
            "recut complete: {} variants re-rendered (captions off)",
            succeeded.len()
        ),
        (false, _) => format!(
            "recut complete: {} succeeded, {} failed",
            succeeded.len(),
            errors.len()
        ),
    };
    state
        .event_bus
        .emit(crate::events::PipelineEvent::PipelineStage {
            job_id: clip.job_id.clone(),
            stage_id: "recut".into(),
            status: if errors.is_empty() { "done".into() } else { "failed".into() },
            progress: Some(100),
            message: Some(final_msg),
        });

    // Return the freshly-loaded clip wrapped with status flags so the
    // dashboard can give honest feedback: "captions applied" vs "skipped
    // because no words" vs "off by choice".
    match state.storage.get_clip(&clip.id).await {
        Ok(Some(c)) => {
            let mut payload = enrich_clip(&c, &state.work_dir);
            if let Value::Object(ref mut map) = payload {
                map.insert(
                    "captions_applied".into(),
                    Value::Bool(captions_will_apply),
                );
                map.insert(
                    "captions_requested".into(),
                    Value::Bool(burn),
                );
                map.insert(
                    "has_captionable_words".into(),
                    Value::Bool(!filtered_words.is_empty()),
                );
                map.insert(
                    "variants_failed".into(),
                    Value::Number(serde_json::Number::from(errors.len())),
                );
            }
            (StatusCode::OK, Json(payload)).into_response()
        }
        Ok(None) => err_json(StatusCode::NOT_FOUND, "clip vanished post-update")
            .into_response(),
        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// Locate the original source video for a job. Prefers `jobs.local_path`
/// (set on file-upload jobs) and falls back to scanning the upload dir
/// for any `.mp4` (URL-ingest jobs land there via yt-dlp). Returns
/// `None` when nothing usable remains on disk.
async fn resolve_source_video(
    job: &crate::storage::JobRow,
    work_dir: &str,
) -> Option<PathBuf> {
    if let Some(p) = job.local_path.as_deref() {
        let path = PathBuf::from(p);
        if tokio::fs::metadata(&path).await.is_ok() {
            return Some(path);
        }
    }
    let dir = PathBuf::from(work_dir).join("uploads").join(&job.id);
    let mut entries = tokio::fs::read_dir(&dir).await.ok()?;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("mp4"))
        {
            return Some(path);
        }
    }
    None
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
        .route("/clips/{id}/recut", post(recut_clip))
}
