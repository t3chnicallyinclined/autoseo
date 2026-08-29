//! Editable prompt registry — exposes the `prompts/` directory as a
//! REST-ish surface so the dashboard can read + edit each LLM prompt without
//! shelling into the box.
//!
//! Endpoints (mounted at `/api/prompts`):
//!   - `GET  /api/prompts`            → list every editable prompt + content
//!   - `GET  /api/prompts/{slug}`     → single prompt (content + metadata)
//!   - `PUT  /api/prompts/{slug}`     → save new content (.bak written first)
//!   - `POST /api/prompts/{slug}/revert` → restore the last .bak
//!
//! Safety model: slugs are baked in. User input never builds a path. The
//! handler resolves slug → registry entry → fixed relative path. No traversal
//! risk even if the body is malformed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use super::AppState;

/// Static description of one editable prompt. The path is relative to the
/// process CWD (which is always the project root for autoseo).
struct PromptDef {
    slug: &'static str,
    label: &'static str,
    path: &'static str,
    group: &'static str,
    description: &'static str,
    /// Template placeholders the runtime will substitute. Surfaced in the
    /// dashboard so the user knows what's available without grepping code.
    variables: &'static [&'static str],
}

/// The full set of prompts the dashboard is allowed to edit. Adding a new
/// prompt is: drop a .txt file + add an entry here + (optional) reference
/// from a `_prompt_path` config knob.
const PROMPTS: &[PromptDef] = &[
    // ── Clip ranker ─────────────────────────────────────────────────
    PromptDef {
        slug: "ranker_system",
        label: "Clip Ranker — System",
        path: "prompts/clips/ranker_system.txt",
        group: "Clip Ranker",
        description: "Defines short-form ranking strategy: 'one beat, one hook', scoring heuristics, JSON output shape.",
        variables: &[],
    },
    PromptDef {
        slug: "ranker_user",
        label: "Clip Ranker — User",
        path: "prompts/clips/ranker_user.txt",
        group: "Clip Ranker",
        description: "Per-batch ranker request. Receives candidates_json + show context + duration constraints + trends.",
        variables: &[
            "{{show_name}}", "{{hosts}}", "{{guest}}",
            "{{performance_history}}", "{{current_trends}}",
            "{{candidates_json}}",
            "{{min_secs}}", "{{max_secs}}", "{{target_secs}}",
        ],
    },
    // ── Social copy ─────────────────────────────────────────────────
    PromptDef {
        slug: "social_system",
        label: "Social Copy — System",
        path: "prompts/clips/social_system.txt",
        group: "Social Copy",
        description: "Per-platform caption + hashtag policy. Length budgets per platform. Output JSON shape.",
        variables: &[],
    },
    PromptDef {
        slug: "social_user",
        label: "Social Copy — User",
        path: "prompts/clips/social_user.txt",
        group: "Social Copy",
        description: "Per-clip social-copy request. Receives full clip transcript + ranker's hook/reasoning + show context.",
        variables: &[
            "{{show_name}}", "{{hosts}}", "{{guest}}",
            "{{time_range}}", "{{duration_secs}}",
            "{{hook}}", "{{reasoning}}",
            "{{transcript}}",
        ],
    },
    // ── SEO mode (non-clipper path) ─────────────────────────────────
    PromptDef {
        slug: "seo_system",
        label: "SEO Mode — System",
        path: "prompts/seo_system.txt",
        group: "SEO Mode",
        description: "System prompt for the non-clipper SEO path (legacy email-digest mode).",
        variables: &[],
    },
    PromptDef {
        slug: "seo_user",
        label: "SEO Mode — User",
        path: "prompts/seo_user.txt",
        group: "SEO Mode",
        description: "SEO mode user prompt. Receives full episode transcript.",
        variables: &["{{transcript}}"],
    },
    PromptDef {
        slug: "seo_variants",
        label: "SEO Mode — Variants",
        path: "prompts/seo_variants.txt",
        group: "SEO Mode",
        description: "Variant specifications — generates N distinct title/description/hashtag variants per episode.",
        variables: &[],
    },
    // ── Thumbnail picker ────────────────────────────────────────────
    PromptDef {
        slug: "thumbnail_system",
        label: "Thumbnail Picker — System",
        path: "prompts/thumbnail_system.txt",
        group: "Thumbnail",
        description: "Picks thumbnail-worthy timestamps from an episode transcript.",
        variables: &[],
    },
    PromptDef {
        slug: "thumbnail_user",
        label: "Thumbnail Picker — User",
        path: "prompts/thumbnail_user.txt",
        group: "Thumbnail",
        description: "Asks for N candidate timestamps across the episode duration.",
        variables: &["{{count}}", "{{minutes}}"],
    },
];

// ── Wire format ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PromptEntry {
    slug: String,
    label: String,
    path: String,
    group: String,
    description: String,
    variables: Vec<String>,
    content: String,
    /// True when a `.bak` exists next to the file (i.e. a prior save can be reverted).
    has_backup: bool,
    /// Detected differences from the on-disk default. `false` when content
    /// matches what was originally checked in to git. We don't actually
    /// diff against git — we set `true` whenever a `.bak` exists, which is
    /// only created on the first user edit.
    is_modified: bool,
}

#[derive(Deserialize)]
struct SaveBody {
    content: String,
}

// ── Router ──────────────────────────────────────────────────────────────

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/prompts", get(list_prompts))
        .route("/prompts/{slug}", get(get_prompt).put(save_prompt))
        .route("/prompts/{slug}/revert", post(revert_prompt))
}

// ── Handlers ────────────────────────────────────────────────────────────

async fn list_prompts(State(_): State<Arc<AppState>>) -> Json<Vec<PromptEntry>> {
    let mut out = Vec::with_capacity(PROMPTS.len());
    for def in PROMPTS {
        out.push(build_entry(def).await);
    }
    Json(out)
}

async fn get_prompt(
    State(_): State<Arc<AppState>>,
    AxumPath(slug): AxumPath<String>,
) -> impl IntoResponse {
    let Some(def) = lookup(&slug) else {
        return not_found();
    };
    let entry = build_entry(def).await;
    (StatusCode::OK, Json(entry)).into_response()
}

async fn save_prompt(
    State(_): State<Arc<AppState>>,
    AxumPath(slug): AxumPath<String>,
    Json(body): Json<SaveBody>,
) -> impl IntoResponse {
    let Some(def) = lookup(&slug) else {
        return not_found();
    };
    let path = PathBuf::from(def.path);
    match write_with_backup(&path, &body.content).await {
        Ok(()) => {
            tracing::info!(slug = def.slug, bytes = body.content.len(), "prompt saved");
            let entry = build_entry(def).await;
            (StatusCode::OK, Json(entry)).into_response()
        }
        Err(e) => {
            tracing::warn!(slug = def.slug, error = ?e, "prompt save failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("{e:#}") })),
            )
                .into_response()
        }
    }
}

async fn revert_prompt(
    State(_): State<Arc<AppState>>,
    AxumPath(slug): AxumPath<String>,
) -> impl IntoResponse {
    let Some(def) = lookup(&slug) else {
        return not_found();
    };
    let path = PathBuf::from(def.path);
    let bak = path.with_extension("txt.bak");
    if !bak.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "no backup to revert to" })),
        )
            .into_response();
    }
    match tokio::fs::rename(&bak, &path).await {
        Ok(()) => {
            tracing::info!(slug = def.slug, "prompt reverted from backup");
            let entry = build_entry(def).await;
            (StatusCode::OK, Json(entry)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("{e}") })),
        )
            .into_response(),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn lookup(slug: &str) -> Option<&'static PromptDef> {
    PROMPTS.iter().find(|p| p.slug == slug)
}

fn not_found() -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": "unknown prompt slug" })),
    )
        .into_response()
}

async fn build_entry(def: &PromptDef) -> PromptEntry {
    let path = PathBuf::from(def.path);
    let content = tokio::fs::read_to_string(&path)
        .await
        .unwrap_or_else(|e| format!("/* failed to read {}: {} */", def.path, e));
    let has_backup = path.with_extension("txt.bak").exists();
    PromptEntry {
        slug: def.slug.to_string(),
        label: def.label.to_string(),
        path: def.path.to_string(),
        group: def.group.to_string(),
        description: def.description.to_string(),
        variables: def.variables.iter().map(|s| s.to_string()).collect(),
        content,
        has_backup,
        is_modified: has_backup,
    }
}

/// Write `content` to `path` atomically with a `.bak` copy of the previous
/// version. The `.bak` enables the revert endpoint to restore the last
/// known-good prompt.
async fn write_with_backup(path: &Path, content: &str) -> anyhow::Result<()> {
    // Snapshot the current file into a `.bak` so revert can restore it.
    // Skip when the file doesn't exist yet (fresh install).
    if path.exists() {
        let bak = path.with_extension("txt.bak");
        tokio::fs::copy(path, &bak).await?;
    }
    // Atomic rewrite: write to a sibling temp file, then rename.
    let tmp = path.with_extension("txt.tmp");
    tokio::fs::write(&tmp, content).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}
