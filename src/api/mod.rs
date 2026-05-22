mod auth;
mod clips;
mod cloudflare;
pub mod config_store;
mod jobs;
mod services;
mod stubs;
mod ws;

use axum::{
    Json, Router,
    extract::{Path, State},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::ServeDir;

use crate::events::EventBus;
use crate::storage::Storage;
use config_store::ConfigStore;

/// Shared application state passed to all handlers via Axum's state extractor.
#[derive(Clone)]
pub struct AppState {
    pub storage: Storage,
    pub work_dir: String,
    pub dashboard_dist: Option<PathBuf>,
    pub config_store: Arc<ConfigStore>,
    /// Broadcast channel for pipeline events. Cloned into the worker so it
    /// can publish job status transitions; subscribed from the `/ws` route
    /// on every dashboard connection.
    pub event_bus: EventBus,
}

#[derive(serde::Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

// ── System inspection ─────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct SystemResponse {
    specs: crate::system_specs::SystemSpecs,
    /// Effective render concurrency we'll actually use for the next job.
    /// Reflects either `RENDER_CONCURRENCY` env override or the auto value
    /// derived from `specs.logical_cores`.
    render_concurrency: usize,
    /// `true` if `RENDER_CONCURRENCY` is unset/0 and we picked the auto
    /// value; `false` if the user explicitly pinned it.
    render_concurrency_auto: bool,
}

/// `GET /api/system` — surface CPU / RAM / effective concurrency so the
/// dashboard can show what we detected and tell the user whether they're
/// being throttled by an explicit override.
///
/// Reads `RENDER_CONCURRENCY` straight from the process env so dashboard
/// edits land here without a code restart (the env var is reloaded by each
/// worker run via [`crate::config::Config::parse`]).
async fn get_system(State(state): State<Arc<AppState>>) -> Json<SystemResponse> {
    let _ = state;
    let specs = crate::system_specs::SystemSpecs::detect();
    let explicit = std::env::var("RENDER_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let (render_concurrency, render_concurrency_auto) = if explicit == 0 {
        (crate::system_specs::auto_render_concurrency(&specs), true)
    } else {
        (explicit, false)
    };
    Json(SystemResponse {
        specs,
        render_concurrency,
        render_concurrency_auto,
    })
}

// ── Font inventory + installer ───────────────────────────────────────

#[derive(serde::Serialize)]
struct FontsResponse {
    /// Sorted list of installed font family names. Empty when fontconfig
    /// (`fc-list`) is unavailable on this host.
    families: Vec<String>,
}

/// `GET /api/fonts` — enumerate font families known to the system via
/// fontconfig. libass picks fonts by family name, so this is the truth
/// the renderer will see. The dashboard cross-references it against its
/// curated short-form preview catalog so users know whether their pick
/// will actually render or fall back to DejaVu.
async fn get_fonts() -> Json<FontsResponse> {
    let families = list_installed_fonts().await.unwrap_or_else(|e| {
        tracing::warn!(error = ?e, "fc-list unavailable; returning empty font list");
        Vec::new()
    });
    Json(FontsResponse { families })
}

async fn list_installed_fonts() -> anyhow::Result<Vec<String>> {
    let out = tokio::process::Command::new("fc-list")
        .args([":", "family"])
        .output()
        .await?;
    if !out.status.success() {
        anyhow::bail!("fc-list exit {}", out.status);
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut families: Vec<String> = text
        .lines()
        .map(|line| {
            // fc-list emits "Family1,Family2" for fonts exposed under
            // multiple names. Take the first — it's what libass tries
            // first against an `Fontname:` request.
            line.split(',').next().unwrap_or("").trim().to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();
    families.sort();
    families.dedup();
    Ok(families)
}

#[derive(serde::Deserialize)]
struct InstallFontBody {
    /// Google Fonts family name (e.g. "Montserrat", "Bebas Neue"). The
    /// server hits the Google Fonts CSS API to discover the actual TTF
    /// URLs, downloads them into `~/.fonts/autoseo/<family>/`, then runs
    /// `fc-cache` so libass picks the new font up immediately.
    family: String,
    /// Optional weight subset. When omitted, installs weights 400 + 700.
    /// Caller can pass `["400", "700", "900"]` for heavier hooks.
    weights: Option<Vec<String>>,
}

#[derive(serde::Serialize)]
struct InstallFontResponse {
    family: String,
    installed: Vec<String>,
}

/// `POST /api/fonts/install` — fetch a Google Font into `~/.fonts/autoseo/`
/// and refresh the fontconfig cache. Idempotent — re-runs overwrite the
/// existing files. No auth check beyond the existing bearer middleware.
async fn install_font(Json(body): Json<InstallFontBody>) -> impl IntoResponse {
    match install_google_font(&body.family, body.weights.as_deref()).await {
        Ok(installed) => (
            axum::http::StatusCode::OK,
            Json(InstallFontResponse {
                family: body.family,
                installed,
            }),
        )
            .into_response(),
        Err(e) => {
            tracing::warn!(family = %body.family, error = ?e, "font install failed");
            (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("{e:#}") })),
            )
                .into_response()
        }
    }
}

async fn install_google_font(family: &str, _weights: Option<&[String]>) -> anyhow::Result<Vec<String>> {
    // Pull TTFs straight from github.com/google/fonts via the contents
    // API. Why not the CSS API? Google's CSS endpoint maps `font-weight:400`
    // to whatever static TTF instance it chose for the family's default,
    // which for some families (Montserrat, Inter) is mis-labeled as
    // "Thin" or "Light" in the embedded font name table. fontconfig
    // then registers those files under the wrong family name, so libass
    // can't find them when the user picks "Montserrat".
    //
    // The github repo holds canonical TTFs whose `name` table actually
    // matches the family. We try `ofl/`, `apache/`, then `ufl/` —
    // Google Fonts splits licenses across those three folders.
    let slug = family.to_lowercase().replace(' ', "");
    let licenses = ["ofl", "apache", "ufl"];

    let client = reqwest::Client::builder()
        // The GitHub API rejects requests without a UA header.
        .user_agent("autoseo-font-installer")
        .build()?;

    let mut listing: Option<(String, serde_json::Value)> = None;
    for lic in licenses {
        let api_url = format!("https://api.github.com/repos/google/fonts/contents/{lic}/{slug}");
        let resp = client.get(&api_url).send().await?;
        if resp.status().is_success() {
            let json: serde_json::Value = resp.json().await?;
            listing = Some((lic.to_string(), json));
            break;
        }
    }
    let (license, listing) =
        listing.ok_or_else(|| anyhow::anyhow!("family {family:?} not found in google/fonts repo"))?;
    let entries = listing
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("github API returned non-array contents for {family:?}"))?;

    // Pick TTFs at the family root first (variable fonts ship as
    // `Family[wght].ttf` / `Family-Italic[wght].ttf` at the top level).
    // If only static instances exist, dive into the `static/` subdir.
    let mut ttf_files: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|e| {
            e.get("type").and_then(|v| v.as_str()) == Some("file")
                && e.get("name")
                    .and_then(|v| v.as_str())
                    .is_some_and(|n| n.to_ascii_lowercase().ends_with(".ttf"))
        })
        .collect();
    let mut subdir_listing: Vec<serde_json::Value> = Vec::new();
    if ttf_files.is_empty() {
        let static_url = format!(
            "https://api.github.com/repos/google/fonts/contents/{license}/{slug}/static"
        );
        let resp = client.get(&static_url).send().await?;
        if resp.status().is_success() {
            let json: serde_json::Value = resp.json().await?;
            if let Some(arr) = json.as_array() {
                subdir_listing = arr.clone();
            }
        }
        ttf_files = subdir_listing
            .iter()
            .filter(|e| {
                e.get("type").and_then(|v| v.as_str()) == Some("file")
                    && e.get("name")
                        .and_then(|v| v.as_str())
                        .is_some_and(|n| n.to_ascii_lowercase().ends_with(".ttf"))
            })
            .collect();
    }
    anyhow::ensure!(!ttf_files.is_empty(), "no TTF files for {family:?} in google/fonts");

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let dest_dir = std::path::PathBuf::from(home)
        .join(".fonts")
        .join("autoseo")
        .join(family);
    tokio::fs::create_dir_all(&dest_dir).await?;

    let mut installed = Vec::new();
    for entry in ttf_files.iter() {
        let Some(download_url) = entry.get("download_url").and_then(|v| v.as_str()) else {
            continue;
        };
        let bytes = client
            .get(download_url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        let filename = entry
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("font.ttf");
        let dest = dest_dir.join(filename);
        tokio::fs::write(&dest, &bytes).await?;
        installed.push(dest.display().to_string());
        tracing::info!(path = %dest.display(), bytes = bytes.len(), "font installed");
    }

    // Refresh fontconfig so libass sees the new files this run.
    let cache_status = tokio::process::Command::new("fc-cache")
        .args(["-f"])
        .arg(&dest_dir)
        .status()
        .await;
    match cache_status {
        Ok(s) if s.success() => {}
        Ok(s) => tracing::warn!(status = ?s, "fc-cache returned non-zero; fonts may not be picked up until next process start"),
        Err(e) => tracing::warn!(error = ?e, "fc-cache failed to spawn; fonts may not be picked up until next process start"),
    }

    Ok(installed)
}

// ── Config endpoints ──────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct ConfigResponse {
    config: config_store::ConfigData,
    needs_setup: bool,
    config_path: String,
}

async fn get_config(State(state): State<Arc<AppState>>) -> Json<ConfigResponse> {
    let masked = state.config_store.get_masked().await;
    let needs_setup = state.config_store.needs_setup().await;
    Json(ConfigResponse {
        config: masked,
        needs_setup,
        config_path: state.config_store.path().display().to_string(),
    })
}

async fn patch_config(
    State(state): State<Arc<AppState>>,
    Json(updates): Json<HashMap<String, serde_json::Value>>,
) -> Result<Json<ConfigResponse>, (axum::http::StatusCode, String)> {
    state
        .config_store
        .patch(updates)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let masked = state.config_store.get_masked().await;
    let needs_setup = state.config_store.needs_setup().await;
    Ok(Json(ConfigResponse {
        config: masked,
        needs_setup,
        config_path: state.config_store.path().display().to_string(),
    }))
}

#[derive(serde::Serialize)]
struct TestResult {
    service: String,
    ok: bool,
    message: String,
}

/// Shape an error response the same way the per-module handlers do, so
/// dashboard error parsing stays consistent. Local-to-mod.rs since the
/// peer helpers in `clips.rs` / `jobs.rs` aren't pub.
fn err_json(status: axum::http::StatusCode, msg: impl Into<String>) -> impl IntoResponse {
    (status, Json(serde_json::json!({ "error": msg.into() })))
}

async fn test_service(
    State(state): State<Arc<AppState>>,
    Path(service): Path<String>,
) -> Json<TestResult> {
    let result = run_service_test(&state.config_store, &service).await;
    Json(result)
}

/// Body posted by the setup wizard's "Connect Cloudflare" button.
#[derive(serde::Deserialize)]
struct CloudflareProvisionBody {
    /// CF API token with `Account:Read` + `Workers R2 Storage:Edit` scopes.
    token: String,
    /// Bucket name to create. Defaults to `autoseo-clips`.
    #[serde(default)]
    bucket: Option<String>,
}

/// POST /api/cloudflare/provision
///
/// One-shot R2 bootstrap: creates the bucket, enables the managed
/// pub-<hash>.r2.dev domain, best-effort mints S3 access keys, PATCHes
/// the resulting config into ConfigStore so the dashboard sees the new
/// values immediately. Returns the provisioning report so the wizard can
/// surface "X created, Y already existed, mint S3 keys manually here" to
/// the user.
async fn cloudflare_provision(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CloudflareProvisionBody>,
) -> impl IntoResponse {
    let token = body.token.trim().to_string();
    if token.is_empty() {
        return err_json(
            axum::http::StatusCode::BAD_REQUEST,
            "token is required",
        )
        .into_response();
    }
    let bucket = body
        .bucket
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("autoseo-clips")
        .to_string();

    let client = cloudflare::CloudflareClient::new(token);
    let result = match client.provision(&bucket).await {
        Ok(r) => r,
        Err(e) => {
            return err_json(
                axum::http::StatusCode::BAD_REQUEST,
                format!("cloudflare provision failed: {e:#}"),
            )
            .into_response();
        }
    };

    // PATCH the provisioned values into ConfigStore. Skip access keys if we
    // didn't get them — the wizard will collect those manually. Skip the
    // public URL if it wasn't returned (caller can fill it later).
    let mut updates: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    updates.insert(
        "R2_ENDPOINT".to_string(),
        serde_json::Value::String(result.endpoint.clone()),
    );
    updates.insert(
        "R2_BUCKET".to_string(),
        serde_json::Value::String(result.bucket.clone()),
    );
    if let Some(url) = result.public_url.as_deref() {
        updates.insert(
            "R2_PUBLIC_BASE_URL".to_string(),
            serde_json::Value::String(url.to_string()),
        );
    }
    if let Some(id) = result.access_key_id.as_deref() {
        updates.insert(
            "R2_ACCESS_KEY_ID".to_string(),
            serde_json::Value::String(id.to_string()),
        );
    }
    if let Some(secret) = result.secret_access_key.as_deref() {
        updates.insert(
            "R2_SECRET_ACCESS_KEY".to_string(),
            serde_json::Value::String(secret.to_string()),
        );
    }
    if let Err(e) = state.config_store.patch(updates).await {
        return err_json(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("config persist failed: {e:#}"),
        )
        .into_response();
    }

    (axum::http::StatusCode::OK, Json(result)).into_response()
}

async fn run_service_test(store: &ConfigStore, service: &str) -> TestResult {
    match service {
        "openai" => {
            let key = store.get_value("OPENAI_API_KEY").await.unwrap_or_default();
            let base = store
                .get_value("OPENAI_BASE_URL")
                .await
                .unwrap_or_else(|| "https://api.openai.com".to_string());
            if key.is_empty() {
                return TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: "OPENAI_API_KEY not set".to_string(),
                };
            }
            match test_openai(&base, &key).await {
                Ok(msg) => TestResult {
                    service: service.to_string(),
                    ok: true,
                    message: msg,
                },
                Err(e) => TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: e.to_string(),
                },
            }
        }
        "huggingface" => {
            let key = store.get_value("HF_API_KEY").await.unwrap_or_default();
            if key.is_empty() {
                return TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: "HF_API_KEY not set".to_string(),
                };
            }
            match test_huggingface(&key).await {
                Ok(msg) => TestResult {
                    service: service.to_string(),
                    ok: true,
                    message: msg,
                },
                Err(e) => TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: e.to_string(),
                },
            }
        }
        "google" => {
            let client_id = store
                .get_value("GOOGLE_CLIENT_ID")
                .await
                .unwrap_or_default();
            let client_secret = store
                .get_value("GOOGLE_CLIENT_SECRET")
                .await
                .unwrap_or_default();
            let refresh_token = store
                .get_value("GOOGLE_REFRESH_TOKEN")
                .await
                .unwrap_or_default();
            if client_id.is_empty() || client_secret.is_empty() || refresh_token.is_empty() {
                return TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: "Google OAuth credentials incomplete".to_string(),
                };
            }
            match test_google(&client_id, &client_secret, &refresh_token).await {
                Ok(msg) => TestResult {
                    service: service.to_string(),
                    ok: true,
                    message: msg,
                },
                Err(e) => TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: e.to_string(),
                },
            }
        }
        "bluesky" => {
            let handle = store.get_value("BLUESKY_HANDLE").await.unwrap_or_default();
            let password = store
                .get_value("BLUESKY_APP_PASSWORD")
                .await
                .unwrap_or_default();
            let pds = store
                .get_value("BLUESKY_PDS_URL")
                .await
                .unwrap_or_else(|| "https://bsky.social".to_string());
            if handle.is_empty() || password.is_empty() {
                return TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: "BLUESKY_HANDLE and BLUESKY_APP_PASSWORD required".to_string(),
                };
            }
            match test_bluesky(&pds, &handle, &password).await {
                Ok(msg) => TestResult {
                    service: service.to_string(),
                    ok: true,
                    message: msg,
                },
                Err(e) => TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: e.to_string(),
                },
            }
        }
        "ayrshare" => {
            let key = store
                .get_value("AYRSHARE_API_KEY")
                .await
                .unwrap_or_default();
            if key.is_empty() {
                return TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: "AYRSHARE_API_KEY not set".to_string(),
                };
            }
            match test_ayrshare(&key).await {
                Ok(msg) => TestResult {
                    service: service.to_string(),
                    ok: true,
                    message: msg,
                },
                Err(e) => TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: e.to_string(),
                },
            }
        }
        "r2" => {
            let endpoint = store.get_value("R2_ENDPOINT").await.unwrap_or_default();
            let access = store
                .get_value("R2_ACCESS_KEY_ID")
                .await
                .unwrap_or_default();
            let secret = store
                .get_value("R2_SECRET_ACCESS_KEY")
                .await
                .unwrap_or_default();
            let bucket = store.get_value("R2_BUCKET").await.unwrap_or_default();
            let public = store
                .get_value("R2_PUBLIC_BASE_URL")
                .await
                .unwrap_or_default();
            let region = store
                .get_value("R2_REGION")
                .await
                .unwrap_or_else(|| "auto".to_string());
            if endpoint.is_empty()
                || access.is_empty()
                || secret.is_empty()
                || bucket.is_empty()
                || public.is_empty()
            {
                return TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: "R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BASE_URL all required".to_string(),
                };
            }
            match test_r2(&endpoint, &access, &secret, &bucket, &region).await {
                Ok(msg) => TestResult {
                    service: service.to_string(),
                    ok: true,
                    message: msg,
                },
                Err(e) => TestResult {
                    service: service.to_string(),
                    ok: false,
                    message: e.to_string(),
                },
            }
        }
        _ => TestResult {
            service: service.to_string(),
            ok: false,
            message: format!("Unknown service: {service}"),
        },
    }
}

async fn test_openai(base_url: &str, api_key: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    // Same normalization as the clipper's OpenAiClient — accept either
    // `https://api.openai.com` or `https://api.groq.com/openai/v1`.
    let base = crate::openai::normalize_base_url(base_url);
    let resp = client
        .get(format!("{base}/v1/models"))
        .header("Authorization", format!("Bearer {api_key}"))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?;
    if resp.status().is_success() {
        Ok("Connected — models endpoint reachable".to_string())
    } else {
        anyhow::bail!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )
    }
}

async fn test_huggingface(api_key: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://huggingface.co/api/whoami-v2")
        .header("Authorization", format!("Bearer {api_key}"))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?;
    if resp.status().is_success() {
        Ok("Connected — token valid".to_string())
    } else {
        anyhow::bail!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )
    }
}

async fn test_google(
    client_id: &str,
    client_secret: &str,
    refresh_token: &str,
) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://oauth2.googleapis.com/token")
        .form(&[
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("refresh_token", refresh_token),
            ("grant_type", "refresh_token"),
        ])
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?;
    if resp.status().is_success() {
        Ok("Connected — token refresh successful".to_string())
    } else {
        anyhow::bail!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )
    }
}

async fn test_bluesky(pds_url: &str, handle: &str, password: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "identifier": handle,
        "password": password,
    });
    let resp = client
        .post(format!("{pds_url}/xrpc/com.atproto.server.createSession"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?;
    if resp.status().is_success() {
        Ok("Connected — session created".to_string())
    } else {
        anyhow::bail!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )
    }
}

/// Test an R2 (S3-compatible) configuration by listing the bucket. Returns
/// quickly without uploading any data.
async fn test_r2(
    endpoint: &str,
    access_key: &str,
    secret_key: &str,
    bucket: &str,
    region: &str,
) -> anyhow::Result<String> {
    use aws_credential_types::Credentials;
    use aws_sdk_s3::Client;
    use aws_sdk_s3::config::Region;

    let creds = Credentials::new(access_key, secret_key, None, None, "autoseo-r2-test");
    let cfg = aws_sdk_s3::config::Builder::new()
        .endpoint_url(endpoint)
        .region(Region::new(region.to_string()))
        .credentials_provider(creds)
        .force_path_style(true)
        .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
        .build();
    let client = Client::from_conf(cfg);

    // HeadBucket is the cheapest auth + bucket-exists check.
    match client.head_bucket().bucket(bucket).send().await {
        Ok(_) => Ok(format!("Connected — bucket {bucket} reachable")),
        Err(e) => anyhow::bail!("{}", e.into_service_error()),
    }
}

async fn test_ayrshare(api_key: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://app.ayrshare.com/api/user")
        .header("Authorization", format!("Bearer {api_key}"))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await?;
    if resp.status().is_success() {
        Ok("Connected — API key valid".to_string())
    } else {
        anyhow::bail!(
            "HTTP {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )
    }
}

/// Build the application router with CORS and shared state.
pub fn router(state: AppState, cors_origins: &str) -> Router {
    let origins: Vec<_> = cors_origins
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PUT,
            axum::http::Method::PATCH,
            axum::http::Method::DELETE,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ]);

    let dashboard_dist = state.dashboard_dist.clone();
    let work_dir = PathBuf::from(&state.work_dir);

    // Routes that stay open even when DASHBOARD_TOKEN is set. /health is
    // here so external monitoring (and the dashboard bootstrap) can probe
    // the server before a token is in hand.
    let public_api = Router::new()
        .route("/health", get(health))
        .route("/system", get(get_system))
        .route("/fonts", get(get_fonts));

    // Everything else under /api is gated by the optional bearer middleware
    // (no-op when DASHBOARD_TOKEN is unset).
    let protected_api = Router::new()
        // PATCH is the canonical method for partial-update of config keys;
        // PUT is accepted as an alias because the dashboard's existing
        // `useUpdateConfig` mutation hits PUT. Both call the same handler.
        .route(
            "/config",
            get(get_config).patch(patch_config).put(patch_config),
        )
        .route("/config/test/{service}", post(test_service))
        .route("/cloudflare/provision", post(cloudflare_provision))
        .route("/fonts/install", post(install_font))
        .merge(clips::router())
        .merge(jobs::router())
        .merge(services::router())
        .merge(stubs::router())
        .layer(axum::middleware::from_fn(auth::require_token));

    let api_routes = public_api
        .merge(protected_api)
        // Unknown /api/* paths must return JSON 404, not fall through to the
        // SPA index.html fallback below.
        .fallback(api_not_found);

    // Mount the clipper output dir as a static file tree at /media/clipper/*
    // so the dashboard can stream rendered videos + cover JPGs by URL. The
    // tower-http ServeDir layer takes care of Range support and refuses
    // path-traversal (`..`) requests by default.
    let media_serve =
        ServeDir::new(work_dir.join("clipper")).append_index_html_on_directories(false);
    // Wrap the ServeDir in a Router so the auth middleware applies — clip
    // URLs need to be gated too, otherwise anyone who guesses one leaks the
    // rendered video. `fallback_service` is the supported way to mount a
    // Service at the root of a Router; `nest_service("/", …)` is rejected
    // by current axum.
    let media_routes = Router::new()
        .fallback_service(media_serve)
        .layer(axum::middleware::from_fn(auth::require_token));

    // WS sits behind auth via `?token=...` (browsers can't set headers on
    // raw WebSocket handshakes); the dashboard appends the query.
    let ws_routes = ws::router().layer(axum::middleware::from_fn(auth::require_token));

    let app = Router::new()
        .nest("/api", api_routes)
        // WS at /ws (not /api/ws) so dashboard's `ws://host/ws` default works
        // and a single cloudflared HTTP tunnel covers both API and WS.
        .merge(ws_routes)
        .nest("/media/clipper", media_routes)
        .layer(cors)
        .with_state(Arc::new(state));

    // Serve the dashboard frontend from dist/. The dashboard's WebSocket hook
    // looks at `window.__AUTOSEO_WS_URL` first; we patch the served index.html
    // with a tiny inline script that derives the WS URL from the page's own
    // origin, so it works for both `localhost:8080` and a cloudflared tunnel
    // without rebuilding the dashboard.
    match dashboard_dist {
        Some(dist_path) if dist_path.is_dir() => {
            let index_route = {
                let index_file = dist_path.join("index.html");
                get(move || serve_index_with_ws_inject(index_file.clone()))
            };
            // ServeDir handles assets (js/css/img). Unknown paths fall through
            // to the SPA index route (also patched) for client-side routing.
            let fallback_index = dist_path.join("index.html");
            let serve_dir = ServeDir::new(&dist_path).not_found_service(get(move || {
                serve_index_with_ws_inject(fallback_index.clone())
            }));
            app.route("/", index_route).fallback_service(serve_dir)
        }
        _ => app.fallback(dist_not_found),
    }
}

/// Serve the dashboard's index.html with a `window.__AUTOSEO_WS_URL` inject
/// so the WebSocket hook in the dashboard auto-picks the right ws/wss origin
/// (same as the page itself). The dashboard hook checks
/// `window.__AUTOSEO_WS_URL` before falling back to `ws://<host>:9090/ws`, so
/// this lets local and cloudflared deployments both work without rebuilding.
async fn serve_index_with_ws_inject(path: PathBuf) -> impl IntoResponse {
    use axum::http::{StatusCode, header};
    match tokio::fs::read_to_string(&path).await {
        Ok(body) => {
            // Inline script — minimal, defensive (lower-cased protocol match,
            // works whether `</head>` is upper or lowercase).
            const INJECT: &str = "<script>(function(){try{var p=location.protocol==='https:'?'wss://':'ws://';window.__AUTOSEO_WS_URL=p+location.host+'/ws';}catch(e){}})();</script>";
            let patched = if let Some(idx) = body.to_lowercase().find("</head>") {
                let mut s = String::with_capacity(body.len() + INJECT.len());
                s.push_str(&body[..idx]);
                s.push_str(INJECT);
                s.push_str(&body[idx..]);
                s
            } else {
                // Fall back to prepending — uncommon but keep the page usable.
                format!("{INJECT}{body}")
            };
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                patched,
            )
                .into_response()
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "failed to read dashboard index.html");
            (StatusCode::INTERNAL_SERVER_ERROR, "dashboard index.html unavailable").into_response()
        }
    }
}

/// JSON 404 for unmatched routes under `/api`.
async fn api_not_found() -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        axum::http::StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": "not found" })),
    )
}

/// Shown when the dashboard dist/ folder doesn't exist.
async fn dist_not_found() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html>
<head><title>autoseo - Dashboard Not Built</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 0 20px;">
  <h1>Dashboard not found</h1>
  <p>The frontend has not been built yet. To build it:</p>
  <pre style="background: #f4f4f4; padding: 16px; border-radius: 4px;">
cd dashboard
npm install
npm run build</pre>
  <p>Then restart the server. The built files will be served automatically.</p>
  <p><strong>API is running:</strong> <a href="/api/health">/api/health</a></p>
</body>
</html>"#,
    )
}

/// Start the API server on the given port with graceful shutdown.
pub async fn serve(
    state: AppState,
    port: u16,
    cors_origins: &str,
    open_browser: bool,
) -> anyhow::Result<()> {
    let app = router(state, cors_origins);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "API server listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;

    if open_browser {
        let url = format!("http://localhost:{port}");
        tracing::info!(url = %url, "opening browser");
        if let Err(e) = open::that(&url) {
            tracing::warn!(error = %e, "failed to open browser");
        }
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("API server shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {},
            _ = sigterm.recv() => {},
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }

    tracing::info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt as _;

    async fn test_state() -> AppState {
        let storage = Storage::open_in_memory_sync();
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let config_store = Arc::new(ConfigStore::load(config_path).await.unwrap());
        // Leak the tempdir so it lives for the test duration
        std::mem::forget(dir);
        AppState {
            storage,
            work_dir: "/tmp/test_work".to_string(),
            dashboard_dist: None,
            config_store,
            event_bus: EventBus::new(),
        }
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = router(test_state().await, "http://localhost:5173");

        let req = Request::builder()
            .uri("/api/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn cors_allows_configured_origin() {
        let app = router(test_state().await, "http://localhost:5173");

        let req = Request::builder()
            .method("OPTIONS")
            .uri("/api/health")
            .header("Origin", "http://localhost:5173")
            .header("Access-Control-Request-Method", "GET")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let acl = resp
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok());
        assert_eq!(acl, Some("http://localhost:5173"));
    }

    #[tokio::test]
    async fn list_jobs_returns_empty_when_db_empty() {
        let app = router(test_state().await, "http://localhost:5173");
        let req = Request::builder()
            .uri("/api/jobs")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn list_jobs_returns_dashboard_shape() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-1", Some("the-show"), Some("ep.mp4"), None)
            .await
            .unwrap();

        let app = router(state, "http://localhost:5173");
        let req = Request::builder()
            .uri("/api/jobs")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let j = &arr[0];
        assert_eq!(j["id"], "job-1");
        assert_eq!(j["showId"], "the-show");
        assert_eq!(j["media"], "ep.mp4");
        assert_eq!(j["status"], "pending");
        assert_eq!(j["stage"], "Queued");
        assert!(j["progress"].is_number());
        assert!(j["created"].is_string());
    }

    #[tokio::test]
    async fn get_job_returns_404_when_missing() {
        let app = router(test_state().await, "http://localhost:5173");
        let req = Request::builder()
            .uri("/api/jobs/no-such")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_job_returns_clip_summary() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-2", Some("show"), Some("ep.mp4"), None)
            .await
            .unwrap();
        // Insert a clip so the summary is exercised.
        state
            .storage
            .insert_clip(
                "clip-1",
                "job-2",
                1000,
                61000,
                Some(1),
                Some(82.0),
                Some("hook"),
                None,
                None,
            )
            .await
            .unwrap();

        let app = router(state, "http://localhost:5173");
        let req = Request::builder()
            .uri("/api/jobs/job-2")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], "job-2");
        assert_eq!(json["clipsGenerated"], 1);
        let clips = json["clips"].as_array().unwrap();
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0]["id"], "clip-1");
        assert_eq!(clips[0]["hook"], "hook");
    }

    #[tokio::test]
    async fn retry_job_404_when_missing() {
        let app = router(test_state().await, "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/no-such/retry")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn retry_job_409_when_not_failed() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-3", None, Some("ep.mp4"), None)
            .await
            .unwrap();

        let app = router(state, "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/job-3/retry")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn cancel_pending_job_via_api() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-5", None, Some("ep.mp4"), None)
            .await
            .unwrap();
        let app = router(state.clone(), "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/job-5/cancel")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let row = state.storage.get_job("job-5").await.unwrap().unwrap();
        assert_eq!(row.status.as_str(), "cancelled");
    }

    #[tokio::test]
    async fn cancel_rejects_non_pending_job() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-6", None, Some("ep.mp4"), None)
            .await
            .unwrap();
        state
            .storage
            .update_job_status("job-6", crate::storage::JobStatus::Done, None)
            .await
            .unwrap();
        let app = router(state, "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/job-6/cancel")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn delete_job_removes_row_and_cascades_clips() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-7", None, Some("ep.mp4"), None)
            .await
            .unwrap();
        state
            .storage
            .insert_clip("clip-z", "job-7", 0, 30_000, Some(1), Some(80.0), Some("h"), None, None)
            .await
            .unwrap();
        let app = router(state.clone(), "http://localhost:5173");
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/jobs/job-7")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Job and its clip should both be gone.
        assert!(state.storage.get_job("job-7").await.unwrap().is_none());
        assert!(state.storage.get_clip("clip-z").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_job_returns_404_when_missing() {
        let app = router(test_state().await, "http://localhost:5173");
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/jobs/nope")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rerun_clones_source_into_new_pending_job() {
        let state = test_state().await;
        state
            .storage
            .enqueue_job(
                "job-8",
                Some("the-show"),
                Some("ep.mp4"),
                Some("/tmp/source.mp4"),
                None,
                None,
            )
            .await
            .unwrap();
        state
            .storage
            .update_job_status("job-8", crate::storage::JobStatus::Done, None)
            .await
            .unwrap();

        let app = router(state.clone(), "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/job-8/rerun")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let new_id = json["id"].as_str().unwrap().to_string();
        assert_ne!(new_id, "job-8");
        // Original unchanged
        let orig = state.storage.get_job("job-8").await.unwrap().unwrap();
        assert_eq!(orig.status.as_str(), "done");
        // Clone is pending + carries source fields
        let clone = state.storage.get_job(&new_id).await.unwrap().unwrap();
        assert_eq!(clone.status.as_str(), "pending");
        assert_eq!(clone.show_slug.as_deref(), Some("the-show"));
        assert_eq!(clone.media_name.as_deref(), Some("ep.mp4"));
        assert_eq!(clone.local_path.as_deref(), Some("/tmp/source.mp4"));
    }

    #[tokio::test]
    async fn rerun_404_when_source_missing() {
        let app = router(test_state().await, "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/nope/rerun")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn retry_job_flips_failed_back_to_pending() {
        let state = test_state().await;
        state
            .storage
            .create_job("job-4", None, Some("ep.mp4"), None)
            .await
            .unwrap();
        state
            .storage
            .update_job_status(
                "job-4",
                crate::storage::JobStatus::Failed,
                Some("ffmpeg died"),
            )
            .await
            .unwrap();

        let app = router(state.clone(), "http://localhost:5173");
        let req = Request::builder()
            .method("POST")
            .uri("/api/jobs/job-4/retry")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "pending");

        // Verify the DB matches.
        let row = state.storage.get_job("job-4").await.unwrap().unwrap();
        assert_eq!(row.status.as_str(), "pending");
        assert!(row.error.is_none());
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let app = router(test_state().await, "http://localhost:5173");

        let req = Request::builder()
            .uri("/api/nonexistent")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_config_returns_needs_setup() {
        let app = router(test_state().await, "http://localhost:5173");

        let req = Request::builder()
            .uri("/api/config")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["needs_setup"], true);
    }

    #[tokio::test]
    async fn put_config_works_as_patch_alias() {
        // The dashboard's useUpdateConfig sends PUT; we accept both methods
        // so old built dist works without rebuild.
        let state = test_state().await;
        let app = router(state, "http://localhost:5173");
        let body = serde_json::json!({"CLIP_TOP_K": "5"});
        let req = Request::builder()
            .method("PUT")
            .uri("/api/config")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "PUT should be accepted as a PATCH alias"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["config"]["CLIP_TOP_K"], "5");
    }

    #[tokio::test]
    async fn patch_config_stores_and_masks_secrets() {
        let state = test_state().await;
        let app = router(state.clone(), "http://localhost:5173");

        // PATCH config to set a key
        let patch_body = serde_json::json!({
            "OPENAI_API_KEY": "sk-realkey1234567890abcdef",
            "MODE": "clipper"
        });
        let req = Request::builder()
            .method("PATCH")
            .uri("/api/config")
            .header("Content-Type", "application/json")
            .body(Body::from(serde_json::to_vec(&patch_body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Secret should be masked
        let key = json["config"]["OPENAI_API_KEY"].as_str().unwrap();
        assert!(key.contains("••••"), "secret should be masked: {key}");
        assert!(
            !key.contains("realkey"),
            "secret should not be in clear text"
        );

        // Non-secret should be visible
        assert_eq!(json["config"]["MODE"], "clipper");

        // needs_setup should now be false
        assert_eq!(json["needs_setup"], false);
    }

    #[tokio::test]
    async fn test_unknown_service() {
        let app = router(test_state().await, "http://localhost:5173");

        let req = Request::builder()
            .method("POST")
            .uri("/api/config/test/foobar")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], false);
        assert!(
            json["message"]
                .as_str()
                .unwrap()
                .contains("Unknown service")
        );
    }
}
