mod clips;
pub mod config_store;
mod jobs;
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

async fn test_service(
    State(state): State<Arc<AppState>>,
    Path(service): Path<String>,
) -> Json<TestResult> {
    let result = run_service_test(&state.config_store, &service).await;
    Json(result)
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
    let resp = client
        .get(format!("{base_url}/v1/models"))
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
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let dashboard_dist = state.dashboard_dist.clone();
    let work_dir = PathBuf::from(&state.work_dir);

    // API routes nested under /api — unknown /api/* paths still return 404
    let api_routes = Router::new()
        .route("/health", get(health))
        // PATCH is the canonical method for partial-update of config keys;
        // PUT is accepted as an alias because the dashboard's existing
        // `useUpdateConfig` mutation hits PUT. Both call the same handler.
        .route(
            "/config",
            get(get_config).patch(patch_config).put(patch_config),
        )
        .route("/config/test/{service}", post(test_service))
        .merge(clips::router())
        .merge(jobs::router())
        .merge(stubs::router())
        // Unknown /api/* paths must return JSON 404, not fall through to the
        // SPA index.html fallback below.
        .fallback(api_not_found);

    // Mount the clipper output dir as a static file tree at /media/clipper/*
    // so the dashboard can stream rendered videos + cover JPGs by URL. The
    // tower-http ServeDir layer takes care of Range support and refuses
    // path-traversal (`..`) requests by default.
    let media_serve =
        ServeDir::new(work_dir.join("clipper")).append_index_html_on_directories(false);

    let app = Router::new()
        .nest("/api", api_routes)
        // WS at /ws (not /api/ws) so dashboard's `ws://host/ws` default works
        // and a single cloudflared HTTP tunnel covers both API and WS.
        .merge(ws::router())
        .nest_service("/media/clipper", media_serve)
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
