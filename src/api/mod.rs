pub mod config_store;

use axum::{Json, Router, extract::{Path, State}, response::Html, routing::{get, patch, post}};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};

use crate::storage::Storage;
use config_store::ConfigStore;

/// Shared application state passed to all handlers via Axum's state extractor.
#[derive(Clone)]
pub struct AppState {
    pub storage: Storage,
    pub work_dir: String,
    pub dashboard_dist: Option<PathBuf>,
    pub config_store: Arc<ConfigStore>,
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
            let client_id = store.get_value("GOOGLE_CLIENT_ID").await.unwrap_or_default();
            let client_secret = store.get_value("GOOGLE_CLIENT_SECRET").await.unwrap_or_default();
            let refresh_token = store.get_value("GOOGLE_REFRESH_TOKEN").await.unwrap_or_default();
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
            let password = store.get_value("BLUESKY_APP_PASSWORD").await.unwrap_or_default();
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
            let key = store.get_value("AYRSHARE_API_KEY").await.unwrap_or_default();
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
        anyhow::bail!("HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())
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
        anyhow::bail!("HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())
    }
}

async fn test_google(client_id: &str, client_secret: &str, refresh_token: &str) -> anyhow::Result<String> {
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
        anyhow::bail!("HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())
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
        anyhow::bail!("HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())
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
        anyhow::bail!("HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())
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
            axum::http::Method::DELETE,
            axum::http::Method::PATCH,
        ])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let dashboard_dist = state.dashboard_dist.clone();

    // API routes nested under /api — unknown /api/* paths still return 404
    let api_routes = Router::new()
        .route("/health", get(health))
        .route("/config", get(get_config).patch(patch_config))
        .route("/config/test/{service}", post(test_service));

    let app = Router::new()
        .nest("/api", api_routes)
        .layer(cors)
        .with_state(Arc::new(state));

    // Serve the dashboard frontend from dist/ as a fallback for non-API paths
    match dashboard_dist {
        Some(dist_path) if dist_path.is_dir() => {
            let index_file = dist_path.join("index.html");
            // SPA fallback: serve index.html for any path not matched by a static file
            let serve_dir = ServeDir::new(&dist_path)
                .not_found_service(ServeFile::new(index_file));
            app.fallback_service(serve_dir)
        }
        _ => {
            app.fallback(dist_not_found)
        }
    }
}

/// Shown when the dashboard dist/ folder doesn't exist.
async fn dist_not_found() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
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
</html>"#)
}

/// Start the API server on the given port with graceful shutdown.
pub async fn serve(state: AppState, port: u16, cors_origins: &str, open_browser: bool) -> anyhow::Result<()> {
    let app = router(state, cors_origins);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "API server listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;

    if open_browser {
        let url = format!("http://localhost:{}", port);
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
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
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

    fn test_state() -> AppState {
        let storage = Storage::open_in_memory_sync();
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        let config_store = Arc::new(
            tokio::runtime::Handle::current()
                .block_on(ConfigStore::load(config_path))
                .unwrap(),
        );
        // Leak the tempdir so it lives for the test duration
        std::mem::forget(dir);
        AppState {
            storage,
            work_dir: "/tmp/test_work".to_string(),
            dashboard_dist: None,
            config_store,
        }
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = router(test_state(), "http://localhost:5173");

        let req = Request::builder()
            .uri("/api/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn cors_allows_configured_origin() {
        let app = router(test_state(), "http://localhost:5173");

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
    async fn unknown_route_returns_404() {
        let app = router(test_state(), "http://localhost:5173");

        let req = Request::builder()
            .uri("/api/nonexistent")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_config_returns_needs_setup() {
        let app = router(test_state(), "http://localhost:5173");

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
    async fn patch_config_stores_and_masks_secrets() {
        let state = test_state();
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
        assert!(!key.contains("realkey"), "secret should not be in clear text");

        // Non-secret should be visible
        assert_eq!(json["config"]["MODE"], "clipper");

        // needs_setup should now be false
        assert_eq!(json["needs_setup"], false);
    }

    #[tokio::test]
    async fn test_unknown_service() {
        let app = router(test_state(), "http://localhost:5173");

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
        assert!(json["message"].as_str().unwrap().contains("Unknown service"));
    }
}
