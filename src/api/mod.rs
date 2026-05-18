use axum::{Json, Router, extract::State, response::Html, routing::get};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};

use crate::storage::Storage;

/// Shared application state passed to all handlers via Axum's state extractor.
#[derive(Clone)]
pub struct AppState {
    pub storage: Storage,
    pub work_dir: String,
    pub dashboard_dist: Option<PathBuf>,
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
        ])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    let dashboard_dist = state.dashboard_dist.clone();

    // API routes nested under /api — unknown /api/* paths still return 404
    let api_routes = Router::new()
        .route("/health", get(health));

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
        AppState {
            storage,
            work_dir: "/tmp/test_work".to_string(),
            dashboard_dist: None,
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
}
