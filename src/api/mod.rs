use axum::{Json, Router, extract::State, routing::get};
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::storage::Storage;

/// Shared application state passed to all handlers via Axum's state extractor.
#[derive(Clone)]
pub struct AppState {
    pub storage: Storage,
    pub work_dir: String,
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

    Router::new()
        .route("/api/health", get(health))
        .layer(cors)
        .with_state(Arc::new(state))
}

/// Start the API server on the given port with graceful shutdown.
pub async fn serve(state: AppState, port: u16, cors_origins: &str) -> anyhow::Result<()> {
    let app = router(state, cors_origins);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!(%addr, "API server listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
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
