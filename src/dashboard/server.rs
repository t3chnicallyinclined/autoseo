//! axum server bootstrap. Constructs `AppState`, builds the router, and
//! serves on the configured bind address until SIGINT/SIGTERM.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Router;
use tokio::signal;
use tower_http::{
    compression::CompressionLayer, timeout::TimeoutLayer, trace::TraceLayer,
};

use crate::config::Config;
use crate::storage::Storage;

use super::config::DashboardArgs;
use super::routes;
use super::state::{AppState, SchedulerStatus};

/// Boot the dashboard server. Returns when the listener exits (Ctrl-C, SIGTERM).
pub async fn run(cfg: Config, args: DashboardArgs) -> anyhow::Result<()> {
    let storage = Arc::new(
        Storage::open(&cfg.clipper_db)
            .await
            .with_context(|| format!("open dashboard storage at {}", cfg.clipper_db))?,
    );

    let state = AppState {
        storage,
        version: env!("CARGO_PKG_VERSION"),
        schema_version: 1, // bumped to 2 by Slice 1's migration
        scheduler_status: SchedulerStatus::Disabled, // → Running after Slice 6 wiring
    };

    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("bind dashboard listener to {}", &args.bind))?;
    tracing::info!(
        bind = %args.bind,
        insecure = args.insecure,
        version = env!("CARGO_PKG_VERSION"),
        "dashboard listening"
    );
    println!("autoseo dashboard listening on http://{}", &args.bind);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum::serve")?;

    tracing::info!("dashboard shutdown complete");
    Ok(())
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .merge(routes::health::router())
        // Slice 2 inserts: .merge(routes::pages::router()).merge(routes::users::router())
        // Slice 3+: .merge(routes::jobs::router())...
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(TimeoutLayer::new(Duration::from_secs(60)))
        .with_state(state)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl-C received; shutting down"),
        _ = terminate => tracing::info!("SIGTERM received; shutting down"),
    }
}
