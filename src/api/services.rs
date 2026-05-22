//! `/api/services/*` — status surfaces for optional sidecars.
//!
//! Currently exposes the android-agent browser-posting worker. Mirrors
//! the master-switch product model: when `BROWSER_POSTING_ENABLED=false`,
//! status reports `enabled=false` and skips the health probe. When true, it
//! forwards `/healthz` to the worker and reports reachability so the
//! dashboard can show "enabled but unreachable" instead of failing silently.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::State,
    response::IntoResponse,
    routing::get,
};
use serde::Serialize;

use super::AppState;
use crate::platforms::browser::client as browser_client;

const DEFAULT_WORKER_URL: &str = "http://localhost:8090";
const HEALTH_PROBE_TIMEOUT_SECS: u64 = 3;

#[derive(Serialize)]
pub struct BrowserServiceStatus {
    /// True if `BROWSER_POSTING_ENABLED` is set. When false, autoseo never
    /// constructs `Platform::Browser` and never probes the worker.
    pub enabled: bool,
    /// True if the worker responded to `/healthz` within the probe timeout.
    /// `null` when `enabled == false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reachable: Option<bool>,
    /// Worker URL that was probed.
    pub worker_url: String,
    /// Platform IDs the worker has registered drivers for (e.g. `["x"]`).
    /// Empty when unreachable.
    pub drivers: Vec<String>,
    /// Last error encountered while probing the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

async fn browser_status(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    let enabled = std::env::var("BROWSER_POSTING_ENABLED")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let worker_url = std::env::var("BROWSER_WORKER_URL").unwrap_or_else(|_| DEFAULT_WORKER_URL.into());

    if !enabled {
        return Json(BrowserServiceStatus {
            enabled: false,
            reachable: None,
            worker_url,
            drivers: vec![],
            error: None,
        });
    }

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(HEALTH_PROBE_TIMEOUT_SECS))
        .build()
        .expect("reqwest client");

    match browser_client::healthz(&http, &worker_url).await {
        Ok(resp) => Json(BrowserServiceStatus {
            enabled: true,
            reachable: Some(resp.ok && resp.cdp_connected),
            worker_url,
            drivers: resp.drivers,
            error: None,
        }),
        Err(e) => Json(BrowserServiceStatus {
            enabled: true,
            reachable: Some(false),
            worker_url,
            drivers: vec![],
            error: Some(format!("{e:#}")),
        }),
    }
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/services/browser", get(browser_status))
}
