//! `GET /health` — liveness + dashboard self-report.
//!
//! Used by:
//! - Reverse proxies / load balancers to confirm the process is up.
//! - The operator to confirm the binary booted with the expected schema and
//!   scheduler state.

use crate::dashboard::prelude::*;

pub fn router() -> Router<AppState> {
    Router::new().route("/health", get(health))
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "version": state.version,
        "schema": state.schema_version,
        "scheduler": state.scheduler_status.as_str(),
    }))
}
