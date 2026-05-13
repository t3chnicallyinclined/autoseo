//! Shared application state passed to every axum handler.
//!
//! Slice 0: just storage + version. Later slices add Vault, SSE bus, scheduler
//! handle, http client for platform tests, render semaphore.

use std::sync::Arc;

use crate::storage::Storage;

#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<Storage>,
    pub version: &'static str,
    pub schema_version: u32,
    pub scheduler_status: SchedulerStatus,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerStatus {
    /// Slice 0–5: scheduler isn't wired up yet.
    Disabled,
    /// Slice 6+: scheduler task is running.
    Running,
}

impl SchedulerStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SchedulerStatus::Disabled => "disabled",
            SchedulerStatus::Running => "running",
        }
    }
}
