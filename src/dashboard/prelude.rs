//! Common imports for dashboard route handlers.

#![allow(unused_imports)]

pub use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Redirect, Response},
    routing::{delete, get, patch, post, put},
};
pub use serde::{Deserialize, Serialize};
pub use serde_json::{Value, json};

pub use super::error::{DashboardError, Result};
pub use super::state::AppState;
