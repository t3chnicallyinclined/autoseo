//! Dashboard error type — uniform JSON 4xx/5xx for `/api/*`.
//!
//! HTMX-targeted handlers can attach `HX-Retarget: #toast` via response
//! extensions later; slice 0 just emits a clean JSON body.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

#[derive(Debug)]
pub struct DashboardError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl DashboardError {
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal",
            message: message.into(),
        }
    }

    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: message.into(),
        }
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "forbidden",
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found",
            message: message.into(),
        }
    }
}

impl IntoResponse for DashboardError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": self.code,
            "message": self.message,
        }));
        (self.status, body).into_response()
    }
}

impl From<anyhow::Error> for DashboardError {
    fn from(e: anyhow::Error) -> Self {
        Self::internal(format!("{e:#}"))
    }
}

pub type Result<T> = std::result::Result<T, DashboardError>;
