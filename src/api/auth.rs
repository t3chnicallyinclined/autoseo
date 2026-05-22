//! Optional bearer-token middleware.
//!
//! When `DASHBOARD_TOKEN` is set (non-empty) at process start, every
//! protected route must carry the same token, supplied via either:
//!   - `Authorization: Bearer <token>` header (preferred, used by fetch)
//!   - `?token=<token>` query string (used by WebSocket; raw `ws://` has
//!     no clean header story)
//!   - `autoseo_token=<token>` cookie (set by the dashboard so plain
//!     browser navigation works after the first prompt)
//!
//! When `DASHBOARD_TOKEN` is unset or empty, the middleware is a no-op —
//! local development keeps working without any setup. The intended use is
//! to lock down a cloudflared tunnel before sharing its URL.
//!
//! `/api/health` and the SPA static routes intentionally don't carry this
//! layer so the dashboard can boot, hit `/api/config`, see a 401, and
//! prompt the user for the token.

use axum::{
    extract::Request,
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Read the token the server expects. `None` = no auth required.
fn expected_token() -> Option<String> {
    std::env::var("DASHBOARD_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
}

/// Extract `Bearer <token>` from the Authorization header, case-insensitive.
fn header_token(req: &Request) -> Option<String> {
    let raw = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let stripped = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))?;
    Some(stripped.to_string())
}

/// Extract `?token=...` from the URI query.
fn query_token(req: &Request) -> Option<String> {
    let q = req.uri().query()?;
    for kv in q.split('&') {
        let mut split = kv.splitn(2, '=');
        let k = split.next()?;
        let v = split.next()?;
        if k == "token" {
            // Don't URL-decode here; the dashboard sends the raw token.
            return Some(v.to_string());
        }
    }
    None
}

/// Extract `autoseo_token=...` from the Cookie header.
fn cookie_token(req: &Request) -> Option<String> {
    let raw = req.headers().get(header::COOKIE)?.to_str().ok()?;
    for kv in raw.split(';') {
        let kv = kv.trim();
        if let Some(v) = kv.strip_prefix("autoseo_token=") {
            return Some(v.to_string());
        }
    }
    None
}

/// Axum middleware that gates downstream handlers behind the configured
/// token. See module docs for the supplying mechanisms.
pub async fn require_token(req: Request, next: Next) -> Response {
    let Some(expected) = expected_token() else {
        return next.run(req).await;
    };

    let supplied = header_token(&req)
        .or_else(|| query_token(&req))
        .or_else(|| cookie_token(&req));

    if supplied.as_deref() == Some(expected.as_str()) {
        return next.run(req).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": "auth required",
            "hint": "Send `Authorization: Bearer <DASHBOARD_TOKEN>` or set the autoseo_token cookie.",
        })),
    )
        .into_response()
}
