//! HTTP route composition. Each resource module exposes a `router()`
//! returning a `Router<AppState>`. They get nested under the top-level Router
//! in `dashboard::server::build_router`.

pub mod health;
