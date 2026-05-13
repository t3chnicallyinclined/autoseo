//! Admin dashboard — `autoseo dashboard` subcommand.
//!
//! Single binary. axum 0.8 HTTP server + Tower middleware. Reads/writes the
//! same SQLite database as the worker. Slice 0 ships only `GET /health`;
//! later slices add auth, jobs, clips, posts, schedules, ingest, etc.

pub use config::DashboardArgs;
pub use server::run;

pub mod config;
pub mod error;
pub mod middleware;
pub mod prelude;
pub mod routes;
pub mod server;
pub mod state;
