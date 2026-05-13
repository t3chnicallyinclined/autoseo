//! Async typed DAOs wrapping `spawn_blocking` over the SQLite `Storage`.
//!
//! Each submodule owns one logical resource (workspaces, jobs, clips, …) and
//! exposes async functions that take an `Arc<Storage>` (so handlers can clone
//! once and pass to multiple awaits).
//!
//! Multi-tenant scaffolding from day one: every user-owned table carries a
//! `workspace_id` and every repo method takes it as the first arg after
//! storage. In v1 we always pass `WS_DEFAULT`; v2 (SaaS) extracts it from the
//! session payload.
//!
//! The trait definitions in [`traits`] document the swap-out shape for v2
//! (Postgres, NATS, R2, Vault) — single SQLite/tokio/FS/AES impls in v1.

pub mod audit;
pub mod traits;
pub mod workspaces;

/// The single workspace id used in v1. v2 (SaaS) will replace this with a
/// per-tenant id pulled from the session payload.
pub const WS_DEFAULT: &str = "ws_default";
