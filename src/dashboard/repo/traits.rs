//! Repository traits — the SaaS-readiness seam.
//!
//! v1 ships exactly one implementation per trait (SQLite, in-process tokio,
//! local FS, ChaCha20-Poly1305). The day we add Postgres / NATS JetStream /
//! R2 / Vault, every call site stays unchanged — only the trait impl wired
//! into `AppState` changes.
//!
//! Slice 1 declares the shapes; the concrete impls are filled in over the
//! following slices (clips/posts/schedules: slices 3–8, creds: slice 7,
//! events: slice 6). The method surfaces below intentionally only carry the
//! few calls needed for v1's `/health` wiring + slice-1 unit tests; later
//! slices append more methods to each trait as they land.
//!
//! These traits use `impl Future` return positions (Rust 2024) rather than
//! pulling in `async_trait`, since the only consumers are inside the
//! dashboard module and we don't need dyn-dispatch yet. When v2 introduces
//! a second backend, the easiest path is `boxed-future`-style returns or a
//! switch to `async_trait::async_trait`.

#![allow(dead_code)] // traits are forward-declared; impls land slice-by-slice.

use std::future::Future;

/// Errors returned by repository operations. v1 collapses to `anyhow::Error`
/// since handlers convert via `DashboardError::Internal`.
pub type RepoResult<T> = anyhow::Result<T>;

/// Persistence + query surface for clips, edits, history. Slice 3+ fills in
/// the SQLite impl; v2 adds a Postgres impl behind the same trait.
pub trait ClipRepo: Send + Sync {
    fn ping(&self) -> impl Future<Output = RepoResult<()>> + Send;
}

/// Schedule + post queue. v1 polls SQLite every 30s with a `Notify` wake.
/// v2 backs this with NATS JetStream — same `claim_due` / `mark_done` API.
pub trait ScheduleQueue: Send + Sync {
    fn ping(&self) -> impl Future<Output = RepoResult<()>> + Send;
}

/// Blob storage abstraction for clips + source MP4s + waveform sidecars.
/// v1 reads/writes the local filesystem under `WORK_DIR`. v2 swaps in R2 /
/// S3 / B2; the `/media/:job/:file` 302→signed-URL handler doesn't change.
pub trait ClipBlobStore: Send + Sync {
    fn ping(&self) -> impl Future<Output = RepoResult<()>> + Send;
}

/// Encrypted credential vault. v1 uses XChaCha20-Poly1305 with a 32-byte
/// master key from env. v2 (SaaS) swaps this for HashiCorp Vault transit
/// engine with per-tenant keys.
pub trait SecretStore: Send + Sync {
    fn ping(&self) -> impl Future<Output = RepoResult<()>> + Send;
}

/// Append-only event log + subscriber bus. v1 writes a row to the `events`
/// table and notifies in-process SSE subscribers via `tokio::sync::Notify`.
/// v2 also mirrors writes to NATS JetStream — same `append` signature.
pub trait EventBus: Send + Sync {
    fn ping(&self) -> impl Future<Output = RepoResult<()>> + Send;
}
