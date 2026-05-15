//! Library re-exports for integration tests.
//!
//! The binary entry point is `main.rs`; this file exposes the pipeline modules
//! so that `tests/` integration tests can exercise them with synthetic media
//! without needing real API keys.

pub mod ai_pipeline;
pub mod align;
pub mod candidates;
pub mod captions;
pub mod config;
pub mod linguistic_markers;
pub mod media;
pub mod openai;
pub mod prosody;
pub mod render;
pub mod scene;
pub mod vad;

// Modules below are not needed by integration tests but must be compiled
// alongside the public ones because of internal `use crate::` imports.
mod embed;
mod rate_limit;
