//! Dashboard-specific tower middleware.
//!
//! Slice 0 is bare (only tracing + tower-http defaults applied at server
//! build). Slice 2 adds: request_id, auth, CSRF.
