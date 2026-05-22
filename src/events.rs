//! Pipeline event types and broadcast infrastructure for real-time WebSocket
//! updates. Events are emitted by the clipper pipeline at each stage transition
//! and broadcast to all connected WebSocket clients via the `/ws` axum route.
//!
//! Wire schema is dashboard-facing: each variant serializes as a flat object
//! with `{"type": "...", ...camelCase fields...}` to match the
//! `WSMessage` union in `autoseo-dashboard/src/contexts/WebSocketContext.tsx`.
//! Do not rename or restructure variants without updating the dashboard too.
//!
//! Variants currently emitted from the pipeline:
//! - `job_update` — every storage status transition
//! - `job_complete` — terminal Done
//! - `job_failed`  — terminal Failed
//!
//! Variants the dashboard understands but the pipeline does not yet publish
//! (left here so future emitters fit the schema): `pipeline_stage`,
//! `post_complete`, `stat_update`, `cost_update`, `agent_status`.

use serde::Serialize;
use tokio::sync::broadcast;

/// Capacity of the broadcast channel. Late/slow readers get `Lagged` and can
/// reconnect. 256 is generous for the volume we emit.
const CHANNEL_CAPACITY: usize = 256;

/// A pipeline event sent to WebSocket clients.
///
/// Each variant flattens its `type` discriminator and field set into a single
/// JSON object (no nested `data`) to match the dashboard's `WSMessage` union.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineEvent {
    /// Every status transition fires one of these. Mirrors `JobUpdate` in
    /// the dashboard.
    JobUpdate {
        #[serde(rename = "jobId")]
        job_id: String,
        /// Dashboard-facing status (`pending`, `transcribing`, `rendering`,
        /// `done`, `failed`) — already collapsed from the internal FSM.
        status: String,
        /// Human-readable stage label (`Queued`, `Transcribed`, …).
        stage: String,
        /// 0..=100 progress percentage.
        progress: u8,
        #[serde(rename = "clipsGenerated", skip_serializing_if = "Option::is_none")]
        clips_generated: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        media: Option<String>,
    },

    /// Terminal success event. Mirrors `JobComplete`.
    JobComplete {
        #[serde(rename = "jobId")]
        job_id: String,
        media: String,
        #[serde(rename = "clipsGenerated")]
        clips_generated: i64,
    },

    /// Terminal failure event. Mirrors `JobFailed`.
    JobFailed {
        #[serde(rename = "jobId")]
        job_id: String,
        media: String,
        error: String,
    },

    /// Per-platform post outcome. Not yet emitted by the pipeline; defined
    /// so future emitters match the dashboard schema.
    PostComplete {
        #[serde(rename = "clipId")]
        clip_id: String,
        #[serde(rename = "clipHook")]
        clip_hook: String,
        platform: String,
        /// `posted` or `failed`.
        status: String,
    },
}

/// Map an internal job status string to the dashboard's (status, stage, progress).
/// Single source of truth shared by the HTTP `Job` mapper, the WS emitter, and
/// the `/api/pipeline/status` projection.
///
/// The internal FSM uses *past-tense gate names* — `transcribed` means STT has
/// completed, not that we're mid-STT. The dashboard's `JobStatus` enum is small
/// (`pending` | `transcribing` | `rendering` | `done` | `failed`), so every
/// non-terminal post-transcribe gate collapses to `rendering`. The `stage`
/// label then describes *what is actually running right now*, which is the
/// stage **after** the gate's name (e.g. internal=`transcribed` → currently
/// running feature extraction + ranking).
pub fn dashboard_view(internal_status: &str) -> (&'static str, &'static str, u8) {
    match internal_status {
        "pending" => ("pending", "Queued", 5),
        // STT done → now extracting features, ranking, generating social copy.
        "transcribed" => ("rendering", "Ranking", 40),
        // Ranker done → now VLM re-rank + ASD + cutting/rendering clips.
        "ranked" => ("rendering", "Rendering clips", 65),
        // Renders done → uploading to R2 + posting to platforms.
        "rendered" => ("rendering", "Posting", 85),
        // Posts attempted → finalizing cost + writing digest.
        "posted" => ("rendering", "Finalizing", 95),
        "done" => ("done", "Complete", 100),
        "failed" => ("failed", "Failed", 0),
        "cancelled" => ("failed", "Cancelled", 0),
        _ => ("pending", "Unknown", 5),
    }
}

/// Thin wrapper around `tokio::sync::broadcast` for pipeline events.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<PipelineEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self { tx }
    }

    /// Send an event. Returns Ok even if there are no receivers
    /// (fire-and-forget — silent drop is correct when no dashboard is open).
    pub fn emit(&self, event: PipelineEvent) {
        let _ = self.tx.send(event);
    }

    /// Subscribe to the event stream. Each subscriber gets its own independent
    /// cursor into the broadcast channel.
    pub fn subscribe(&self) -> broadcast::Receiver<PipelineEvent> {
        self.tx.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_update_flat_camelcase() {
        let event = PipelineEvent::JobUpdate {
            job_id: "abc-123".into(),
            status: "transcribing".into(),
            stage: "Transcribed".into(),
            progress: 30,
            clips_generated: None,
            media: Some("ep.mp4".into()),
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
        assert_eq!(v["type"], "job_update");
        assert_eq!(v["jobId"], "abc-123"); // flat, camelCase
        assert_eq!(v["progress"], 30);
        assert_eq!(v["media"], "ep.mp4");
        assert!(v.get("clipsGenerated").is_none()); // None skipped
    }

    #[test]
    fn job_complete_flat_camelcase() {
        let event = PipelineEvent::JobComplete {
            job_id: "job-1".into(),
            media: "ep.mp4".into(),
            clips_generated: 5,
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
        assert_eq!(v["type"], "job_complete");
        assert_eq!(v["clipsGenerated"], 5);
        assert_eq!(v["media"], "ep.mp4");
    }

    #[test]
    fn job_failed_flat_camelcase() {
        let event = PipelineEvent::JobFailed {
            job_id: "job-2".into(),
            media: "ep.mp4".into(),
            error: "out of memory".into(),
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
        assert_eq!(v["type"], "job_failed");
        assert_eq!(v["error"], "out of memory");
    }

    #[test]
    fn post_complete_flat_camelcase() {
        let event = PipelineEvent::PostComplete {
            clip_id: "clip-1".into(),
            clip_hook: "the punchline".into(),
            platform: "youtube".into(),
            status: "posted".into(),
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
        assert_eq!(v["type"], "post_complete");
        assert_eq!(v["clipId"], "clip-1");
        assert_eq!(v["clipHook"], "the punchline");
    }

    #[test]
    fn dashboard_view_covers_all_internal_statuses() {
        assert_eq!(dashboard_view("pending").0, "pending");
        // Post-transcribe gates all collapse to "rendering" (dashboard's
        // small JobStatus enum has no "ranking" value); the stage label is
        // what changes to reflect what's actually running.
        assert_eq!(dashboard_view("transcribed").0, "rendering");
        assert_eq!(dashboard_view("transcribed").1, "Ranking");
        assert_eq!(dashboard_view("ranked").0, "rendering");
        assert_eq!(dashboard_view("ranked").1, "Rendering clips");
        assert_eq!(dashboard_view("rendered").0, "rendering");
        assert_eq!(dashboard_view("rendered").1, "Posting");
        assert_eq!(dashboard_view("posted").0, "rendering");
        assert_eq!(dashboard_view("posted").1, "Finalizing");
        assert_eq!(dashboard_view("done").0, "done");
        assert_eq!(dashboard_view("failed").0, "failed");
        // Unknown collapses to pending.
        assert_eq!(dashboard_view("weird").0, "pending");
        // Done is 100%, failed is 0%.
        assert_eq!(dashboard_view("done").2, 100);
        assert_eq!(dashboard_view("failed").2, 0);
    }

    #[tokio::test]
    async fn event_bus_broadcast_to_multiple_receivers() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.emit(PipelineEvent::JobUpdate {
            job_id: "j1".into(),
            status: "rendering".into(),
            stage: "Ranked".into(),
            progress: 55,
            clips_generated: None,
            media: None,
        });

        let j1 = serde_json::to_string(&rx1.recv().await.unwrap()).unwrap();
        let j2 = serde_json::to_string(&rx2.recv().await.unwrap()).unwrap();
        assert_eq!(j1, j2);
    }

    #[tokio::test]
    async fn emit_without_receivers_does_not_panic() {
        let bus = EventBus::new();
        bus.emit(PipelineEvent::JobComplete {
            job_id: "j1".into(),
            media: "ep.mp4".into(),
            clips_generated: 3,
        });
    }
}
