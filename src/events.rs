//! Pipeline event types and broadcast infrastructure for real-time WebSocket
//! updates. Events are emitted by the clipper pipeline at each stage transition
//! and broadcast to all connected WebSocket clients.

use serde::Serialize;
use tokio::sync::broadcast;

/// Capacity of the broadcast channel. Late/slow readers will get a `Lagged`
/// error and can reconnect. 256 is generous for the volume of events we emit.
const CHANNEL_CAPACITY: usize = 256;

/// A pipeline event sent to WebSocket clients.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum PipelineEvent {
    /// A job transitioned to a new pipeline stage.
    #[serde(rename = "job_stage_change")]
    JobStageChange {
        job_id: String,
        stage: String,
        progress: f64,
    },

    /// A job completed successfully.
    #[serde(rename = "job_complete")]
    JobComplete {
        job_id: String,
        clips_count: usize,
    },

    /// A job failed.
    #[serde(rename = "job_failed")]
    JobFailed {
        job_id: String,
        error: String,
    },

    /// A clip was posted to a social platform.
    #[serde(rename = "post_complete")]
    PostComplete {
        clip_id: String,
        platform: String,
        url: String,
    },

    /// Cost tracking update.
    #[serde(rename = "cost_update")]
    CostUpdate {
        total_cents: u64,
        breakdown: serde_json::Value,
    },
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

    /// Send an event. Returns Ok even if there are no receivers (fire-and-forget).
    pub fn emit(&self, event: PipelineEvent) {
        // Ignore send errors (no active receivers).
        let _ = self.tx.send(event);
    }

    /// Subscribe to the event stream. Each subscriber gets its own independent
    /// cursor into the broadcast channel.
    pub fn subscribe(&self) -> broadcast::Receiver<PipelineEvent> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serializes_as_tagged_json() {
        let event = PipelineEvent::JobStageChange {
            job_id: "abc-123".into(),
            stage: "transcribing".into(),
            progress: 0.42,
        };
        let json = serde_json::to_string(&event).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "job_stage_change");
        assert_eq!(v["data"]["job_id"], "abc-123");
        assert_eq!(v["data"]["stage"], "transcribing");
        assert!((v["data"]["progress"].as_f64().unwrap() - 0.42).abs() < 1e-9);
    }

    #[test]
    fn job_complete_serializes() {
        let event = PipelineEvent::JobComplete {
            job_id: "job-1".into(),
            clips_count: 5,
        };
        let json = serde_json::to_string(&event).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "job_complete");
        assert_eq!(v["data"]["clips_count"], 5);
    }

    #[test]
    fn job_failed_serializes() {
        let event = PipelineEvent::JobFailed {
            job_id: "job-2".into(),
            error: "out of memory".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "job_failed");
        assert_eq!(v["data"]["error"], "out of memory");
    }

    #[test]
    fn post_complete_serializes() {
        let event = PipelineEvent::PostComplete {
            clip_id: "clip-1".into(),
            platform: "youtube".into(),
            url: "https://youtube.com/shorts/abc".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "post_complete");
        assert_eq!(v["data"]["platform"], "youtube");
    }

    #[test]
    fn cost_update_serializes() {
        let event = PipelineEvent::CostUpdate {
            total_cents: 150,
            breakdown: serde_json::json!({"transcription": 80, "llm": 70}),
        };
        let json = serde_json::to_string(&event).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "cost_update");
        assert_eq!(v["data"]["total_cents"], 150);
        assert_eq!(v["data"]["breakdown"]["transcription"], 80);
    }

    #[tokio::test]
    async fn event_bus_broadcast_to_multiple_receivers() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        bus.emit(PipelineEvent::JobStageChange {
            job_id: "j1".into(),
            stage: "ranking".into(),
            progress: 0.5,
        });

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();

        let j1 = serde_json::to_string(&e1).unwrap();
        let j2 = serde_json::to_string(&e2).unwrap();
        assert_eq!(j1, j2);
    }

    #[tokio::test]
    async fn emit_without_receivers_does_not_panic() {
        let bus = EventBus::new();
        bus.emit(PipelineEvent::JobComplete {
            job_id: "j1".into(),
            clips_count: 3,
        });
        // No panic — fire-and-forget.
    }
}
