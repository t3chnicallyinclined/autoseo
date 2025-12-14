use std::time::Duration;

use tokio::sync::Mutex;
use tokio::time::{Instant, Interval, MissedTickBehavior};

#[derive(Debug)]
pub struct RpmGate {
    interval: Mutex<Interval>,
}

impl RpmGate {
    pub fn new(rpm: u32) -> Self {
        // Period per request: 60s / rpm. Round up so we never exceed rpm.
        let rpm = rpm.max(1) as f64;
        let period_secs = 60.0 / rpm;
        let nanos = (period_secs * 1_000_000_000.0).ceil() as u64;
        let period = Duration::from_nanos(nanos.max(1));

        let start = Instant::now();
        let mut interval = tokio::time::interval_at(start, period);
        // Important: do NOT allow catch-up bursts if we were previously idle.
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self {
            interval: Mutex::new(interval),
        }
    }

    pub async fn wait(&self) {
        let mut interval = self.interval.lock().await;
        interval.tick().await;
    }
}
