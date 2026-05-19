//! One-Euro filter for smooth crop panning when following the active speaker.
//!
//! Prevents jarring jumps between frames by adaptively filtering the crop
//! center trajectory. B-roll aware: resets the filter when no face is detected
//! for >= 0.5 s so that cuts to B-roll footage don't drag the old position.
//!
//! Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
//! for Noisy Input in Interactive Systems", CHI 2012.

use std::f64::consts::PI;

/// Tuning knobs for the One-Euro filter.
#[derive(Debug, Clone, Copy)]
pub struct OneEuroParams {
    /// Minimum cutoff frequency (Hz). Lower = smoother but more latency.
    pub min_cutoff: f64,
    /// Speed coefficient. Higher = less smoothing when moving fast.
    pub beta: f64,
    /// Cutoff frequency for the derivative low-pass (Hz).
    pub d_cutoff: f64,
}

impl Default for OneEuroParams {
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.007,
            d_cutoff: 1.0,
        }
    }
}

/// An observed crop-center sample (may have `None` position when no face is detected).
#[derive(Debug, Clone, Copy)]
pub struct CropSample {
    pub timestamp: f64,
    pub center: Option<(f64, f64)>,
}

/// Smoothed output point.
#[derive(Debug, Clone, Copy)]
pub struct SmoothedPoint {
    pub timestamp: f64,
    pub x: f64,
    pub y: f64,
}

/// Duration of missing face detections before the filter resets (seconds).
const BROLL_GAP_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// Low-pass exponential filter (internal building block)
// ---------------------------------------------------------------------------

fn smoothing_factor(te: f64, cutoff: f64) -> f64 {
    let r = 2.0 * PI * cutoff * te;
    r / (r + 1.0)
}

fn exponential_smooth(alpha: f64, x: f64, prev: f64) -> f64 {
    alpha * x + (1.0 - alpha) * prev
}

/// Per-axis One-Euro filter state.
struct AxisFilter {
    prev_value: f64,
    prev_derivative: f64,
    prev_timestamp: f64,
    initialised: bool,
}

impl AxisFilter {
    fn new() -> Self {
        Self {
            prev_value: 0.0,
            prev_derivative: 0.0,
            prev_timestamp: 0.0,
            initialised: false,
        }
    }

    fn reset(&mut self) {
        self.initialised = false;
    }

    fn filter(&mut self, t: f64, x: f64, params: &OneEuroParams) -> f64 {
        if !self.initialised {
            self.prev_value = x;
            self.prev_derivative = 0.0;
            self.prev_timestamp = t;
            self.initialised = true;
            return x;
        }

        let te = t - self.prev_timestamp;
        if te <= 0.0 {
            return self.prev_value;
        }

        // Estimate derivative with a low-pass filter.
        let alpha_d = smoothing_factor(te, params.d_cutoff);
        let dx = (x - self.prev_value) / te;
        let edx = exponential_smooth(alpha_d, dx, self.prev_derivative);

        // Adaptive cutoff based on speed.
        let cutoff = params.min_cutoff + params.beta * edx.abs();
        let alpha = smoothing_factor(te, cutoff);
        let result = exponential_smooth(alpha, x, self.prev_value);

        self.prev_value = result;
        self.prev_derivative = edx;
        self.prev_timestamp = t;
        result
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply the One-Euro filter to a sequence of crop-center samples.
///
/// Samples with `center = None` indicate frames where no face was detected.
/// If no face is detected for >= [`BROLL_GAP_THRESHOLD`] seconds the filter
/// resets so that the next detection starts fresh (B-roll gap handling).
///
/// Returns one [`SmoothedPoint`] per input sample that had a face detection.
pub fn smooth_trajectory(samples: &[CropSample], params: &OneEuroParams) -> Vec<SmoothedPoint> {
    let mut fx = AxisFilter::new();
    let mut fy = AxisFilter::new();
    let mut last_face_time: Option<f64> = None;
    let mut out = Vec::with_capacity(samples.len());

    for s in samples {
        match s.center {
            Some((x, y)) => {
                // B-roll gap detection: reset if the gap since the last face
                // exceeds the threshold.
                if let Some(prev_t) = last_face_time {
                    if s.timestamp - prev_t >= BROLL_GAP_THRESHOLD {
                        fx.reset();
                        fy.reset();
                    }
                }

                let sx = fx.filter(s.timestamp, x, params);
                let sy = fy.filter(s.timestamp, y, params);
                last_face_time = Some(s.timestamp);

                out.push(SmoothedPoint {
                    timestamp: s.timestamp,
                    x: sx,
                    y: sy,
                });
            }
            None => {
                // No face detected — do not produce output; gap may trigger
                // reset on the next detection.
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> OneEuroParams {
        OneEuroParams::default()
    }

    /// A constant position should pass through unmodified.
    #[test]
    fn stationary_input_is_unchanged() {
        let samples: Vec<CropSample> = (0..30)
            .map(|i| CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((540.0, 960.0)),
            })
            .collect();

        let smoothed = smooth_trajectory(&samples, &default_params());
        assert_eq!(smoothed.len(), 30);
        for p in &smoothed {
            assert!((p.x - 540.0).abs() < 1e-9, "x drifted: {}", p.x);
            assert!((p.y - 960.0).abs() < 1e-9, "y drifted: {}", p.y);
        }
    }

    /// A sudden jump should be smoothed (output should lag behind the raw jump).
    #[test]
    fn sudden_jump_is_smoothed() {
        let mut samples = Vec::new();
        // 15 frames at position A, then 15 frames at position B.
        for i in 0..15 {
            samples.push(CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((100.0, 100.0)),
            });
        }
        for i in 15..30 {
            samples.push(CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((500.0, 500.0)),
            });
        }

        let smoothed = smooth_trajectory(&samples, &default_params());
        assert_eq!(smoothed.len(), 30);

        // Frame right after the jump (index 15) should NOT equal the raw value
        // 500 — it should be somewhere between 100 and 500.
        let after_jump = &smoothed[15];
        assert!(
            after_jump.x > 100.0 && after_jump.x < 500.0,
            "expected smoothed value between 100 and 500, got {}",
            after_jump.x
        );

        // Last frame should be close to 500 (converged).
        let last = smoothed.last().unwrap();
        assert!(
            (last.x - 500.0).abs() < 50.0,
            "expected convergence near 500, got {}",
            last.x
        );
    }

    /// After a B-roll gap (>= 0.5 s with no face), the filter should reset and
    /// snap to the new position instead of smoothly interpolating from the old one.
    #[test]
    fn broll_gap_resets_filter() {
        let mut samples = Vec::new();

        // 10 frames at (100, 100) over 0.0–0.3 s.
        for i in 0..10 {
            samples.push(CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((100.0, 100.0)),
            });
        }

        // 0.6 s gap with no face (B-roll).
        for i in 0..18 {
            samples.push(CropSample {
                timestamp: 0.333 + i as f64 / 30.0,
                center: None,
            });
        }

        // Face reappears at (800, 800) after the gap.
        for i in 0..10 {
            samples.push(CropSample {
                timestamp: 1.0 + i as f64 / 30.0,
                center: Some((800.0, 800.0)),
            });
        }

        let smoothed = smooth_trajectory(&samples, &default_params());

        // Should have 20 output points (10 before gap + 10 after).
        assert_eq!(smoothed.len(), 20);

        // The first point after the gap should snap directly to 800 (filter reset).
        let first_after_gap = &smoothed[10];
        assert!(
            (first_after_gap.x - 800.0).abs() < 1e-9,
            "expected snap to 800 after B-roll gap, got {}",
            first_after_gap.x
        );
    }

    /// Short gaps (< 0.5 s) should NOT reset the filter — continuity is preserved.
    #[test]
    fn short_gap_does_not_reset() {
        let mut samples = Vec::new();

        // 10 frames at (100, 100).
        for i in 0..10 {
            samples.push(CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((100.0, 100.0)),
            });
        }

        // Short gap: 0.3 s without face (< 0.5 s threshold).
        for i in 0..9 {
            samples.push(CropSample {
                timestamp: 0.333 + i as f64 / 30.0,
                center: None,
            });
        }

        // Face reappears at (500, 500) — should be smoothed, not snapped.
        for i in 0..10 {
            samples.push(CropSample {
                timestamp: 0.633 + i as f64 / 30.0,
                center: Some((500.0, 500.0)),
            });
        }

        let smoothed = smooth_trajectory(&samples, &default_params());
        assert_eq!(smoothed.len(), 20);

        // First point after short gap should be smoothed (NOT 500).
        let first_after = &smoothed[10];
        assert!(
            first_after.x > 100.0 && first_after.x < 500.0,
            "expected smoothed value after short gap, got {}",
            first_after.x
        );
    }

    /// Custom params should affect smoothing strength.
    #[test]
    fn custom_params_affect_smoothing() {
        let mut samples = Vec::new();
        for i in 0..10 {
            samples.push(CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((100.0, 100.0)),
            });
        }
        for i in 10..20 {
            samples.push(CropSample {
                timestamp: i as f64 / 30.0,
                center: Some((500.0, 500.0)),
            });
        }

        // Very aggressive smoothing (low min_cutoff, low beta).
        let aggressive = OneEuroParams {
            min_cutoff: 0.1,
            beta: 0.0,
            d_cutoff: 1.0,
        };
        let smooth_agg = smooth_trajectory(&samples, &aggressive);

        // Very responsive (high min_cutoff, high beta).
        let responsive = OneEuroParams {
            min_cutoff: 10.0,
            beta: 1.0,
            d_cutoff: 1.0,
        };
        let smooth_resp = smooth_trajectory(&samples, &responsive);

        // Right after the jump, aggressive should be further from 500 than responsive.
        let agg_after = smooth_agg[10].x;
        let resp_after = smooth_resp[10].x;
        assert!(
            (500.0 - agg_after).abs() > (500.0 - resp_after).abs(),
            "aggressive ({agg_after}) should lag more than responsive ({resp_after})"
        );
    }

    /// Empty input produces empty output.
    #[test]
    fn empty_input() {
        let smoothed = smooth_trajectory(&[], &default_params());
        assert!(smoothed.is_empty());
    }

    /// All-None samples produce no output.
    #[test]
    fn all_none_samples() {
        let samples: Vec<CropSample> = (0..10)
            .map(|i| CropSample {
                timestamp: i as f64 / 30.0,
                center: None,
            })
            .collect();
        let smoothed = smooth_trajectory(&samples, &default_params());
        assert!(smoothed.is_empty());
    }
}
