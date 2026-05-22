//! Dense candidate window generation + per-window feature aggregation.
//!
//! Walks the episode in sliding strides, proposes a target-length window at each
//! step, snaps the boundaries to natural pauses (silence) and shot cuts, then attaches
//! every signal the ranker will use:
//!
//! - transcript text (from word timestamps)
//! - linguistic markers (counts + quotable lines)
//! - prosody (RMS peak + mean in window)
//! - speaking rate (words / sec)
//! - novelty score (optional — filled in by `attach_novelty` if an embedder is available)
//!
//! The novelty pass is separated so generation stays pure / sync; only the embedding
//! pass touches async and an ML model.

use anyhow::Result;

use crate::align::{self, AlignedWord};
use crate::ast::{self, AudioEvents, ScoredWindow};
use crate::embed::{self, Embedder};
use crate::linguistic_markers::{self, LinguisticFeatures};
use crate::prosody::{self, F0Sample, RmsWindow};
use crate::scene;
use crate::vad::{self, SilenceWindow};

#[derive(Debug, Clone)]
pub struct CandidateWindow {
    pub start_secs: f64,
    pub end_secs: f64,
    pub transcript: String,
    pub word_count: usize,
    pub linguistic: LinguisticFeatures,
    pub rms_peak_db: Option<f64>,
    pub rms_mean_db: Option<f64>,
    pub f0_mean_hz: Option<f64>,
    pub f0_variance_hz2: Option<f64>,
    pub f0_peak_hz: Option<f64>,
    pub speaking_rate_wps: Option<f64>,
    /// 0..=1, set by [`attach_novelty`]. `None` until the embedding pass runs.
    pub novelty_score: Option<f64>,
    /// Per-window audio event scores from AST. `None` until the AST pass runs.
    pub audio_events: Option<AudioEvents>,
}

impl CandidateWindow {
    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct CandidateGenerator {
    /// Target clip length the LLM ranker should aim for.
    pub target_secs: f64,
    /// Reject candidates shorter than this after snapping.
    pub min_secs: f64,
    /// Truncate candidates longer than this after snapping.
    pub max_secs: f64,
    /// Sliding-window stride between candidate proposals.
    pub stride_secs: f64,
    /// Max distance to drift when snapping start/end to silence.
    pub silence_drift_secs: f64,
    /// Max distance to drift when snapping start/end to a shot boundary.
    pub shot_drift_secs: f64,
    /// Minimum words required for the candidate to be retained — silent or
    /// monologue-free spans are useless to the ranker.
    pub min_words: usize,
}

impl Default for CandidateGenerator {
    fn default() -> Self {
        Self {
            target_secs: 60.0,
            min_secs: 25.0,
            max_secs: 90.0,
            stride_secs: 30.0,
            silence_drift_secs: 2.0,
            shot_drift_secs: 1.0,
            min_words: 30,
        }
    }
}

impl CandidateGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a generator from the runtime [`crate::config::Config`] so the
    /// clip-duration knobs (`CLIP_MIN_SECS` / `CLIP_MAX_SECS` /
    /// `CLIP_TARGET_SECS` / `CLIP_STRIDE_SECS` / `CLIP_MIN_WORDS`) flow
    /// through to candidate generation without touching this struct's
    /// internals. Defaults from `Default` apply for unset knobs.
    ///
    /// The min/max/target are floored at sane minima (5s / >min / >=min)
    /// to keep downstream code from receiving nonsense windows; the
    /// generator already filters by `min_words` so very short candidates
    /// with too few words drop out anyway.
    pub fn from_config(cfg: &crate::config::Config) -> Self {
        let mut g = Self::default();
        g.min_secs = cfg.clip_min_secs.max(5.0);
        g.max_secs = cfg.clip_max_secs.max(g.min_secs + 1.0);
        g.target_secs = cfg.clip_target_secs.clamp(g.min_secs, g.max_secs);
        if cfg.clip_stride_secs > 0.0 {
            g.stride_secs = cfg.clip_stride_secs;
        }
        if cfg.clip_min_words > 0 {
            g.min_words = cfg.clip_min_words;
        }
        g
    }

    /// Produce candidate windows over `[0, total_duration_secs)`. Pure, sync.
    /// `novelty_score` is left `None` on every window; call [`attach_novelty`] later.
    pub fn generate(
        &self,
        total_duration_secs: f64,
        words: &[AlignedWord],
        silences: &[SilenceWindow],
        shots: &[f64],
        rms_curve: &[RmsWindow],
        f0_curve: &[F0Sample],
    ) -> Vec<CandidateWindow> {
        if total_duration_secs < self.min_secs {
            return Vec::new();
        }

        let mut out: Vec<CandidateWindow> = Vec::new();
        let mut t = 0.0;
        while t + self.min_secs <= total_duration_secs {
            let proposed_start = t;
            let proposed_end = (t + self.target_secs).min(total_duration_secs);

            let (start, end) = self.snap_window(
                proposed_start,
                proposed_end,
                silences,
                shots,
                total_duration_secs,
            );

            if end - start >= self.min_secs {
                let window = build_window(start, end, words, rms_curve, f0_curve);
                if window.word_count >= self.min_words {
                    // Skip near-duplicate of previous window.
                    let dup = out.last().is_some_and(|prev| {
                        (prev.start_secs - window.start_secs).abs() < 0.5
                            && (prev.end_secs - window.end_secs).abs() < 0.5
                    });
                    if !dup {
                        out.push(window);
                    }
                }
            }
            t += self.stride_secs;
        }
        out
    }

    fn snap_window(
        &self,
        start: f64,
        end: f64,
        silences: &[SilenceWindow],
        shots: &[f64],
        total_duration: f64,
    ) -> (f64, f64) {
        // Prefer silence snap (more semantic), fall back to shot snap.
        let snapped_start = {
            let by_silence =
                vad::snap_to_silence_boundary(start, silences, self.silence_drift_secs);
            if (by_silence - start).abs() < f64::EPSILON {
                scene::snap_to_shot(start, shots, self.shot_drift_secs)
            } else {
                by_silence
            }
        };
        let snapped_end = {
            let by_silence = vad::snap_to_silence_boundary(end, silences, self.silence_drift_secs);
            if (by_silence - end).abs() < f64::EPSILON {
                scene::snap_to_shot(end, shots, self.shot_drift_secs)
            } else {
                by_silence
            }
        };

        let start = snapped_start.max(0.0).min(total_duration);
        let mut end = snapped_end.max(start).min(total_duration);
        // Clamp to [min, max] relative to start.
        let max_end = (start + self.max_secs).min(total_duration);
        if end > max_end {
            end = max_end;
        }
        let min_end = (start + self.min_secs).min(total_duration);
        if end < min_end {
            end = min_end;
        }
        (start, end)
    }
}

/// Attach novelty scores to a candidate set by embedding each window's transcript
/// and computing cosine distance to the centroid (normalized 0..=1 within the batch).
/// Mutates in place.
pub async fn attach_novelty(windows: &mut [CandidateWindow], embedder: &Embedder) -> Result<()> {
    if windows.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> = windows.iter().map(|w| w.transcript.clone()).collect();
    let vecs = embedder.embed(texts).await?;
    let scores = embed::score_novelty(&vecs);
    for (w, s) in windows.iter_mut().zip(scores) {
        w.novelty_score = Some(s);
    }
    Ok(())
}

fn build_window(
    start: f64,
    end: f64,
    words: &[AlignedWord],
    rms_curve: &[RmsWindow],
    f0_curve: &[F0Sample],
) -> CandidateWindow {
    let in_window: Vec<&AlignedWord> = words
        .iter()
        .filter(|w| w.start_secs >= start && w.start_secs < end)
        .collect();
    let transcript = in_window
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let word_count = in_window.len();
    let linguistic = linguistic_markers::extract_features(&transcript);
    let speaking_rate_wps = align::speaking_rate_wps(words, start, end);
    let rms_peak_db = prosody::peak_in_range(rms_curve, start, end).map(|w| w.rms_db);
    let rms_mean_db = prosody::mean_in_range(rms_curve, start, end);
    let f0_stats = prosody::f0_stats_in_range(f0_curve, start, end);

    CandidateWindow {
        start_secs: start,
        end_secs: end,
        transcript,
        word_count,
        linguistic,
        rms_peak_db,
        rms_mean_db,
        f0_mean_hz: f0_stats.as_ref().map(|s| s.mean_hz),
        f0_variance_hz2: f0_stats.as_ref().map(|s| s.variance_hz2),
        f0_peak_hz: f0_stats.map(|s| s.peak_hz),
        speaking_rate_wps,
        novelty_score: None,
        audio_events: None,
    }
}

/// Attach AST audio-event scores to each candidate by aggregating overlapping
/// scored windows. Mutates in place.
pub fn attach_audio_events(windows: &mut [CandidateWindow], scored: &[ScoredWindow]) {
    for w in windows.iter_mut() {
        w.audio_events = ast::aggregate_events(scored, w.start_secs, w.end_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> AlignedWord {
        AlignedWord {
            text: text.into(),
            start_secs: start,
            end_secs: end,
        }
    }

    fn dense_words(total_duration: f64, words_per_sec: f64) -> Vec<AlignedWord> {
        let n = (total_duration * words_per_sec) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / words_per_sec;
                word("word", t, t + (1.0 / words_per_sec) * 0.8)
            })
            .collect()
    }

    #[test]
    fn empty_duration_returns_empty() {
        let g = CandidateGenerator::new();
        let out = g.generate(0.0, &[], &[], &[], &[], &[]);
        assert!(out.is_empty());
        let out = g.generate(10.0, &[], &[], &[], &[], &[]);
        // 10s < default min_secs (25s) — should be empty.
        assert!(out.is_empty());
    }

    #[test]
    fn generates_regular_windows_with_no_signals() {
        // 5 minutes of dense speech, no silences or shots, no RMS.
        let words = dense_words(300.0, 3.0); // 900 words at 3wps
        let g = CandidateGenerator::new();
        let out = g.generate(300.0, &words, &[], &[], &[], &[]);
        assert!(!out.is_empty(), "expected several candidate windows");
        for w in &out {
            assert!(
                w.duration_secs() >= g.min_secs - 1e-6,
                "window {:?} below min duration",
                (w.start_secs, w.end_secs)
            );
            assert!(
                w.duration_secs() <= g.max_secs + 1e-6,
                "window {:?} above max duration",
                (w.start_secs, w.end_secs)
            );
            assert!(w.word_count >= g.min_words);
        }

        // Windows should advance by roughly stride_secs from one to the next.
        for pair in out.windows(2) {
            let delta = pair[1].start_secs - pair[0].start_secs;
            assert!(
                delta > 0.0 && delta <= g.stride_secs * 2.0,
                "non-monotonic or huge gap between windows: {delta}"
            );
        }
    }

    #[test]
    fn drops_silent_or_low_word_windows() {
        // 2 minutes total but only a few words at the very start.
        let mut words = vec![
            word("hello", 0.5, 0.7),
            word("world", 0.8, 1.0),
            word("test", 1.1, 1.3),
        ];
        // Pad time but no more words.
        let g = CandidateGenerator::new();
        let out = g.generate(120.0, &mut words, &[], &[], &[], &[]);
        // Below default min_words (30), so even though windows fit, none pass.
        assert!(
            out.is_empty(),
            "expected empty result for low-word episode, got {} windows",
            out.len()
        );
    }

    #[test]
    fn snaps_start_to_nearby_silence() {
        // Propose window starting at t=0; place a silence_end at 0.5s within drift budget.
        let words = dense_words(200.0, 3.0);
        let silences = vec![SilenceWindow {
            start_secs: 0.0,
            end_secs: 0.5,
        }];
        let g = CandidateGenerator::new();
        let out = g.generate(200.0, &words, &silences, &[], &[], &[]);
        let first = out.first().expect("at least one window");
        // First proposed start is 0.0; the snap-target boundary is 0.0 (silence_start)
        // — already there. The next thing snap_to_silence_boundary might prefer is 0.5
        // (silence_end). It picks the closest within drift; for target 0.0 the closest
        // is 0.0 itself, so the start should be 0.0.
        assert_eq!(first.start_secs, 0.0);
    }

    #[test]
    fn aggregates_features_per_window() {
        let words = vec![
            word("Honestly", 0.0, 0.5),
            word("nobody", 0.5, 0.9),
            word("talks", 0.9, 1.2),
            word("about", 1.2, 1.5),
            word("this", 1.5, 1.8),
        ];
        // Extend with filler so the window passes min_words.
        let mut all = words.clone();
        for i in 0..40 {
            let t = 2.0 + (i as f64) * 0.3;
            all.push(word("filler", t, t + 0.2));
        }
        let total = 30.0;
        let rms = vec![
            RmsWindow {
                start_secs: 0.0,
                rms_db: -10.0,
            },
            RmsWindow {
                start_secs: 10.0,
                rms_db: -30.0,
            },
        ];
        let g = CandidateGenerator {
            target_secs: 25.0,
            min_secs: 20.0,
            stride_secs: 25.0,
            min_words: 5,
            ..Default::default()
        };
        let out = g.generate(total, &all, &[], &[], &rms, &[]);
        assert!(!out.is_empty(), "expected at least one window");
        let w = &out[0];
        assert!(w.transcript.contains("Honestly"));
        assert!(
            w.linguistic.strong_claim_count >= 1,
            "expected to detect 'Honestly' / 'nobody talks about'"
        );
        assert!(w.rms_peak_db.is_some());
        assert!(w.speaking_rate_wps.is_some());
        assert!(w.novelty_score.is_none(), "novelty filled by separate pass");
    }
}
