//! Word-level alignment helpers for the clipper pipeline.
//!
//! The transcription call lives in [`crate::openai::OpenAiClient::transcribe_words`].
//! This module provides post-processing utilities:
//!
//! - [`shift_words`] — translate a chunk's word timestamps into absolute episode time.
//! - [`snap_to_word_boundary`] — align a target time to the nearest word edge so
//!   clip cuts never slice mid-word.
//! - [`AlignedWord`] — the absolute-time representation the rest of the clipper consumes.

use crate::openai::TranscriptionWord;

#[derive(Debug, Clone, PartialEq)]
pub struct AlignedWord {
    pub text: String,
    pub start_secs: f64,
    pub end_secs: f64,
}

impl AlignedWord {
    pub fn from_transcription(w: &TranscriptionWord, offset_secs: f64) -> Self {
        Self {
            text: w.word.clone(),
            start_secs: (w.start + offset_secs).max(0.0),
            end_secs: (w.end + offset_secs).max(0.0),
        }
    }

    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }
}

/// Translate a chunk's per-chunk word timestamps into the global episode timeline.
pub fn shift_words(words: &[TranscriptionWord], offset_secs: f64) -> Vec<AlignedWord> {
    words
        .iter()
        .map(|w| AlignedWord::from_transcription(w, offset_secs))
        .collect()
}

/// Snap a target time to the nearest word boundary (start or end of any word) within
/// `max_drift_secs`. Returns the original target if no boundary is close enough.
/// Use this so clip cuts land at word edges, never mid-word.
pub fn snap_to_word_boundary(
    target_secs: f64,
    words: &[AlignedWord],
    max_drift_secs: f64,
) -> f64 {
    if words.is_empty() {
        return target_secs;
    }
    let mut best = target_secs;
    let mut best_drift = max_drift_secs;
    for w in words {
        for boundary in [w.start_secs, w.end_secs] {
            let drift = (boundary - target_secs).abs();
            if drift <= best_drift {
                best_drift = drift;
                best = boundary;
            }
        }
    }
    best
}

/// Compute speaking rate (words per second) inside a `[start, end)` window.
/// Useful as a ranker feature — rapid-fire delivery often signals high energy.
pub fn speaking_rate_wps(words: &[AlignedWord], start_secs: f64, end_secs: f64) -> Option<f64> {
    let duration = (end_secs - start_secs).max(0.0);
    if duration <= 0.0 {
        return None;
    }
    let count = words
        .iter()
        .filter(|w| w.start_secs >= start_secs && w.start_secs < end_secs)
        .count();
    Some(count as f64 / duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::TranscriptionVerboseJson;

    fn tw(word: &str, start: f64, end: f64) -> TranscriptionWord {
        TranscriptionWord {
            word: word.into(),
            start,
            end,
        }
    }

    #[test]
    fn shift_words_adds_offset() {
        let words = vec![tw("hello", 0.0, 0.5), tw("world", 0.6, 1.0)];
        let shifted = shift_words(&words, 10.0);
        assert_eq!(shifted.len(), 2);
        assert_eq!(shifted[0].text, "hello");
        assert!((shifted[0].start_secs - 10.0).abs() < 1e-9);
        assert!((shifted[1].end_secs - 11.0).abs() < 1e-9);
    }

    #[test]
    fn shift_clamps_negative_to_zero() {
        let words = vec![tw("oops", -0.1, 0.2)];
        let shifted = shift_words(&words, 0.0);
        assert_eq!(shifted[0].start_secs, 0.0);
    }

    #[test]
    fn snap_to_word_boundary_picks_closest() {
        let aligned = vec![
            AlignedWord {
                text: "hello".into(),
                start_secs: 1.0,
                end_secs: 1.5,
            },
            AlignedWord {
                text: "world".into(),
                start_secs: 1.6,
                end_secs: 2.0,
            },
        ];
        // Snap to start of "hello"
        assert!((snap_to_word_boundary(0.95, &aligned, 0.1) - 1.0).abs() < 1e-9);
        // Snap to end of "world"
        assert!((snap_to_word_boundary(2.05, &aligned, 0.1) - 2.0).abs() < 1e-9);
        // Outside budget — return original target
        assert!((snap_to_word_boundary(3.0, &aligned, 0.1) - 3.0).abs() < 1e-9);
        // Empty input
        assert!((snap_to_word_boundary(0.5, &[], 1.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn speaking_rate_counts_words_per_second() {
        let words = vec![
            AlignedWord {
                text: "one".into(),
                start_secs: 0.1,
                end_secs: 0.3,
            },
            AlignedWord {
                text: "two".into(),
                start_secs: 0.4,
                end_secs: 0.6,
            },
            AlignedWord {
                text: "three".into(),
                start_secs: 0.7,
                end_secs: 0.9,
            },
            AlignedWord {
                text: "four".into(),
                start_secs: 1.5,
                end_secs: 1.8,
            },
        ];
        // 3 words in [0, 1) => 3 wps
        let r = speaking_rate_wps(&words, 0.0, 1.0).unwrap();
        assert!((r - 3.0).abs() < 1e-9);

        assert!(speaking_rate_wps(&words, 0.0, 0.0).is_none());
    }

    #[test]
    fn deserializes_groq_word_response_shape() {
        // Shape matches what Groq's whisper-large-v3-turbo returns when
        // timestamp_granularities[]=word is requested.
        let body = r#"{
          "task": "transcribe",
          "language": "english",
          "duration": 2.5,
          "text": "hello world",
          "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}
          ],
          "words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0}
          ]
        }"#;
        let parsed: TranscriptionVerboseJson =
            serde_json::from_str(body).expect("parse verbose json");
        assert_eq!(parsed.text, "hello world");
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.words.len(), 2);
        assert_eq!(parsed.words[0].word, "hello");
        assert!((parsed.words[1].end - 1.0).abs() < 1e-9);
    }

    #[test]
    fn deserializes_response_without_words() {
        // Older providers (e.g. OpenAI whisper-1) won't include `words`.
        let body = r#"{
          "task": "transcribe",
          "text": "hello",
          "segments": []
        }"#;
        let parsed: TranscriptionVerboseJson =
            serde_json::from_str(body).expect("parse verbose json");
        assert_eq!(parsed.words.len(), 0, "words should default to empty");
    }
}
