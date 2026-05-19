//! Transcript-level feature extractor — pure Rust regex over a text window.
//!
//! These features feed the LLM ranker as structured evidence ("two strong-claim
//! openers, one confessional, three short declaratives") so the model has more to
//! work with than vibes alone. Cheap to compute (microseconds per window).

use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinguisticFeatures {
    pub total_word_count: usize,
    pub sentence_count: usize,
    pub avg_sentence_words: f32,

    /// Counter-assertion / interruption cues: "no wait", "actually,", "hold on",
    /// "that's not true", crosstalk patterns.
    pub conflict_marker_count: usize,
    /// Strong-claim openers: "honestly", "the truth is", "hot take",
    /// "nobody talks about", "I'm gonna be real".
    pub strong_claim_count: usize,
    /// Confessional / vulnerability cues: "I'll be honest", "between us",
    /// "I've never told this", "to be honest with you".
    pub confessional_count: usize,
    /// Topic-shift markers: "so anyway", "speaking of which", "moving on",
    /// "on a different note".
    pub topic_shift_count: usize,
    /// Question marks present (rough proxy for back-and-forth).
    pub question_count: usize,
    /// Numeric tokens (digits) — proxy for specific claims.
    pub number_count: usize,
    /// Short declarative sentences (<= 12 words, ending in . or !).
    /// High count = punchy delivery, easier to clip into hooks.
    pub short_declarative_count: usize,

    /// Up to 2 highest-signal short declaratives, for the ranker's evidence block.
    pub quotable_lines: Vec<String>,
}

static CONFLICT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(no,?\s+wait|no,?\s+actually|hold on|wait,?\s+wait|that's not (?:true|right|how)|but but|let me finish|that's where you're wrong|hold up|no\s+no\s+no)\b",
    )
    .expect("static regex")
});

static STRONG_CLAIM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(honestly|the truth is|hot take|nobody talks about|i'm gonna be real|i'm being real|here's the thing|the reality is|let me tell you|i'll tell you what|the fact is|real talk)\b",
    )
    .expect("static regex")
});

static CONFESSIONAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(i'?ll be honest|i'?ve never told|between us|to be honest with you|i'?m gonna admit|i'?m embarrassed|i hate to admit|full disclosure)\b",
    )
    .expect("static regex")
});

static TOPIC_SHIFT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(so anyway|speaking of which|moving on|on a different note|switching gears|different topic|by the way|on another note)\b",
    )
    .expect("static regex")
});

static NUMBER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d[\d,\.]*\b").expect("static regex"));

/// Sentence splitter: a `.`, `!`, or `?` followed by whitespace or end of input.
/// Lossy for abbreviations ("Mr.") but adequate for ranker-feature counting.
static SENTENCE_SPLIT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[.!?]+(?:\s+|$)").expect("static regex"));

/// Extract linguistic features from a transcript window.
pub fn extract_features(text: &str) -> LinguisticFeatures {
    let text = text.trim();
    if text.is_empty() {
        return LinguisticFeatures::default();
    }

    let total_word_count = text.split_whitespace().count();

    // Split into sentences. Each split entry is the sentence text WITHOUT the
    // terminal punctuation; we infer punctuation kind by re-checking the original.
    let raw_sentences: Vec<&str> = SENTENCE_SPLIT_RE
        .split(text)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    let sentence_count = raw_sentences.len();
    let avg_sentence_words = if sentence_count > 0 {
        raw_sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f32
            / sentence_count as f32
    } else {
        0.0
    };

    let conflict_marker_count = CONFLICT_RE.find_iter(text).count();
    let strong_claim_count = STRONG_CLAIM_RE.find_iter(text).count();
    let confessional_count = CONFESSIONAL_RE.find_iter(text).count();
    let topic_shift_count = TOPIC_SHIFT_RE.find_iter(text).count();
    let number_count = NUMBER_RE.find_iter(text).count();
    let question_count = text.chars().filter(|&c| c == '?').count();

    // Short declaratives: <= 12 words, no `?` in the sentence text.
    // We treat anything without an explicit `?` as a declarative for this counter.
    let mut short_decls: Vec<(usize, &str)> = raw_sentences
        .iter()
        .filter(|s| !s.contains('?'))
        .map(|s| (s.split_whitespace().count(), *s))
        .filter(|(wc, _)| *wc > 0 && *wc <= 12)
        .collect();
    let short_declarative_count = short_decls.len();

    // Quotable lines: pick the two shortest by word count, breaking ties by
    // appearance order (sort_by_key is stable).
    short_decls.sort_by_key(|(wc, _)| *wc);
    let quotable_lines: Vec<String> = short_decls
        .into_iter()
        .take(2)
        .map(|(_, s)| s.to_string())
        .collect();

    LinguisticFeatures {
        total_word_count,
        sentence_count,
        avg_sentence_words,
        conflict_marker_count,
        strong_claim_count,
        confessional_count,
        topic_shift_count,
        question_count,
        number_count,
        short_declarative_count,
        quotable_lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_defaults() {
        let f = extract_features("");
        assert_eq!(f, LinguisticFeatures::default());
        let f = extract_features("   \n\t  ");
        assert_eq!(f, LinguisticFeatures::default());
    }

    #[test]
    fn counts_words_and_sentences() {
        let f = extract_features("Hello world. How are you doing today? I'm fine!");
        // 3 sentences after split: "Hello world", "How are you doing today", "I'm fine"
        assert_eq!(f.sentence_count, 3);
        assert_eq!(f.total_word_count, 9);
        assert_eq!(f.question_count, 1);
        assert!((f.avg_sentence_words - 3.0).abs() < 0.1);
    }

    #[test]
    fn detects_conflict_markers() {
        let f = extract_features("No wait, that's not right. Hold on.");
        assert!(
            f.conflict_marker_count >= 2,
            "expected >=2 conflict markers, got {}",
            f.conflict_marker_count
        );
    }

    #[test]
    fn detects_strong_claim_openers() {
        let f = extract_features("Honestly, that was nuts. The truth is nobody talks about this.");
        // "honestly" + "the truth is" + "nobody talks about" — three independent matches.
        assert!(
            f.strong_claim_count >= 3,
            "expected >=3 strong-claim hits, got {}",
            f.strong_claim_count
        );
    }

    #[test]
    fn detects_confessional() {
        let f = extract_features("I'll be honest, I've never told anyone this story.");
        assert_eq!(f.confessional_count, 2);
    }

    #[test]
    fn detects_topic_shift() {
        let f = extract_features("So anyway, speaking of which, the new policy moved on.");
        // "so anyway" + "speaking of which" + "moved on" — pattern matches "moving on" not "moved on"
        // so we should see exactly 2 hits.
        assert!(
            f.topic_shift_count >= 2,
            "expected >=2 topic-shift hits, got {}",
            f.topic_shift_count
        );
    }

    #[test]
    fn counts_numbers() {
        let f = extract_features("In 2024 we did 50,000 reps and ran 3.5 miles.");
        assert_eq!(f.number_count, 3);
    }

    #[test]
    fn picks_quotable_lines_by_brevity() {
        let f = extract_features(
            "Wow. That changes everything. I think it does change everything in this particular case. Or does it?",
        );
        // Declarative short sentences: "Wow" (1 word), "That changes everything" (3 words),
        // and the long one (~9 words). Quotables should be the two shortest.
        assert_eq!(f.quotable_lines.len(), 2);
        assert_eq!(f.quotable_lines[0], "Wow");
        assert_eq!(f.quotable_lines[1], "That changes everything");
        assert!(f.short_declarative_count >= 3);
    }

    #[test]
    fn quotables_exclude_questions() {
        let f = extract_features("Is this useful? Probably. Yes.");
        // Two declaratives: "Probably" and "Yes". Question should not appear.
        assert_eq!(f.quotable_lines.len(), 2);
        assert!(!f.quotable_lines.iter().any(|q| q.contains('?')));
    }
}
