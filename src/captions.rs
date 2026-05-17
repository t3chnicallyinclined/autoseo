//! Generate an Advanced SubStation Alpha (`.ass`) subtitle file from per-word
//! transcript timestamps. ffmpeg burns the result via `subtitles='path.ass'` —
//! see [`crate::render::render_clip`].
//!
//! Supports two modes:
//! - **Phrase-only** (M1): uniform white text, no per-word highlight. Used when
//!   word-level timestamps are unavailable or all words in a phrase share the
//!   same timing.
//! - **Karaoke** (M3): each word lights up as it's spoken via ASS `{\k}` tags.
//!   The highlight color is configurable via [`CaptionStyle::highlight_bgr`].

use anyhow::Context;
use std::path::Path;

use crate::align::AlignedWord;

#[derive(Debug, Clone)]
pub struct CaptionStyle {
    pub font_name: String,
    pub font_size_pt: u32,
    /// 0xBBGGRR — ASS uses BGR not RGB.
    pub primary_bgr: u32,
    /// Outline color (also BGR).
    pub outline_bgr: u32,
    /// Outline thickness in pixels.
    pub outline_px: u32,
    /// Vertical margin from the bottom of the frame, in pixels.
    pub margin_v_px: u32,
    /// Maximum words allowed in a single phrase line.
    pub max_words_per_phrase: usize,
    /// Soft cap on phrase character count before forcing a flush.
    pub max_chars_per_phrase: usize,
    /// 0xBBGGRR — highlight (active-word) color for karaoke mode.
    /// When set to `Some`, per-word `{\k}` tags are emitted.
    /// `None` disables karaoke and produces plain phrase captions (M1 behavior).
    pub highlight_bgr: Option<u32>,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self::for_vertical()
    }
}

impl CaptionStyle {
    /// Caption style tuned for 9:16 vertical (1080×1920). Large font, bottom-center
    /// with a thumb-safe margin from the bottom edge.
    pub fn for_vertical() -> Self {
        Self {
            font_name: "Montserrat".into(),
            font_size_pt: 80,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 4,
            margin_v_px: 200,
            max_words_per_phrase: 4,
            max_chars_per_phrase: 28,
            highlight_bgr: Some(0x00FFFF), // yellow in BGR
        }
    }

    /// Caption style tuned for 1:1 square (1080×1080). Smaller font, tighter margin.
    pub fn for_square() -> Self {
        Self {
            font_name: "Montserrat".into(),
            font_size_pt: 60,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 3,
            margin_v_px: 80,
            max_words_per_phrase: 5,
            max_chars_per_phrase: 36,
            highlight_bgr: Some(0x00FFFF),
        }
    }

    /// Caption style tuned for 16:9 landscape (1920×1080). Smaller font (more
    /// horizontal real estate, longer phrases possible).
    pub fn for_landscape() -> Self {
        Self {
            font_name: "Montserrat".into(),
            font_size_pt: 50,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 3,
            margin_v_px: 60,
            max_words_per_phrase: 7,
            max_chars_per_phrase: 52,
            highlight_bgr: Some(0x00FFFF),
        }
    }
}

/// Write an `.ass` file for the given word stream. Word timestamps must be in
/// CLIP-LOCAL seconds (so subtract the clip's start_secs before calling).
/// `play_res_w` and `play_res_h` should match the rendered video size so libass
/// scales the font correctly.
pub async fn write_ass(
    path: &Path,
    words: &[AlignedWord],
    play_res_w: u32,
    play_res_h: u32,
    style: &CaptionStyle,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let body = render_ass(words, play_res_w, play_res_h, style);
    tokio::fs::write(path, body)
        .await
        .with_context(|| format!("write ass file {}", path.display()))?;
    Ok(())
}

/// Overlay style — for the "WAIT FOR IT"-style hook shown over the first
/// ~1.5 seconds of a clip. Big, bold, centered, fades in/out.
#[derive(Debug, Clone)]
pub struct OverlayStyle {
    pub font_name: String,
    pub font_size_pt: u32,
    pub primary_bgr: u32,
    pub outline_bgr: u32,
    pub outline_px: u32,
    /// Vertical margin used as a hint (ASS alignment 5 is middle-center, so this is unused;
    /// kept for future flexibility).
    pub margin_v_px: u32,
    /// Total on-screen duration in seconds (default 1.5).
    pub duration_secs: f64,
    /// Fade in / fade out duration in seconds (default 0.3 each).
    pub fade_secs: f64,
}

impl OverlayStyle {
    pub fn for_vertical() -> Self {
        Self {
            font_name: "Montserrat".into(),
            font_size_pt: 130,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 5,
            margin_v_px: 0,
            duration_secs: 1.5,
            fade_secs: 0.3,
        }
    }

    pub fn for_square() -> Self {
        Self {
            font_size_pt: 100,
            ..Self::for_vertical()
        }
    }

    pub fn for_landscape() -> Self {
        Self {
            font_size_pt: 80,
            ..Self::for_vertical()
        }
    }
}

/// Write an overlay-only `.ass` file containing a single Dialogue event for
/// the hook text, centered in the frame with a fade-in/out animation.
pub async fn write_overlay_ass(
    path: &Path,
    hook: &str,
    play_res_w: u32,
    play_res_h: u32,
    style: &OverlayStyle,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    let body = render_overlay_ass(hook, play_res_w, play_res_h, style);
    tokio::fs::write(path, body)
        .await
        .with_context(|| format!("write overlay ass {}", path.display()))?;
    Ok(())
}

/// Pure function: produce the overlay `.ass` text for a single hook event.
pub fn render_overlay_ass(
    hook: &str,
    play_res_w: u32,
    play_res_h: u32,
    style: &OverlayStyle,
) -> String {
    let mut out = String::new();
    out.push_str(&header(play_res_w, play_res_h));
    out.push_str(&overlay_style_block(style));
    out.push_str("[Events]\n");
    out.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );

    let text = sanitize_text(hook.trim());
    if text.is_empty() {
        return out;
    }
    let start = format_ass_time(0.0);
    let end = format_ass_time(style.duration_secs.max(0.1));
    let fade_ms = (style.fade_secs.max(0.0) * 1000.0).round() as u64;
    let fade_tag = if fade_ms > 0 {
        format!("{{\\fade({fade_ms},{fade_ms})}}")
    } else {
        String::new()
    };
    // Layer 1 so overlay draws above any captions burned via a later subtitles filter.
    out.push_str(&format!(
        "Dialogue: 1,{start},{end},Overlay,,0,0,0,,{fade_tag}{text}\n"
    ));
    out
}

fn overlay_style_block(style: &OverlayStyle) -> String {
    let primary = format!("&H00{:06X}", style.primary_bgr & 0xFFFFFF);
    let outline = format!("&H00{:06X}", style.outline_bgr & 0xFFFFFF);
    // Alignment 5 = middle-center.
    format!(
        "[V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Overlay,{font},{size},{primary},{primary},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{out_px},0,5,40,40,0,1\n\
         \n",
        font = style.font_name,
        size = style.font_size_pt,
        primary = primary,
        outline = outline,
        out_px = style.outline_px,
    )
}

/// Pure function for testability: produce the full `.ass` text.
pub fn render_ass(
    words: &[AlignedWord],
    play_res_w: u32,
    play_res_h: u32,
    style: &CaptionStyle,
) -> String {
    let phrases = group_into_phrases(words, style.max_words_per_phrase, style.max_chars_per_phrase);
    let karaoke = style.highlight_bgr.is_some();

    let mut out = String::new();
    out.push_str(&header(play_res_w, play_res_h));
    out.push_str(&style_block(style));
    out.push_str("[Events]\n");
    out.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );
    for p in &phrases {
        let start = format_ass_time(p.start_secs);
        let end = format_ass_time(p.end_secs);
        let text = if karaoke && p.has_word_timing() {
            build_karaoke_text(&p.words)
        } else {
            sanitize_text(&p.text)
        };
        out.push_str(&format!(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
        ));
    }
    out
}

/// Build ASS dialogue text with per-word `{\kN}` (karaoke fill) tags.
/// `N` is the word duration in centiseconds.
fn build_karaoke_text(words: &[PhraseWord]) -> String {
    let mut out = String::new();
    for (i, pw) in words.iter().enumerate() {
        let dur_cs = ((pw.end_secs - pw.start_secs).max(0.0) * 100.0).round() as u64;
        // {\kN} — karaoke fill: text transitions from SecondaryColour to
        // PrimaryColour over N centiseconds.
        out.push_str(&format!("{{\\k{dur_cs}}}"));
        out.push_str(&sanitize_word(&pw.text));
        if i + 1 < words.len() {
            out.push(' ');
        }
    }
    out
}

/// Sanitize a single word for use inside a karaoke-tagged dialogue line.
/// Unlike `sanitize_text`, we do NOT escape `{` / `}` because those are our
/// own override tags — but we still escape backslashes and strip newlines.
fn sanitize_word(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', " ")
}

#[derive(Debug, Clone, PartialEq)]
struct PhraseWord {
    text: String,
    start_secs: f64,
    end_secs: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Phrase {
    start_secs: f64,
    end_secs: f64,
    text: String,
    words: Vec<PhraseWord>,
}

impl Phrase {
    /// Returns true if words have distinct timing (i.e. not all identical).
    fn has_word_timing(&self) -> bool {
        if self.words.len() <= 1 {
            return self.words.len() == 1;
        }
        // Check that at least two words have different start times.
        let first_start = self.words[0].start_secs;
        self.words.iter().any(|w| (w.start_secs - first_start).abs() > 0.001)
    }
}

fn group_into_phrases(
    words: &[AlignedWord],
    max_words: usize,
    max_chars: usize,
) -> Vec<Phrase> {
    let max_words = max_words.max(1);
    let max_chars = max_chars.max(8);

    let mut out: Vec<Phrase> = Vec::new();
    let mut buf: Vec<&AlignedWord> = Vec::new();
    let mut chars = 0usize;

    let flush = |out: &mut Vec<Phrase>, buf: &mut Vec<&AlignedWord>, chars: &mut usize| {
        if buf.is_empty() {
            return;
        }
        let start = buf.first().unwrap().start_secs;
        let end = buf.last().unwrap().end_secs;
        let phrase_words: Vec<PhraseWord> = buf
            .iter()
            .filter(|w| !w.text.trim().is_empty())
            .map(|w| PhraseWord {
                text: w.text.trim().to_string(),
                start_secs: w.start_secs,
                end_secs: w.end_secs,
            })
            .collect();
        let text = phrase_words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if end > start && !text.is_empty() {
            out.push(Phrase {
                start_secs: start,
                end_secs: end,
                text,
                words: phrase_words,
            });
        }
        buf.clear();
        *chars = 0;
    };

    for w in words {
        let token = w.text.trim();
        if token.is_empty() {
            continue;
        }
        let would_be_chars = chars + token.chars().count() + if buf.is_empty() { 0 } else { 1 };
        let would_be_words = buf.len() + 1;

        // Hard flush if the next word would exceed budgets.
        if !buf.is_empty() && (would_be_words > max_words || would_be_chars > max_chars) {
            flush(&mut out, &mut buf, &mut chars);
        }

        let token_chars = token.chars().count();
        chars += token_chars + if buf.is_empty() { 0 } else { 1 };
        buf.push(w);

        // Soft flush on strong sentence punctuation.
        if token.ends_with('.') || token.ends_with('!') || token.ends_with('?') {
            flush(&mut out, &mut buf, &mut chars);
        }
    }
    flush(&mut out, &mut buf, &mut chars);
    out
}

fn header(w: u32, h: u32) -> String {
    format!(
        "[Script Info]\n\
         Title: autoseo-clip\n\
         ScriptType: v4.00+\n\
         PlayResX: {w}\n\
         PlayResY: {h}\n\
         WrapStyle: 2\n\
         ScaledBorderAndShadow: yes\n\
         \n"
    )
}

fn style_block(style: &CaptionStyle) -> String {
    // ASS color literal: &HAABBGGRR — alpha is the first byte, 00 = opaque.
    let primary = format!("&H00{:06X}", reverse_bytes_24(style.primary_bgr));
    // In karaoke mode SecondaryColour is the "before highlight" color (dimmed text);
    // PrimaryColour is the "after highlight" (lit-up) color. We swap roles:
    // the highlight color becomes Primary so words glow as {\k} sweeps through,
    // and the original text color stays as Secondary (the pre-highlight state).
    let (primary_col, secondary_col) = if let Some(hl) = style.highlight_bgr {
        let highlight = format!("&H00{:06X}", reverse_bytes_24(hl));
        // ASS \k fills from SecondaryColour → PrimaryColour over the duration.
        // So PrimaryColour = highlight (lit), SecondaryColour = normal (unlit).
        (highlight, primary.clone())
    } else {
        (primary.clone(), primary.clone())
    };
    let outline = format!("&H00{:06X}", reverse_bytes_24(style.outline_bgr));
    let back = "&H80000000".to_string(); // semi-transparent black shadow
    format!(
        "[V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Default,{font},{size},{primary_col},{secondary_col},{outline},{back},-1,0,0,0,100,100,0,0,1,{out_px},0,2,40,40,{margin_v},1\n\
         \n",
        font = style.font_name,
        size = style.font_size_pt,
        primary_col = primary_col,
        secondary_col = secondary_col,
        outline = outline,
        back = back,
        out_px = style.outline_px,
        margin_v = style.margin_v_px
    )
}

/// libass color literal is `&HAABBGGRR`. Caller stores BGR as a u32 already;
/// we just hex-format it. This indirection exists so the public field name reads
/// naturally even though ASS storage is BGR.
fn reverse_bytes_24(bgr: u32) -> u32 {
    bgr & 0xFFFFFF
}

fn format_ass_time(t_secs: f64) -> String {
    let t = t_secs.max(0.0);
    let total_cs = (t * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_s = total_cs / 100;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

fn sanitize_text(s: &str) -> String {
    // ASS dialogue uses `\N` for hard line break and treats `{` `}` specially.
    s.replace('\\', "\\\\")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, start: f64, end: f64) -> AlignedWord {
        AlignedWord {
            text: text.into(),
            start_secs: start,
            end_secs: end,
        }
    }

    #[test]
    fn format_time_basic() {
        assert_eq!(format_ass_time(0.0), "0:00:00.00");
        assert_eq!(format_ass_time(1.5), "0:00:01.50");
        assert_eq!(format_ass_time(61.25), "0:01:01.25");
        assert_eq!(format_ass_time(3661.99), "1:01:01.99");
    }

    #[test]
    fn group_phrases_respects_word_limit() {
        let words = vec![
            w("one", 0.0, 0.2),
            w("two", 0.2, 0.4),
            w("three", 0.4, 0.6),
            w("four", 0.6, 0.8),
            w("five", 0.8, 1.0),
            w("six", 1.0, 1.2),
            w("seven", 1.2, 1.4),
        ];
        let phrases = group_into_phrases(&words, 3, 100);
        // 7 words / max 3 per phrase → ceiling(7/3) = 3 phrases.
        assert_eq!(phrases.len(), 3, "got: {:?}", phrases);
        assert_eq!(phrases[0].text, "one two three");
        assert_eq!(phrases[1].text, "four five six");
        assert_eq!(phrases[2].text, "seven");
    }

    #[test]
    fn group_phrases_flushes_on_terminal_punctuation() {
        let words = vec![
            w("Hello", 0.0, 0.3),
            w("world.", 0.3, 0.6),
            w("How", 0.7, 0.9),
            w("are", 0.9, 1.0),
            w("you?", 1.0, 1.3),
            w("Fine.", 1.4, 1.7),
        ];
        let phrases = group_into_phrases(&words, 10, 200);
        assert_eq!(phrases.len(), 3, "got: {:?}", phrases);
        assert_eq!(phrases[0].text, "Hello world.");
        assert_eq!(phrases[1].text, "How are you?");
        assert_eq!(phrases[2].text, "Fine.");
    }

    #[test]
    fn group_phrases_respects_char_limit() {
        let words = vec![
            w("supercalifragilistic", 0.0, 0.5),
            w("expialidocious", 0.5, 1.0),
            w("word", 1.0, 1.2),
        ];
        // Char limit 25 — first long word fills on its own; second should flush.
        let phrases = group_into_phrases(&words, 10, 25);
        assert!(phrases.len() >= 2);
        assert!(phrases[0].text.chars().count() <= 25 + 1);
    }

    #[test]
    fn group_phrases_handles_empty_and_whitespace_only() {
        let phrases = group_into_phrases(&[], 4, 28);
        assert!(phrases.is_empty());
        let words = vec![w("  ", 0.0, 0.5), w("", 0.5, 1.0)];
        let phrases = group_into_phrases(&words, 4, 28);
        assert!(phrases.is_empty());
    }

    #[test]
    fn render_emits_valid_ass_structure() {
        let words = vec![
            w("Hello", 1.0, 1.3),
            w("world", 1.3, 1.6),
            w("today", 1.6, 1.9),
        ];
        let style = CaptionStyle::default();
        let body = render_ass(&words, 1080, 1920, &style);

        assert!(body.contains("[Script Info]"));
        assert!(body.contains("PlayResX: 1080"));
        assert!(body.contains("PlayResY: 1920"));
        assert!(body.contains("[V4+ Styles]"));
        assert!(body.contains("Style: Default,Montserrat,80"));
        assert!(body.contains("[Events]"));
        assert!(body.contains("Dialogue: "));
        // Time should be in H:MM:SS.CS shape.
        assert!(body.contains("0:00:01.00"));
        assert!(body.contains("0:00:01.90"));
    }

    #[test]
    fn render_with_no_words_produces_header_only() {
        let body = render_ass(&[], 1080, 1920, &CaptionStyle::default());
        assert!(body.contains("[Events]"));
        assert!(!body.contains("Dialogue: "));
    }

    #[test]
    fn sanitize_escapes_special_chars() {
        assert_eq!(sanitize_text("plain"), "plain");
        assert_eq!(sanitize_text("a{b}c"), r"a\{b\}c");
        assert_eq!(sanitize_text("line1\nline2"), "line1 line2");
        assert_eq!(sanitize_text(r"back\slash"), r"back\\slash");
    }

    #[tokio::test]
    async fn write_ass_creates_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("nested/clip.ass");
        let words = vec![w("test", 0.0, 0.5)];
        write_ass(&path, &words, 1080, 1920, &CaptionStyle::default()).await?;
        let body = tokio::fs::read_to_string(&path).await?;
        assert!(body.contains("Dialogue: "));
        Ok(())
    }

    #[test]
    fn overlay_ass_emits_single_event_with_fade() {
        let body = render_overlay_ass("WAIT FOR IT", 1080, 1920, &OverlayStyle::for_vertical());
        assert!(body.contains("[Script Info]"));
        assert!(body.contains("Style: Overlay,Montserrat,130"));
        // Alignment 5 = middle-center.
        assert!(body.contains(",5,"));
        // Single Dialogue with fade tag.
        let dialogue_count = body.matches("Dialogue: ").count();
        assert_eq!(dialogue_count, 1);
        assert!(body.contains("WAIT FOR IT"));
        assert!(body.contains("\\fade(300,300)"));
        // Duration 1.5s → end at 0:00:01.50.
        assert!(body.contains("0:00:01.50"));
    }

    #[test]
    fn overlay_with_empty_hook_produces_no_dialogue() {
        let body = render_overlay_ass("   ", 1080, 1920, &OverlayStyle::for_vertical());
        assert!(body.contains("[Events]"));
        assert!(!body.contains("Dialogue: "));
    }

    #[test]
    fn overlay_aspect_specific_sizes_differ() {
        assert_eq!(OverlayStyle::for_vertical().font_size_pt, 130);
        assert_eq!(OverlayStyle::for_square().font_size_pt, 100);
        assert_eq!(OverlayStyle::for_landscape().font_size_pt, 80);
    }

    #[test]
    fn overlay_sanitizes_braces_in_hook() {
        let body = render_overlay_ass("{tricky}", 1080, 1920, &OverlayStyle::for_vertical());
        assert!(body.contains(r"\{tricky\}"));
    }

    // ── Karaoke tests ──────────────────────────────────────────────────

    #[test]
    fn karaoke_emits_k_tags_per_word() {
        let words = vec![
            w("Hello", 0.0, 0.3),
            w("world", 0.3, 0.7),
            w("today", 0.7, 1.0),
        ];
        let style = CaptionStyle::default(); // highlight_bgr = Some(...)
        let body = render_ass(&words, 1080, 1920, &style);

        // Each word should have a {\kN} tag.
        assert!(body.contains("{\\k30}Hello"), "body: {body}");
        assert!(body.contains("{\\k40}world"), "body: {body}");
        assert!(body.contains("{\\k30}today"), "body: {body}");
    }

    #[test]
    fn karaoke_disabled_when_highlight_none() {
        let words = vec![
            w("Hello", 0.0, 0.3),
            w("world", 0.3, 0.7),
        ];
        let mut style = CaptionStyle::default();
        style.highlight_bgr = None;
        let body = render_ass(&words, 1080, 1920, &style);

        // No karaoke tags — plain text.
        assert!(!body.contains("{\\k"), "body: {body}");
        assert!(body.contains("Hello world"), "body: {body}");
    }

    #[test]
    fn karaoke_style_block_has_highlight_primary() {
        let style = CaptionStyle::default();
        let block = style_block(&style);
        // highlight_bgr = 0x00FFFF → reversed = 0x00FFFF → &H0000FFFF
        assert!(block.contains("&H0000FFFF"), "block: {block}");
    }

    #[test]
    fn karaoke_preserves_phrase_grouping() {
        let words = vec![
            w("one", 0.0, 0.2),
            w("two", 0.2, 0.4),
            w("three", 0.4, 0.6),
            w("four", 0.6, 0.8),
            w("five", 0.8, 1.0),
        ];
        let mut style = CaptionStyle::default();
        style.max_words_per_phrase = 3;
        style.max_chars_per_phrase = 100;
        let body = render_ass(&words, 1080, 1920, &style);

        // Should have 2 Dialogue lines (3 + 2 words).
        let dialogue_count = body.matches("Dialogue: ").count();
        assert_eq!(dialogue_count, 2, "body: {body}");

        // First phrase: words with {\k} tags.
        assert!(body.contains("{\\k20}one {\\k20}two {\\k20}three"), "body: {body}");
        // Second phrase.
        assert!(body.contains("{\\k20}four {\\k20}five"), "body: {body}");
    }

    #[test]
    fn karaoke_word_timing_centisecond_rounding() {
        let words = vec![
            w("fast", 0.0, 0.05),   // 5cs
            w("slow", 0.05, 1.555), // ~150cs
        ];
        let body = render_ass(&words, 1080, 1920, &CaptionStyle::default());
        assert!(body.contains("{\\k5}fast"), "body: {body}");
        assert!(body.contains("{\\k150}slow") || body.contains("{\\k151}slow"), "body: {body}");
    }

    #[test]
    fn build_karaoke_text_handles_single_word() {
        let words = vec![PhraseWord {
            text: "alone".into(),
            start_secs: 0.0,
            end_secs: 0.5,
        }];
        let text = build_karaoke_text(&words);
        assert_eq!(text, "{\\k50}alone");
    }
}
