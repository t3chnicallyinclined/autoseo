//! Generate an Advanced SubStation Alpha (`.ass`) subtitle file from per-word
//! transcript timestamps. ffmpeg burns the result via `subtitles='path.ass'` —
//! see [`crate::render::render_clip`].
//!
//! M1 style: "popping" phrase captions — 2–4 words per phrase, replaced when the
//! next phrase starts. No active-word color highlight; uniform white text with a
//! bold black outline at the bottom-center. Karaoke `{\k}` tag highlighting is
//! a future polish, not required to ship M1.

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
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            font_name: "Montserrat".into(),
            font_size_pt: 80,
            primary_bgr: 0xFFFFFF, // white
            outline_bgr: 0x000000, // black
            outline_px: 4,
            margin_v_px: 200,
            max_words_per_phrase: 4,
            max_chars_per_phrase: 28,
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

/// Pure function for testability: produce the full `.ass` text.
pub fn render_ass(
    words: &[AlignedWord],
    play_res_w: u32,
    play_res_h: u32,
    style: &CaptionStyle,
) -> String {
    let phrases = group_into_phrases(words, style.max_words_per_phrase, style.max_chars_per_phrase);

    let mut out = String::new();
    out.push_str(&header(play_res_w, play_res_h));
    out.push_str(&style_block(style));
    out.push_str("[Events]\n");
    out.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );
    for p in phrases {
        let start = format_ass_time(p.start_secs);
        let end = format_ass_time(p.end_secs);
        let text = sanitize_text(&p.text);
        out.push_str(&format!(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
        ));
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
struct Phrase {
    start_secs: f64,
    end_secs: f64,
    text: String,
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
        let text = buf
            .iter()
            .map(|w| w.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if end > start && !text.is_empty() {
            out.push(Phrase {
                start_secs: start,
                end_secs: end,
                text,
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
    // Secondary is unused for the "popping" style; reuse primary.
    let secondary = primary.clone();
    let outline = format!("&H00{:06X}", reverse_bytes_24(style.outline_bgr));
    let back = "&H80000000".to_string(); // semi-transparent black shadow
    format!(
        "[V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Default,{font},{size},{primary},{secondary},{outline},{back},-1,0,0,0,100,100,0,0,1,{out_px},0,2,40,40,{margin_v},1\n\
         \n",
        font = style.font_name,
        size = style.font_size_pt,
        primary = primary,
        secondary = secondary,
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
}
