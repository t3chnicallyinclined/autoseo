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
use serde::Deserialize;
use std::path::Path;

use crate::align::AlignedWord;

/// Per-aspect, per-show overrides applied on top of the built-in caption /
/// overlay defaults. Every field is optional — when `None`, the aspect's
/// hardcoded value wins, so loading an empty `CaptionOverrides` keeps the
/// existing look.
///
/// JSON shape (used by per-show `prompts/shows/{slug}/captions.json`):
/// ```json
/// {
///   "font_name": "Inter",
///   "highlight_bgr": "FF00FF",
///   "primary_bgr": "FFFFFF",
///   "outline_bgr": "000000",
///   "disable_karaoke": false,
///   "vertical": {
///     "font_size_pt": 72,
///     "outline_px": 3,
///     "margin_v_px": 220,
///     "max_words_per_phrase": 3,
///     "max_chars_per_phrase": 24,
///     "overlay_font_size_pt": 110
///   },
///   "square":    { "font_size_pt": 56 },
///   "landscape": { "font_size_pt": 48 }
/// }
/// ```
/// Colors are 0xBBGGRR hex strings (ASS storage order); leading `0x`/`#` and
/// upper/lowercase both accepted.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptionOverrides {
    pub font_name: Option<String>,
    #[serde(deserialize_with = "de_opt_bgr", default)]
    pub highlight_bgr: Option<u32>,
    #[serde(deserialize_with = "de_opt_bgr", default)]
    pub primary_bgr: Option<u32>,
    #[serde(deserialize_with = "de_opt_bgr", default)]
    pub outline_bgr: Option<u32>,
    pub disable_karaoke: Option<bool>,
    pub vertical: Option<PerAspectOverride>,
    pub square: Option<PerAspectOverride>,
    pub landscape: Option<PerAspectOverride>,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerAspectOverride {
    pub font_size_pt: Option<u32>,
    pub outline_px: Option<u32>,
    pub margin_v_px: Option<u32>,
    pub max_words_per_phrase: Option<usize>,
    pub max_chars_per_phrase: Option<usize>,
    /// Override the 1.5s hook-overlay font size for this aspect.
    pub overlay_font_size_pt: Option<u32>,
}

impl CaptionOverrides {
    /// Apply selected env vars on top of the receiver. Env wins over
    /// in-struct values so the env can be used as a hard global override.
    /// Recognized env keys (all optional):
    /// - `CAPTION_FONT_NAME`
    /// - `CAPTION_HIGHLIGHT_BGR` / `CAPTION_PRIMARY_BGR` / `CAPTION_OUTLINE_BGR` (hex)
    /// - `CAPTION_DISABLE_KARAOKE` (`true`/`false`)
    pub fn apply_env(&mut self) {
        if let Ok(v) = std::env::var("CAPTION_FONT_NAME") {
            if !v.trim().is_empty() {
                self.font_name = Some(v);
            }
        }
        if let Ok(v) = std::env::var("CAPTION_HIGHLIGHT_BGR") {
            if let Some(c) = parse_bgr_hex(&v) {
                self.highlight_bgr = Some(c);
            }
        }
        if let Ok(v) = std::env::var("CAPTION_PRIMARY_BGR") {
            if let Some(c) = parse_bgr_hex(&v) {
                self.primary_bgr = Some(c);
            }
        }
        if let Ok(v) = std::env::var("CAPTION_OUTLINE_BGR") {
            if let Some(c) = parse_bgr_hex(&v) {
                self.outline_bgr = Some(c);
            }
        }
        if let Ok(v) = std::env::var("CAPTION_DISABLE_KARAOKE") {
            match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => self.disable_karaoke = Some(true),
                "0" | "false" | "no" | "off" => self.disable_karaoke = Some(false),
                _ => {}
            }
        }
    }

    /// Load `${shows_dir}/{show_slug}/captions.json` if present and merge into
    /// `self`. Per-show JSON wins over the in-struct base; the caller usually
    /// runs `apply_env()` *after* this if env should be the hard override, or
    /// before if per-show should win. Pipeline default = env first, then
    /// per-show (see `load_for_show`).
    pub async fn merge_show_file(
        &mut self,
        shows_dir: &Path,
        show_slug: &str,
    ) -> anyhow::Result<bool> {
        let path = shows_dir.join(show_slug).join("captions.json");
        if tokio::fs::metadata(&path).await.is_err() {
            return Ok(false);
        }
        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("read {}", path.display()))?;
        let parsed: CaptionOverrides = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse {}", path.display()))?;
        self.merge_from(parsed);
        Ok(true)
    }

    /// In-place merge: any `Some(...)` in `other` overwrites this.
    pub fn merge_from(&mut self, other: CaptionOverrides) {
        if other.font_name.is_some() {
            self.font_name = other.font_name;
        }
        if other.highlight_bgr.is_some() {
            self.highlight_bgr = other.highlight_bgr;
        }
        if other.primary_bgr.is_some() {
            self.primary_bgr = other.primary_bgr;
        }
        if other.outline_bgr.is_some() {
            self.outline_bgr = other.outline_bgr;
        }
        if other.disable_karaoke.is_some() {
            self.disable_karaoke = other.disable_karaoke;
        }
        // Per-aspect overrides merge field-by-field so a show JSON can change
        // just `font_size_pt` without wiping the other fields the env set.
        merge_per_aspect(&mut self.vertical, other.vertical);
        merge_per_aspect(&mut self.square, other.square);
        merge_per_aspect(&mut self.landscape, other.landscape);
    }

    /// Convenience: build the final overrides for a render by stacking
    /// env → per-show JSON. `show_slug = None` skips the per-show step.
    pub async fn load_for_show(shows_dir: &Path, show_slug: Option<&str>) -> Self {
        let mut o = Self::default();
        o.apply_env();
        if let Some(slug) = show_slug.filter(|s| !s.is_empty()) {
            if let Err(e) = o.merge_show_file(shows_dir, slug).await {
                tracing::warn!(error = ?e, show_slug = slug, "captions: per-show overrides failed to load");
            }
        }
        o
    }
}

fn merge_per_aspect(base: &mut Option<PerAspectOverride>, incoming: Option<PerAspectOverride>) {
    let Some(inc) = incoming else { return };
    let target = base.get_or_insert_with(PerAspectOverride::default);
    if inc.font_size_pt.is_some() {
        target.font_size_pt = inc.font_size_pt;
    }
    if inc.outline_px.is_some() {
        target.outline_px = inc.outline_px;
    }
    if inc.margin_v_px.is_some() {
        target.margin_v_px = inc.margin_v_px;
    }
    if inc.max_words_per_phrase.is_some() {
        target.max_words_per_phrase = inc.max_words_per_phrase;
    }
    if inc.max_chars_per_phrase.is_some() {
        target.max_chars_per_phrase = inc.max_chars_per_phrase;
    }
    if inc.overlay_font_size_pt.is_some() {
        target.overlay_font_size_pt = inc.overlay_font_size_pt;
    }
}

fn parse_bgr_hex(s: &str) -> Option<u32> {
    let trimmed = s.trim().trim_start_matches('#').trim_start_matches("0x");
    if trimmed.is_empty() || trimmed.len() > 6 {
        return None;
    }
    u32::from_str_radix(trimmed, 16).ok()
}

fn de_opt_bgr<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    let opt = Option::<String>::deserialize(deserializer)?;
    match opt {
        None => Ok(None),
        Some(s) => parse_bgr_hex(&s)
            .map(Some)
            .ok_or_else(|| D::Error::custom(format!("invalid BGR hex: {s:?}"))),
    }
}

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
        Self::for_vertical_with(&CaptionOverrides::default())
    }

    /// Caption style tuned for 1:1 square (1080×1080). Smaller font, tighter margin.
    pub fn for_square() -> Self {
        Self::for_square_with(&CaptionOverrides::default())
    }

    /// Caption style tuned for 16:9 landscape (1920×1080). Smaller font (more
    /// horizontal real estate, longer phrases possible).
    pub fn for_landscape() -> Self {
        Self::for_landscape_with(&CaptionOverrides::default())
    }

    /// Apply `overrides` on top of the vertical defaults.
    pub fn for_vertical_with(overrides: &CaptionOverrides) -> Self {
        let mut s = Self {
            font_name: "Montserrat".into(),
            font_size_pt: 80,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 4,
            margin_v_px: 200,
            max_words_per_phrase: 4,
            max_chars_per_phrase: 28,
            highlight_bgr: Some(0x00FFFF), // yellow in BGR
        };
        apply_overrides(&mut s, overrides, overrides.vertical.as_ref());
        s
    }

    /// Apply `overrides` on top of the 1:1 defaults.
    pub fn for_square_with(overrides: &CaptionOverrides) -> Self {
        let mut s = Self {
            font_name: "Montserrat".into(),
            font_size_pt: 60,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 3,
            margin_v_px: 80,
            max_words_per_phrase: 5,
            max_chars_per_phrase: 36,
            highlight_bgr: Some(0x00FFFF),
        };
        apply_overrides(&mut s, overrides, overrides.square.as_ref());
        s
    }

    /// Apply `overrides` on top of the 16:9 defaults.
    pub fn for_landscape_with(overrides: &CaptionOverrides) -> Self {
        let mut s = Self {
            font_name: "Montserrat".into(),
            font_size_pt: 50,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 3,
            margin_v_px: 60,
            max_words_per_phrase: 7,
            max_chars_per_phrase: 52,
            highlight_bgr: Some(0x00FFFF),
        };
        apply_overrides(&mut s, overrides, overrides.landscape.as_ref());
        s
    }
}

fn apply_overrides(
    style: &mut CaptionStyle,
    global: &CaptionOverrides,
    aspect: Option<&PerAspectOverride>,
) {
    if let Some(ref f) = global.font_name {
        if !f.trim().is_empty() {
            style.font_name = f.clone();
        }
    }
    if let Some(c) = global.primary_bgr {
        style.primary_bgr = c;
    }
    if let Some(c) = global.outline_bgr {
        style.outline_bgr = c;
    }
    if let Some(c) = global.highlight_bgr {
        style.highlight_bgr = Some(c);
    }
    if global.disable_karaoke == Some(true) {
        style.highlight_bgr = None;
    }
    if let Some(a) = aspect {
        if let Some(v) = a.font_size_pt {
            style.font_size_pt = v;
        }
        if let Some(v) = a.outline_px {
            style.outline_px = v;
        }
        if let Some(v) = a.margin_v_px {
            style.margin_v_px = v;
        }
        if let Some(v) = a.max_words_per_phrase {
            style.max_words_per_phrase = v;
        }
        if let Some(v) = a.max_chars_per_phrase {
            style.max_chars_per_phrase = v;
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
        Self::for_vertical_with(&CaptionOverrides::default())
    }

    pub fn for_square() -> Self {
        Self::for_square_with(&CaptionOverrides::default())
    }

    pub fn for_landscape() -> Self {
        Self::for_landscape_with(&CaptionOverrides::default())
    }

    pub fn for_vertical_with(overrides: &CaptionOverrides) -> Self {
        let mut s = Self {
            font_name: "Montserrat".into(),
            font_size_pt: 130,
            primary_bgr: 0xFFFFFF,
            outline_bgr: 0x000000,
            outline_px: 5,
            margin_v_px: 0,
            duration_secs: 1.5,
            fade_secs: 0.3,
        };
        apply_overlay_overrides(&mut s, overrides, overrides.vertical.as_ref());
        s
    }

    pub fn for_square_with(overrides: &CaptionOverrides) -> Self {
        let mut s = Self {
            font_size_pt: 100,
            ..Self::for_vertical_with(&CaptionOverrides::default())
        };
        apply_overlay_overrides(&mut s, overrides, overrides.square.as_ref());
        s
    }

    pub fn for_landscape_with(overrides: &CaptionOverrides) -> Self {
        let mut s = Self {
            font_size_pt: 80,
            ..Self::for_vertical_with(&CaptionOverrides::default())
        };
        apply_overlay_overrides(&mut s, overrides, overrides.landscape.as_ref());
        s
    }
}

fn apply_overlay_overrides(
    style: &mut OverlayStyle,
    global: &CaptionOverrides,
    aspect: Option<&PerAspectOverride>,
) {
    if let Some(ref f) = global.font_name {
        if !f.trim().is_empty() {
            style.font_name = f.clone();
        }
    }
    if let Some(c) = global.primary_bgr {
        style.primary_bgr = c;
    }
    if let Some(c) = global.outline_bgr {
        style.outline_bgr = c;
    }
    if let Some(a) = aspect {
        if let Some(v) = a.overlay_font_size_pt {
            style.font_size_pt = v;
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
    let phrases = group_into_phrases(
        words,
        style.max_words_per_phrase,
        style.max_chars_per_phrase,
    );
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
        self.words
            .iter()
            .any(|w| (w.start_secs - first_start).abs() > 0.001)
    }
}

fn group_into_phrases(words: &[AlignedWord], max_words: usize, max_chars: usize) -> Vec<Phrase> {
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
        let words = vec![w("Hello", 0.0, 0.3), w("world", 0.3, 0.7)];
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
        assert!(
            body.contains("{\\k20}one {\\k20}two {\\k20}three"),
            "body: {body}"
        );
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
        assert!(
            body.contains("{\\k150}slow") || body.contains("{\\k151}slow"),
            "body: {body}"
        );
    }

    // ── Override / per-show config tests ────────────────────────────────

    #[test]
    fn overrides_default_is_noop() {
        let s = CaptionStyle::for_vertical_with(&CaptionOverrides::default());
        let ref_s = CaptionStyle::for_vertical();
        assert_eq!(s.font_name, ref_s.font_name);
        assert_eq!(s.font_size_pt, ref_s.font_size_pt);
        assert_eq!(s.highlight_bgr, ref_s.highlight_bgr);
        assert_eq!(s.margin_v_px, ref_s.margin_v_px);
    }

    #[test]
    fn overrides_change_color_and_font() {
        let mut o = CaptionOverrides::default();
        o.font_name = Some("Inter".into());
        o.highlight_bgr = Some(0xFF00FF);
        o.primary_bgr = Some(0xCCCCCC);
        let s = CaptionStyle::for_vertical_with(&o);
        assert_eq!(s.font_name, "Inter");
        assert_eq!(s.highlight_bgr, Some(0xFF00FF));
        assert_eq!(s.primary_bgr, 0xCCCCCC);
    }

    #[test]
    fn overrides_disable_karaoke_clears_highlight() {
        let mut o = CaptionOverrides::default();
        o.disable_karaoke = Some(true);
        let s = CaptionStyle::for_vertical_with(&o);
        assert!(s.highlight_bgr.is_none());
    }

    #[test]
    fn per_aspect_override_only_touches_matching_aspect() {
        let mut o = CaptionOverrides::default();
        o.vertical = Some(PerAspectOverride {
            font_size_pt: Some(72),
            margin_v_px: Some(180),
            ..Default::default()
        });
        let v = CaptionStyle::for_vertical_with(&o);
        let sq = CaptionStyle::for_square_with(&o);
        assert_eq!(v.font_size_pt, 72);
        assert_eq!(v.margin_v_px, 180);
        // Square aspect should be untouched.
        assert_eq!(sq.font_size_pt, 60);
        assert_eq!(sq.margin_v_px, 80);
    }

    #[test]
    fn overlay_overrides_pick_up_aspect_font() {
        let mut o = CaptionOverrides::default();
        o.vertical = Some(PerAspectOverride {
            overlay_font_size_pt: Some(110),
            ..Default::default()
        });
        let v = OverlayStyle::for_vertical_with(&o);
        let sq = OverlayStyle::for_square_with(&o);
        assert_eq!(v.font_size_pt, 110);
        // Square aspect override unset → default 100.
        assert_eq!(sq.font_size_pt, 100);
    }

    #[test]
    fn parse_bgr_hex_accepts_common_forms() {
        assert_eq!(parse_bgr_hex("FF00FF"), Some(0xFF00FF));
        assert_eq!(parse_bgr_hex("#ff00ff"), Some(0xFF00FF));
        assert_eq!(parse_bgr_hex("0xFf00FF"), Some(0xFF00FF));
        assert_eq!(parse_bgr_hex("123"), Some(0x123));
        assert!(parse_bgr_hex("").is_none());
        assert!(parse_bgr_hex("ZZZZZZ").is_none());
        assert!(parse_bgr_hex("1234567").is_none());
    }

    #[test]
    fn json_overrides_parse_partial() {
        let json = r#"{
            "font_name": "Inter",
            "highlight_bgr": "FF00FF",
            "vertical": { "font_size_pt": 72, "margin_v_px": 180 }
        }"#;
        let o: CaptionOverrides = serde_json::from_str(json).unwrap();
        assert_eq!(o.font_name.as_deref(), Some("Inter"));
        assert_eq!(o.highlight_bgr, Some(0xFF00FF));
        assert!(o.square.is_none());
        let v = o.vertical.unwrap();
        assert_eq!(v.font_size_pt, Some(72));
        assert_eq!(v.margin_v_px, Some(180));
        assert!(v.outline_px.is_none());
    }

    #[test]
    fn merge_from_overwrites_some_keeps_none() {
        let mut base = CaptionOverrides {
            font_name: Some("Montserrat".into()),
            primary_bgr: Some(0xFFFFFF),
            ..Default::default()
        };
        let incoming = CaptionOverrides {
            font_name: Some("Inter".into()),
            highlight_bgr: Some(0x00FF00),
            ..Default::default()
        };
        base.merge_from(incoming);
        assert_eq!(base.font_name.as_deref(), Some("Inter"));
        assert_eq!(base.highlight_bgr, Some(0x00FF00));
        assert_eq!(base.primary_bgr, Some(0xFFFFFF));
    }

    #[tokio::test]
    async fn show_file_overrides_layer_onto_env() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let shows = dir.path();
        let slug = "the-show";
        let slug_dir = shows.join(slug);
        tokio::fs::create_dir_all(&slug_dir).await?;
        tokio::fs::write(
            slug_dir.join("captions.json"),
            r#"{
                "font_name": "Bricolage",
                "vertical": { "font_size_pt": 64 }
            }"#,
        )
        .await?;

        let mut o = CaptionOverrides::default();
        // Pretend env set the primary color (we set it directly to avoid touching
        // real process env in tests).
        o.primary_bgr = Some(0xEEEEEE);
        let loaded = o.merge_show_file(shows, slug).await?;
        assert!(loaded);
        assert_eq!(o.font_name.as_deref(), Some("Bricolage"));
        assert_eq!(o.primary_bgr, Some(0xEEEEEE)); // env survived
        assert_eq!(o.vertical.as_ref().unwrap().font_size_pt, Some(64));
        Ok(())
    }

    #[tokio::test]
    async fn show_file_missing_returns_false_and_leaves_base() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut o = CaptionOverrides::default();
        o.font_name = Some("Inter".into());
        let loaded = o.merge_show_file(dir.path(), "nope").await?;
        assert!(!loaded);
        assert_eq!(o.font_name.as_deref(), Some("Inter"));
        Ok(())
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
