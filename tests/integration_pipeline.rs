//! Integration tests that exercise the full autoseo pipeline with synthetic
//! video/audio generated via ffmpeg. No real API keys are needed — transcription
//! and ranking are tested via their pure/sync sub-components (candidate generation,
//! caption rendering, etc.), while ffmpeg-dependent tests skip gracefully when the
//! tool is unavailable.
//!
//! Run with: `cargo test --test integration_pipeline`

use std::path::Path;
use tokio::process::Command;

use autoseo::align::AlignedWord;
use autoseo::candidates::CandidateGenerator;
use autoseo::captions::{
    CaptionStyle, OverlayStyle, render_ass, render_overlay_ass, write_ass, write_overlay_ass,
};
use autoseo::media;
use autoseo::prosody::{self, RmsWindow};
use autoseo::render::{RenderProfile, render_clip};
use autoseo::scene;
use autoseo::vad::{self, SilenceWindow};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn tool_ok(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Generate a 30-second synthetic video with a 1920×1080 testsrc pattern and a
/// 1 kHz sine wave tone. Returns the path to the generated file inside `dir`.
async fn generate_synthetic_video(dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let src = dir.join("synthetic_30s.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=30"])
        .args(["-f", "lavfi", "-i", "sine=frequency=1000:duration=30"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
        .arg(&src)
        .status()
        .await?;
    anyhow::ensure!(status.success(), "synthetic video generation failed");
    Ok(src)
}

/// Generate a synthetic audio file with alternating silence/tone patterns.
/// Pattern: 2s silence + 3s 440Hz tone + 2s silence + 3s 880Hz tone = 10s.
async fn generate_speech_like_audio(dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let wav = dir.join("speech_like.wav");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=2"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:r=16000:d=3"])
        .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=2"])
        .args(["-f", "lavfi", "-i", "sine=frequency=880:r=16000:d=3"])
        .args([
            "-filter_complex",
            "[0][1][2][3]concat=n=4:v=0:a=1[a]",
        ])
        .args(["-map", "[a]"])
        .arg(&wav)
        .status()
        .await?;
    anyhow::ensure!(status.success(), "speech-like audio generation failed");
    Ok(wav)
}

/// Generate a short 6s video with 3 distinct colored segments for shot detection.
async fn generate_multi_scene_video(dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let video = dir.join("multi_scene.mp4");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "color=red:size=1920x1080:duration=2:rate=30"])
        .args(["-f", "lavfi", "-i", "color=green:size=1920x1080:duration=2:rate=30"])
        .args(["-f", "lavfi", "-i", "color=blue:size=1920x1080:duration=2:rate=30"])
        .args(["-f", "lavfi", "-i", "sine=frequency=1000:duration=6"])
        .args([
            "-filter_complex",
            "[0][1][2]concat=n=3:v=1:a=0[v]",
        ])
        .args(["-map", "[v]", "-map", "3:a"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
        .arg(&video)
        .status()
        .await?;
    anyhow::ensure!(status.success(), "multi-scene video generation failed");
    Ok(video)
}

/// Helper to probe video dimensions via ffprobe.
async fn probe_dimensions(path: &Path) -> anyhow::Result<(u32, u32)> {
    let output = Command::new("ffprobe")
        .args(["-v", "error"])
        .args([
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
        ])
        .arg(path)
        .output()
        .await?;
    let dims = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let parts: Vec<&str> = dims.split('x').collect();
    anyhow::ensure!(parts.len() == 2, "unexpected probe output: {dims}");
    Ok((parts[0].parse()?, parts[1].parse()?))
}

/// Create dense synthetic word timestamps for candidate generation testing.
fn dense_words(total_duration: f64, words_per_sec: f64) -> Vec<AlignedWord> {
    let n = (total_duration * words_per_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / words_per_sec;
            AlignedWord {
                text: format!("word{i}"),
                start_secs: t,
                end_secs: t + (1.0 / words_per_sec) * 0.8,
            }
        })
        .collect()
}

// ===========================================================================
// Test: Full render pipeline across all three aspect ratios
// ===========================================================================

#[tokio::test]
async fn render_all_three_formats_from_synthetic_video() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let src = generate_synthetic_video(dir.path()).await?;

    let profiles = [
        ("vertical", RenderProfile::shorts_vertical(), 1080u32, 1920u32),
        ("square", RenderProfile::linkedin_square(), 1080, 1080),
        ("landscape", RenderProfile::bluesky_landscape(), 1920, 1080),
    ];

    for (label, profile, expected_w, expected_h) in &profiles {
        let out = dir.path().join(format!("clip_{label}.mp4"));
        render_clip("ffmpeg", &src, 2.0, 8.0, &out, profile, &[]).await?;

        let meta = tokio::fs::metadata(&out).await?;
        assert!(meta.len() > 0, "{label}: output file is empty");

        let (w, h) = probe_dimensions(&out).await?;
        assert_eq!(w, *expected_w, "{label}: width mismatch");
        assert_eq!(h, *expected_h, "{label}: height mismatch");
    }
    Ok(())
}

// ===========================================================================
// Test: Audio extraction from synthetic video
// ===========================================================================

#[tokio::test]
async fn extract_audio_and_probe_duration() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let src = generate_synthetic_video(dir.path()).await?;

    let audio = dir.path().join("audio.m4a");
    media::extract_audio_m4a("ffmpeg", &src, &audio).await?;

    let meta = tokio::fs::metadata(&audio).await?;
    assert!(meta.len() > 0, "extracted audio is empty");

    let dur = media::duration_secs("ffprobe", &audio).await?;
    // Should be close to 30s (within 1s tolerance for codec padding).
    assert!(
        (dur - 30.0).abs() < 1.5,
        "expected ~30s duration, got {dur}"
    );
    Ok(())
}

// ===========================================================================
// Test: Audio segmentation produces correct chunks
// ===========================================================================

#[tokio::test]
async fn segment_audio_produces_multiple_chunks() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let src = generate_synthetic_video(dir.path()).await?;

    let audio = dir.path().join("audio.m4a");
    media::extract_audio_m4a("ffmpeg", &src, &audio).await?;

    let chunks_dir = dir.path().join("chunks");
    let chunks = media::segment_audio("ffmpeg", &audio, &chunks_dir, 10).await?;

    // 30s / 10s per chunk = 3 chunks (possibly 4 with codec overhead).
    assert!(
        chunks.len() >= 3,
        "expected >= 3 chunks from 30s/10s segmentation, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        let m = tokio::fs::metadata(chunk).await?;
        assert!(m.len() > 0, "chunk {:?} is empty", chunk.file_name());
    }
    Ok(())
}

// ===========================================================================
// Test: Silence detection on synthetic audio
// ===========================================================================

#[tokio::test]
async fn silence_detection_on_speech_like_audio() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await {
        eprintln!("skipping: ffmpeg not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let wav = generate_speech_like_audio(dir.path()).await?;

    let silences = vad::detect_silences("ffmpeg", &wav, -30.0, 0.5).await?;
    assert!(
        !silences.is_empty(),
        "expected silence windows in speech-like audio"
    );

    // Verify silences are in the expected regions (~0-2s and ~5-7s).
    let has_leading_silence = silences.iter().any(|s| s.start_secs < 0.5 && s.end_secs > 1.0);
    let has_middle_silence = silences.iter().any(|s| s.start_secs > 4.0 && s.start_secs < 6.0);
    assert!(
        has_leading_silence,
        "expected silence in the leading 0-2s region: {silences:?}"
    );
    assert!(
        has_middle_silence,
        "expected silence in the middle 5-7s region: {silences:?}"
    );

    // Invert to speech and verify we get speech segments.
    let speech = vad::invert_to_speech(&silences, 10.0, 0.5);
    assert!(
        !speech.is_empty(),
        "expected at least one speech segment after inverting silences"
    );
    Ok(())
}

// ===========================================================================
// Test: Shot detection on multi-scene video
// ===========================================================================

#[tokio::test]
async fn shot_detection_on_multi_scene_video() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await {
        eprintln!("skipping: ffmpeg not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let video = generate_multi_scene_video(dir.path()).await?;

    let shots = scene::detect_shots("ffmpeg", &video, 0.1).await?;
    assert!(
        !shots.is_empty(),
        "expected shot boundaries in multi-scene video"
    );

    // Transitions should be near 2.0s and 4.0s.
    for &cut in &shots {
        let near_first = (cut - 2.0).abs() < 0.5;
        let near_second = (cut - 4.0).abs() < 0.5;
        assert!(
            near_first || near_second,
            "unexpected cut at {cut}s (expected near 2.0 or 4.0)"
        );
    }
    Ok(())
}

// ===========================================================================
// Test: RMS prosody curve on speech-like audio
// ===========================================================================

#[tokio::test]
async fn rms_curve_on_speech_like_audio() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await {
        eprintln!("skipping: ffmpeg not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let wav = generate_speech_like_audio(dir.path()).await?;

    let curve = prosody::rms_curve("ffmpeg", &wav, 1.0).await?;
    assert!(
        !curve.is_empty(),
        "expected at least one RMS window from 10s audio"
    );

    // The loud-tone windows (~2-5s, ~7-10s) should be louder than silence (~0-2s).
    let loud = prosody::mean_in_range(&curve, 2.0, 5.0);
    let silent = prosody::mean_in_range(&curve, 0.0, 2.0);
    if let (Some(loud_db), Some(silent_db)) = (loud, silent) {
        assert!(
            loud_db > silent_db + 5.0,
            "tone ({loud_db} dB) should be louder than silence ({silent_db} dB)"
        );
    }

    // Peak should exist in the loud region.
    let peak = prosody::peak_in_range(&curve, 2.0, 5.0);
    assert!(peak.is_some(), "expected a peak RMS in the tone region");
    Ok(())
}

// ===========================================================================
// Test: Caption ASS generation with all three aspect ratio styles
// ===========================================================================

#[tokio::test]
async fn caption_generation_all_styles() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;

    let words: Vec<AlignedWord> = vec![
        AlignedWord { text: "This".into(), start_secs: 0.0, end_secs: 0.3 },
        AlignedWord { text: "is".into(), start_secs: 0.3, end_secs: 0.5 },
        AlignedWord { text: "a".into(), start_secs: 0.5, end_secs: 0.6 },
        AlignedWord { text: "test.".into(), start_secs: 0.6, end_secs: 1.0 },
        AlignedWord { text: "Captions".into(), start_secs: 1.2, end_secs: 1.6 },
        AlignedWord { text: "should".into(), start_secs: 1.6, end_secs: 1.8 },
        AlignedWord { text: "work".into(), start_secs: 1.8, end_secs: 2.0 },
        AlignedWord { text: "correctly!".into(), start_secs: 2.0, end_secs: 2.5 },
    ];

    let configs = [
        ("vertical", CaptionStyle::for_vertical(), 1080u32, 1920u32),
        ("square", CaptionStyle::for_square(), 1080, 1080),
        ("landscape", CaptionStyle::for_landscape(), 1920, 1080),
    ];

    for (label, style, w, h) in &configs {
        let path = dir.path().join(format!("captions_{label}.ass"));
        write_ass(&path, &words, *w, *h, style).await?;

        let body = tokio::fs::read_to_string(&path).await?;
        assert!(body.contains("[Script Info]"), "{label}: missing Script Info");
        assert!(body.contains(&format!("PlayResX: {w}")), "{label}: wrong PlayResX");
        assert!(body.contains(&format!("PlayResY: {h}")), "{label}: wrong PlayResY");
        assert!(body.contains("[V4+ Styles]"), "{label}: missing styles");
        assert!(body.contains("[Events]"), "{label}: missing events");
        assert!(body.contains("Dialogue: "), "{label}: no dialogue events");

        // Should have "test." as a sentence-ending flush.
        assert!(body.contains("test."), "{label}: missing test phrase");
    }
    Ok(())
}

// ===========================================================================
// Test: Overlay ASS generation (hook text)
// ===========================================================================

#[tokio::test]
async fn overlay_generation_all_styles() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;

    let overlays = [
        ("vertical", OverlayStyle::for_vertical(), 1080u32, 1920u32, 130u32),
        ("square", OverlayStyle::for_square(), 1080, 1080, 100),
        ("landscape", OverlayStyle::for_landscape(), 1920, 1080, 80),
    ];

    for (label, style, w, h, expected_font_size) in &overlays {
        let path = dir.path().join(format!("overlay_{label}.ass"));
        write_overlay_ass(&path, "WAIT FOR IT", *w, *h, style).await?;

        let body = tokio::fs::read_to_string(&path).await?;
        assert!(body.contains("WAIT FOR IT"), "{label}: hook text missing");
        assert!(
            body.contains(&format!(",{expected_font_size},")),
            "{label}: expected font size {expected_font_size}"
        );
        assert!(body.contains("\\fade(300,300)"), "{label}: missing fade");
        // Alignment 5 (middle-center).
        assert!(body.contains(",5,"), "{label}: missing alignment 5");
        // Layer 1 (above captions).
        assert!(body.contains("Dialogue: 1,"), "{label}: overlay should be layer 1");
    }
    Ok(())
}

// ===========================================================================
// Test: Render with burned captions and overlay
// ===========================================================================

#[tokio::test]
async fn render_with_captions_and_overlay() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let src = generate_synthetic_video(dir.path()).await?;

    // Generate caption ASS.
    let words: Vec<AlignedWord> = vec![
        AlignedWord { text: "Hello".into(), start_secs: 0.0, end_secs: 0.5 },
        AlignedWord { text: "world.".into(), start_secs: 0.5, end_secs: 1.0 },
        AlignedWord { text: "Testing".into(), start_secs: 1.5, end_secs: 2.0 },
        AlignedWord { text: "captions.".into(), start_secs: 2.0, end_secs: 2.5 },
    ];
    let caption_path = dir.path().join("captions.ass");
    write_ass(&caption_path, &words, 1080, 1920, &CaptionStyle::for_vertical()).await?;

    // Generate overlay ASS.
    let overlay_path = dir.path().join("overlay.ass");
    write_overlay_ass(
        &overlay_path,
        "WAIT FOR IT",
        1080,
        1920,
        &OverlayStyle::for_vertical(),
    )
    .await?;

    // Render vertical clip with both subtitle layers burned in.
    let out = dir.path().join("clip_with_subs.mp4");
    let profile = RenderProfile::shorts_vertical();
    render_clip(
        "ffmpeg",
        &src,
        1.0,
        6.0,
        &out,
        &profile,
        &[caption_path.as_path(), overlay_path.as_path()],
    )
    .await?;

    let meta = tokio::fs::metadata(&out).await?;
    assert!(meta.len() > 0, "output with subs is empty");

    let (w, h) = probe_dimensions(&out).await?;
    assert_eq!(w, 1080);
    assert_eq!(h, 1920);
    Ok(())
}

// ===========================================================================
// Test: Candidate generation with synthetic word stream
// ===========================================================================

#[test]
fn candidate_generation_end_to_end() {
    // 5 minutes of dense speech at 3 words/sec = 900 words.
    let total_duration = 300.0;
    let words = dense_words(total_duration, 3.0);

    // Simulated silences at ~60s and ~120s.
    let silences = vec![
        SilenceWindow { start_secs: 59.5, end_secs: 60.5 },
        SilenceWindow { start_secs: 119.5, end_secs: 120.5 },
        SilenceWindow { start_secs: 179.5, end_secs: 180.5 },
        SilenceWindow { start_secs: 239.5, end_secs: 240.5 },
    ];

    // Simulated shot boundaries.
    let shots = vec![30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0];

    // RMS curve — alternating loud/quiet windows.
    let rms: Vec<RmsWindow> = (0..300)
        .map(|i| RmsWindow {
            start_secs: i as f64,
            rms_db: if i % 2 == 0 { -15.0 } else { -25.0 },
        })
        .collect();

    let cgen = CandidateGenerator::default();
    let candidates = cgen.generate(total_duration, &words, &silences, &shots, &rms);

    assert!(
        !candidates.is_empty(),
        "expected candidate windows from 5-min episode"
    );

    // Every candidate should respect min/max duration.
    for c in &candidates {
        let dur = c.duration_secs();
        assert!(
            dur >= cgen.min_secs - 1e-6,
            "candidate {}-{}s below min duration: {dur}",
            c.start_secs, c.end_secs
        );
        assert!(
            dur <= cgen.max_secs + 1e-6,
            "candidate {}-{}s above max duration: {dur}",
            c.start_secs, c.end_secs
        );
    }

    // Every candidate should have enough words.
    for c in &candidates {
        assert!(
            c.word_count >= cgen.min_words,
            "candidate {}-{}s has only {} words (min {})",
            c.start_secs, c.end_secs, c.word_count, cgen.min_words
        );
    }

    // Candidates should have linguistic features computed.
    for c in &candidates {
        assert!(
            c.linguistic.total_word_count > 0,
            "candidate should have extracted linguistic features"
        );
    }

    // Candidates should have prosody features.
    let has_rms = candidates.iter().any(|c| c.rms_peak_db.is_some());
    assert!(has_rms, "at least some candidates should have RMS features");

    // Candidates should have speaking rate.
    let has_rate = candidates.iter().any(|c| c.speaking_rate_wps.is_some());
    assert!(has_rate, "at least some candidates should have speaking rate");

    // Novelty should be None (requires embedding pass).
    for c in &candidates {
        assert!(c.novelty_score.is_none(), "novelty should be None before attach_novelty");
    }

    // Windows should be monotonically advancing.
    for pair in candidates.windows(2) {
        assert!(
            pair[1].start_secs > pair[0].start_secs,
            "windows not advancing: {} then {}",
            pair[0].start_secs, pair[1].start_secs
        );
    }
}

// ===========================================================================
// Test: Candidate generation snaps to silence/shot boundaries
// ===========================================================================

#[test]
fn candidate_snapping_to_silence_and_shots() {
    let total_duration = 200.0;
    let words = dense_words(total_duration, 3.0);

    // Place silences where candidate boundaries might snap to.
    let silences = vec![
        SilenceWindow { start_secs: 29.0, end_secs: 31.0 },
        SilenceWindow { start_secs: 59.0, end_secs: 61.0 },
        SilenceWindow { start_secs: 89.0, end_secs: 91.0 },
    ];

    let cgen = CandidateGenerator::default();
    let candidates = cgen.generate(total_duration, &words, &silences, &[], &[]);

    assert!(!candidates.is_empty());

    // Check that at least some boundaries aligned with silence edges.
    let snapped_to_silence = candidates.iter().any(|c| {
        silences.iter().any(|s| {
            (c.start_secs - s.start_secs).abs() < 0.1
                || (c.start_secs - s.end_secs).abs() < 0.1
                || (c.end_secs - s.start_secs).abs() < 0.1
                || (c.end_secs - s.end_secs).abs() < 0.1
        })
    });
    assert!(
        snapped_to_silence,
        "expected at least one boundary to snap to a silence edge"
    );
}

// ===========================================================================
// Test: Caption ASS pure rendering (no file I/O)
// ===========================================================================

#[test]
fn caption_render_ass_phrase_grouping() {
    let words: Vec<AlignedWord> = vec![
        AlignedWord { text: "First".into(), start_secs: 0.0, end_secs: 0.3 },
        AlignedWord { text: "phrase".into(), start_secs: 0.3, end_secs: 0.5 },
        AlignedWord { text: "here.".into(), start_secs: 0.5, end_secs: 0.8 },
        AlignedWord { text: "Second".into(), start_secs: 1.0, end_secs: 1.3 },
        AlignedWord { text: "phrase!".into(), start_secs: 1.3, end_secs: 1.6 },
        AlignedWord { text: "And".into(), start_secs: 2.0, end_secs: 2.2 },
        AlignedWord { text: "a".into(), start_secs: 2.2, end_secs: 2.3 },
        AlignedWord { text: "third".into(), start_secs: 2.3, end_secs: 2.5 },
        AlignedWord { text: "one?".into(), start_secs: 2.5, end_secs: 2.8 },
    ];

    let body = render_ass(&words, 1080, 1920, &CaptionStyle::for_vertical());
    let dialogue_count = body.matches("Dialogue: ").count();

    // Terminal punctuation (.!?) triggers flush — expect 3 phrases.
    assert_eq!(
        dialogue_count, 3,
        "expected 3 Dialogue events for 3 sentences, got {dialogue_count}"
    );
    assert!(body.contains("First phrase here."));
    assert!(body.contains("Second phrase!"));
    assert!(body.contains("And a third one?"));
}

// ===========================================================================
// Test: Overlay sanitizes special characters
// ===========================================================================

#[test]
fn overlay_sanitizes_braces_and_backslashes() {
    let body = render_overlay_ass(
        r"Look at {this} \ thing",
        1080,
        1920,
        &OverlayStyle::for_vertical(),
    );
    assert!(body.contains(r"Look at \{this\} \\ thing"));
    assert!(!body.contains(r"{this}"));
}

// ===========================================================================
// Test: Full pipeline integration — video → audio → silence → shots → candidates
// ===========================================================================

#[tokio::test]
async fn full_pipeline_video_to_candidates() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let video = generate_multi_scene_video(dir.path()).await?;

    // Step 1: Extract audio.
    let audio = dir.path().join("audio.m4a");
    media::extract_audio_m4a("ffmpeg", &video, &audio).await?;

    // Step 2: Probe duration.
    let duration = media::duration_secs("ffprobe", &audio).await?;
    assert!(duration > 5.0, "expected >5s duration, got {duration}");

    // Step 3: Detect silences.
    let silences = vad::detect_silences("ffmpeg", &audio, -30.0, 0.3).await?;
    // Synthetic 6s with constant tone may have no silence — that's OK.

    // Step 4: Detect shots.
    let shots = scene::detect_shots("ffmpeg", &video, 0.1).await?;
    assert!(
        !shots.is_empty(),
        "multi-scene video should have shot boundaries"
    );

    // Step 5: Build RMS curve.
    let rms = prosody::rms_curve("ffmpeg", &audio, 1.0).await?;
    // May be empty for very short audio; that's acceptable.

    // Step 6: Generate candidate windows with synthetic words.
    // Use a relaxed generator to work with the short 6s duration.
    let words = dense_words(duration, 10.0); // High density to pass min_words.
    let cgen = CandidateGenerator {
        target_secs: 5.0,
        min_secs: 4.0,
        max_secs: 7.0,
        stride_secs: 3.0,
        min_words: 5,
        ..CandidateGenerator::default()
    };
    let candidates = cgen.generate(duration, &words, &silences, &shots, &rms);
    assert!(
        !candidates.is_empty(),
        "expected at least one candidate from 6s video with dense words"
    );

    for c in &candidates {
        assert!(c.transcript.len() > 0, "candidate should have transcript");
        assert!(c.word_count >= cgen.min_words);
    }
    Ok(())
}

// ===========================================================================
// Test: Full pipeline — candidates → captions → render all formats
// ===========================================================================

#[tokio::test]
async fn full_pipeline_candidates_to_rendered_clips() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let src = generate_synthetic_video(dir.path()).await?;

    // Simulate a ranked clip: clip from 5s to 15s with hook text.
    let clip_start = 5.0;
    let clip_end = 15.0;
    let hook = "You won't believe this!";

    // Generate clip-local word timestamps.
    let clip_words: Vec<AlignedWord> = vec![
        AlignedWord { text: "You".into(), start_secs: 0.0, end_secs: 0.3 },
        AlignedWord { text: "won't".into(), start_secs: 0.3, end_secs: 0.6 },
        AlignedWord { text: "believe".into(), start_secs: 0.6, end_secs: 1.0 },
        AlignedWord { text: "this!".into(), start_secs: 1.0, end_secs: 1.3 },
        AlignedWord { text: "The".into(), start_secs: 2.0, end_secs: 2.2 },
        AlignedWord { text: "secret".into(), start_secs: 2.2, end_secs: 2.6 },
        AlignedWord { text: "is".into(), start_secs: 2.6, end_secs: 2.8 },
        AlignedWord { text: "out.".into(), start_secs: 2.8, end_secs: 3.2 },
    ];

    let format_configs: Vec<(&str, RenderProfile, CaptionStyle, OverlayStyle, u32, u32)> = vec![
        (
            "9x16",
            RenderProfile::shorts_vertical(),
            CaptionStyle::for_vertical(),
            OverlayStyle::for_vertical(),
            1080,
            1920,
        ),
        (
            "1x1",
            RenderProfile::linkedin_square(),
            CaptionStyle::for_square(),
            OverlayStyle::for_square(),
            1080,
            1080,
        ),
        (
            "16x9",
            RenderProfile::bluesky_landscape(),
            CaptionStyle::for_landscape(),
            OverlayStyle::for_landscape(),
            1920,
            1080,
        ),
    ];

    for (label, profile, cap_style, ovl_style, exp_w, exp_h) in &format_configs {
        // Write captions ASS.
        let cap_path = dir.path().join(format!("captions_{label}.ass"));
        write_ass(&cap_path, &clip_words, *exp_w, *exp_h, cap_style).await?;

        // Write overlay ASS.
        let ovl_path = dir.path().join(format!("overlay_{label}.ass"));
        write_overlay_ass(&ovl_path, hook, *exp_w, *exp_h, ovl_style).await?;

        // Render clip with burned subs.
        let out = dir.path().join(format!("final_{label}.mp4"));
        render_clip(
            "ffmpeg",
            &src,
            clip_start,
            clip_end,
            &out,
            profile,
            &[cap_path.as_path(), ovl_path.as_path()],
        )
        .await?;

        let meta = tokio::fs::metadata(&out).await?;
        assert!(meta.len() > 0, "{label}: rendered clip is empty");

        let (w, h) = probe_dimensions(&out).await?;
        assert_eq!(w, *exp_w, "{label}: width mismatch");
        assert_eq!(h, *exp_h, "{label}: height mismatch");
    }
    Ok(())
}

// ===========================================================================
// Test: Screenshot extraction from synthetic video
// ===========================================================================

#[tokio::test]
async fn screenshot_extraction() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await {
        eprintln!("skipping: ffmpeg not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let src = generate_synthetic_video(dir.path()).await?;

    let screenshot = dir.path().join("thumb.jpg");
    media::screenshot_jpeg("ffmpeg", &src, 10.0, &screenshot, 720).await?;

    let meta = tokio::fs::metadata(&screenshot).await?;
    assert!(meta.len() > 0, "screenshot JPEG is empty");
    Ok(())
}

// ===========================================================================
// Test: Empty / edge-case inputs
// ===========================================================================

#[test]
fn candidate_generation_empty_inputs() {
    let cgen = CandidateGenerator::default();

    // Zero duration.
    let out = cgen.generate(0.0, &[], &[], &[], &[]);
    assert!(out.is_empty());

    // Duration below min_secs.
    let out = cgen.generate(10.0, &[], &[], &[], &[]);
    assert!(out.is_empty());

    // Enough duration but no words.
    let out = cgen.generate(120.0, &[], &[], &[], &[]);
    assert!(out.is_empty(), "no words should produce no candidates");
}

#[test]
fn caption_with_no_words_produces_no_dialogue() {
    let body = render_ass(&[], 1080, 1920, &CaptionStyle::for_vertical());
    assert!(body.contains("[Events]"));
    assert!(!body.contains("Dialogue: "));
}

#[test]
fn overlay_with_empty_hook_produces_no_dialogue() {
    let body = render_overlay_ass("", 1080, 1920, &OverlayStyle::for_vertical());
    assert!(body.contains("[Events]"));
    assert!(!body.contains("Dialogue: "));
}

#[tokio::test]
async fn render_rejects_invalid_time_range() {
    let dir = tempfile::tempdir().unwrap();
    let dummy_in = dir.path().join("in.mp4");
    let dummy_out = dir.path().join("out.mp4");
    let profile = RenderProfile::shorts_vertical();

    // end == start → should fail.
    let result = render_clip("ffmpeg", &dummy_in, 5.0, 5.0, &dummy_out, &profile, &[]).await;
    assert!(result.is_err());

    // end < start → should fail.
    let result = render_clip("ffmpeg", &dummy_in, 10.0, 3.0, &dummy_out, &profile, &[]).await;
    assert!(result.is_err());
}

// ===========================================================================
// Test: Audio transcode (audio-only input)
// ===========================================================================

#[tokio::test]
async fn transcode_audio_to_m4a() -> anyhow::Result<()> {
    if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let wav = generate_speech_like_audio(dir.path()).await?;

    let m4a = dir.path().join("transcoded.m4a");
    media::transcode_audio_to_m4a("ffmpeg", &wav, &m4a).await?;

    let meta = tokio::fs::metadata(&m4a).await?;
    assert!(meta.len() > 0, "transcoded m4a is empty");

    let dur = media::duration_secs("ffprobe", &m4a).await?;
    assert!(
        (dur - 10.0).abs() < 1.0,
        "expected ~10s transcoded duration, got {dur}"
    );
    Ok(())
}
