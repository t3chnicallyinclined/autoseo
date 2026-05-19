//! Silence/speech-window detection with pluggable backends.
//!
//! Two backends are available:
//! - **Silero VAD** (default): ONNX model via `ort`, more accurate speech/silence detection.
//! - **ffmpeg silencedetect** (fallback): the original M1 implementation.
//!
//! Backend is selected via `VAD_BACKEND` env var (`silero` or `ffmpeg`).
//! If Silero is requested but the model file is missing, falls back to ffmpeg automatically.
//!
//! The module exposes two views over the same data:
//! - [`SilenceWindow`] — explicit silence regions.
//! - [`SpeechSegment`] — derived by inverting silences against the total duration.

use anyhow::Context;
use regex::Regex;
use std::path::Path;
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct SilenceWindow {
    pub start_secs: f64,
    pub end_secs: f64,
}

impl SilenceWindow {
    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }

    pub fn center_secs(&self) -> f64 {
        (self.start_secs + self.end_secs) / 2.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSegment {
    pub start_secs: f64,
    pub end_secs: f64,
}

impl SpeechSegment {
    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }
}

/// Which VAD backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadBackend {
    Silero,
    Ffmpeg,
}

impl VadBackend {
    pub fn from_config(backend_str: &str) -> Self {
        match backend_str.to_lowercase().as_str() {
            "ffmpeg" => VadBackend::Ffmpeg,
            _ => VadBackend::Silero,
        }
    }
}

/// Detect silence windows using the configured backend.
///
/// For `Silero`: reads audio via `hound`, runs Silero VAD ONNX model, derives silence windows.
/// For `Ffmpeg`: uses `silencedetect` filter (original behavior).
///
/// Falls back to ffmpeg if Silero model file is missing.
pub async fn detect_silences(
    ffmpeg: &str,
    media_path: &Path,
    noise_db: f64,
    min_duration_secs: f64,
    backend: VadBackend,
    model_path: &str,
    threshold: f64,
) -> anyhow::Result<Vec<SilenceWindow>> {
    match backend {
        VadBackend::Silero => {
            let model_file = Path::new(model_path);
            if !model_file.exists() {
                tracing::warn!(
                    path = model_path,
                    "silero_vad.onnx not found, falling back to ffmpeg silencedetect"
                );
                return detect_silences_ffmpeg(ffmpeg, media_path, noise_db, min_duration_secs)
                    .await;
            }
            let canonical = model_file
                .canonicalize()
                .with_context(|| format!("canonicalize VAD model path: {model_path}"))?;
            let canonical_str = canonical.to_str().ok_or_else(|| {
                anyhow::anyhow!("VAD model path is not valid UTF-8: {}", canonical.display())
            })?;
            detect_silences_silero(
                ffmpeg,
                media_path,
                canonical_str,
                threshold,
                min_duration_secs,
            )
            .await
        }
        VadBackend::Ffmpeg => {
            detect_silences_ffmpeg(ffmpeg, media_path, noise_db, min_duration_secs).await
        }
    }
}

// ── ffmpeg backend ──────────────────────────────────────────────────────────

async fn detect_silences_ffmpeg(
    ffmpeg: &str,
    media_path: &Path,
    noise_db: f64,
    min_duration_secs: f64,
) -> anyhow::Result<Vec<SilenceWindow>> {
    let min_duration = min_duration_secs.max(0.01);
    let af = format!("silencedetect=noise={noise_db}dB:d={min_duration}");

    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-nostats"])
        .args(["-vn", "-sn", "-dn"])
        .arg("-i")
        .arg(media_path)
        .args(["-af", &af])
        .args(["-f", "null", "-"])
        .output()
        .await
        .with_context(|| format!("run ffmpeg silencedetect on {}", media_path.display()))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg silencedetect failed: {} {}", output.status, err);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(parse_silencedetect(&stderr))
}

// ── Silero VAD backend ──────────────────────────────────────────────────────

/// Extract 16kHz mono f32 PCM from any media file using ffmpeg, then run Silero VAD.
async fn detect_silences_silero(
    ffmpeg: &str,
    media_path: &Path,
    model_path: &str,
    threshold: f64,
    min_silence_secs: f64,
) -> anyhow::Result<Vec<SilenceWindow>> {
    let samples = extract_16k_mono_pcm(ffmpeg, media_path).await?;
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let model_path_owned = model_path.to_string();
    let silences = tokio::task::spawn_blocking(move || {
        run_silero_vad(&samples, &model_path_owned, threshold, min_silence_secs)
    })
    .await??;

    Ok(silences)
}

/// Use ffmpeg to convert any media to raw 16kHz mono f32le PCM bytes, then parse as f32 samples.
async fn extract_16k_mono_pcm(ffmpeg: &str, media_path: &Path) -> anyhow::Result<Vec<f32>> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-nostats", "-loglevel", "error"])
        .args(["-vn", "-sn", "-dn"])
        .arg("-i")
        .arg(media_path)
        .args(["-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .output()
        .await
        .with_context(|| format!("ffmpeg PCM extraction from {}", media_path.display()))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg PCM extraction failed: {} {}", output.status, err);
    }

    let bytes = &output.stdout;
    if bytes.len() % 4 != 0 {
        anyhow::bail!("PCM output length {} is not a multiple of 4", bytes.len());
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(samples)
}

/// Run Silero VAD v5 ONNX model on 16kHz mono f32 samples.
///
/// The model processes audio in chunks of 512 samples (32ms at 16kHz).
/// Returns silence windows derived from speech probability output.
fn run_silero_vad(
    samples: &[f32],
    model_path: &str,
    threshold: f64,
    min_silence_secs: f64,
) -> anyhow::Result<Vec<SilenceWindow>> {
    use ndarray::Array;
    use ort::session::Session;

    let session = Session::builder()
        .context("create ORT session builder")?
        .with_intra_threads(1)
        .context("set intra threads")?
        .commit_from_file(model_path)
        .with_context(|| format!("load Silero VAD model from {model_path}"))?;

    let sample_rate: i64 = 16000;
    let chunk_size: usize = 512; // 32ms at 16kHz (Silero v5 expects this for 16kHz)
    let secs_per_chunk = chunk_size as f64 / sample_rate as f64;

    // State tensor: shape [2, 1, 128] initialized to zeros (Silero v5)
    let mut state = Array::zeros((2, 1, 128_usize)).into_dyn();
    let sr_tensor = Array::from_elem((1,), sample_rate).into_dyn();

    let total_chunks = samples.len().div_ceil(chunk_size);
    let mut speech_probs: Vec<f64> = Vec::with_capacity(total_chunks);

    for chunk_idx in 0..total_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(samples.len());

        // Pad last chunk with zeros if needed
        let mut chunk_data = vec![0.0f32; chunk_size];
        chunk_data[..end - start].copy_from_slice(&samples[start..end]);

        let input_tensor = Array::from_shape_vec((1, chunk_size), chunk_data)
            .context("shape input tensor")?
            .into_dyn();

        let outputs = session.run(ort::inputs![
            "input" => input_tensor,
            "state" => state.clone(),
            "sr" => sr_tensor.clone(),
        ]?)?;

        // Output probability
        let prob_output = outputs["output"]
            .try_extract_tensor::<f32>()
            .context("extract output probability")?;
        let prob = prob_output.iter().next().copied().unwrap_or(0.0) as f64;
        speech_probs.push(prob);

        // Update state for next iteration
        let new_state = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .context("extract new state")?;
        state = new_state.to_owned().into_dyn();
    }

    // Convert speech probabilities to silence windows
    Ok(probs_to_silence_windows(
        &speech_probs,
        secs_per_chunk,
        threshold,
        min_silence_secs,
    ))
}

/// Convert per-frame speech probabilities into silence windows.
/// A frame with prob < threshold is considered silence.
/// Adjacent silence frames are merged; short silences (< min_silence_secs) are discarded.
fn probs_to_silence_windows(
    probs: &[f64],
    secs_per_frame: f64,
    threshold: f64,
    min_silence_secs: f64,
) -> Vec<SilenceWindow> {
    let mut windows = Vec::new();
    let mut silence_start: Option<usize> = None;

    for (i, &prob) in probs.iter().enumerate() {
        let is_silence = prob < threshold;
        match (is_silence, silence_start) {
            (true, None) => {
                silence_start = Some(i);
            }
            (false, Some(start)) => {
                let start_secs = start as f64 * secs_per_frame;
                let end_secs = i as f64 * secs_per_frame;
                if end_secs - start_secs >= min_silence_secs {
                    windows.push(SilenceWindow {
                        start_secs,
                        end_secs,
                    });
                }
                silence_start = None;
            }
            _ => {}
        }
    }

    // Close trailing silence
    if let Some(start) = silence_start {
        let start_secs = start as f64 * secs_per_frame;
        let end_secs = probs.len() as f64 * secs_per_frame;
        if end_secs - start_secs >= min_silence_secs {
            windows.push(SilenceWindow {
                start_secs,
                end_secs,
            });
        }
    }

    windows
}

/// Invert silence windows to get speech segments over a given total duration.
/// Drops segments shorter than `min_speech_secs` to avoid sub-word fragments.
pub fn invert_to_speech(
    silences: &[SilenceWindow],
    total_duration_secs: f64,
    min_speech_secs: f64,
) -> Vec<SpeechSegment> {
    if total_duration_secs <= 0.0 {
        return Vec::new();
    }

    // Sort & coalesce overlapping silences defensively.
    let mut sorted: Vec<SilenceWindow> = silences.to_vec();
    sorted.sort_by(|a, b| {
        a.start_secs
            .partial_cmp(&b.start_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = Vec::new();
    let mut cursor = 0.0_f64;
    for s in sorted.iter() {
        let s_start = s.start_secs.max(0.0).min(total_duration_secs);
        let s_end = s.end_secs.max(s_start).min(total_duration_secs);
        if s_start > cursor {
            let speech_start = cursor;
            let speech_end = s_start;
            if speech_end - speech_start >= min_speech_secs {
                out.push(SpeechSegment {
                    start_secs: speech_start,
                    end_secs: speech_end,
                });
            }
        }
        if s_end > cursor {
            cursor = s_end;
        }
    }
    if cursor < total_duration_secs {
        let speech_start = cursor;
        let speech_end = total_duration_secs;
        if speech_end - speech_start >= min_speech_secs {
            out.push(SpeechSegment {
                start_secs: speech_start,
                end_secs: speech_end,
            });
        }
    }
    out
}

/// Snap a target timestamp to the nearest silence boundary (start or end of any silence)
/// within `max_drift_secs`. Useful for ensuring clip cuts land on a natural pause.
/// Returns the original target if no boundary is close enough.
pub fn snap_to_silence_boundary(
    target_secs: f64,
    silences: &[SilenceWindow],
    max_drift_secs: f64,
) -> f64 {
    if silences.is_empty() {
        return target_secs;
    }
    let mut best = target_secs;
    let mut best_drift = max_drift_secs;
    for s in silences {
        for boundary in [s.start_secs, s.end_secs] {
            let drift = (boundary - target_secs).abs();
            if drift <= best_drift {
                best_drift = drift;
                best = boundary;
            }
        }
    }
    best
}

fn parse_silencedetect(stderr: &str) -> Vec<SilenceWindow> {
    // silencedetect emits two log lines per silence window, e.g.:
    //   [silencedetect @ 0x...] silence_start: 3.456
    //   [silencedetect @ 0x...] silence_end: 5.123 | silence_duration: 1.667
    // At EOF, ffmpeg may or may not emit a closing silence_end depending on version;
    // we discard unmatched starts.
    let start_re = Regex::new(r"silence_start:\s*(-?\d+(?:\.\d+)?)").expect("static regex");
    let end_re = Regex::new(r"silence_end:\s*(-?\d+(?:\.\d+)?)").expect("static regex");

    let mut out = Vec::new();
    let mut pending_start: Option<f64> = None;
    for line in stderr.lines() {
        if let Some(c) = start_re.captures(line) {
            if let Some(s) = c.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) {
                pending_start = Some(s.max(0.0));
            }
            continue;
        }
        if let Some(c) = end_re.captures(line)
            && let Some(e) = c.get(1).and_then(|m| m.as_str().parse::<f64>().ok())
            && let Some(s) = pending_start.take()
            && e > s
        {
            out.push(SilenceWindow {
                start_secs: s,
                end_secs: e,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn parses_paired_silence_lines() {
        let stderr = "\
[silencedetect @ 0x55a] silence_start: 0
[silencedetect @ 0x55a] silence_end: 1.5 | silence_duration: 1.5
[silencedetect @ 0x55a] silence_start: 2.5
[silencedetect @ 0x55a] silence_end: 4 | silence_duration: 1.5
";
        let windows = parse_silencedetect(stderr);
        assert_eq!(windows.len(), 2);
        assert!((windows[0].start_secs - 0.0).abs() < 1e-6);
        assert!((windows[0].end_secs - 1.5).abs() < 1e-6);
        assert!((windows[1].start_secs - 2.5).abs() < 1e-6);
        assert!((windows[1].end_secs - 4.0).abs() < 1e-6);
    }

    #[test]
    fn parser_drops_unmatched_start_at_eof() {
        let stderr = "\
[silencedetect @ 0x55a] silence_start: 1.0
[silencedetect @ 0x55a] silence_end: 2.0 | silence_duration: 1.0
[silencedetect @ 0x55a] silence_start: 5.0
";
        let windows = parse_silencedetect(stderr);
        assert_eq!(windows.len(), 1);
    }

    #[test]
    fn parser_ignores_negative_start() {
        let stderr = "\
[silencedetect @ 0x55a] silence_start: -0.000016
[silencedetect @ 0x55a] silence_end: 1.5 | silence_duration: 1.5
";
        let windows = parse_silencedetect(stderr);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_secs, 0.0);
    }

    #[test]
    fn invert_basic() {
        let silences = vec![
            SilenceWindow {
                start_secs: 0.0,
                end_secs: 1.5,
            },
            SilenceWindow {
                start_secs: 2.5,
                end_secs: 4.0,
            },
        ];
        let speech = invert_to_speech(&silences, 6.0, 0.1);
        assert_eq!(speech.len(), 2);
        assert_eq!(speech[0].start_secs, 1.5);
        assert_eq!(speech[0].end_secs, 2.5);
        assert_eq!(speech[1].start_secs, 4.0);
        assert_eq!(speech[1].end_secs, 6.0);
    }

    #[test]
    fn invert_no_silence_yields_full_speech() {
        let speech = invert_to_speech(&[], 10.0, 0.1);
        assert_eq!(speech.len(), 1);
        assert_eq!(speech[0].start_secs, 0.0);
        assert_eq!(speech[0].end_secs, 10.0);
    }

    #[test]
    fn invert_drops_micro_speech() {
        let silences = vec![
            SilenceWindow {
                start_secs: 0.0,
                end_secs: 1.0,
            },
            SilenceWindow {
                start_secs: 1.05,
                end_secs: 2.0,
            },
        ];
        let speech = invert_to_speech(&silences, 3.0, 0.5);
        assert_eq!(speech.len(), 1);
        assert_eq!(speech[0].start_secs, 2.0);
    }

    #[test]
    fn snap_to_silence_boundary_picks_closest() {
        let silences = vec![SilenceWindow {
            start_secs: 1.5,
            end_secs: 2.5,
        }];
        assert_eq!(snap_to_silence_boundary(1.4, &silences, 0.5), 1.5);
        assert_eq!(snap_to_silence_boundary(2.6, &silences, 0.5), 2.5);
        assert_eq!(snap_to_silence_boundary(0.0, &silences, 0.5), 0.0);
        assert_eq!(snap_to_silence_boundary(5.0, &[], 1.0), 5.0);
    }

    #[test]
    fn probs_to_silence_basic() {
        // 10 frames at 32ms each = 0.32s total
        // Frames 0-2: silence (prob=0.1), 3-6: speech (prob=0.9), 7-9: silence (prob=0.05)
        let probs = vec![0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.05, 0.05, 0.05];
        let secs_per_frame = 0.032;
        let windows = probs_to_silence_windows(&probs, secs_per_frame, 0.5, 0.05);
        assert_eq!(windows.len(), 2);
        // First silence: frames 0-2 → 0.0 to 0.096s
        assert!((windows[0].start_secs - 0.0).abs() < 1e-6);
        assert!((windows[0].end_secs - 0.096).abs() < 1e-6);
        // Second silence: frames 7-9 → 0.224 to 0.320s
        assert!((windows[1].start_secs - 0.224).abs() < 1e-6);
        assert!((windows[1].end_secs - 0.320).abs() < 1e-6);
    }

    #[test]
    fn probs_to_silence_drops_short() {
        // Single short silence frame shouldn't be reported with min_silence_secs=0.1
        let probs = vec![0.9, 0.1, 0.9, 0.9];
        let windows = probs_to_silence_windows(&probs, 0.032, 0.5, 0.1);
        assert!(windows.is_empty());
    }

    #[tokio::test]
    async fn detect_silences_finds_gaps_in_synthetic_audio() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let wav = dir.path().join("test.wav");

        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1.5"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:r=16000:d=1"])
            .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1.5"])
            .args(["-filter_complex", "[0][1][2]concat=n=3:v=0:a=1[a]"])
            .args(["-map", "[a]"])
            .arg(&wav)
            .status()
            .await?;
        anyhow::ensure!(status.success(), "ffmpeg synthetic wav generation failed");

        // Test ffmpeg backend explicitly
        let silences =
            detect_silences("ffmpeg", &wav, -30.0, 0.3, VadBackend::Ffmpeg, "", 0.5).await?;
        assert!(
            !silences.is_empty(),
            "expected silence windows in synthetic audio, got none"
        );
        for s in &silences {
            let is_leading = s.start_secs < 0.5 && s.end_secs > 1.0;
            let is_trailing = s.start_secs > 2.0;
            assert!(
                is_leading || is_trailing,
                "silence {s:?} doesn't match either expected gap"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn detect_silences_constant_tone_has_none() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let wav = dir.path().join("tone.wav");
        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:r=16000:d=3"])
            .arg(&wav)
            .status()
            .await?;
        anyhow::ensure!(status.success());

        let silences =
            detect_silences("ffmpeg", &wav, -30.0, 0.3, VadBackend::Ffmpeg, "", 0.5).await?;
        assert!(
            silences.is_empty(),
            "got silences in tone-only audio: {silences:?}"
        );
        Ok(())
    }
}
