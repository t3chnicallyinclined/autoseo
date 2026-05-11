//! Silence/speech-window detection backed by ffmpeg's `silencedetect` audio filter.
//!
//! M1 implementation: pure ffmpeg, no ML deps. Silero VAD via `ort` is deferred
//! to M3, at which point it can be swapped in behind the same public API.
//!
//! The module exposes two views over the same data:
//! - [`SilenceWindow`] — explicit silence regions (what `silencedetect` reports).
//! - [`SpeechSegment`] — derived by inverting silences against the total duration.
//!   Used by the ranker for turn-density features.

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

/// Detect silence windows in an audio (or video w/ audio) file.
///
/// `noise_db` is the silence threshold relative to full-scale (e.g. `-30.0` = -30 dB).
/// Lower (more negative) = more sensitive, reports softer audio as silence.
/// Typical: -30 dB for clean studio, -40 dB for noisier recordings.
///
/// `min_duration_secs` is the shortest silence to report (e.g. 0.3 catches natural pauses,
/// 1.0 only reports long gaps).
pub async fn detect_silences(
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
    let mut sorted: Vec<SilenceWindow> = silences.iter().cloned().collect();
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
        if let Some(c) = end_re.captures(line) {
            if let Some(e) = c.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) {
                if let Some(s) = pending_start.take() {
                    if e > s {
                        out.push(SilenceWindow {
                            start_secs: s,
                            end_secs: e,
                        });
                    }
                }
            }
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
        // Some ffmpeg builds emit "silence_start: -0.000016" at t=0.
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
        // 0.05s sliver between silences should be dropped by min_speech_secs=0.5
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
        // Closest to silence start
        assert_eq!(snap_to_silence_boundary(1.4, &silences, 0.5), 1.5);
        // Closest to silence end
        assert_eq!(snap_to_silence_boundary(2.6, &silences, 0.5), 2.5);
        // Outside drift budget
        assert_eq!(snap_to_silence_boundary(0.0, &silences, 0.5), 0.0);
        // Empty silences
        assert_eq!(snap_to_silence_boundary(5.0, &[], 1.0), 5.0);
    }

    #[tokio::test]
    async fn detect_silences_finds_gaps_in_synthetic_audio() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let wav = dir.path().join("test.wav");

        // 1.5s silence + 1s 440Hz tone + 1.5s silence = 4s total.
        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1.5"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:r=16000:d=1"])
            .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1.5"])
            .args([
                "-filter_complex",
                "[0][1][2]concat=n=3:v=0:a=1[a]",
            ])
            .args(["-map", "[a]"])
            .arg(&wav)
            .status()
            .await?;
        anyhow::ensure!(status.success(), "ffmpeg synthetic wav generation failed");

        let silences = detect_silences("ffmpeg", &wav, -30.0, 0.3).await?;
        assert!(
            !silences.is_empty(),
            "expected silence windows in synthetic audio, got none"
        );
        // Every detected silence must be either at start (~0..1.5) or end (~2.5..4.0).
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

        let silences = detect_silences("ffmpeg", &wav, -30.0, 0.3).await?;
        assert!(silences.is_empty(), "got silences in tone-only audio: {silences:?}");
        Ok(())
    }
}
