//! Per-window prosody features: RMS energy curve via ffmpeg's `astats` filter
//! and fundamental frequency (F0) extraction via the `aubio` CLI.
//!
//! F0 extraction is non-fatal: if the `aubio` binary is not on PATH, the F0
//! features are silently skipped and callers receive an empty vector.
//!
//! Speaking rate is derived directly from word timestamps in `candidates.rs`,
//! not here.

use anyhow::Context;
use regex::Regex;
use std::path::Path;
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct RmsWindow {
    /// Start time of the window in seconds.
    pub start_secs: f64,
    /// Mean RMS level over the window, in dBFS (e.g. -23.0). Lower = quieter.
    pub rms_db: f64,
}

/// Compute the per-window RMS energy curve for an audio (or video w/ audio) file.
/// `window_secs` controls bucket size — 1.0 is a reasonable default.
/// Output is sorted by `start_secs` ascending.
pub async fn rms_curve(
    ffmpeg: &str,
    media_path: &Path,
    window_secs: f64,
) -> anyhow::Result<Vec<RmsWindow>> {
    let window = window_secs.max(0.1);
    // astats with length=N emits metadata once per N-second window;
    // ametadata=print writes those key/value pairs (and a pts_time per frame) to stdout.
    let af = format!("astats=metadata=1:length={window},ametadata=print:file=-");

    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-nostats"])
        .args(["-vn", "-sn", "-dn"])
        .arg("-i")
        .arg(media_path)
        .args(["-af", &af])
        .args(["-f", "null", "-"])
        .output()
        .await
        .with_context(|| format!("run ffmpeg astats on {}", media_path.display()))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg astats failed: {} {}", output.status, err);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_astats_metadata(&stdout))
}

/// Find the window with the highest RMS within `[start_secs, end_secs)`. Returns `None`
/// if the curve has no overlapping windows.
pub fn peak_in_range(curve: &[RmsWindow], start_secs: f64, end_secs: f64) -> Option<&RmsWindow> {
    curve
        .iter()
        .filter(|w| w.start_secs >= start_secs && w.start_secs < end_secs)
        .max_by(|a, b| {
            a.rms_db
                .partial_cmp(&b.rms_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ---------------------------------------------------------------------------
// F0 (fundamental frequency) via aubio CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct F0Sample {
    /// Time in seconds.
    pub time_secs: f64,
    /// Detected pitch in Hz. 0.0 means unvoiced / no pitch detected.
    pub pitch_hz: f64,
}

/// Aggregate F0 statistics over a time range.
#[derive(Debug, Clone, PartialEq)]
pub struct F0Stats {
    pub mean_hz: f64,
    pub variance_hz2: f64,
    pub peak_hz: f64,
}

/// Extract per-frame F0 (pitch) from an audio file using the `aubio pitch` CLI.
///
/// Returns an empty `Vec` (not an error) if `aubio` is not on PATH, so callers
/// can treat F0 as a best-effort signal.
pub async fn f0_curve(
    aubio_bin: &str,
    media_path: &Path,
) -> Vec<F0Sample> {
    let output = Command::new(aubio_bin)
        .arg("pitch")
        .arg("-i")
        .arg(media_path)
        .args(["-p", "yin"])
        .args(["-B", "2048"])
        .args(["-H", "512"])
        .output()
        .await;

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            tracing::warn!(
                status = %o.status,
                "aubio pitch exited with error; skipping F0 features"
            );
            return Vec::new();
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                tracing::debug!("aubio not found on PATH; skipping F0 features");
            } else {
                tracing::warn!(error = %e, "aubio pitch failed; skipping F0 features");
            }
            return Vec::new();
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_aubio_pitch(&stdout)
}

/// Compute F0 statistics over the voiced samples in `[start_secs, end_secs)`.
/// Unvoiced frames (pitch_hz == 0) are excluded. Returns `None` if no voiced
/// frames fall in the range.
pub fn f0_stats_in_range(
    curve: &[F0Sample],
    start_secs: f64,
    end_secs: f64,
) -> Option<F0Stats> {
    let voiced: Vec<f64> = curve
        .iter()
        .filter(|s| s.time_secs >= start_secs && s.time_secs < end_secs && s.pitch_hz > 0.0)
        .map(|s| s.pitch_hz)
        .collect();

    if voiced.is_empty() {
        return None;
    }

    let n = voiced.len() as f64;
    let mean = voiced.iter().sum::<f64>() / n;
    let variance = voiced.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let peak = voiced
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    Some(F0Stats {
        mean_hz: mean,
        variance_hz2: variance,
        peak_hz: peak,
    })
}

fn parse_aubio_pitch(stdout: &str) -> Vec<F0Sample> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // aubio pitch outputs: <time_secs>\t<pitch_hz>
        let mut parts = line.split_whitespace();
        let time = parts.next().and_then(|s| s.parse::<f64>().ok());
        let pitch = parts.next().and_then(|s| s.parse::<f64>().ok());
        if let (Some(t), Some(p)) = (time, pitch) {
            if t.is_finite() && p.is_finite() {
                out.push(F0Sample {
                    time_secs: t,
                    pitch_hz: p.max(0.0),
                });
            }
        }
    }
    out
}

/// Mean RMS (in dBFS) over the windows that overlap `[start_secs, end_secs)`.
pub fn mean_in_range(curve: &[RmsWindow], start_secs: f64, end_secs: f64) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for w in curve {
        if w.start_secs >= start_secs && w.start_secs < end_secs && w.rms_db.is_finite() {
            sum += w.rms_db;
            count += 1;
        }
    }
    (count > 0).then(|| sum / count as f64)
}

fn parse_astats_metadata(stdout: &str) -> Vec<RmsWindow> {
    // ametadata=print interleaves frame headers with key=value metadata lines:
    //   frame:0    pts:0       pts_time:0
    //   lavfi.astats.Overall.RMS_level=-23.456789
    //   ...
    //   frame:1    pts:44100   pts_time:1.0
    //   lavfi.astats.Overall.RMS_level=-25.123456
    //
    // We carry the most-recent pts_time as the window start and emit a row each time
    // we see a fresh RMS_level value. (Multiple per-channel RMS_level lines may appear;
    // we use Overall when present, otherwise the first level seen for that frame.)
    let pts_re = Regex::new(r"pts_time:(-?\d+(?:\.\d+)?)").expect("static regex");
    let overall_re =
        Regex::new(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+(?:\.\d+)?|-?inf)").expect("static");
    let channel_re =
        Regex::new(r"lavfi\.astats\.\d+\.RMS_level=(-?\d+(?:\.\d+)?|-?inf)").expect("static");

    #[derive(Default)]
    struct PendingFrame {
        pts: Option<f64>,
        overall: Option<f64>,
        first_channel: Option<f64>,
    }
    impl PendingFrame {
        fn flush(&self, out: &mut Vec<RmsWindow>) {
            if let (Some(pts), Some(rms)) = (self.pts, self.overall.or(self.first_channel))
                && rms.is_finite()
            {
                out.push(RmsWindow {
                    start_secs: pts.max(0.0),
                    rms_db: rms,
                });
            }
        }
    }

    let mut out = Vec::new();
    let mut pending = PendingFrame::default();

    for line in stdout.lines() {
        if let Some(c) = pts_re.captures(line) {
            // New frame header — flush previous before starting.
            pending.flush(&mut out);
            pending = PendingFrame::default();
            pending.pts = c.get(1).and_then(|m| parse_db(m.as_str()));
            continue;
        }
        if let Some(c) = overall_re.captures(line) {
            pending.overall = c.get(1).and_then(|m| parse_db(m.as_str()));
            continue;
        }
        if pending.first_channel.is_none()
            && let Some(c) = channel_re.captures(line)
        {
            pending.first_channel = c.get(1).and_then(|m| parse_db(m.as_str()));
        }
    }
    pending.flush(&mut out);

    out.sort_by(|a, b| {
        a.start_secs
            .partial_cmp(&b.start_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn parse_db(s: &str) -> Option<f64> {
    // ffmpeg emits "-inf" for true silence; we drop those at the call site by .is_finite().
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("-inf") {
        Some(f64::NEG_INFINITY)
    } else if trimmed.eq_ignore_ascii_case("inf") {
        Some(f64::INFINITY)
    } else {
        trimmed.parse::<f64>().ok()
    }
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
    fn parses_overall_rms_metadata() {
        let stdout = "\
frame:0    pts:0       pts_time:0
lavfi.astats.Overall.RMS_level=-23.456789
lavfi.astats.1.RMS_level=-23.5
frame:1    pts:44100   pts_time:1.0
lavfi.astats.Overall.RMS_level=-25.123456
frame:2    pts:88200   pts_time:2.0
lavfi.astats.Overall.RMS_level=-inf
";
        let curve = parse_astats_metadata(stdout);
        assert_eq!(curve.len(), 2, "expected 2 finite windows, got {curve:?}");
        assert!((curve[0].start_secs - 0.0).abs() < 1e-6);
        assert!((curve[0].rms_db - -23.456789).abs() < 1e-4);
        assert!((curve[1].start_secs - 1.0).abs() < 1e-6);
        assert!((curve[1].rms_db - -25.123456).abs() < 1e-4);
    }

    #[test]
    fn falls_back_to_first_channel_when_no_overall() {
        let stdout = "\
frame:0    pts:0       pts_time:0
lavfi.astats.1.RMS_level=-20.0
lavfi.astats.2.RMS_level=-21.0
";
        let curve = parse_astats_metadata(stdout);
        assert_eq!(curve.len(), 1);
        assert!((curve[0].rms_db - -20.0).abs() < 1e-6);
    }

    #[test]
    fn peak_and_mean_helpers() {
        let curve = vec![
            RmsWindow {
                start_secs: 0.0,
                rms_db: -30.0,
            },
            RmsWindow {
                start_secs: 1.0,
                rms_db: -10.0,
            },
            RmsWindow {
                start_secs: 2.0,
                rms_db: -20.0,
            },
            RmsWindow {
                start_secs: 3.0,
                rms_db: -40.0,
            },
        ];
        let peak = peak_in_range(&curve, 0.5, 2.5).unwrap();
        assert!((peak.rms_db - -10.0).abs() < 1e-6);

        let mean = mean_in_range(&curve, 0.0, 3.0).unwrap();
        // mean of -30, -10, -20
        assert!((mean - (-20.0)).abs() < 1e-6);

        assert!(peak_in_range(&curve, 100.0, 200.0).is_none());
        assert!(mean_in_range(&curve, 100.0, 200.0).is_none());
    }

    #[test]
    fn parses_aubio_pitch_output() {
        let stdout = "\
0.000000 220.500000
0.032000 221.300000
0.064000 0.000000
0.096000 440.100000
";
        let curve = parse_aubio_pitch(stdout);
        assert_eq!(curve.len(), 4);
        assert!((curve[0].time_secs - 0.0).abs() < 1e-6);
        assert!((curve[0].pitch_hz - 220.5).abs() < 1e-3);
        assert!((curve[2].pitch_hz - 0.0).abs() < 1e-6, "unvoiced frame");
        assert!((curve[3].pitch_hz - 440.1).abs() < 1e-3);
    }

    #[test]
    fn f0_stats_computes_mean_variance_peak() {
        let curve = vec![
            F0Sample { time_secs: 0.0, pitch_hz: 100.0 },
            F0Sample { time_secs: 0.5, pitch_hz: 200.0 },
            F0Sample { time_secs: 1.0, pitch_hz: 0.0 },   // unvoiced, excluded
            F0Sample { time_secs: 1.5, pitch_hz: 300.0 },
            F0Sample { time_secs: 5.0, pitch_hz: 999.0 },  // outside range
        ];
        let stats = f0_stats_in_range(&curve, 0.0, 2.0).unwrap();
        // voiced in [0,2): 100, 200, 300 => mean=200, var=6666.67, peak=300
        assert!((stats.mean_hz - 200.0).abs() < 1e-6);
        assert!((stats.variance_hz2 - 6666.666666).abs() < 1.0);
        assert!((stats.peak_hz - 300.0).abs() < 1e-6);
    }

    #[test]
    fn f0_stats_none_when_no_voiced_frames() {
        let curve = vec![
            F0Sample { time_secs: 0.5, pitch_hz: 0.0 },
        ];
        assert!(f0_stats_in_range(&curve, 0.0, 2.0).is_none());
    }

    #[test]
    fn f0_stats_none_when_out_of_range() {
        let curve = vec![
            F0Sample { time_secs: 5.0, pitch_hz: 440.0 },
        ];
        assert!(f0_stats_in_range(&curve, 0.0, 2.0).is_none());
    }

    #[tokio::test]
    async fn f0_curve_returns_empty_when_aubio_missing() {
        // Use a nonexistent binary name to verify graceful fallback.
        let result = f0_curve("__nonexistent_aubio_binary__", Path::new("/dev/null")).await;
        assert!(result.is_empty(), "should return empty vec when aubio not found");
    }

    #[tokio::test]
    async fn rms_curve_distinguishes_silence_from_tone() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let wav = dir.path().join("mixed.wav");

        // 1s silence + 1s loud tone + 1s silence + 1s loud tone = 4s
        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:r=16000:d=1"])
            .args(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:r=16000:d=1"])
            .args([
                "-filter_complex",
                "[0][1][2][3]concat=n=4:v=0:a=1[a]",
            ])
            .args(["-map", "[a]"])
            .arg(&wav)
            .status()
            .await?;
        anyhow::ensure!(status.success());

        let curve = rms_curve("ffmpeg", &wav, 1.0).await?;
        assert!(!curve.is_empty(), "expected at least one rms window");

        // The mean over the loud second (1-2) should be meaningfully louder
        // than the mean over the silent second (0-1).
        let loud = mean_in_range(&curve, 1.0, 2.0);
        let silent = mean_in_range(&curve, 0.0, 1.0);

        // If ffmpeg emitted -inf for true silence we won't have a silent mean — that's fine,
        // the assertion only fires when both ranges produced finite values.
        if let (Some(loud_db), Some(silent_db)) = (loud, silent) {
            assert!(
                loud_db > silent_db + 10.0,
                "loud window ({loud_db} dB) should be >10dB louder than silent ({silent_db} dB)"
            );
        } else {
            // Otherwise just check we got at least one loud window above the silent floor.
            let max = curve
                .iter()
                .map(|w| w.rms_db)
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(max > -30.0, "expected a loud window above -30dB, got peak {max}");
        }
        Ok(())
    }
}
