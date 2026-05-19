use anyhow::Context;
use regex::Regex;
use std::path::Path;
use tokio::process::Command;

/// Detect shot/scene boundaries in a video using ffmpeg's `select='gt(scene,T)'` filter
/// plus `showinfo`. Returns a sorted vector of timestamps (seconds) at which a shot change
/// starts. No external ML — pure ffmpeg.
///
/// `threshold` is in [0, 1]. 0.30 = sensitive (cuts on subtle changes), 0.40 = default,
/// 0.50+ = conservative (only hard cuts).
pub async fn detect_shots(
    ffmpeg: &str,
    video_path: &Path,
    threshold: f64,
) -> anyhow::Result<Vec<f64>> {
    let threshold = threshold.clamp(0.0, 1.0);
    let vf = format!("select='gt(scene,{threshold})',showinfo");

    // Discard everything except the showinfo stderr; drop non-video streams to keep ffmpeg cheap.
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-nostats"])
        .args(["-an", "-sn", "-dn"])
        .arg("-i")
        .arg(video_path)
        .args(["-vf", &vf])
        .args(["-f", "null", "-"])
        .output()
        .await
        .with_context(|| format!("run ffmpeg scenedetect on {}", video_path.display()))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg scenedetect failed: {} {}", output.status, err);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(parse_showinfo(&stderr))
}

/// Parse `pts_time:<float>` markers out of ffmpeg showinfo stderr lines.
fn parse_showinfo(stderr: &str) -> Vec<f64> {
    // Showinfo emits one line per passed frame, e.g.:
    // [Parsed_showinfo_1 @ 0x...] n: 0 pts: 256 pts_time:0.106667 fmt:yuv420p ...
    let re = Regex::new(r"pts_time:(\d+(?:\.\d+)?)").expect("static regex");
    let mut out: Vec<f64> = re
        .captures_iter(stderr)
        .filter_map(|c| c.get(1).and_then(|m| m.as_str().parse::<f64>().ok()))
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    out
}

/// Snap a target timestamp to the nearest shot boundary within `max_drift_secs`.
/// Returns the original target if no boundary is close enough.
/// Use this to avoid cutting mid-shot when the LLM picks a clip start/end.
pub fn snap_to_shot(target_secs: f64, shots: &[f64], max_drift_secs: f64) -> f64 {
    if shots.is_empty() {
        return target_secs;
    }
    let mut best = target_secs;
    let mut best_drift = max_drift_secs;
    for &cut in shots {
        let drift = (cut - target_secs).abs();
        if drift <= best_drift {
            best_drift = drift;
            best = cut;
        }
    }
    best
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
    fn parses_pts_time_lines() {
        let stderr = "[Parsed_showinfo_1 @ 0x0] n: 0 pts: 256 pts_time:0.106667 fmt:yuv420p\n\
                      garbage line with no pts\n\
                      [Parsed_showinfo_1 @ 0x0] n: 1 pts:12800 pts_time:5.123456 fmt:yuv420p\n\
                      [Parsed_showinfo_1 @ 0x0] n: 2 pts:12800 pts_time:5.123456 dup\n";
        let cuts = parse_showinfo(stderr);
        assert_eq!(cuts.len(), 2, "deduped: {:?}", cuts);
        assert!((cuts[0] - 0.106667).abs() < 1e-4);
        assert!((cuts[1] - 5.123456).abs() < 1e-4);
    }

    #[test]
    fn parses_empty_stderr() {
        assert!(parse_showinfo("").is_empty());
        assert!(parse_showinfo("no markers here").is_empty());
    }

    #[test]
    fn snap_to_shot_within_drift() {
        let shots = vec![1.0, 5.5, 10.0, 30.0];
        assert_eq!(snap_to_shot(5.6, &shots, 0.5), 5.5);
        assert_eq!(snap_to_shot(5.6, &shots, 0.05), 5.6); // no snap — too far
        assert_eq!(snap_to_shot(0.0, &shots, 2.0), 1.0);
        assert_eq!(snap_to_shot(100.0, &[], 5.0), 100.0); // empty shots
    }

    #[tokio::test]
    async fn detect_shots_finds_color_transitions() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let video = dir.path().join("test.mp4");

        // Synthesize a 6s video with three distinct colored segments (2s each).
        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=red:size=320x180:duration=2:rate=15",
            ])
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=green:size=320x180:duration=2:rate=15",
            ])
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=blue:size=320x180:duration=2:rate=15",
            ])
            .args(["-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]"])
            .args(["-map", "[v]"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&video)
            .status()
            .await?;
        anyhow::ensure!(status.success(), "ffmpeg test video generation failed");

        // Sensitive threshold — verifies the ffmpeg wiring catches at least one obvious cut.
        // The scene filter's score for synthetic color transitions varies by ffmpeg build,
        // so don't assert an exact count; assert the plumbing works end-to-end.
        let cuts = detect_shots("ffmpeg", &video, 0.1).await?;
        assert!(
            !cuts.is_empty(),
            "expected at least one cut in 3-color test video, got none"
        );
        // Every cut should land near a real transition (2.0s or 4.0s).
        for &c in &cuts {
            let near_red_to_green = (c - 2.0).abs() < 0.5;
            let near_green_to_blue = (c - 4.0).abs() < 0.5;
            assert!(
                near_red_to_green || near_green_to_blue,
                "cut at {c}s doesn't match either expected transition"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn detect_shots_constant_video_has_no_cuts() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await {
            eprintln!("skipping: ffmpeg not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let video = dir.path().join("flat.mp4");

        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-hide_banner", "-loglevel", "error"])
            .args([
                "-f",
                "lavfi",
                "-i",
                "color=red:size=320x180:duration=3:rate=15",
            ])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&video)
            .status()
            .await?;
        anyhow::ensure!(status.success(), "ffmpeg flat video generation failed");

        let cuts = detect_shots("ffmpeg", &video, 0.3).await?;
        assert!(
            cuts.is_empty(),
            "expected no cuts on constant video, got {cuts:?}"
        );
        Ok(())
    }
}
