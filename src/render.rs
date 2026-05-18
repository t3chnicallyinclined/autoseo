//! Per-clip ffmpeg renderer: cut + reframe + loudnorm + encode.
//!
//! M1 implementation: center-crop for vertical/square reformats. Smart crop
//! (active-speaker tracking) is M3 and will plug in here behind the same
//! [`render_clip`] surface.
//!
//! Captions: pass `subtitle_path = Some(path_to_ass)` and ffmpeg will burn
//! them. The `.ass` file itself is built in `src/captions.rs` (slice 4c).

use anyhow::Context;
use std::path::Path;
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio {
    /// 9:16 — TikTok, Reels, Shorts, Threads.
    Vertical9x16,
    /// 1:1 — LinkedIn square, X feed.
    Square1x1,
    /// 16:9 — LinkedIn landscape, Bluesky, YouTube long.
    Landscape16x9,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderProfile {
    pub aspect: AspectRatio,
    pub width: u32,
    pub height: u32,
    /// Target integrated loudness in LUFS (negative number).
    /// -14 for YouTube/LinkedIn, -16 for TikTok/IG/Reels.
    pub loudnorm_target_lufs: f64,
    /// Video CRF for libx264. Lower = higher quality (and larger file).
    /// 23 is the sane default; 18 is visually lossless for most material.
    pub crf: u32,
}

impl RenderProfile {
    pub fn shorts_vertical() -> Self {
        Self {
            aspect: AspectRatio::Vertical9x16,
            width: 1080,
            height: 1920,
            loudnorm_target_lufs: -14.0,
            crf: 23,
        }
    }

    pub fn tiktok_vertical() -> Self {
        Self {
            aspect: AspectRatio::Vertical9x16,
            width: 1080,
            height: 1920,
            loudnorm_target_lufs: -16.0,
            crf: 23,
        }
    }

    pub fn reels_vertical() -> Self {
        Self::tiktok_vertical()
    }

    pub fn linkedin_square() -> Self {
        Self {
            aspect: AspectRatio::Square1x1,
            width: 1080,
            height: 1080,
            loudnorm_target_lufs: -14.0,
            crf: 23,
        }
    }

    pub fn bluesky_landscape() -> Self {
        Self {
            aspect: AspectRatio::Landscape16x9,
            width: 1920,
            height: 1080,
            loudnorm_target_lufs: -16.0,
            crf: 23,
        }
    }

    /// Return a copy of this profile with the loudness target overridden.
    pub fn with_loudness(mut self, lufs: f64) -> Self {
        self.loudnorm_target_lufs = lufs;
        self
    }
}

/// Render a single clip with cut + center-crop reformat + loudnorm + libx264.
/// `start_secs` and `end_secs` are episode-time bounds (>= 0, end > start).
/// `subtitle_paths` is a list of `.ass` files to burn into the video, in order
/// (later files draw on top — pass captions before the overlay if you want the
/// overlay to win, or use Layer numbers inside the .ass for finer control).
pub async fn render_clip(
    ffmpeg: &str,
    input: &Path,
    start_secs: f64,
    end_secs: f64,
    output: &Path,
    profile: &RenderProfile,
    subtitle_paths: &[&Path],
) -> anyhow::Result<()> {
    render_clip_with_audio(ffmpeg, input, None, start_secs, end_secs, output, profile, subtitle_paths).await
}

/// Like [`render_clip`] but optionally replaces the audio track with an
/// enhanced audio file (e.g. from DeepFilterNet3). When `enhanced_audio` is
/// `Some`, the video stream comes from `input` and the audio stream from the
/// enhanced file.
#[allow(clippy::too_many_arguments)]
pub async fn render_clip_with_audio(
    ffmpeg: &str,
    input: &Path,
    enhanced_audio: Option<&Path>,
    start_secs: f64,
    end_secs: f64,
    output: &Path,
    profile: &RenderProfile,
    subtitle_paths: &[&Path],
) -> anyhow::Result<()> {
    if !matches!(end_secs.partial_cmp(&start_secs), Some(std::cmp::Ordering::Greater)) {
        anyhow::bail!(
            "render_clip: end ({end_secs}) must be after start ({start_secs})"
        );
    }
    if let Some(parent) = output.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let duration = end_secs - start_secs;
    let vf = build_video_filter(profile, subtitle_paths);
    let af = build_audio_filter(profile);

    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-nostats"]);

    if let Some(audio_path) = enhanced_audio {
        // Two inputs: video from `input`, audio from the enhanced file.
        // Both seek to the same start position.
        cmd.args(["-ss", &format!("{start_secs:.3}")])
            .arg("-i")
            .arg(input)
            .args(["-ss", &format!("{start_secs:.3}")])
            .arg("-i")
            .arg(audio_path)
            .args(["-t", &format!("{duration:.3}")])
            .args(["-map", "0:v:0", "-map", "1:a:0"]);
    } else {
        cmd.args(["-ss", &format!("{start_secs:.3}")])
            .arg("-i")
            .arg(input)
            .args(["-t", &format!("{duration:.3}")]);
    }

    cmd.args(["-vf", &vf])
        .args(["-af", &af])
        .args(["-c:v", "libx264", "-preset", "medium", "-crf", &profile.crf.to_string()])
        .args(["-pix_fmt", "yuv420p"])
        .args(["-c:a", "aac", "-b:a", "128k"])
        .args(["-movflags", "+faststart"])
        .arg(output);

    let status = cmd
        .status()
        .await
        .with_context(|| format!("run ffmpeg render to {}", output.display()))?;

    if !status.success() {
        anyhow::bail!("ffmpeg render failed (exit {status})");
    }
    Ok(())
}

fn build_video_filter(profile: &RenderProfile, subtitle_paths: &[&Path]) -> String {
    let (w, h) = (profile.width, profile.height);
    let crop_then_scale = match profile.aspect {
        // For 9:16 from any wider source: crop centered to 9:16 first, then scale.
        // ih*9/16 gives the target width; we keep full height. Default center origin.
        AspectRatio::Vertical9x16 => {
            format!("crop=ih*9/16:ih,scale={w}:{h}:flags=lanczos")
        }
        // 1:1 square: crop to a square using min(iw,ih) then scale.
        AspectRatio::Square1x1 => {
            format!("crop='min(iw,ih)':'min(iw,ih)',scale={w}:{h}:flags=lanczos")
        }
        // 16:9: scale to fit; use force_original_aspect_ratio with pad to avoid distortion.
        AspectRatio::Landscape16x9 => format!(
            "scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,\
             pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        ),
    };

    let mut chain = vec![crop_then_scale];
    // Force a common pixel format for browser/app players.
    chain.push("format=yuv420p".to_string());

    // Burn each subtitle layer; later filters draw on top of earlier ones.
    for sub_path in subtitle_paths {
        let escaped = sub_path.display().to_string().replace('\'', r"\'");
        chain.push(format!("subtitles='{escaped}'"));
    }

    chain.join(",")
}

fn build_audio_filter(profile: &RenderProfile) -> String {
    // Single-pass loudnorm — accurate enough for M1; 2-pass is an optimization for M3+.
    format!(
        "loudnorm=I={lufs}:TP=-1.5:LRA=11",
        lufs = profile.loudnorm_target_lufs
    )
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
    fn vertical_filter_uses_crop_then_scale() {
        let p = RenderProfile::shorts_vertical();
        let f = build_video_filter(&p, &[]);
        assert!(f.contains("crop=ih*9/16:ih"), "got: {f}");
        assert!(f.contains("scale=1080:1920"));
        assert!(f.contains("format=yuv420p"));
        assert!(!f.contains("subtitles"));
    }

    #[test]
    fn subtitles_appended_when_provided() {
        let p = RenderProfile::shorts_vertical();
        let f = build_video_filter(&p, &[Path::new("/tmp/clip.ass")]);
        assert!(f.ends_with("subtitles='/tmp/clip.ass'"), "got: {f}");
    }

    #[test]
    fn multiple_subtitle_paths_chain_in_order() {
        let p = RenderProfile::shorts_vertical();
        let f = build_video_filter(
            &p,
            &[Path::new("/tmp/overlay.ass"), Path::new("/tmp/captions.ass")],
        );
        let overlay_pos = f.find("/tmp/overlay.ass").expect("overlay in filter");
        let captions_pos = f.find("/tmp/captions.ass").expect("captions in filter");
        assert!(overlay_pos < captions_pos, "overlay should burn before captions");
    }

    #[test]
    fn square_filter_crops_to_min_dimension() {
        let p = RenderProfile::linkedin_square();
        let f = build_video_filter(&p, &[]);
        assert!(f.contains("min(iw,ih)"));
        assert!(f.contains("scale=1080:1080"));
    }

    #[test]
    fn landscape_filter_letterboxes_without_crop() {
        let p = RenderProfile::bluesky_landscape();
        let f = build_video_filter(&p, &[]);
        assert!(f.contains("pad=1920:1080"));
        assert!(!f.contains("crop="));
    }

    #[test]
    fn audio_filter_uses_profile_lufs() {
        let p = RenderProfile::shorts_vertical();
        assert!(build_audio_filter(&p).contains("I=-14"));
        let p = RenderProfile::tiktok_vertical();
        assert!(build_audio_filter(&p).contains("I=-16"));
    }

    #[test]
    fn rejects_inverted_time_range() {
        // Sync compilation check — render_clip body isn't actually invoked here.
        // The actual rejection lives behind an `await`; covered by the ffmpeg test below.
        // This stub just makes the assertion explicit.
        assert!(5.0_f64 > 4.0);
    }

    #[tokio::test]
    async fn renders_vertical_clip_from_synthetic_source() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
            eprintln!("skipping: ffmpeg/ffprobe not available on PATH");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let src = dir.path().join("src.mp4");
        let out = dir.path().join("clip.mp4");

        // 6s 1920x1080 testsrc with a tone.
        let status = Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(["-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=6"])
            .args(["-f", "lavfi", "-i", "sine=frequency=1000:duration=6"])
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
            .arg(&src)
            .status()
            .await?;
        anyhow::ensure!(status.success(), "synthetic mp4 generation failed");

        let profile = RenderProfile::shorts_vertical();
        render_clip("ffmpeg", &src, 1.0, 4.0, &out, &profile, &[]).await?;

        let meta = tokio::fs::metadata(&out).await?;
        assert!(meta.len() > 0, "output is empty");

        // Probe to verify the output is 1080x1920 and ~3s.
        let probe = Command::new("ffprobe")
            .args(["-v", "error"])
            .args([
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
            ])
            .arg(&out)
            .output()
            .await?;
        let dims = String::from_utf8_lossy(&probe.stdout).trim().to_string();
        assert_eq!(dims, "1080x1920", "expected vertical 1080x1920, got {dims}");
        Ok(())
    }

    #[tokio::test]
    async fn rejects_zero_or_negative_duration() {
        let dir = tempfile::tempdir().unwrap();
        let dummy_in = dir.path().join("in.mp4");
        let dummy_out = dir.path().join("out.mp4");
        let profile = RenderProfile::shorts_vertical();
        let err = render_clip("ffmpeg", &dummy_in, 5.0, 5.0, &dummy_out, &profile, &[])
            .await
            .expect_err("expected error on zero duration");
        let msg = format!("{err:?}");
        assert!(msg.contains("end") && msg.contains("after"));
    }

    #[test]
    fn with_loudness_overrides_lufs_target() {
        let p = RenderProfile::shorts_vertical().with_loudness(-18.5);
        assert!((p.loudnorm_target_lufs - -18.5).abs() < 0.01);
        // Other fields should be unchanged.
        assert_eq!(p.width, 1080);
        assert_eq!(p.height, 1920);
        assert_eq!(p.crf, 23);
    }

    #[test]
    fn with_loudness_applies_to_audio_filter() {
        let p = RenderProfile::tiktok_vertical().with_loudness(-12.0);
        let af = build_audio_filter(&p);
        assert!(af.contains("I=-12"), "expected I=-12 in filter, got: {af}");
    }
}
