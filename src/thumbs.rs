use futures_util::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};

use crate::{ai_pipeline::ThumbnailMoment, media, mime::Attachment};

#[allow(clippy::too_many_arguments)]
pub async fn generate_thumbnails(
    ffmpeg: &str,
    video_path: &std::path::Path,
    thumbs_dir: &std::path::Path,
    moments: &[ThumbnailMoment],
    total_duration_secs: f64,
    thumbnail_window_secs: u64,
    thumbnail_count: usize,
    thumbnail_max_height: u32,
    thumbnail_ffmpeg_concurrency: usize,
    progress: Option<ProgressBar>,
) -> Vec<Attachment> {
    let window = thumbnail_window_secs as f64;
    let total = total_duration_secs.max(0.0);
    // Avoid requesting a frame right at EOF; many files return "nothing was encoded".
    let eof_epsilon_secs = 0.25;
    let last_frame_ts = if total.is_finite() && total > 0.0 {
        (total - eof_epsilon_secs).max(0.0)
    } else {
        0.0
    };

    // Build screenshot tasks.
    // Prefer 1 shot per moment when we already have enough moments; it halves ffmpeg work vs center+alt.
    // Add a small buffer of extra tasks to survive occasional screenshot failures/empty outputs.
    let mut tasks: Vec<(std::path::PathBuf, f64)> = Vec::new();
    if thumbnail_count == 0 {
        return Vec::new();
    }

    let buffer = (thumbnail_count / 4).max(2); // small safety margin
    let max_tasks = thumbnail_count.saturating_add(buffer);

    if moments.len() >= thumbnail_count {
        for (i, m) in moments.iter().take(thumbnail_count).enumerate() {
            let center = m.center_seconds.max(0.0).min(last_frame_ts);
            let out_path = thumbs_dir.join(format!("thumb_{i:02}_{sec:.0}.jpg", sec = center));
            tasks.push((out_path, center));
        }
    } else {
        for (i, m) in moments.iter().enumerate() {
            if tasks.len() >= max_tasks {
                break;
            }
            let center = m.center_seconds.max(0.0).min(last_frame_ts);
            let out_path = thumbs_dir.join(format!("thumb_{i:02}_{sec:.0}.jpg", sec = center));
            tasks.push((out_path, center));

            if tasks.len() >= max_tasks {
                break;
            }
            let alt = (center + (window / 2.0)).min(last_frame_ts);
            let out_path2 = thumbs_dir.join(format!("thumb_{i:02}_alt_{sec:.0}.jpg", sec = alt));
            tasks.push((out_path2, alt));
        }
    }

    if tasks.is_empty() {
        return Vec::new();
    }

    let pb = progress.unwrap_or_else(ProgressBar::hidden);
    pb.set_length(tasks.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} thumbnails {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb.set_message("rendering");

    let concurrency = thumbnail_ffmpeg_concurrency.max(1);

    let results = stream::iter(tasks.into_iter())
        .map(|(out_path, at)| {
            let ffmpeg = ffmpeg.to_string();
            let video_path = video_path.to_path_buf();
            let pb = pb.clone();
            async move {
                let _ = media::screenshot_jpeg(
                    &ffmpeg,
                    &video_path,
                    at,
                    &out_path,
                    thumbnail_max_height,
                )
                .await;
                let bytes = tokio::fs::read(&out_path)
                    .await
                    .ok()
                    .filter(|b| !b.is_empty());
                pb.inc(1);
                bytes.map(|bytes| Attachment {
                    filename: out_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("thumb.jpg")
                        .to_string(),
                    content_type: "image/jpeg".to_string(),
                    bytes,
                })
            }
        })
        .buffer_unordered(concurrency)
        .collect::<Vec<_>>()
        .await;

    pb.finish_with_message("thumbnails complete");

    let mut attachments: Vec<Attachment> = results.into_iter().flatten().collect();
    attachments.sort_by(|a, b| a.filename.cmp(&b.filename));
    attachments.truncate(thumbnail_count);
    attachments
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use tokio::process::Command;

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

    #[tokio::test]
    async fn thumbnail_generation_smoke() -> anyhow::Result<()> {
        if !tool_ok("ffmpeg").await || !tool_ok("ffprobe").await {
            eprintln!("skipping: ffmpeg/ffprobe not available on PATH");
            return Ok(());
        }

        let dir = tempfile::tempdir().context("tempdir")?;
        let video_path = dir.path().join("test.mp4");

        // Generate a tiny synthetic video with audio.
        let status = Command::new("ffmpeg")
            .arg("-y")
            .args(["-f", "lavfi", "-i", "testsrc=size=640x360:rate=30"])
            .args(["-f", "lavfi", "-i", "sine=frequency=1000"])
            .args(["-t", "6"])
            .args(["-c:v", "libx264"])
            .args(["-pix_fmt", "yuv420p"])
            .args(["-c:a", "aac"])
            .arg("-shortest")
            .arg(&video_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .context("ffmpeg generate test video")?;
        anyhow::ensure!(status.success(), "ffmpeg failed to generate test video");

        let audio_path = dir.path().join("audio.m4a");
        media::extract_audio_m4a("ffmpeg", &video_path, &audio_path)
            .await
            .context("extract_audio_m4a")?;

        let dur = media::duration_secs("ffprobe", &audio_path)
            .await
            .context("duration_secs")?;
        anyhow::ensure!(dur > 1.0, "unexpected duration {dur}");

        let chunks_dir = dir.path().join("chunks");
        let chunks = media::segment_audio("ffmpeg", &audio_path, &chunks_dir, 2)
            .await
            .context("segment_audio")?;
        anyhow::ensure!(chunks.len() >= 2, "expected multiple chunks");

        let thumbs_dir = dir.path().join("thumbs");
        tokio::fs::create_dir_all(&thumbs_dir).await.ok();

        let moments = vec![
            ThumbnailMoment {
                center_seconds: 1.0,
                reason: "test".to_string(),
            },
            ThumbnailMoment {
                center_seconds: 3.0,
                reason: "test".to_string(),
            },
            ThumbnailMoment {
                center_seconds: 5.0,
                reason: "test".to_string(),
            },
        ];

        let atts = generate_thumbnails(
            "ffmpeg",
            &video_path,
            &thumbs_dir,
            &moments,
            6.0,
            2,
            3,
            720,
            4,
            None,
        )
        .await;

        anyhow::ensure!(atts.len() == 3, "expected 3 thumbnails, got {}", atts.len());
        for a in &atts {
            anyhow::ensure!(!a.bytes.is_empty(), "empty attachment bytes");
        }

        Ok(())
    }
}
