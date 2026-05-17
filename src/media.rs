use anyhow::Context;
use serde::{Deserialize, Deserializer, de::Error as DeError};
use tokio::process::Command;

pub async fn extract_audio_m4a(
    ffmpeg: &str,
    video_path: &std::path::Path,
    audio_path: &std::path::Path,
) -> anyhow::Result<()> {
    if let Some(parent) = audio_path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    // Extract a low-bitrate mono track; good enough for STT and cheap to upload.
    let status = Command::new(ffmpeg)
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .arg("-i")
        .arg(video_path)
        .args(["-vn", "-ac", "1", "-ar", "16000"])
        .args(["-c:a", "aac", "-b:a", "32k"])
        .arg(audio_path)
        .status()
        .await
        .with_context(|| format!("run ffmpeg extract audio from {}", video_path.display()))?;

    if !status.success() {
        anyhow::bail!("ffmpeg audio extract failed (exit {status})");
    }

    Ok(())
}

pub async fn transcode_audio_to_m4a(
    ffmpeg: &str,
    input_audio_path: &std::path::Path,
    audio_path: &std::path::Path,
) -> anyhow::Result<()> {
    if let Some(parent) = audio_path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    // Transcode arbitrary audio into a low-bitrate mono track; good enough for STT and cheap to upload.
    let status = Command::new(ffmpeg)
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .arg("-i")
        .arg(input_audio_path)
        .args(["-ac", "1", "-ar", "16000"])
        .args(["-c:a", "aac", "-b:a", "32k"])
        .arg(audio_path)
        .status()
        .await
        .with_context(|| {
            format!(
                "run ffmpeg transcode audio from {}",
                input_audio_path.display()
            )
        })?;

    if !status.success() {
        anyhow::bail!("ffmpeg audio transcode failed (exit {status})");
    }

    Ok(())
}

pub async fn segment_audio(
    ffmpeg: &str,
    audio_path: &std::path::Path,
    out_dir: &std::path::Path,
    segment_secs: u64,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    tokio::fs::create_dir_all(out_dir).await.ok();

    // Remove stale chunks from previous runs so we don't accidentally re-use them
    // when the new segmentation yields fewer files.
    let mut existing = tokio::fs::read_dir(out_dir).await?;
    while let Some(ent) = existing.next_entry().await? {
        let p = ent.path();
        let is_chunk = p
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.starts_with("chunk_"))
            .unwrap_or(false);
        let is_m4a = p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("m4a"))
            .unwrap_or(false);
        if is_chunk && is_m4a {
            let _ = tokio::fs::remove_file(&p).await;
        }
    }

    let pattern = out_dir.join("chunk_%05d.m4a");

    let status = Command::new(ffmpeg)
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .arg("-i")
        .arg(audio_path)
        .args(["-f", "segment"])
        .args(["-segment_time", &segment_secs.to_string()])
        .args(["-reset_timestamps", "1"])
        .args(["-c", "copy"])
        .arg(pattern)
        .status()
        .await
        .with_context(|| format!("run ffmpeg segment audio {}", audio_path.display()))?;

    if !status.success() {
        anyhow::bail!("ffmpeg segment failed (exit {status})");
    }

    let mut entries = tokio::fs::read_dir(out_dir).await?;
    let mut chunks = Vec::new();
    while let Some(ent) = entries.next_entry().await? {
        let p = ent.path();
        if p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("m4a"))
            .unwrap_or(false)
            && p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with("chunk_"))
                .unwrap_or(false)
        {
            chunks.push(p);
        }
    }
    chunks.sort();
    Ok(chunks)
}

pub async fn duration_secs(ffprobe: &str, media_path: &std::path::Path) -> anyhow::Result<f64> {
    let output = Command::new(ffprobe)
        .arg("-v")
        .arg("error")
        .args(["-show_entries", "format=duration"])
        .args(["-of", "json"])
        .arg(media_path)
        .output()
        .await
        .with_context(|| format!("run ffprobe duration {}", media_path.display()))?;

    if !output.status.success() {
        anyhow::bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let parsed: Probe = serde_json::from_slice(&output.stdout).context("parse ffprobe json")?;
    let dur = parsed
        .format
        .and_then(|f| f.duration)
        .context("ffprobe duration missing")?;
    Ok(dur)
}

pub async fn wav_duration_secs(path: &std::path::Path) -> anyhow::Result<f64> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || -> anyhow::Result<f64> {
        let reader = hound::WavReader::open(&path)
            .with_context(|| format!("open wav {}", path.display()))?;
        let spec = reader.spec();
        let frames = reader.duration() as f64;
        let sr = spec.sample_rate as f64;
        Ok(frames / sr)
    })
    .await
    .context("join wav_duration_secs")?
}

pub async fn segment_wav(
    wav_path: &std::path::Path,
    out_dir: &std::path::Path,
    segment_secs: u64,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let wav_path = wav_path.to_owned();
    let out_dir = out_dir.to_owned();

    tokio::fs::create_dir_all(&out_dir).await.ok();

    // Remove stale chunks from previous runs.
    let mut existing = tokio::fs::read_dir(&out_dir).await?;
    while let Some(ent) = existing.next_entry().await? {
        let p = ent.path();
        let is_chunk = p
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.starts_with("chunk_"))
            .unwrap_or(false);
        let is_wav = p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("wav"))
            .unwrap_or(false);
        if is_chunk && is_wav {
            let _ = tokio::fs::remove_file(&p).await;
        }
    }

    tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<std::path::PathBuf>> {
            let mut reader = hound::WavReader::open(&wav_path)
                .with_context(|| format!("open wav {}", wav_path.display()))?;
            let spec = reader.spec();
            let channels = spec.channels as u64;
            let samples_per_chunk = spec.sample_rate as u64 * segment_secs * channels;
            if samples_per_chunk == 0 {
                anyhow::bail!("invalid wav chunk size");
            }

            let mut out_paths = Vec::new();
            let mut chunk_idx: u64 = 0;
            let mut written_in_chunk: u64 = 0;

            let mut writer_i16: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = None;
            let mut writer_i32: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = None;
            let mut writer_f32: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = None;

            let open_new = |chunk_idx: u64,
                            out_paths: &mut Vec<std::path::PathBuf>|
             -> anyhow::Result<std::path::PathBuf> {
                let out_path = out_dir.join(format!("chunk_{chunk_idx:05}.wav"));
                out_paths.push(out_path.clone());
                Ok(out_path)
            };

            match spec.sample_format {
                hound::SampleFormat::Int => {
                    if spec.bits_per_sample <= 16 {
                        for s in reader.samples::<i16>() {
                            let sample = s.context("read wav sample(i16)")?;
                            if writer_i16.is_none() {
                                let out_path = open_new(chunk_idx, &mut out_paths)?;
                                writer_i16 =
                                    Some(hound::WavWriter::create(&out_path, spec).with_context(
                                        || format!("create wav chunk {}", out_path.display()),
                                    )?);
                            }
                            writer_i16
                                .as_mut()
                                .context("missing wav writer")?
                                .write_sample(sample)
                                .context("write wav sample")?;
                            written_in_chunk += 1;
                            if written_in_chunk >= samples_per_chunk {
                                writer_i16.take().unwrap().finalize().ok();
                                chunk_idx += 1;
                                written_in_chunk = 0;
                            }
                        }
                        if let Some(w) = writer_i16.take() {
                            w.finalize().ok();
                        }
                    } else {
                        for s in reader.samples::<i32>() {
                            let sample = s.context("read wav sample(i32)")?;
                            if writer_i32.is_none() {
                                let out_path = open_new(chunk_idx, &mut out_paths)?;
                                writer_i32 =
                                    Some(hound::WavWriter::create(&out_path, spec).with_context(
                                        || format!("create wav chunk {}", out_path.display()),
                                    )?);
                            }
                            writer_i32
                                .as_mut()
                                .context("missing wav writer")?
                                .write_sample(sample)
                                .context("write wav sample")?;
                            written_in_chunk += 1;
                            if written_in_chunk >= samples_per_chunk {
                                writer_i32.take().unwrap().finalize().ok();
                                chunk_idx += 1;
                                written_in_chunk = 0;
                            }
                        }
                        if let Some(w) = writer_i32.take() {
                            w.finalize().ok();
                        }
                    }
                }
                hound::SampleFormat::Float => {
                    for s in reader.samples::<f32>() {
                        let sample = s.context("read wav sample(f32)")?;
                        if writer_f32.is_none() {
                            let out_path = open_new(chunk_idx, &mut out_paths)?;
                            writer_f32 =
                                Some(hound::WavWriter::create(&out_path, spec).with_context(
                                    || format!("create wav chunk {}", out_path.display()),
                                )?);
                        }
                        writer_f32
                            .as_mut()
                            .context("missing wav writer")?
                            .write_sample(sample)
                            .context("write wav sample")?;
                        written_in_chunk += 1;
                        if written_in_chunk >= samples_per_chunk {
                            writer_f32.take().unwrap().finalize().ok();
                            chunk_idx += 1;
                            written_in_chunk = 0;
                        }
                    }
                    if let Some(w) = writer_f32.take() {
                        w.finalize().ok();
                    }
                }
            }

            // If we produced an empty last chunk file (possible when samples align exactly), drop it.
            out_paths.retain(|p| {
                std::fs::metadata(p)
                    .map(|m| m.len() > 44) // WAV header is 44 bytes
                    .unwrap_or(false)
            });
            out_paths.sort();
            Ok(out_paths)
    })
    .await
    .context("join segment_wav")?
}

#[derive(Debug, Deserialize)]
struct Probe {
    #[serde(default)]
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    #[serde(default, deserialize_with = "opt_number_from_str")]
    duration: Option<f64>,
}

fn opt_number_from_str<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum NumOrString {
        Num(f64),
        Str(String),
    }

    let value = Option::<NumOrString>::deserialize(deserializer)?;
    match value {
        Some(NumOrString::Num(n)) => Ok(Some(n)),
        Some(NumOrString::Str(s)) => {
            let parsed = s
                .trim()
                .parse::<f64>()
                .map_err(|e| DeError::custom(format!("parse duration '{s}': {e}")))?;
            Ok(Some(parsed))
        }
        None => Ok(None),
    }
}

pub async fn screenshot_jpeg(
    ffmpeg: &str,
    video_path: &std::path::Path,
    at_secs: f64,
    out_path: &std::path::Path,
    max_height: u32,
) -> anyhow::Result<()> {
    if let Some(parent) = out_path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let ts = format!("{at_secs:.3}");

    // If max_height is set, downscale while preserving aspect ratio.
    // Use -2 to ensure even width for yuv420-based pixel formats.
    let vf = if max_height > 0 {
        // High-quality downscale when requested.
        format!("scale=-2:{max_height}:flags=lanczos,format=yuvj420p")
    } else {
        // Truly no-resample: keep original size.
        // Prefer 4:4:4 JPEG for maximum fidelity; Gmail/clients generally render it fine.
        "format=yuvj444p".to_string()
    };

    let status = Command::new(ffmpeg)
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        // Reduce startup overhead and avoid probing/decoding non-video streams.
        .args(["-an", "-sn", "-dn"])
        .args(["-probesize", "32k", "-analyzeduration", "0"])
        .args(["-ss", &ts])
        .arg("-i")
        .arg(video_path)
        .args(["-frames:v", "1"])
        // Faster thumbnails: optional downscale + cheaper pixel format.
        // -q:v: lower is higher quality; 1 is near-lossless JPEG.
        .args(["-vf", &vf])
        .args(["-q:v", "1"])
        .arg(out_path)
        .status()
        .await
        .with_context(|| format!("run ffmpeg screenshot at {ts}s"))?;

    if !status.success() {
        anyhow::bail!("ffmpeg screenshot failed (exit {status})");
    }

    Ok(())
}
