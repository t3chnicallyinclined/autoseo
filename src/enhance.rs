//! DeepFilterNet3 speech enhancement pre-stage.
//!
//! When `ENHANCE_AUDIO=true` (and the `enhance` cargo feature is enabled),
//! extracted audio is denoised before chunking / STT and before final clip
//! rendering. The enhancement operates at 48 kHz mono (DeepFilterNet's native
//! sample rate) and writes a cleaned WAV that downstream stages consume.
//!
//! Graceful fallback: if the model file is missing or the crate fails to
//! initialise, a warning is logged and the original audio is used unchanged.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Download the DeepFilterNet3 model from GitHub releases if the local path
/// does not exist. Returns `Ok(path)` on success, or an error if download fails.
pub async fn ensure_model(model_path: &str) -> Result<PathBuf> {
    let path = PathBuf::from(model_path);
    if path.exists() {
        return Ok(path);
    }

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let url =
        "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/DeepFilterNet3.tar.gz";
    tracing::info!(url, dest = %path.display(), "downloading DeepFilterNet3 model");

    let resp = reqwest::get(url)
        .await
        .context("download DeepFilterNet3 model")?
        .error_for_status()
        .context("DeepFilterNet3 model download HTTP error")?;

    let bytes = resp
        .bytes()
        .await
        .context("read DeepFilterNet3 model bytes")?;
    tokio::fs::write(&path, &bytes)
        .await
        .with_context(|| format!("write model to {}", path.display()))?;

    tracing::info!(bytes = bytes.len(), dest = %path.display(), "DeepFilterNet3 model downloaded");
    Ok(path)
}

/// Extract audio from a media file to a 48 kHz mono WAV (DeepFilterNet's native rate).
pub async fn extract_48k_wav(ffmpeg: &str, input: &Path, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let status = Command::new(ffmpeg)
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .arg("-i")
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar", "48000"])
        .args(["-c:a", "pcm_s16le"])
        .arg(output)
        .status()
        .await
        .with_context(|| format!("ffmpeg extract 48kHz wav from {}", input.display()))?;

    if !status.success() {
        anyhow::bail!("ffmpeg 48kHz wav extraction failed (exit {status})");
    }
    Ok(())
}

/// Run DeepFilterNet3 speech enhancement on a 48 kHz mono WAV file.
///
/// Reads the input WAV, processes it frame-by-frame through the model, and
/// writes the enhanced result to `output_path`. This is CPU-bound work so it
/// runs on a blocking thread.
///
/// Returns `Ok(())` on success. On any failure (model load, processing, I/O)
/// returns an error — callers should fall back to the original audio.
#[cfg(feature = "enhance")]
pub async fn enhance_audio(model_path: &Path, input_path: &Path, output_path: &Path) -> Result<()> {
    let model_path = model_path.to_owned();
    let input_path = input_path.to_owned();
    let output_path = output_path.to_owned();

    tokio::task::spawn_blocking(move || enhance_blocking(&model_path, &input_path, &output_path))
        .await
        .context("join enhance_audio task")?
}

#[cfg(feature = "enhance")]
fn enhance_blocking(model_path: &Path, input_path: &Path, output_path: &Path) -> Result<()> {
    use df::tract::*;
    use ndarray::Array2;

    // DeepFilterNet3 operates at 48 kHz with a hop size of 480 samples (10 ms).
    const SR: usize = 48000;
    const HOP_SIZE: usize = 480;
    const CHANNELS: usize = 1;

    let dfp = DfParams::new(model_path.to_owned())
        .with_context(|| format!("load DeepFilterNet3 model from {}", model_path.display()))?;
    let rp = RuntimeParams::default_with_ch(CHANNELS);
    let mut df = DfTract::new(dfp, &rp).context("initialize DeepFilterNet3 runtime")?;

    // Read input WAV
    let reader = hound::WavReader::open(input_path)
        .with_context(|| format!("open input wav {}", input_path.display()))?;
    let spec = reader.spec();

    if spec.sample_rate != SR as u32 {
        anyhow::bail!(
            "DeepFilterNet3 requires {}Hz input, got {}Hz",
            SR,
            spec.sample_rate
        );
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("read wav samples")?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("read wav samples")?,
    };

    let total_samples = samples.len();
    tracing::info!(
        total_samples,
        sr = SR,
        hop_size = HOP_SIZE,
        duration_secs = total_samples as f64 / SR as f64,
        "enhancing audio with DeepFilterNet3"
    );

    // Process frame by frame
    let mut enhanced = Vec::with_capacity(total_samples);
    let num_frames = (total_samples + HOP_SIZE - 1) / HOP_SIZE;

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_SIZE;
        let end = (start + HOP_SIZE).min(total_samples);
        let frame_len = end - start;

        // Pad last frame if needed
        let mut frame_buf = vec![0.0f32; HOP_SIZE];
        frame_buf[..frame_len].copy_from_slice(&samples[start..end]);

        let noisy =
            Array2::from_shape_vec((CHANNELS, HOP_SIZE), frame_buf).context("shape noisy frame")?;
        let mut enh = Array2::zeros((CHANNELS, HOP_SIZE));

        df.process(noisy.view(), enh.view_mut())
            .context("DeepFilterNet3 process frame")?;

        // Only keep actual samples (not padding) for the last frame
        enhanced.extend_from_slice(&enh.row(0).as_slice().unwrap()[..frame_len]);
    }

    // Write enhanced WAV
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, out_spec)
        .with_context(|| format!("create output wav {}", output_path.display()))?;

    for &s in &enhanced {
        let clamped = s.clamp(-1.0, 1.0);
        let int_val = (clamped * 32767.0) as i16;
        writer
            .write_sample(int_val)
            .context("write enhanced sample")?;
    }
    writer.finalize().context("finalize enhanced wav")?;

    tracing::info!(
        output = %output_path.display(),
        samples = enhanced.len(),
        "DeepFilterNet3 enhancement complete"
    );
    Ok(())
}

/// Stub when the `enhance` feature is not compiled in.
#[cfg(not(feature = "enhance"))]
pub async fn enhance_audio(
    _model_path: &Path,
    _input_path: &Path,
    _output_path: &Path,
) -> Result<()> {
    anyhow::bail!(
        "ENHANCE_AUDIO=true requires the `enhance` cargo feature. \
         Recompile with: cargo build --features enhance"
    )
}

/// Top-level entry point: conditionally enhances audio for the pipeline.
///
/// If `enhance_audio` is false, returns `None` (use original audio).
/// If true, extracts 48kHz WAV, runs DeepFilterNet3, and returns the path
/// to the enhanced WAV. On failure, logs a warning and returns `None`
/// (graceful fallback).
pub async fn maybe_enhance(
    ffmpeg: &str,
    input_media: &Path,
    job_dir: &Path,
    model_path: &str,
    enhance_enabled: bool,
) -> Option<PathBuf> {
    if !enhance_enabled {
        return None;
    }

    let raw_wav = job_dir.join("audio_48k_raw.wav");
    let enhanced_wav = job_dir.join("audio_48k_enhanced.wav");

    // If already enhanced on a previous run, reuse it.
    if enhanced_wav.exists() {
        tracing::info!(path = %enhanced_wav.display(), "reusing previously enhanced audio");
        return Some(enhanced_wav);
    }

    let model = match ensure_model(model_path).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(error = %e, "failed to obtain DeepFilterNet3 model; skipping enhancement");
            return None;
        }
    };

    if let Err(e) = extract_48k_wav(ffmpeg, input_media, &raw_wav).await {
        tracing::warn!(error = %e, "failed to extract 48kHz WAV for enhancement; skipping");
        return None;
    }

    match enhance_audio(&model, &raw_wav, &enhanced_wav).await {
        Ok(()) => {
            tracing::info!(path = %enhanced_wav.display(), "speech enhancement complete");
            Some(enhanced_wav)
        }
        Err(e) => {
            tracing::warn!(error = %e, "DeepFilterNet3 enhancement failed; using original audio");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn maybe_enhance_disabled_returns_none() {
        let result = maybe_enhance(
            "ffmpeg",
            Path::new("/nonexistent"),
            Path::new("/tmp"),
            "./models/nope.tar.gz",
            false,
        )
        .await;
        assert!(
            result.is_none(),
            "should return None when enhance_enabled=false"
        );
    }

    #[tokio::test]
    async fn maybe_enhance_missing_model_returns_none() {
        // With a bogus model path and no network, should gracefully return None.
        let tmp = tempfile::tempdir().unwrap();
        let result = maybe_enhance(
            "ffmpeg",
            Path::new("/nonexistent.mp4"),
            tmp.path(),
            tmp.path()
                .join("nonexistent_model.tar.gz")
                .to_str()
                .unwrap(),
            true,
        )
        .await;
        assert!(
            result.is_none(),
            "should gracefully fall back when model is missing"
        );
    }

    #[tokio::test]
    async fn extract_48k_wav_from_synthetic_source() {
        if !ffmpeg_available().await {
            eprintln!("skipping: ffmpeg not found");
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("test_input.wav");
        let dst = tmp.path().join("test_48k.wav");

        // Generate a 1-second 16kHz mono sine wave via ffmpeg.
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1:sample_rate=16000",
            ])
            .args(["-ac", "1", "-c:a", "pcm_s16le"])
            .arg(&src)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .unwrap();
        assert!(status.success(), "failed to generate test wav");

        extract_48k_wav("ffmpeg", &src, &dst).await.unwrap();

        // Verify the output is 48kHz mono.
        let reader = hound::WavReader::open(&dst).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 48000, "output should be 48kHz");
        assert_eq!(spec.channels, 1, "output should be mono");
        assert!(reader.duration() > 0, "output should have samples");
    }

    #[tokio::test]
    async fn maybe_enhance_reuses_existing_enhanced_file() {
        let tmp = tempfile::tempdir().unwrap();
        let enhanced_path = tmp.path().join("audio_48k_enhanced.wav");

        // Create a dummy enhanced file.
        tokio::fs::write(&enhanced_path, b"fake enhanced wav")
            .await
            .unwrap();

        let result = maybe_enhance(
            "ffmpeg",
            Path::new("/nonexistent.mp4"),
            tmp.path(),
            "./models/nope.tar.gz",
            true,
        )
        .await;

        assert_eq!(
            result,
            Some(enhanced_path),
            "should reuse existing enhanced file"
        );
    }
}
