//! Audio Spectrogram Transformer (AST) via ONNX Runtime for audio-event detection.
//!
//! Classifies audio windows into event categories (laughter, applause, music, speech)
//! as a tier-2 ranking signal for candidate scoring.
//!
//! The model auto-downloads to `{WORK_DIR}/models/ast/` on first use. If the model
//! is unavailable (download failure, missing ONNX runtime), the scorer logs a warning
//! and returns `None` — intentionally non-fatal.

use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::Session;
use rustfft::{FftPlanner, num_complex::Complex};
use std::path::Path;
use tokio::process::Command;

// ── AudioSet class indices (standard 527-class ontology) ────────────────────
const SPEECH_INDICES: &[usize] = &[0, 1, 2, 3, 4, 5];
const LAUGHTER_INDICES: &[usize] = &[13];
const APPLAUSE_INDICES: &[usize] = &[36];
const MUSIC_INDICES: &[usize] = &[137];
const NUM_AUDIOSET_CLASSES: usize = 527;

// ── AST preprocessing constants (matches HuggingFace MIT/ast-finetuned-audioset) ─
const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400; // 25ms at 16 kHz
const HOP_LENGTH: usize = 160; // 10ms at 16 kHz
const N_MELS: usize = 128;
const TARGET_FRAMES: usize = 1024; // model expects exactly 1024 mel frames

// AudioSet normalization stats (dataset-level mean/std of log-mel features).
const AUDIOSET_MEAN: f32 = -4.2677393;
const AUDIOSET_STD: f32 = 4.5689974;

// Stride between inference windows (seconds).
const WINDOW_SAMPLES: usize = TARGET_FRAMES * HOP_LENGTH + WIN_LENGTH - HOP_LENGTH; // 163840
const WINDOW_SECS: f64 = WINDOW_SAMPLES as f64 / SAMPLE_RATE as f64; // ~10.24s
const STRIDE_SECS: f64 = 5.0;
const STRIDE_SAMPLES: usize = (STRIDE_SECS * SAMPLE_RATE as f64) as usize; // 80000

/// Per-window audio event scores (0.0..=1.0 probability).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AudioEvents {
    pub laughter: f64,
    pub applause: f64,
    pub music: f64,
    pub speech: f64,
}

/// A scored time window from the AST model.
#[derive(Debug, Clone)]
pub struct ScoredWindow {
    pub start_secs: f64,
    pub end_secs: f64,
    pub events: AudioEvents,
}

/// ONNX-backed AST scorer. Wraps an `ort::Session` for inference.
pub struct AstScorer {
    session: Session,
    mel_filterbank: Vec<Vec<f32>>,
    hann_window: Vec<f32>,
}

impl AstScorer {
    /// Load the AST model from `models_dir/ast/model.onnx`. If the model file
    /// doesn't exist and `model_url` is provided, download it first.
    pub async fn load(models_dir: &Path, model_url: Option<&str>) -> Result<Self> {
        let ast_dir = models_dir.join("ast");
        let model_path = ast_dir.join("model.onnx");

        if !model_path.exists() {
            if let Some(url) = model_url {
                tracing::info!(url, path = %model_path.display(), "AST model not found; downloading");
                download_model(url, &model_path).await?;
            } else {
                anyhow::bail!(
                    "AST model not found at {} and no download URL configured",
                    model_path.display()
                );
            }
        }

        tracing::info!(path = %model_path.display(), "loading AST ONNX model");
        let session = Session::builder()
            .context("create ort session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("load ONNX model from {}", model_path.display()))?;

        let mel_filterbank = build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE);
        let hann_window = build_hann_window(WIN_LENGTH);

        Ok(Self {
            session,
            mel_filterbank,
            hann_window,
        })
    }

    /// Score an audio file, returning per-window event probabilities.
    ///
    /// Extracts 16 kHz mono PCM via ffmpeg, slides a ~10.24s window with a 5s
    /// stride, computes mel spectrogram, runs ONNX inference, and maps AudioSet
    /// class probabilities to the four event categories.
    pub async fn score_file(
        &self,
        ffmpeg: &str,
        audio_path: &Path,
    ) -> Result<Vec<ScoredWindow>> {
        let pcm = extract_pcm_f32(ffmpeg, audio_path).await?;
        if pcm.len() < WIN_LENGTH {
            return Ok(Vec::new());
        }

        let total_samples = pcm.len();
        let mut windows = Vec::new();
        let mut offset: usize = 0;

        while offset + WIN_LENGTH <= total_samples {
            let end = (offset + WINDOW_SAMPLES).min(total_samples);
            let chunk = &pcm[offset..end];

            let fbank = compute_fbank(chunk, &self.hann_window, &self.mel_filterbank);
            let events = self.infer(&fbank)?;

            let start_secs = offset as f64 / SAMPLE_RATE as f64;
            let end_secs = end as f64 / SAMPLE_RATE as f64;

            windows.push(ScoredWindow {
                start_secs,
                end_secs,
                events,
            });

            offset += STRIDE_SAMPLES;
        }

        Ok(windows)
    }

    /// Run inference on a single mel-spectrogram frame.
    fn infer(&self, fbank: &[Vec<f32>]) -> Result<AudioEvents> {
        // Pad or truncate to exactly TARGET_FRAMES.
        let mut flat = Vec::with_capacity(TARGET_FRAMES * N_MELS);
        for i in 0..TARGET_FRAMES {
            if i < fbank.len() {
                flat.extend_from_slice(&fbank[i]);
            } else {
                flat.extend(std::iter::repeat_n(0.0f32, N_MELS));
            }
        }

        // Normalize with AudioSet stats.
        for v in flat.iter_mut() {
            *v = (*v - AUDIOSET_MEAN) / (2.0 * AUDIOSET_STD);
        }

        let input =
            Array3::<f32>::from_shape_vec((1, TARGET_FRAMES, N_MELS), flat)
                .context("build AST input tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![input].context("build ort inputs")?)
            .context("AST inference")?;

        let output = outputs
            .values()
            .next()
            .context("no output from AST model")?;
        let logits = output
            .try_extract_raw_tensor::<f32>()
            .context("extract AST output tensor")?
            .1;

        // Apply softmax to get probabilities.
        let probs = softmax(logits);
        Ok(extract_events(&probs))
    }
}

/// Aggregate event scores from scored windows for a candidate time range.
/// Returns the mean of each event score across windows that overlap the range.
pub fn aggregate_events(
    windows: &[ScoredWindow],
    start_secs: f64,
    end_secs: f64,
) -> Option<AudioEvents> {
    let overlapping: Vec<&ScoredWindow> = windows
        .iter()
        .filter(|w| w.start_secs < end_secs && w.end_secs > start_secs)
        .collect();

    if overlapping.is_empty() {
        return None;
    }

    let n = overlapping.len() as f64;
    let mut agg = AudioEvents::default();
    for w in &overlapping {
        agg.laughter += w.events.laughter;
        agg.applause += w.events.applause;
        agg.music += w.events.music;
        agg.speech += w.events.speech;
    }
    agg.laughter /= n;
    agg.applause /= n;
    agg.music /= n;
    agg.speech /= n;

    Some(agg)
}

// ── Private helpers ─────────────────────────────────────────────────────────

/// Extract 16 kHz mono f32 PCM from any audio file via ffmpeg.
async fn extract_pcm_f32(ffmpeg: &str, audio_path: &Path) -> Result<Vec<f32>> {
    let output = Command::new(ffmpeg)
        .arg("-y")
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .arg("-i")
        .arg(audio_path)
        .args(["-ac", "1", "-ar", &SAMPLE_RATE.to_string()])
        .args(["-f", "f32le", "-acodec", "pcm_f32le"])
        .arg("pipe:1")
        .output()
        .await
        .with_context(|| format!("run ffmpeg PCM extract from {}", audio_path.display()))?;

    if !output.status.success() {
        anyhow::bail!(
            "ffmpeg PCM extract failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let bytes = &output.stdout;
    if bytes.len() % 4 != 0 {
        anyhow::bail!("PCM output length {} not aligned to f32", bytes.len());
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(samples)
}

/// Compute log-mel filterbank features from raw PCM samples.
fn compute_fbank(
    samples: &[f32],
    hann_window: &[f32],
    mel_filterbank: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let n_frames = if samples.len() >= WIN_LENGTH {
        (samples.len() - WIN_LENGTH) / HOP_LENGTH + 1
    } else {
        0
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP_LENGTH;

        // Windowed frame, zero-padded to N_FFT.
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
        for i in 0..WIN_LENGTH {
            let idx = start + i;
            if idx < samples.len() {
                buffer[i] = Complex::new(samples[idx] * hann_window[i], 0.0);
            }
        }

        fft.process(&mut buffer);

        // Power spectrum (positive frequencies only).
        let power: Vec<f32> = buffer[..n_freqs]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // Apply mel filterbank.
        let mut mel: Vec<f32> = Vec::with_capacity(N_MELS);
        for filter in mel_filterbank {
            let energy: f32 = filter
                .iter()
                .zip(power.iter())
                .map(|(f, p)| f * p)
                .sum();
            // Log-mel (floor to avoid log(0)).
            mel.push((energy.max(1e-10)).ln());
        }

        frames.push(mel);
    }

    frames
}

/// Build a Hann window of the given length.
fn build_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / length as f32).cos())
        })
        .collect()
}

/// Build a mel filterbank matrix: `n_mels` triangular filters over `n_fft/2+1`
/// frequency bins, using the HTK mel scale. Uses float-precision bin positions
/// for smooth interpolation (avoids degenerate all-zero filters at low frequencies).
fn build_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 equally spaced points in mel scale.
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Float bin positions for smooth interpolation.
    let fbin: Vec<f64> = hz_points
        .iter()
        .map(|&hz| (n_fft as f64 + 1.0) * hz / sample_rate as f64)
        .collect();

    let mut filterbank = Vec::with_capacity(n_mels);
    for i in 0..n_mels {
        let mut filter = vec![0.0f32; n_freqs];
        let f_start = fbin[i];
        let f_center = fbin[i + 1];
        let f_end = fbin[i + 2];

        for j in 0..n_freqs {
            let jf = j as f64;
            if jf > f_start && jf <= f_center && f_center > f_start {
                filter[j] = ((jf - f_start) / (f_center - f_start)) as f32;
            } else if jf > f_center && jf < f_end && f_end > f_center {
                filter[j] = ((f_end - jf) / (f_end - f_center)) as f32;
            }
        }
        filterbank.push(filter);
    }

    filterbank
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Softmax over a logits slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Map softmax probabilities to event categories by taking the max probability
/// among the relevant AudioSet class indices.
fn extract_events(probs: &[f32]) -> AudioEvents {
    let max_prob = |indices: &[usize]| -> f64 {
        indices
            .iter()
            .filter_map(|&i| probs.get(i).copied())
            .fold(0.0f32, f32::max) as f64
    };

    AudioEvents {
        speech: max_prob(SPEECH_INDICES),
        laughter: max_prob(LAUGHTER_INDICES),
        applause: max_prob(APPLAUSE_INDICES),
        music: max_prob(MUSIC_INDICES),
    }
}

/// Download a file from `url` to `dest`, creating parent directories.
async fn download_model(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!("AST model download failed: HTTP {}", resp.status());
    }

    let bytes = resp.bytes().await.context("read AST model bytes")?;
    tokio::fs::write(dest, &bytes)
        .await
        .with_context(|| format!("write AST model to {}", dest.display()))?;

    tracing::info!(
        path = %dest.display(),
        bytes = bytes.len(),
        "AST model downloaded"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_properties() {
        let w = build_hann_window(400);
        assert_eq!(w.len(), 400);
        // Endpoints should be ~0 (periodic Hann).
        assert!(w[0].abs() < 1e-6, "start should be ~0, got {}", w[0]);
        // Midpoint should be ~1.
        assert!(
            (w[200] - 1.0).abs() < 0.01,
            "mid should be ~1, got {}",
            w[200]
        );
    }

    #[test]
    fn mel_filterbank_shape() {
        let fb = build_mel_filterbank(128, 512, 16000);
        assert_eq!(fb.len(), 128);
        assert_eq!(fb[0].len(), 257); // N_FFT/2 + 1
        // The first few filters may be sub-bin resolution at low frequencies
        // (mel spacing < FFT bin width). Check that the majority are non-zero.
        let nonzero_count = fb
            .iter()
            .filter(|f| f.iter().any(|&v| v > 0.0))
            .count();
        assert!(
            nonzero_count >= 120,
            "expected most filters to be non-zero, got {nonzero_count}/128"
        );
        // High-frequency filters should definitely be non-zero.
        for i in 10..128 {
            let sum: f32 = fb[i].iter().sum();
            assert!(sum > 0.0, "filter {i} is all zeros");
        }
    }

    #[test]
    fn mel_hz_roundtrip() {
        for &hz in &[0.0, 100.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!(
                (hz - back).abs() < 0.01,
                "roundtrip failed: {hz} -> {mel} -> {back}"
            );
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
        // Monotonic: higher logit → higher prob.
        for i in 0..probs.len() - 1 {
            assert!(probs[i] < probs[i + 1]);
        }
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn extract_events_from_probs() {
        let mut probs = vec![0.0f32; NUM_AUDIOSET_CLASSES];
        probs[0] = 0.8; // Speech
        probs[13] = 0.5; // Laughter
        probs[36] = 0.3; // Applause
        probs[137] = 0.1; // Music

        let events = extract_events(&probs);
        assert!((events.speech - 0.8).abs() < 1e-6);
        assert!((events.laughter - 0.5).abs() < 1e-6);
        assert!((events.applause - 0.3).abs() < 1e-6);
        assert!((events.music - 0.1).abs() < 1e-6);
    }

    #[test]
    fn aggregate_events_basic() {
        let windows = vec![
            ScoredWindow {
                start_secs: 0.0,
                end_secs: 10.0,
                events: AudioEvents {
                    laughter: 0.8,
                    applause: 0.0,
                    music: 0.0,
                    speech: 0.9,
                },
            },
            ScoredWindow {
                start_secs: 5.0,
                end_secs: 15.0,
                events: AudioEvents {
                    laughter: 0.4,
                    applause: 0.6,
                    music: 0.0,
                    speech: 0.7,
                },
            },
            ScoredWindow {
                start_secs: 20.0,
                end_secs: 30.0,
                events: AudioEvents {
                    laughter: 0.0,
                    applause: 0.0,
                    music: 0.9,
                    speech: 0.1,
                },
            },
        ];

        // Range [3, 12] overlaps windows 0 and 1.
        let agg = aggregate_events(&windows, 3.0, 12.0).unwrap();
        assert!((agg.laughter - 0.6).abs() < 1e-6); // (0.8 + 0.4) / 2
        assert!((agg.applause - 0.3).abs() < 1e-6); // (0.0 + 0.6) / 2
        assert!((agg.speech - 0.8).abs() < 1e-6); // (0.9 + 0.7) / 2

        // Range [50, 60] overlaps nothing.
        assert!(aggregate_events(&windows, 50.0, 60.0).is_none());
    }

    #[test]
    fn compute_fbank_shape() {
        // 1 second of silence at 16 kHz.
        let samples = vec![0.0f32; 16000];
        let hann = build_hann_window(WIN_LENGTH);
        let fb = build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE);

        let fbank = compute_fbank(&samples, &hann, &fb);

        // Expected frames: (16000 - 400) / 160 + 1 = 98.
        let expected_frames = (16000 - WIN_LENGTH) / HOP_LENGTH + 1;
        assert_eq!(fbank.len(), expected_frames);
        assert_eq!(fbank[0].len(), N_MELS);
    }

    #[test]
    fn compute_fbank_with_tone() {
        // 440 Hz sine wave, 1 second.
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let hann = build_hann_window(WIN_LENGTH);
        let fb = build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE);

        let fbank = compute_fbank(&samples, &hann, &fb);
        assert!(!fbank.is_empty());

        // The mel bin containing 440 Hz should have higher energy than silence.
        // 440 Hz in mel space should be around bin 20-30 (rough estimate).
        let mid_frame = &fbank[fbank.len() / 2];
        let max_energy = mid_frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_energy > -20.0, "expected detectable energy for 440 Hz tone");
    }
}
