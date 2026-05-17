//! Offline STT via whisper.cpp (whisper-rs bindings).
//!
//! Feature-gated behind `local-stt`. Provides the same output shape as the
//! API path (`TranscriptionText` with segments + words) so the rest of the
//! pipeline is backend-agnostic.

use anyhow::Context;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::openai::{TranscriptionSegment, TranscriptionText, TranscriptionWord};

/// A reusable handle around a loaded whisper.cpp model.
#[derive(Clone)]
pub struct WhisperLocal {
    ctx: Arc<WhisperContext>,
}

impl std::fmt::Debug for WhisperLocal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperLocal").finish()
    }
}

impl WhisperLocal {
    /// Load a GGML model file. Errors if the path does not exist.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        if !model_path.exists() {
            anyhow::bail!(
                "Whisper model not found at {}.\n\
                 Download a GGML model (e.g. ggml-large-v3-turbo.bin) from:\n  \
                 https://huggingface.co/ggerganov/whisper.cpp/tree/main\n\
                 and place it at the path above, or set WHISPER_MODEL_PATH to point to it.",
                model_path.display()
            );
        }
        let model_str = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let ctx = WhisperContext::new_with_params(model_str, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))?;
        Ok(Self {
            ctx: Arc::new(ctx),
        })
    }

    /// Resolve the default model path: `{work_dir}/models/whisper/ggml-large-v3-turbo.bin`
    pub fn default_model_path(work_dir: &str) -> PathBuf {
        PathBuf::from(work_dir)
            .join("models")
            .join("whisper")
            .join("ggml-large-v3-turbo.bin")
    }

    /// Transcribe a single audio file. The file should be WAV 16kHz mono
    /// (the caller is responsible for conversion via ffmpeg beforehand).
    ///
    /// This is CPU-bound, so we run it on a blocking thread.
    pub async fn transcribe(&self, audio_path: &Path) -> anyhow::Result<TranscriptionText> {
        let audio_path = audio_path.to_path_buf();
        let ctx = self.ctx.clone();
        tokio::task::spawn_blocking(move || transcribe_blocking(&ctx, &audio_path))
            .await
            .context("whisper blocking task panicked")?
    }
}

fn transcribe_blocking(
    ctx: &WhisperContext,
    audio_path: &Path,
) -> anyhow::Result<TranscriptionText> {
    let samples = read_wav_samples(audio_path)?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(true);
    params.set_language(Some("en"));

    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("create whisper state: {e}"))?;
    state
        .full(params, &samples)
        .map_err(|e| anyhow::anyhow!("whisper inference failed: {e}"))?;

    let n_segments = state.full_n_segments().unwrap_or(0) as i32;
    let mut segments = Vec::new();
    let mut words = Vec::new();
    let mut full_text = String::new();

    for i in 0..n_segments {
        let seg_start = state.full_get_segment_t0(i).unwrap_or(0) as f64 / 100.0;
        let seg_end = state.full_get_segment_t1(i).unwrap_or(0) as f64 / 100.0;
        let seg_text = state
            .full_get_segment_text(i)
            .unwrap_or_default();

        segments.push(TranscriptionSegment {
            id: Some(i as u64),
            start: seg_start,
            end: seg_end,
            text: seg_text.clone(),
        });

        if !seg_text.trim().is_empty() {
            full_text.push_str(seg_text.trim());
            full_text.push('\n');
        }

        // Extract word-level timestamps from tokens.
        let n_tokens = state.full_n_tokens(i).unwrap_or(0) as i32;
        for t in 0..n_tokens {
            let token_text = state.full_get_token_text(i, t).unwrap_or_default();
            let trimmed = token_text.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Skip special tokens (they start with [, <, or are whitespace-only)
            if trimmed.starts_with('[') || trimmed.starts_with('<') {
                continue;
            }
            let token_data = state.full_get_token_data(i, t);
            match token_data {
                Ok(data) => {
                    let w_start = data.t0 as f64 / 100.0;
                    let w_end = data.t1 as f64 / 100.0;
                    words.push(TranscriptionWord {
                        word: trimmed.to_string(),
                        start: w_start,
                        end: w_end,
                    });
                }
                Err(_) => {
                    // If we can't get token data, use segment-level timestamps.
                    words.push(TranscriptionWord {
                        word: trimmed.to_string(),
                        start: seg_start,
                        end: seg_end,
                    });
                }
            }
        }
    }

    Ok(TranscriptionText {
        text: full_text,
        segments,
        words,
    })
}

/// Read a WAV file and return f32 samples at 16kHz mono (what whisper.cpp expects).
fn read_wav_samples(path: &Path) -> anyhow::Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("open wav file {}", path.display()))?;
    let spec = reader.spec();

    let samples_i16: Vec<i16> = if spec.bits_per_sample == 16 && spec.sample_format == hound::SampleFormat::Int {
        reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .context("read wav samples")?
    } else if spec.bits_per_sample == 32 && spec.sample_format == hound::SampleFormat::Float {
        let float_samples: Vec<f32> = reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("read wav float samples")?;
        // Convert to mono if needed, then return directly as f32.
        let mono = if spec.channels > 1 {
            float_samples
                .chunks(spec.channels as usize)
                .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
                .collect()
        } else {
            float_samples
        };
        // whisper.cpp expects float samples in [-1, 1], which WAV float already is.
        return Ok(mono);
    } else {
        // Try reading as i16 anyway; hound will convert.
        reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .context("read wav samples (fallback i16)")?
    };

    // Mix down to mono.
    let mono: Vec<i16> = if spec.channels > 1 {
        samples_i16
            .chunks(spec.channels as usize)
            .map(|ch| {
                let sum: i32 = ch.iter().map(|&s| s as i32).sum();
                (sum / ch.len() as i32) as i16
            })
            .collect()
    } else {
        samples_i16
    };

    // Convert i16 -> f32 in [-1, 1].
    let float_samples: Vec<f32> = mono.iter().map(|&s| s as f32 / 32768.0).collect();

    Ok(float_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_path() {
        let path = WhisperLocal::default_model_path("./work");
        assert!(path.ends_with("ggml-large-v3-turbo.bin"));
        assert!(path.to_str().unwrap().contains("models"));
        assert!(path.to_str().unwrap().contains("whisper"));
    }

    #[test]
    fn test_load_missing_model_errors() {
        let result = WhisperLocal::load(Path::new("/nonexistent/model.bin"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"), "error should mention not found: {err}");
        assert!(err.contains("huggingface"), "error should mention download URL: {err}");
    }

    #[test]
    fn test_read_wav_samples_missing_file() {
        let result = read_wav_samples(Path::new("/nonexistent/audio.wav"));
        assert!(result.is_err());
    }
}
