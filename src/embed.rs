//! Sentence embeddings for within-episode novelty scoring.
//!
//! Two backends behind one [`Embedder`] enum:
//!
//! - [`FastembedEmbedder`] — local ONNX via `fastembed`, default model
//!   `all-MiniLM-L6-v2` (384-dim, 2020). Used when no HF API key is configured.
//! - [`HfEmbedder`] — HF Inference Providers (OpenAI-compatible `/v1/embeddings`).
//!   Default model `Qwen/Qwen3-Embedding-0.6B` (1024-dim, 32K context, MTEB
//!   leader for its size class). Used when `HF_API_KEY` is set.
//!
//! Both implement the same `embed(Vec<String>) -> Vec<Vec<f32>>` API; the
//! novelty scorer is backend-agnostic.

use anyhow::Context;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::config::Config;
use crate::cost::CostTracker;

#[derive(Clone)]
pub enum Embedder {
    Fastembed(FastembedEmbedder),
    Hf(HfEmbedder),
}

impl Embedder {
    /// Pick a backend from config. Prefers HF Inference Providers when
    /// `HF_API_KEY` is set; falls back to local `fastembed` ONNX otherwise.
    pub fn from_config(cfg: &Config) -> anyhow::Result<Self> {
        Self::from_config_with_tracker(cfg, None)
    }

    /// Like [`from_config`] but attaches a cost tracker for usage estimation.
    pub fn from_config_with_tracker(
        cfg: &Config,
        cost_tracker: Option<&CostTracker>,
    ) -> anyhow::Result<Self> {
        if let Some(key) = cfg.hf_api_key.as_ref().filter(|k| !k.is_empty()) {
            let mut embedder = HfEmbedder::new(
                cfg.hf_router_url.clone(),
                cfg.hf_embed_provider.clone(),
                key.clone(),
                cfg.embed_model.clone(),
            );
            if let Some(tracker) = cost_tracker {
                embedder.cost_tracker = Some(tracker.clone());
            }
            Ok(Embedder::Hf(embedder))
        } else {
            let cache_dir =
                (!cfg.embed_model_dir.is_empty()).then(|| PathBuf::from(&cfg.embed_model_dir));
            Ok(Embedder::Fastembed(FastembedEmbedder::try_new(cache_dir)?))
        }
    }

    pub async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Embedder::Fastembed(e) => e.embed(texts).await,
            Embedder::Hf(e) => e.embed(texts).await,
        }
    }

    /// Force backend init / first call. Use on startup to fail fast if the
    /// remote endpoint is unreachable or the local model can't load.
    pub async fn warmup(&self) -> anyhow::Result<()> {
        let _ = self.embed(vec!["warmup".to_string()]).await?;
        Ok(())
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Embedder::Fastembed(_) => "fastembed-local",
            Embedder::Hf(_) => "hf-inference",
        }
    }
}

#[derive(Clone)]
pub struct FastembedEmbedder {
    inner: Arc<TextEmbedding>,
}

impl FastembedEmbedder {
    pub fn try_new(cache_dir: Option<PathBuf>) -> anyhow::Result<Self> {
        let mut opts = InitOptions::new(EmbeddingModel::AllMiniLML6V2);
        if let Some(dir) = cache_dir {
            std::fs::create_dir_all(&dir).ok();
            opts = opts.with_cache_dir(dir);
        }
        let model = TextEmbedding::try_new(opts).context("init fastembed model")?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    pub async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<f32>>> {
            inner.embed(texts, None).context("fastembed embed")
        })
        .await
        .context("join embed")?
    }
}

#[derive(Clone)]
pub struct HfEmbedder {
    router_url: String,
    provider: String,
    api_key: String,
    model: String,
    http: reqwest::Client,
    batch_size: usize,
    cost_tracker: Option<CostTracker>,
}

impl HfEmbedder {
    /// `router_url` is the root (no `/v1`), e.g. `https://router.huggingface.co`.
    /// `provider` routes the request, e.g. `hf-inference` or `scaleway`.
    pub fn new(router_url: String, provider: String, api_key: String, model: String) -> Self {
        Self {
            router_url: router_url.trim_end_matches('/').to_string(),
            provider: provider.trim_matches('/').to_string(),
            api_key,
            model,
            http: reqwest::Client::new(),
            batch_size: 32,
            cost_tracker: None,
        }
    }

    fn endpoint_url(&self) -> String {
        // Final URL: {router}/{provider}/models/{model}/pipeline/feature-extraction
        format!(
            "{}/{}/models/{}/pipeline/feature-extraction",
            self.router_url, self.provider, self.model
        )
    }

    pub async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let batch = chunk.to_vec();
            let mut vecs = self.embed_batch(batch).await?;
            out.append(&mut vecs);
        }
        Ok(out)
    }

    async fn embed_batch(&self, batch: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let url = self.endpoint_url();
        let body = FeatureExtractionRequest {
            inputs: batch.clone(),
        };

        const MAX_ATTEMPTS: usize = 5;
        let mut backoff = Duration::from_millis(400);
        for attempt in 1..=MAX_ATTEMPTS {
            let res = self
                .http
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;

            match res {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        // The hf-inference feature-extraction task returns a bare
                        // array of arrays — one vector per input, in order.
                        let vecs: Vec<Vec<f32>> = r
                            .json()
                            .await
                            .context("parse hf feature-extraction response")?;
                        if vecs.len() != batch.len() {
                            anyhow::bail!(
                                "hf feature-extraction returned {} vectors for {} inputs",
                                vecs.len(),
                                batch.len()
                            );
                        }
                        if let Some(tracker) = &self.cost_tracker {
                            let total_chars: usize = batch.iter().map(|t| t.len()).sum();
                            tracker.record_embedding_call(&self.model, total_chars);
                        }
                        return Ok(vecs);
                    }

                    let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
                    if retryable && attempt < MAX_ATTEMPTS {
                        let body_text = r.text().await.unwrap_or_default();
                        tracing::warn!(
                            attempt,
                            status = %status,
                            body = %truncate(&body_text, 400),
                            "hf feature-extraction retryable; backing off"
                        );
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    let body_text = r.text().await.unwrap_or_default();
                    anyhow::bail!("hf feature-extraction failed: {status} {body_text}");
                }
                Err(e) => {
                    if attempt < MAX_ATTEMPTS {
                        tracing::warn!(attempt, error = %e, "hf feature-extraction request error; backing off");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    return Err(anyhow::Error::new(e)).context("hf feature-extraction POST");
                }
            }
        }
        anyhow::bail!("hf feature-extraction failed after {MAX_ATTEMPTS} attempts")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "…"
}

#[derive(Debug, Serialize)]
struct FeatureExtractionRequest {
    inputs: Vec<String>,
}

/// Cosine similarity. Tolerant of non-unit vectors (re-normalizes).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    dot / (na * nb)
}

pub fn mean_vec(vecs: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vecs.first()?;
    let dim = first.len();
    if dim == 0 {
        return None;
    }
    let mut acc = vec![0.0_f32; dim];
    for v in vecs {
        if v.len() != dim {
            return None;
        }
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += x;
        }
    }
    let n = vecs.len() as f32;
    for a in acc.iter_mut() {
        *a /= n;
    }
    Some(acc)
}

pub fn score_novelty(vecs: &[Vec<f32>]) -> Vec<f64> {
    if vecs.is_empty() {
        return Vec::new();
    }
    if vecs.len() == 1 {
        return vec![0.0];
    }
    let mut centroid = match mean_vec(vecs) {
        Some(c) => c,
        None => return vec![0.0; vecs.len()],
    };
    l2_normalize(&mut centroid);

    let raw: Vec<f64> = vecs
        .iter()
        .map(|v| 1.0 - cosine_similarity(v, &centroid) as f64)
        .collect();

    let max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = raw.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max - min).max(1e-9);
    raw.iter().map(|d| (d - min) / range).collect()
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_handles_zero_vectors() {
        let zero = vec![0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&zero, &v), 0.0);
        assert_eq!(cosine_similarity(&v, &zero), 0.0);
    }

    #[test]
    fn cosine_known_values() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
        assert!((cosine_similarity(&a, &d) - -1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_vec_basic() {
        let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let m = mean_vec(&vecs).unwrap();
        assert!((m[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((m[1] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn mean_vec_empty_or_ragged() {
        assert!(mean_vec(&[]).is_none());
        let ragged = vec![vec![1.0, 0.0], vec![0.0]];
        assert!(mean_vec(&ragged).is_none());
    }

    #[test]
    fn score_novelty_normalizes_to_unit_range() {
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let scores = score_novelty(&vecs);
        assert_eq!(scores.len(), 3);
        assert!(scores[2] >= scores[0] - 1e-9);
        assert!(scores[2] >= scores[1] - 1e-9);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((max - 1.0).abs() < 1e-6);
        assert!((min - 0.0).abs() < 1e-6);
    }

    #[test]
    fn score_novelty_single_input_is_zero() {
        let scores = score_novelty(&[vec![1.0, 2.0, 3.0]]);
        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn score_novelty_empty_is_empty() {
        assert!(score_novelty(&[]).is_empty());
    }

    #[test]
    fn hf_embedder_builds_native_url() {
        let e = HfEmbedder::new(
            "https://router.huggingface.co/".into(),
            "hf-inference".into(),
            "k".into(),
            "BAAI/bge-large-en-v1.5".into(),
        );
        assert_eq!(
            e.endpoint_url(),
            "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction"
        );
    }

    // Integration test — requires HF_API_KEY in env. Skipped by default.
    #[ignore]
    #[tokio::test]
    async fn hf_embed_end_to_end() -> anyhow::Result<()> {
        let key = match std::env::var("HF_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!("skipping: HF_API_KEY not set");
                return Ok(());
            }
        };
        let e = HfEmbedder::new(
            "https://router.huggingface.co".into(),
            "hf-inference".into(),
            key,
            "BAAI/bge-large-en-v1.5".into(),
        );
        let texts = vec![
            "The quarterback threw a perfect spiral.".to_string(),
            "Quantum entanglement defies classical intuition.".to_string(),
        ];
        let vecs = e.embed(texts).await?;
        assert_eq!(vecs.len(), 2);
        assert!(
            vecs[0].len() >= 384,
            "embedding dim should be large; got {}",
            vecs[0].len()
        );
        let sim = cosine_similarity(&vecs[0], &vecs[1]);
        assert!(
            sim < 0.7,
            "unrelated texts should have low cosine; got {sim}"
        );
        Ok(())
    }

    #[ignore]
    #[tokio::test]
    async fn fastembed_end_to_end() -> anyhow::Result<()> {
        let cache = std::env::temp_dir().join("autoseo_fastembed_test");
        let e = FastembedEmbedder::try_new(Some(cache))?;
        let texts = vec![
            "The quarterback threw a perfect spiral.".to_string(),
            "The pitcher delivered a fastball.".to_string(),
            "Quantum entanglement defies classical intuition.".to_string(),
        ];
        let vecs = e.embed(texts).await?;
        assert_eq!(vecs.len(), 3);
        assert_eq!(vecs[0].len(), 384);
        let scores = score_novelty(&vecs);
        assert!(scores[2] > scores[0] && scores[2] > scores[1]);
        Ok(())
    }
}
