//! Sentence-embedding wrapper for within-episode novelty scoring.
//!
//! Backs the ranker's "is this minute unusual for this episode?" feature by embedding
//! per-window transcript text and computing cosine distance to the episode centroid.
//!
//! Uses `fastembed` (ONNX-runtime-backed) with `all-MiniLM-L6-v2` (384-dim, ~90 MB).
//! First call downloads the model under `EMBED_MODEL_DIR`; subsequent calls reuse the cache.
//! Embeddings are L2-normalized by fastembed; we re-normalize the centroid before cosine.

use anyhow::Context;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct Embedder {
    inner: Arc<TextEmbedding>,
}

impl Embedder {
    /// Construct the embedder, optionally with a custom on-disk cache directory.
    /// `None` falls back to fastembed's default (~/.cache/fastembed).
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

    /// Embed a batch of strings. Returns one 384-dim, L2-normalized vector per input,
    /// in the same order. Runs on a blocking thread so it doesn't stall the async runtime.
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

    /// Force model load up-front (fastembed lazy-loads on first embed call).
    /// Use this on startup to fail fast if the model isn't reachable.
    pub async fn warmup(&self) -> anyhow::Result<()> {
        let _ = self.embed(vec!["warmup".to_string()]).await?;
        Ok(())
    }
}

/// Cosine similarity of two equal-length vectors. Assumes L2-normalized inputs for
/// best fidelity, but works on any vector.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Element-wise mean of a non-empty set of vectors. Returns `None` if empty or ragged.
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

/// Score per-chunk novelty: cosine distance to the centroid of the set, normalized to
/// `[0, 1]` within this batch. 1.0 = the most novel chunk relative to the episode mean;
/// 0.0 = the closest to the centroid. Returns one score per input vector.
///
/// Pass the L2-normalized embeddings from `Embedder::embed`; this function re-normalizes
/// the centroid internally so cosine is computed correctly.
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

    let raw_distances: Vec<f64> = vecs
        .iter()
        .map(|v| 1.0 - cosine_similarity(v, &centroid) as f64)
        .collect();

    let max = raw_distances
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min = raw_distances.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max - min).max(1e-9);
    raw_distances.iter().map(|d| (d - min) / range).collect()
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
        // Three vectors: two close together, one orthogonal outlier.
        // Outlier should score 1.0 (most novel), the near-twins should be lower.
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let scores = score_novelty(&vecs);
        assert_eq!(scores.len(), 3);
        // Outlier (index 2) should be the max.
        assert!(scores[2] >= scores[0] - 1e-9);
        assert!(scores[2] >= scores[1] - 1e-9);
        // Normalized range should hit both endpoints.
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((max - 1.0).abs() < 1e-6, "max should be 1.0, got {max}");
        assert!((min - 0.0).abs() < 1e-6, "min should be 0.0, got {min}");
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

    // Integration test — downloads the model on first run (~90 MB). Skipped by default;
    // run with `cargo test -- --ignored embed_end_to_end` to verify the live wiring.
    #[ignore]
    #[tokio::test]
    async fn embed_end_to_end() -> anyhow::Result<()> {
        let cache = std::env::temp_dir().join("autoseo_fastembed_test");
        let embedder = Embedder::try_new(Some(cache))?;
        let texts = vec![
            "The quarterback threw a perfect spiral.".to_string(),
            "The pitcher delivered a fastball.".to_string(),
            "Quantum entanglement defies classical intuition.".to_string(),
        ];
        let vecs = embedder.embed(texts).await?;
        assert_eq!(vecs.len(), 3);
        assert_eq!(vecs[0].len(), 384, "all-MiniLM-L6-v2 emits 384-dim");

        // First two are both about sports; third is about physics. The third should be
        // farthest from the centroid (most novel).
        let scores = score_novelty(&vecs);
        assert_eq!(scores.len(), 3);
        assert!(
            scores[2] > scores[0] && scores[2] > scores[1],
            "physics line should be most novel, got {scores:?}"
        );
        Ok(())
    }
}
