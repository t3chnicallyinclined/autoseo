//! Per-job API cost tracking.
//!
//! [`CostTracker`] accumulates token usage and estimated cost across all API
//! calls within a single pipeline run. Thread-safe via `Arc<Mutex<_>>` so it
//! can be shared across concurrent STT/embedding tasks.

use std::fmt;
use std::sync::{Arc, Mutex};

/// Category of API call for cost breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CostCategory {
    Stt,
    Chat,
    Embeddings,
    Vlm,
    VlmPremium,
}

impl CostCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            CostCategory::Stt => "stt",
            CostCategory::Chat => "chat",
            CostCategory::Embeddings => "embeddings",
            CostCategory::Vlm => "vlm",
            CostCategory::VlmPremium => "vlm_premium",
        }
    }
}

impl fmt::Display for CostCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-category accumulator.
#[derive(Debug, Clone, Default)]
pub struct CategoryCost {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_cents: f64,
}

/// Hardcoded per-model cost rates (USD per 1K tokens). Covers the models
/// configured by default in autoseo. Add new entries as needed.
///
/// Returns `(input_cost_per_1k, output_cost_per_1k)`.
///
/// **Match order matters**: more-specific entries must come first because
/// `m if m.contains("whisper")` would shadow `m == "whisper-large-v3-turbo"`
/// if put in the wrong order. Groq's turbo whisper is ~3× cheaper than the
/// non-turbo; collapsing them into one rate overestimates by 3× on turbo.
///
/// Source: provider pricing pages as of 2026-05.
fn model_rate(model: &str) -> (f64, f64) {
    match model {
        // ── Groq chat models ────────────────────────────────────────────
        // From console.groq.com/pricing. Convert per-1M → per-1K: ×0.001.
        "openai/gpt-oss-120b" => (0.00015, 0.00075),
        "openai/gpt-oss-20b" => (0.000075, 0.0003),
        "meta-llama/llama-4-scout-17b-16e-instruct" => (0.00011, 0.00034),
        "llama-3.3-70b-versatile" => (0.00059, 0.00079),
        "llama-3.1-8b-instant" => (0.00005, 0.00008),
        "qwen/qwen3-32b" => (0.00029, 0.00059),

        // ── Groq STT — priced per audio-second, not tokens ──────────────
        // Caller (record_stt_call) converts seconds → "tokens" at 25/sec,
        // so per-1K-token rates here back out to the real per-second cost.
        // Groq whisper-large-v3-turbo: $0.04/hr ≈ $0.0000111/sec ≈ $0.000444/1K toks.
        // Groq whisper-large-v3      : $0.111/hr ≈ $0.0000308/sec ≈ $0.00123/1K toks.
        "whisper-large-v3-turbo" => (0.000444, 0.0),
        "whisper-large-v3" => (0.00123, 0.0),

        // ── OpenAI ──────────────────────────────────────────────────────
        m if m.starts_with("gpt-5.2-pro") => (0.002, 0.010),
        m if m.starts_with("gpt-5") => (0.002, 0.010),
        m if m.starts_with("gpt-4o") => (0.0025, 0.010),
        m if m.starts_with("gpt-4") => (0.01, 0.03),
        // OpenAI's older whisper-1 — billed per minute of audio. ~$0.006/min
        // → $0.0001/sec → 25 tokens/sec → $0.004/1K "tokens".
        "whisper-1" => (0.004, 0.0),
        // Whisper fallback for any other variant (defensive).
        m if m.contains("whisper") => (0.00123, 0.0),

        // ── HuggingFace Inference Providers ─────────────────────────────
        m if m.contains("bge") || m.contains("embed") => (0.0001, 0.0),
        m if m.contains("Qwen3-VL-8B") || m.contains("qwen3-vl-8b") => (0.0003, 0.0006),
        m if m.contains("qwen2.5-vl-72b") || m.contains("Qwen2.5-VL-72B") => (0.001, 0.002),

        // ── Anthropic via OpenRouter (rough per-1M rates as of 2026-05) ──
        m if m.contains("claude-opus-4.7") || m.contains("claude-opus-4-7") => (0.015, 0.075),
        m if m.contains("claude-sonnet-4.6") || m.contains("claude-sonnet-4-6") => (0.003, 0.015),
        m if m.contains("claude-haiku-4.5") || m.contains("claude-haiku-4-5") => (0.001, 0.005),

        // ── Catch-all ───────────────────────────────────────────────────
        // Picked to be roughly OpenRouter "mid-tier" so unknown premium VLM
        // models on OpenRouter aren't wildly under-counted. Logs a warning
        // so we can spot missing entries in production.
        _ => {
            tracing::warn!(model, "unknown model in cost.rs::model_rate — using catch-all rate; add to the match arm");
            (0.001, 0.003)
        }
    }
}

/// Record a single API call with known token counts.
#[derive(Debug, Clone)]
pub struct UsageRecord {
    pub category: CostCategory,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Thread-safe, cloneable cost tracker. Create one per pipeline run.
#[derive(Clone, Debug)]
pub struct CostTracker {
    inner: Arc<Mutex<CostTrackerInner>>,
    enabled: bool,
}

#[derive(Debug, Default)]
struct CostTrackerInner {
    stt: CategoryCost,
    chat: CategoryCost,
    embeddings: CategoryCost,
    vlm: CategoryCost,
    vlm_premium: CategoryCost,
}

impl CostTrackerInner {
    fn category_mut(&mut self, cat: CostCategory) -> &mut CategoryCost {
        match cat {
            CostCategory::Stt => &mut self.stt,
            CostCategory::Chat => &mut self.chat,
            CostCategory::Embeddings => &mut self.embeddings,
            CostCategory::Vlm => &mut self.vlm,
            CostCategory::VlmPremium => &mut self.vlm_premium,
        }
    }
}

impl CostTracker {
    pub fn new(enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CostTrackerInner::default())),
            enabled,
        }
    }

    /// Whether cost tracking is active.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Record token usage from an API call. Cost is estimated from the model name.
    pub fn record(&self, usage: UsageRecord) {
        if !self.enabled {
            return;
        }
        let (in_rate, out_rate) = model_rate(&usage.model);
        let cost = (usage.input_tokens as f64 * in_rate + usage.output_tokens as f64 * out_rate)
            / 1000.0
            * 100.0; // convert USD to cents

        let mut inner = self.inner.lock().unwrap();
        let cat = inner.category_mut(usage.category);
        cat.calls += 1;
        cat.input_tokens += usage.input_tokens;
        cat.output_tokens += usage.output_tokens;
        cat.cost_cents += cost;
    }

    /// Record an STT call. STT is priced per audio-second; we pass audio_secs
    /// as "input_tokens" with a whisper model rate for estimation.
    pub fn record_stt_call(&self, model: &str, audio_duration_secs: f64) {
        if !self.enabled {
            return;
        }
        // Approximate: 1 audio-second → 25 "tokens" for cost estimation.
        let approx_tokens = (audio_duration_secs * 25.0).round() as u64;
        self.record(UsageRecord {
            category: CostCategory::Stt,
            model: model.to_string(),
            input_tokens: approx_tokens,
            output_tokens: 0,
        });
    }

    /// Record an embedding call. HF embeddings don't report tokens; estimate
    /// from input text length (~4 chars per token).
    pub fn record_embedding_call(&self, model: &str, total_chars: usize) {
        if !self.enabled {
            return;
        }
        let approx_tokens = (total_chars as u64) / 4;
        self.record(UsageRecord {
            category: CostCategory::Embeddings,
            model: model.to_string(),
            input_tokens: approx_tokens,
            output_tokens: 0,
        });
    }

    /// Snapshot of all categories. Returns `(stt, chat, embeddings, vlm, vlm_premium)`.
    pub fn snapshot(&self) -> CostSnapshot {
        let inner = self.inner.lock().unwrap();
        CostSnapshot {
            stt: inner.stt.clone(),
            chat: inner.chat.clone(),
            embeddings: inner.embeddings.clone(),
            vlm: inner.vlm.clone(),
            vlm_premium: inner.vlm_premium.clone(),
        }
    }

    /// Total estimated cost in cents across all categories.
    pub fn total_cost_cents(&self) -> f64 {
        let snap = self.snapshot();
        snap.total_cost_cents()
    }
}

#[derive(Debug, Clone)]
pub struct CostSnapshot {
    pub stt: CategoryCost,
    pub chat: CategoryCost,
    pub embeddings: CategoryCost,
    pub vlm: CategoryCost,
    pub vlm_premium: CategoryCost,
}

impl CostSnapshot {
    pub fn total_cost_cents(&self) -> f64 {
        self.stt.cost_cents
            + self.chat.cost_cents
            + self.embeddings.cost_cents
            + self.vlm.cost_cents
            + self.vlm_premium.cost_cents
    }

    pub fn total_calls(&self) -> u64 {
        self.stt.calls
            + self.chat.calls
            + self.embeddings.calls
            + self.vlm.calls
            + self.vlm_premium.calls
    }

    /// Format a human-readable cost summary for the digest.
    pub fn format_digest(&self) -> String {
        let mut out = String::new();
        out.push_str("Cost breakdown:\n");

        let cats: &[(&str, &CategoryCost)] = &[
            ("STT", &self.stt),
            ("Chat/LLM", &self.chat),
            ("Embeddings", &self.embeddings),
            ("VLM", &self.vlm),
            ("VLM Premium", &self.vlm_premium),
        ];

        for (label, cat) in cats {
            if cat.calls == 0 {
                continue;
            }
            out.push_str(&format!(
                "  {label:<14} {calls:>3} calls  {in_tok:>8} in  {out_tok:>8} out  ${cost:.4}\n",
                calls = cat.calls,
                in_tok = cat.input_tokens,
                out_tok = cat.output_tokens,
                cost = cat.cost_cents / 100.0,
            ));
        }

        let total = self.total_cost_cents();
        out.push_str(&format!(
            "  {:<14} {:>3} calls  total ${:.4}\n",
            "TOTAL",
            self.total_calls(),
            total / 100.0,
        ));
        out
    }

    /// Build a JSON value for the manifest.
    pub fn to_json(&self) -> serde_json::Value {
        let cat_json = |label: &str, cat: &CategoryCost| -> serde_json::Value {
            serde_json::json!({
                "category": label,
                "calls": cat.calls,
                "input_tokens": cat.input_tokens,
                "output_tokens": cat.output_tokens,
                "cost_usd": (cat.cost_cents / 100.0 * 10000.0).round() / 10000.0,
            })
        };

        let categories: Vec<serde_json::Value> = [
            ("stt", &self.stt),
            ("chat", &self.chat),
            ("embeddings", &self.embeddings),
            ("vlm", &self.vlm),
            ("vlm_premium", &self.vlm_premium),
        ]
        .iter()
        .filter(|(_, c)| c.calls > 0)
        .map(|(label, c)| cat_json(label, c))
        .collect();

        let total = self.total_cost_cents();
        serde_json::json!({
            "total_cost_usd": (total / 100.0 * 10000.0).round() / 10000.0,
            "total_calls": self.total_calls(),
            "categories": categories,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_accumulates_costs() {
        let tracker = CostTracker::new(true);

        tracker.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5.2-pro-2025-12-11".to_string(),
            input_tokens: 1000,
            output_tokens: 200,
        });
        tracker.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5.2-pro-2025-12-11".to_string(),
            input_tokens: 500,
            output_tokens: 100,
        });

        let snap = tracker.snapshot();
        assert_eq!(snap.chat.calls, 2);
        assert_eq!(snap.chat.input_tokens, 1500);
        assert_eq!(snap.chat.output_tokens, 300);
        assert!(snap.chat.cost_cents > 0.0);
    }

    #[test]
    fn tracker_disabled_is_noop() {
        let tracker = CostTracker::new(false);
        tracker.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5.2-pro".to_string(),
            input_tokens: 1000,
            output_tokens: 200,
        });
        let snap = tracker.snapshot();
        assert_eq!(snap.chat.calls, 0);
    }

    #[test]
    fn stt_call_recording() {
        let tracker = CostTracker::new(true);
        tracker.record_stt_call("whisper-large-v3-turbo", 30.0);
        let snap = tracker.snapshot();
        assert_eq!(snap.stt.calls, 1);
        assert_eq!(snap.stt.input_tokens, 750); // 30 * 25
    }

    #[test]
    fn embedding_call_recording() {
        let tracker = CostTracker::new(true);
        tracker.record_embedding_call("BAAI/bge-large-en-v1.5", 4000);
        let snap = tracker.snapshot();
        assert_eq!(snap.embeddings.calls, 1);
        assert_eq!(snap.embeddings.input_tokens, 1000); // 4000/4
    }

    #[test]
    fn total_cost_sums_all_categories() {
        let tracker = CostTracker::new(true);
        tracker.record(UsageRecord {
            category: CostCategory::Stt,
            model: "whisper-1".to_string(),
            input_tokens: 100,
            output_tokens: 0,
        });
        tracker.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5.2-pro".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        });
        let total = tracker.total_cost_cents();
        assert!(total > 0.0);
    }

    #[test]
    fn snapshot_format_digest_includes_totals() {
        let tracker = CostTracker::new(true);
        tracker.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5.2-pro".to_string(),
            input_tokens: 5000,
            output_tokens: 1000,
        });
        let snap = tracker.snapshot();
        let digest = snap.format_digest();
        assert!(digest.contains("Chat/LLM"));
        assert!(digest.contains("TOTAL"));
        assert!(digest.contains("$"));
    }

    #[test]
    fn snapshot_to_json_structure() {
        let tracker = CostTracker::new(true);
        tracker.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5.2-pro".to_string(),
            input_tokens: 1000,
            output_tokens: 200,
        });
        let json = tracker.snapshot().to_json();
        assert!(json["total_cost_usd"].as_f64().unwrap() > 0.0);
        assert_eq!(json["total_calls"].as_u64().unwrap(), 1);
        let cats = json["categories"].as_array().unwrap();
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0]["category"], "chat");
    }

    #[test]
    fn model_rate_known_models() {
        let (i, o) = model_rate("gpt-5.2-pro-2025-12-11");
        assert!(i > 0.0);
        assert!(o > 0.0);

        let (i, _) = model_rate("whisper-large-v3-turbo");
        assert!(i > 0.0);
    }

    #[test]
    fn groq_rates_are_cheaper_than_fallback() {
        // Regression guard: the bug we just fixed was Groq models falling
        // through to the OpenRouter-tier catch-all rate, overcharging ~5×.
        let (fallback_in, fallback_out) = model_rate("some-unknown-model");
        let (groq_in, groq_out) = model_rate("openai/gpt-oss-120b");
        assert!(
            groq_in < fallback_in,
            "Groq input rate ({groq_in}) should be < fallback ({fallback_in})"
        );
        assert!(
            groq_out < fallback_out,
            "Groq output rate ({groq_out}) should be < fallback ({fallback_out})"
        );
    }

    #[test]
    fn turbo_whisper_cheaper_than_non_turbo() {
        // Specificity check: whisper-large-v3-turbo must NOT collapse into
        // the generic whisper rate. Turbo is ~3× cheaper on Groq.
        let (turbo, _) = model_rate("whisper-large-v3-turbo");
        let (v3, _) = model_rate("whisper-large-v3");
        assert!(turbo < v3, "turbo ({turbo}) should be cheaper than v3 ({v3})");
    }

    #[test]
    fn llama_4_scout_priced_low() {
        // VLM lane runs on Llama 4 Scout via Groq. Should be ~$0.0001/$0.0003,
        // not the $0.001/$0.003 catch-all.
        let (in_rate, out_rate) = model_rate("meta-llama/llama-4-scout-17b-16e-instruct");
        assert!(in_rate < 0.0005, "Scout input ({in_rate}) too high");
        assert!(out_rate < 0.001, "Scout output ({out_rate}) too high");
    }

    #[test]
    fn clone_shares_state() {
        let t1 = CostTracker::new(true);
        let t2 = t1.clone();
        t1.record(UsageRecord {
            category: CostCategory::Chat,
            model: "gpt-5".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        });
        assert_eq!(t2.snapshot().chat.calls, 1);
    }
}
