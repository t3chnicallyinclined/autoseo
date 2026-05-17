//! LLM-driven candidate scorer. Sends batches of [`CandidateWindow`]s with their
//! attached features to an OpenAI-compatible chat model and parses back a score,
//! hook, refined start/end, and reasoning per candidate.
//!
//! The ranker is intentionally batched (default 10 candidates per LLM call) so a
//! 2-hour episode with ~240 candidates costs ~20 calls instead of 240.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ai_pipeline::ShowContext;
use crate::candidates::CandidateWindow;
use crate::openai::OpenAiClient;

/// One LLM-ranked clip. `refined_start_secs` / `refined_end_secs` are clamped on
/// our side to ±5s of the original candidate so an off-the-rails LLM can't make
/// us cut into adjacent content.
#[derive(Debug, Clone)]
pub struct RankedClip {
    pub candidate_index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
    pub score: i32,
    pub hook: String,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
pub struct Ranker {
    pub openai: OpenAiClient,
    pub chat_model: String,
    pub system_prompt: String,
    pub user_prompt_template: String,
    pub batch_size: usize,
    pub refine_drift_secs: f64,
}

impl Ranker {
    pub fn new(
        openai: OpenAiClient,
        chat_model: String,
        system_prompt: String,
        user_prompt_template: String,
    ) -> Self {
        Self {
            openai,
            chat_model,
            system_prompt,
            user_prompt_template,
            batch_size: 10,
            refine_drift_secs: 5.0,
        }
    }

    /// Score every candidate via the LLM, batched, then return the top `top_k`
    /// sorted by score desc. Set `top_k = usize::MAX` to keep them all.
    pub async fn rank(
        &self,
        candidates: &[CandidateWindow],
        top_k: usize,
        show_context: Option<&ShowContext>,
    ) -> Result<Vec<RankedClip>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_ranked: Vec<RankedClip> = Vec::with_capacity(candidates.len());

        let batch_size = self.batch_size.max(1);
        for (batch_idx, batch) in candidates.chunks(batch_size).enumerate() {
            let batch_start_index = batch_idx * batch_size;
            let batch_json = serialize_batch(batch, batch_start_index);

            let user = self.build_user_prompt(&batch_json, show_context);

            let raw = self
                .openai
                .chat_json(&self.chat_model, &self.system_prompt, &user)
                .await
                .with_context(|| format!("ranker batch {batch_idx}"))?;
            let parsed: RankerResponse =
                serde_json::from_value(raw).context("parse ranker JSON")?;

            for clip in parsed.clips {
                let local_idx = clip.index;
                let absolute_idx = batch_start_index + local_idx;
                let Some(c) = candidates.get(absolute_idx) else {
                    tracing::warn!(
                        absolute_idx,
                        batch_size = batch.len(),
                        "ranker returned out-of-range index; skipping"
                    );
                    continue;
                };

                let (refined_start, refined_end) =
                    self.clamp_refinement(c, clip.refined_start_secs, clip.refined_end_secs);

                all_ranked.push(RankedClip {
                    candidate_index: absolute_idx,
                    start_secs: refined_start,
                    end_secs: refined_end,
                    score: clip.score.clamp(0, 100),
                    hook: clip.hook.trim().to_string(),
                    reasoning: clip.reasoning.trim().to_string(),
                });
            }
        }

        all_ranked.sort_by(|a, b| b.score.cmp(&a.score));
        all_ranked.truncate(top_k);
        Ok(all_ranked)
    }

    fn build_user_prompt(
        &self,
        candidates_json: &str,
        show_context: Option<&ShowContext>,
    ) -> String {
        let mut out = self
            .user_prompt_template
            .replace("{{candidates_json}}", candidates_json);

        let (show_name, hosts, guest) = match show_context {
            Some(ctx) => (
                ctx.show_name.as_deref().unwrap_or("").to_string(),
                ctx.hosts.join(", "),
                ctx.guest.as_deref().unwrap_or("").to_string(),
            ),
            None => (String::new(), String::new(), String::new()),
        };
        out = out
            .replace("{{show_name}}", &show_name)
            .replace("{{hosts}}", &hosts)
            .replace("{{guest}}", &guest);
        out
    }

    fn clamp_refinement(
        &self,
        c: &CandidateWindow,
        suggested_start: f64,
        suggested_end: f64,
    ) -> (f64, f64) {
        let drift = self.refine_drift_secs;
        let lo_start = (c.start_secs - drift).max(0.0);
        let hi_start = c.start_secs + drift;
        let start = suggested_start.clamp(lo_start, hi_start);

        let lo_end = c.end_secs - drift;
        let hi_end = c.end_secs + drift;
        let mut end = suggested_end.clamp(lo_end, hi_end);
        if end <= start {
            end = c.end_secs.max(start + 1.0);
        }
        (start, end)
    }
}

fn serialize_batch(batch: &[CandidateWindow], start_index_in_episode: usize) -> String {
    let payload: Vec<_> = batch
        .iter()
        .enumerate()
        .map(|(local_idx, c)| BatchCandidate {
            index: local_idx,
            episode_index: start_index_in_episode + local_idx,
            start_secs: c.start_secs,
            end_secs: c.end_secs,
            transcript: c.transcript.clone(),
            linguistic: LinguisticPayload {
                conflict_markers: c.linguistic.conflict_marker_count,
                strong_claim: c.linguistic.strong_claim_count,
                confessional: c.linguistic.confessional_count,
                topic_shift: c.linguistic.topic_shift_count,
                questions: c.linguistic.question_count,
                numbers: c.linguistic.number_count,
                short_declaratives: c.linguistic.short_declarative_count,
                quotable_lines: c.linguistic.quotable_lines.clone(),
            },
            prosody: ProsodyPayload {
                rms_peak_db: c.rms_peak_db,
                rms_mean_db: c.rms_mean_db,
                f0_mean_hz: c.f0_mean_hz,
                f0_variance_hz2: c.f0_variance_hz2,
                f0_peak_hz: c.f0_peak_hz,
                speaking_rate_wps: c.speaking_rate_wps,
            },
            novelty: c.novelty_score,
        })
        .collect();
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "[]".to_string())
}

#[derive(Debug, Serialize)]
struct BatchCandidate {
    index: usize,
    episode_index: usize,
    start_secs: f64,
    end_secs: f64,
    transcript: String,
    linguistic: LinguisticPayload,
    prosody: ProsodyPayload,
    novelty: Option<f64>,
}

#[derive(Debug, Serialize)]
struct LinguisticPayload {
    conflict_markers: usize,
    strong_claim: usize,
    confessional: usize,
    topic_shift: usize,
    questions: usize,
    numbers: usize,
    short_declaratives: usize,
    quotable_lines: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ProsodyPayload {
    rms_peak_db: Option<f64>,
    rms_mean_db: Option<f64>,
    f0_mean_hz: Option<f64>,
    f0_variance_hz2: Option<f64>,
    f0_peak_hz: Option<f64>,
    speaking_rate_wps: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RankerResponse {
    clips: Vec<RankerClip>,
}

#[derive(Debug, Deserialize)]
struct RankerClip {
    index: usize,
    score: i32,
    #[serde(default)]
    hook: String,
    #[serde(default)]
    refined_start_secs: f64,
    #[serde(default)]
    refined_end_secs: f64,
    #[serde(default)]
    reasoning: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateWindow;
    use crate::linguistic_markers::LinguisticFeatures;

    fn dummy_candidate(start: f64, end: f64) -> CandidateWindow {
        CandidateWindow {
            start_secs: start,
            end_secs: end,
            transcript: "hello world".to_string(),
            word_count: 2,
            linguistic: LinguisticFeatures::default(),
            rms_peak_db: Some(-15.0),
            rms_mean_db: Some(-22.0),
            f0_mean_hz: Some(185.0),
            f0_variance_hz2: Some(420.0),
            f0_peak_hz: Some(310.0),
            speaking_rate_wps: Some(2.5),
            novelty_score: Some(0.4),
        }
    }

    #[test]
    fn serialize_batch_emits_per_candidate_records() {
        let batch = vec![dummy_candidate(10.0, 70.0), dummy_candidate(60.0, 120.0)];
        let json = serialize_batch(&batch, 0);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        let arr = parsed.as_array().expect("array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["index"], 0);
        assert_eq!(arr[1]["index"], 1);
        assert_eq!(arr[1]["episode_index"], 1);
        assert_eq!(arr[0]["start_secs"], 10.0);
        assert!(arr[0]["prosody"]["rms_peak_db"].is_number());
    }

    #[test]
    fn clamp_refinement_caps_drift() {
        let openai = OpenAiClient::new("https://example.com".into(), "x".into());
        let r = Ranker::new(openai, "model".into(), "sys".into(), "user".into());
        let c = dummy_candidate(60.0, 120.0);

        // LLM suggests way outside the budget — clamp to ±5s.
        let (s, e) = r.clamp_refinement(&c, 30.0, 200.0);
        assert!((s - 55.0).abs() < 1e-6, "start should clamp to 55, got {s}");
        assert!((e - 125.0).abs() < 1e-6, "end should clamp to 125, got {e}");

        // LLM suggests modest shift inside budget — keep as-is.
        let (s2, e2) = r.clamp_refinement(&c, 62.0, 118.0);
        assert_eq!(s2, 62.0);
        assert_eq!(e2, 118.0);

        // Degenerate suggestion where end < start — bump end forward.
        let (s3, e3) = r.clamp_refinement(&c, 70.0, 60.0);
        assert_eq!(s3, 65.0);
        assert!(e3 > s3, "end should be after start, got start={s3} end={e3}");
    }

    #[test]
    fn build_user_prompt_substitutes_placeholders() {
        let openai = OpenAiClient::new("https://example.com".into(), "x".into());
        let r = Ranker::new(
            openai,
            "model".into(),
            "sys".into(),
            "show={{show_name}} hosts={{hosts}} guest={{guest}}\nCANDS:\n{{candidates_json}}"
                .into(),
        );
        let ctx = ShowContext {
            show_name: Some("TFATK".into()),
            hosts: vec!["Brendan".into(), "Bryan".into()],
            guest: Some("Joe".into()),
            evidence: vec![],
        };
        let out = r.build_user_prompt("[]", Some(&ctx));
        assert!(out.contains("show=TFATK"));
        assert!(out.contains("hosts=Brendan, Bryan"));
        assert!(out.contains("guest=Joe"));
        assert!(out.contains("CANDS:\n[]"));

        let out_no_ctx = r.build_user_prompt("[]", None);
        assert!(out_no_ctx.contains("show="));
        assert!(!out_no_ctx.contains("{{show_name}}"));
    }

    #[test]
    fn parses_ranker_response() {
        let body = r#"{
          "clips": [
            {"index": 0, "score": 85, "hook": "the punchline", "refined_start_secs": 12.5, "refined_end_secs": 71.0, "reasoning": "strong claim + payoff"},
            {"index": 1, "score": 40, "hook": "filler", "refined_start_secs": 60.0, "refined_end_secs": 120.0, "reasoning": "long setup no payoff"}
          ]
        }"#;
        let parsed: RankerResponse = serde_json::from_str(body).expect("parse");
        assert_eq!(parsed.clips.len(), 2);
        assert_eq!(parsed.clips[0].score, 85);
        assert_eq!(parsed.clips[0].hook, "the punchline");
    }
}
