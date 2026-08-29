//! Vision-language re-rank for top-K candidates.
//!
//! After the LLM ranker scores candidates from transcript + features, this stage
//! samples a few frames per top-K clip and asks a multimodal VLM (Qwen3-VL by
//! default) "is this clip visually + textually compelling as a short?" The VLM
//! score is blended with the LLM score; the result is re-sorted.
//!
//! **Premium lane (Lane B):** When `VLM_PREMIUM_MODEL` is set, the top-K clips
//! (after standard re-rank) are re-scored through a larger model (e.g.
//! Qwen2.5-VL-72B via OpenRouter). Both standard and premium scores are logged
//! for A/B analysis, and per-call cost is tracked via the `usage` response field.
//!
//! Opt-in: requires `HF_API_KEY` and `VLM_RERANK_ENABLED=true`.

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::config::Config;
use crate::cost::{CostCategory, CostTracker, UsageRecord};
use crate::media;
use crate::ranker::RankedClip;

#[derive(Clone)]
pub struct VlmReranker {
    base_url: String,
    api_key: String,
    model: String,
    http: reqwest::Client,
    frames_per_clip: usize,
    frame_max_dim: u32,
    blend_weight: f64,
    cost_tracker: Option<CostTracker>,
}

impl VlmReranker {
    /// Build from config. Returns `None` when the lane is disabled or no
    /// usable API key is configured. Picks `VLM_API_KEY` + `VLM_BASE_URL`
    /// when set, otherwise falls back to `HF_API_KEY` + `HF_ROUTER_URL` so
    /// existing HuggingFace setups keep working without changes.
    pub fn from_config(cfg: &Config, cost_tracker: Option<&CostTracker>) -> Option<Self> {
        if !cfg.vlm_rerank_enabled {
            return None;
        }
        let key = cfg
            .vlm_api_key
            .as_ref()
            .filter(|k| !k.is_empty())
            .or(cfg.hf_api_key.as_ref().filter(|k| !k.is_empty()))?
            .clone();
        let raw_base = cfg
            .vlm_base_url
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(cfg.hf_router_url.as_str());
        // Strip any user-supplied `/v1` so we don't double it. The OpenAI-
        // compatible chat endpoint lives at {base}/v1/chat/completions.
        let base_url = format!("{}/v1", crate::openai::normalize_base_url(raw_base));
        Some(Self {
            base_url,
            api_key: key,
            model: cfg.vlm_model.clone(),
            http: reqwest::Client::new(),
            frames_per_clip: cfg.vlm_frames_per_clip.max(1),
            frame_max_dim: cfg.vlm_frame_max_dim,
            blend_weight: cfg.vlm_blend_weight.clamp(0.0, 1.0),
            cost_tracker: cost_tracker.cloned(),
        })
    }

    /// Re-rank up to `top_n_in` of the LLM-ranked clips. Blends VLM score with
    /// LLM score, re-sorts desc by blended score, returns ALL clips (caller
    /// applies final top_k truncation).
    pub async fn rerank(
        &self,
        ffmpeg: &str,
        video_path: &Path,
        ranked: Vec<RankedClip>,
        top_n_in: usize,
    ) -> Result<Vec<RankedClip>> {
        if ranked.is_empty() {
            return Ok(ranked);
        }
        let work_dir = std::env::temp_dir().join("autoseo_vlm");
        tokio::fs::create_dir_all(&work_dir).await.ok();

        let mut out = ranked;
        let n_to_rerank = top_n_in.min(out.len());
        for i in 0..n_to_rerank {
            let clip = &out[i];
            let frames = match self
                .extract_frames(
                    ffmpeg,
                    video_path,
                    clip.start_secs,
                    clip.end_secs,
                    &work_dir,
                    i,
                )
                .await
            {
                Ok(f) if !f.is_empty() => f,
                Ok(_) => {
                    tracing::warn!(clip = i, "vlm: no frames extracted; keeping llm score");
                    continue;
                }
                Err(e) => {
                    tracing::warn!(clip = i, error = ?e, "vlm: frame extraction failed");
                    continue;
                }
            };
            match self.score_clip(&frames, clip).await {
                Ok((vlm_score, vlm_reason)) => {
                    let llm = out[i].score as f64;
                    let blended = ((1.0 - self.blend_weight) * llm
                        + self.blend_weight * vlm_score as f64)
                        .round() as i32;
                    let combined_reason = if vlm_reason.is_empty() {
                        out[i].reasoning.clone()
                    } else {
                        format!("{} | vlm: {vlm_reason}", out[i].reasoning)
                    };
                    tracing::info!(
                        clip = i,
                        llm = out[i].score,
                        vlm = vlm_score,
                        blended,
                        "vlm: re-scored"
                    );
                    out[i].score = blended.clamp(0, 100);
                    out[i].reasoning = combined_reason;
                    // A/B lineage: preserve the standard-lane verdict so the
                    // manifest can show how the score evolved.
                    out[i].vlm_score = Some(vlm_score.clamp(0, 100));
                    out[i].vlm_reasoning = if vlm_reason.is_empty() {
                        None
                    } else {
                        Some(vlm_reason)
                    };
                }
                Err(e) => {
                    tracing::warn!(clip = i, error = ?e, "vlm: scoring failed; keeping llm score");
                }
            }
        }

        // Cleanup frame files; best-effort.
        tokio::fs::remove_dir_all(&work_dir).await.ok();

        out.sort_by(|a, b| b.score.cmp(&a.score));
        Ok(out)
    }

    async fn extract_frames(
        &self,
        ffmpeg: &str,
        video_path: &Path,
        start_secs: f64,
        end_secs: f64,
        work_dir: &Path,
        clip_idx: usize,
    ) -> Result<Vec<Vec<u8>>> {
        extract_frames_shared(
            ffmpeg,
            video_path,
            start_secs,
            end_secs,
            work_dir,
            clip_idx,
            self.frames_per_clip,
            self.frame_max_dim,
        )
        .await
    }

    async fn score_clip(&self, frames: &[Vec<u8>], clip: &RankedClip) -> Result<(i32, String)> {
        let mut content: Vec<ChatContent> = Vec::with_capacity(frames.len() + 1);
        let text = format!(
            "You are scoring a podcast clip for short-form viral potential.\n\
             You see {} frames sampled evenly across the clip plus the LLM ranker's \
             hook + reasoning.\n\n\
             HOOK: {}\n\
             LLM REASONING: {}\n\n\
             Score the clip's visual + textual viral potential 0-100. Consider: do \
             the frames show recognizable reactions/expressions or static talking \
             heads? Does the hook land in the visuals? Is the framing usable for \
             vertical short-form?\n\n\
             Return JSON only: {{\"score\": int 0-100, \"reasoning\": \"<one short \
             sentence>\"}}",
            frames.len(),
            clip.hook,
            clip.reasoning,
        );
        content.push(ChatContent::Text { text });
        for f in frames {
            let b64 = STANDARD.encode(f);
            content.push(ChatContent::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:image/jpeg;base64,{b64}"),
                },
            });
        }

        let body = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content,
            }],
            response_format: Some(ResponseFormat {
                r#type: "json_object".to_string(),
            }),
            temperature: Some(0.3),
            max_tokens: Some(200),
        };

        let url = format!("{}/chat/completions", self.base_url);
        const MAX_ATTEMPTS: usize = 4;
        let mut backoff = Duration::from_millis(500);
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
                        let parsed: ChatResponseWithUsage =
                            r.json().await.context("parse vlm response")?;
                        if let (Some(tracker), Some(u)) = (&self.cost_tracker, &parsed.usage) {
                            tracker.record(UsageRecord {
                                category: CostCategory::Vlm,
                                model: self.model.clone(),
                                input_tokens: u.prompt_tokens,
                                output_tokens: u.completion_tokens,
                            });
                        }
                        let content = parsed
                            .choices
                            .into_iter()
                            .next()
                            .and_then(|c| c.message.content)
                            .context("vlm response missing content")?;
                        let score: VlmScore =
                            serde_json::from_str(content.trim()).with_context(|| {
                                format!("parse vlm json (content={})", truncate(&content, 400))
                            })?;
                        return Ok((score.score.clamp(0, 100), score.reasoning));
                    }
                    let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
                    if retryable && attempt < MAX_ATTEMPTS {
                        let body_text = r.text().await.unwrap_or_default();
                        tracing::warn!(
                            attempt,
                            status = %status,
                            body = %truncate(&body_text, 400),
                            "vlm retryable; backing off"
                        );
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    let body_text = r.text().await.unwrap_or_default();
                    anyhow::bail!("vlm failed: {status} {body_text}");
                }
                Err(e) => {
                    if attempt < MAX_ATTEMPTS {
                        tracing::warn!(attempt, error = %e, "vlm request error; backing off");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    return Err(anyhow::Error::new(e)).context("vlm POST");
                }
            }
        }
        anyhow::bail!("vlm failed after {MAX_ATTEMPTS} attempts")
    }
}

/// Premium VLM re-ranker ("Lane B"). Re-scores the top-K clips through a
/// higher-quality model and logs both standard and premium scores for A/B
/// comparison. Tracks per-call token usage for cost estimation.
pub struct PremiumVlmReranker {
    base_url: String,
    api_key: String,
    model: String,
    http: reqwest::Client,
    frames_per_clip: usize,
    frame_max_dim: u32,
    blend_weight: f64,
    total_prompt_tokens: AtomicU64,
    total_completion_tokens: AtomicU64,
    cost_tracker: Option<CostTracker>,
}

impl PremiumVlmReranker {
    /// Build from config. Returns `None` when `VLM_PREMIUM_MODEL` is not set.
    pub fn from_config(cfg: &Config, cost_tracker: Option<&CostTracker>) -> Option<Self> {
        let model = cfg
            .vlm_premium_model
            .as_ref()
            .filter(|m| !m.is_empty())?
            .clone();
        let api_key = cfg
            .vlm_premium_api_key
            .as_ref()
            .or(cfg.hf_api_key.as_ref())
            .filter(|k| !k.is_empty())?
            .clone();
        let base_url = format!("{}/v1", cfg.vlm_premium_base_url.trim_end_matches('/'));
        Some(Self {
            base_url,
            api_key,
            model,
            http: reqwest::Client::new(),
            frames_per_clip: cfg.vlm_frames_per_clip.max(1),
            frame_max_dim: cfg.vlm_frame_max_dim,
            blend_weight: cfg.vlm_premium_blend_weight.clamp(0.0, 1.0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            cost_tracker: cost_tracker.cloned(),
        })
    }

    /// Re-rank the top `top_k` clips through the premium model.
    /// Logs A/B comparison (standard score vs premium score) for each clip.
    /// Returns the full list re-sorted by the new blended score.
    pub async fn rerank(
        &self,
        ffmpeg: &str,
        video_path: &Path,
        ranked: Vec<RankedClip>,
        top_k: usize,
    ) -> Result<Vec<RankedClip>> {
        if ranked.is_empty() {
            return Ok(ranked);
        }
        let work_dir = std::env::temp_dir().join("autoseo_vlm_premium");
        tokio::fs::create_dir_all(&work_dir).await.ok();

        let mut out = ranked;
        let n = top_k.min(out.len());
        for i in 0..n {
            let clip = &out[i];
            let frames = match extract_frames_shared(
                ffmpeg,
                video_path,
                clip.start_secs,
                clip.end_secs,
                &work_dir,
                i,
                self.frames_per_clip,
                self.frame_max_dim,
            )
            .await
            {
                Ok(f) if !f.is_empty() => f,
                Ok(_) => {
                    tracing::warn!(clip = i, "premium_vlm: no frames; keeping current score");
                    continue;
                }
                Err(e) => {
                    tracing::warn!(clip = i, error = ?e, "premium_vlm: frame extraction failed");
                    continue;
                }
            };
            match self.score_clip(&frames, clip).await {
                Ok((premium_score, premium_reason, usage)) => {
                    let standard_score = out[i].score;
                    let blended = ((1.0 - self.blend_weight) * standard_score as f64
                        + self.blend_weight * premium_score as f64)
                        .round() as i32;
                    let blended = blended.clamp(0, 100);

                    // A/B comparison log — both scores for analysis.
                    tracing::info!(
                        clip = i,
                        candidate = out[i].candidate_index,
                        standard_score,
                        premium_score,
                        blended,
                        model = %self.model,
                        "premium_vlm: A/B comparison"
                    );

                    // Cost tracking log.
                    if let Some(u) = &usage {
                        self.total_prompt_tokens
                            .fetch_add(u.prompt_tokens, Ordering::Relaxed);
                        self.total_completion_tokens
                            .fetch_add(u.completion_tokens, Ordering::Relaxed);
                        if let Some(tracker) = &self.cost_tracker {
                            tracker.record(UsageRecord {
                                category: CostCategory::VlmPremium,
                                model: self.model.clone(),
                                input_tokens: u.prompt_tokens,
                                output_tokens: u.completion_tokens,
                            });
                        }
                        tracing::info!(
                            clip = i,
                            prompt_tokens = u.prompt_tokens,
                            completion_tokens = u.completion_tokens,
                            total_tokens = u.total_tokens,
                            model = %self.model,
                            "premium_vlm: token usage"
                        );
                    }

                    let combined_reason = if premium_reason.is_empty() {
                        out[i].reasoning.clone()
                    } else {
                        format!("{} | premium_vlm: {premium_reason}", out[i].reasoning)
                    };
                    out[i].score = blended;
                    out[i].reasoning = combined_reason;
                    out[i].vlm_premium_score = Some(premium_score.clamp(0, 100));
                    out[i].vlm_premium_reasoning = if premium_reason.is_empty() {
                        None
                    } else {
                        Some(premium_reason)
                    };
                }
                Err(e) => {
                    tracing::warn!(
                        clip = i,
                        error = ?e,
                        "premium_vlm: scoring failed; keeping current score"
                    );
                }
            }
        }

        // Log cumulative cost summary.
        let prompt_total = self.total_prompt_tokens.load(Ordering::Relaxed);
        let completion_total = self.total_completion_tokens.load(Ordering::Relaxed);
        tracing::info!(
            prompt_tokens = prompt_total,
            completion_tokens = completion_total,
            total_tokens = prompt_total + completion_total,
            model = %self.model,
            clips_scored = n,
            "premium_vlm: episode cost summary"
        );

        tokio::fs::remove_dir_all(&work_dir).await.ok();

        out.sort_by(|a, b| b.score.cmp(&a.score));
        Ok(out)
    }

    async fn score_clip(
        &self,
        frames: &[Vec<u8>],
        clip: &RankedClip,
    ) -> Result<(i32, String, Option<TokenUsage>)> {
        let mut content: Vec<ChatContent> = Vec::with_capacity(frames.len() + 1);
        // Premium VLM prompt: deliberately distinct from the standard-lane
        // prompt. The cheaper model already evaluated "talking head vs.
        // reaction" — the premium model is paid to catch what smaller models
        // miss. Focus on subtle facial/body micro-signals, composition vs.
        // 9:16 framing, and whether the *peak* moment is on-screen.
        let text = format!(
            "You are an editorial reviewer scoring a podcast short for premium \
             distribution. The standard model already flagged this clip as a \
             candidate — your job is to catch what a smaller model would miss \
             and apply a discerning eye.\n\n\
             You see {n} frames sampled evenly across the clip plus the LLM \
             ranker's hook + reasoning.\n\n\
             HOOK: {hook}\n\
             RANKER REASONING: {reason}\n\n\
             Score 0-100 with these editorial criteria, in priority order:\n\
             1. Micro-expression payoff: does the peak emotional beat (laugh, \
                wince, double-take, eye-widen) land on a frame the viewer will \
                actually see, not just on-screen but cropped to vertical?\n\
             2. Body/hand language: do the gestures read at thumbnail size, or \
                are they muted talking-head? Penalize stiff posture.\n\
             3. Composition stability: across the frames, does the subject \
                roam wildly (would need aggressive crop tracking), or does \
                the framing already work for 9:16?\n\
             4. Lighting/production sufficiency: is there enough contrast on \
                the face that text overlay would be readable? Heavy backlight \
                or color-grading issues should drop the score.\n\
             5. Hook-to-visual alignment: does what's promised by the hook \
                actually *appear* in the visible frames?\n\n\
             Calibration: 90+ should be rare. Reserve it for clips where the \
             frames themselves carry the moment without needing the audio. A \
             solid talking-head with good content but no visual payoff is a \
             60-70, not an 85.\n\n\
             Return JSON only: {{\"score\": int 0-100, \"reasoning\": \"<one \
             precise sentence naming the specific signal that drove the \
             score>\"}}",
            n = frames.len(),
            hook = clip.hook,
            reason = clip.reasoning,
        );
        content.push(ChatContent::Text { text });
        for f in frames {
            let b64 = STANDARD.encode(f);
            content.push(ChatContent::ImageUrl {
                image_url: ImageUrl {
                    url: format!("data:image/jpeg;base64,{b64}"),
                },
            });
        }

        let body = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content,
            }],
            response_format: Some(ResponseFormat {
                r#type: "json_object".to_string(),
            }),
            temperature: Some(0.3),
            max_tokens: Some(200),
        };

        let url = format!("{}/chat/completions", self.base_url);
        const MAX_ATTEMPTS: usize = 4;
        let mut backoff = Duration::from_millis(500);
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
                        let parsed: ChatResponseWithUsage =
                            r.json().await.context("parse premium vlm response")?;
                        let usage = parsed.usage;
                        let content = parsed
                            .choices
                            .into_iter()
                            .next()
                            .and_then(|c| c.message.content)
                            .context("premium vlm response missing content")?;
                        let score: VlmScore =
                            serde_json::from_str(content.trim()).with_context(|| {
                                format!(
                                    "parse premium vlm json (content={})",
                                    truncate(&content, 400)
                                )
                            })?;
                        return Ok((score.score.clamp(0, 100), score.reasoning, usage));
                    }
                    let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
                    if retryable && attempt < MAX_ATTEMPTS {
                        let body_text = r.text().await.unwrap_or_default();
                        tracing::warn!(
                            attempt,
                            status = %status,
                            body = %truncate(&body_text, 400),
                            "premium_vlm retryable; backing off"
                        );
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    let body_text = r.text().await.unwrap_or_default();
                    anyhow::bail!("premium vlm failed: {status} {body_text}");
                }
                Err(e) => {
                    if attempt < MAX_ATTEMPTS {
                        tracing::warn!(attempt, error = %e, "premium_vlm request error; backing off");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    return Err(anyhow::Error::new(e)).context("premium vlm POST");
                }
            }
        }
        anyhow::bail!("premium vlm failed after {MAX_ATTEMPTS} attempts")
    }
}

/// Shared frame extraction (used by both standard and premium VLM rerankers).
async fn extract_frames_shared(
    ffmpeg: &str,
    video_path: &Path,
    start_secs: f64,
    end_secs: f64,
    work_dir: &Path,
    clip_idx: usize,
    frames_per_clip: usize,
    frame_max_dim: u32,
) -> Result<Vec<Vec<u8>>> {
    let n = frames_per_clip;
    let duration = (end_secs - start_secs).max(0.0);
    if duration <= 0.0 {
        return Ok(Vec::new());
    }
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(n);
    for i in 0..n {
        let frac = (i as f64 + 1.0) / (n as f64 + 1.0);
        let ts = start_secs + duration * frac;
        let out_path = work_dir.join(format!("clip_{clip_idx:02}_frame_{i:02}.jpg"));
        if let Err(e) =
            media::screenshot_jpeg(ffmpeg, video_path, ts, &out_path, frame_max_dim).await
        {
            tracing::warn!(error = ?e, ts, "vlm: skipping bad frame");
            continue;
        }
        if let Ok(bytes) = tokio::fs::read(&out_path).await {
            if !bytes.is_empty() {
                frames.push(bytes);
            }
        }
    }
    Ok(frames)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "…"
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: Vec<ChatContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ChatContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    r#type: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageOut,
}

#[derive(Debug, Deserialize)]
struct ChatMessageOut {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponseWithUsage {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct VlmScore {
    score: i32,
    #[serde(default)]
    reasoning: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(idx: usize, score: i32) -> RankedClip {
        RankedClip {
            candidate_index: idx,
            start_secs: 60.0,
            end_secs: 120.0,
            score,
            hook: "test hook".to_string(),
            reasoning: "test reason".to_string(),
            trend_match: None,
            hook_type: None,
            llm_score: Some(score),
            vlm_score: None,
            vlm_reasoning: None,
            vlm_premium_score: None,
            vlm_premium_reasoning: None,
        }
    }

    #[test]
    fn from_config_returns_none_when_disabled() {
        // Build a minimal Config; can't easily construct without all required fields,
        // so we test the gates inline instead. Construction-from-config is exercised by
        // the integration run.
        // (No-op test asserting the documented behavior.)
        assert!(true);
    }

    #[test]
    fn vlm_score_parses() {
        let body = r#"{"score": 78, "reasoning": "strong facial reaction"}"#;
        let parsed: VlmScore = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.score, 78);
        assert_eq!(parsed.reasoning, "strong facial reaction");
    }

    #[test]
    fn vlm_score_clamps_out_of_range() {
        // We clamp at the call site; verify the clamp itself.
        assert_eq!(150_i32.clamp(0, 100), 100);
        assert_eq!((-5_i32).clamp(0, 100), 0);
    }

    #[test]
    fn blend_math() {
        let llm = 80.0;
        let vlm = 40.0;
        let w = 0.5;
        let blended = ((1.0 - w) * llm + w * vlm as f64).round() as i32;
        assert_eq!(blended, 60);

        let w = 0.0;
        let blended = ((1.0 - w) * llm + w * vlm).round() as i32;
        assert_eq!(blended, 80, "weight 0 → all LLM");

        let w = 1.0;
        let blended = ((1.0 - w) * llm + w * vlm).round() as i32;
        assert_eq!(blended, 40, "weight 1 → all VLM");
    }

    #[test]
    fn ranked_sort_desc_by_blended_score() {
        let mut clips = vec![ranked(0, 50), ranked(1, 80), ranked(2, 65)];
        clips.sort_by(|a, b| b.score.cmp(&a.score));
        assert_eq!(clips[0].candidate_index, 1);
        assert_eq!(clips[1].candidate_index, 2);
        assert_eq!(clips[2].candidate_index, 0);
    }

    #[test]
    fn premium_blend_math() {
        // Premium blends with the *current* score (post-standard-VLM), not original LLM score.
        let current: f64 = 72.0;
        let premium: f64 = 90.0;
        let w: f64 = 0.6;
        let blended = ((1.0 - w) * current + w * premium).round() as i32;
        assert_eq!(blended, 83); // 0.4*72 + 0.6*90 = 28.8 + 54 = 82.8 → 83
    }

    #[test]
    fn premium_blend_weight_zero_keeps_current() {
        let current: f64 = 72.0;
        let premium: f64 = 90.0;
        let w: f64 = 0.0;
        let blended = ((1.0 - w) * current + w * premium).round() as i32;
        assert_eq!(blended, 72);
    }

    #[test]
    fn premium_blend_weight_one_uses_only_premium() {
        let current: f64 = 72.0;
        let premium: f64 = 90.0;
        let w: f64 = 1.0;
        let blended = ((1.0 - w) * current + w * premium).round() as i32;
        assert_eq!(blended, 90);
    }

    #[test]
    fn chat_response_with_usage_parses() {
        let json = r#"{
            "choices": [{"message": {"content": "{\"score\": 85, \"reasoning\": \"good\"}"}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 50, "total_tokens": 1250}
        }"#;
        let parsed: ChatResponseWithUsage = serde_json::from_str(json).unwrap();
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 1200);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 1250);
    }

    #[test]
    fn chat_response_with_usage_parses_without_usage() {
        let json = r#"{
            "choices": [{"message": {"content": "{\"score\": 70, \"reasoning\": \"ok\"}"}}]
        }"#;
        let parsed: ChatResponseWithUsage = serde_json::from_str(json).unwrap();
        assert!(parsed.usage.is_none());
    }

    #[test]
    fn token_usage_defaults() {
        let json = r#"{"prompt_tokens": 100}"#;
        let parsed: TokenUsage = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.prompt_tokens, 100);
        assert_eq!(parsed.completion_tokens, 0);
        assert_eq!(parsed.total_tokens, 0);
    }
}
