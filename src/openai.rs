use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::cost::{CostCategory, CostTracker, UsageRecord};

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
    cost_tracker: Option<CostTracker>,
}

/// Normalize an OpenAI-compatible base URL to autoseo's convention: host root
/// only (no `/v1` suffix, no trailing slash). The clipper code paths append
/// `/v1/<endpoint>` themselves, so a base that already ends in `/v1` would
/// otherwise double — `https://api.groq.com/openai/v1` → `/openai/v1/v1/…`.
/// Accepting either form lets users paste the provider's documented URL
/// verbatim.
pub fn normalize_base_url(s: &str) -> String {
    let s = s.trim_end_matches('/');
    let s = s.strip_suffix("/v1").unwrap_or(s);
    s.trim_end_matches('/').to_string()
}

impl OpenAiClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url: normalize_base_url(&base_url),
            api_key,
            http: reqwest::Client::new(),
            cost_tracker: None,
        }
    }

    /// Attach a cost tracker to record token usage from API calls.
    pub fn with_cost_tracker(mut self, tracker: CostTracker) -> Self {
        self.cost_tracker = Some(tracker);
        self
    }

    /// Record token usage to the cost tracker if present.
    fn record_usage(&self, category: CostCategory, model: &str, input: u64, output: u64) {
        if let Some(ref tracker) = self.cost_tracker {
            tracker.record(UsageRecord {
                category,
                model: model.to_string(),
                input_tokens: input,
                output_tokens: output,
            });
        }
    }

    /// Access the cost tracker (if attached) for external recording (e.g. STT).
    pub fn cost_tracker(&self) -> Option<&CostTracker> {
        self.cost_tracker.as_ref()
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.bearer_auth(&self.api_key)
    }

    pub async fn transcribe_text(
        &self,
        model: &str,
        audio_path: &std::path::Path,
    ) -> anyhow::Result<TranscriptionText> {
        match self
            .transcribe_internal(model, audio_path, "verbose_json", &[])
            .await
        {
            Ok(SttResponse::Verbose(v)) => Ok(TranscriptionText {
                text: v.text,
                segments: v.segments,
                words: v.words,
            }),
            Ok(SttResponse::Json(j)) => Ok(TranscriptionText {
                text: j.text,
                segments: Vec::new(),
                words: Vec::new(),
            }),
            Err(e) => Err(e),
        }
    }

    /// Transcribe with word-level timestamps. Returns segments AND words.
    /// Requires a provider that honors `timestamp_granularities[]=word` (e.g.
    /// Groq's `whisper-large-v3-turbo`). OpenAI's `whisper-1` does not support
    /// word-level granularity; the response will come back with an empty `words`.
    pub async fn transcribe_words(
        &self,
        model: &str,
        audio_path: &std::path::Path,
    ) -> anyhow::Result<TranscriptionText> {
        match self
            .transcribe_internal(model, audio_path, "verbose_json", &["word"])
            .await
        {
            Ok(SttResponse::Verbose(v)) => Ok(TranscriptionText {
                text: v.text,
                segments: v.segments,
                words: v.words,
            }),
            Ok(SttResponse::Json(j)) => Ok(TranscriptionText {
                text: j.text,
                segments: Vec::new(),
                words: Vec::new(),
            }),
            Err(e) => Err(e),
        }
    }

    async fn transcribe_internal(
        &self,
        model: &str,
        audio_path: &std::path::Path,
        primary_response_format: &str,
        granularities: &[&str],
    ) -> anyhow::Result<SttResponse> {
        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        let try_once =
            |response_format: &str, part: reqwest::multipart::Part| -> reqwest::RequestBuilder {
                let mut form = reqwest::multipart::Form::new()
                    .text("model", model.to_string())
                    .text("response_format", response_format.to_string())
                    .part("file", part);
                for g in granularities {
                    form = form.text("timestamp_granularities[]", g.to_string());
                }
                self.auth(self.http.post(url.clone())).multipart(form)
            };

        const MAX_ATTEMPTS: usize = 5;
        let mut backoff = Duration::from_millis(400);

        let mut last_status: Option<reqwest::StatusCode> = None;
        let mut last_body: String = String::new();

        for attempt in 1..=MAX_ATTEMPTS {
            // reqwest multipart parts are consumed on send; rebuild per attempt.
            let file_bytes = tokio::fs::read(audio_path)
                .await
                .with_context(|| format!("read audio {}", audio_path.display()))?;
            let file_name = audio_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("audio.bin")
                .to_string();
            let part = match guess_audio_mime(audio_path) {
                Some(mime) => reqwest::multipart::Part::bytes(file_bytes)
                    .file_name(file_name)
                    .mime_str(mime)
                    .unwrap(),
                None => reqwest::multipart::Part::bytes(file_bytes).file_name(file_name),
            };

            let res = try_once(primary_response_format, part).send().await;

            let res = match res {
                Ok(r) => r,
                Err(e) => {
                    if attempt < MAX_ATTEMPTS {
                        tracing::warn!(attempt, error = %e, "POST /v1/audio/transcriptions request error; backing off");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    return Err(anyhow::Error::new(e)).context("POST /v1/audio/transcriptions");
                }
            };

            let status = res.status();
            if status.is_success() {
                if primary_response_format == "verbose_json" {
                    return Ok(SttResponse::Verbose(
                        res.json::<TranscriptionVerboseJson>()
                            .await
                            .context("parse stt response")?,
                    ));
                }
                return Ok(SttResponse::Json(
                    res.json::<TranscriptionJson>()
                        .await
                        .context("parse stt response")?,
                ));
            }

            // Capture the Retry-After header (if any) BEFORE consuming the
            // body — calling `.text()` moves `res`. Groq returns this on 429
            // with the same value referenced in the error body's
            // "Please try again in Ns" message.
            let retry_after_secs = res
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            let body = res.text().await.unwrap_or_default();
            last_status = Some(status);
            last_body = body.clone();

            // Don't waste time retrying quota/billing failures.
            if status.as_u16() == 429 && body.contains("insufficient_quota") {
                anyhow::bail!("stt failed: {status} {body}");
            }

            let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
            if retryable && attempt < MAX_ATTEMPTS {
                // Prefer the server's Retry-After hint over blind exponential
                // backoff. Groq emits "Please try again in 3s" → we sleep
                // ~3.3s and re-fire instead of marching the backoff schedule
                // up to 8s. Header takes precedence; fall back to a regex
                // sniff of the message body for providers that omit the
                // header but include the duration in the JSON.
                let sleep = retry_after_secs
                    .map(|s| Duration::from_millis(s.saturating_mul(1000) + 300))
                    .or_else(|| sniff_retry_after_body(&body))
                    .unwrap_or(backoff);
                tracing::warn!(
                    attempt,
                    status = %status,
                    sleep_ms = sleep.as_millis() as u64,
                    "stt failed with retryable status; honoring retry hint"
                );
                tokio::time::sleep(sleep).await;
                backoff = (backoff * 2).min(Duration::from_secs(8));
                continue;
            }

            break;
        }

        let status = last_status.unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
        let body = last_body;
        // Fallback: verbose_json rejected -> retry with json
        if primary_response_format == "verbose_json"
            && status.as_u16() == 400
            && body.contains("response_format")
            && body.contains("not compatible")
        {
            let file_bytes = tokio::fs::read(audio_path)
                .await
                .with_context(|| format!("read audio {}", audio_path.display()))?;
            let file_name = audio_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("audio.bin")
                .to_string();
            let part = match guess_audio_mime(audio_path) {
                Some(mime) => reqwest::multipart::Part::bytes(file_bytes)
                    .file_name(file_name)
                    .mime_str(mime)
                    .unwrap(),
                None => reqwest::multipart::Part::bytes(file_bytes).file_name(file_name),
            };

            let res2 = try_once("json", part)
                .send()
                .await
                .context("POST /v1/audio/transcriptions (fallback json)")?;
            let status2 = res2.status();
            if !status2.is_success() {
                let body2 = res2.text().await.unwrap_or_default();
                anyhow::bail!("stt failed: {status2} {body2}");
            }
            return Ok(SttResponse::Json(
                res2.json::<TranscriptionJson>()
                    .await
                    .context("parse stt response")?,
            ));
        }

        anyhow::bail!("stt failed: {status} {body}");
    }

    pub async fn chat_json(
        &self,
        model: &str,
        system: &str,
        user: &str,
    ) -> anyhow::Result<serde_json::Value> {
        // gpt-5.* (and some other newer models) are served via the Responses API, not chat.
        if model.starts_with("gpt-5") {
            return self
                .responses_json(model, system, user)
                .await
                .with_context(|| format!("/v1/responses model={model}"));
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatRequest {
            model: model.to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user.to_string(),
                },
            ],
            response_format: Some(ResponseFormat {
                r#type: "json_object".to_string(),
            }),
            temperature: Some(0.4),
        };

        let res = self
            .post_json_with_retry(&url, &body, "POST /v1/chat/completions")
            .await?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            if status.as_u16() == 404
                && (body.contains("not a chat model")
                    || body.contains("v1/chat/completions")
                    || body.contains("Did you mean to use v1/responses")
                    || body.contains("Did you mean to use v1/completions"))
            {
                return self
                    .responses_json(model, system, user)
                    .await
                    .with_context(|| format!("fallback /v1/responses model={model}"));
            }

            anyhow::bail!("chat failed: {status} {body}");
        }

        let parsed: ChatResponse = res.json().await.context("parse chat response")?;
        if let Some(u) = &parsed.usage {
            self.record_usage(
                CostCategory::Chat,
                model,
                u.prompt_tokens,
                u.completion_tokens,
            );
        }
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .context("missing chat content")?;

        serde_json::from_str(&content).context("chat content was not JSON")
    }

    pub async fn chat_text(&self, model: &str, system: &str, user: &str) -> anyhow::Result<String> {
        // gpt-5.* (and some other newer models) are served via the Responses API, not chat.
        if model.starts_with("gpt-5") {
            return self
                .responses_text(model, system, user)
                .await
                .with_context(|| format!("/v1/responses model={model}"));
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatRequest {
            model: model.to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user.to_string(),
                },
            ],
            response_format: None,
            temperature: Some(0.7),
        };

        let res = self
            .post_json_with_retry(&url, &body, "POST /v1/chat/completions")
            .await?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            if status.as_u16() == 404
                && (body.contains("not a chat model")
                    || body.contains("v1/chat/completions")
                    || body.contains("Did you mean to use v1/responses")
                    || body.contains("Did you mean to use v1/completions"))
            {
                return self
                    .responses_text(model, system, user)
                    .await
                    .with_context(|| format!("fallback /v1/responses model={model}"));
            }

            anyhow::bail!("chat failed: {status} {body}");
        }

        let parsed: ChatResponse = res.json().await.context("parse chat response")?;
        if let Some(u) = &parsed.usage {
            self.record_usage(
                CostCategory::Chat,
                model,
                u.prompt_tokens,
                u.completion_tokens,
            );
        }
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .context("missing chat content")?;

        Ok(content)
    }

    async fn responses_json(
        &self,
        model: &str,
        system: &str,
        user: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/v1/responses", self.base_url);
        let allow_temperature = !model.starts_with("gpt-5");
        let body = ResponsesRequest {
            model: model.to_string(),
            input: vec![
                ResponsesInputMessage {
                    role: "system".to_string(),
                    content: vec![ResponsesInputContent::InputText {
                        text: system.to_string(),
                    }],
                },
                ResponsesInputMessage {
                    role: "user".to_string(),
                    content: vec![ResponsesInputContent::InputText {
                        text: user.to_string(),
                    }],
                },
            ],
            text: Some(ResponsesTextConfig {
                format: ResponsesTextFormat {
                    r#type: "json_object".to_string(),
                },
            }),
            temperature: if allow_temperature { Some(0.4) } else { None },
        };

        let res = self
            .post_json_with_retry(&url, &body, "POST /v1/responses")
            .await?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("responses failed: {status} {body}");
        }

        let parsed: ResponsesResponse = res.json().await.context("parse responses response")?;
        if let Some(u) = &parsed.usage {
            self.record_usage(CostCategory::Chat, model, u.input_tokens, u.output_tokens);
        }
        let text = parsed
            .output_text()
            .context("missing responses output_text")?;
        serde_json::from_str(&text).context("responses content was not JSON")
    }

    async fn responses_text(
        &self,
        model: &str,
        system: &str,
        user: &str,
    ) -> anyhow::Result<String> {
        let url = format!("{}/v1/responses", self.base_url);
        let allow_temperature = !model.starts_with("gpt-5");
        let body = ResponsesRequest {
            model: model.to_string(),
            input: vec![
                ResponsesInputMessage {
                    role: "system".to_string(),
                    content: vec![ResponsesInputContent::InputText {
                        text: system.to_string(),
                    }],
                },
                ResponsesInputMessage {
                    role: "user".to_string(),
                    content: vec![ResponsesInputContent::InputText {
                        text: user.to_string(),
                    }],
                },
            ],
            // No enforced JSON format.
            text: None,
            temperature: if allow_temperature { Some(0.7) } else { None },
        };

        let res = self
            .post_json_with_retry(&url, &body, "POST /v1/responses")
            .await?;

        let status = res.status();
        if !status.is_success() {
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("responses failed: {status} {body}");
        }

        let parsed: ResponsesResponse = res.json().await.context("parse responses response")?;
        if let Some(u) = &parsed.usage {
            self.record_usage(CostCategory::Chat, model, u.input_tokens, u.output_tokens);
        }
        parsed
            .output_text()
            .context("missing responses output_text")
    }

    async fn post_json_with_retry<T: Serialize + ?Sized>(
        &self,
        url: &str,
        body: &T,
        label: &str,
    ) -> anyhow::Result<reqwest::Response> {
        // Retry transient failures from upstream (Cloudflare 502, etc) and rate limiting.
        // Keep this small and bounded to avoid hanging the worker.
        const MAX_ATTEMPTS: usize = 5;
        let mut backoff = Duration::from_millis(400);

        for attempt in 1..=MAX_ATTEMPTS {
            let res = self.auth(self.http.post(url)).json(body).send().await;

            match res {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        return Ok(r);
                    }

                    let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);

                    // Capture Retry-After before consuming the body. Groq's
                    // chat 429s carry both the header AND a "try again in Ns"
                    // string in the JSON error.message (TPM-window-relative,
                    // can be sub-second like 757ms or up to ~60s on a full
                    // window). Honoring it is way better than the previous
                    // fixed exponential schedule, which capped at 8s and
                    // would burn all 5 attempts before the window resets.
                    let retry_after_secs = r
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok());
                    let body_text = r.text().await.unwrap_or_default();

                    if retryable && attempt < MAX_ATTEMPTS {
                        let sleep = retry_after_secs
                            .map(|s| Duration::from_millis(s.saturating_mul(1000) + 300))
                            .or_else(|| sniff_retry_after_body(&body_text))
                            .unwrap_or(backoff);
                        let body_snip = truncate_for_log(&body_text, 600);
                        tracing::warn!(
                            attempt,
                            status = %status,
                            sleep_ms = sleep.as_millis() as u64,
                            body = %body_snip,
                            "{label} failed with retryable status; honoring retry hint"
                        );
                        tokio::time::sleep(sleep).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }

                    anyhow::bail!("request failed: {status} {body_text}");
                }
                Err(e) => {
                    if attempt < MAX_ATTEMPTS {
                        tracing::warn!(attempt, error = %e, "{label} request error; backing off");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }
                    Err(anyhow::Error::new(e)).with_context(|| label.to_string())?;
                    unreachable!()
                }
            }
        }

        anyhow::bail!("{label} failed after {MAX_ATTEMPTS} attempts")
    }
}

fn truncate_for_log(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect::<String>() + "…"
}

enum SttResponse {
    Verbose(TranscriptionVerboseJson),
    Json(TranscriptionJson),
}

#[derive(Debug, Deserialize)]
struct TranscriptionJson {
    #[serde(default, deserialize_with = "null_as_default")]
    text: String,
}

#[derive(Debug)]
pub struct TranscriptionText {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub words: Vec<TranscriptionWord>,
}

/// True when the character is in a Unicode block typical of non-Latin
/// scripts that Whisper occasionally hallucinates into English transcripts
/// (Hebrew, Arabic, Indic scripts, Thai, Lao, Tibetan, Myanmar, Georgian,
/// Hangul, CJK, kana). Latin / Latin-1 supplement / Latin extended / Greek
/// / Cyrillic / General Punctuation are NOT flagged — they appear
/// legitimately in English transcripts (accents, smart quotes, named
/// entities). If you transcribe genuinely non-English audio, this filter
/// would false-positive — gate behind a config knob if that becomes a need.
fn is_foreign_script_char(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        0x0590..=0x05FF  // Hebrew
        | 0x0600..=0x07FF  // Arabic, Syriac, Thaana, NKo
        | 0x0900..=0x097F  // Devanagari
        | 0x0980..=0x09FF  // Bengali
        | 0x0A00..=0x0AFF  // Gurmukhi, Gujarati
        | 0x0B00..=0x0BFF  // Oriya, Tamil
        | 0x0C00..=0x0CFF  // Telugu, Kannada
        | 0x0D00..=0x0DFF  // Malayalam, Sinhala
        | 0x0E00..=0x0E7F  // Thai
        | 0x0E80..=0x0EFF  // Lao
        | 0x0F00..=0x0FFF  // Tibetan
        | 0x1000..=0x109F  // Myanmar
        | 0x10A0..=0x10FF  // Georgian
        | 0x1100..=0x11FF  // Hangul Jamo
        | 0x3040..=0x309F  // Hiragana
        | 0x30A0..=0x30FF  // Katakana
        | 0x3400..=0x4DBF  // CJK Ext A
        | 0x4E00..=0x9FFF  // CJK Unified
        | 0xAC00..=0xD7AF  // Hangul Syllables
        | 0xF900..=0xFAFF  // CJK Compat
    )
}

/// Heuristic detector for Whisper STT hallucinations on non-speech chunks.
///
/// When you feed Whisper an audio chunk that's silence, music, or background
/// noise, it doesn't return empty — it confidently writes plausible-looking
/// English (or fragments of song lyrics, generic conversational filler, or
/// occasionally bursts of CJK/Thai characters). Downstream LLM passes treat
/// the transcript as ground truth and confabulate full social-copy hooks +
/// ranker reasoning around words that don't exist.
///
/// This function checks three independent signals; any one tripping marks
/// the chunk as hallucinated. Caller should drop the returned words from the
/// transcript so the ranker doesn't see them.
///
/// Signals:
///   1. **Foreign-script characters** — real English transcription contains
///      Latin + common punctuation + occasional Latin-extended (é, ñ, smart
///      quotes). It never contains Thai (ั), Hangul (장난), CJK (中文), etc.
///      Even one such character mid-sentence is a strong hallucination flag.
///   2. **Repeated 3-gram > 3×** — Whisper's other failure on silence is
///      looping the same phrase (e.g. "Thank you. Thank you. Thank you...").
///   3. **Word density < 0.5 wps** on a chunk longer than 5s — real speech
///      runs 2-3 words/sec; hallucinations are sparse fragments at 0.1-0.3.
///
/// Returns `Some(reason)` when one trips, `None` when the chunk looks clean.
pub fn detect_stt_hallucination(
    text: &str,
    words: &[TranscriptionWord],
    chunk_duration_secs: f64,
) -> Option<String> {
    detect_stt_hallucination_with(text, words, chunk_duration_secs, HallucinationGuard::Default)
}

/// Strictness profile for the Whisper hallucination detector. Sourced from
/// `STT_HALLUCINATION_GUARD` env (`lax` / `default` / `strict`) so users can
/// trade off recall (catching hallucinated music chunks) vs. precision
/// (false-positives on legitimate quiet stretches).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HallucinationGuard {
    /// Foreign-script + extreme repetition only. Lowest false-positive
    /// rate. Use when transcripts come from clean studio audio.
    Lax,
    /// Recommended default — all three signals at their original
    /// thresholds (foreign script, 3-gram > 3×, word density < 0.5 wps
    /// over chunks > 5s).
    Default,
    /// Aggressive: same signals as default, but tighter thresholds
    /// (repeat > 2×, density < 0.75 wps over chunks > 3s). Use when
    /// the source is noisy or Whisper is invasively hallucinating on
    /// music/silence stretches.
    Strict,
}

impl HallucinationGuard {
    /// Parse the `STT_HALLUCINATION_GUARD` env knob. Unknown values fall
    /// back to `Default` (with a tracing warning).
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "lax" | "loose" => HallucinationGuard::Lax,
            "default" | "balanced" | "" => HallucinationGuard::Default,
            "strict" | "aggressive" => HallucinationGuard::Strict,
            other => {
                tracing::warn!(value = other, "unknown STT_HALLUCINATION_GUARD; using 'default'");
                HallucinationGuard::Default
            }
        }
    }
}

pub fn detect_stt_hallucination_with(
    text: &str,
    words: &[TranscriptionWord],
    chunk_duration_secs: f64,
    guard: HallucinationGuard,
) -> Option<String> {
    // 1. Foreign script presence — fires under every profile (all three
    //    strictness levels treat CJK/Thai in an English transcription as a
    //    hard signal).
    if let Some(c) = text.chars().find(|c| is_foreign_script_char(*c)) {
        return Some(format!(
            "foreign-script character {:?} (U+{:04X}) in transcript",
            c, c as u32
        ));
    }
    // 2. Repeated 3-grams. Threshold tightens under Strict.
    let repeat_threshold = match guard {
        HallucinationGuard::Lax => 4,
        HallucinationGuard::Default => 3,
        HallucinationGuard::Strict => 2,
    };
    let toks: Vec<&str> = text.split_whitespace().collect();
    if toks.len() >= 6 {
        let mut counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for w in toks.windows(3) {
            let key = w.join(" ").to_lowercase();
            *counts.entry(key).or_insert(0) += 1;
        }
        if let Some((trigram, count)) = counts.iter().max_by_key(|(_, c)| **c) {
            if *count > repeat_threshold {
                return Some(format!(
                    "repeated 3-gram {:?} appears {} times",
                    trigram, count
                ));
            }
        }
    }
    // 3. Word density. Lax skips this entirely; Default keeps 0.5 wps over
    //    5s; Strict tightens to 0.75 wps over 3s.
    let (min_duration, min_wps) = match guard {
        HallucinationGuard::Lax => (f64::INFINITY, 0.0),
        HallucinationGuard::Default => (5.0, 0.5),
        HallucinationGuard::Strict => (3.0, 0.75),
    };
    if chunk_duration_secs > min_duration && !words.is_empty() {
        let wps = words.len() as f64 / chunk_duration_secs;
        if wps < min_wps {
            return Some(format!(
                "density {:.2} wps < {:.2} ({} words over {:.1}s)",
                wps,
                min_wps,
                words.len(),
                chunk_duration_secs
            ));
        }
    }
    None
}

#[cfg(test)]
mod hallucination_tests {
    use super::*;

    fn word(w: &str, start: f64, end: f64) -> TranscriptionWord {
        TranscriptionWord {
            word: w.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn clean_english_passes() {
        let text = "And then he told me the whole story about how it really went down.";
        let words: Vec<TranscriptionWord> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| word(w, i as f64 * 0.3, (i + 1) as f64 * 0.3))
            .collect();
        assert!(detect_stt_hallucination(text, &words, 5.0).is_none());
    }

    #[test]
    fn cjk_splice_flagged() {
        let text = "For now at this time, Incast the Fall of theั 장난 Ippo Tape";
        let words = vec![word("dummy", 0.0, 1.0); 12];
        let res = detect_stt_hallucination(text, &words, 60.0);
        assert!(res.is_some(), "expected hallucination flag");
        assert!(res.unwrap().contains("foreign-script"));
    }

    #[test]
    fn latin_accents_pass() {
        // Real English transcripts can include accented Latin (café, naïve)
        // and smart quotes ("don't"). Don't false-positive on these.
        let text = "She said it was naïve to think the café was open.";
        let words: Vec<TranscriptionWord> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| word(w, i as f64 * 0.3, (i + 1) as f64 * 0.3))
            .collect();
        assert!(detect_stt_hallucination(text, &words, 5.0).is_none());
    }

    #[test]
    fn repeated_phrase_flagged() {
        let text = "Thank you very much. Thank you very much. \
                    Thank you very much. Thank you very much. \
                    Thank you very much.";
        let words = vec![word("dummy", 0.0, 1.0); 25];
        let res = detect_stt_hallucination(text, &words, 30.0);
        assert!(res.is_some());
        assert!(res.unwrap().contains("3-gram"));
    }

    #[test]
    fn sparse_words_flagged() {
        // 2 words over 60s = 0.033 wps
        let text = "Hello world";
        let words = vec![word("Hello", 0.0, 0.5), word("world", 30.0, 30.5)];
        let res = detect_stt_hallucination(text, &words, 60.0);
        assert!(res.is_some());
        assert!(res.unwrap().contains("wps"));
    }

    #[test]
    fn empty_text_is_clean() {
        // Silent chunks shouldn't be treated as hallucination; downstream
        // accumulator already skips empty text.
        assert!(detect_stt_hallucination("", &[], 60.0).is_none());
    }

    #[test]
    fn short_chunk_skips_density_check() {
        // A 3-second chunk with one word shouldn't trip density (the floor
        // is 5s — too aggressive on quick replies otherwise).
        let words = vec![word("yes", 0.0, 0.4)];
        assert!(detect_stt_hallucination("yes", &words, 3.0).is_none());
    }
}

/// Best-effort: extract "Please try again in <N><unit>" from a 429 body
/// when the server didn't set a `Retry-After` header. Handles both seconds
/// (`3s`, `3.5s`) and milliseconds (`757.5ms`) — Groq emits sub-second
/// hints in `ms` for TPM bursts and full-second hints in `s` for window
/// rollovers. Returns `None` on no match so the caller can fall back to
/// exponential backoff.
fn sniff_retry_after_body(body: &str) -> Option<Duration> {
    let needle = "try again in ";
    let idx = body.to_lowercase().find(needle)?;
    let rest = &body[idx + needle.len()..];
    // Take the leading numeric run (digits + optional decimal point).
    let num_str: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let amount: f64 = num_str.parse().ok()?;
    if !amount.is_finite() || amount <= 0.0 {
        return None;
    }
    // Look at the next char(s) to disambiguate ms vs s.
    let suffix = rest[num_str.len()..]
        .chars()
        .take(2)
        .collect::<String>()
        .to_lowercase();
    let total_ms = if suffix.starts_with("ms") {
        amount
    } else if suffix.starts_with('s') {
        amount * 1000.0
    } else {
        // Unknown unit — bail out rather than guess.
        return None;
    };
    // Clamp to a sane window. 120s is enough for any Groq TPM rollover;
    // anything larger is suspect (likely a parse mistake).
    if total_ms > 120_000.0 {
        return None;
    }
    // Add 300ms jitter so parallel retriers don't all wake together.
    Some(Duration::from_millis(total_ms as u64 + 300))
}

#[cfg(test)]
mod sniff_tests {
    use super::sniff_retry_after_body;
    use std::time::Duration;

    #[test]
    fn parses_seconds() {
        let d = sniff_retry_after_body(
            r#"{"error":{"message":"Please try again in 3s. Need more?"}}"#,
        )
        .unwrap();
        // 3s + 300ms jitter
        assert!(d >= Duration::from_millis(3300) && d <= Duration::from_millis(3400));
    }

    #[test]
    fn parses_decimal_seconds() {
        let d = sniff_retry_after_body(
            r#"{"error":{"message":"Please try again in 3.5s."}}"#,
        )
        .unwrap();
        assert!(d >= Duration::from_millis(3800) && d <= Duration::from_millis(3900));
    }

    #[test]
    fn parses_milliseconds() {
        let d = sniff_retry_after_body(
            r#"{"error":{"message":"Rate limit ... Please try again in 757.5ms. Need more?"}}"#,
        )
        .unwrap();
        // 757ms + 300ms jitter
        assert!(d >= Duration::from_millis(1050) && d <= Duration::from_millis(1100));
    }

    #[test]
    fn rejects_unknown_unit() {
        assert!(sniff_retry_after_body("Please try again in 5 minutes").is_none());
    }

    #[test]
    fn rejects_no_match() {
        assert!(sniff_retry_after_body("totally unrelated body").is_none());
    }
}

fn guess_audio_mime(path: &std::path::Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "m4a" | "mp4" => "audio/mp4",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "webm" => "audio/webm",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        _ => return None,
    })
}

/// Coerce `null` → `T::default()` during deserialization. `#[serde(default)]`
/// alone only fills MISSING fields; OpenAI-compatible providers vary in
/// whether they omit a field or emit explicit `null` (Groq returns
/// `segments: null` when you ask only for `timestamp_granularities[]=word`,
/// and `whisper-1` returns `words: null` even when you do ask for words).
/// Pairing `default` with this deserializer makes both spellings accept.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Option::deserialize(deserializer).map(|x: Option<T>| x.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionVerboseJson {
    #[serde(default, deserialize_with = "null_as_default")]
    pub text: String,
    #[serde(default, deserialize_with = "null_as_default")]
    pub segments: Vec<TranscriptionSegment>,
    #[serde(default, deserialize_with = "null_as_default")]
    pub words: Vec<TranscriptionWord>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionWord {
    #[serde(default, deserialize_with = "null_as_default")]
    pub word: String,
    #[serde(default, deserialize_with = "null_as_default")]
    pub start: f64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub end: f64,
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionSegment {
    #[allow(dead_code)]
    #[serde(default)]
    pub id: Option<u64>,
    #[serde(default, deserialize_with = "null_as_default")]
    pub start: f64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub end: f64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub text: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    r#type: String,
}

#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInputMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<ResponsesTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ResponsesInputMessage {
    role: String,
    content: Vec<ResponsesInputContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesInputContent {
    #[serde(rename = "input_text")]
    InputText { text: String },
}

#[derive(Debug, Serialize)]
struct ResponsesTextConfig {
    format: ResponsesTextFormat,
}

#[derive(Debug, Serialize)]
struct ResponsesTextFormat {
    r#type: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<ResponsesOutputItem>,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

impl ResponsesResponse {
    fn output_text(&self) -> Option<String> {
        let mut out = String::new();
        for item in &self.output {
            if let ResponsesOutputItem::Message { role, content } = item {
                if role != "assistant" {
                    continue;
                }
                for c in content {
                    if let ResponsesMessageContent::OutputText { text } = c {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(text);
                    }
                }
            }
        }
        if out.trim().is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponsesOutputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        #[serde(default)]
        content: Vec<ResponsesMessageContent>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponsesMessageContent {
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
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
