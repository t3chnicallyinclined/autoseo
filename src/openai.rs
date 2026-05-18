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

impl OpenAiClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
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

            let body = res.text().await.unwrap_or_default();
            last_status = Some(status);
            last_body = body.clone();

            // Don't waste time retrying quota/billing failures.
            if status.as_u16() == 429 && body.contains("insufficient_quota") {
                anyhow::bail!("stt failed: {status} {body}");
            }

            let retryable = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
            if retryable && attempt < MAX_ATTEMPTS {
                tracing::warn!(attempt, status = %status, "stt failed with retryable status; backing off");
                tokio::time::sleep(backoff).await;
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
            self.record_usage(CostCategory::Chat, model, u.prompt_tokens, u.completion_tokens);
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
            self.record_usage(CostCategory::Chat, model, u.prompt_tokens, u.completion_tokens);
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

                    if retryable && attempt < MAX_ATTEMPTS {
                        let body_text = r.text().await.unwrap_or_default();
                        let body_snip = truncate_for_log(&body_text, 600);
                        tracing::warn!(
                            attempt,
                            status = %status,
                            body = %body_snip,
                            "{label} failed with retryable status; backing off"
                        );
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(8));
                        continue;
                    }

                    let body_text = r.text().await.unwrap_or_default();
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
    #[serde(default)]
    text: String,
}

#[derive(Debug)]
pub struct TranscriptionText {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub words: Vec<TranscriptionWord>,
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

#[derive(Debug, Deserialize)]
pub struct TranscriptionVerboseJson {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
    #[serde(default)]
    pub words: Vec<TranscriptionWord>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionWord {
    #[serde(default)]
    pub word: String,
    #[serde(default)]
    pub start: f64,
    #[serde(default)]
    pub end: f64,
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionSegment {
    #[allow(dead_code)]
    #[serde(default)]
    pub id: Option<u64>,
    #[serde(default)]
    pub start: f64,
    #[serde(default)]
    pub end: f64,
    #[serde(default)]
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
