use anyhow::Context;
use futures_util::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::align::{self, AlignedWord};
use crate::openai::OpenAiClient;
use crate::rate_limit::RpmGate;

#[derive(Debug, Clone)]
pub struct AiPipeline {
    pub openai: OpenAiClient,
    pub stt_model: String,
    pub chat_model: String,
    stt_concurrency: usize,
    stt_rpm_gate: Option<Arc<RpmGate>>,
    seo_system_prompt: String,
    thumbnail_system_prompt: String,
    seo_user_prompt_template: String,
    thumbnail_user_prompt_template: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start: f64,
    #[allow(dead_code)]
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Transcript {
    pub full_text: String,
    pub segments: Vec<TranscriptSegment>,
}

/// Like [`Transcript`] but also carries word-level timestamps in absolute episode time.
/// Produced by [`AiPipeline::transcribe_word_chunks`] for the clipper pipeline.
#[derive(Debug, Clone)]
pub struct WordTranscript {
    pub full_text: String,
    pub segments: Vec<TranscriptSegment>,
    pub words: Vec<AlignedWord>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ShowContext {
    /// Canonical show name if explicitly supported by evidence.
    #[serde(default)]
    pub show_name: Option<String>,

    /// Host names if explicitly supported by evidence.
    #[serde(default)]
    pub hosts: Vec<String>,

    /// Primary guest name if explicitly supported by evidence.
    #[serde(default)]
    pub guest: Option<String>,

    /// Short evidence snippet(s) or notes (optional).
    #[allow(dead_code)]
    #[serde(default)]
    pub evidence: Vec<String>,
}

impl AiPipeline {
    pub fn new(
        openai: OpenAiClient,
        stt_model: String,
        chat_model: String,
        stt_concurrency: usize,
        stt_rpm_limit: u32,
        seo_system_prompt: String,
        thumbnail_system_prompt: String,
        seo_user_prompt_template: String,
        thumbnail_user_prompt_template: String,
    ) -> Self {
        let stt_rpm_gate = if stt_rpm_limit > 0 {
            Some(Arc::new(RpmGate::new(stt_rpm_limit)))
        } else {
            None
        };
        Self {
            openai,
            stt_model,
            chat_model,
            stt_concurrency: stt_concurrency.max(1),
            stt_rpm_gate,
            seo_system_prompt,
            thumbnail_system_prompt,
            seo_user_prompt_template,
            thumbnail_user_prompt_template,
        }
    }

    pub async fn transcribe_chunks(
        &self,
        chunks: &[(std::path::PathBuf, f64, f64)],
        progress: Option<ProgressBar>,
    ) -> anyhow::Result<Transcript> {
        let client = self.openai.clone();
        let stt_model = self.stt_model.clone();
        let concurrency = self.stt_concurrency;
        let stt_rpm_gate = self.stt_rpm_gate.clone();
        if chunks.is_empty() {
            anyhow::bail!("no audio chunks to transcribe");
        }

        let total_chunks = chunks.len();
        let progress = progress.unwrap_or_else(ProgressBar::hidden);
        progress.set_length(total_chunks as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} transcribing {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise}"
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        progress.set_message("starting");
        let completed = Arc::new(AtomicUsize::new(0));

        let results = stream::iter(chunks.iter().cloned().enumerate())
            .map(|(idx, (chunk_path, offset_secs, chunk_duration_secs))| {
                let client = client.clone();
                let stt_model = stt_model.clone();
                let progress = progress.clone();
                let completed = completed.clone();
                let total_chunks = total_chunks;
                let stt_rpm_gate = stt_rpm_gate.clone();
                async move {
                    let chunk_label = chunk_path.display().to_string();
                    tracing::debug!(chunk=%chunk_label, offset_secs, "transcribing audio chunk");

                    if let Some(gate) = stt_rpm_gate.as_ref() {
                        gate.wait().await;
                    }
                    let tr = client
                        .transcribe_text(&stt_model, &chunk_path)
                        .await
                        .with_context(|| format!("stt for {chunk_label}"))?;

                    let mapped = map_segments_or_synthesize(tr, offset_secs, chunk_duration_secs);
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.set_message(format!("chunk {done}/{total_chunks}"));
                    progress.inc(1);
                    Ok::<_, anyhow::Error>((idx, mapped))
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await;
        progress.finish_with_message("transcription complete");

        let mut per_chunk = Vec::with_capacity(results.len());
        for res in results {
            per_chunk.push(res?);
        }
        per_chunk.sort_by_key(|(idx, _)| *idx);

        let mut segments_out = Vec::new();
        let mut text_out = String::new();
        for (_, mapped) in per_chunk {
            for seg in mapped {
                if !seg.text.trim().is_empty() {
                    segments_out.push(seg.clone());
                    text_out.push_str(seg.text.trim());
                    text_out.push('\n');
                }
            }
        }

        Ok(Transcript {
            full_text: text_out,
            segments: segments_out,
        })
    }

    /// Parallel STT over chunked audio with word-level timestamps. Mirrors
    /// [`transcribe_chunks`] but uses [`OpenAiClient::transcribe_words`] (Groq's
    /// `whisper-large-v3-turbo` or any provider that honors
    /// `timestamp_granularities=word`). Word timestamps are shifted into the global
    /// episode timeline.
    pub async fn transcribe_word_chunks(
        &self,
        chunks: &[(std::path::PathBuf, f64, f64)],
        progress: Option<ProgressBar>,
    ) -> anyhow::Result<WordTranscript> {
        if chunks.is_empty() {
            anyhow::bail!("no audio chunks to transcribe");
        }
        let client = self.openai.clone();
        let stt_model = self.stt_model.clone();
        let concurrency = self.stt_concurrency;
        let stt_rpm_gate = self.stt_rpm_gate.clone();
        let total_chunks = chunks.len();

        let progress = progress.unwrap_or_else(ProgressBar::hidden);
        progress.set_length(total_chunks as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} transcribing-words {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        let completed = Arc::new(AtomicUsize::new(0));

        let results = stream::iter(chunks.iter().cloned().enumerate())
            .map(|(idx, (chunk_path, offset_secs, chunk_duration_secs))| {
                let client = client.clone();
                let stt_model = stt_model.clone();
                let progress = progress.clone();
                let completed = completed.clone();
                let total_chunks = total_chunks;
                let stt_rpm_gate = stt_rpm_gate.clone();
                async move {
                    let chunk_label = chunk_path.display().to_string();
                    if let Some(gate) = stt_rpm_gate.as_ref() {
                        gate.wait().await;
                    }
                    let tr = client
                        .transcribe_words(&stt_model, &chunk_path)
                        .await
                        .with_context(|| format!("stt-words for {chunk_label}"))?;

                    let segments =
                        map_segments_or_synthesize_words(&tr, offset_secs, chunk_duration_secs);
                    let words = align::shift_words(&tr.words, offset_secs);

                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.set_message(format!("chunk {done}/{total_chunks}"));
                    progress.inc(1);
                    Ok::<_, anyhow::Error>((idx, segments, words))
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await;
        progress.finish_with_message("word transcription complete");

        let mut per_chunk = Vec::with_capacity(results.len());
        for res in results {
            per_chunk.push(res?);
        }
        per_chunk.sort_by_key(|(idx, _, _)| *idx);

        let mut segments_out = Vec::new();
        let mut words_out = Vec::new();
        let mut text_out = String::new();
        for (_, segs, words) in per_chunk {
            for seg in segs {
                if !seg.text.trim().is_empty() {
                    text_out.push_str(seg.text.trim());
                    text_out.push('\n');
                    segments_out.push(seg);
                }
            }
            words_out.extend(words);
        }

        Ok(WordTranscript {
            full_text: text_out,
            segments: segments_out,
            words: words_out,
        })
    }

    pub async fn infer_show_context(
        &self,
        media_name: &str,
        transcript_text: &str,
    ) -> anyhow::Result<ShowContext> {
        // Keep this cheap: use the filename and just the beginning of the transcript.
        let transcript_head = clamp_chars(transcript_text, 18_000);

        let system = r#"You extract show metadata from weak signals.

Hard rules:
- Only output a show name/host/guest if it is explicitly present in the provided filename or transcript snippet.
- If not explicit, use null/empty.
- Do NOT guess based on style, topic, or vibes.

Return JSON only."#;

        let user = format!(
            r#"Media filename:
{media_name}

Transcript snippet (start of episode):
{transcript_head}

Task:
Infer the podcast/show context ONLY if explicitly stated.

Return JSON with this shape:
{{
  "show_name": string|null,
  "hosts": string[],
  "guest": string|null,
  "evidence": string[]
}}

Rules:
- evidence: 0-3 short quotes or notes showing where you found it (filename or transcript).
- If you only see an acronym, you may return it as show_name ONLY if it appears verbatim.
"#
        );

        let json = self
            .openai
            .chat_json(&self.chat_model, system, &user)
            .await
            .context("infer_show_context LLM call")?;

        let parsed: ShowContext =
            serde_json::from_value(json).context("parse infer_show_context JSON")?;
        Ok(parsed)
    }

    pub async fn seo_variant_text_with_context(
        &self,
        transcript_text: &str,
        variant_instructions: &str,
        variant_index: usize,
        variant_total: usize,
        show_context: Option<&ShowContext>,
        media_name: Option<&str>,
    ) -> anyhow::Result<String> {
        // Keep prompts bounded.
        let transcript_text = clamp_chars(transcript_text, 120_000);

        let system = self.seo_system_prompt.as_str();
        let mut user = self
            .seo_user_prompt_template
            .replace("{{transcript}}", transcript_text.as_str());

        user = user
            .replace("{{variant_instructions}}", variant_instructions)
            .replace("{{variant_index}}", &(variant_index + 1).to_string())
            .replace("{{variant_total}}", &variant_total.to_string());

        if let Some(name) = media_name {
            user = user.replace("{{media_name}}", name);
        } else {
            user = user.replace("{{media_name}}", "");
        }

        if let Some(ctx) = show_context {
            let show_name = ctx.show_name.as_deref().unwrap_or("");
            let hosts = if ctx.hosts.is_empty() {
                "".to_string()
            } else {
                ctx.hosts.join(", ")
            };
            let guest = ctx.guest.as_deref().unwrap_or("");
            user = user
                .replace("{{show_name}}", show_name)
                .replace("{{hosts}}", hosts.as_str())
                .replace("{{guest}}", guest);
        } else {
            user = user
                .replace("{{show_name}}", "")
                .replace("{{hosts}}", "")
                .replace("{{guest}}", "");
        }

        let text = self
            .openai
            .chat_text(&self.chat_model, system, &user)
            .await?;
        Ok(text.trim().to_string())
    }

    pub async fn thumbnail_windows(
        &self,
        segments: &[TranscriptSegment],
        count: usize,
    ) -> anyhow::Result<Vec<ThumbnailMoment>> {
        // Provide a compact time-indexed summary to the LLM.
        let minutes = minute_index(segments, 180 /* max minutes */);

        let system = self.thumbnail_system_prompt.as_str();
        let user = self
            .thumbnail_user_prompt_template
            .replace("{{count}}", &count.to_string())
            .replace("{{minutes}}", &minutes);

        let json = self
            .openai
            .chat_json(&self.chat_model, system, &user)
            .await?;
        let resp: ThumbnailResponse =
            serde_json::from_value(json).context("parse thumbnail JSON")?;

        let mut out = resp.moments;

        // Normalize and clamp.
        for m in &mut out {
            if !m.center_seconds.is_finite() || m.center_seconds < 0.0 {
                m.center_seconds = 0.0;
            }
            if m.reason.trim().is_empty() {
                m.reason = "thumbnail moment".to_string();
            }
        }

        // If the model under-delivers, pad deterministically so the pipeline still generates
        // enough thumbnails.
        if count > 0 && out.len() < count {
            let approx_duration = segments
                .iter()
                .map(|s| s.end.max(s.start))
                .fold(0.0_f64, |a, b| a.max(b))
                .max(segments.last().map(|s| s.start).unwrap_or(0.0).max(0.0));

            // Avoid placing filler right at EOF.
            let last_ts = (approx_duration - 0.25).max(0.0);

            let missing = count - out.len();
            let base = if count > 1 {
                last_ts / (count.saturating_sub(1) as f64)
            } else {
                0.0
            };

            for i in 0..missing {
                let idx = out.len() + i;
                let mut ts = (idx as f64) * base;
                if !ts.is_finite() {
                    ts = 0.0;
                }
                ts = ts.max(0.0).min(last_ts);
                out.push(ThumbnailMoment {
                    center_seconds: ts,
                    reason: "big reaction — safe filler".to_string(),
                });
            }
        }

        // Deduplicate near-identical timestamps (keep earliest), then truncate.
        out.sort_by(|a, b| {
            a.center_seconds
                .partial_cmp(&b.center_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.dedup_by(|a, b| (a.center_seconds - b.center_seconds).abs() < 0.25);
        out.truncate(count);
        Ok(out)
    }
}

fn map_segments_or_synthesize(
    tr: crate::openai::TranscriptionText,
    offset_secs: f64,
    chunk_duration_secs: f64,
) -> Vec<TranscriptSegment> {
    if !tr.segments.is_empty() {
        return tr
            .segments
            .into_iter()
            .map(|s| TranscriptSegment {
                start: s.start + offset_secs,
                end: s.end + offset_secs,
                text: s.text,
            })
            .collect();
    }

    synthesize_minute_segments(&tr.text, offset_secs, chunk_duration_secs)
}

/// Variant of `map_segments_or_synthesize` that borrows the response (we still need
/// `tr.words` after this call for the word-shift pass).
fn map_segments_or_synthesize_words(
    tr: &crate::openai::TranscriptionText,
    offset_secs: f64,
    chunk_duration_secs: f64,
) -> Vec<TranscriptSegment> {
    if !tr.segments.is_empty() {
        return tr
            .segments
            .iter()
            .map(|s| TranscriptSegment {
                start: s.start + offset_secs,
                end: s.end + offset_secs,
                text: s.text.clone(),
            })
            .collect();
    }

    synthesize_minute_segments(&tr.text, offset_secs, chunk_duration_secs)
}

fn synthesize_minute_segments(
    text: &str,
    offset_secs: f64,
    chunk_duration_secs: f64,
) -> Vec<TranscriptSegment> {
    let dur = if chunk_duration_secs.is_finite() && chunk_duration_secs > 0.0 {
        chunk_duration_secs
    } else {
        60.0
    };
    let minutes = ((dur / 60.0).ceil() as usize).max(1);

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![TranscriptSegment {
            start: offset_secs,
            end: offset_secs + dur,
            text: text.to_string(),
        }];
    }

    let mut out = Vec::with_capacity(minutes);
    for i in 0..minutes {
        let start_word = i * words.len() / minutes;
        let end_word = ((i + 1) * words.len() / minutes).max(start_word + 1);
        let seg_text = words[start_word..end_word.min(words.len())].join(" ");

        let start = offset_secs + (i as f64) * 60.0;
        let end = (start + 60.0).min(offset_secs + dur);
        out.push(TranscriptSegment {
            start,
            end,
            text: seg_text,
        });
    }
    out
}

fn clamp_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect()
}

fn minute_index(segments: &[TranscriptSegment], max_minutes: usize) -> String {
    let mut buckets: std::collections::BTreeMap<u64, String> = std::collections::BTreeMap::new();
    for seg in segments {
        let minute = (seg.start.max(0.0) as u64) / 60;
        let entry = buckets.entry(minute).or_default();
        if entry.len() < 800 {
            entry.push_str(seg.text.trim());
            entry.push(' ');
        }
    }

    let mut out = String::new();
    for (minute, text) in buckets.into_iter().take(max_minutes) {
        out.push_str(&format!(
            "[{start:>5}s] {text}\n",
            start = minute * 60,
            text = clamp_chars(&text, 600)
        ));
    }
    out
}

#[derive(Debug, Deserialize)]
struct ThumbnailResponse {
    moments: Vec<ThumbnailMoment>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ThumbnailMoment {
    pub center_seconds: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub reason: String,
}
