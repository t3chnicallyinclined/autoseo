//! Clipper orchestrator. Takes a local media file, extracts audio, runs the full
//! M1 feature pipeline, ranks candidates via LLM, renders top clips with burned
//! captions, sends a digest email with clip attachments.
//!
//! Polling-from-Gmail and posting-to-social-platforms are M2+ — for M1 the
//! acceptance path is `MODE=clipper LOCAL_VIDEO_PATH=foo.mp4`.

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ai_pipeline::AiPipeline;
use crate::align::AlignedWord;
use crate::candidates::CandidateGenerator;
use crate::captions::{self, CaptionStyle};
use crate::config::Config;
use crate::gmail::GmailClient;
use crate::google_auth::GoogleAuth;
use crate::media;
use crate::mime::{Attachment, build_mime_email};
use crate::prosody;
use crate::ranker::{RankedClip, Ranker};
use crate::render::{self, RenderProfile};
use crate::scene;
use crate::vad;

const MAX_DIGEST_ATTACHMENT_BYTES_TOTAL: u64 = 20 * 1024 * 1024;
const MAX_DIGEST_ATTACHMENT_BYTES_EACH: u64 = 18 * 1024 * 1024;

pub async fn run_clipper_local_once(
    cfg: &Config,
    google: &GoogleAuth,
    gmail: &GmailClient,
    ai: &AiPipeline,
    local_path: &str,
) -> Result<()> {
    let result_to = cfg
        .result_to
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("RESULT_TO is required in clipper mode"))?;

    let input_path = PathBuf::from(local_path);
    if tokio::fs::metadata(&input_path).await.is_err() {
        anyhow::bail!("local input does not exist: {}", input_path.display());
    }

    let file_name = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("local_media")
        .to_string();
    let lower = file_name.to_lowercase();

    let is_audio_only = !is_video_extension(&lower);
    if is_audio_only {
        // Clipper renders video clips; an audio-only input can't produce vertical reformat.
        anyhow::bail!(
            "clipper mode requires a video input, got: {file_name}"
        );
    }

    ensure_tool(&cfg.ffmpeg, "ffmpeg").await?;
    ensure_tool(&cfg.ffprobe, "ffprobe").await?;

    let job_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let job_dir = PathBuf::from(&cfg.work_dir)
        .join("clipper")
        .join(sanitize_filename(&file_name))
        .join(job_ts.to_string());
    tokio::fs::create_dir_all(&job_dir).await.ok();

    let audio_path = job_dir.join("audio.m4a");
    if !audio_path.exists() {
        media::extract_audio_m4a(&cfg.ffmpeg, &input_path, &audio_path).await?;
    }

    let total_duration_secs = media::duration_secs(&cfg.ffprobe, &audio_path)
        .await
        .context("ffprobe duration")?;
    let chunk_secs = choose_chunk_secs(cfg, total_duration_secs);
    tracing::info!(
        total_secs = total_duration_secs,
        chunk_secs,
        "clipper: chunk sizing"
    );

    let chunks_dir = job_dir.join("audio_chunks");
    let chunk_paths =
        media::segment_audio(&cfg.ffmpeg, &audio_path, &chunks_dir, chunk_secs).await?;
    if chunk_paths.is_empty() {
        anyhow::bail!("no audio chunks produced");
    }

    let chunks: Vec<(PathBuf, f64, f64)> = chunk_paths
        .into_iter()
        .enumerate()
        .map(|(idx, p)| {
            let offset = (idx as f64) * (chunk_secs as f64);
            let mut dur = chunk_secs as f64;
            if offset + dur > total_duration_secs {
                dur = (total_duration_secs - offset).max(0.0);
            }
            if !dur.is_finite() || dur <= 0.0 {
                dur = chunk_secs as f64;
            }
            (p, offset, dur)
        })
        .collect();

    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    let bar_style = ProgressStyle::with_template(
        "{spinner:.green} {msg:18} {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar());

    let pb_transcribe = mp.add(ProgressBar::new(chunks.len() as u64));
    pb_transcribe.set_style(bar_style.clone());
    pb_transcribe.set_message("transcribing");
    pb_transcribe.enable_steady_tick(std::time::Duration::from_millis(120));

    let pb_features = mp.add(ProgressBar::new_spinner());
    if let Ok(style) = ProgressStyle::with_template("{spinner:.green} {msg}") {
        pb_features.set_style(style);
    }
    pb_features.set_message("scene + vad + prosody");
    pb_features.enable_steady_tick(std::time::Duration::from_millis(120));

    let (transcript, shots, silences, rms) = tokio::try_join!(
        ai.transcribe_word_chunks(&chunks, Some(pb_transcribe.clone())),
        scene::detect_shots(&cfg.ffmpeg, &input_path, 0.4),
        vad::detect_silences(&cfg.ffmpeg, &audio_path, -30.0, 0.3),
        prosody::rms_curve(&cfg.ffmpeg, &audio_path, 1.0),
    )?;
    pb_features.finish_with_message("scene + vad + prosody complete");

    tracing::info!(
        words = transcript.words.len(),
        shots = shots.len(),
        silences = silences.len(),
        rms_windows = rms.len(),
        "clipper: feature extraction complete"
    );

    if transcript.words.is_empty() {
        anyhow::bail!(
            "transcription returned no word-level timestamps — provider likely doesn't \
             support timestamp_granularities=word (configure OPENAI_BASE_URL + \
             OPENAI_STT_MODEL to point at Groq or another supporting provider)"
        );
    }

    let show_context = ai
        .infer_show_context(&file_name, &transcript.full_text)
        .await
        .ok();

    let cand_gen = CandidateGenerator::new();
    let candidates = cand_gen.generate(total_duration_secs, &transcript.words, &silences, &shots, &rms);
    tracing::info!(candidates = candidates.len(), "clipper: candidates generated");

    if candidates.is_empty() {
        anyhow::bail!("no candidate windows produced — episode may be too short or too quiet");
    }

    let ranker_system = tokio::fs::read_to_string(&cfg.clip_ranker_system_prompt_path)
        .await
        .with_context(|| {
            format!(
                "read ranker system prompt at {}",
                &cfg.clip_ranker_system_prompt_path
            )
        })?
        .trim()
        .to_string();
    let ranker_user_template = tokio::fs::read_to_string(&cfg.clip_ranker_user_prompt_path)
        .await
        .with_context(|| {
            format!(
                "read ranker user prompt at {}",
                &cfg.clip_ranker_user_prompt_path
            )
        })?
        .trim()
        .to_string();

    let ranker = Ranker::new(
        ai.openai.clone(),
        ai.chat_model.clone(),
        ranker_system,
        ranker_user_template,
    );

    let top_k = cfg.clip_top_k.max(1);
    let ranked = ranker
        .rank(&candidates, top_k, show_context.as_ref())
        .await
        .context("LLM rank")?;
    tracing::info!(top_k = ranked.len(), "clipper: ranked clips");

    if ranked.is_empty() {
        anyhow::bail!("LLM ranker returned no clips");
    }

    // Render each ranked clip with burned captions.
    let clips_dir = job_dir.join("clips");
    tokio::fs::create_dir_all(&clips_dir).await.ok();
    let profile = RenderProfile::shorts_vertical();
    let style = CaptionStyle::default();

    let mut rendered: Vec<RenderedClip> = Vec::with_capacity(ranked.len());
    for (i, clip) in ranked.iter().enumerate() {
        let idx = i + 1;
        let basename = format!("clip_{idx:02}_{}-{}.mp4", to_mmss(clip.start_secs), to_mmss(clip.end_secs));
        let out_path = clips_dir.join(&basename);
        let ass_path = clips_dir.join(format!("clip_{idx:02}.ass"));

        let clip_words: Vec<AlignedWord> = transcript
            .words
            .iter()
            .filter(|w| w.start_secs >= clip.start_secs && w.end_secs <= clip.end_secs + 0.5)
            .map(|w| AlignedWord {
                text: w.text.clone(),
                start_secs: (w.start_secs - clip.start_secs).max(0.0),
                end_secs: (w.end_secs - clip.start_secs).max(0.0),
            })
            .collect();

        captions::write_ass(&ass_path, &clip_words, profile.width, profile.height, &style)
            .await
            .with_context(|| format!("write .ass for clip {idx}"))?;

        let render_result = render::render_clip(
            &cfg.ffmpeg,
            &input_path,
            clip.start_secs,
            clip.end_secs,
            &out_path,
            &profile,
            Some(&ass_path),
        )
        .await;

        match render_result {
            Ok(()) => {
                let bytes = tokio::fs::metadata(&out_path).await.map(|m| m.len()).unwrap_or(0);
                tracing::info!(
                    clip = idx,
                    path = %out_path.display(),
                    score = clip.score,
                    bytes,
                    "rendered clip"
                );
                rendered.push(RenderedClip {
                    rank: idx,
                    path: out_path,
                    bytes,
                    ranked: clip.clone(),
                });
            }
            Err(e) => {
                tracing::warn!(clip = idx, error = ?e, "render failed; skipping");
            }
        }
    }

    if rendered.is_empty() {
        anyhow::bail!("all clip renders failed");
    }

    let body = build_digest_body(&file_name, total_duration_secs, &rendered);
    let attachments = build_attachments(&rendered).await;

    let subject = format!(
        "{} CLIPPER: {} clips for {}",
        cfg.result_subject_prefix,
        rendered.len(),
        file_name
    );

    let access_token = google.access_token().await?;
    let raw_mime = build_mime_email("me", result_to, &subject, &body, &attachments);
    let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
    let sent_message_id = gmail.send_raw(&access_token, &raw_b64url).await?;
    tracing::info!(sent_message_id, "clipper: digest email sent");
    Ok(())
}

struct RenderedClip {
    rank: usize,
    path: PathBuf,
    bytes: u64,
    ranked: RankedClip,
}

fn build_digest_body(media_name: &str, total_duration_secs: f64, rendered: &[RenderedClip]) -> String {
    let mut out = String::new();
    out.push_str(&format!("Episode: {media_name}\n"));
    out.push_str(&format!(
        "Total duration: {}\n",
        fmt_hms(total_duration_secs)
    ));
    out.push_str(&format!("Clips produced: {}\n", rendered.len()));
    out.push_str("\n");
    out.push_str(&"=".repeat(72));
    out.push('\n');

    for r in rendered {
        out.push_str(&format!(
            "Clip {idx:02}  score {score}  {start}-{end}  ({dur}s)\n",
            idx = r.rank,
            score = r.ranked.score,
            start = to_mmss(r.ranked.start_secs),
            end = to_mmss(r.ranked.end_secs),
            dur = (r.ranked.end_secs - r.ranked.start_secs) as i64,
        ));
        if !r.ranked.hook.is_empty() {
            out.push_str(&format!("  hook: {}\n", r.ranked.hook));
        }
        if !r.ranked.reasoning.is_empty() {
            out.push_str(&format!("  why:  {}\n", r.ranked.reasoning));
        }
        out.push_str(&format!(
            "  file: {} ({:.1} MB)\n",
            r.path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?"),
            r.bytes as f64 / (1024.0 * 1024.0)
        ));
        out.push_str("\n");
        out.push_str(&"-".repeat(72));
        out.push('\n');
    }

    out.push('\n');
    out.push_str("Local job dir contains the clip MP4s; clips small enough to attach are\n");
    out.push_str("included on this email. Larger clips are referenced by filename above.\n");
    out
}

async fn build_attachments(rendered: &[RenderedClip]) -> Vec<Attachment> {
    let mut out = Vec::new();
    let mut total: u64 = 0;
    for r in rendered {
        if r.bytes > MAX_DIGEST_ATTACHMENT_BYTES_EACH {
            tracing::info!(
                clip = r.rank,
                bytes = r.bytes,
                "skipping attachment: clip exceeds per-file budget"
            );
            continue;
        }
        if total.saturating_add(r.bytes) > MAX_DIGEST_ATTACHMENT_BYTES_TOTAL {
            tracing::info!(
                clip = r.rank,
                bytes = r.bytes,
                total,
                "skipping attachment: would exceed total digest budget"
            );
            continue;
        }
        let bytes = match tokio::fs::read(&r.path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(clip = r.rank, error = ?e, "failed to read clip for attachment");
                continue;
            }
        };
        let filename = r
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("clip.mp4")
            .to_string();
        total += bytes.len() as u64;
        out.push(Attachment {
            filename,
            content_type: "video/mp4".to_string(),
            bytes,
        });
    }
    out
}

fn choose_chunk_secs(cfg: &Config, total_duration: f64) -> u64 {
    if !cfg.auto_chunking || !total_duration.is_finite() || total_duration <= 0.0 {
        return cfg.audio_chunk_secs;
    }
    let target_chunks = if cfg.auto_chunk_target_chunks > 0 {
        cfg.auto_chunk_target_chunks
    } else {
        let concurrency = cfg.stt_concurrency.max(1);
        let factor = cfg.auto_chunk_target_factor.max(1);
        concurrency.saturating_mul(factor).max(1)
    };
    let mut chunk = (total_duration / target_chunks as f64).ceil();
    if chunk < cfg.auto_chunk_min_secs as f64 {
        chunk = cfg.auto_chunk_min_secs as f64;
    }
    if chunk > cfg.auto_chunk_max_secs as f64 {
        chunk = cfg.auto_chunk_max_secs as f64;
    }
    chunk as u64
}

async fn ensure_tool(cmd: &str, label: &str) -> Result<()> {
    let status = tokio::process::Command::new(cmd)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(_) => anyhow::bail!("{label} exists but failed: {cmd}"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!("{label} not found on PATH (tried: {cmd})")
        }
        Err(e) => Err(e.into()),
    }
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

fn is_video_extension(name_lower: &str) -> bool {
    name_lower.ends_with(".mp4")
        || name_lower.ends_with(".mov")
        || name_lower.ends_with(".mkv")
        || name_lower.ends_with(".webm")
        || name_lower.ends_with(".m4v")
        || name_lower.ends_with(".avi")
}

fn to_mmss(secs: f64) -> String {
    let s = secs.max(0.0) as i64;
    let m = s / 60;
    let r = s % 60;
    format!("{m:02}:{r:02}")
}

fn fmt_hms(secs: f64) -> String {
    let s = secs.max(0.0) as i64;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let r = s % 60;
    if h > 0 {
        format!("{h}h {m:02}m {r:02}s")
    } else {
        format!("{m}m {r:02}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranker::RankedClip;

    #[test]
    fn to_mmss_basic() {
        assert_eq!(to_mmss(0.0), "00:00");
        assert_eq!(to_mmss(59.9), "00:59");
        assert_eq!(to_mmss(61.0), "01:01");
        assert_eq!(to_mmss(3601.5), "60:01");
    }

    #[test]
    fn fmt_hms_basic() {
        assert_eq!(fmt_hms(45.0), "0m 45s");
        assert_eq!(fmt_hms(125.0), "2m 05s");
        assert_eq!(fmt_hms(3725.0), "1h 02m 05s");
    }

    #[test]
    fn sanitize_strips_path_chars() {
        assert_eq!(sanitize_filename("a/b\\c:d*e?f.mp4"), "a_b_c_d_e_f.mp4");
    }

    #[test]
    fn is_video_detects_common_extensions() {
        assert!(is_video_extension("foo.mp4"));
        assert!(is_video_extension("foo.mov"));
        assert!(is_video_extension("foo.mkv"));
        assert!(!is_video_extension("foo.mp3"));
        assert!(!is_video_extension("foo.wav"));
    }

    fn dummy_clip(rank: usize, score: i32, start: f64, end: f64, hook: &str) -> RenderedClip {
        RenderedClip {
            rank,
            path: PathBuf::from(format!("/tmp/clip_{rank:02}.mp4")),
            bytes: 2_500_000,
            ranked: RankedClip {
                candidate_index: rank,
                start_secs: start,
                end_secs: end,
                score,
                hook: hook.to_string(),
                reasoning: "test".to_string(),
            },
        }
    }

    #[test]
    fn digest_body_includes_each_clip() {
        let clips = vec![
            dummy_clip(1, 90, 60.0, 120.0, "first hook"),
            dummy_clip(2, 75, 240.0, 300.0, "second hook"),
        ];
        let body = build_digest_body("episode.mp4", 7320.0, &clips);
        assert!(body.contains("episode.mp4"));
        assert!(body.contains("2h"));
        assert!(body.contains("Clip 01"));
        assert!(body.contains("Clip 02"));
        assert!(body.contains("first hook"));
        assert!(body.contains("second hook"));
        assert!(body.contains("01:00-02:00"));
        assert!(body.contains("score 90"));
    }
}
