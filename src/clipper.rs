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
use crate::candidates::{self, CandidateGenerator};
use crate::captions::{self, CaptionStyle, OverlayStyle};
use crate::config::Config;
use crate::embed::Embedder;
use crate::gmail::GmailClient;
use crate::google_auth::GoogleAuth;
use crate::media;
use crate::mime::build_mime_email;
use crate::prosody;
use crate::ranker::{RankedClip, Ranker};
use crate::render::{self, RenderProfile};
use crate::platforms::{self, PostResult, PostStatus};
use crate::posting;
use crate::scene;
use crate::show_config::DigestMode;
use crate::social_copy::{SocialCopy, SocialCopyGenerator};
use crate::storage::{JobStatus, Storage};
use crate::vad;
use crate::vlm_ranker::VlmReranker;

/// Update job status in storage if both storage and job_id are provided.
/// Errors from the status update itself are logged but not propagated,
/// so a DB hiccup never masks the real pipeline error.
async fn set_job_status(storage: Option<&Storage>, job_id: Option<&str>, status: JobStatus, error: Option<&str>) {
    if let (Some(st), Some(jid)) = (storage, job_id) {
        if let Err(e) = st.update_job_status(jid, status, error).await {
            tracing::warn!(job_id = jid, status = status.as_str(), error = ?e, "failed to update job status");
        }
    }
}

pub async fn run_clipper_local_once(
    cfg: &Config,
    google: Option<&GoogleAuth>,
    gmail: &GmailClient,
    ai: &AiPipeline,
    local_path: &str,
    digest_mode: DigestMode,
    storage: Option<&Storage>,
    job_id: Option<&str>,
) -> Result<()> {
    let result = run_clipper_inner(cfg, google, gmail, ai, local_path, digest_mode, storage, job_id).await;
    if let Err(ref e) = result {
        let msg = format!("{e:#}");
        set_job_status(storage, job_id, JobStatus::Failed, Some(&msg)).await;
    }
    result
}

async fn run_clipper_inner(
    cfg: &Config,
    google: Option<&GoogleAuth>,
    gmail: &GmailClient,
    ai: &AiPipeline,
    local_path: &str,
    digest_mode: DigestMode,
    storage: Option<&Storage>,
    job_id: Option<&str>,
) -> Result<()> {
    if digest_mode.sends_email() {
        if google.is_none() {
            anyhow::bail!(
                "DIGEST_MODE includes email but Google credentials are missing"
            );
        }
        if cfg.result_to.as_deref().unwrap_or("").is_empty() {
            anyhow::bail!("DIGEST_MODE includes email but RESULT_TO is unset");
        }
    }

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

    // F0 pitch extraction via aubio — non-fatal, returns empty vec if unavailable.
    let f0 = prosody::f0_curve("aubio", &audio_path).await;
    if f0.is_empty() {
        tracing::info!("clipper: F0 features unavailable (aubio not on PATH or failed)");
    }

    pb_features.finish_with_message("scene + vad + prosody complete");

    tracing::info!(
        words = transcript.words.len(),
        shots = shots.len(),
        silences = silences.len(),
        rms_windows = rms.len(),
        f0_samples = f0.len(),
        "clipper: feature extraction complete"
    );

    if transcript.words.is_empty() {
        anyhow::bail!(
            "transcription returned no word-level timestamps — provider likely doesn't \
             support timestamp_granularities=word (configure OPENAI_BASE_URL + \
             OPENAI_STT_MODEL to point at Groq or another supporting provider)"
        );
    }

    // ── Status: transcribed ──
    set_job_status(storage, job_id, JobStatus::Transcribed, None).await;

    let show_context = ai
        .infer_show_context(&file_name, &transcript.full_text)
        .await
        .ok();

    let cand_gen = CandidateGenerator::new();
    let mut candidates =
        cand_gen.generate(total_duration_secs, &transcript.words, &silences, &shots, &rms, &f0);
    tracing::info!(candidates = candidates.len(), "clipper: candidates generated");

    if candidates.is_empty() {
        anyhow::bail!("no candidate windows produced — episode may be too short or too quiet");
    }

    // Attach within-episode novelty scores. Non-fatal — if the embedder fails
    // (network blip, missing model cache), we proceed without the signal.
    match Embedder::from_config(cfg) {
        Ok(embedder) => {
            tracing::info!(backend = embedder.backend_name(), "clipper: embedding novelty");
            if let Err(e) = candidates::attach_novelty(&mut candidates, &embedder).await {
                tracing::warn!(error = ?e, "novelty scoring failed; proceeding without it");
            } else {
                let with_scores = candidates.iter().filter(|c| c.novelty_score.is_some()).count();
                tracing::info!(
                    with_scores,
                    total = candidates.len(),
                    "clipper: novelty attached"
                );
            }
        }
        Err(e) => {
            tracing::warn!(error = ?e, "embedder init failed; proceeding without novelty");
        }
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

    let final_top_k = cfg.clip_top_k.max(1);
    // If VLM re-rank is on, pass a wider top-K through the LLM ranker so the
    // VLM has more to choose from before final truncation.
    let llm_top_k = if cfg.vlm_rerank_enabled {
        cfg.vlm_rerank_top_k.max(final_top_k)
    } else {
        final_top_k
    };
    let ranked = ranker
        .rank(&candidates, llm_top_k, show_context.as_ref())
        .await
        .context("LLM rank")?;
    tracing::info!(top_k = ranked.len(), "clipper: LLM ranked clips");

    if ranked.is_empty() {
        anyhow::bail!("LLM ranker returned no clips");
    }

    // ── Status: ranked ──
    set_job_status(storage, job_id, JobStatus::Ranked, None).await;

    // Write clip rows to DB now that we have ranked clips.
    if let (Some(st), Some(jid)) = (storage, job_id) {
        for (i, clip) in ranked.iter().enumerate() {
            let clip_id = format!("{jid}_clip_{}", i + 1);
            let start_ms = (clip.start_secs * 1000.0) as i64;
            let end_ms = (clip.end_secs * 1000.0) as i64;
            if let Err(e) = st
                .insert_clip(
                    &clip_id,
                    jid,
                    start_ms,
                    end_ms,
                    Some((i + 1) as i64),
                    Some(clip.score as f64),
                    Some(&clip.hook),
                    Some(&clip.reasoning),
                )
                .await
            {
                tracing::warn!(clip = i + 1, error = ?e, "failed to write clip to DB");
            }
        }
    }

    // Optional: VLM re-rank top-N using frames + transcript.
    let ranked = match VlmReranker::from_config(cfg) {
        Some(reranker) => {
            tracing::info!(
                model = %cfg.vlm_model,
                frames_per_clip = cfg.vlm_frames_per_clip,
                blend_weight = cfg.vlm_blend_weight,
                "clipper: VLM re-rank starting"
            );
            match reranker
                .rerank(&cfg.ffmpeg, &input_path, ranked, cfg.vlm_rerank_top_k)
                .await
            {
                Ok(re) => {
                    tracing::info!(reranked = re.len(), "clipper: VLM re-rank complete");
                    re
                }
                Err(e) => {
                    tracing::warn!(error = ?e, "VLM re-rank failed; falling back to LLM order");
                    // re-fetch from ranker since we moved it; just re-do the call cheaply
                    ranker
                        .rank(&candidates, llm_top_k, show_context.as_ref())
                        .await
                        .context("LLM rank (refetch after VLM failure)")?
                }
            }
        }
        None => ranked,
    };

    let ranked: Vec<_> = ranked.into_iter().take(final_top_k).collect();

    // Generate per-platform social-media copy for each top clip. One LLM call
    // per clip; non-fatal — a single failure logs and continues.
    let social_copies: Vec<Option<SocialCopy>> = if cfg.clip_social_copy_disabled {
        tracing::info!("clipper: social-copy generation disabled by config");
        vec![None; ranked.len()]
    } else {
        let sys = tokio::fs::read_to_string(&cfg.clip_social_system_prompt_path)
            .await
            .with_context(|| {
                format!(
                    "read social system prompt at {}",
                    &cfg.clip_social_system_prompt_path
                )
            })?
            .trim()
            .to_string();
        let user_tmpl = tokio::fs::read_to_string(&cfg.clip_social_user_prompt_path)
            .await
            .with_context(|| {
                format!(
                    "read social user prompt at {}",
                    &cfg.clip_social_user_prompt_path
                )
            })?
            .trim()
            .to_string();
        let generator = SocialCopyGenerator::new(
            ai.openai.clone(),
            ai.chat_model.clone(),
            sys,
            user_tmpl,
        );
        tracing::info!(top_k = ranked.len(), "clipper: generating per-platform social copy");
        let mut copies: Vec<Option<SocialCopy>> = Vec::with_capacity(ranked.len());
        for (i, clip) in ranked.iter().enumerate() {
            let candidate = candidates.get(clip.candidate_index);
            let copy = match candidate {
                Some(cand) => match generator.generate(clip, cand, show_context.as_ref()).await {
                    Ok(c) => {
                        tracing::info!(
                            clip = i + 1,
                            overlay = %c.overlay_hook,
                            "social copy generated"
                        );
                        Some(c)
                    }
                    Err(e) => {
                        tracing::warn!(clip = i + 1, error = ?e, "social copy failed");
                        None
                    }
                },
                None => {
                    tracing::warn!(
                        clip = i + 1,
                        candidate_idx = clip.candidate_index,
                        "social copy: candidate lookup failed"
                    );
                    None
                }
            };
            copies.push(copy);
        }
        copies
    };

    // Render each ranked clip in every requested aspect ratio. Captions use a
    // per-aspect style (font size + margins tuned for the frame).
    let clips_dir = job_dir.join("clips");
    tokio::fs::create_dir_all(&clips_dir).await.ok();

    let formats = parse_render_formats(&cfg.clip_render_formats);
    if formats.is_empty() {
        anyhow::bail!(
            "CLIP_RENDER_FORMATS produced no recognized formats (got '{}'); expected any of 9x16,1x1,16x9",
            cfg.clip_render_formats
        );
    }
    tracing::info!(
        formats = ?formats.iter().map(|f| f.label).collect::<Vec<_>>(),
        "clipper: render formats"
    );

    let mut rendered: Vec<RenderedClip> = Vec::with_capacity(ranked.len());
    for (i, (clip, social)) in ranked.iter().zip(social_copies.iter()).enumerate() {
        let idx = i + 1;

        // Shift word timestamps into clip-local time once; reused for every aspect.
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

        let mut variants: Vec<RenderedVariant> = Vec::with_capacity(formats.len());
        for spec in &formats {
            let basename = format!(
                "clip_{idx:02}_{}-{}_{}.mp4",
                to_mmss(clip.start_secs),
                to_mmss(clip.end_secs),
                spec.label
            );
            let out_path = clips_dir.join(&basename);
            let ass_path =
                clips_dir.join(format!("clip_{idx:02}_{}.ass", spec.label));

            let profile = (spec.profile)();
            let style = (spec.style)();

            if let Err(e) =
                captions::write_ass(&ass_path, &clip_words, profile.width, profile.height, &style)
                    .await
            {
                tracing::warn!(clip = idx, format = spec.label, error = ?e, "ass write failed");
                continue;
            }

            // Optional overlay hook .ass for the first ~1.5s. Only write if the
            // social-copy LLM produced a hook. The render filter chain takes both
            // (captions first, overlay second so it draws on top).
            let overlay_ass_path = clips_dir.join(format!("clip_{idx:02}_{}_overlay.ass", spec.label));
            let mut overlay_written = false;
            if let Some(sc) = social {
                let hook = sc.overlay_hook.trim();
                if !hook.is_empty() {
                    let overlay_style = (spec.overlay_style)();
                    match captions::write_overlay_ass(
                        &overlay_ass_path,
                        hook,
                        profile.width,
                        profile.height,
                        &overlay_style,
                    )
                    .await
                    {
                        Ok(()) => overlay_written = true,
                        Err(e) => {
                            tracing::warn!(clip = idx, format = spec.label, error = ?e, "overlay ass write failed");
                        }
                    }
                }
            }

            // Build subtitle layer list: captions first, overlay last so it draws on top.
            let mut layers: Vec<&std::path::Path> = vec![ass_path.as_ref()];
            if overlay_written {
                layers.push(overlay_ass_path.as_ref());
            }

            let res = render::render_clip(
                &cfg.ffmpeg,
                &input_path,
                clip.start_secs,
                clip.end_secs,
                &out_path,
                &profile,
                &layers,
            )
            .await;

            match res {
                Ok(()) => {
                    let bytes = tokio::fs::metadata(&out_path)
                        .await
                        .map(|m| m.len())
                        .unwrap_or(0);
                    tracing::info!(
                        clip = idx,
                        format = spec.label,
                        path = %out_path.display(),
                        bytes,
                        "rendered variant"
                    );
                    variants.push(RenderedVariant {
                        label: spec.label.to_string(),
                        path: out_path,
                        bytes,
                        width: profile.width,
                        height: profile.height,
                    });
                }
                Err(e) => {
                    tracing::warn!(clip = idx, format = spec.label, error = ?e, "render failed");
                }
            }
        }

        if variants.is_empty() {
            tracing::warn!(clip = idx, "no variants rendered for this clip");
            continue;
        }

        rendered.push(RenderedClip {
            rank: idx,
            ranked: clip.clone(),
            social: social.clone(),
            variants,
            posts: Vec::new(),
        });
    }

    if rendered.is_empty() {
        anyhow::bail!("all clip renders failed");
    }

    // ── Status: rendered ──
    set_job_status(storage, job_id, JobStatus::Rendered, None).await;

    // Write clip_renders rows to DB.
    if let (Some(st), Some(jid)) = (storage, job_id) {
        for r in &rendered {
            let clip_id = format!("{jid}_clip_{}", r.rank);
            for v in &r.variants {
                let path_str = v.path.display().to_string();
                let dur_ms = r.ranked.end_secs - r.ranked.start_secs;
                if let Err(e) = st
                    .insert_clip_render(
                        &clip_id,
                        &v.label,
                        &path_str,
                        Some(v.bytes as i64),
                        Some((dur_ms * 1000.0) as i64),
                    )
                    .await
                {
                    tracing::warn!(clip = r.rank, variant = %v.label, error = ?e, "failed to write clip_render to DB");
                }
            }
        }
    }

    // Post each clip to enabled platforms (default: no platforms enabled,
    // POST_DRY_RUN=true). The 9x16 variant is used for both YouTube Shorts and
    // Bluesky video posts.
    let platforms = platforms::Platform::from_config(cfg, google);
    if !platforms.is_empty() {
        let platform_names: Vec<&str> = platforms.iter().map(|p| p.name()).collect();
        tracing::info!(
            platforms = ?platform_names,
            dry_run = cfg.post_dry_run,
            "clipper: posting starting"
        );
        for r in rendered.iter_mut() {
            let video_9x16 = r
                .variants
                .iter()
                .find(|v| v.label == "9x16")
                .map(|v| v.path.as_path());
            let results = posting::post_one_clip(
                &platforms,
                cfg.post_dry_run,
                r.rank,
                video_9x16,
                r.social.as_ref(),
            )
            .await;
            r.posts = results;
        }
        let total_posted = rendered
            .iter()
            .flat_map(|r| &r.posts)
            .filter(|p| p.status == PostStatus::Posted)
            .count();
        tracing::info!(total_posted, "clipper: posting complete");

        // ── Status: posted ──
        set_job_status(storage, job_id, JobStatus::Posted, None).await;

        // Write post rows to DB.
        if let (Some(st), Some(jid)) = (storage, job_id) {
            for r in &rendered {
                let clip_id = format!("{jid}_clip_{}", r.rank);
                for p in &r.posts {
                    let status_str = match p.status {
                        PostStatus::Posted => "posted",
                        PostStatus::DryRun => "dry_run",
                        PostStatus::Skipped => "skipped",
                        PostStatus::Failed => "failed",
                    };
                    if let Err(e) = st
                        .insert_post(
                            &clip_id,
                            &p.platform,
                            status_str,
                            p.external_id.as_deref(),
                            p.external_url.as_deref(),
                            p.posted_at_unix,
                            p.error.as_deref(),
                        )
                        .await
                    {
                        tracing::warn!(clip = r.rank, platform = %p.platform, error = ?e, "failed to write post to DB");
                    }
                }
            }
        }
    }

    let body = build_digest_body(&file_name, total_duration_secs, &clips_dir, &rendered);

    if digest_mode.writes_file() {
        let digest_path = clips_dir.join("digest.md");
        tokio::fs::write(&digest_path, &body)
            .await
            .with_context(|| format!("write digest at {}", digest_path.display()))?;
        let abs = digest_path.canonicalize().unwrap_or(digest_path);
        tracing::info!(path = %abs.display(), "clipper: digest written to disk");

        // Structured manifest for the HTML viewer (rich UI loads this instead of
        // regex-parsing digest.md).
        let manifest_path = clips_dir.join("manifest.json");
        let manifest =
            build_manifest_json(&file_name, total_duration_secs, &clips_dir, &rendered);
        let manifest_text = serde_json::to_string_pretty(&manifest)
            .unwrap_or_else(|_| "{}".to_string());
        if let Err(e) = tokio::fs::write(&manifest_path, manifest_text).await {
            tracing::warn!(error = ?e, "manifest.json write failed (non-fatal)");
        } else {
            tracing::info!(path = %manifest_path.display(), "clipper: manifest.json written");
        }
    }

    if digest_mode.sends_email() {
        let google = google.expect("validated at function entry");
        let result_to = cfg
            .result_to
            .as_deref()
            .expect("validated at function entry");

        let subject = format!(
            "{} CLIPPER: {} clips for {}",
            cfg.result_subject_prefix,
            rendered.len(),
            file_name
        );

        let access_token = google.access_token().await?;
        let raw_mime = build_mime_email("me", result_to, &subject, &body, &[]);
        let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
        let sent_message_id = gmail.send_raw(&access_token, &raw_b64url).await?;
        tracing::info!(sent_message_id, "clipper: digest email sent");
    }

    // ── Status: done ──
    set_job_status(storage, job_id, JobStatus::Done, None).await;

    Ok(())
}

struct RenderedClip {
    rank: usize,
    ranked: RankedClip,
    social: Option<SocialCopy>,
    variants: Vec<RenderedVariant>,
    posts: Vec<PostResult>,
}

struct RenderedVariant {
    label: String, // "9x16" | "1x1" | "16x9"
    path: PathBuf,
    bytes: u64,
    width: u32,
    height: u32,
}

/// One requested render format: aspect-ratio label + factories for the matching
/// `RenderProfile`, `CaptionStyle`, and `OverlayStyle`.
struct FormatSpec {
    label: &'static str,
    profile: fn() -> RenderProfile,
    style: fn() -> CaptionStyle,
    overlay_style: fn() -> OverlayStyle,
}

fn parse_render_formats(spec: &str) -> Vec<FormatSpec> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw in spec.split(',') {
        let label = raw.trim().to_ascii_lowercase();
        if label.is_empty() || !seen.insert(label.clone()) {
            continue;
        }
        let spec = match label.as_str() {
            "9x16" | "vertical" | "shorts" => Some(FormatSpec {
                label: "9x16",
                profile: RenderProfile::shorts_vertical,
                style: CaptionStyle::for_vertical,
                overlay_style: OverlayStyle::for_vertical,
            }),
            "1x1" | "square" => Some(FormatSpec {
                label: "1x1",
                profile: RenderProfile::linkedin_square,
                style: CaptionStyle::for_square,
                overlay_style: OverlayStyle::for_square,
            }),
            "16x9" | "landscape" | "horizontal" => Some(FormatSpec {
                label: "16x9",
                profile: RenderProfile::bluesky_landscape,
                style: CaptionStyle::for_landscape,
                overlay_style: OverlayStyle::for_landscape,
            }),
            other => {
                tracing::warn!(format = other, "unknown render format; ignoring");
                None
            }
        };
        if let Some(s) = spec {
            out.push(s);
        }
    }
    out
}

fn build_digest_body(
    media_name: &str,
    total_duration_secs: f64,
    clips_dir: &std::path::Path,
    rendered: &[RenderedClip],
) -> String {
    // Resolve to an absolute path if possible so the email reader can copy/paste
    // it straight into a `vlc` / `open` command.
    let abs_clips_dir = clips_dir
        .canonicalize()
        .unwrap_or_else(|_| clips_dir.to_path_buf());

    let mut out = String::new();
    out.push_str(&format!("Episode: {media_name}\n"));
    out.push_str(&format!(
        "Total duration: {}\n",
        fmt_hms(total_duration_secs)
    ));
    out.push_str(&format!("Clips produced: {}\n", rendered.len()));
    out.push_str(&format!("Clips directory: {}\n", abs_clips_dir.display()));
    out.push('\n');
    out.push_str(&"=".repeat(72));
    out.push('\n');

    for r in rendered {
        out.push_str(&format!(
            "## Clip {idx:02}  score {score}  {start}-{end}  ({dur}s)\n",
            idx = r.rank,
            score = r.ranked.score,
            start = to_mmss(r.ranked.start_secs),
            end = to_mmss(r.ranked.end_secs),
            dur = (r.ranked.end_secs - r.ranked.start_secs) as i64,
        ));
        if !r.ranked.hook.is_empty() {
            out.push_str(&format!("  hook:    {}\n", r.ranked.hook));
        }
        if !r.ranked.reasoning.is_empty() {
            out.push_str(&format!("  why:     {}\n", r.ranked.reasoning));
        }
        out.push_str("  files:\n");
        for v in &r.variants {
            let abs = v.path.canonicalize().unwrap_or_else(|_| v.path.clone());
            out.push_str(&format!(
                "    [{}]  {wx}x{hx}  {sz:.1}MB  {p}\n",
                v.label,
                wx = v.width,
                hx = v.height,
                sz = v.bytes as f64 / (1024.0 * 1024.0),
                p = abs.display(),
            ));
        }
        if let Some(social) = &r.social {
            if !social.overlay_hook.is_empty() {
                out.push_str(&format!("  overlay: {}\n", social.overlay_hook));
            }
            append_social_copy(&mut out, social);
        }
        if !r.posts.is_empty() {
            out.push_str("\n  ── Posts ──────────────────────\n");
            for p in &r.posts {
                let tag = match p.status {
                    PostStatus::Posted => "POSTED ",
                    PostStatus::DryRun => "DRYRUN ",
                    PostStatus::Skipped => "SKIPPED",
                    PostStatus::Failed => "FAILED ",
                };
                let detail = match p.status {
                    PostStatus::Posted => p
                        .external_url
                        .clone()
                        .or_else(|| p.external_id.clone())
                        .unwrap_or_default(),
                    PostStatus::Skipped | PostStatus::Failed => {
                        p.error.clone().unwrap_or_default()
                    }
                    PostStatus::DryRun => String::new(),
                };
                out.push_str(&format!("  [{tag}] {:<10} {detail}\n", p.platform));
            }
        }
        out.push('\n');
        out.push_str(&"-".repeat(72));
        out.push('\n');
    }

    out.push('\n');
    out.push_str("Clips are on disk at the paths above. Pick the winners and post them.\n");
    out
}

fn append_social_copy(out: &mut String, s: &SocialCopy) {
    out.push('\n');
    out.push_str("  ── YouTube Shorts ──────────────\n");
    if !s.youtube_shorts.title.is_empty() {
        out.push_str(&format!("  Title:        {}\n", s.youtube_shorts.title));
    }
    if !s.youtube_shorts.description.is_empty() {
        out.push_str(&indent_block("Description", &s.youtube_shorts.description));
    }
    if !s.youtube_shorts.hashtags.is_empty() {
        out.push_str(&format!(
            "  Hashtags:     {}\n",
            s.youtube_shorts.hashtags.join(" ")
        ));
    }
    if !s.youtube_shorts.pinned_comment.is_empty() {
        out.push_str(&format!(
            "  Pinned cmt:   {}\n",
            s.youtube_shorts.pinned_comment
        ));
    }

    out.push_str("\n  ── TikTok ──────────────────────\n");
    if !s.tiktok.caption.is_empty() {
        out.push_str(&indent_block("Caption", &s.tiktok.caption));
    }
    if !s.tiktok.hashtags.is_empty() {
        out.push_str(&format!("  Hashtags:     {}\n", s.tiktok.hashtags.join(" ")));
    }

    out.push_str("\n  ── Instagram Reels ─────────────\n");
    if !s.instagram_reels.caption.is_empty() {
        out.push_str(&indent_block("Caption", &s.instagram_reels.caption));
    }
    if !s.instagram_reels.hashtags.is_empty() {
        out.push_str(&format!(
            "  Hashtags:     {}\n",
            s.instagram_reels.hashtags.join(" ")
        ));
    }

    out.push_str("\n  ── Threads ─────────────────────\n");
    if !s.threads.text.is_empty() {
        out.push_str(&indent_block("Text", &s.threads.text));
    }
    if !s.threads.hashtags.is_empty() {
        out.push_str(&format!(
            "  Hashtags:     {}\n",
            s.threads.hashtags.join(" ")
        ));
    }

    out.push_str("\n  ── LinkedIn ────────────────────\n");
    if !s.linkedin.post_text.is_empty() {
        out.push_str(&indent_block("Post", &s.linkedin.post_text));
    }
    if !s.linkedin.hashtags.is_empty() {
        out.push_str(&format!(
            "  Hashtags:     {}\n",
            s.linkedin.hashtags.join(" ")
        ));
    }

    out.push_str("\n  ── X / Twitter ─────────────────\n");
    if !s.x.text.is_empty() {
        out.push_str(&indent_block("Text", &s.x.text));
    }
    if !s.x.hashtags.is_empty() {
        out.push_str(&format!("  Hashtags:     {}\n", s.x.hashtags.join(" ")));
    }

    out.push_str("\n  ── Bluesky ─────────────────────\n");
    if !s.bluesky.text.is_empty() {
        out.push_str(&indent_block("Text", &s.bluesky.text));
    }
    if !s.bluesky.hashtags.is_empty() {
        out.push_str(&format!(
            "  Hashtags:     {}\n",
            s.bluesky.hashtags.join(" ")
        ));
    }
}

/// Structured manifest of a clipper run for downstream tools (HTML viewer,
/// future posting bot, analytics). One per run; lives at `clips_dir/manifest.json`.
fn build_manifest_json(
    media_name: &str,
    total_duration_secs: f64,
    clips_dir: &std::path::Path,
    rendered: &[RenderedClip],
) -> serde_json::Value {
    let abs_clips_dir = clips_dir
        .canonicalize()
        .unwrap_or_else(|_| clips_dir.to_path_buf());

    let clips: Vec<serde_json::Value> = rendered
        .iter()
        .map(|r| {
            let variants: Vec<serde_json::Value> = r
                .variants
                .iter()
                .map(|v| {
                    let abs = v.path.canonicalize().unwrap_or_else(|_| v.path.clone());
                    serde_json::json!({
                        "label": v.label,
                        "filename": v.path.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(""),
                        "abs_path": abs.display().to_string(),
                        "width": v.width,
                        "height": v.height,
                        "bytes": v.bytes,
                    })
                })
                .collect();

            let social = r
                .social
                .as_ref()
                .and_then(|s| serde_json::to_value(s).ok())
                .unwrap_or(serde_json::Value::Null);

            let posts: Vec<serde_json::Value> = r
                .posts
                .iter()
                .filter_map(|p| serde_json::to_value(p).ok())
                .collect();

            serde_json::json!({
                "rank": r.rank,
                "score": r.ranked.score,
                "start_secs": r.ranked.start_secs,
                "end_secs": r.ranked.end_secs,
                "duration_secs": (r.ranked.end_secs - r.ranked.start_secs).max(0.0),
                "time_range_mmss": format!(
                    "{}-{}",
                    to_mmss(r.ranked.start_secs),
                    to_mmss(r.ranked.end_secs),
                ),
                "hook": r.ranked.hook,
                "reasoning": r.ranked.reasoning,
                "variants": variants,
                "social": social,
                "posts": posts,
            })
        })
        .collect();

    serde_json::json!({
        "schema_version": 1,
        "episode": media_name,
        "total_duration_secs": total_duration_secs,
        "clips_dir": abs_clips_dir.display().to_string(),
        "generated_at_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "clips": clips,
    })
}

/// Pretty-print a multi-line block with consistent left padding.
fn indent_block(label: &str, body: &str) -> String {
    let mut out = String::new();
    let mut first = true;
    for line in body.lines() {
        if first {
            out.push_str(&format!("  {label}: {pad}{line}\n", pad = " ".repeat(13_usize.saturating_sub(label.len() + 2))));
            first = false;
        } else {
            out.push_str(&format!("  {pad}{line}\n", pad = " ".repeat(15)));
        }
    }
    if first {
        // Empty body — emit just the label.
        out.push_str(&format!("  {label}:\n"));
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
            ranked: RankedClip {
                candidate_index: rank,
                start_secs: start,
                end_secs: end,
                score,
                hook: hook.to_string(),
                reasoning: "test".to_string(),
            },
            social: None,
            variants: vec![RenderedVariant {
                label: "9x16".into(),
                path: PathBuf::from(format!("/tmp/clip_{rank:02}_9x16.mp4")),
                bytes: 2_500_000,
                width: 1080,
                height: 1920,
            }],
            posts: Vec::new(),
        }
    }

    #[test]
    fn digest_body_includes_each_clip() {
        let clips = vec![
            dummy_clip(1, 90, 60.0, 120.0, "first hook"),
            dummy_clip(2, 75, 240.0, 300.0, "second hook"),
        ];
        let dir = std::path::PathBuf::from("/tmp/clips");
        let body = build_digest_body("episode.mp4", 7320.0, &dir, &clips);
        assert!(body.contains("episode.mp4"));
        assert!(body.contains("2h"));
        assert!(body.contains("Clip 01"));
        assert!(body.contains("Clip 02"));
        assert!(body.contains("first hook"));
        assert!(body.contains("second hook"));
        assert!(body.contains("01:00-02:00"));
        assert!(body.contains("score 90"));
        // File entries should appear under each clip's variants block.
        assert!(body.contains("clip_01_9x16.mp4"), "expected variant filename, got:\n{body}");
        assert!(body.contains("[9x16]"));
        assert!(body.contains("Clips directory:"));
    }
}
