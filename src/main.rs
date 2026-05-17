mod ai_pipeline;
mod align;
mod candidates;
mod captions;
mod clipper;
mod config;
mod drive;
mod embed;
mod gmail;
mod google_auth;
mod linguistic_markers;
mod media;
mod mime;
mod openai;
mod parse;
mod platforms;
mod posting;
mod prosody;
mod ranker;
mod rate_limit;
mod render;
mod scene;
mod show_config;
mod social_copy;
mod storage;
mod thumbs;
mod vad;
mod vlm_ranker;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use clap::Parser;
use futures_util::future::try_join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use ai_pipeline::{AiPipeline, ThumbnailMoment};
use config::Config;
use drive::DriveClient;
use gmail::GmailClient;
use google_auth::GoogleAuth;
use mime::build_mime_email;
use openai::OpenAiClient;
use std::time::{SystemTime, UNIX_EPOCH};
use storage::Storage;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Ensure we always get useful logs in Docker (even if RUST_LOG is unset/invalid).
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cfg = Config::parse();

    let mode = show_config::Mode::parse(&cfg.mode)?;
    let digest_mode = show_config::DigestMode::parse(&cfg.digest_mode)?;

    // Print a minimal startup banner to stdout so `docker logs` is never empty.
    println!(
        "autoseo starting (mode={:?}, digest_mode={:?}, poll_interval_secs={}, work_dir={}, clipper_db={})",
        mode, digest_mode, cfg.poll_interval_secs, cfg.work_dir, cfg.clipper_db
    );

    // Build Google auth only if creds are all present; downstream code validates
    // its own requirement and bails with a clear message if missing.
    let google: Option<GoogleAuth> = match (
        cfg.google_client_id.as_ref(),
        cfg.google_client_secret.as_ref(),
        cfg.google_refresh_token.as_ref(),
    ) {
        (Some(id), Some(secret), Some(token))
            if !id.is_empty() && !secret.is_empty() && !token.is_empty() =>
        {
            Some(GoogleAuth::new(id.clone(), secret.clone(), token.clone()))
        }
        _ => None,
    };
    let gmail = GmailClient::new();
    let drive = DriveClient::new();

    // Validation matrix: which paths actually require Google?
    //   - seo-only mode (always sends per-variant Gmail emails): requires Google + RESULT_TO
    //   - clipper mode + DIGEST_MODE includes email: requires Google + RESULT_TO
    //   - clipper mode + DIGEST_MODE=file (default): no Google required
    //   - polling loop (no LOCAL_VIDEO_PATH) for seo-only: requires Google (Gmail/Drive ingest)
    let needs_google = mode.produces_seo_emails()
        || (mode.produces_clips() && digest_mode.sends_email());
    if needs_google && google.is_none() {
        anyhow::bail!(
            "GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET + GOOGLE_REFRESH_TOKEN are required \
             for the requested MODE/DIGEST_MODE combination. Set them in .env or switch to \
             MODE=clipper with DIGEST_MODE=file to run Google-free."
        );
    }
    let needs_result_to = mode.produces_seo_emails()
        || (mode.produces_clips() && digest_mode.sends_email());
    if needs_result_to && cfg.result_to.as_deref().unwrap_or("").is_empty() && !cfg.dry_run {
        anyhow::bail!(
            "RESULT_TO is required when any code path sends email. Set it or use \
             MODE=clipper DIGEST_MODE=file for disk-only output."
        );
    }
    let storage = Storage::open(&cfg.clipper_db).await?;
    let imported = storage.import_legacy_dedupe(&cfg.dedupe_file).await?;
    if imported > 0 {
        tracing::info!(
            imported,
            dedupe_file = %cfg.dedupe_file,
            clipper_db = %cfg.clipper_db,
            "imported legacy dedupe entries into sqlite"
        );
    }

    let (ai, seo_variant_blocks) = if cfg.dry_run {
        (None, Vec::new())
    } else {
        let api_key = cfg
            .openai_api_key
            .clone()
            .ok_or_else(|| anyhow::anyhow!("OPENAI_API_KEY is required unless --dry-run"))?;
        let openai = OpenAiClient::new(cfg.openai_base_url.clone(), api_key);

        let seo_system_prompt = tokio::fs::read_to_string(&cfg.seo_system_prompt_path)
            .await
            .with_context(|| format!("read seo system prompt at {}", &cfg.seo_system_prompt_path))?
            .trim()
            .to_string();

        let thumbnail_system_prompt = tokio::fs::read_to_string(&cfg.thumbnail_system_prompt_path)
            .await
            .with_context(|| {
                format!(
                    "read thumbnail system prompt at {}",
                    &cfg.thumbnail_system_prompt_path
                )
            })?
            .trim()
            .to_string();

        let seo_user_prompt_template = tokio::fs::read_to_string(&cfg.seo_user_prompt_path)
            .await
            .with_context(|| format!("read seo user prompt at {}", &cfg.seo_user_prompt_path))?
            .trim()
            .to_string();

        let thumbnail_user_prompt_template =
            tokio::fs::read_to_string(&cfg.thumbnail_user_prompt_path)
                .await
                .with_context(|| {
                    format!(
                        "read thumbnail user prompt at {}",
                        &cfg.thumbnail_user_prompt_path
                    )
                })?
                .trim()
                .to_string();

        let variants_raw = tokio::fs::read_to_string(&cfg.seo_variants_prompt_path)
            .await
            .with_context(|| {
                format!(
                    "read seo variants prompt at {}",
                    &cfg.seo_variants_prompt_path
                )
            })?;
        let seo_variant_blocks = parse_variants_prompt_file(&variants_raw);

        (
            Some(AiPipeline::new(
                openai,
                cfg.openai_stt_model.clone(),
                cfg.openai_chat_model.clone(),
                cfg.stt_concurrency,
                cfg.stt_rpm_limit,
                seo_system_prompt,
                thumbnail_system_prompt,
                seo_user_prompt_template,
                thumbnail_user_prompt_template,
            )),
            seo_variant_blocks,
        )
    };

    if let Some(local_path) = cfg.local_video_path.as_deref() {
        if mode.produces_clips() {
            let ai_ref = ai.as_ref().ok_or_else(|| {
                anyhow::anyhow!("clipper mode requires OPENAI_API_KEY (unset)")
            })?;
            // Generate a stable job ID from the local file path so retries
            // reuse the same row.
            let job_id = format!("local:{}", sanitize_filename(
                std::path::Path::new(local_path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(local_path),
            ));
            storage.create_job(&job_id, None, Some(local_path), None).await?;
            clipper::run_clipper_local_once(
                &cfg,
                google.as_ref(),
                &gmail,
                ai_ref,
                local_path,
                digest_mode,
                Some(&storage),
                Some(&job_id),
            )
            .await?;
        }
        if mode.produces_seo_emails() {
            let g = google
                .as_ref()
                .expect("google creds validated at startup for seo-only path");
            run_local_once(&cfg, g, &gmail, ai.as_ref(), local_path, &seo_variant_blocks)
                .await?;
        }
        return Ok(());
    }

    if mode.produces_clips() {
        tracing::warn!(
            "MODE={} but no LOCAL_VIDEO_PATH — the clipper polling-from-Gmail flow \
             is M2; M1 only supports MODE=clipper with LOCAL_VIDEO_PATH. The SEO-email \
             polling loop will run if MODE=both.",
            cfg.mode
        );
    }

    if !mode.produces_seo_emails() {
        tracing::info!("MODE={} has no polling work; exiting.", cfg.mode);
        return Ok(());
    }

    let g = google
        .as_ref()
        .expect("google creds validated at startup for polling path");
    loop {
        if let Err(e) = run_once(
            &cfg,
            g,
            &gmail,
            &drive,
            ai.as_ref(),
            &storage,
            &seo_variant_blocks,
        )
        .await
        {
            tracing::error!(error = ?e, "run_once failed");
        }

        if cfg.once {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(cfg.poll_interval_secs)).await;
    }

    Ok(())
}

async fn run_once(
    cfg: &Config,
    google: &GoogleAuth,
    gmail: &GmailClient,
    drive: &DriveClient,
    ai: Option<&AiPipeline>,
    storage: &Storage,
    seo_variant_blocks: &[String],
) -> anyhow::Result<()> {
    let access_token = google.access_token().await?;

    let message_ids = gmail
        .list_message_ids(&access_token, &cfg.gmail_query, cfg.gmail_max_results)
        .await?;
    if message_ids.is_empty() {
        tracing::info!("no messages matched");
        return Ok(());
    }

    for message_id in message_ids {
        if storage.job_exists(&message_id).await? {
            continue;
        }

        let msg = gmail.get_message_full(&access_token, &message_id).await?;
        let bodies = match gmail
            .extract_text_bodies_resolving_attachments(&access_token, &message_id, &msg)
            .await
        {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(message_id, error = %err, "failed to resolve message bodies via attachments.get; falling back to inline bodies only");
                GmailClient::extract_text_bodies(&msg)
            }
        };
        let mut file_ids = parse::extract_drive_file_ids(&bodies);
        if file_ids.is_empty() {
            // Some Drive-share emails don't embed a direct drive.google.com link in the MIME parts we decoded.
            // Fallback to parsing the raw RFC822 message.
            if let Ok(raw) = gmail
                .get_message_raw_rfc822(&access_token, &message_id)
                .await
            {
                file_ids = parse::extract_drive_file_ids(&raw);
                if file_ids.is_empty() {
                    if let Some(dump_dir) = &cfg.dump_dir {
                        let msg_json = serde_json::to_string_pretty(&msg).unwrap_or_default();
                        dump_failed_message(dump_dir, &message_id, &bodies, &raw, &msg_json)
                            .await
                            .ok();
                    }
                }
            } else if let Some(dump_dir) = &cfg.dump_dir {
                let msg_json = serde_json::to_string_pretty(&msg).unwrap_or_default();
                dump_failed_message(dump_dir, &message_id, &bodies, "", &msg_json)
                    .await
                    .ok();
            }
        }
        if file_ids.is_empty() {
            tracing::info!(message_id, "no drive file ids found in message");
            storage.mark_processed(&message_id).await?;
            continue;
        }

        // For MVP: process the first file id.
        let file_id = &file_ids[0];
        tracing::info!(message_id, file_id, "processing drive file");

        let meta = match drive.get_metadata(&access_token, file_id).await {
            Ok(m) => m,
            Err(e) => {
                // Common case: forwarded/old drive links or links you no longer have access to.
                tracing::warn!(message_id, file_id, error=?e, "failed to fetch drive metadata; skipping");
                storage.mark_processed(&message_id).await?;
                continue;
            }
        };
        tracing::info!(name=%meta.name, mime_type=%meta.mime_type, size=?meta.size, "drive metadata");

        if cfg.dry_run {
            // Mark processed so repeated dry-runs don't spam the same message.
            storage.mark_processed(&message_id).await?;
            continue;
        }

        let ai = ai.ok_or_else(|| anyhow::anyhow!("AI pipeline not configured"))?;
        let result_to = cfg
            .result_to
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("RESULT_TO is required unless --dry-run"))?;

        let name_lower = meta.name.to_lowercase();
        let mime_lower = meta.mime_type.to_lowercase();

        let is_wav = mime_lower.eq("audio/x-wav")
            || mime_lower.eq("audio/wav")
            || name_lower.ends_with(".wav");

        let is_audio = mime_lower.starts_with("audio/")
            || name_lower.ends_with(".wav")
            || name_lower.ends_with(".mp3")
            || name_lower.ends_with(".m4a")
            || name_lower.ends_with(".aac")
            || name_lower.ends_with(".flac")
            || name_lower.ends_with(".ogg")
            || name_lower.ends_with(".opus")
            || name_lower.ends_with(".wma")
            || name_lower.ends_with(".aiff")
            || name_lower.ends_with(".aif");

        let is_video = mime_lower.starts_with("video/")
            || name_lower.ends_with(".mp4")
            || name_lower.ends_with(".mov")
            || name_lower.ends_with(".mkv")
            || name_lower.ends_with(".webm")
            || name_lower.ends_with(".m4v")
            || name_lower.ends_with(".avi");

        if cfg.require_video && !is_video {
            tracing::info!(
                message_id,
                file_id,
                name = %meta.name,
                mime_type = %meta.mime_type,
                "skipping non-video drive file (require_video=true)"
            );
            storage.mark_processed(&message_id).await?;
            continue;
        }

        if !is_video && !is_audio {
            tracing::info!(
                message_id,
                file_id,
                name = %meta.name,
                mime_type = %meta.mime_type,
                "skipping unrecognized media type (not audio/video)"
            );
            storage.mark_processed(&message_id).await?;
            continue;
        }

        // Preflight: ensure ffmpeg tools exist before we download huge files (unless WAV).
        if !is_wav {
            ensure_tool_available(&cfg.ffmpeg, "ffmpeg").await?;
            ensure_tool_available(&cfg.ffprobe, "ffprobe").await?;
        }

        let job_dir = std::path::PathBuf::from(&cfg.work_dir)
            .join(sanitize_filename(&meta.name))
            .join(&message_id);
        tokio::fs::create_dir_all(&job_dir).await.ok();

        let video_path = job_dir.join(&meta.name);
        let expected_size_bytes = meta
            .size
            .as_deref()
            .and_then(|s| s.trim().parse::<u64>().ok());

        let need_download = match tokio::fs::metadata(&video_path).await {
            Ok(m) => {
                let actual = m.len();
                match expected_size_bytes {
                    Some(expected) if expected > 0 => {
                        if actual == expected {
                            false
                        } else {
                            tracing::warn!(
                                dest=%video_path.display(),
                                actual_bytes=actual,
                                expected_bytes=expected,
                                "existing file size mismatch; re-downloading"
                            );
                            true
                        }
                    }
                    _ => false,
                }
            }
            Err(_) => true,
        };

        if need_download {
            tracing::info!(dest=%video_path.display(), "downloading media from drive");
            drive
                .download_to_path(&access_token, file_id, &video_path)
                .await?;
        }

        // Build chunk list with offsets/durations.
        // Avoid per-chunk probing (N x ffprobe / wav decode) — derive from total duration and chunk size.
        let (chunks, total_duration_secs) = if is_wav {
            let total_duration = media::wav_duration_secs(&video_path)
                .await
                .context("wav duration")?;
            let chunk_secs = choose_chunk_secs(cfg, Some(total_duration));
            tracing::info!(
                total_secs = total_duration,
                chunk_secs,
                "chunk sizing (wav)"
            );

            let chunks_dir = job_dir.join("audio_chunks");
            clear_chunk_dir(&chunks_dir).await.ok();
            let chunk_paths = media::segment_wav(&video_path, &chunks_dir, chunk_secs).await?;
            if chunk_paths.is_empty() {
                anyhow::bail!("no wav chunks produced");
            }

            let base = chunk_secs as f64;
            let mut chunks: Vec<(std::path::PathBuf, f64, f64)> =
                Vec::with_capacity(chunk_paths.len());
            for (idx, p) in chunk_paths.into_iter().enumerate() {
                let offset = (idx as f64) * base;
                let mut dur = base;
                if offset + dur > total_duration {
                    dur = (total_duration - offset).max(0.0);
                }
                if !dur.is_finite() || dur <= 0.0 {
                    dur = base;
                }
                chunks.push((p, offset, dur));
            }
            (chunks, total_duration)
        } else {
            let audio_path = job_dir.join("audio.m4a");
            if !audio_path.exists() {
                if is_video {
                    media::extract_audio_m4a(&cfg.ffmpeg, &video_path, &audio_path).await?;
                } else {
                    media::transcode_audio_to_m4a(&cfg.ffmpeg, &video_path, &audio_path).await?;
                }
            }

            let total_duration = media::duration_secs(&cfg.ffprobe, &audio_path)
                .await
                .with_context(|| format!("ffprobe duration for {}", audio_path.display()))?;
            let chunk_secs = choose_chunk_secs(cfg, Some(total_duration));
            tracing::info!(
                total_secs = total_duration,
                chunk_secs,
                "chunk sizing (media)"
            );

            let chunks_dir = job_dir.join("audio_chunks");
            clear_chunk_dir(&chunks_dir).await.ok();
            let chunk_paths =
                media::segment_audio(&cfg.ffmpeg, &audio_path, &chunks_dir, chunk_secs).await?;
            if chunk_paths.is_empty() {
                anyhow::bail!("no audio chunks produced");
            }

            let base = chunk_secs as f64;
            let mut chunks: Vec<(std::path::PathBuf, f64, f64)> =
                Vec::with_capacity(chunk_paths.len());
            for (idx, p) in chunk_paths.into_iter().enumerate() {
                let offset = (idx as f64) * base;
                let mut dur = base;
                if offset + dur > total_duration {
                    dur = (total_duration - offset).max(0.0);
                }
                if !dur.is_finite() || dur <= 0.0 {
                    dur = base;
                }
                chunks.push((p, offset, dur));
            }
            (chunks, total_duration)
        };

        // Render stable progress bars (avoid scrolling output).
        let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        let bar_style = ProgressStyle::with_template(
            "{spinner:.green} {msg:12} {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar());

        let pb_transcribe = mp.add(ProgressBar::new(chunks.len() as u64));
        pb_transcribe.set_style(bar_style.clone());
        pb_transcribe.set_message("transcribing");
        pb_transcribe.enable_steady_tick(std::time::Duration::from_millis(120));

        let pb_thumbs = mp.add(ProgressBar::new(1));
        pb_thumbs.set_style(bar_style);
        pb_thumbs.set_message("waiting");
        pb_thumbs.enable_steady_tick(std::time::Duration::from_millis(120));

        let transcript = ai
            .transcribe_chunks(&chunks, Some(pb_transcribe.clone()))
            .await?;

        let show_context = match ai
            .infer_show_context(&meta.name, &transcript.full_text)
            .await
        {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                tracing::warn!(error = %e, name=%meta.name, "show inference failed; proceeding without show context");
                None
            }
        };

        if let Some(ctx) = show_context.as_ref() {
            tracing::info!(
                message_id,
                media_name = %meta.name,
                show_name = ?ctx.show_name,
                hosts = ?ctx.hosts,
                guest = ?ctx.guest,
                "inferred show context"
            );
        } else {
            tracing::info!(message_id, media_name = %meta.name, "no explicit show context inferred");
        }

        let variant_total = cfg.seo_variants.max(1);
        let selected_variant_instructions: Vec<String> = (0..variant_total)
            .map(|i| select_variant_instructions(seo_variant_blocks, i))
            .collect();

        if is_video {
            pb_thumbs.set_message("LLM (seo variants + thumbs)");
        } else {
            pb_thumbs.set_message("LLM (seo variants)");
        }

        let transcript_text = std::sync::Arc::new(transcript.full_text.clone());
        let media_name = meta.name.clone();
        let seo_fut = try_join_all(selected_variant_instructions.iter().enumerate().map(
            |(i, inst)| {
                let show_context = show_context.clone();
                let media_name = media_name.clone();
                let transcript_text = transcript_text.clone();
                async move {
                    ai.seo_variant_text_with_context(
                        transcript_text.as_str(),
                        inst.as_str(),
                        i,
                        variant_total,
                        show_context.as_ref(),
                        Some(media_name.as_str()),
                    )
                    .await
                }
            },
        ));

        let (seo_texts, moments) = if is_video {
            let (seo_texts_res, moments_res) =
                tokio::join!(seo_fut, ai.thumbnail_windows(&transcript.segments, cfg.thumbnail_slots));
            let seo_texts = seo_texts_res?;
            let moments = match moments_res {
                Ok(m) if !m.is_empty() => m,
                Ok(_) => fallback_thumbnail_moments(cfg.thumbnail_slots, total_duration_secs),
                Err(e) => {
                    tracing::warn!(error = %e, "thumbnail moment selection failed; using deterministic fallback moments");
                    fallback_thumbnail_moments(cfg.thumbnail_slots, total_duration_secs)
                }
            };
            (seo_texts, moments)
        } else {
            let seo_texts = seo_fut.await?;
            (seo_texts, Vec::new())
        };

        let mut attachments = Vec::new();

        if is_video {
            let thumbs_dir = job_dir.join("thumbnails");
            tokio::fs::create_dir_all(&thumbs_dir).await.ok();

            pb_thumbs.set_message("rendering");
            attachments = thumbs::generate_thumbnails(
                &cfg.ffmpeg,
                &video_path,
                &thumbs_dir,
                &moments,
                total_duration_secs,
                cfg.thumbnail_window_secs,
                cfg.thumbnail_count,
                cfg.thumbnail_max_height,
                cfg.thumbnail_ffmpeg_concurrency,
                Some(pb_thumbs.clone()),
            )
            .await;
        }

        let subject_base = if is_video {
            if attachments.is_empty() {
                format!("{} SEO package: {}", cfg.result_subject_prefix, meta.name)
            } else {
                format!(
                    "{} SEO package + thumbnails: {}",
                    cfg.result_subject_prefix, meta.name
                )
            }
        } else {
            format!("{} SEO package (audio): {}", cfg.result_subject_prefix, meta.name)
        };

        for (i, body) in seo_texts.iter().enumerate() {
            let subject = format!("{subject_base} ({}/{})", i + 1, variant_total);
            let raw_mime = build_mime_email("me", result_to, &subject, body, &attachments);
            let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
            let sent_message_id = gmail.send_raw(&access_token, &raw_b64url).await?;
            tracing::info!(
                sent_message_id,
                variant = i + 1,
                variants = variant_total,
                "sent result email"
            );
        }

        // Mark as processed only after all variant emails are sent.
        storage.mark_processed(&message_id).await?;
        tracing::info!("done");

        // MVP: only process the newest matching media message.
        break;
    }

    Ok(())
}

async fn run_local_once(
    cfg: &Config,
    google: &GoogleAuth,
    gmail: &GmailClient,
    ai: Option<&AiPipeline>,
    local_path: &str,
    seo_variant_blocks: &[String],
) -> anyhow::Result<()> {
    let ai = ai.ok_or_else(|| anyhow::anyhow!("AI pipeline not configured"))?;
    let result_to = cfg
        .result_to
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("RESULT_TO is required unless --dry-run"))?;

    let input_path = std::path::PathBuf::from(local_path);
    if tokio::fs::metadata(&input_path).await.is_err() {
        anyhow::bail!("local input does not exist: {}", input_path.display());
    }

    let file_name = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("local_media")
        .to_string();
    let lower = file_name.to_lowercase();

    let is_wav = lower.ends_with(".wav");
    let is_audio = is_wav
        || lower.ends_with(".mp3")
        || lower.ends_with(".m4a")
        || lower.ends_with(".aac")
        || lower.ends_with(".flac")
        || lower.ends_with(".ogg")
        || lower.ends_with(".opus")
        || lower.ends_with(".wma")
        || lower.ends_with(".aiff")
        || lower.ends_with(".aif");
    let is_video = lower.ends_with(".mp4")
        || lower.ends_with(".mov")
        || lower.ends_with(".mkv")
        || lower.ends_with(".webm")
        || lower.ends_with(".m4v")
        || lower.ends_with(".avi");

    if !is_video && !is_audio {
        anyhow::bail!("local input is not a recognized audio/video file: {file_name}");
    }

    if cfg.require_video && !is_video {
        anyhow::bail!("local input is not treated as video and REQUIRE_VIDEO=true: {file_name}");
    }

    if !is_wav {
        ensure_tool_available(&cfg.ffmpeg, "ffmpeg").await?;
        ensure_tool_available(&cfg.ffprobe, "ffprobe").await?;
    }

    let job_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let job_dir = std::path::PathBuf::from(&cfg.work_dir)
        .join("local")
        .join(sanitize_filename(&file_name))
        .join(job_ts.to_string());
    tokio::fs::create_dir_all(&job_dir).await.ok();

    let (chunks, total_duration_secs) = if is_wav {
        let total_duration = media::wav_duration_secs(&input_path)
            .await
            .context("wav duration")?;
        let chunk_secs = choose_chunk_secs(cfg, Some(total_duration));
        tracing::info!(
            total_secs = total_duration,
            chunk_secs,
            "chunk sizing (wav)"
        );

        let chunks_dir = job_dir.join("audio_chunks");
        clear_chunk_dir(&chunks_dir).await.ok();
        let chunk_paths = media::segment_wav(&input_path, &chunks_dir, chunk_secs).await?;
        if chunk_paths.is_empty() {
            anyhow::bail!("no wav chunks produced");
        }

        let base = chunk_secs as f64;
        let mut chunks: Vec<(std::path::PathBuf, f64, f64)> = Vec::with_capacity(chunk_paths.len());
        for (idx, p) in chunk_paths.into_iter().enumerate() {
            let offset = (idx as f64) * base;
            let mut dur = base;
            if offset + dur > total_duration {
                dur = (total_duration - offset).max(0.0);
            }
            if !dur.is_finite() || dur <= 0.0 {
                dur = base;
            }
            chunks.push((p, offset, dur));
        }
        (chunks, total_duration)
    } else {
        let audio_path = job_dir.join("audio.m4a");
        if !audio_path.exists() {
            if is_video {
                media::extract_audio_m4a(&cfg.ffmpeg, &input_path, &audio_path).await?;
            } else {
                media::transcode_audio_to_m4a(&cfg.ffmpeg, &input_path, &audio_path).await?;
            }
        }

        let total_duration = media::duration_secs(&cfg.ffprobe, &audio_path)
            .await
            .with_context(|| format!("ffprobe duration for {}", audio_path.display()))?;
        let chunk_secs = choose_chunk_secs(cfg, Some(total_duration));
        tracing::info!(
            total_secs = total_duration,
            chunk_secs,
            "chunk sizing (media)"
        );

        let chunks_dir = job_dir.join("audio_chunks");
        clear_chunk_dir(&chunks_dir).await.ok();
        let chunk_paths =
            media::segment_audio(&cfg.ffmpeg, &audio_path, &chunks_dir, chunk_secs).await?;
        if chunk_paths.is_empty() {
            anyhow::bail!("no audio chunks produced");
        }

        let base = chunk_secs as f64;
        let mut chunks: Vec<(std::path::PathBuf, f64, f64)> = Vec::with_capacity(chunk_paths.len());
        for (idx, p) in chunk_paths.into_iter().enumerate() {
            let offset = (idx as f64) * base;
            let mut dur = base;
            if offset + dur > total_duration {
                dur = (total_duration - offset).max(0.0);
            }
            if !dur.is_finite() || dur <= 0.0 {
                dur = base;
            }
            chunks.push((p, offset, dur));
        }
        (chunks, total_duration)
    };

    // Render stable progress bars (avoid scrolling output).
    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    let bar_style = ProgressStyle::with_template(
        "{spinner:.green} {msg:12} {pos}/{len} [{wide_bar:.cyan/blue}] {elapsed_precise}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar());

    let pb_transcribe = mp.add(ProgressBar::new(chunks.len() as u64));
    pb_transcribe.set_style(bar_style.clone());
    pb_transcribe.set_message("transcribing");
    pb_transcribe.enable_steady_tick(std::time::Duration::from_millis(120));

    let pb_thumbs = mp.add(ProgressBar::new(1));
    pb_thumbs.set_style(bar_style);
    pb_thumbs.set_message("waiting");
    pb_thumbs.enable_steady_tick(std::time::Duration::from_millis(120));

    let transcript = ai
        .transcribe_chunks(&chunks, Some(pb_transcribe.clone()))
        .await?;

    let show_context = match ai.infer_show_context(&file_name, &transcript.full_text).await {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            tracing::warn!(error = %e, file_name=%file_name, "show inference failed; proceeding without show context");
            None
        }
    };

    if let Some(ctx) = show_context.as_ref() {
        tracing::info!(
            media_name = %file_name,
            show_name = ?ctx.show_name,
            hosts = ?ctx.hosts,
            guest = ?ctx.guest,
            "inferred show context"
        );
    } else {
        tracing::info!(media_name = %file_name, "no explicit show context inferred");
    }

    let variant_total = cfg.seo_variants.max(1);
    let selected_variant_instructions: Vec<String> = (0..variant_total)
        .map(|i| select_variant_instructions(seo_variant_blocks, i))
        .collect();

    if is_video {
        pb_thumbs.set_message("LLM (seo variants + thumbs)");
    } else {
        pb_thumbs.set_message("LLM (seo variants)");
    }
    let transcript_text = std::sync::Arc::new(transcript.full_text.clone());
    let media_name = file_name.clone();
    let seo_fut = try_join_all(selected_variant_instructions.iter().enumerate().map(
        |(i, inst)| {
            let show_context = show_context.clone();
            let media_name = media_name.clone();
            let transcript_text = transcript_text.clone();
            async move {
                ai.seo_variant_text_with_context(
                    transcript_text.as_str(),
                    inst.as_str(),
                    i,
                    variant_total,
                    show_context.as_ref(),
                    Some(media_name.as_str()),
                )
                .await
            }
        },
    ));

    let (seo_texts, moments) = if is_video {
        let (seo_texts_res, moments_res) =
            tokio::join!(seo_fut, ai.thumbnail_windows(&transcript.segments, cfg.thumbnail_slots));
        let seo_texts = seo_texts_res?;
        let moments = match moments_res {
            Ok(m) if !m.is_empty() => m,
            Ok(_) => fallback_thumbnail_moments(cfg.thumbnail_slots, total_duration_secs),
            Err(e) => {
                tracing::warn!(error = %e, "thumbnail moment selection failed; using deterministic fallback moments");
                fallback_thumbnail_moments(cfg.thumbnail_slots, total_duration_secs)
            }
        };
        (seo_texts, moments)
    } else {
        let seo_texts = seo_fut.await?;
        (seo_texts, Vec::new())
    };

    let mut attachments = Vec::new();
    if is_video {
        let thumbs_dir = job_dir.join("thumbnails");
        tokio::fs::create_dir_all(&thumbs_dir).await.ok();
        pb_thumbs.set_message("rendering");
        attachments = thumbs::generate_thumbnails(
            &cfg.ffmpeg,
            &input_path,
            &thumbs_dir,
            &moments,
            total_duration_secs,
            cfg.thumbnail_window_secs,
            cfg.thumbnail_count,
            cfg.thumbnail_max_height,
            cfg.thumbnail_ffmpeg_concurrency,
            Some(pb_thumbs.clone()),
        )
        .await;
    }

    let access_token = google.access_token().await?;
    let subject_base = if is_video {
        if attachments.is_empty() {
            format!("{} SEO package: {}", cfg.result_subject_prefix, file_name)
        } else {
            format!(
                "{} SEO package + thumbnails: {}",
                cfg.result_subject_prefix, file_name
            )
        }
    } else {
        format!("{} SEO package (audio): {}", cfg.result_subject_prefix, file_name)
    };

    for (i, body) in seo_texts.iter().enumerate() {
        let subject = format!("{subject_base} ({}/{})", i + 1, variant_total);
        let raw_mime = build_mime_email("me", result_to, &subject, body, &attachments);
        let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
        let sent_message_id = gmail.send_raw(&access_token, &raw_b64url).await?;
        tracing::info!(
            sent_message_id,
            variant = i + 1,
            variants = variant_total,
            "sent result email (local)"
        );
    }

    Ok(())
}

fn parse_variants_prompt_file(s: &str) -> Vec<String> {
    s.split("\n---\n")
        .map(|block| {
            let mut lines = block.lines();
            // Drop leading empty/comment lines to avoid leaking header comments into the first variant.
            while let Some(line) = lines.clone().next() {
                let t = line.trim();
                if t.is_empty() || t.starts_with('#') {
                    lines.next();
                    continue;
                }
                break;
            }
            lines.collect::<Vec<_>>().join("\n").trim().to_string()
        })
        .filter(|b| !b.trim().is_empty())
        .collect()
}

fn select_variant_instructions(blocks: &[String], idx: usize) -> String {
    if blocks.is_empty() {
        return String::new();
    }
    blocks[idx % blocks.len()].trim().to_string()
}

fn fallback_thumbnail_moments(slots: usize, total_duration_secs: f64) -> Vec<ThumbnailMoment> {
    let n = slots.max(1);
    let total = total_duration_secs.max(0.0);
    let eof_epsilon_secs = 0.25;
    let last_ts = if total.is_finite() && total > 0.0 {
        (total - eof_epsilon_secs).max(0.0)
    } else {
        0.0
    };

    (0..n)
        .map(|i| {
            let frac = (i as f64 + 1.0) / (n as f64 + 1.0);
            let ts = (total * frac).max(0.0).min(last_ts);
            ThumbnailMoment {
                center_seconds: ts,
                reason: "fallback".to_string(),
            }
        })
        .collect()
}

async fn ensure_tool_available(cmd: &str, label: &str) -> anyhow::Result<()> {
    let status = tokio::process::Command::new(cmd)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(_) => anyhow::bail!("{label} exists but failed to run: {cmd}"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "{label} not found: set env {label_upper} or install it (e.g. `sudo apt-get install -y ffmpeg`). Tried: {cmd}",
                label_upper = label.to_ascii_uppercase()
            );
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
        .collect::<String>()
}

async fn clear_chunk_dir(dir: &std::path::Path) -> anyhow::Result<()> {
    if tokio::fs::metadata(dir).await.is_err() {
        tokio::fs::create_dir_all(dir).await.ok();
        return Ok(());
    }

    let mut entries = tokio::fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        if file_name.starts_with("chunk_") {
            tokio::fs::remove_file(&path).await.ok();
        }
    }
    Ok(())
}

fn choose_chunk_secs(cfg: &Config, total_duration: Option<f64>) -> u64 {
    if !cfg.auto_chunking {
        return cfg.audio_chunk_secs;
    }

    let duration = match total_duration {
        Some(d) if d.is_finite() && d > 0.0 => d,
        _ => return cfg.audio_chunk_secs,
    };

    let target_chunks = if cfg.auto_chunk_target_chunks > 0 {
        cfg.auto_chunk_target_chunks
    } else {
        let concurrency = cfg.stt_concurrency.max(1);
        let factor = cfg.auto_chunk_target_factor.max(1);
        concurrency.saturating_mul(factor).max(1)
    };
    let mut chunk = (duration / target_chunks as f64).ceil();
    if chunk < cfg.auto_chunk_min_secs as f64 {
        chunk = cfg.auto_chunk_min_secs as f64;
    }
    if chunk > cfg.auto_chunk_max_secs as f64 {
        chunk = cfg.auto_chunk_max_secs as f64;
    }

    chunk as u64
}

async fn dump_failed_message(
    dump_dir: &str,
    message_id: &str,
    decoded_parts: &str,
    raw_rfc822: &str,
    message_json: &str,
) -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(dump_dir);
    tokio::fs::create_dir_all(&dir).await.ok();

    let parts_path = dir.join(format!("{message_id}_parts.txt"));
    tokio::fs::write(&parts_path, decoded_parts).await.ok();

    if !message_json.is_empty() {
        let json_path = dir.join(format!("{message_id}_message.json"));
        tokio::fs::write(&json_path, message_json).await.ok();
    }

    if !raw_rfc822.is_empty() {
        let raw_path = dir.join(format!("{message_id}.eml"));
        tokio::fs::write(&raw_path, raw_rfc822).await.ok();

        // Also write a URL list for quick inspection.
        let url_re = regex::Regex::new(r#"https?://[^\s\"'<>()]+"#).expect("valid url regex");
        let mut urls: Vec<&str> = url_re.find_iter(raw_rfc822).map(|m| m.as_str()).collect();
        urls.sort();
        urls.dedup();
        let urls_path = dir.join(format!("{message_id}_urls.txt"));
        tokio::fs::write(&urls_path, urls.join("\n")).await.ok();
    }

    Ok(())
}
