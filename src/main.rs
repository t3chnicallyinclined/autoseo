mod ai_pipeline;
mod config;
mod dedupe;
mod drive;
mod gmail;
mod google_auth;
mod media;
mod mime;
mod openai;
mod parse;
mod rate_limit;
mod thumbs;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use ai_pipeline::AiPipeline;
use config::Config;
use dedupe::FileBackedDedupe;
use drive::DriveClient;
use gmail::GmailClient;
use google_auth::GoogleAuth;
use mime::build_mime_email;
use openai::OpenAiClient;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cfg = Config::parse();

    let google = GoogleAuth::new(
        cfg.google_client_id.clone(),
        cfg.google_client_secret.clone(),
        cfg.google_refresh_token.clone(),
    );
    let gmail = GmailClient::new();
    let drive = DriveClient::new();
    let mut dedupe = FileBackedDedupe::load(&cfg.dedupe_file).await?;

    let ai = if cfg.dry_run {
        None
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
        ))
    };

    if let Some(local_path) = cfg.local_video_path.as_deref() {
        run_local_once(&cfg, &google, &gmail, ai.as_ref(), local_path).await?;
        return Ok(());
    }

    loop {
        if let Err(e) = run_once(&cfg, &google, &gmail, &drive, ai.as_ref(), &mut dedupe).await {
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
    dedupe: &mut FileBackedDedupe,
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
        if dedupe.contains(&message_id) {
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
            dedupe.insert(message_id).await?;
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
                dedupe.insert(message_id).await?;
                continue;
            }
        };
        tracing::info!(name=%meta.name, mime_type=%meta.mime_type, size=?meta.size, "drive metadata");

        if cfg.dry_run {
            // Mark processed so repeated dry-runs don't spam the same message.
            dedupe.insert(message_id).await?;
            continue;
        }

        let ai = ai.ok_or_else(|| anyhow::anyhow!("AI pipeline not configured"))?;
        let result_to = cfg
            .result_to
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("RESULT_TO is required unless --dry-run"))?;

        let is_wav = meta.mime_type.eq_ignore_ascii_case("audio/x-wav")
            || meta.mime_type.eq_ignore_ascii_case("audio/wav")
            || meta.name.to_lowercase().ends_with(".wav");

        let is_video = meta.mime_type.to_lowercase().starts_with("video/")
            || meta.name.to_lowercase().ends_with(".mp4");

        if cfg.require_video && !is_video {
            tracing::info!(
                message_id,
                file_id,
                name = %meta.name,
                mime_type = %meta.mime_type,
                "skipping non-video drive file (require_video=true)"
            );
            dedupe.insert(message_id).await?;
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
        if !video_path.exists() {
            tracing::info!(dest=%video_path.display(), "downloading video from drive");
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
                media::extract_audio_m4a(&cfg.ffmpeg, &video_path, &audio_path).await?;
            }

            let total_duration = media::duration_secs(&cfg.ffprobe, &audio_path)
                .await
                .with_context(|| format!("ffprobe duration for {}", audio_path.display()))?;
            let chunk_secs = choose_chunk_secs(cfg, Some(total_duration));
            tracing::info!(
                total_secs = total_duration,
                chunk_secs,
                "chunk sizing (video)"
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

        pb_thumbs.set_message("LLM (seo+thumbs)");
        let (seo_res, moments_res) = tokio::join!(
            ai.seo_package(&transcript.full_text),
            ai.thumbnail_windows(&transcript.segments, cfg.thumbnail_slots)
        );
        let seo = seo_res?;
        let moments = match moments_res {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "thumbnail moment selection failed; continuing without moments");
                Vec::new()
            }
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

        let subject = if attachments.is_empty() {
            format!("{} SEO package: {}", cfg.result_subject_prefix, meta.name)
        } else {
            format!(
                "{} SEO package + thumbnails: {}",
                cfg.result_subject_prefix, meta.name
            )
        };

        let mut body = format!(
            "SEO DESCRIPTION:\n\n{}\n\nHASHTAGS:\n{}\n\nTAGS (<=500 chars, comma-separated):\n{}\n\n",
            seo.description.trim(),
            seo.hashtags
                .iter()
                .map(|h| format!("#{}", h))
                .collect::<Vec<_>>()
                .join(" "),
            seo.tags_csv.trim()
        );

        if !moments.is_empty() {
            body.push_str("THUMBNAIL-WORTHY MOMENTS (timestamps):\n");
            for m in &moments {
                body.push_str(&format!(
                    "- {} — {}\n",
                    format_hhmmss(m.center_seconds),
                    m.reason.trim()
                ));
            }
            body.push('\n');
        }

        if attachments.is_empty() && is_video {
            body.push_str("NOTE: No thumbnails attached (ffmpeg/screenshot failed).\n\n");
        } else if attachments.is_empty() && !is_video {
            body.push_str("NOTE: No thumbnails attached (input is not a video).\n\n");
        }

        // Gmail API will set the From based on authenticated user, but RFC headers still help.
        let raw_mime = build_mime_email("me", result_to, &subject, &body, &attachments);
        let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
        let sent_message_id = gmail.send_raw(&access_token, &raw_b64url).await?;
        tracing::info!(sent_message_id, "sent result email");

        // Mark as processed.
        dedupe.insert(message_id).await?;
        tracing::info!("done");

        // MVP: only process the newest matching *video* message.
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
    let is_video = lower.ends_with(".mp4") || !is_wav;

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
        tracing::info!(total_secs = total_duration, chunk_secs, "chunk sizing (wav)");

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
            media::extract_audio_m4a(&cfg.ffmpeg, &input_path, &audio_path).await?;
        }

        let total_duration = media::duration_secs(&cfg.ffprobe, &audio_path)
            .await
            .with_context(|| format!("ffprobe duration for {}", audio_path.display()))?;
        let chunk_secs = choose_chunk_secs(cfg, Some(total_duration));
        tracing::info!(total_secs = total_duration, chunk_secs, "chunk sizing (video)");

        let chunks_dir = job_dir.join("audio_chunks");
        clear_chunk_dir(&chunks_dir).await.ok();
        let chunk_paths = media::segment_audio(&cfg.ffmpeg, &audio_path, &chunks_dir, chunk_secs).await?;
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

    pb_thumbs.set_message("LLM (seo+thumbs)");
    let (seo_res, moments_res) = tokio::join!(
        ai.seo_package(&transcript.full_text),
        ai.thumbnail_windows(&transcript.segments, cfg.thumbnail_slots)
    );
    let seo = seo_res?;
    let moments = match moments_res {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "thumbnail moment selection failed; continuing without moments");
            Vec::new()
        }
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

    let subject = if attachments.is_empty() {
        format!("{} SEO package: {}", cfg.result_subject_prefix, file_name)
    } else {
        format!(
            "{} SEO package + thumbnails: {}",
            cfg.result_subject_prefix, file_name
        )
    };

    let mut body = format!(
        "SEO DESCRIPTION:\n\n{}\n\nHASHTAGS:\n{}\n\nTAGS (<=500 chars, comma-separated):\n{}\n\n",
        seo.description.trim(),
        seo.hashtags
            .iter()
            .map(|h| format!("#{}", h))
            .collect::<Vec<_>>()
            .join(" "),
        seo.tags_csv.trim()
    );

    if !moments.is_empty() {
        body.push_str("THUMBNAIL-WORTHY MOMENTS (timestamps):\n");
        for m in &moments {
            body.push_str(&format!(
                "- {} — {}\n",
                format_hhmmss(m.center_seconds),
                m.reason.trim()
            ));
        }
        body.push('\n');
    }

    if attachments.is_empty() && is_video {
        body.push_str("NOTE: No thumbnails attached (ffmpeg/screenshot failed).\n\n");
    } else if attachments.is_empty() && !is_video {
        body.push_str("NOTE: No thumbnails attached (input is not a video).\n\n");
    }

    let access_token = google.access_token().await?;
    let raw_mime = build_mime_email("me", result_to, &subject, &body, &attachments);
    let raw_b64url = URL_SAFE_NO_PAD.encode(raw_mime);
    let sent_message_id = gmail.send_raw(&access_token, &raw_b64url).await?;
    tracing::info!(sent_message_id, "sent result email (local)");
    Ok(())
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

fn format_hhmmss(secs: f64) -> String {
    let mut s = secs;
    if !s.is_finite() || s < 0.0 {
        s = 0.0;
    }
    let total = s.floor() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let sec = total % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{sec:02}")
    } else {
        format!("{m:02}:{sec:02}")
    }
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
