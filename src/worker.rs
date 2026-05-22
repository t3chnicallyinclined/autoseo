//! Background job-runner loop, used when the dashboard server (`MODE=server`)
//! wants to also drive clipper runs without a separate worker process.
//!
//! Each tick:
//!   1. atomically claim the oldest `pending` job (status flips to `transcribed`
//!      — that's the first non-pending state the existing schema knows about),
//!   2. download the source if it's URL-only,
//!   3. invoke the existing clipper pipeline,
//!   4. let the pipeline mark the job done/failed.
//!
//! Runs a single job at a time on purpose: clipper renders are
//! ffmpeg-CPU-bound and STT calls already saturate the configured concurrency
//! budget, so parallel jobs would just thrash. Multi-job throughput is a
//! later concern (would need a job queue with worker pool).

use std::time::Duration;

use anyhow::{Context, Result};
use tokio::time::{MissedTickBehavior, interval};

use crate::ai_pipeline::{AiPipeline, SttBackend};
use crate::clipper;
use crate::config::Config;
use crate::events::{EventBus, PipelineEvent, dashboard_view};
use crate::gmail::GmailClient;
use crate::openai::OpenAiClient;
use crate::show_config::{DigestMode, GlobalPromptPaths, PromptLoader, PromptName};
use crate::storage::{JobStatus, Storage};

const POLL_INTERVAL_SECS: u64 = 5;

/// Run the worker loop until the host process is cancelled. Designed to be
/// spawned via `tokio::spawn` alongside the axum server. `bus` is cloned into
/// each pipeline invocation so per-stage events reach all connected
/// dashboard WebSocket clients.
pub async fn run(cfg: Config, storage: Storage, bus: EventBus) {
    let mut tick = interval(Duration::from_secs(POLL_INTERVAL_SECS));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tracing::info!(
        poll_secs = POLL_INTERVAL_SECS,
        "dashboard worker: polling for pending jobs"
    );
    loop {
        tick.tick().await;
        match storage.claim_next_pending_job().await {
            Ok(Some(job)) => {
                let job_id = job.id.clone();
                tracing::info!(job_id, media = ?job.media_name, "worker: starting job");
                let result = run_job(&cfg, &storage, &bus, &job).await;
                match result {
                    Ok(()) => tracing::info!(job_id, "worker: job complete"),
                    Err(e) => {
                        let err_msg = format!("{e:#}");
                        tracing::error!(job_id, error = ?e, "worker: job failed");
                        // The clipper marks the row failed on its own error
                        // path; fall through in case it didn't. Either way
                        // we publish a final event so the dashboard's stuck
                        // job indicator clears.
                        let _ = storage
                            .update_job_status(&job_id, JobStatus::Failed, Some(&err_msg))
                            .await;
                        let (status, stage, progress) = dashboard_view("failed");
                        bus.emit(PipelineEvent::JobUpdate {
                            job_id: job_id.clone(),
                            status: status.to_string(),
                            stage: stage.to_string(),
                            progress,
                            clips_generated: None,
                            media: job.media_name.clone(),
                        });
                        bus.emit(PipelineEvent::JobFailed {
                            job_id,
                            media: job.media_name.clone().unwrap_or_default(),
                            error: err_msg,
                        });
                    }
                }
            }
            Ok(None) => { /* nothing to do this tick */ }
            Err(e) => tracing::warn!(error = ?e, "worker: claim_next_pending_job failed"),
        }
    }
}

async fn run_job(
    cfg: &Config,
    storage: &Storage,
    bus: &EventBus,
    job: &crate::storage::JobRow,
) -> Result<()> {
    let local_path = ensure_local_source(cfg, storage, job).await?;

    let openai_api_key = cfg
        .openai_api_key
        .clone()
        .context("OPENAI_API_KEY is required to run a clipper job")?;
    let openai_client = OpenAiClient::new(cfg.openai_base_url.clone(), openai_api_key);

    // Build the AiPipeline with prompts resolved through the same per-show
    // override loader the standalone MODE=clipper path uses.
    let loader = PromptLoader::new(
        &cfg.shows_dir,
        GlobalPromptPaths {
            seo_system: std::path::PathBuf::from(&cfg.seo_system_prompt_path),
            seo_user: std::path::PathBuf::from(&cfg.seo_user_prompt_path),
            seo_variants: std::path::PathBuf::from(&cfg.seo_variants_prompt_path),
            thumbnail_system: std::path::PathBuf::from(&cfg.thumbnail_system_prompt_path),
            thumbnail_user: std::path::PathBuf::from(&cfg.thumbnail_user_prompt_path),
        },
    );
    let show_slug = job.show_slug.as_deref();
    let seo_system = loader.load(PromptName::SeoSystem, show_slug).await?;
    let seo_user = loader.load(PromptName::SeoUser, show_slug).await?;
    let thumbnail_system = loader.load(PromptName::ThumbnailSystem, show_slug).await?;
    let thumbnail_user = loader.load(PromptName::ThumbnailUser, show_slug).await?;

    let stt_backend = SttBackend::parse(&cfg.stt_backend)?;
    let ai = AiPipeline::new(
        openai_client,
        cfg.openai_stt_model.clone(),
        cfg.openai_chat_model.clone(),
        cfg.stt_concurrency,
        cfg.stt_rpm_limit,
        seo_system,
        thumbnail_system,
        seo_user,
        thumbnail_user,
    )
    .with_stt_backend(stt_backend)
    .with_ffmpeg_path(cfg.ffmpeg.clone());

    let gmail = GmailClient::new();
    clipper::run_clipper_local_once(
        cfg,
        None, // no Google credentials needed for file-mode digest
        &gmail,
        &ai,
        &local_path,
        DigestMode::File,
        Some(storage),
        Some(&job.id),
        Some(bus),
    )
    .await
}

/// If the job came in as a URL rather than an upload, download it into
/// `WORK_DIR/uploads/<job_id>/` before handing off to the clipper. Returns
/// the absolute local path the clipper should read.
async fn ensure_local_source(
    cfg: &Config,
    storage: &Storage,
    job: &crate::storage::JobRow,
) -> Result<String> {
    if let Some(path) = job.local_path.as_deref() {
        if tokio::fs::metadata(path).await.is_ok() {
            return Ok(path.to_string());
        }
        tracing::warn!(
            job_id = job.id,
            stale_path = path,
            "worker: local_path set but file missing — falling back to source_url"
        );
    }
    let url = job
        .source_url
        .as_deref()
        .context("job has no local_path and no source_url")?;

    let dest_dir = std::path::PathBuf::from(&cfg.work_dir)
        .join("uploads")
        .join(&job.id);
    tokio::fs::create_dir_all(&dest_dir).await.ok();

    // YouTube / TikTok / Twitch / etc.: use yt-dlp (handles HLS, signed URLs,
    // the whole zoo). Plain mp4 URLs: just stream the body.
    let dest = if is_streaming_site(url) {
        download_via_ytdlp(url, &dest_dir)
            .await
            .context("yt-dlp download")?
    } else {
        let filename = filename_from_url(url).unwrap_or_else(|| format!("{}.mp4", job.id));
        let path = dest_dir.join(&filename);
        tracing::info!(job_id = job.id, url, dest = %path.display(), "worker: downloading source (direct)");
        download_to_file(url, &path)
            .await
            .context("download source")?;
        path
    };

    let path_str = dest.display().to_string();
    storage.update_job_local_path(&job.id, &path_str).await.ok();
    Ok(path_str)
}

/// Heuristic: if the URL host looks like a streaming-site landing page,
/// route it through yt-dlp instead of a naive HTTP GET (which would just
/// fetch the HTML watch page).
fn is_streaming_site(url: &str) -> bool {
    let lower = url.to_lowercase();
    [
        "youtube.com",
        "youtu.be",
        "tiktok.com",
        "twitch.tv",
        "vimeo.com",
        "drive.google.com",
        "x.com/",
        "twitter.com/",
    ]
    .iter()
    .any(|d| lower.contains(d))
}

async fn download_via_ytdlp(url: &str, dest_dir: &std::path::Path) -> Result<std::path::PathBuf> {
    // Resolve binary: PATH first, then ~/.local/bin/yt-dlp as a fallback.
    let bin =
        which_ytdlp().context("yt-dlp binary not found — install via `pip install yt-dlp`")?;
    let template = dest_dir.join("%(title).80s.%(ext)s");

    tracing::info!(url, %bin, dest_dir = %dest_dir.display(), "worker: yt-dlp downloading");

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("--no-progress")
        .arg("--no-playlist")
        // Prefer mp4 / merged with the best video+audio combo we can.
        .arg("-f")
        .arg("bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b")
        .arg("--merge-output-format")
        .arg("mp4")
        .arg("-o")
        .arg(&template)
        .arg("--print")
        .arg("after_move:filepath");

    // YouTube increasingly blocks anonymous downloads. The operator can wire
    // cookies via either:
    //   YTDLP_COOKIES_BROWSER=firefox|chrome|chromium|edge|safari|brave
    //   YTDLP_COOKIES_FILE=/path/to/cookies.txt
    // The browser variant pulls cookies from the locally-installed browser's
    // profile (no manual export step). The file variant is a Netscape-format
    // cookies.txt exported via an extension like "Get cookies.txt".
    if let Ok(browser) = std::env::var("YTDLP_COOKIES_BROWSER") {
        if !browser.trim().is_empty() {
            cmd.arg("--cookies-from-browser").arg(browser.trim());
        }
    }
    if let Ok(file) = std::env::var("YTDLP_COOKIES_FILE") {
        if !file.trim().is_empty() {
            cmd.arg("--cookies").arg(file.trim());
        }
    }

    // yt-dlp's newer YouTube extractor wants a JS runtime to evaluate the
    // player script. Try deno first (yt-dlp's default), then fall back to
    // node (which most systems already have). Either avoids the "no JS
    // runtime" warning and unlocks the latest extractor formats.
    if let Some(deno) = which_runtime("deno", ".deno/bin/deno") {
        cmd.arg("--js-runtimes").arg(format!("deno:{deno}"));
    } else if let Some(node) = which_node() {
        cmd.arg("--js-runtimes").arg(format!("node:{node}"));
    }

    // Forward a real-looking user agent — vanilla yt-dlp UA is increasingly
    // flagged. Configurable for power users.
    let ua = std::env::var("YTDLP_USER_AGENT").ok().unwrap_or_else(||
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_string()
    );
    if !ua.is_empty() {
        cmd.arg("--user-agent").arg(ua);
    }

    cmd.arg(url);
    let output = cmd.output().await.context("spawn yt-dlp")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("yt-dlp exit {} — {}", output.status, stderr.trim());
    }

    // The very last non-empty line of stdout is the saved filepath (from
    // --print after_move:filepath). Trust nothing else, since yt-dlp can
    // print other diagnostic lines first.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let path = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|l| std::path::PathBuf::from(l.trim()))
        .context("yt-dlp printed no filepath")?;
    if !path.is_file() {
        anyhow::bail!("yt-dlp reported {} but file is missing", path.display());
    }
    Ok(path)
}

fn which_ytdlp() -> Option<String> {
    if let Ok(p) = std::process::Command::new("yt-dlp")
        .arg("--version")
        .output()
    {
        if p.status.success() {
            return Some("yt-dlp".to_string());
        }
    }
    let home = std::env::var("HOME").ok()?;
    let local = format!("{home}/.local/bin/yt-dlp");
    if std::path::Path::new(&local).is_file() {
        Some(local)
    } else {
        None
    }
}

/// Probe for an executable: PATH first, then `~/relative_fallback`.
fn which_runtime(name: &str, home_fallback: &str) -> Option<String> {
    if let Ok(p) = std::process::Command::new(name).arg("--version").output() {
        if p.status.success() {
            return Some(name.to_string());
        }
    }
    let home = std::env::var("HOME").ok()?;
    let candidate = format!("{home}/{home_fallback}");
    if std::path::Path::new(&candidate).is_file() {
        Some(candidate)
    } else {
        None
    }
}

/// Find a node binary. Checks PATH, then `~/.nvm/versions/node/*/bin/node`
/// for users using nvm.
fn which_node() -> Option<String> {
    if let Some(n) = which_runtime("node", ".nvm/current/bin/node") {
        return Some(n);
    }
    let home = std::env::var("HOME").ok()?;
    let nvm_dir = std::path::PathBuf::from(format!("{home}/.nvm/versions/node"));
    if !nvm_dir.is_dir() {
        return None;
    }
    // Pick the lexicographically-highest version (good-enough proxy for
    // newest, since nvm dirs are named like v22.22.0).
    let mut best: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(&nvm_dir).ok()? {
        let entry = entry.ok()?;
        let bin = entry.path().join("bin/node");
        if bin.is_file() && (best.is_none() || best.as_ref().is_some_and(|b| bin > *b)) {
            best = Some(bin);
        }
    }
    best.map(|p| p.display().to_string())
}

async fn download_to_file(url: &str, dest: &std::path::Path) -> Result<()> {
    use tokio::io::AsyncWriteExt as _;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;
    let mut resp = client.get(url).send().await?.error_for_status()?;
    let mut file = tokio::fs::File::create(dest).await?;
    while let Some(chunk) = resp.chunk().await? {
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    Ok(())
}

fn filename_from_url(url: &str) -> Option<String> {
    let no_query = url.split('?').next()?;
    let last = no_query.rsplit('/').next()?;
    if last.is_empty() {
        None
    } else {
        Some(last.to_string())
    }
}
