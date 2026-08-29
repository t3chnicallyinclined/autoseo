use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "autoseo")]
pub struct Config {
    /// Gmail search query used to find candidate messages.
    #[arg(
        long,
        env = "GMAIL_QUERY",
        default_value = "from:drive-shares-dm-noreply@google.com subject:\"Item shared with you\" has:drive"
    )]
    pub gmail_query: String,

    /// Where to send the result email.
    /// Required unless running with `--dry-run`.
    #[arg(long, env = "RESULT_TO")]
    pub result_to: Option<String>,

    /// Optional subject prefix for result emails.
    #[arg(long, env = "RESULT_SUBJECT_PREFIX", default_value = "[autoseo]")]
    pub result_subject_prefix: String,

    /// Poll interval in seconds.
    #[arg(long, env = "POLL_INTERVAL_SECS", default_value_t = 60)]
    pub poll_interval_secs: u64,

    /// Max number of Gmail messages to fetch per poll (newest-first).
    #[arg(long, env = "GMAIL_MAX_RESULTS", default_value_t = 10)]
    pub gmail_max_results: u32,

    /// If set, only process Drive files that look like videos (mimeType starts with video/ or common video extensions).
    /// When false, audio files (e.g. mp3/m4a/wav) are also accepted.
    #[arg(long, env = "REQUIRE_VIDEO", default_value_t = false)]
    pub require_video: bool,

    /// If set, run a single poll cycle and exit.
    #[arg(long)]
    pub once: bool,

    /// If set, only fetch matching Gmail messages, extract Drive file IDs, and print metadata.
    /// Skips download, ffmpeg, transcription, LLM calls, thumbnails, and sending email.
    #[arg(long)]
    pub dry_run: bool,

    /// Directory for large working files (video, audio, thumbnails).
    #[arg(long, env = "WORK_DIR", default_value = "./work")]
    pub work_dir: String,

    /// File-backed dedupe list (newline-delimited Gmail message IDs).
    /// Read once on startup and imported into the SQLite jobs table; thereafter
    /// dedupe lives in `clipper_db`.
    #[arg(
        long,
        env = "DEDUPE_FILE",
        default_value = "./work/processed_message_ids.txt"
    )]
    pub dedupe_file: String,

    /// SQLite database path. Holds jobs/clips/posts/analytics/trends.
    /// Survives container restarts when WORK_DIR is mounted.
    #[arg(long, env = "CLIPPER_DB", default_value = "./work/clipper.db")]
    pub clipper_db: String,

    /// Pipeline mode. Accepted: "seo-only" (current behavior — emails SEO packages only),
    /// "clipper" (new — produces clips and digest email; M1+), "both" (runs both).
    /// Default keeps existing behavior unchanged until the clipper path is feature-complete.
    #[arg(long, env = "MODE", default_value = "seo-only")]
    pub mode: String,

    /// Root directory for per-show prompt overrides.
    /// Layout: {shows_dir}/{show_slug}/{seo_system,seo_user,seo_variants,thumbnail_system,thumbnail_user}.txt
    /// A missing file falls back to the global prompt path.
    #[arg(long, env = "SHOWS_DIR", default_value = "./prompts/shows")]
    pub shows_dir: String,

    /// Cache directory for fastembed model files (MiniLM-L6-v2, ~90MB).
    /// First run downloads from Hugging Face; subsequent runs reuse the cached files.
    /// Set this under WORK_DIR so the download persists across container restarts.
    #[arg(
        long,
        env = "EMBED_MODEL_DIR",
        default_value = "./work/models/fastembed"
    )]
    pub embed_model_dir: String,

    /// LLM clip-ranker system prompt path.
    #[arg(
        long,
        env = "CLIP_RANKER_SYSTEM_PROMPT_PATH",
        default_value = "./prompts/clips/ranker_system.txt"
    )]
    pub clip_ranker_system_prompt_path: String,

    /// LLM clip-ranker user prompt template path. Supports `{{candidates_json}}`,
    /// `{{show_name}}`, `{{hosts}}`, `{{guest}}`.
    #[arg(
        long,
        env = "CLIP_RANKER_USER_PROMPT_PATH",
        default_value = "./prompts/clips/ranker_user.txt"
    )]
    pub clip_ranker_user_prompt_path: String,

    /// Number of top-ranked clips to render and include in the digest email.
    #[arg(long, env = "CLIP_TOP_K", default_value_t = 10)]
    pub clip_top_k: usize,

    /// System prompt for the per-clip social-media copy generator.
    #[arg(
        long,
        env = "CLIP_SOCIAL_SYSTEM_PROMPT_PATH",
        default_value = "./prompts/clips/social_system.txt"
    )]
    pub clip_social_system_prompt_path: String,

    /// User prompt template for the per-clip social-media copy generator.
    /// Supports `{{show_name}}`, `{{hosts}}`, `{{guest}}`, `{{time_range}}`,
    /// `{{duration_secs}}`, `{{hook}}`, `{{reasoning}}`, `{{transcript}}`.
    #[arg(
        long,
        env = "CLIP_SOCIAL_USER_PROMPT_PATH",
        default_value = "./prompts/clips/social_user.txt"
    )]
    pub clip_social_user_prompt_path: String,

    /// Disable per-clip social copy generation (saves ~10 LLM calls per episode).
    #[arg(long, env = "CLIP_SOCIAL_COPY_DISABLED", default_value_t = false)]
    pub clip_social_copy_disabled: bool,

    /// Source for overlay hook text: "llm" (default, from social copy LLM),
    /// "ranker" (truncated ranker hook), or "ab_test" (alternate per clip).
    #[arg(long, env = "CLIP_HOOK_SOURCE", default_value = "llm")]
    pub clip_hook_source: String,

    /// Which aspect ratios to render per clip. Comma-separated list of any of:
    /// `9x16` (Shorts/TikTok/Reels/Threads), `1x1` (LinkedIn/X feed),
    /// `16x9` (LinkedIn landscape / Bluesky / YouTube re-upload).
    #[arg(long, env = "CLIP_RENDER_FORMATS", default_value = "9x16,1x1,16x9")]
    pub clip_render_formats: String,

    /// libx264 CRF for clip renders. **Lower = higher quality**. Defaults
    /// to `18` (visually lossless) — treats every clip as a master that
    /// the platform will re-encode further. Drop to `16` for absolute
    /// master grade (~2× file size), or `23` to match the old "balanced"
    /// output. Valid range: 0 (truly lossless, huge files) — 51 (worst).
    #[arg(long, env = "CLIP_VIDEO_CRF", default_value_t = 18)]
    pub clip_video_crf: u32,

    /// libx264 preset. Slower presets compress better at the same CRF.
    /// `slow` is the master default — ~1.5× the wall time of `medium`
    /// for a notable bitrate/size win at identical quality. Accepted:
    /// `ultrafast` / `superfast` / `veryfast` / `faster` / `fast` /
    /// `medium` / `slow` / `slower` / `veryslow` / `placebo`.
    #[arg(long, env = "CLIP_VIDEO_PRESET", default_value = "slow")]
    pub clip_video_preset: String,

    /// AAC audio bitrate in kbps for clip renders. `192` is the master
    /// default — clips are short so extra bits cost negligible disk.
    /// Drop to `128` for "balanced" output; `256` buys little for voice.
    #[arg(long, env = "CLIP_AUDIO_BITRATE_KBPS", default_value_t = 192)]
    pub clip_audio_bitrate_kbps: u32,

    // ── Clip duration & detection ───────────────────────────────────────

    /// Minimum candidate clip duration in seconds. Defaults to **10s** —
    /// short-form sweet spot. Bump to 25+ for monologue/explainer formats.
    /// Floored at 5s by the candidate generator to keep transcription
    /// + caption work usable.
    #[arg(long, env = "CLIP_MIN_SECS", default_value_t = 10.0)]
    pub clip_min_secs: f64,

    /// Maximum candidate clip duration in seconds. Defaults to **30s** —
    /// keeps clips inside the TikTok/IG Reels native short-form window.
    /// Raise to 60-90s if you want longer YouTube Shorts-friendly clips.
    #[arg(long, env = "CLIP_MAX_SECS", default_value_t = 30.0)]
    pub clip_max_secs: f64,

    /// Target candidate duration the generator aims for. The LLM ranker is
    /// told to refine windows around this length; if you set it outside
    /// `[CLIP_MIN_SECS, CLIP_MAX_SECS]` it's clamped. Defaults to **20s**.
    #[arg(long, env = "CLIP_TARGET_SECS", default_value_t = 20.0)]
    pub clip_target_secs: f64,

    /// Stride between candidate proposals in seconds. Smaller stride =
    /// denser candidate grid (more LLM tokens; higher recall). Default 15s
    /// balances coverage against ranker cost for 10-30s clips.
    #[arg(long, env = "CLIP_STRIDE_SECS", default_value_t = 15.0)]
    pub clip_stride_secs: f64,

    /// Minimum word count required for a candidate window to survive
    /// filtering. Silent or near-silent spans are useless to the ranker.
    /// Default 15 (lower than the 25s-clip default of 30 because shorter
    /// clips have fewer words).
    #[arg(long, env = "CLIP_MIN_WORDS", default_value_t = 15)]
    pub clip_min_words: usize,

    // ── Caption typography ──────────────────────────────────────────────

    /// Caption font point size for **9:16 vertical** renders. When unset
    /// (0), uses the per-aspect default baked into [`crate::captions`].
    /// Per-show captions.json still wins over this env knob.
    #[arg(long, env = "CAPTION_FONT_SIZE_VERTICAL", default_value_t = 0)]
    pub caption_font_size_vertical: u32,

    /// Caption font point size for **1:1 square** renders. 0 = default.
    #[arg(long, env = "CAPTION_FONT_SIZE_SQUARE", default_value_t = 0)]
    pub caption_font_size_square: u32,

    /// Caption font point size for **16:9 landscape** renders. 0 = default.
    #[arg(long, env = "CAPTION_FONT_SIZE_LANDSCAPE", default_value_t = 0)]
    pub caption_font_size_landscape: u32,

    /// Hook-overlay font size for **9:16 vertical** renders. 0 = default.
    #[arg(long, env = "CAPTION_OVERLAY_FONT_SIZE_VERTICAL", default_value_t = 0)]
    pub caption_overlay_font_size_vertical: u32,

    /// Hook-overlay font size for **1:1 square** renders. 0 = default.
    #[arg(long, env = "CAPTION_OVERLAY_FONT_SIZE_SQUARE", default_value_t = 0)]
    pub caption_overlay_font_size_square: u32,

    /// Hook-overlay font size for **16:9 landscape** renders. 0 = default.
    #[arg(long, env = "CAPTION_OVERLAY_FONT_SIZE_LANDSCAPE", default_value_t = 0)]
    pub caption_overlay_font_size_landscape: u32,

    /// Master switch for burning karaoke captions into the rendered video.
    /// Off = clean speakerphones with no overlay (good for explainers that
    /// will get manual captions on the platform side).
    #[arg(long, env = "CAPTION_BURN_ENABLED", default_value_t = true)]
    pub caption_burn_enabled: bool,

    /// Master switch for the hook overlay (top-of-frame title text).
    /// Off = pure captions, no hook headline.
    #[arg(long, env = "CAPTION_HOOK_OVERLAY_ENABLED", default_value_t = true)]
    pub caption_hook_overlay_enabled: bool,

    // ── Ranker audience targeting ────────────────────────────────────────

    /// Audience target injected into the ranker prompt — drives whether the
    /// LLM optimizes for new-viewer accessibility or existing-fan reward.
    /// Accepted: `broad` / `core` / `growth`. Default `broad`.
    ///   - `broad`  — pulling new viewers; penalizes inside-baseball and
    ///                guest-specific references.
    ///   - `core`   — rewarding existing fans; relaxes the self-contained
    ///                axis so inside jokes + deep cuts score higher.
    ///   - `growth` — middle ground; hook broadly accessible, payoff can
    ///                reward the niche. Best for accounts trying to grow.
    #[arg(long, env = "CLIP_AUDIENCE_MODE", default_value = "broad")]
    pub clip_audience_mode: String,

    // ── STT hallucination guard strictness ──────────────────────────────

    /// Hallucination-guard strictness for whisper transcripts.
    /// - `lax`: only foreign-script + extreme repetition detected
    /// - `default`: balanced (recommended)
    /// - `strict`: aggressive — also drops low-word-density chunks &
    ///   anything VAD flags as silence-dominant
    #[arg(long, env = "STT_HALLUCINATION_GUARD", default_value = "default")]
    pub stt_hallucination_guard: String,

    /// Which platforms to post to. Comma-separated, any of: `youtube`, `bluesky`,
    /// `instagram`, `ayrshare`. Default empty (no posting). Posting also requires
    /// `POST_DRY_RUN=false` to actually send — both knobs are opt-in for safety.
    #[arg(long, env = "POST_ENABLED_PLATFORMS", default_value = "")]
    pub post_enabled_platforms: String,

    /// Safety net. When true (default), posting code constructs platforms and
    /// resolves credentials but does NOT actually POST — every result is
    /// `DryRun`. Flip to `false` to enable real posting.
    #[arg(long, env = "POST_DRY_RUN", default_value_t = true)]
    pub post_dry_run: bool,

    /// YouTube Shorts privacy on upload: `unlisted` (default — link-only),
    /// `private` (only you), or `public`.
    #[arg(long, env = "YOUTUBE_PRIVACY_STATUS", default_value = "unlisted")]
    pub youtube_privacy_status: String,

    /// YouTube category id. Common values: `22` (People & Blogs), `24`
    /// (Entertainment), `23` (Comedy), `17` (Sports).
    #[arg(long, env = "YOUTUBE_CATEGORY_ID", default_value = "24")]
    pub youtube_category_id: String,

    /// Bluesky handle (e.g. `you.bsky.social` or your custom domain).
    #[arg(long, env = "BLUESKY_HANDLE")]
    pub bluesky_handle: Option<String>,

    /// Bluesky app password (`xxxx-xxxx-xxxx-xxxx`). Generate at
    /// https://bsky.app/settings/app-passwords. NOT your main login password.
    #[arg(long, env = "BLUESKY_APP_PASSWORD")]
    pub bluesky_app_password: Option<String>,

    /// Bluesky PDS (Personal Data Server) URL. Default: bsky.social. Override
    /// only if you self-host an ATProto PDS.
    #[arg(long, env = "BLUESKY_PDS_URL", default_value = "https://bsky.social")]
    pub bluesky_pds_url: String,

    /// Bluesky video service URL. Default: video.bsky.app.
    #[arg(
        long,
        env = "BLUESKY_VIDEO_SERVICE_URL",
        default_value = "https://video.bsky.app"
    )]
    pub bluesky_video_service_url: String,

    /// Instagram Graph API long-lived access token. Required when
    /// `POST_ENABLED_PLATFORMS` includes `instagram`.
    #[arg(long, env = "INSTAGRAM_ACCESS_TOKEN")]
    pub instagram_access_token: Option<String>,

    /// Instagram Business/Creator account user ID (numeric). Required alongside
    /// `INSTAGRAM_ACCESS_TOKEN`.
    #[arg(long, env = "INSTAGRAM_USER_ID")]
    pub instagram_user_id: Option<String>,

    /// Ayrshare API key. Required when `POST_ENABLED_PLATFORMS` includes
    /// `ayrshare`. Obtain from https://app.ayrshare.com.
    #[arg(long, env = "AYRSHARE_API_KEY")]
    pub ayrshare_api_key: Option<String>,

    /// Comma-separated list of platforms Ayrshare should post to.
    /// Typical values: `tiktok`, `instagram`. These are Ayrshare platform
    /// names, not autoseo platform names.
    #[arg(long, env = "AYRSHARE_PLATFORMS", default_value = "tiktok,instagram")]
    pub ayrshare_platforms: String,

    // --- Browser-backed posting (android-agent sidecar) ---
    /// Master switch for talking to the android-agent browser worker at all.
    /// When false (default), autoseo never constructs `Platform::Browser`
    /// instances and the dashboard hides browser-posting UI. Flip to true to
    /// enable the path; `POSTING_BACKEND` below then determines fan-out.
    #[arg(long, env = "BROWSER_POSTING_ENABLED", default_value_t = false)]
    pub browser_posting_enabled: bool,

    /// Which posting backend to use when `BROWSER_POSTING_ENABLED=true`.
    /// `api` (default) uses the existing HTTP API posters
    /// (YouTube/Bluesky/Instagram-Graph/Ayrshare). `browser` routes all
    /// enabled platforms through the android-agent worker. `mixed`
    /// constructs both — useful while migrating one platform at a time.
    /// Ignored when `BROWSER_POSTING_ENABLED=false`.
    #[arg(long, env = "POSTING_BACKEND", default_value = "api")]
    pub posting_backend: String,

    /// HTTP base URL for the browser_worker sidecar. In docker compose this is
    /// `http://browser_worker:8090`; for local dev override to `http://localhost:8090`.
    #[arg(
        long,
        env = "BROWSER_WORKER_URL",
        default_value = "http://localhost:8090"
    )]
    pub browser_worker_url: String,

    /// All known browser-backed accounts. Comma-separated `platform:account_id`
    /// pairs. Example: `x:tris_main,x:tris_alt,linkedin:tris_pro`. Each pair
    /// becomes its own `Platform::Browser` entry; the worker maintains a
    /// persistent profile per pair on disk.
    #[arg(long, env = "BROWSER_ACCOUNTS", default_value = "")]
    pub browser_accounts: String,

    /// Subset of `BROWSER_ACCOUNTS` used by the *automatic* post path
    /// (one primary per platform). Manual dashboard posts can target any
    /// account. Example: `x:tris_main,linkedin:tris_pro`. If empty, the first
    /// account per platform in `BROWSER_ACCOUNTS` is treated as primary.
    #[arg(long, env = "BROWSER_PRIMARY_ACCOUNTS", default_value = "")]
    pub browser_primary_accounts: String,

    /// Default daily posts-per-account cap when no per-platform override is set.
    /// Overrides via `BROWSER_POST_DAILY_CAP_<PLATFORM_UPPER>` (e.g.
    /// `BROWSER_POST_DAILY_CAP_X=3`).
    #[arg(long, env = "BROWSER_POST_DAILY_CAP_DEFAULT", default_value_t = 5)]
    pub browser_post_daily_cap_default: u32,

    /// Enable CloakBrowser's humanize mode (Bézier mouse paths, per-char
    /// typing, scroll easing). Adds latency in exchange for less mechanical
    /// behavior. Recommended on.
    #[arg(long, env = "BROWSER_HUMANIZE", default_value_t = true)]
    pub browser_humanize: bool,

    /// Hugging Face API key (used for HF Inference Providers — embeddings + VLM re-rank).
    /// When set, the clipper routes embeddings through HF instead of the local
    /// `fastembed` ONNX model. Required to enable VLM re-rank.
    #[arg(long, env = "HF_API_KEY")]
    pub hf_api_key: Option<String>,

    /// HF Inference Providers router URL (root, no trailing /v1).
    /// - VLM (chat) endpoint:    `{HF_ROUTER_URL}/v1/chat/completions`
    /// - Embedding endpoint:     `{HF_ROUTER_URL}/{HF_EMBED_PROVIDER}/models/{EMBED_MODEL}/pipeline/feature-extraction`
    #[arg(
        long,
        env = "HF_ROUTER_URL",
        default_value = "https://router.huggingface.co"
    )]
    pub hf_router_url: String,

    /// Which HF Inference Provider to route embedding requests to. Most English
    /// embedding models are on `hf-inference`; some larger Qwen variants are on
    /// `scaleway`.
    #[arg(long, env = "HF_EMBED_PROVIDER", default_value = "hf-inference")]
    pub hf_embed_provider: String,

    /// Embedding model id on HF (used when `HF_API_KEY` is set).
    /// `BAAI/bge-large-en-v1.5` is the workhorse English embedder — 1024-dim,
    /// reliably warm on hf-inference, MIT license. Swap to a multilingual or
    /// larger model if your content demands it.
    #[arg(long, env = "EMBED_MODEL", default_value = "BAAI/bge-large-en-v1.5")]
    pub embed_model: String,

    /// Enable VLM-based re-rank of top candidates after the LLM ranker.
    /// Sends frames + transcript to a vision-language model and blends the score.
    /// Requires `HF_API_KEY`.
    #[arg(long, env = "VLM_RERANK_ENABLED", default_value_t = false)]
    pub vlm_rerank_enabled: bool,

    /// Vision-language model id (any OpenAI-compatible chat endpoint that
    /// accepts `image_url` content blocks). Defaults to Qwen3-VL-8B via HF
    /// Inference / Novita; users on Groq can point at
    /// `meta-llama/llama-4-scout-17b-16e-instruct` instead by also setting
    /// VLM_BASE_URL + VLM_API_KEY.
    #[arg(
        long,
        env = "VLM_MODEL",
        default_value = "Qwen/Qwen3-VL-8B-Instruct:novita"
    )]
    pub vlm_model: String,

    /// Base URL for the standard VLM re-rank lane. When unset, falls back to
    /// `HF_ROUTER_URL` so existing HuggingFace-keyed setups keep working with
    /// no migration. Set this (plus `VLM_API_KEY`) to route the VLM lane
    /// through Groq (`https://api.groq.com/openai`), OpenRouter, or any other
    /// OpenAI-compatible host.
    #[arg(long, env = "VLM_BASE_URL")]
    pub vlm_base_url: Option<String>,

    /// API key for the standard VLM re-rank lane. When unset, falls back to
    /// `HF_API_KEY`. Pair with `VLM_BASE_URL` to consolidate VLM onto a
    /// non-HF provider (e.g. the same key you already use for Groq chat/STT).
    #[arg(long, env = "VLM_API_KEY")]
    pub vlm_api_key: Option<String>,

    /// How many candidates the LLM ranker passes to the VLM re-rank pass.
    /// VLM-re-ranked candidates are then truncated to `CLIP_TOP_K`.
    #[arg(long, env = "VLM_RERANK_TOP_K", default_value_t = 20)]
    pub vlm_rerank_top_k: usize,

    /// Number of frames sampled from each candidate for the VLM (evenly spaced
    /// across the clip window). 4-8 is the sweet spot.
    #[arg(long, env = "VLM_FRAMES_PER_CLIP", default_value_t = 5)]
    pub vlm_frames_per_clip: usize,

    /// Max dimension (longer edge) of frames sent to the VLM. Smaller = faster +
    /// cheaper; 512 keeps faces readable.
    #[arg(long, env = "VLM_FRAME_MAX_DIM", default_value_t = 512)]
    pub vlm_frame_max_dim: u32,

    /// Weight (0..=1) of the VLM score in the final blended score.
    /// final = (1 - w) * llm_score + w * vlm_score
    #[arg(long, env = "VLM_BLEND_WEIGHT", default_value_t = 0.5)]
    pub vlm_blend_weight: f64,

    /// Premium VLM model id (e.g. `qwen/qwen2.5-vl-72b-instruct`). When set,
    /// the top-K clips (after standard VLM re-rank) are re-scored through this
    /// higher-quality model via OpenRouter or any OpenAI-compatible endpoint.
    #[arg(long, env = "VLM_PREMIUM_MODEL")]
    pub vlm_premium_model: Option<String>,

    /// Base URL for the premium VLM API (OpenRouter or self-hosted).
    /// Defaults to OpenRouter's endpoint.
    #[arg(
        long,
        env = "VLM_PREMIUM_BASE_URL",
        default_value = "https://openrouter.ai/api"
    )]
    pub vlm_premium_base_url: String,

    /// API key for the premium VLM endpoint (e.g. OpenRouter API key).
    /// Falls back to `HF_API_KEY` if not set.
    #[arg(long, env = "VLM_PREMIUM_API_KEY")]
    pub vlm_premium_api_key: Option<String>,

    /// How many top clips go through the premium VLM re-rank (default 3).
    #[arg(long, env = "VLM_PREMIUM_TOP_K", default_value_t = 3)]
    pub vlm_premium_top_k: usize,

    /// Blend weight for the premium VLM score with the current blended score.
    /// final = (1 - w) * current + w * premium_vlm
    #[arg(long, env = "VLM_PREMIUM_BLEND_WEIGHT", default_value_t = 0.6)]
    pub vlm_premium_blend_weight: f64,

    /// Enable CTR history injection into the ranker prompt. When true, the
    /// ranker prompt includes top/worst historical clip performance data from
    /// the analytics table, helping the LLM learn from past successes/failures.
    /// Disable for A/B comparison runs.
    #[arg(long, env = "CTR_HISTORY_ENABLED", default_value_t = true)]
    pub ctr_history_enabled: bool,

    /// VAD backend: "silero" (Silero VAD ONNX, default) or "ffmpeg" (silencedetect fallback).
    /// If silero is selected but the model file is missing, falls back to ffmpeg automatically.
    #[arg(long, env = "VAD_BACKEND", default_value = "silero")]
    pub vad_backend: String,

    /// Path to silero_vad.onnx model file. Downloaded automatically if missing.
    #[arg(
        long,
        env = "VAD_MODEL_PATH",
        default_value = "./models/silero_vad.onnx"
    )]
    pub vad_model_path: String,

    /// Silero VAD speech probability threshold (0.0–1.0). Frames above this are speech.
    #[arg(long, env = "VAD_THRESHOLD", default_value_t = 0.5)]
    pub vad_threshold: f64,

    /// If set, writes debug dumps (raw RFC822 + extracted URLs) for messages we can't parse.
    #[arg(long, env = "DUMP_DIR")]
    pub dump_dir: Option<String>,

    /// Path to ffmpeg binary.
    #[arg(long, env = "FFMPEG", default_value = "ffmpeg")]
    pub ffmpeg: String,

    /// Path to ffprobe binary.
    #[arg(long, env = "FFPROBE", default_value = "ffprobe")]
    pub ffprobe: String,

    /// Base audio chunk duration in seconds (used if auto chunking is disabled or heuristics fail).
    #[arg(long, env = "AUDIO_CHUNK_SECS", default_value_t = 900)]
    pub audio_chunk_secs: u64,

    /// Enable automatic chunk sizing using heuristics tied to STT concurrency.
    #[arg(long, env = "AUTO_CHUNKING", default_value_t = true)]
    pub auto_chunking: bool,

    /// Target factor for chunk count (e.g. concurrency * factor).
    #[arg(long, env = "AUTO_CHUNK_TARGET_FACTOR", default_value_t = 2)]
    pub auto_chunk_target_factor: usize,

    /// If >0, override the chunk target calculation (e.g., 60 chunks divides
    /// duration by 60). Set 0 to fall back to `concurrency * target_factor`.
    /// 60 is tuned for Groq's free-tier whisper-large-v3-turbo: a 1hr episode
    /// becomes 60 × 60-second chunks → ~3.5 min wall time at 18 RPM.
    #[arg(long, env = "AUTO_CHUNK_TARGET_CHUNKS", default_value_t = 60)]
    pub auto_chunk_target_chunks: usize,

    /// Minimum chunk length enforced by auto chunking (seconds). Going below
    /// ~30s for word-timestamp STT wastes request budget without improving
    /// boundary accuracy on a per-RPM-limited provider.
    #[arg(long, env = "AUTO_CHUNK_MIN_SECS", default_value_t = 30)]
    pub auto_chunk_min_secs: u64,

    /// Maximum chunk length enforced by auto chunking (seconds).
    #[arg(long, env = "AUTO_CHUNK_MAX_SECS", default_value_t = 1800)]
    pub auto_chunk_max_secs: u64,

    /// STT backend. `api` sends audio to the OpenAI-compatible STT endpoint
    /// (Groq, OpenAI, etc.). `local` uses whisper.cpp via whisper-rs (requires
    /// the `local-stt` cargo feature). Default: `api`.
    #[arg(long, env = "STT_BACKEND", default_value = "api")]
    pub stt_backend: String,

    /// Path to a GGML whisper model file (e.g. `ggml-large-v3-turbo.bin`).
    /// When `STT_BACKEND=local`, the model is loaded from this path. If it
    /// does not exist the pipeline errors with download instructions.
    /// Default: `{WORK_DIR}/models/whisper/ggml-large-v3-turbo.bin`.
    #[arg(long, env = "WHISPER_MODEL_PATH")]
    pub whisper_model_path: Option<String>,

    /// Maximum number of concurrent STT requests. Tuned for Groq free tier
    /// (20 RPM cap) — pushing above 8 just queues behind the RPM gate and
    /// inflates 429s without throughput gain. Users on Dev/Pro tiers should
    /// raise this (e.g. 32) plus `STT_RPM_LIMIT` together.
    #[arg(long, env = "STT_CONCURRENCY", default_value_t = 8)]
    pub stt_concurrency: usize,

    /// Maximum number of concurrent ffmpeg render processes. **`0` (default) =
    /// auto** — detect logical cores via `std::thread::available_parallelism()`
    /// and pick `cores / 4` clamped to `[1, 8]`. Each ffmpeg `-preset medium`
    /// encode pegs ~3-4 cores, so this scales to the box: 1 on a 4-core
    /// laptop, 4 on a 16-core workstation, 8 on a 32+-core server.
    ///
    /// Set an explicit non-zero value to pin (e.g. `RENDER_CONCURRENCY=1` to
    /// preserve the old sequential behavior, or `RENDER_CONCURRENCY=12` if
    /// your disk can sustain more than the auto cap).
    #[arg(long, env = "RENDER_CONCURRENCY", default_value_t = 0)]
    pub render_concurrency: usize,

    /// STT requests-per-minute cap (0 disables). 18 leaves headroom under
    /// Groq free tier's 20 RPM for `whisper-large-v3-turbo`; set 0 if your
    /// provider is unmetered, raise to ~95 on Groq Dev tier (100 RPM).
    #[arg(long, env = "STT_RPM_LIMIT", default_value_t = 18)]
    pub stt_rpm_limit: u32,

    /// Number of thumbnail timestamps to request from the LLM.
    #[arg(long, env = "THUMBNAIL_SLOTS", default_value_t = 5)]
    pub thumbnail_slots: usize,

    /// When generating thumbnails, capture this many seconds around the center.
    #[arg(long, env = "THUMBNAIL_WINDOW_SECS", default_value_t = 6)]
    pub thumbnail_window_secs: u64,

    /// Total number of thumbnail images to generate.
    #[arg(long, env = "THUMBNAIL_COUNT", default_value_t = 10)]
    pub thumbnail_count: usize,

    /// Max number of concurrent ffmpeg screenshot processes for thumbnails.
    /// Keep this small to avoid disk/seek thrash.
    #[arg(long, env = "THUMBNAIL_FFMPEG_CONCURRENCY", default_value_t = 4)]
    pub thumbnail_ffmpeg_concurrency: usize,

    /// Maximum thumbnail height in pixels (preserve aspect ratio). Set 0 to disable scaling.
    /// Lower values are faster to generate and smaller to email.
    #[arg(long, env = "THUMBNAIL_MAX_HEIGHT", default_value_t = 1080)]
    pub thumbnail_max_height: u32,

    /// OpenAI-compatible base URL (no trailing slash), e.g. https://api.openai.com
    #[arg(
        long,
        env = "OPENAI_BASE_URL",
        default_value = "https://api.openai.com"
    )]
    pub openai_base_url: String,

    /// OpenAI API key.
    /// Required unless running with `--dry-run`.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub openai_api_key: Option<String>,

    /// Chat model for SEO + thumbnail selection.
    #[arg(
        long,
        env = "OPENAI_CHAT_MODEL",
        default_value = "gpt-5.2-pro-2025-12-11"
    )]
    pub openai_chat_model: String,

    /// Audio transcription model.
    #[arg(long, env = "OPENAI_STT_MODEL", default_value = "whisper-1")]
    pub openai_stt_model: String,

    /// Path to the SEO system prompt text file.
    #[arg(
        long,
        env = "SEO_SYSTEM_PROMPT_PATH",
        default_value = "./prompts/seo_system.txt"
    )]
    pub seo_system_prompt_path: String,

    /// Path to the SEO user prompt template text file. Use {{transcript}} placeholder.
    #[arg(
        long,
        env = "SEO_USER_PROMPT_PATH",
        default_value = "./prompts/seo_user.txt"
    )]
    pub seo_user_prompt_path: String,

    /// Number of distinct SEO variants to generate per video.
    #[arg(long, env = "SEO_VARIANTS", default_value_t = 3)]
    pub seo_variants: usize,

    /// Path to the SEO variants prompt file.
    /// Suggested format: variant blocks separated by a line containing only `---`.
    #[arg(
        long,
        env = "SEO_VARIANTS_PROMPT_PATH",
        default_value = "./prompts/seo_variants.txt"
    )]
    pub seo_variants_prompt_path: String,

    /// Path to the thumbnail-selection system prompt text file.
    #[arg(
        long,
        env = "THUMBNAIL_SYSTEM_PROMPT_PATH",
        default_value = "./prompts/thumbnail_system.txt"
    )]
    pub thumbnail_system_prompt_path: String,

    /// Path to the thumbnail-selection user prompt template text file.
    /// Use {{count}} and {{minutes}} placeholders.
    #[arg(
        long,
        env = "THUMBNAIL_USER_PROMPT_PATH",
        default_value = "./prompts/thumbnail_user.txt"
    )]
    pub thumbnail_user_prompt_path: String,

    /// Google OAuth client id. Required only when a code path actually talks to
    /// Google (Gmail polling, Drive download, or email digest delivery). The
    /// clipper's `DIGEST_MODE=file` path works without it.
    #[arg(long, env = "GOOGLE_CLIENT_ID")]
    pub google_client_id: Option<String>,

    /// Google OAuth client secret. Required alongside `google_client_id`.
    #[arg(long, env = "GOOGLE_CLIENT_SECRET")]
    pub google_client_secret: Option<String>,

    /// Google OAuth refresh token (from OAuth Playground). Required alongside
    /// `google_client_id`.
    #[arg(long, env = "GOOGLE_REFRESH_TOKEN")]
    pub google_refresh_token: Option<String>,

    /// How the clipper delivers its summary: `file` writes `digest.md` to the
    /// clips dir on disk (no external service needed); `email` sends via Gmail
    /// (requires Google credentials + `RESULT_TO`); `both` does both. Defaults
    /// to `file` so the clipper has no required external dependencies beyond
    /// `OPENAI_API_KEY`.
    #[arg(long, env = "DIGEST_MODE", default_value = "file")]
    pub digest_mode: String,

    /// If set, run the pipeline on a local media file (e.g. .mp4 or .wav) and send the result email.
    /// This bypasses Gmail/Drive ingest and the dedupe list.
    #[arg(long, env = "LOCAL_VIDEO_PATH")]
    pub local_video_path: Option<String>,

    /// Port for the API server (MODE=server).
    #[arg(long, env = "API_PORT", default_value_t = 8080)]
    pub api_port: u16,

    /// Comma-separated CORS allowed origins for the API server.
    /// Default allows the Vite dev server.
    #[arg(
        long,
        env = "API_CORS_ORIGINS",
        default_value = "http://localhost:5173"
    )]
    pub api_cors_origins: String,

    /// Start the API server instead of the pipeline. Can also be set via
    /// `MODE=server` or `--serve-api`.
    #[arg(long, env = "SERVE_API", default_value_t = false)]
    pub serve_api: bool,

    /// Enable per-show loudness persistence. When true (default), the clipper
    /// measures integrated LUFS on the first episode of a show, stores it in
    /// the database, and reuses it for subsequent episodes so all clips from the
    /// same show have consistent audio levels. Falls back to platform defaults
    /// when no show history exists or the show is unknown.
    #[arg(long, env = "LOUDNESS_PER_SHOW", default_value_t = true)]
    pub loudness_per_show: bool,

    /// Enable DeepFilterNet3 speech enhancement as a pre-processing stage.
    /// When enabled, extracted audio is denoised before chunking/STT and
    /// the enhanced audio is used for final clip rendering.
    /// Requires the `enhance` cargo feature.
    #[arg(long, env = "ENHANCE_AUDIO", default_value_t = false)]
    pub enhance_audio: bool,

    /// Path to the DeepFilterNet3 model tar.gz file.
    /// Downloaded automatically from GitHub releases if missing.
    #[arg(
        long,
        env = "ENHANCE_MODEL_PATH",
        default_value = "./models/DeepFilterNet3.tar.gz"
    )]
    pub enhance_model_path: String,

    /// Enable AST (Audio Spectrogram Transformer) audio-event detection.
    /// When enabled, classifies audio windows for laughter, applause, music, and
    /// speech as a tier-2 ranking signal. Requires the ONNX model file.
    #[arg(long, env = "AST_ENABLED", default_value_t = false)]
    pub ast_enabled: bool,

    /// URL to download the AST ONNX model from on first use.
    /// The model is saved to `{WORK_DIR}/models/ast/model.onnx`.
    #[arg(
        long,
        env = "AST_MODEL_URL",
        // The original MIT/ast-finetuned-audioset repo removed its onnx/
        // subfolder; this onnx-community export is the actively-maintained
        // mirror with the same model weights.
        default_value = "https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX/resolve/main/onnx/model.onnx"
    )]
    pub ast_model_url: String,

    /// Master switch for active-speaker detection (ASD). When true (default),
    /// the clipper runs face detection + landmark-based speaker selection per
    /// clip and uses the resulting trajectory to pan the 9:16 crop. When false,
    /// every clip falls back to a static centered crop. The legacy
    /// `SCRFD_ENABLED` flag is honored for backwards compat but the new
    /// recommended path is `ASD_ENABLED=true` + `FACE_DETECTOR=yunet`.
    #[arg(long, env = "ASD_ENABLED", default_value_t = true)]
    pub asd_enabled: bool,

    /// Which face detector implementation drives ASD. `yunet` (default) — the
    /// OpenCV Zoo YuNet ONNX, ~340 KB, no auth gate. `scrfd` — legacy SCRFD
    /// path; mirrored models often have output-shape drift and are kept
    /// behind an opt-in for users who source matching weights.
    #[arg(long, env = "FACE_DETECTOR", default_value = "yunet")]
    pub face_detector: String,

    /// URL to download the YuNet ONNX model from on first use. Saved to
    /// `{WORK_DIR}/models/yunet/face_detection_yunet_2023mar.onnx`.
    #[arg(
        long,
        env = "YUNET_MODEL_URL",
        default_value = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )]
    pub yunet_model_url: String,

    /// YuNet input resolution (square). 320 keeps inference cheap on CPU while
    /// retaining usable accuracy for podcast-style framing (one or two faces
    /// roughly center-frame). Bump to 640 for crowd scenes.
    #[arg(long, env = "YUNET_INPUT_SIZE", default_value_t = 320)]
    pub yunet_input_size: u32,

    /// YuNet confidence threshold (post sqrt(conf*iou) blending). 0.6 is the
    /// upstream OpenCV demo default; lower to 0.5 if real faces are being
    /// dropped, raise to 0.7 if background noise frames are leaking through.
    #[arg(long, env = "YUNET_CONF_THRESHOLD", default_value_t = 0.6)]
    pub yunet_conf_threshold: f32,

    /// YuNet NMS IoU threshold. Overlapping boxes above this are suppressed.
    #[arg(long, env = "YUNET_NMS_THRESHOLD", default_value_t = 0.3)]
    pub yunet_nms_threshold: f32,

    /// Enable SCRFD face detection for active-speaker crop. Requires the ONNX
    /// model file; auto-downloads on first use if `SCRFD_MODEL_URL` is set.
    ///
    /// Deprecated: prefer `ASD_ENABLED=true` + `FACE_DETECTOR=scrfd` if you
    /// need the SCRFD lane. Kept for env-file backwards compat.
    #[arg(long, env = "SCRFD_ENABLED", default_value_t = false)]
    pub scrfd_enabled: bool,

    /// URL to download the SCRFD ONNX model from on first use.
    /// The model is saved to `{WORK_DIR}/models/scrfd/scrfd_10g_bnkps.onnx`.
    ///
    /// The original `deepinsight/scrfd_10g_bnkps` repo is gated/restructured;
    /// `cromsc/scrfd-10g` is a public mirror with the same weights.
    #[arg(
        long,
        env = "SCRFD_MODEL_URL",
        default_value = "https://huggingface.co/cromsc/scrfd-10g/resolve/main/scrfd_10g_bnkps.onnx"
    )]
    pub scrfd_model_url: String,

    /// SCRFD input resolution (square). Larger = more accurate but slower.
    /// 640 is the standard SCRFD inference size.
    #[arg(long, env = "SCRFD_INPUT_SIZE", default_value_t = 640)]
    pub scrfd_input_size: u32,

    /// SCRFD confidence threshold. Detections below this are discarded.
    #[arg(long, env = "SCRFD_CONF_THRESHOLD", default_value_t = 0.5)]
    pub scrfd_conf_threshold: f32,

    /// SCRFD NMS IoU threshold. Overlapping boxes above this are suppressed.
    #[arg(long, env = "SCRFD_NMS_THRESHOLD", default_value_t = 0.4)]
    pub scrfd_nms_threshold: f32,

    /// IoU threshold for the frame-to-frame face tracker.
    /// Higher = stricter matching (fewer identity switches but more lost tracks).
    #[arg(long, env = "SCRFD_TRACKER_IOU", default_value_t = 0.3)]
    pub scrfd_tracker_iou: f32,

    /// Enable the Google Trends poller. When true, fetches daily trending
    /// searches and stores them in the `trends` table for ranker context.
    #[arg(long, env = "GOOGLE_TRENDS_ENABLED", default_value_t = false)]
    pub google_trends_enabled: bool,

    /// Google Trends RSS feed base URL (without query params).
    #[arg(
        long,
        env = "GOOGLE_TRENDS_RSS_URL",
        default_value = "https://trends.google.com/trending/rss"
    )]
    pub google_trends_rss_url: String,

    /// ISO 3166-1 alpha-2 geo code for Google Trends (e.g. US, GB, CA).
    #[arg(long, env = "GOOGLE_TRENDS_GEO", default_value = "US")]
    pub google_trends_geo: String,

    /// Google Trends poll RPM limit. Google is aggressive about rate-limiting;
    /// keep this low (1-2) to avoid 429s.
    #[arg(long, env = "GOOGLE_TRENDS_RPM", default_value_t = 1)]
    pub google_trends_rpm: u32,

    /// How often to refresh Google Trends, in seconds. Default 86400 (daily).
    #[arg(long, env = "GOOGLE_TRENDS_REFRESH_SECS", default_value_t = 86400)]
    pub google_trends_refresh_secs: u64,

    /// Maximum number of trending topics to inject into the ranker prompt.
    #[arg(long, env = "RANKER_TRENDS_TOP_N", default_value_t = 10)]
    pub ranker_trends_top_n: usize,
    /// Enable veto-via-Gmail-reply polling. When true, the system polls for
    /// replies to digest emails containing `veto: clip_XX` directives and
    /// removes/unlists the matching posts on each platform.
    #[arg(long, env = "VETO_ENABLED", default_value_t = false)]
    pub veto_enabled: bool,

    /// Gmail search query for veto reply messages. Defaults to replies to
    /// autoseo CLIPPER digest emails.
    #[arg(
        long,
        env = "VETO_GMAIL_QUERY",
        default_value = "subject:\"CLIPPER\" label:inbox newer_than:7d"
    )]
    pub veto_gmail_query: String,
    /// Bind address for the API server. Defaults to 0.0.0.0 (all interfaces).
    #[arg(long, env = "API_BIND", default_value = "0.0.0.0")]
    pub api_bind: String,

    /// Path to the built dashboard frontend (dist/ folder). When MODE=server,
    /// static files from this directory are served at /* as a fallback behind /api/*.
    #[arg(long, env = "DASHBOARD_DIST", default_value = "./dashboard/dist")]
    pub dashboard_dist: String,

    /// Open the browser automatically when the server starts.
    #[arg(long, env = "OPEN_BROWSER", default_value_t = false)]
    pub open_browser: bool,

    /// Enable per-job API cost tracking. When true (default), the pipeline
    /// estimates costs from token usage across STT, chat, embedding, and VLM
    /// calls. The cost summary appears in the digest email/file and is
    /// persisted to `jobs.cost_cents` in the database.
    #[arg(long, env = "COST_TRACKING_ENABLED", default_value_t = true)]
    pub cost_tracking_enabled: bool,
}

impl Config {
    /// Resolve `render_concurrency`, honoring `0` as "auto". When auto, picks
    /// a value from [`crate::system_specs::auto_render_concurrency`] based on
    /// the host CPU count; otherwise returns the explicit setting (floored at
    /// 1 so a misconfigured `RENDER_CONCURRENCY=0` doesn't deadlock).
    pub fn effective_render_concurrency(&self) -> usize {
        if self.render_concurrency == 0 {
            let specs = crate::system_specs::SystemSpecs::detect();
            crate::system_specs::auto_render_concurrency(&specs)
        } else {
            self.render_concurrency.max(1)
        }
    }
}
