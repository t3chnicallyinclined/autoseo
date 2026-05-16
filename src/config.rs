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
    #[arg(long, env = "EMBED_MODEL_DIR", default_value = "./work/models/fastembed")]
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

    /// Which aspect ratios to render per clip. Comma-separated list of any of:
    /// `9x16` (Shorts/TikTok/Reels/Threads), `1x1` (LinkedIn/X feed),
    /// `16x9` (LinkedIn landscape / Bluesky / YouTube re-upload).
    #[arg(long, env = "CLIP_RENDER_FORMATS", default_value = "9x16,1x1,16x9")]
    pub clip_render_formats: String,

    /// Which platforms to post to. Comma-separated, any of: `youtube`, `bluesky`.
    /// Default empty (no posting). Posting also requires `POST_DRY_RUN=false`
    /// to actually send — both knobs are opt-in for safety.
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

    /// Hugging Face API key (used for HF Inference Providers — embeddings + VLM re-rank).
    /// When set, the clipper routes embeddings through HF instead of the local
    /// `fastembed` ONNX model. Required to enable VLM re-rank.
    #[arg(long, env = "HF_API_KEY")]
    pub hf_api_key: Option<String>,

    /// HF Inference Providers router URL (root, no trailing /v1).
    /// - VLM (chat) endpoint:    `{HF_ROUTER_URL}/v1/chat/completions`
    /// - Embedding endpoint:     `{HF_ROUTER_URL}/{HF_EMBED_PROVIDER}/models/{EMBED_MODEL}/pipeline/feature-extraction`
    #[arg(long, env = "HF_ROUTER_URL", default_value = "https://router.huggingface.co")]
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

    /// Vision-language model id on HF (for re-rank). Qwen3-VL-8B-Instruct is the
    /// current pick for long-video understanding at this size class.
    #[arg(long, env = "VLM_MODEL", default_value = "Qwen/Qwen3-VL-8B-Instruct")]
    pub vlm_model: String,

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

    /// If >0, override the chunk target calculation (e.g., 400 chunks divides duration by 400). Set 0 to disable.
    #[arg(long, env = "AUTO_CHUNK_TARGET_CHUNKS", default_value_t = 400)]
    pub auto_chunk_target_chunks: usize,

    /// Minimum chunk length enforced by auto chunking (seconds).
    #[arg(long, env = "AUTO_CHUNK_MIN_SECS", default_value_t = 10)]
    pub auto_chunk_min_secs: u64,

    /// Maximum chunk length enforced by auto chunking (seconds).
    #[arg(long, env = "AUTO_CHUNK_MAX_SECS", default_value_t = 1800)]
    pub auto_chunk_max_secs: u64,

    /// Maximum number of concurrent STT requests (limits Whisper RPM usage).
    #[arg(long, env = "STT_CONCURRENCY", default_value_t = 500)]
    pub stt_concurrency: usize,

    /// Optional STT requests-per-minute cap (0 disables). Useful to push near provider RPM without 429s.
    #[arg(long, env = "STT_RPM_LIMIT", default_value_t = 0)]
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
    #[arg(long, env = "API_CORS_ORIGINS", default_value = "http://localhost:5173")]
    pub api_cors_origins: String,

    /// Start the API server instead of the pipeline. Can also be set via
    /// `MODE=server` or `--serve-api`.
    #[arg(long, env = "SERVE_API", default_value_t = false)]
    pub serve_api: bool,
}
