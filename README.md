# autoseo

A Rust pipeline that processes long-form video/audio into SEO packages and short-form clips for social media.

**Three operating modes:**

| Mode | What it does |
|------|-------------|
| `MODE=seo-only` (default) | Polls Gmail for Google Drive share emails, transcribes media, generates YouTube SEO packages + thumbnails, emails results |
| `MODE=clipper` | Takes a local video, extracts viral-worthy clips, renders in multiple aspect ratios with burned captions, optionally posts to YouTube Shorts / Bluesky |
| `MODE=both` | Runs both pipelines |

## Requirements

- Rust toolchain (1.85+)
- `ffmpeg` and `ffprobe` on PATH

## Architecture

```
Gmail poll ──► Drive download ──► Audio extraction ──► STT (chunked) ──► Transcript
                                                                            │
                              ┌─────────────────────────────────────────────┘
                              ▼
                   ┌──── MODE=seo-only ────┐     ┌──── MODE=clipper ──────────────┐
                   │                       │     │                                │
                   │  LLM SEO generation   │     │  Feature extraction (parallel) │
                   │  Thumbnail selection  │     │  ├─ Shot detection (scdet)     │
                   │  Email delivery       │     │  ├─ VAD (Silero ONNX)          │
                   │                       │     │  └─ Prosody (RMS + F0)         │
                   └───────────────────────┘     │                                │
                                                 │  Candidate generation          │
                                                 │  ├─ Dense windowing (30-90s)   │
                                                 │  ├─ Embedding novelty scores   │
                                                 │  └─ Linguistic markers          │
                                                 │                                │
                                                 │  LLM ranking → top-K clips     │
                                                 │  Optional VLM re-rank          │
                                                 │  Social copy generation        │
                                                 │  Multi-format rendering        │
                                                 │  ├─ 9x16 (Shorts/Reels/TikTok)│
                                                 │  ├─ 1x1  (LinkedIn)            │
                                                 │  └─ 16x9 (Bluesky/YouTube)     │
                                                 │                                │
                                                 │  Platform posting (opt-in)     │
                                                 │  Digest output (file/email)    │
                                                 └────────────────────────────────┘
```

## Google setup (OAuth Playground bootstrap)

You need OAuth to read a private Gmail inbox and download private Drive files.

1. Create a Google Cloud project
2. Enable APIs:
   - Gmail API
   - Google Drive API
   - YouTube Data API v3 *(only if posting to YouTube Shorts)*
3. Configure OAuth consent screen (Testing is fine) and add your Google account as a test user.
4. Create OAuth credentials:
   - OAuth Client ID (type: **Web application** or **Desktop app**)
   - Copy the **Client ID** and **Client secret**.
5. Use **OAuth 2.0 Playground** to get a refresh token:
   - Click the gear icon and set **OAuth client** to your client id/secret
   - Authorize scopes:
     - `https://www.googleapis.com/auth/gmail.readonly`
     - `https://www.googleapis.com/auth/gmail.send`
     - `https://www.googleapis.com/auth/drive.readonly`
     - `https://www.googleapis.com/auth/youtube.upload` *(only if posting to YouTube Shorts)*
   - Exchange authorization code for tokens
   - Copy the **refresh_token**

## OpenAI-compatible setup

Set `OPENAI_BASE_URL` (default: `https://api.openai.com`) and `OPENAI_API_KEY`.

This project uses:
- `POST /v1/audio/transcriptions` with `response_format=verbose_json` (and `timestamp_granularities=word` for clipper mode)
- `POST /v1/chat/completions` (or `POST /v1/responses` for `gpt-5*`) for SEO / clip ranking / social copy
- `POST /v1/chat/completions` with `response_format={"type":"json_object"}` for thumbnail timestamp selection

## Configure

Copy `.env.example` to `.env` and fill in your credentials. See `.env.example` for the full list of variables with comments.

### Core variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_CLIENT_ID` | yes | — | OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | yes | — | OAuth client secret |
| `GOOGLE_REFRESH_TOKEN` | yes | — | Long-lived refresh token |
| `OPENAI_API_KEY` | yes* | — | OpenAI-compatible API key (*not needed with `--dry-run`) |
| `RESULT_TO` | seo-only | — | Email address for SEO results |
| `MODE` | no | `seo-only` | Pipeline mode: `seo-only`, `clipper`, `both` |

### MODE=seo-only (default)

Polls Gmail for Google Drive "Item shared with you" emails, downloads the shared media (video or audio), transcribes via an OpenAI-compatible API, asks an LLM for YouTube SEO packages, optionally generates thumbnail screenshots for videos, then sends the results back via Gmail API.

| Variable | Default | Description |
|----------|---------|-------------|
| `GMAIL_QUERY` | `from:drive-shares-dm-noreply@google.com subject:"Item shared with you" has:drive` | Gmail search query |
| `GMAIL_MAX_RESULTS` | `10` | Max messages per poll |
| `REQUIRE_VIDEO` | `false` | Skip audio-only files |
| `POLL_INTERVAL_SECS` | `60` | Seconds between polls |
| `RESULT_SUBJECT_PREFIX` | `[autoseo]` | Email subject prefix |
| `SEO_VARIANTS` | `3` | Number of SEO variant outputs |
| `SEO_SYSTEM_PROMPT_PATH` | `./prompts/seo_system.txt` | System prompt for SEO generation |
| `SEO_USER_PROMPT_PATH` | `./prompts/seo_user.txt` | User prompt (supports `{{media_name}}`, `{{show_name}}`, `{{hosts}}`, `{{guest}}`) |
| `SEO_VARIANTS_PROMPT_PATH` | `./prompts/seo_variants.txt` | Variants prompt |
| `THUMBNAIL_SLOTS` | `5` | LLM-selected timestamp slots |
| `THUMBNAIL_WINDOW_SECS` | `6` | Seconds around each timestamp |
| `THUMBNAIL_COUNT` | `10` | Total thumbnails to generate |
| `THUMBNAIL_MAX_HEIGHT` | `1080` | Max height; `0` for native resolution |
| `THUMBNAIL_FFMPEG_CONCURRENCY` | `4` | Parallel ffmpeg processes |
| `THUMBNAIL_SYSTEM_PROMPT_PATH` | `./prompts/thumbnail_system.txt` | Thumbnail system prompt |
| `THUMBNAIL_USER_PROMPT_PATH` | `./prompts/thumbnail_user.txt` | Thumbnail user prompt |

### MODE=clipper

Takes a video (via `LOCAL_VIDEO_PATH` or Gmail/Drive), extracts viral-worthy short clips, renders them in multiple aspect ratios with burned karaoke captions, generates per-platform social copy, and optionally posts to enabled platforms.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_VIDEO_PATH` | — | Bypass Gmail/Drive; process a local video file |
| `CLIP_TOP_K` | `10` | Number of top clips to render |
| `CLIP_RENDER_FORMATS` | `9x16,1x1,16x9` | Aspect ratios: `9x16` (Shorts/Reels/TikTok), `1x1` (LinkedIn), `16x9` (Bluesky) |
| `CLIP_RANKER_SYSTEM_PROMPT_PATH` | `./prompts/clips/ranker_system.txt` | Ranker system prompt |
| `CLIP_RANKER_USER_PROMPT_PATH` | `./prompts/clips/ranker_user.txt` | Ranker user prompt |
| `CLIP_SOCIAL_COPY_DISABLED` | `false` | Skip per-clip social copy generation |
| `CLIP_SOCIAL_SYSTEM_PROMPT_PATH` | `./prompts/clips/social_system.txt` | Social copy system prompt |
| `CLIP_SOCIAL_USER_PROMPT_PATH` | `./prompts/clips/social_user.txt` | Social copy user prompt |
| `DIGEST_MODE` | `file` | Output delivery: `file` (disk), `email` (Gmail), `both` |
| `CLIPPER_DB` | `./work/clipper.db` | SQLite DB for jobs/clips/posts |

### Embedding & VLM re-ranking (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL_DIR` | `./work/models/fastembed` | Cache for fastembed ONNX models (~90 MB) |
| `HF_API_KEY` | — | Hugging Face API key for embeddings + VLM |
| `HF_ROUTER_URL` | `https://router.huggingface.co` | HF Inference Providers router |
| `HF_EMBED_PROVIDER` | `hf-inference` | HF embedding provider (e.g. `scaleway`) |
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model on HuggingFace |
| `VLM_RERANK_ENABLED` | `false` | Enable VLM re-rank after LLM ranker |
| `VLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Vision-language model for re-ranking |
| `VLM_RERANK_TOP_K` | `20` | Candidates passed to VLM |
| `VLM_FRAMES_PER_CLIP` | `5` | Frames sampled per candidate (4-8 recommended) |
| `VLM_FRAME_MAX_DIM` | `512` | Max frame dimension for VLM |
| `VLM_BLEND_WEIGHT` | `0.5` | VLM weight: `final = (1-w)*llm + w*vlm` |

### Platform posting (opt-in)

Posting is **off by default**. Two safety gates must both be opened:

1. `POST_ENABLED_PLATFORMS` must list the platforms (empty by default)
2. `POST_DRY_RUN` must be set to `false` (defaults to `true`)

| Variable | Default | Description |
|----------|---------|-------------|
| `POST_ENABLED_PLATFORMS` | *(empty)* | Comma-separated: `youtube`, `bluesky` (aliases: `yt`, `shorts`, `bsky`) |
| `POST_DRY_RUN` | `true` | Safety net; set `false` to actually post |

**YouTube Shorts** — requires `youtube.upload` OAuth scope on your refresh token:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOUTUBE_PRIVACY_STATUS` | `unlisted` | `unlisted`, `private`, or `public` |
| `YOUTUBE_CATEGORY_ID` | `24` | YouTube category (24=Entertainment, 22=People & Blogs, 17=Sports) |

**Bluesky** — requires an app password from https://bsky.app/settings/app-passwords:

| Variable | Default | Description |
|----------|---------|-------------|
| `BLUESKY_HANDLE` | — | Your handle (e.g. `you.bsky.social`) |
| `BLUESKY_APP_PASSWORD` | — | App password (NOT your login password) |
| `BLUESKY_PDS_URL` | `https://bsky.social` | Personal Data Server URL |
| `BLUESKY_VIDEO_SERVICE_URL` | `https://video.bsky.app` | Video upload service |

**Posting result states:** `Posted` (published), `DryRun` (logged only), `Skipped` (disabled/missing creds), `Failed` (API error).

### Media processing & STT

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `./work` | Working directory for downloads/processing |
| `DEDUPE_FILE` | `./work/processed_message_ids.txt` | File-backed deduplication |
| `FFMPEG` | `ffmpeg` | Path to ffmpeg binary |
| `FFPROBE` | `ffprobe` | Path to ffprobe binary |
| `AUDIO_CHUNK_SECS` | `900` | Base chunk size for STT (15 min) |
| `AUTO_CHUNKING` | `true` | Auto-size chunks based on concurrency |
| `AUTO_CHUNK_TARGET_FACTOR` | `2` | Chunk count = concurrency x factor |
| `AUTO_CHUNK_TARGET_CHUNKS` | `400` | Override chunk count (if > 0) |
| `AUTO_CHUNK_MIN_SECS` | `10` | Minimum chunk duration |
| `AUTO_CHUNK_MAX_SECS` | `1800` | Maximum chunk duration (30 min) |
| `STT_CONCURRENCY` | `500` | Max concurrent STT requests |
| `STT_RPM_LIMIT` | `0` | Requests-per-minute cap (0 = unlimited) |
| `OPENAI_BASE_URL` | `https://api.openai.com` | OpenAI-compatible endpoint |
| `OPENAI_CHAT_MODEL` | `gpt-5.2-pro-2025-12-11` | Chat model |
| `OPENAI_STT_MODEL` | `whisper-1` | Speech-to-text model |

### Show configuration & debug

| Variable | Default | Description |
|----------|---------|-------------|
| `SHOWS_DIR` | `./prompts/shows` | Per-show prompt overrides (layout: `{shows_dir}/{show_slug}/*.txt`) |
| `DUMP_DIR` | — | Dump failed message parsing for debugging |

## Run

### SEO-only mode (default)

One poll cycle:

```bash
export RESULT_TO="you@example.com"
export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
export GOOGLE_REFRESH_TOKEN="..."
export OPENAI_API_KEY="..."

cargo run -- --once
```

Dry-run (no OpenAI, no email send — only needs Google credentials):

```bash
cargo run -- --once --dry-run
```

Continuous poller:

```bash
cargo run
```

### Clipper mode

Process a local video, render top 10 clips:

```bash
export MODE=clipper
export LOCAL_VIDEO_PATH="/path/to/episode.mp4"
export OPENAI_API_KEY="..."

cargo run -- --once
```

With platform posting enabled:

```bash
export MODE=clipper
export LOCAL_VIDEO_PATH="/path/to/episode.mp4"
export OPENAI_API_KEY="..."
export POST_ENABLED_PLATFORMS="youtube,bluesky"
export POST_DRY_RUN=false
export YOUTUBE_PRIVACY_STATUS=unlisted
export BLUESKY_HANDLE="you.bsky.social"
export BLUESKY_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"

cargo run -- --once
```

### Both modes

```bash
export MODE=both
export LOCAL_VIDEO_PATH="/path/to/episode.mp4"
export RESULT_TO="you@example.com"
# ... all credentials ...

cargo run -- --once
```

## Docker (long-running worker)

This repo includes a multi-stage `Dockerfile` that builds a release binary and ships `ffmpeg` in the runtime image.

Build:

```bash
docker build -t autoseo:release .
```

Run as a long-lived poller with persisted work + dedupe:

```bash
mkdir -p work

docker rm -f autoseo 2>/dev/null || true
docker run -d --rm --name autoseo \
   -v "$PWD/work:/work" \
   --env-file .env \
   -e WORK_DIR=/work \
   -e DEDUPE_FILE=/work/processed_message_ids.txt \
   autoseo:release
```

Run in clipper mode with a local video:

```bash
docker run --rm --name autoseo-clipper \
   -v "$PWD/work:/work" \
   -v "/path/to/video.mp4:/input/video.mp4:ro" \
   --env-file .env \
   -e MODE=clipper \
   -e LOCAL_VIDEO_PATH=/input/video.mp4 \
   -e WORK_DIR=/work \
   autoseo:release --once
```

Logs / ops:

```bash
docker logs -f --tail=200 autoseo
docker ps --filter name=autoseo
docker restart autoseo
```

## Notes

- Dedupe is file-backed but not a real job DB. If the process restarts, it won't reprocess messages already recorded in `DEDUPE_FILE`.
- The clipper pipeline uses SQLite (`CLIPPER_DB`) for job/clip/post tracking.
- The standard Google Drive share email contains a direct link like `https://drive.google.com/file/d/<fileId>/view?...` which is what we parse.
- For 1-3 hour videos, transcription uploads are done as small audio chunks (default 15 minutes, auto-sized when `AUTO_CHUNKING=true`).
- For audio-only files, the SEO pipeline runs transcript + SEO only (no thumbnails). Clipper mode requires video.
- Show context (show name/hosts/guest) is inferred from the media filename + early transcript, but only when explicitly present; otherwise the output stays generic.
- `prompts/seo_user.txt` can use injected placeholders: `{{media_name}}`, `{{show_name}}`, `{{hosts}}`, `{{guest}}`.
- Per-show prompt overrides live in `SHOWS_DIR/{show_slug}/` — any standard prompt file placed there overrides the global default.
