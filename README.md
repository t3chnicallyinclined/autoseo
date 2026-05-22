# autoseo

[![CI](https://github.com/t3chnicallyinclined/autoseo/actions/workflows/ci.yml/badge.svg)](https://github.com/t3chnicallyinclined/autoseo/actions/workflows/ci.yml)

A self-hosted clipper agent for long-form podcasts. Drop a video in (Drive share, local file, or URL), and autoseo transcribes it, ranks the highest-CTR moments, renders them as vertical/square/landscape clips with burned captions and a 1.5s hook overlay, writes per-platform social copy, optionally posts to YouTube Shorts / Bluesky / Instagram / TikTok, and emails a digest. The original SEO-email mode (per-variant YouTube SEO packages for the long-form upload) is still here and runs alongside.

Built as a single Rust binary. All ML runs in-process (ONNX Runtime, `fastembed`, optional `whisper-rs`, ffmpeg filters) — no Python sidecar.

## Modes

Selected via the `MODE` env var (or `--serve-api`):

| Mode | What it does |
|---|---|
| `seo-only` *(default)* | Original behavior: Gmail → Drive download → STT → SEO variants + thumbnails → email |
| `clipper` | Long-form video in → ranked vertical/square/landscape clips + per-platform social copy + optional posting → digest |
| `both` | Runs both pipelines per poll cycle |
| `server` *(or `--serve-api`)* | Axum API + dashboard frontend + background worker that picks up jobs created via the dashboard |

The clipper can run from Gmail/Drive polling, a local file (`LOCAL_VIDEO_PATH=...`), or a job submitted through the dashboard (file upload or `video_url` — yt-dlp pulls from YouTube/TikTok/Drive/Twitch/Vimeo/X).

## Pipeline (clipper mode)

```
video in → ffmpeg audio extract → (optional DeepFilterNet3 enhance)
  → Groq Whisper word-timestamps (or whisper.cpp offline)
  → parallel feature extraction:
       • Silero VAD (ONNX)        • PySceneDetect (ffmpeg)
       • RMS energy (ffmpeg)      • aubio F0 pitch
       • linguistic markers       • embedding novelty (fastembed or HF BGE)
       • AST audio events (optional)
       • current trends from GDELT / Reddit / Google Trends
  → dense candidate windows snapped to silence + shot bounds
  → LLM ranker (batched, feature-injected, CTR history + trend context)
  → optional VLM re-rank (Qwen3-VL via HF) + premium VLM (OpenRouter)
  → for each top-K clip:
       SCRFD face detect → rule-based active-speaker → One-Euro smoothed crop
       → ffmpeg cut + reframe (9×16 / 1×1 / 16×9) → loudnorm (per-show LUFS)
       → ASS karaoke captions + 1.5s hook overlay burn
       → per-platform social copy (LLM)
       → post to YouTube / Bluesky / Instagram / Ayrshare (TikTok+IG)
  → digest (file + optional Gmail) with cost summary
  → 24h/72h analytics pull → feedback into next ranker prompt
```

## Quick start (clipper, no Google needed)

```bash
export OPENAI_API_KEY="..."
cargo run -- --once \
  --mode clipper \
  --local-video-path /path/to/episode.mp4
# clips + manifest.json + digest.md land in ./work/<name>/<ts>/clips/
```

Optional add-ons:
- `HF_API_KEY=...` enables the BGE embeddings + VLM re-rank lane.
- `VLM_RERANK_ENABLED=true` turns on the Qwen3-VL re-ranker.
- `ENHANCE_AUDIO=true` (with the `enhance` cargo feature) adds DeepFilterNet3 pre-processing.
- `STT_BACKEND=local` + `--features local-stt` runs offline via whisper.cpp.

## Quick start (dashboard / API server)

```bash
cargo run -- --serve-api
# API on :8080, dashboard served from ./dashboard/dist if built
# config.json under WORK_DIR is hot-imported into env on startup
```

The dashboard frontend lives in its own repo: [t3chnicallyinclined/autoseo-dashboard](https://github.com/t3chnicallyinclined/autoseo-dashboard) — React 19 + Vite + shadcn/ui + Tailwind v4 + Recharts. Quick wire-up:

```bash
# Clone alongside autoseo, build, symlink dist into autoseo/dashboard/dist
git clone https://github.com/t3chnicallyinclined/autoseo-dashboard.git ../autoseo-dashboard
( cd ../autoseo-dashboard && npm install && npm run build )
mkdir -p dashboard && ln -snf ../../autoseo-dashboard/dist dashboard/dist
```

Full instructions (alternative layouts, sanity checks, what to do if you don't have the dashboard repo at all) live in [DEV.md § Dashboard wiring](DEV.md#dashboard-wiring-one-time-per-machine). The symlink path is gitignored.

Then `cargo run -- --serve-api` from this repo serves the API at `:8080` and the built dashboard at `/`. Override the location via `DASHBOARD_DIST=/abs/path/to/dist`. The Rust API is the source of truth — the dashboard repo also ships a Node/Express dev server (`npm run dev:api`), but it's a convenience for frontend-only iteration; production should always go through the Rust binary.

### Available endpoints
- `GET /api/health`
- `GET|PATCH /api/config` and `POST /api/config/test/{service}`
- `GET /api/jobs` (list, dashboard-shape) · `POST /api/jobs` (multipart upload or `{"video_url": "..."}`) · `GET /api/jobs/{id}` (detail + clip summary) · `POST /api/jobs/{id}/retry`
- `GET /api/clips`, `GET /api/clips/{id}`, `POST /api/clips/bulk`
- `POST /api/clips/{id}/approve|veto|post`
- `GET /api/pipeline/status`, `GET /api/episodes`, `GET /api/analytics`, `GET /api/cost`
- `GET /api/shows`, `GET /api/platforms`, `GET /api/trends`, `GET /api/agents`

A WebSocket at `/ws` on the same port (`:8080`) streams live pipeline events to the dashboard. The served `index.html` is patched with a `window.__AUTOSEO_WS_URL` global so the dashboard auto-picks the right origin (local or tunnel) without rebuilding.

## Quick start (legacy SEO-only)

You'll need Google OAuth (Gmail + Drive scopes) — see the [Google setup](#google-setup-oauth-playground-bootstrap) section below.

```bash
export RESULT_TO="you@example.com"
export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
export GOOGLE_REFRESH_TOKEN="..."
export OPENAI_API_KEY="..."
cargo run -- --once       # one poll cycle
cargo run                 # continuous poller
```

Dry-run (no OpenAI + no send, only lists Gmail + Drive metadata):

```bash
cargo run -- --once --dry-run
```

## What you need installed

- Rust toolchain (1.88+)
- `ffmpeg` and `ffprobe` on PATH
- `yt-dlp` on PATH (only if you submit jobs by `video_url` through the dashboard)
- One of: an OpenAI-compatible STT API (Groq for word timestamps), or compile with `--features local-stt` to use whisper.cpp

The Docker image bundles ffmpeg automatically.

## OpenAI-compatible setup

- `OPENAI_BASE_URL` (default: `https://api.openai.com`)
- `OPENAI_API_KEY`
- `OPENAI_CHAT_MODEL` (default: `gpt-5.2-pro-2025-12-11`)
- `OPENAI_STT_MODEL` (default: `whisper-1` — set to `whisper-large-v3-turbo` if you use Groq)

The pipeline uses:
- `POST /v1/audio/transcriptions` (chunked, optional word-level timestamps via Groq)
- `POST /v1/chat/completions` for ranker, SEO, thumbnails, social copy
- `POST /v1/responses` fallback for `gpt-5*` models
- `POST /v1/chat/completions` against any OpenAI-compatible endpoint (OpenRouter, Groq, vLLM, etc.) for the premium VLM lane

## Google setup (OAuth Playground bootstrap)

Only needed for `MODE=seo-only`, `DIGEST_MODE=email`, the Gmail/Drive polling clipper path, or the veto-reply path.

1. Create a Google Cloud project.
2. Enable the **Gmail API** and **Google Drive API**.
3. Configure the OAuth consent screen (Testing is fine) and add your Google account as a test user.
4. Create OAuth credentials (Desktop app or Web application). Copy the **Client ID** and **Client secret**.
5. Use the [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/) to mint a refresh token. Required scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
   - `https://www.googleapis.com/auth/drive.readonly`
   - `https://www.googleapis.com/auth/youtube.upload` *(only if posting Shorts via YouTube Data API)*

## Configuration

The full list of knobs is in [src/config.rs](src/config.rs). The most commonly-set ones:

### Core
| Env | Default | Notes |
|---|---|---|
| `MODE` | `seo-only` | `seo-only` \| `clipper` \| `both` \| `server` |
| `DIGEST_MODE` | `file` | `file` \| `email` \| `both`. File mode needs no Google creds. |
| `LOCAL_VIDEO_PATH` | — | Run a single file end-to-end and exit. |
| `WORK_DIR` | `./work` | Big-file scratch + SQLite DB + model cache. |
| `CLIPPER_DB` | `./work/clipper.db` | Jobs / clips / posts / analytics / trends. |
| `POLL_INTERVAL_SECS` | `60` | Gmail polling cadence. |

### Server / dashboard
| Env | Default | Notes |
|---|---|---|
| `API_BIND` | `0.0.0.0` | |
| `API_PORT` | `8080` | |
| `API_CORS_ORIGINS` | `http://localhost:5173` | Vite dev server. |
| `DASHBOARD_DIST` | `./dashboard/dist` | Built frontend; falls back to a help page if missing. |
| `OPEN_BROWSER` | `false` | Pop the dashboard on startup. |

### Clipper ranking + render
| Env | Default | Notes |
|---|---|---|
| `CLIP_TOP_K` | `10` | Clips per episode. |
| `CLIP_RENDER_FORMATS` | `9x16,1x1,16x9` | |
| `CLIP_HOOK_SOURCE` | `llm` | `llm` \| `ranker` \| `ab_test` |
| `CLIP_SOCIAL_COPY_DISABLED` | `false` | Skip per-platform copy LLM calls. |
| `CTR_HISTORY_ENABLED` | `true` | Inject past clip CTR into the ranker prompt. |
| `LOUDNESS_PER_SHOW` | `true` | Measure once per show, reuse across episodes. |
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Used when `HF_API_KEY` is set; otherwise local fastembed MiniLM-L6-v2. |

### Caption styling
Applied to every clip; per-show JSON at `${SHOWS_DIR}/{slug}/captions.json` overlays on top of these envs (see [src/captions.rs](src/captions.rs) `CaptionOverrides` for the full schema).

| Env | Default | Notes |
|---|---|---|
| `CAPTION_FONT_NAME` | `Montserrat` | Used for both burned captions and the 1.5s hook overlay. |
| `CAPTION_HIGHLIGHT_BGR` | `00FFFF` (yellow) | Karaoke active-word color (BGR hex). |
| `CAPTION_PRIMARY_BGR` | `FFFFFF` | Text color (BGR hex). |
| `CAPTION_OUTLINE_BGR` | `000000` | Outline color (BGR hex). |
| `CAPTION_DISABLE_KARAOKE` | `false` | Set `true` for flat per-phrase captions (no per-word highlight). |

### VLM re-rank lanes
| Env | Default | Notes |
|---|---|---|
| `HF_API_KEY` | — | Enables HF Inference Providers (embeddings + VLM). |
| `VLM_RERANK_ENABLED` | `false` | |
| `VLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | |
| `VLM_FRAMES_PER_CLIP` | `5` | |
| `VLM_BLEND_WEIGHT` | `0.5` | `final = (1-w)*llm + w*vlm` |
| `VLM_PREMIUM_MODEL` | — | Set to `qwen/qwen2.5-vl-72b-instruct` (etc.) for top-3 re-rank via OpenRouter. |
| `VLM_PREMIUM_API_KEY` | falls back to `HF_API_KEY` | |
| `VLM_PREMIUM_BLEND_WEIGHT` | `0.6` | |

### Audio / video stages
| Env | Default | Notes |
|---|---|---|
| `STT_BACKEND` | `api` | `api` \| `local` (needs `--features local-stt`). |
| `STT_CONCURRENCY` | `500` | |
| `STT_RPM_LIMIT` | `0` | `0` disables. |
| `VAD_BACKEND` | `silero` | `silero` (ONNX) \| `ffmpeg` (silencedetect fallback). |
| `VAD_MODEL_PATH` | `./models/silero_vad.onnx` | Auto-downloaded. |
| `ENHANCE_AUDIO` | `false` | DeepFilterNet3 pre-stage (needs `--features enhance`). |
| `AST_ENABLED` | `false` | AST audio-event scorer (laughter/applause/music). |
| `SCRFD_ENABLED` | `false` | Face detect for active-speaker crop. |

### Posting
| Env | Default | Notes |
|---|---|---|
| `POST_ENABLED_PLATFORMS` | (empty) | Comma-separated: `youtube`, `bluesky`, `instagram`, `ayrshare`. |
| `POST_DRY_RUN` | `true` | Set `false` to actually publish. Both knobs are opt-in for safety. |
| `YOUTUBE_PRIVACY_STATUS` | `unlisted` | |
| `YOUTUBE_CATEGORY_ID` | `24` | |
| `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD` | — | App password, not your main login. |
| `INSTAGRAM_ACCESS_TOKEN` / `INSTAGRAM_USER_ID` | — | Graph API long-lived token. |
| `AYRSHARE_API_KEY` / `AYRSHARE_PLATFORMS` | — | Shim for TikTok + Instagram. |
| `VETO_ENABLED` | `false` | Reply to a digest with `veto: clip_03` to unlist/remove. |

### Context / trends
| Env | Default |
|---|---|
| `GOOGLE_TRENDS_ENABLED` | `false` |
| `GOOGLE_TRENDS_GEO` | `US` |
| `GOOGLE_TRENDS_REFRESH_SECS` | `86400` |
| `RANKER_TRENDS_TOP_N` | `10` |

### R2 / S3 object storage (optional, for dashboard playback)
| Env | Notes |
|---|---|
| `R2_ENDPOINT` | e.g. `https://<accountid>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | |
| `R2_BUCKET` | |
| `R2_PUBLIC_BASE_URL` | Public read prefix that maps to the bucket. |
| `R2_REGION` | Default `auto`. |
| `R2_KEY_PREFIX` | Default `clipper`. |

### yt-dlp (URL ingest via dashboard)
- `YTDLP_COOKIES_BROWSER` (`firefox`/`chrome`/`chromium`/`edge`/`safari`/`brave`)
- `YTDLP_COOKIES_FILE`
- `YTDLP_USER_AGENT`

### Cost tracking
- `COST_TRACKING_ENABLED` (default `true`) — per-job cents in `jobs.cost_cents` + digest summary.

### Settings via dashboard
On startup, `${WORK_DIR}/config.json` is hot-imported into the process environment for any key not already set. The dashboard's settings page writes to this file via `PATCH /api/config`, so most knobs above can be flipped from the UI without touching `.env`.

## Docker

Long-lived clipper/server image:

```bash
docker build -t autoseo:release .
mkdir -p work

docker rm -f autoseo 2>/dev/null || true
docker run -d --rm --name autoseo \
   -v "$PWD/work:/work" \
   -p 8080:8080 -p 9823:9823 \
   --env-file .env \
   -e WORK_DIR=/work \
   -e CLIPPER_DB=/work/clipper.db \
   -e EMBED_MODEL_DIR=/work/models/fastembed \
   -e MODE=server \
   autoseo:release
```

Or via compose:

```bash
docker compose up -d
docker compose logs -f
```

## Notes

- The SQLite DB at `CLIPPER_DB` is the source of truth for dedupe/jobs/clips/posts; the legacy `processed_message_ids.txt` is imported once on startup, then ignored.
- Per-show prompt overrides live under `${SHOWS_DIR:-./prompts/shows}/{show_slug}/{seo_system,seo_user,seo_variants,thumbnail_system,thumbnail_user}.txt`. Show slug is inferred from the media filename + early transcript via an LLM call.
- The clip viewer (`tools/serve_clips.py`) renders `manifest.json` as an HTML grid — per-format tabs, per-platform copy buttons, post-status pills. Used as the standing test harness while iterating on the clipper.
- For posting, both `POST_ENABLED_PLATFORMS` *and* `POST_DRY_RUN=false` must be set. Defaults are intentionally inert.
- See [CHANGELOG.md](CHANGELOG.md) for the per-feature history and [PLAN.md](PLAN.md) for the milestone roadmap.
