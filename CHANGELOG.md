# Changelog

## 2026-05-21

### Clipper pipeline features (items 8–11 from the dashboard backlog)
- **#8 Active-speaker smart-crop**: new `src/asd_pipeline.rs` orchestrator (SCRFD → temporal-smoothed active-speaker election → One-Euro smoothed trajectory), wired into `src/render.rs` via a piecewise-linear x-expression in the ffmpeg crop filter for 9:16. Falls back to static center-crop when SCRFD is disabled or fails. New env: `SCRFD_ENABLED` (default false). Note: the current `cromsc/scrfd-10g` ONNX output shape doesn't match the inference assumptions in `src/face_detect.rs` and panics at runtime; tracked separately.
- **#9 AST audio events into ranker**: `audio_events` field now documented in `prompts/clips/ranker_user.txt` + scoring heuristic added to `ranker_system.txt`. The LLM ranker can use laughter / applause / music / speech signals to break ties and bump strong reactions.
- **#10 Caption overrides**: new `CaptionOverrides` struct in `src/captions.rs` with env vars (`CAPTION_FONT_NAME`, `CAPTION_HIGHLIGHT_BGR`, `CAPTION_PRIMARY_BGR`, `CAPTION_OUTLINE_BGR`, `CAPTION_DISABLE_KARAOKE`) and optional per-show JSON at `prompts/shows/{slug}/captions.json`. `FormatSpec` refactored to a `FormatAspect` enum.
- **#11 Premium VLM lane differentiation + A/B manifest**: rewrote the premium VLM prompt with 5 distinct editorial criteria (micro-expressions, body language, composition stability, lighting, hook-to-visual alignment) so it earns its cost over the standard lane. Added `llm_score` / `vlm_score` / `vlm_reasoning` / `vlm_premium_score` / `vlm_premium_reasoning` to `RankedClip`; manifest schema bumped to v3 with a `scores` block per clip.

### WebSocket multiplexed on main HTTP port
- New axum WS extractor at `/ws` (same port as `/api/*`), replacing the standalone `tokio-tungstenite` server. One `cloudflared tunnel --url http://localhost:8080` now covers both API and WS.
- `EventBus` added to `AppState` and threaded through `main.rs → worker → clipper`, publishing `job_update` / `job_complete` / `job_failed` events at every status transition. Wire schema in `src/events.rs` matches the dashboard's `WSMessage` union (flat camelCase, no nested `data`).
- Served `index.html` is patched with a `window.__AUTOSEO_WS_URL` inject so the dashboard auto-picks ws:// vs wss:// from the page origin without rebuilding.
- Deleted `src/ws.rs` (dead from an earlier incomplete wiring attempt). Removed `WS_PORT` config field.

### Job management API (`src/api/jobs.rs`)
- `GET /api/jobs` (list, dashboard-shape mapping) — moved out of stubs.
- `GET /api/jobs/{id}` — detail with `clips` summary.
- `POST /api/jobs/{id}/retry` — flip `failed → pending`.
- `POST /api/jobs/{id}/cancel` — flip `pending → cancelled` (mid-flight cancel deferred; needs cooperative cancellation tokens).
- `POST /api/jobs/{id}/rerun` — clone the source row into a new pending job; original artifacts untouched.
- `DELETE /api/jobs/{id}[?purge=true]` — remove row (CASCADE removes clips / renders / posts via FK); optional disk purge of `work/clipper/<media>/` and `work/uploads/<id>/`.
- New `JobStatus::Cancelled` variant.

### Dashboard plumbing fixes
- Single `events::dashboard_view()` is the source of truth for the FSM → (status, stage, progress) collapse used by both the `/api/jobs` Job mapper and `/api/pipeline/status`. Fixes the previous Pipeline-card-vs-Jobs-card label mismatch where internal status `transcribed` was rendered as "Transcribing" (it actually means STT *done* and feature extraction is running).
- `GET /api/pipeline/status` (was a stub returning `[]`) now projects the most-recent job's status onto the 8 dashboard pipeline stages — the Pipeline Architecture card lights up live.
- `PUT /api/config` aliased to `PATCH` so the dashboard's `useUpdateConfig` mutation actually persists (previous behavior 405'd silently).
- `clipsGenerated` count now matches `CLIP_TOP_K` exactly: `insert_clip` moved to AFTER the post-VLM truncate so the `clips` table only holds clips that actually got rendered (was inserting `vlm_rerank_top_k` candidates and leaving ghost rows).

### Model defaults + HF auth fixes
- AST default URL → `onnx-community/ast-finetuned-audioset-…-ONNX` (the original MIT repo restructured its ONNX exports → 404).
- SCRFD default URL → `cromsc/scrfd-10g` (the original `deepinsight/scrfd_10g_bnkps` is now gated → 401).
- `VLM_MODEL` default → `Qwen/Qwen3-VL-8B-Instruct:novita` (the `:novita` suffix routes via Novita Inference Provider; the bare model returns `model_not_available` against `hf-inference`).
- Both `face_detect.rs` and `ast.rs` `download_model()` now send `Authorization: Bearer $HF_API_KEY` so future-gated models keep working.

### Docs
- New `docs/dashboard-mock-data-workstream.md` — full audit of all 15 dashboard pages with prioritized batched backlog for replacing mock data with live data.
- New `DEV.md` — runbook for the local dev stack (autoseo + dashboard + cloudflared tunnel) plus dashboard-as-test-harness rule.
- README rewrite — covers the four `MODE` values, dashboard wiring, full env reference (clipper render knobs, caption overrides, VLM re-rank, audio/video stages, posting, context/trends, R2/S3, yt-dlp ingest, cost tracking), full `/api/*` endpoint list.

### Tests
- 323 passing, 0 failing, 2 ignored. Up from 285.

## 2026-05-18
- Added DeepFilterNet3 speech enhancement pre-stage (`src/enhance.rs`). New envs: `ENHANCE_AUDIO` (default false), `ENHANCE_MODEL_PATH` (default `./models/DeepFilterNet3.tar.gz`). New cargo feature: `enhance`. When enabled, extracted audio is denoised at 48 kHz via DeepFilterNet3 before chunking/STT and clip rendering. Model is auto-downloaded from GitHub releases on first use. Graceful fallback: if the model fails to load or enhancement errors, the pipeline continues with the original audio unchanged. Enhanced audio is also passed to `render_clip_with_audio` so final clips use the denoised track. 4 tests covering disabled path, missing model fallback, 48kHz extraction, and file reuse.

## 2026-05-11
- Started clipper-agent extension on `feat/clipper` branch — turning autoseo into an autonomous clipper that produces N platform-specific short clips per episode. See [PLAN.md](PLAN.md) for the locked-in design.
- Added SQLite storage layer (`src/storage.rs`) backed by bundled `rusqlite`. Schema covers the full M1 surface: `jobs`, `clips`, `clip_renders`, `posts`, `analytics`, `trends`. WAL mode + foreign keys enabled. Idempotent migration via `PRAGMA user_version`.
- Migrated dedupe from flat `processed_message_ids.txt` to the SQLite `jobs` table. Legacy file is imported once on startup; thereafter the authoritative state lives in `CLIPPER_DB` (default `./work/clipper.db`). The legacy file is left on disk untouched.
- Removed `src/dedupe.rs`.
- Added pipeline mode toggle via `MODE` env: `seo-only` (default, current behavior), `clipper` (M1 work in progress), or `both`. Invalid values fail fast on startup.
- Added multi-show prompt override layer (`src/show_config.rs`): `PromptLoader` resolves `{SHOWS_DIR}/{slug}/{prompt}.txt` first and falls back to the global prompt path when no override exists. Includes a `slugify` helper for deriving stable directory keys from free-form show names. Not yet wired into the live SEO email path (lands in M1 slice 4); available for the clipper path.
- Existing SEO-email pipeline behavior is unchanged.
- **Architecture pivot:** dropped the Python sidecar from M1. Every M1 ML stage has a viable Rust path: Silero VAD via `ort`, embeddings via `fastembed`, shot detection via ffmpeg's `scenedetect` filter, prosody via `aubio` + ffmpeg `astats`, word timestamps via Groq's OpenAI-compatible API (`whisper-rs` as offline fallback). Python sidecar decision deferred to M3, only if Light-ASD active-speaker detection is required. PLAN.md updated.
- Added shot-boundary detector (`src/scene.rs`): wraps ffmpeg's `select='gt(scene,T)',showinfo` filter and parses `pts_time:` markers from stderr. Includes `snap_to_shot()` helper for aligning clip start/end times to nearest shot boundary within a drift budget. Pure ffmpeg, no new Cargo deps. 5 tests including end-to-end synthetic-video verification.
- Added silence/speech-window detector (`src/vad.rs`): wraps ffmpeg's `silencedetect` audio filter, pairs the `silence_start`/`silence_end` log lines into `SilenceWindow`s. Includes `invert_to_speech()` to derive `SpeechSegment`s for turn-density features, and `snap_to_silence_boundary()` for natural-pause clip cuts. Configurable noise threshold (dB) and minimum-duration parameters. Silero VAD via `ort` is deferred to M3 — the public API will swap behind the same surface. Pure ffmpeg, no new Cargo deps. 9 tests including end-to-end synthetic-audio verification.
- Added per-window RMS energy curve (`src/prosody.rs`): wraps ffmpeg's `astats=metadata=1:length=N,ametadata=print:file=-` to emit `RmsWindow` rows in dBFS. Includes `peak_in_range()` and `mean_in_range()` helpers for feature aggregation inside candidate windows. F0/pitch (via `aubio`) is deferred — the LLM ranker with linguistic markers + RMS + word density is sufficient for M1. Pure ffmpeg, no new Cargo deps. 4 tests including end-to-end synthetic-audio verification.
- Added embedding-based novelty scorer (`src/embed.rs`): wraps `fastembed` with `all-MiniLM-L6-v2` (384-dim, ONNX-runtime-backed). Provides `Embedder::embed()` for batched text embedding, `cosine_similarity` / `mean_vec` helpers, and `score_novelty()` which computes per-chunk cosine distance from the episode centroid and normalizes to `[0,1]`. New env: `EMBED_MODEL_DIR` (default `./work/models/fastembed`) — model auto-downloads on first run (~90 MB) and persists under WORK_DIR. New deps: `fastembed = "4"` (pulls `ort` transitively). 7 unit tests + 1 ignored end-to-end test (`cargo test -- --ignored embed_end_to_end`).
- Added word-level transcription support: extended `OpenAiClient` with `transcribe_words()` that requests `timestamp_granularities[]=word` (Groq's `whisper-large-v3-turbo` returns word-level timing in the same verbose_json shape). Added `TranscriptionWord` struct and a `words` field on `TranscriptionText` / `TranscriptionVerboseJson` (defaults to empty when not requested, so the existing SEO pipeline is unaffected).
- Added clipper-side alignment helpers (`src/align.rs`): `AlignedWord` in absolute episode time, `shift_words()` to translate chunk-local timestamps into the global timeline, `snap_to_word_boundary()` so clip cuts never slice mid-word, and `speaking_rate_wps()` for a per-window ranker feature. 6 tests including JSON-shape verification against Groq's response.
- Added linguistic-marker extractor (`src/linguistic_markers.rs`): pure-Rust regex over a transcript window, counts conflict markers, strong-claim openers, confessional cues, topic-shift markers, numbers, questions, and short declaratives. Also picks up to 2 quotable lines (shortest declaratives) for the ranker's evidence block. 9 tests.
- Added candidate-window generator + feature aggregator (`src/candidates.rs`): walks the episode in `stride_secs` strides, snaps proposed boundaries to silence (preferred) or shot cuts (fallback), enforces `[min_secs, max_secs]` clamps, and attaches per-window features from every extractor (linguistic, prosody RMS peak/mean, speaking rate, transcript text). Novelty score is filled in by a separate async pass through the embedder so the generation path stays pure/sync. Drops near-duplicate and low-word windows. 5 tests covering edge cases and feature aggregation.
- Added LLM-driven clip ranker (`src/ranker.rs` + `prompts/clips/ranker_system.txt` + `prompts/clips/ranker_user.txt`): batches candidate windows (default 10/call) with all features serialized as structured evidence, calls existing OpenAI-compatible `chat_json`, parses `{score, hook, refined_start, refined_end, reasoning}` per candidate. Refined boundaries are clamped to ±5s of the original candidate so a hallucinating LLM can't slice into adjacent content. Top-K sorted by score on return. 4 tests.
- Added per-clip ffmpeg renderer (`src/render.rs`): cut + center-crop reformat (vertical 9:16 / square 1:1 / letterboxed 16:9) + single-pass `loudnorm` + libx264. `RenderProfile` presets for Shorts (-14 LUFS), TikTok/Reels/Threads (-16 LUFS), LinkedIn square, Bluesky landscape. Optional `subtitle_path` parameter burns an `.ass` file via the `subtitles=` filter. 8 tests including end-to-end synthetic-video verification of 1080×1920 output.
- Added `.ass` caption writer (`src/captions.rs`): pure-Rust ASS v4+ generator producing "popping" phrase captions (2–4 words per phrase, white text + bold black outline at bottom-center). Groups words on word-count, char-budget, and terminal-punctuation boundaries. Karaoke `{\k}` highlighting is a future polish. 9 tests.
- Added word-level transcription helper (`AiPipeline::transcribe_word_chunks`): mirror of the existing chunked STT path but calls `transcribe_words` (Groq) and merges word timestamps back into the global episode timeline. Returns a `WordTranscript` carrying full_text + segments + words.
- Added clipper orchestrator (`src/clipper.rs`) — the M1 closer. End-to-end pipeline for `MODE=clipper LOCAL_VIDEO_PATH=…`: extract audio → chunk → parallel (word-transcribe + scene-detect + silence-detect + RMS-curve via `tokio::try_join!`) → candidate generation → LLM ranker → render top-K with burned captions → digest email with attachments (capped at 18MB/file, 20MB total; over-budget clips referenced by filename in the body). New env: `CLIP_RANKER_SYSTEM_PROMPT_PATH`, `CLIP_RANKER_USER_PROMPT_PATH`, `CLIP_TOP_K` (default 10). 5 tests covering body formatting + helpers.
- Main routing now dispatches on `mode`: `seo-only` (default, existing behavior unchanged) | `clipper` (new pipeline, M1 supports LOCAL_VIDEO_PATH only) | `both` (run both). The Gmail/Drive polling-to-clipper integration is M2.
- Made Google credentials optional. New `DIGEST_MODE` env: `file` (default — writes `digest.md` to the clips directory; no external service needed), `email` (Gmail digest send; requires Google creds + `RESULT_TO`), `both`. The clipper can now run with just `OPENAI_API_KEY` + `LOCAL_VIDEO_PATH` set — no Gmail/Drive/OAuth in the loop. Validation matrix at startup gates the Gmail-dependent paths with clear error messages. The seo-only mode still requires Google (it sends per-variant emails by design).
- Added HF Inference Providers embeddings backend (slice 5a). `src/embed.rs` now exposes an `Embedder` enum that picks between `FastembedEmbedder` (local ONNX, default `all-MiniLM-L6-v2`) and `HfEmbedder` (OpenAI-compatible `/v1/embeddings`, default `Qwen/Qwen3-Embedding-0.6B`). When `HF_API_KEY` is set, the clipper routes embeddings through HF — 2026-grade model, 1024-dim, 32K context, ~$0.05/episode. Otherwise falls back to fastembed. `attach_novelty()` is now wired into the clipper pipeline (was built but unused in M1). Non-fatal: novelty failure logs a warning and the ranker proceeds without it.
- Added VLM re-rank stage (slice 5b). `src/vlm_ranker.rs` extracts N evenly-spaced frames from each top-K candidate via the existing `screenshot_jpeg` helper, base64-encodes them as data URIs, and sends frames + transcript hook + LLM reasoning to a vision-language model via HF Inference Providers (default `Qwen/Qwen3-VL-8B-Instruct`). Returns a 0-100 score that's blended with the LLM score via `VLM_BLEND_WEIGHT` (default 0.5). Opt-in via `VLM_RERANK_ENABLED=true`; requires `HF_API_KEY`. Cost: ~$0.20-0.50/episode at typical Together/Fireworks routing. New env: `VLM_RERANK_ENABLED`, `VLM_MODEL`, `VLM_RERANK_TOP_K` (default 20), `VLM_FRAMES_PER_CLIP` (default 5), `VLM_FRAME_MAX_DIM` (default 512), `VLM_BLEND_WEIGHT`.
- Fixed HF embeddings URL pattern. The OpenAI-compatible `/v1/embeddings` endpoint does NOT exist on HF Inference Providers (chat-only per the docs); the embeddings path is the native feature-extraction task at `{HF_ROUTER_URL}/{HF_EMBED_PROVIDER}/models/{EMBED_MODEL}/pipeline/feature-extraction` with payload `{"inputs": [...]}` and bare-array response. `Qwen/Qwen3-Embedding-0.6B` (the prior research recommendation) is not actually deployed on HF Inference — switched the default to `BAAI/bge-large-en-v1.5` (1024-dim, MIT, reliably warm on `hf-inference`). Renamed `HF_BASE_URL` to `HF_ROUTER_URL` (root, no `/v1`). New env: `HF_EMBED_PROVIDER` (default `hf-inference`). VLM client now builds its `/v1/chat/completions` URL from the same router root. End-to-end verified against a real 10-min podcast slice: novelty attached to all 20 candidates, VLM re-rank promoted 2 physical-action clips the LLM under-rated, demoted static-talking-head clips the LLM over-rated.
- M2 phase 1: actual posting to YouTube Shorts + Bluesky (the two free, no-app-review platforms).
  - `src/platforms/mod.rs` — `Platform` enum, `PostResult`, `PostStatus` (Posted / DryRun / Skipped / Failed). Constructs configured backends from `POST_ENABLED_PLATFORMS`.
  - `src/platforms/bluesky.rs` — ATProto flow: `createSession` → `getServiceAuth` → `app.bsky.video.uploadVideo` → poll `getJobStatus` until COMPLETED → `com.atproto.repo.createRecord` with `app.bsky.embed.video`. App-password auth (no main login password). 300-char post text composer with hashtag merging + truncation. Converts the returned `at://` URI to a `bsky.app/profile/.../post/...` web URL.
  - `src/platforms/youtube.rs` — Data API v3 resumable upload (`videos.insert`). Reuses the existing `GoogleAuth` token refresher. Auto-appends `#Shorts` to descriptions that lack it. Surfaces a useful error message when the refresh token doesn't have the `youtube.upload` scope (the operator's existing Gmail/Drive token needs re-minting).
  - `src/posting.rs` — orchestrator that selects the 9:16 variant per clip and dispatches to each configured platform.
  - Safety: TWO opt-ins to actually post — `POST_ENABLED_PLATFORMS` must list platforms AND `POST_DRY_RUN=false`. Default is empty + dry-run.
  - New env: `POST_ENABLED_PLATFORMS`, `POST_DRY_RUN` (default true), `YOUTUBE_PRIVACY_STATUS` (default `unlisted`), `YOUTUBE_CATEGORY_ID` (default `24` Entertainment), `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`, `BLUESKY_PDS_URL`, `BLUESKY_VIDEO_SERVICE_URL`.
  - Clipper writes per-platform `PostResult` into `manifest.json` and the digest body. Viewer (`tools/serve_clips.py`) renders a Posts strip per clip card with color-coded status pills (green = posted, purple = dry-run, gray = skipped, red = failed) and clickable external URLs for successful posts.
  - 17 new tests across the three new modules (parse-enabled, post-result constructors, Bluesky text composition + at:// → web URL, YouTube metadata building + tag stripping). Total now 112.

- Slice 6: full multi-platform packaging (M2-prep).
  - **6a.** Per-clip social-media copy generator (`src/social_copy.rs` + `prompts/clips/social_{system,user}.txt`): one LLM call per top-K clip returns structured JSON with platform-appropriate title/description/hashtags/etc. for YouTube Shorts, TikTok, Instagram Reels, Threads, LinkedIn, X, Bluesky — plus a 2-5 word `overlay_hook` for the burned first-1.5s overlay. Opt-out via `CLIP_SOCIAL_COPY_DISABLED=true`. Cost: ~$0.30/episode extra.
  - **6b.** Multi-format render: each clip now renders in every aspect requested via `CLIP_RENDER_FORMATS` (default `9x16,1x1,16x9`). Aspect-aware caption styles in `captions.rs` (`for_vertical`, `for_square`, `for_landscape`) with sane font sizes and margins per format. `RenderedClip` carries a `variants: Vec<RenderedVariant>` instead of one path.
  - **6c.** Overlay hook burn: new `captions::write_overlay_ass` + `OverlayStyle` (aspect-aware). `render::render_clip` now accepts a `&[&Path]` of subtitle layers (captions first, overlay second so it draws on top). The overlay event uses a `{\\fade(300,300)}` animation centered in the frame for the first 1.5s.
  - **6d.** Machine-readable manifest: clipper now writes `manifest.json` alongside `digest.md` — schema-versioned structured data covering every rendered variant + per-platform copy. `tools/serve_clips.py` rewritten to load `manifest.json` and render a rich HTML viewer with format-switcher tabs, per-platform tabs, and copy-to-clipboard buttons per field. Falls back gracefully for older runs without a manifest.
  - Tests: 100 pass, 2 ignored. 5 new social-copy tests + 4 new overlay-ass tests + 1 multi-subtitle render test + updated existing tests for the new RenderedClip shape.

## 2025-12-14
- Added OpenAI Responses API fallback so `gpt-5.*` models work with JSON-only output.
- Added local end-to-end mode (`LOCAL_VIDEO_PATH`) to run transcription → LLM → thumbnails → email without Gmail/Drive ingest.
- Hardened SEO prompts to reduce hallucinations (transcript-grounded, less-specific output).
- Improved thumbnail selection to reliably return 10+ moments (prompt requirement + code-side padding when the model under-delivers).
- Improved thumbnail rendering performance:
	- Bounded concurrent ffmpeg screenshot generation with a progress bar (`THUMBNAIL_FFMPEG_CONCURRENCY`).
	- Reduced screenshot work (avoid generating 2x images when only N are emailed) and lowered ffmpeg startup overhead.
	- Added configurable downscale (`THUMBNAIL_MAX_HEIGHT`, default 720) while keeping JPEG quality high (`-q:v 1`).
- Added stable terminal UX with two on-screen progress bars (transcription + downstream LLM/thumbnails) and silenced ffmpeg log spam.
- Added STT throughput controls:
	- Parallel chunk transcription via `STT_CONCURRENCY`.
	- Optional STT RPM pacing (`STT_RPM_LIMIT`) plus retries/backoff for transient 429/5xx.
- Added auto chunk-sizing heuristics (including explicit target chunk counts, default 400).
- Externalized SEO/thumbnail prompts into `prompts/*.txt` and load them via config.

## 2025-12-13
- Logged Gmail `sent_message_id` for auditability and cleaned build warnings.
- Hardened Gmail/Drive parsing, ensured mp4-only processing when `REQUIRE_VIDEO=true`, and added screenshot improvements groundwork.
