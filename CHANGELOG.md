# Changelog

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
