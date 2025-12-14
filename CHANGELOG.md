# Changelog

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
