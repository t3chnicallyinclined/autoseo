# Changelog

## 2025-12-14
- Added OpenAI Responses API fallback so `gpt-5.*` models work with JSON-only output.
- Improved thumbnail capture quality (LANCZOS scaling, yuv444p, `-q:v 1`).
- Added `STT_CONCURRENCY` option and parallel chunk transcription to better utilize Whisper RPM limits.
- Added auto chunk-sizing heuristics (now supporting explicit target chunk counts, default 400) plus config toggles, and raised default STT concurrency to 500.
- Auto chunking now requires successful duration probes (ffprobe/WAV) to prevent silent fallbacks; runs abort if we can't measure the source.
- Externalized SEO/thumbnail system prompts into `prompts/*.txt` and load them via config for easy tuning.
- Introduced config flags for prompt paths plus default prompt files checked into the repo.

## 2025-12-13
- Logged Gmail `sent_message_id` for auditability and cleaned build warnings.
- Hardened Gmail/Drive parsing, ensured mp4-only processing when `REQUIRE_VIDEO=true`, and added screenshot improvements groundwork.
