# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-binary Rust worker (`autoseo`) that polls Gmail for Google Drive share notifications, downloads the shared media, transcribes it via an OpenAI-compatible STT endpoint, asks an LLM for YouTube SEO packages (multiple variants), generates thumbnail screenshots for videos, and emails the result back. Runs as a long-lived poller in Docker, or as a one-shot for local files.

## Commands

```bash
# Build / check
cargo build
cargo build --release
cargo check

# Run
cargo run                              # continuous poll loop
cargo run -- --once                    # one cycle
cargo run -- --once --dry-run          # Gmail+Drive only (no OpenAI, no send)
LOCAL_VIDEO_PATH=/path/to.mp4 cargo run -- --once   # skip Gmail/Drive, process local file

# Tests
cargo test
cargo test --test <name>               # single integration test (none currently)
cargo test parse::tests::extracts_file_d -- --exact   # single unit test
cargo test thumbnail_generation_smoke  # requires ffmpeg/ffprobe on PATH; auto-skips if missing

# Logs / format
RUST_LOG=debug cargo run -- --once
cargo fmt
cargo clippy

# Docker (preferred deployment)
docker compose up -d --build
docker compose up -d --force-recreate  # after editing .env (e.g. new refresh token)
docker logs -f --tail=200 autoseo
```

Required env for real runs: `RESULT_TO`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`, `OPENAI_API_KEY`. Full list in `.env.example` and as `#[arg(env=...)]` annotations in [src/config.rs](src/config.rs). `--dry-run` drops the OpenAI + `RESULT_TO` requirements.

## Architecture

### Pipeline (one Gmail message → one set of result emails)

Orchestrated by `run_once` in [src/main.rs](src/main.rs) (and the parallel `run_local_once` for the `LOCAL_VIDEO_PATH` bypass path — they share the same shape and any pipeline change must be made in both):

1. **Gmail list/get** ([src/gmail.rs](src/gmail.rs)) → message bodies. Falls back from inline-base64 bodies to `attachments.get` resolution, and finally to raw RFC822 if Drive links still aren't found. Failed parses can be dumped to `DUMP_DIR` for forensics.
2. **Drive ID extraction** ([src/parse.rs](src/parse.rs)) — tolerates `google.com/url?q=` redirect wrappers and several `drive.google.com` URL shapes.
3. **Dedupe** ([src/dedupe.rs](src/dedupe.rs)) — append-only file of processed Gmail message IDs; loaded into a HashSet at startup. Insertion happens *after* all variant emails are sent successfully.
4. **Drive metadata + streamed download** ([src/drive.rs](src/drive.rs)) — writes to `<name>.partial` then atomic rename. Re-uses existing files when size matches Drive metadata.
5. **Media prep** ([src/media.rs](src/media.rs)):
   - Video → `ffmpeg` extracts 16 kHz mono AAC `audio.m4a`.
   - Non-WAV audio → same transcode path.
   - WAV → bypasses ffmpeg entirely; `hound` reads samples and writes per-chunk WAV files. WAV duration also comes from `hound`, not ffprobe.
6. **Chunking** — `choose_chunk_secs` in [src/main.rs](src/main.rs) computes chunk size from total duration and `AUTO_CHUNK_TARGET_CHUNKS` (default 400), clamped to `AUTO_CHUNK_MIN_SECS`..`AUTO_CHUNK_MAX_SECS`. Stale `chunk_*` files in the chunks dir are cleared first so a smaller new run doesn't reuse old chunks.
7. **Parallel STT** ([src/ai_pipeline.rs](src/ai_pipeline.rs) `transcribe_chunks`) — `buffer_unordered(STT_CONCURRENCY)` (default 500), optionally rate-limited by `STT_RPM_LIMIT` via `RpmGate` ([src/rate_limit.rs](src/rate_limit.rs)). Segment offsets are added back from the chunk's offset so timestamps remain global. When the provider returns no segments, `synthesize_minute_segments` fabricates ~1-minute buckets from the text so thumbnail selection still works.
8. **Show inference** — `infer_show_context` runs a small JSON LLM call against the filename + transcript head. Hard rule: only return show/host/guest if explicitly present. Failures are non-fatal.
9. **SEO variants + thumbnail moments run concurrently** via `tokio::join!` — N=`SEO_VARIANTS` chat-completions plus one JSON thumbnail-moment call (videos only). Thumbnail moments are deduped (0.25s tolerance) and padded deterministically via `fallback_thumbnail_moments` if the LLM under-delivers.
10. **Thumbnail render** ([src/thumbs.rs](src/thumbs.rs)) — bounded concurrent ffmpeg seeks (`THUMBNAIL_FFMPEG_CONCURRENCY`, default 4) with `-q:v 1` + optional `scale=-2:H:lanczos`. Uses `-ss` before `-i` and `-probesize 32k -analyzeduration 0` for cheap seeks. Extra "buffer" shots cover occasional empty-output failures, then the list is truncated to `THUMBNAIL_COUNT`.
11. **MIME assembly + Gmail send** ([src/mime.rs](src/mime.rs), [src/gmail.rs](src/gmail.rs)) — multipart/mixed when there are attachments, plain text otherwise; base64url-encoded into `users.messages.send`. One email per SEO variant; subject suffix `(i/N)`.

### OpenAI client model routing

[src/openai.rs](src/openai.rs) auto-routes between `/v1/chat/completions` and `/v1/responses`:

- Models starting with `gpt-5` go straight to `/v1/responses` (no `temperature` field — gpt-5 rejects it).
- For other models, chat is tried first; on `404` with messages like "not a chat model" or "Did you mean to use v1/responses", it falls back to Responses.
- STT (`/v1/audio/transcriptions`) prefers `verbose_json` for segment timing and falls back to `json` if the provider rejects verbose. Multipart parts are rebuilt per retry attempt (they're consumed on send). Retries 429/5xx with exponential backoff up to ~8s, but bails immediately on `insufficient_quota`.

### Prompts (external, hot-editable without recompile)

All prompt text lives in [prompts/](prompts/) and is loaded at startup from paths in `Config`:

- `seo_system.txt` / `seo_user.txt` — base SEO prompt. The user template supports `{{transcript}}`, `{{variant_instructions}}`, `{{variant_index}}`, `{{variant_total}}`, `{{media_name}}`, `{{show_name}}`, `{{hosts}}`, `{{guest}}`. Missing context resolves to empty string.
- `seo_variants.txt` — variant blocks separated by lines containing exactly `---`. Leading `#` comment lines per block are stripped. `parse_variants_prompt_file` in [src/main.rs](src/main.rs) handles the split. `SEO_VARIANTS` selects N blocks (wraps modulo).
- `thumbnail_system.txt` / `thumbnail_user.txt` — supports `{{count}}` and `{{minutes}}` (the minute-indexed transcript summary built by `minute_index`).
- `example_response.txt` — referenced from `seo_user.txt` as the desired output shape (rich-text email body, not JSON).

### State and filesystem

- `WORK_DIR` (default `./work`) holds per-job dirs: `<sanitized_filename>/<gmail_message_id>/` containing the downloaded media, `audio.m4a`, `audio_chunks/chunk_NNNNN.{m4a,wav}`, and `thumbnails/`. The Docker container mounts this volume at `/work` for reboot persistence.
- `DEDUPE_FILE` (default `./work/processed_message_ids.txt`) — newline-delimited; corruption-resistant because it's append-only and loaded with empty-line tolerance.
- `DUMP_DIR` (optional) — when a Gmail message has no extractable Drive ID, writes `<id>_parts.txt`, `<id>_message.json`, `<id>.eml`, and `<id>_urls.txt`.

### Modes summary

- **Continuous poller (default)** — loops `run_once` every `POLL_INTERVAL_SECS`. Processes only the newest matching message per cycle (`break` after first success). Errors in a cycle are logged and the loop continues.
- **`--once`** — single cycle, exits.
- **`--dry-run`** — lists Gmail, extracts Drive IDs, prints metadata, marks the message as processed, and stops. No OpenAI client is constructed.
- **`LOCAL_VIDEO_PATH`** — runs `run_local_once`: skips Gmail/Drive ingest and dedupe, runs the full transcribe → SEO → thumbnail → email pipeline against a local file. Still requires Gmail send credentials and `RESULT_TO`.

## Conventions worth knowing

- The pipeline branches on `is_wav` / `is_audio` / `is_video` from MIME type *or* filename extension; `REQUIRE_VIDEO=true` skips audio entirely. Audio-only runs skip thumbnail generation and tag the email subject with `(audio)`.
- "Skip and mark processed" is the standard outcome for unparseable / inaccessible / wrong-type messages, so the poller doesn't retry them forever.
- Progress UI uses `indicatif::MultiProgress` with two fixed bars (transcription + downstream LLM/thumbs). ffmpeg is invoked with `-hide_banner -loglevel error -nostats` so it doesn't fight the bars.
- `ffmpeg` / `ffprobe` paths come from `FFMPEG` / `FFPROBE` env (default just the bare command names); `ensure_tool_available` preflights them before any large download so we fail fast on missing binaries. The WAV path skips this preflight.
- Atomic file writes via `.partial` rename (Drive downloads) and per-chunk file finalization (WAV writer) — anything that could be interrupted writes to a temp name first.

## Clipper extension (`MODE=clipper` / `LOCAL_VIDEO_PATH`)

The binary is being evolved into an autonomous clipper agent. Long-form podcast → N platform-specific short clips with platform-tailored copy, dry-run posting to YouTube Shorts + Bluesky. Working branch: `feat/clipper`.

- `MODE` env toggles pipeline: `seo-only` (default, original behavior), `clipper`, `both`.
- Clipper orchestrator: [src/clipper.rs](src/clipper.rs). End-to-end: extract audio → chunk → parallel (word-transcribe via Groq + scene detect + silence detect + RMS curve) → candidate windows → embedding novelty → LLM ranker → optional Qwen3-VL re-rank → render top-K in 9:16/1:1/16:9 with burned overlay hook → per-platform `SocialCopy` from `src/social_copy.rs` → `manifest.json` → posting via `src/platforms/`.
- Embeddings via [src/embed.rs](src/embed.rs) — HF Inference Providers (`BAAI/bge-large-en-v1.5`, 1024-dim) when `HF_API_KEY` is set, else local `fastembed` MiniLM. URL pattern is the native feature-extraction task at `{HF_ROUTER_URL}/{HF_EMBED_PROVIDER}/models/{model}/pipeline/feature-extraction` (NOT `/v1/embeddings` — that endpoint doesn't exist on HF Inference Providers).
- VLM re-rank ([src/vlm_ranker.rs](src/vlm_ranker.rs)) — opt-in via `VLM_RERANK_ENABLED=true`. Extracts N frames per candidate, sends to Qwen3-VL via HF chat-completions, blends with LLM score.
- Posting backends ([src/platforms/](src/platforms/)) — `Platform::from_config` builds enabled set from `POST_ENABLED_PLATFORMS`. YouTube via direct multipart upload; Bluesky via ATProto (`createSession` → `getServiceAuth` → `uploadVideo` → `getJobStatus` poll → `createRecord`). YouTube uploads need a refresh token with the `youtube.upload` scope (re-mint at OAuth Playground if missing).
- The standing visual test harness is [tools/serve_clips.py](tools/serve_clips.py) — reads `manifest.json` and renders a static HTML clip browser with format switcher + platform tabs + posts status. New clipper features extend the manifest schema rather than spawning new viewers. **This will be retired in dashboard slice 14** once the Rust port reaches parity.

## Dashboard (`autoseo dashboard`)

A self-hosted admin dashboard / clip command center is being built on the same binary. Subcommands: `worker` (existing default), `dashboard` (axum server), `all` (both in one process). Default port 7788.

- **Plan file:** [/home/tris/.claude/plans/ok-great-lets-write-zazzy-hellman.md](/home/tris/.claude/plans/ok-great-lets-write-zazzy-hellman.md) — full multi-slice roadmap with locked decisions, schema, API surface, slice-by-slice acceptance tests, and v2 SaaS roadmap. **Read this first before extending the dashboard.**
- **Stack:** axum 0.8 + tower-http + maud (compile-time HTML) + HTMX 2 + Alpine + Tailwind (precompiled). Solid.js island for the trim editor (slice 4b). Static assets baked in via `rust-embed`. No Node at runtime.
- **Data:** same SQLite DB (`./work/clipper.db`) the worker uses; v2 migration adds the dashboard surface (11 new tables, 18 ALTER columns on jobs/clips/posts).
- **SaaS-readiness seams** (cheap-now, zero infra creep): `workspace_id` on every user-owned row (v1 always `ws_default`), repo traits at [src/dashboard/repo/traits.rs](src/dashboard/repo/traits.rs), OIDC-shaped session payload, `/media/:job/:file` 302→signed-URL, `clip_embeddings` BLOB table, `events` ur-NATS log table, ULID primary keys.
- **Auth (slice 2, in progress):** local users table with `argon2id` PHC hashes. HMAC-signed session cookie carrying OIDC-shaped claims (`sub`, `aud`, `exp`, `iat`, `jti`, `workspace_id`) so a future Clerk/Better Auth/WorkOS swap is drop-in. Roles: `admin` / `editor` / `viewer`. CSRF via double-submit cookie + `X-CSRF-Token` header. Vault: XChaCha20-Poly1305 with 32-byte master key from `DASHBOARD_MASTER_KEY` env.
- **Slice status** (current state on `feat/clipper`):
  - ✅ Slice 0 — subcommand split, axum boot, `/health`, static asset mount.
  - ✅ Slice 1 — SQLite migrated to `user_version=2` with full dashboard schema. Repo trait scaffolding + `workspaces` + `audit` DAOs.
  - ⏳ Slice 2 — auth + users (next).
  - ⏳ Slices 3–14 — clip browser, edits, trim island, approval, manual repost, vault UI, scheduling, per-show settings, ingest, bulk ops, audit view, deploy polish. (Slice 12 — analytics polling — deferred to v1.1.)
- **Module layout:** [src/dashboard/](src/dashboard/) — `mod.rs`, `server.rs`, `state.rs`, `config.rs`, `error.rs`, `middleware.rs`, `prelude.rs`, `routes/` (slice-by-slice), `repo/` (per-table DAOs), `templates/` (maud, lands slice 3+), `auth/` (lands slice 2), `crypto/` (lands slice 7), `scheduler/` (lands slice 6), `ingest/` (lands slice 10), `island/trim/` (Solid.js source for slice 4b).
- **Verifying the dashboard manually:**
  ```bash
  cargo run -- dashboard --bind 127.0.0.1:7788 --insecure
  curl -s http://127.0.0.1:7788/health | jq
  # → {"ok":true,"version":"0.1.0","schema":2,"scheduler":"disabled"}
  ```
- **Backward compat:** legacy `autoseo --once --dry-run` still works — `Cli` has `Option<Command>` + flattened `Config`; absent subcommand defaults to `worker`.

## Pickup notes for new agents

1. Read [the plan file](/home/tris/.claude/plans/ok-great-lets-write-zazzy-hellman.md) end-to-end — it covers every slice's touches, blockers, and acceptance tests.
2. Run `cargo test` (should pass 114 + 2 ignored).
3. Run `cargo run -- dashboard --insecure` and confirm `/health` reports `schema:2`.
4. Pick up at the next pending slice (track in `CHANGELOG.md`'s most recent entry).
5. Don't bypass the plan — if a decision arises that isn't covered, pause and surface it to the operator.
6. Operator preferences worth remembering:
   - No human-day or hour estimates in plans / status updates. Scope work in terms of WHAT / BLOCKERS.
   - The Python viewer is the standing test harness — keep it through slice 3, retire in slice 14.
   - Commit per-slice with `feat(dashboard): slice N — …` on the `feat/clipper` branch.
