# PLAN.md — Autoseo → Clipper Agent

Locked-in plan for evolving autoseo from an SEO-email producer into a fully autonomous clipper agent. See [CLAUDE.md](CLAUDE.md) for the current system; this document is the forward-looking design.

## Vision

A long-form podcast video lands via Drive share. The system autonomously: selects N high-CTR moments, renders each in platform-specific formats with karaoke captions, writes per-platform SEO copy, posts to the configured social platforms, and emails a digest. Zero human-in-the-loop after the share lands.

## Design principles

1. **API-first, GPU optional.** Default lane runs in the existing CPU Docker container against OpenAI-compatible / Groq / OpenRouter endpoints. A GPU sidecar is opt-in for the active-speaker reframe + WhisperX alignment quality jump.
2. **Transcript + context, not audio reactions.** Studio podcast content has no live audience. Moment selection is LLM-ranked over dense transcript windows, augmented by prosody, linguistic markers, embedding novelty, and external trending context (GDELT, Reddit). Audio-event detection (AST laughter) is a tier-2 signal, not the driver.
3. **Rust-only orchestrator and inference (M1).** ML runs in-process via `ort` (ONNX Runtime), `fastembed`, `whisper-rs`, `aubio`, and ffmpeg filters. No Python sidecar in M1 — single binary, single Docker image, no IPC. The sidecar option is **deferred to M3**, only invoked if Light-ASD active-speaker detection or emotion2vec proves non-negotiable for vertical reframe. Every M1 model has either an ONNX export, a native Rust crate, or an API-friendly equivalent (Groq for Whisper word timestamps).
4. **The clip is the unit of work.** Dedupe becomes `(job_id, clip_id, platform)` in SQLite — each tuple succeeds or retries independently.

## Pipeline

```
Gmail msg → Drive download → ffmpeg audio extract → Whisper STT  (existing)
                                                       │
                              ┌────────────────────────┤
                              ▼                        ▼
                         WhisperX align           PySceneDetect
                         (word-level)             (shot bounds)
                              │                        │
                              └──────┬─────────────────┘
                                     ▼
                              Silero VAD silences
                                     ▼
                       Dense candidate windows (30–90s,
                       snapped to VAD + shot bounds)
                                     │
       ┌──────────┬──────────┬───────┴───────┬──────────┬──────────┐
       ▼          ▼          ▼               ▼          ▼          ▼
  linguistic  prosody   embedding       turn         emotion   trend
  markers     (librosa) novelty         density      (M3)      context
  (Rust)      (sidecar) (sidecar)       (Rust)                 (GDELT
                                                                + Reddit)
       └──────────┴──────────┴───────┬───────┴──────────┴──────────┘
                                     ▼
                       LLM ranker — batched, all features
                       injected as evidence per candidate
                                     ▼
                       Top-K clips with hook + score
                                     │
                       (optional) Lane B re-rank
                       Qwen3-VL on frames + audio
                                     ▼
                       For each (clip, platform variant):
                         cut → reframe → DeepFilterNet → loudnorm
                         → karaoke .ass burn → encode → SEO copy → post
                                     ▼
                                Digest email
```

## Model stack

All M1 stages run in-process in the Rust binary. Python is **not introduced in M1**; the decision is revisited at M3.

| Stage | Primary (CPU, Rust + APIs) | Premium / M3 GPU |
|---|---|---|
| Transcription | Groq `whisper-large-v3-turbo` API (existing OpenAI-compatible client) | `whisper-rs` (whisper.cpp bindings) for offline mode |
| Word timestamps | Groq API `word_timestamps=true` | `whisper-rs` with whisper.cpp word-level timing |
| VAD | Silero VAD ONNX via `ort` crate | same |
| Shot bounds | ffmpeg `scenedetect` filter (`-vf select='gt(scene,T)',showinfo`) — no ML | same |
| Prosody | `aubio` F0 + ffmpeg `astats` RMS + transcript-derived speaking rate | same |
| Embedding novelty | `fastembed` (all-MiniLM-L6-v2 ONNX) | same |
| Linguistic markers | Rust regex + simple NLP | same |
| Audio events (M3) | AST ONNX via `ort` | same |
| Speech enhancement | ffmpeg `loudnorm` 2-pass | `deep_filter` crate (native Rust DeepFilterNet3) |
| Face detect (M3) | SCRFD ONNX via `ort` + simple IoU tracker | InsightFace via `ort` with ByteTrack |
| Active speaker (M3) | — (rule-based: face presence + VAD overlap as fallback) | **Light-ASD — likely requires Python sidecar; decision point at M3** |
| Emotion (M3, optional) | — | emotion2vec — Python sidecar if added |
| LLM ranker | existing OpenAI-compatible chat | Qwen3-VL-8B / Qwen2.5-VL-72B via OpenRouter for top-K only |

## Platform variant matrix

| Platform | Aspect | Max len | Audio target | Notes |
|---|---|---|---|---|
| YouTube Shorts | 9:16 | 60s | -14 LUFS | `#Shorts` in title/desc |
| TikTok | 9:16 | 60s / 90s alt | -16 LUFS | native captions if API allows |
| Instagram Reels | 9:16 | 60s | -16 LUFS | cover frame from thumbnail moments |
| LinkedIn | 16:9 or 1:1 | 90s | -14 LUFS | longer preamble in description |
| Threads | 9:16 | 60s | -16 LUFS | shares Reels render |
| Bluesky | 16:9 or 9:16 | 3min | -16 LUFS | 100MB cap |
| YouTube long | 16:9 | full | -14 LUFS | original re-upload, existing SEO pipeline |

A single clip renders 1–3 MP4 files (9:16 + optional 1:1 + optional 16:9).

## Posting phases

- **Phase 1 (free, no review):** YouTube Data API v3 + Bluesky ATProto
- **Phase 2 (free, light review):** LinkedIn Posts API + Threads Graph API
- **Phase 3 (audit-gated, weeks):** TikTok Content Posting API + Instagram Reels Graph API. Optional Ayrshare shim during wait.
- **Deferred:** X (per-post pricing kills clipper economics), Pinterest (low short-form ROI)

## Project layout (target)

Rust-only for M1. The `sidecar/` directory does not exist; it returns only if M3 forces it.

```
src/
  ai_pipeline.rs          # extend: rank_candidates, seo_for_clip
  align.rs                # NEW: Groq word-timestamp client; whisper-rs offline fallback
  scene.rs                # NEW: ffmpeg scenedetect parser (no ML)
  vad.rs                  # NEW: Silero VAD via ort
  prosody.rs              # NEW: aubio F0 + ffmpeg astats RMS
  embed.rs                # NEW: fastembed for novelty scoring
  linguistic_markers.rs   # NEW: regex feature extractor (pure Rust)
  candidates.rs           # NEW: window generation, snap to VAD + shots, scoring
  render.rs               # NEW: per-clip ffmpeg orchestration, .ass karaoke writer
  context/                # NEW: trend ingestion
    mod.rs                # ContextFetcher trait
    gdelt.rs
    reddit.rs
    google_trends.rs      # v2
    cache.rs              # SQLite-backed, daily refresh
  audio_events.rs         # M3: AST ONNX via ort
  enhance.rs              # M3: DeepFilterNet3 via deep_filter crate
  face_detect.rs          # M3: SCRFD ONNX via ort + simple tracker
  asd.rs                  # M3: active-speaker — rule-based primary; Light-ASD client if Python sidecar lands
  platforms/              # M2+
    mod.rs                # Platform trait
    youtube.rs
    bluesky.rs
    linkedin.rs
    threads.rs
    tiktok.rs
    instagram.rs
    ayrshare.rs
  jobs.rs                 # NEW: JobId, ClipId, status FSM, retry tracking
  storage.rs              # (exists)
  show_config.rs          # (exists)

models/                   # NEW: ONNX model artifacts
  silero_vad.onnx         # ~1.8 MB, bundled in image
  minilm_l6_v2/           # fastembed default location
  scrfd_10g_bnkps.onnx    # M3

prompts/
  clips/
    ranker.txt            # virality scoring template
    youtube.txt           # per-platform copy templates
    tiktok.txt
    instagram.txt
    linkedin.txt
    bluesky.txt
    threads.txt
    hook_overlay.txt      # 1.5s overlay text
  shows/{show_slug}/      # per-show overrides
    seo_system.txt
    seo_user.txt
    seo_variants.txt
    thumbnail_system.txt
    thumbnail_user.txt
    context_sources.txt
```

## SQLite schema (M1)

```sql
CREATE TABLE jobs (
  id              TEXT PRIMARY KEY,        -- = gmail message_id
  show_slug       TEXT,
  media_name      TEXT,
  drive_file_id   TEXT,
  status          TEXT,                    -- pending|transcribed|ranked|rendered|posted|done|failed
  created_at      INTEGER,
  updated_at      INTEGER,
  cost_cents      INTEGER DEFAULT 0,
  error           TEXT
);

CREATE TABLE clips (
  id              TEXT PRIMARY KEY,        -- ulid
  job_id          TEXT REFERENCES jobs(id),
  start_ms        INTEGER,
  end_ms          INTEGER,
  rank            INTEGER,                 -- 1 = best
  score           REAL,
  hook            TEXT,
  reasoning_json  TEXT,                    -- features that drove the score
  trend_match     TEXT                     -- nullable trend id
);

CREATE TABLE clip_renders (
  clip_id         TEXT REFERENCES clips(id),
  variant         TEXT,                    -- '9x16' | '1x1' | '16x9'
  path            TEXT,
  bytes           INTEGER,
  duration_ms     INTEGER,
  PRIMARY KEY (clip_id, variant)
);

CREATE TABLE posts (
  clip_id         TEXT REFERENCES clips(id),
  platform        TEXT,                    -- 'youtube' | 'tiktok' | ...
  status          TEXT,                    -- pending|posted|failed|vetoed
  external_id     TEXT,
  external_url    TEXT,
  posted_at       INTEGER,
  error           TEXT,
  PRIMARY KEY (clip_id, platform)
);

CREATE TABLE analytics (
  clip_id         TEXT,
  platform        TEXT,
  fetched_at      INTEGER,
  views           INTEGER,
  ctr             REAL,
  watch_pct       REAL,
  PRIMARY KEY (clip_id, platform, fetched_at)
);

CREATE TABLE trends (
  source          TEXT,                    -- 'gdelt' | 'reddit' | 'google'
  topic_id        TEXT,
  label           TEXT,
  score           REAL,
  fetched_at      INTEGER,
  PRIMARY KEY (source, topic_id, fetched_at)
);
```

Dedupe legacy file (`processed_message_ids.txt`) is imported into `jobs` on first run, then ignored.

## Milestones

### M1 — clip-aware pipeline, no posting (~1 week)
- SQLite schema + storage module; migrate flat dedupe ✅
- Multi-show prompt loader + `MODE` toggle ✅
- ffmpeg `scenedetect` parser (Rust) — shot bounds, no ML
- Silero VAD via `ort` (Rust ONNX) — silences for clip snapping + turn density
- Prosody features (Rust) — F0 via `aubio`, RMS via ffmpeg `astats`, speaking rate from word timestamps
- Embedding novelty via `fastembed` (Rust) — sentence-transformers MiniLM-L6-v2 in ONNX
- Word-timestamp aligner (Rust) — Groq `whisper-large-v3-turbo` client; `whisper-rs` offline fallback
- Linguistic markers extractor (Rust regex/NLP)
- Dense candidate generation, snapped to VAD + shot bounds
- LLM ranker (existing OpenAI-compatible model) with batched feature-injected prompts
- Render: ffmpeg cut + center-crop 9:16 + simple `.ass` captions + loudnorm 2-pass
- Digest email replaces per-variant emails; clips attached
- **Acceptance:** drop a 2-hour mp4 in via `LOCAL_VIDEO_PATH`, get a digest email with 8–12 ranked clips as 9:16 mp4 attachments

### M1.5 — context awareness (~2 days)
- GDELT 15-min news feed poller
- Reddit `r/all` hot poller
- Daily-cached `trends` table
- Ranker prompt augmentation with `current_trends`
- Per-show topic priors via `prompts/shows/{slug}/context_sources.txt`

### M2 — YouTube Shorts + Bluesky posting (~3 days)
- `platforms/` trait + first two implementations
- Per-platform SEO templates
- YouTube Data API v3 (reuses existing Google OAuth)
- Bluesky ATProto video upload
- Cost meter in digest email

### M3 — smart crop + karaoke captions (~4 days, GPU optional)
- SCRFD face detect via `ort` + simple IoU tracker — Rust
- Active speaker: rule-based primary (face presence ∩ VAD speaking) — Rust
- **Decision point:** if rule-based ASD quality is unacceptable, add Python sidecar here with Light-ASD; revisit then, not now
- One-Euro smoothed crop trajectory — Rust
- Karaoke `{\k}` `.ass` generator — Rust (already planned for M1)
- DeepFilterNet3 pre-stage via `deep_filter` crate — Rust
- emotion2vec — only if Python sidecar lands; otherwise skip
- AST audio-event scoring via `ort` — Rust

### M4 — LinkedIn + Threads posting (~3 days)

### M5 — TikTok + Instagram (audit-gated; submit during M1)
- Optional Ayrshare shim while waiting for direct approval

### M6 — feedback loop + hook overlays (~1 week)
- 24h/72h analytics pull per platform
- Ranker prompt augmentation with own-content CTR history
- LLM-generated 1.5s hook overlay text, burned to first 45 frames

### M7 — Lane B premium ranker (optional)
- Qwen3-VL-8B local or Qwen2.5-VL-72B via OpenRouter
- Used only for top-3 clips per episode

## Features beyond the pipeline

1. Cover-frame selection (reuse `thumbnail_windows` against clip window)
2. Hook overlay for the first 1.5s (M6)
3. Loudness consistency persisted per show
4. B-roll-aware cropping (skip face-track when no face for ≥0.5s)
5. Per-show prompt overrides
6. Veto via Gmail reply (`veto: clip_03`) → roll back post
7. Cost meter per job + rollup in digest

## Open decisions (resolved inline; defaults marked)

- **SEO-email mode preservation** — default: keep as opt-in `MODE=seo-only` env. (User to confirm.)
- **Multi-show config** — default: per-show overrides from day one. (User to confirm.)
- **GPU sidecar timing** — defer to M3; M1/M2 are CPU-only.
- **Auto-publish vs approval-gate** — M1/M2 are unlisted/private by default; auto-publish behind `AUTO_PUBLISH=true`, with the veto-reply path always available.
- **Branch strategy** — feature branch `feat/clipper` recommended; main stays current production.
