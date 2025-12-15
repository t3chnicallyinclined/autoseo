# autoseo (MVP)

Polls Gmail for Google Drive “Item shared with you” emails, downloads the shared video from Drive, extracts/chunks audio with `ffmpeg`, transcribes via an OpenAI-compatible API, asks an OpenAI-compatible LLM for a YouTube SEO package, generates a few thumbnail screenshots, then sends the results back via Gmail API.

## What you need installed

- Rust toolchain
- `ffmpeg` and `ffprobe` on PATH

## Google setup (OAuth Playground bootstrap)

You cannot use an API key to read a private Gmail inbox or download private Drive files. You need OAuth.

1. Create a Google Cloud project
2. Enable APIs:
   - Gmail API
   - Google Drive API
3. Configure OAuth consent screen (Testing is fine) and add your Google account as a test user.
4. Create OAuth credentials:
   - OAuth Client ID (type: **Web application** or **Desktop app**)
   - Copy the **Client ID** and **Client secret**.
5. Use **OAuth 2.0 Playground** to get a refresh token:
   - Click the gear icon and set **OAuth client** to your client id/secret
   - Authorize scopes (minimum for this MVP):
     - `https://www.googleapis.com/auth/gmail.readonly`
     - `https://www.googleapis.com/auth/gmail.send`
     - `https://www.googleapis.com/auth/drive.readonly`
   - Exchange authorization code for tokens
   - Copy the **refresh_token** (this is your long-lived secret)

## OpenAI-compatible setup

You need an OpenAI-compatible base URL + API key:

- `OPENAI_BASE_URL` (default: `https://api.openai.com`)
- `OPENAI_API_KEY`

This MVP uses:
- `POST /v1/audio/transcriptions` with `response_format=verbose_json`
- `POST /v1/chat/completions` (or `POST /v1/responses` for `gpt-5*`) for SEO rich-text output
- `POST /v1/chat/completions` with `response_format={"type":"json_object"}` for thumbnail timestamp selection

If your provider differs, we can adapt.

## Configure

Set env vars:

- `RESULT_TO` (where the SEO package + thumbnails are sent)
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`
- `OPENAI_API_KEY`

Optional:
- `GMAIL_QUERY` (default: `from:drive-shares-dm-noreply@google.com subject:"Item shared with you" has:drive`)
- `GMAIL_MAX_RESULTS` (default: `10`)
- `REQUIRE_VIDEO` (default: `true`, skips non-video Drive files like old `.wav` shares)
- `OPENAI_CHAT_MODEL` (default: `gpt-5.2-pro-2025-12-11`)
- `OPENAI_STT_MODEL` (default: `whisper-1`)
- `SEO_VARIANTS` (default: `3`)
- `SEO_VARIANTS_PROMPT_PATH` (default: `./prompts/seo_variants.txt`)
- `POLL_INTERVAL_SECS` (default: `60`)
- `WORK_DIR` (default: `./work`)
- `DEDUPE_FILE` (default: `./work/processed_message_ids.txt`)
- `AUDIO_CHUNK_SECS` (default: `900`)
- `THUMBNAIL_SLOTS` (default: `5`)
- `THUMBNAIL_COUNT` (default: `10`)
- `THUMBNAIL_MAX_HEIGHT` (default: `1080`; set `0` for native-resolution grabs)

## Run

One poll cycle:

```bash
cd /home/tris/projects/autoseo
export RESULT_TO="you@example.com"
export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
export GOOGLE_REFRESH_TOKEN="..."
export OPENAI_API_KEY="..."

cargo run -- --once
```

Dry-run (no OpenAI + no email send)

Dry-run only needs Google credentials (it lists Gmail, extracts Drive file IDs, and prints Drive metadata). It does **not** require `OPENAI_API_KEY` or `RESULT_TO`.

```bash
cd /home/tris/projects/autoseo
export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
export GOOGLE_REFRESH_TOKEN="..."

cargo run -- --once --dry-run
```

Continuous poller:

```bash
cargo run
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

Logs / ops:

```bash
docker logs -f --tail=200 autoseo
docker ps --filter name=autoseo
docker restart autoseo
```

## Notes

- Dedupe is file-backed but not a real job DB. If the process restarts, it won’t reprocess messages already recorded in `DEDUPE_FILE`.
- The standard Google Drive share email contains a direct link like `https://drive.google.com/file/d/<fileId>/view?...` which is what we parse.
- If you want to narrow matching further, set:
   - `GMAIL_QUERY=from:drive-shares-dm-noreply@google.com subject:"Item shared with you" has:drive`
- For 1–3 hour videos, transcription uploads are done as small audio chunks (default 15 minutes).
