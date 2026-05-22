# Dev workflow

How to run autoseo + the dashboard locally and expose them via a Cloudflare quick-tunnel for testing on any device.

This is the standard loop for testing pipeline features (ASD, AST, captions, VLM, etc.) — **do not invent new CLI flags**, configure everything from the dashboard's Settings page and create jobs from the New Job dialog.

## Repos

- `~/projects/autoseo` — the Rust backend (this repo). Pipeline + API + WS + worker.
- `~/projects/autoseo-dashboard` — React 19 + Vite frontend. Builds to `dist/`.

Both repos are pre-cloned side-by-side and have working `.env` files committed locally (not in git).

## One-time setup

```bash
# Symlink so the autoseo binary's default DASHBOARD_DIST resolves correctly.
mkdir -p ~/projects/autoseo/dashboard
ln -snf ~/projects/autoseo-dashboard/dist ~/projects/autoseo/dashboard/dist

# Install cloudflared (already installed at ~/.local/bin/cloudflared).
# https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
```

## Rebuild after Rust changes

```bash
cd ~/projects/autoseo
cargo build --release        # ~5–8 min cold, ~30s incremental
```

The release binary lands at `target/release/autoseo`.

## Rebuild dashboard after frontend changes

```bash
cd ~/projects/autoseo-dashboard
npm install                  # only when package.json changed
npm run build                # outputs dist/
# Symlink picks up the new dist automatically.
```

## Start the dev stack (single command per piece)

Three terminals (or screen/tmux):

```bash
# Terminal 1 — autoseo API + worker + WS, serves dashboard at /
cd ~/projects/autoseo
MODE=server target/release/autoseo
# default ports: 8080 (HTTP+dashboard), 9823 (WS)
```

`work/config.json` auto-imports its keys (OPENAI_API_KEY, HF_API_KEY, R2_*) into the process env at startup, so you don't have to `source .env`. The Settings page in the dashboard PATCHes that same file via `/api/config`.

```bash
# Terminal 2 — public tunnel for testing on any device
cloudflared tunnel --url http://localhost:8080
# Watch stdout for "https://<random-4-words>.trycloudflare.com" — that's the URL.
# Anyone with the URL can hit your API. Treat it as a public endpoint.
```

That's it. Open the tunnel URL in any browser; you'll see the dashboard. The API is at `<url>/api/*`.

### Skip step (faster iteration, no tunnel)

If you're only testing locally:
```bash
MODE=server target/release/autoseo
# Open http://localhost:8080
```

## Testing pipeline features via the dashboard

Everything that used to be a CLI env var should now be flipped from the dashboard:

| Feature | Where to toggle |
|---|---|
| `MODE` / `DIGEST_MODE` | Settings → Core |
| `SCRFD_ENABLED` (ASD dynamic crop) | Settings → Video |
| `AST_ENABLED` (audio events into ranker) | Settings → Audio |
| `ENHANCE_AUDIO` (DeepFilterNet3) | Settings → Audio (requires `--features enhance`) |
| `CAPTION_*` overrides | Settings → Captions |
| `VLM_RERANK_ENABLED` / `VLM_PREMIUM_*` | Settings → Ranker |
| Per-show prompts / caption JSON | `prompts/shows/{slug}/` (file system, not UI yet) |

Then create a job from the dashboard's New Job dialog (file upload or paste a video URL). The background worker picks it up and pushes WS events as it progresses; the Pipeline / Clips / Jobs pages update live.

## Inspecting outputs

- **Job-level**: dashboard Jobs page → click row for detail + clip summary. Also `GET /api/jobs/{id}`.
- **Clip-level**: dashboard Clips page. The Clip card shows `llmScore` and `vlmScore`.
- **A/B score lineage** (added in manifest schema v3): not yet surfaced in the UI; inspect `work/clipper/<job>/<ts>/clips/manifest.json` → `clips[N].scores`.
- **Captions / dynamic crop**: open the rendered `clip_*_9x16.mp4` in the dashboard's video player or directly from `work/`.
- **Cost summary**: dashboard Cost page (`GET /api/cost`).

## WebSocket — how it's wired

WS is served at **`/ws`** on the same port as the API (`:8080`). A single
cloudflared tunnel covers everything. There is no separate WS port anymore.

The dashboard's WS hook checks `window.__AUTOSEO_WS_URL` before falling back
to its hardcoded `ws://<host>:9090/ws` default. The Rust server patches the
served `index.html` with a tiny inline script that sets that global to
`{ws,wss}://<page-host>/ws`, so local *and* tunneled deployments both work
without any dashboard rebuild. If you ever bypass autoseo's HTML serving
(e.g. `npm run preview` directly), set `VITE_WS_URL` in the dashboard's
`.env` to point at the Rust server.

Pipeline events published today (see [src/events.rs](src/events.rs)):

- `job_update` — fires on every status transition (`pending`, `transcribing`,
  `rendering`, `done`, `failed`).
- `job_complete` — terminal Done; includes `clipsGenerated` count.
- `job_failed` — terminal Failed; includes the error string.

Hot path: `clipper::set_job_status` → `EventBus::emit`. The worker also
emits a final `JobFailed` fallback so a job that crashes before the clipper
gets a chance to publish still surfaces in the dashboard.

## Known caveats

- **Manifest A/B score lineage** (VLM standard/premium per-stage scores, added 2026-05-21) is in `manifest.json` but not yet surfaced in the dashboard's Clip cards — the dashboard's `Clip` type still has just `llmScore` / `vlmScore`. Until the dashboard adopts the new fields, read them from `work/clipper/<job>/<ts>/clips/manifest.json` directly.
- **Cloudflare quick-tunnel + HTTP/2 + curl** — testing the WS upgrade with `curl` against the tunnel needs `--http1.1`. Browsers do this automatically for `wss://`; this only matters when smoke-testing via curl. (`HTTP/2 400` over curl means HTTP/2 was negotiated and rejected the upgrade headers; force HTTP/1.1 to verify the server.)

## Troubleshooting

- **Dashboard 404s on `/`** — `dashboard/dist` symlink is broken or stale. Re-run `npm run build` in the dashboard repo.
- **Settings page can't save** — check that `work/config.json` is writable by the user running autoseo.
- **Tunnel URL stops working** — quick-tunnels are ephemeral. Restart `cloudflared`; you'll get a new URL.
- **Worker doesn't pick up new jobs** — check that `MODE=server` is set; the worker only spawns under that mode (see [src/main.rs](src/main.rs)).
- **Stale UI after a backend code change** — restart the autoseo binary; the dashboard hot-reloads its own JS but the Rust API doesn't.

## What NOT to do

- Don't run `cargo run -- --once --local-video-path …` to test pipeline features. The dashboard is the test harness; CLI runs bypass the worker, the WS event stream, and the dashboard's view of state.
- Don't start the dashboard's own `npm run dev:api` (Express server in `autoseo-dashboard/server/`). That's a separate Node backend reading the same SQLite file — the Rust API is the source of truth (see [README.md](README.md)).
- Don't push the `dashboard/dist` symlink to git. It's a local convenience.
