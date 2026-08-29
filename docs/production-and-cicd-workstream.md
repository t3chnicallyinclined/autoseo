# Production deployment & CI/CD workstream

A senior-DevOps workstream for taking autoseo from "running under nohup behind
a cloudflared tunnel on Tris's dev box" to a hybrid staging+production
deployment with a real CI/CD pipeline. Both repos are in play.

**Last audit:** 2026-05-25
**Current owner:** Tris (until handoff)
**Status:** Phase 0 not started. Existing Docker + CI artifacts inventoried.
**Related:** [DEV.md](../DEV.md) · [dashboard-mock-data-workstream.md](dashboard-mock-data-workstream.md)

---

## How to use this document

Each phase is a self-contained iteration that can be picked up cold by a new
agent or engineer. Every task has:

- **Files touched** — exact paths so you don't grep
- **Acceptance criteria** — measurable "DONE WHEN" so reviewers know when to merge
- **Verification** — the literal command(s) that confirm it works
- **Rollback** — how to undo if it goes sideways

Mark tasks with `[ ]` / `[x]` as you go. Append a note line under any task you
change so the next person sees what shifted.

---

## Two repos in play

| Repo | Path on dev box | Purpose | Runs in container? |
|---|---|---|---|
| `autoseo` | `~/projects/autoseo` | Rust backend (axum API at `:8080`, WS at `/ws`, background worker, ffmpeg renders, this doc) | Yes — primary container |
| `autoseo-dashboard` | `~/projects/autoseo-dashboard` | React 19 + Vite SPA. Built to `dist/`, served by autoseo as static files. | No — bundled INTO autoseo's image at build time via symlink at `autoseo/dashboard/dist` |
| `android-agent` | `~/projects/android-agent` | Browser-posting sidecar (FastAPI + CloakBrowser CDP). Referenced by `docker-compose.yml` but optional for the main pipeline. | Yes — secondary container, only when `BROWSER_POSTING_ENABLED=true` |

The dashboard is **bundled into** the autoseo image. There is no separate
dashboard service. This is a deliberate simplification — one container, one
deploy artifact, one version to roll back. It does mean every dashboard
change requires a full image rebuild.

---

## Current-state inventory (audit 2026-05-25)

### ✅ Already in place

| Area | What exists | File |
|---|---|---|
| Dockerfile (multi-stage) | `rust:1.88-bookworm` builder → `debian:bookworm-slim` runtime | [Dockerfile](../Dockerfile) |
| docker-compose | autoseo + cloakbrowser + browser_worker services, named volumes, `restart: unless-stopped` | [docker-compose.yml](../docker-compose.yml) |
| CI workflow | GitHub Actions `ci.yml` runs `cargo fmt --check`, `cargo check`, `cargo clippy -D warnings`, `cargo test`, `docker build` | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| Axum graceful shutdown | `axum::serve(...).with_graceful_shutdown(shutdown_signal())` — handles SIGTERM/SIGINT for HTTP layer | `src/api/mod.rs` (search for `with_graceful_shutdown`) |
| DB migrations | Idempotent via `PRAGMA user_version`, currently at v8 | [src/storage.rs](../src/storage.rs) `fn migrate()` |
| Schema versioned manifest | `manifest.json` schema_version: 4 (carries per-clip words for caption regen) | [src/clipper.rs](../src/clipper.rs) `build_manifest_json` |
| Bearer-token auth | `DASHBOARD_TOKEN` env gates all `/api/*` except `/api/health` + `/api/system` + `/api/fonts` | [src/api/auth.rs](../src/api/auth.rs) |
| Config snapshotting | Dashboard-set values persisted in `{WORK_DIR}/config.json`, loaded into env on boot | [src/main.rs](../src/main.rs) `load_config_json_into_env` |
| TypeScript strictness (dashboard) | `strict: true`, `noUnusedLocals`, `noUnusedParameters` enforced | `autoseo-dashboard/tsconfig.app.json:20-24` |
| Vitest backend tests (dashboard) | `server/**/*.test.ts` — legacy Express stand-in; verify whether still relevant | `autoseo-dashboard/vitest.config.ts` |

### ⚠️ Known gaps (each becomes a Phase task)

| Gap | Severity | Fix lives in |
|---|---|---|
| Dockerfile missing **fontconfig** — caption fonts fall back to DejaVu Sans in container even after `fc-cache` runs | HIGH (silent caption font regression in prod) | Phase 1.1 |
| Dockerfile missing **yt-dlp** — URL-ingest jobs fail; only file-upload works | HIGH (URL jobs are a primary ingest path) | Phase 1.1 |
| Dockerfile healthcheck runs `autoseo --help` — exits 0 even if server is wedged | MEDIUM (Docker thinks dead containers are alive) | Phase 1.2 |
| Worker loop has no shutdown signal — in-flight job aborts on every deploy | HIGH (current SOP is `pkill && start`; zombie jobs require manual DB cleanup) | Phase 0.1 |
| `/api/health` is a 1-line static "ok" — no DB/worker/model probes | MEDIUM (Docker healthcheck has nothing to consult) | Phase 0.2 |
| Logs are plain text to stderr — no JSON for log aggregation, no rotation | MEDIUM (works under journald, not parseable by Loki/Datadog) | Phase 0.3 |
| `DASHBOARD_TOKEN` cookie has no `Secure` flag — fine on localhost, leaks over plain HTTP in prod | HIGH (security gap when tunnel isn't TLS-only) | Phase 3.4 |
| Models lazy-download on first use — first prod job stalls 30-90s waiting for Silero/YuNet/embed | LOW (one-time per fresh container) | Phase 1.3 (optional bake-in) |
| No CI step pushes the image anywhere — `docker build` is just a lint | HIGH (no artifact to deploy) | Phase 2.2 |
| `.env` in `docker-compose.yml env_file: .env` — must be `chmod 600`'d on the host; no CI-injected secrets path | HIGH (current path: hand-edited file) | Phase 2.3 |
| No `engines` field in `autoseo-dashboard/package.json` — Node version drifts between dev and CI | LOW (npm doesn't enforce, but causes "works on my box") | Phase 1.4 |
| Dashboard tests run only on `server/**` (legacy Express). No frontend tests. | MEDIUM (relies on tsc strictness as the only quality gate) | Phase 0.4 |
| Dashboard bundles to single 1.1MB chunk — no code splitting | LOW (first-paint cost; not blocking) | Deferred to Phase 4 |
| No `.github/workflows/` in `autoseo-dashboard` — its CI is implicit (autoseo image build pulls the dashboard via symlink) | MEDIUM (dashboard PRs don't get tested independently) | Phase 2.1 |
| No backup story for `clipper.db` + `work/` volume — single point of failure | HIGH (data loss on disk death) | Phase 4.1 |

---

## Target state

```
┌──────────────────────────────────────────────────────────────────────┐
│                   GitHub Actions (CI/CD)                              │
│                                                                       │
│  PR opened    → ci.yml (test, lint, typecheck, dashboard build)       │
│  push to main → release.yml → build image → push GHCR → deploy staging│
│  git tag v*   → release-prod.yml → manual approval → deploy prod      │
└─────────────────────────┬─────────────────────────┬──────────────────┘
                          │                         │
                          ▼                         ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │   STAGING           │    │   PRODUCTION        │
              │   Tris's 32c box    │    │   Separate host     │
              │                     │    │   (Hetzner/Fly/etc) │
              │  - autoseo:staging  │    │  - autoseo:vX.Y.Z   │
              │  - cloudflared      │    │  - caddy (TLS)      │
              │  - separate .env    │    │  - separate .env    │
              │  - daily R2 backup  │    │  - daily R2 backup  │
              └─────────────────────┘    └─────────────────────┘
```

**Key principles**:

1. **One artifact, two environments.** Same Docker image runs in staging and
   prod. Only env files + DNS differ. Eliminates "works in staging" surprises.
2. **Staging gets every commit; prod gets tagged releases.** Push to main →
   staging deploys automatically. Tag `v1.2.3` + manual approval → prod
   deploys the same image promoted up.
3. **`:previous` for instant rollback.** Every deploy tags the outgoing image
   as `:previous` so a failed health check or human-spotted regression can
   roll back in one command.
4. **Secrets live on the host, never in CI.** GitHub Actions has SSH + GHCR
   creds; nothing more. Prod API keys are written to `/srv/autoseo/.env` on
   the prod host once, manually, and never touched again.
5. **Health gates everything.** Deploy script polls `/api/health/ready` for
   60s after restart. If not 200 in time → roll back automatically.

---

## Risk register

Ordered by impact × likelihood. Update whenever a risk is mitigated or new
ones appear.

| # | Risk | Impact | Likelihood | Mitigation phase |
|---|---|---|---|---|
| R1 | In-flight ffmpeg render killed by deploy → corrupted output + zombie DB row | High | High | Phase 0.1 (graceful SIGTERM) |
| R2 | Caption fonts fall back to DejaVu in container despite Settings → Captions picks | High | High | Phase 1.1 (fontconfig + fonts in image) |
| R3 | URL-ingest jobs fail in container (no yt-dlp installed) | High | High | Phase 1.1 (yt-dlp in image) |
| R4 | Secrets leak — `.env` committed to git or read by CI logs | Critical | Medium | Phase 0.5 (.gitignore + CI secret scoping) |
| R5 | Production DB lost — no backup; disk dies | Critical | Low | Phase 4.1 (nightly backups) |
| R6 | Deploy lands a broken image, no rollback path | High | Medium | Phase 2.4 (`:previous` tag + auto-rollback) |
| R7 | Schema migration fails mid-deploy → app starts but DB is in an undefined state | High | Low | Phase 0.6 (pre-flight migration test) |
| R8 | Cloudflared tunnel breaks → service inaccessible | Medium | Low | Phase 3.3 (Caddy as redundant ingress, or document the failure mode) |
| R9 | Cookie `DASHBOARD_TOKEN` exfiltrated over HTTP | Medium | Medium (only in non-TLS deployments) | Phase 3.4 (Secure flag + HSTS) |
| R10 | Dashboard PR merged with type errors → autoseo image bake fails 20 minutes later | Medium | Medium | Phase 2.1 (dashboard CI on its own repo) |
| R11 | Model lazy-download stalls first prod job 60-90s | Low | High (first job after every fresh container) | Phase 1.3 (bake models into image, or warm volume on first deploy) |
| R12 | Audio-track issues on android-agent sibling — version skew between repos | Medium | Medium | Document: pin android-agent commit in docker-compose. Phase 3.5. |

---

## Phases overview

```
Phase 0  ── Foundations (in-code prereqs)            ⏱  ~1 day
Phase 1  ── Containerization hardening               ⏱  ~half day
Phase 2  ── CI/CD pipeline (staging auto-deploy)     ⏱  ~half day
Phase 3  ── Production host bring-up                 ⏱  ~1 day
Phase 4  ── Observability + safety (backups, alerts) ⏱  ~1-2 days
```

Each phase is shippable on its own. Phase 0 unblocks everything else. Phase 1
must complete before Phase 2 (the image needs to be deployable). Phases 3 and
4 can run in either order once Phase 2 is green.

---

# Phase 0 — Foundations (in-repo code changes)

**Goal**: Make autoseo behave well when SIGTERM arrives, expose meaningful
health, structure logs for aggregation, and gate dashboard PRs on tests.

**Scope**: code changes only in both repos. No infra work. After Phase 0,
deploys still happen the old way (manual). What changes is that future
deploys won't kill jobs and humans will know what broke faster.

## 0.1 Graceful SIGTERM in the worker loop

The HTTP server already drains cleanly. The worker doesn't.

**Files touched:**
- `src/worker.rs` — `run()` function gains a `CancellationToken` parameter
- `src/main.rs` — server-mode boot creates the token, spawns a signal listener
- (Optional) `src/clipper.rs` — checkpoint between stages

**Implementation sketch:**

```rust
// main.rs (server mode)
use tokio_util::sync::CancellationToken;
let shutdown = CancellationToken::new();

let worker_shutdown = shutdown.clone();
tokio::spawn(worker::run(cfg.clone(), storage.clone(), bus.clone(), worker_shutdown));

let signal_shutdown = shutdown.clone();
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("SIGTERM received; draining");
    signal_shutdown.cancel();
});

// worker.rs run(...)
loop {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            tracing::info!("worker: shutdown signal received; finishing in-flight job");
            // No new claims. Active job already running in this fn's stack
            // will complete via the existing await chain.
            break;
        }
        _ = tick.tick() => { /* existing claim logic */ }
    }
}
```

**Acceptance:**
- [ ] `kill -TERM $(pgrep autoseo)` mid-render lets the current render finish (verify by tailing the log: render messages continue past the SIGTERM line, then "shutdown drained" appears).
- [ ] `set_job_status` for the in-flight job reaches `done` or `failed`, never `rendering` permanently.
- [ ] Exit code is 0.

**Verification:**
```bash
# Start a job, then mid-render:
pkill -TERM -f target/release/autoseo
# Should see in log:
#   INFO autoseo: SIGTERM received; draining
#   INFO autoseo::worker: shutdown signal received; finishing in-flight job
#   (render events continue...)
#   INFO autoseo: shutdown drained cleanly
# Process exits 0.
```

**Rollback:** Revert the commit. Worker reverts to "kill -9 city" behavior.

---

## 0.2 Split `/api/health/live` vs `/api/health/ready`

The existing `/api/health` is a 1-line static "ok". Splitting it gives Docker
+ the deploy script meaningful signals:

- `/api/health/live` — process is alive. Always 200 once HTTP server is up.
  Docker `HEALTHCHECK` consults this.
- `/api/health/ready` — process is ready to take traffic. 200 only when:
  - DB migrations have completed
  - The worker has entered its poll loop
  - Required model files for currently-enabled features exist on disk (or
    have been confirmed downloadable in <5s)
  - 503 with a JSON body listing which checks failed, otherwise

**Files touched:**
- `src/api/mod.rs` — add the two new routes
- `src/api/mod.rs` — add a `ReadinessState` field to `AppState` that the
  worker flips to `ready` once it enters its loop
- Keep `/api/health` as an alias of `/api/health/live` for backwards compat

**Acceptance:**
- [ ] `curl http://localhost:8080/api/health/live` returns 200 within 1s of process start
- [ ] `curl http://localhost:8080/api/health/ready` returns 503 immediately after start, 200 after worker loop is entered (typically within 2s)
- [ ] Body of `/api/health/ready` 503 explains why (e.g., `{"status":"not_ready","missing":["migrations","worker"]}`)

**Rollback:** Both new routes coexist with old one. Revert is safe.

---

## 0.3 Structured logging (JSON when `AUTOSEO_LOG_FORMAT=json`)

Plain text is fine under journald. Under a docker logging driver feeding into
Loki / Datadog / Cloudwatch, JSON is required for searchable fields.

**Files touched:**
- `src/main.rs` — `tracing_subscriber` init reads `AUTOSEO_LOG_FORMAT`
- `Cargo.toml` — `tracing-subscriber` already has `fmt::json` available; verify the `json` feature is enabled

**Implementation:**

```rust
let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
match std::env::var("AUTOSEO_LOG_FORMAT").as_deref() {
    Ok("json") => tracing_subscriber::fmt().json().with_env_filter(filter).init(),
    _ => tracing_subscriber::fmt().with_env_filter(filter).init(),
}
```

**Acceptance:**
- [ ] `AUTOSEO_LOG_FORMAT=json autoseo` emits one JSON object per log line
- [ ] Each line includes `timestamp`, `level`, `target`, `fields` (the structured args), `span` chain
- [ ] Plain-text mode unchanged when env unset

**Rollback:** Single match block — revert is safe.

---

## 0.4 Dashboard CI workflow

The dashboard has no `.github/workflows/`. Today it gets implicit testing
when autoseo's CI builds the Docker image (which sources the dashboard via
symlink). That fails 20 minutes into a build instead of in 3 minutes on the
PR. Fix it.

**Files touched (in `autoseo-dashboard` repo):**
- `.github/workflows/ci.yml` (new)
- `package.json` — add `engines` field for Node version pinning

**Workflow contents:**

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20', cache: 'npm' }
      - run: npm ci
      - run: npm run typecheck
      - run: npm run build
      - run: npm test
```

**Acceptance:**
- [ ] PR to `autoseo-dashboard/main` runs typecheck + build + tests
- [ ] Build failures block merge
- [ ] Bundle size is reported in PR comments (use `actions/github-script` to post a delta)

**Rollback:** Delete the workflow file. Repo reverts to "no CI."

---

## 0.5 Secrets hygiene

**Files touched:**
- `.gitignore` — confirm `.env` is listed in both repos
- `docs/SECRETS.md` (new) — document where every secret lives, who has access, how to rotate

**Tasks:**
- [ ] `grep -n "OPENAI_API_KEY\|GROK_API_KEY\|HF_API_KEY" $(git ls-files)` returns nothing in both repos
- [ ] `.env*` patterns added to both `.gitignore` files
- [ ] If any secret is already in git history: rotate it, document, and either rewrite history (small repo, no public clones) or accept the leak (rotate is sufficient)
- [ ] `docs/SECRETS.md` enumerates every API key with: name, where it's used, who has access, how to rotate, what breaks if it's missing

**Acceptance:**
- [ ] `git log -p | grep -iE "(api[_-]?key|password|token).{0,3}[:=].{0,3}['\"\s]?[a-zA-Z0-9_-]{20,}"` returns nothing recent
- [ ] `docs/SECRETS.md` exists and lists ≥10 known secrets (OPENAI, GROQ, HF, CF API, R2 access/secret, Bluesky, IG, Ayrshare, Google client/secret/refresh)

---

## 0.6 Migration safety net

The migration code in `storage.rs::migrate()` is idempotent, but there's no
CI check that ensures a migration doesn't break an existing DB.

**Files touched:**
- `tests/migration_smoke.rs` (new) — opens a fixture DB at schema_v1, runs
  migrate, asserts schema_v8 + the test data still reads correctly
- `.github/workflows/ci.yml` — add the test to the test step

**Acceptance:**
- [ ] A fixture DB at `tests/fixtures/clipper_v1.db` exists with a known schema and 1 row of test data
- [ ] `cargo test migration_smoke` succeeds: opens fixture, runs migrate, reads back v1 data via v8 schema
- [ ] CI runs this test on every PR

---

# Phase 1 — Containerization hardening

**Goal**: Make the existing Docker image actually deployable to a fresh host
with no surprises. Fix the known gaps in [Dockerfile](../Dockerfile).

## 1.1 Add fontconfig + yt-dlp + curl to runtime stage

**Files touched:**
- [Dockerfile](../Dockerfile) — runtime stage `apt-get install` line

**Diff:**

```diff
 RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         ca-certificates \
         ffmpeg \
+        fontconfig \
         libgomp1 \
+        curl \
+        python3 \
+        python3-pip \
+    && pip3 install --no-cache-dir --break-system-packages yt-dlp \
+    && fc-cache -f \
     && rm -rf /var/lib/apt/lists/*
```

**Acceptance:**
- [ ] `docker run --rm autoseo:dev fc-list | wc -l` ≥ 1 (fontconfig finds at least DejaVu)
- [ ] `docker run --rm autoseo:dev yt-dlp --version` prints a version
- [ ] After `POST /api/fonts/install {"family":"Bebas Neue"}`, `fc-list | grep -i bebas` succeeds inside the container
- [ ] A URL-ingest job (file=Drive share link) downloads inside the container

**Rollback:** Single Dockerfile diff — revert.

---

## 1.2 Real Docker healthcheck

Current `HEALTHCHECK CMD ["autoseo", "--help"]` always exits 0. Replace with
a curl against `/api/health/live` (which Phase 0.2 ships).

**Files touched:**
- [Dockerfile](../Dockerfile) — `HEALTHCHECK` line

**Diff:**

```diff
-HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
-    CMD ["/usr/local/bin/autoseo", "--help"]
+HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
+    CMD curl -fsS http://localhost:8080/api/health/live || exit 1
```

**Acceptance:**
- [ ] `docker inspect autoseo | jq '.[0].State.Health.Status'` shows `healthy` when running
- [ ] If the autoseo process is wedged (e.g., kill -STOP the PID inside container), within 90s Docker reports `unhealthy`

**Rollback:** Revert.

---

## 1.3 (Optional) Model pre-warming

Models (Silero VAD, YuNet, fastembed) lazy-download on first use. First job
on a fresh container stalls 30-90s.

**Two options:**

**Option A — Bake models into image** (recommended for prod, larger image):
- Add a `model-warmup` stage that downloads + caches Silero VAD, YuNet, and
  the embed model
- Final image ships ~150MB heavier but instantly ready

**Option B — Pre-warm volume on first deploy** (smaller image, slightly more ops):
- Deploy script runs `docker compose exec autoseo /usr/local/bin/autoseo --warm-models` once after the volume is provisioned

**Acceptance** (whichever option):
- [ ] First job after a fresh container starts the transcribe stage within 5s of clicking Start

**Decision**: TBD by ops preference. Default to Option B (smaller image).

---

## 1.4 Pin Node version + add dashboard build to compose

**Files touched (autoseo-dashboard):**
- `package.json` — add `engines: { "node": "20.x" }`

**Files touched (autoseo):**
- `Dockerfile` — already includes prompts but not dashboard. Add a dashboard-build stage:

```dockerfile
FROM node:20-bookworm-slim AS dashboard-build
WORKDIR /dashboard
COPY ../autoseo-dashboard/package*.json ./
RUN npm ci
COPY ../autoseo-dashboard/ ./
RUN npm run build
# /dashboard/dist now holds the SPA
```

Then in the runtime stage:

```dockerfile
COPY --from=dashboard-build /dashboard/dist /app/dashboard/dist
ENV DASHBOARD_DIST=/app/dashboard/dist
```

**Catch**: docker can't reference a sibling repo by default. Two options:
- A. CI runs `cp -r ../autoseo-dashboard .` before `docker build .` to make the dashboard a build-context subdir
- B. Switch to a monorepo OR move the dashboard into a git submodule

Recommendation: **A** for now (CI step), revisit if dashboard becomes its
own deploy unit later.

**Acceptance:**
- [ ] `docker run --rm -p 8080:8080 autoseo:dev` serves the dashboard at `http://localhost:8080/` (not just `/api/*`)
- [ ] Visiting `/` returns the dashboard's `index.html`

---

## 1.5 Compose smoke test in CI

The existing CI builds the image but never starts it. Add a stage that runs
`docker compose up -d`, waits for healthy, hits `/api/health/ready`, then
tears down.

**Files touched:**
- `.github/workflows/ci.yml` — append a `compose-smoke` job

**Acceptance:**
- [ ] CI fails if the image starts but doesn't reach `/api/health/ready` within 60s
- [ ] CI uploads the container's log on failure (as a workflow artifact)

---

# Phase 2 — CI/CD pipeline (staging auto-deploy)

**Goal**: Push to main → image gets built, pushed to GHCR, deployed to the
current dev box (which becomes staging).

## 2.1 Dashboard repo: bundle in autoseo CI

Until Phase 0.4 dashboard CI lands, autoseo's image build is the only place
dashboard regressions surface. Add a step in autoseo's CI that clones the
dashboard repo at a pinned commit before `docker build`.

**Files touched:**
- `.github/workflows/ci.yml` — pre-build step
- `.github/workflows/release.yml` (new) — same step

**Pinning strategy:** Track dashboard via a `DASHBOARD_REF` file at repo
root containing a git SHA. Bumping requires a PR. Eventually replaced by
the dashboard's own CI publishing its dist as a release asset.

**Acceptance:**
- [ ] `DASHBOARD_REF` file exists with a valid SHA
- [ ] CI clones dashboard at that SHA, builds it, copies dist into autoseo
- [ ] Bumping the SHA in a PR shows the dashboard rebuild step in CI logs

---

## 2.2 GHCR publish on `main` push

**Files touched:**
- `.github/workflows/release.yml` (new)

**Workflow shape:**

```yaml
name: Release
on:
  push:
    branches: [main]
permissions:
  contents: read
  packages: write   # push to ghcr.io
jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # ...clone dashboard at DASHBOARD_REF...
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build & push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/<owner>/autoseo:sha-${{ github.sha }}
            ghcr.io/<owner>/autoseo:staging
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**Acceptance:**
- [ ] Image `ghcr.io/<owner>/autoseo:sha-<shortsha>` appears in GHCR after main push
- [ ] Image also tagged `:staging`
- [ ] Image build takes <8 minutes (with cache)

---

## 2.3 Deploy to staging via SSH

**Files touched:**
- `.github/workflows/release.yml` — add a `deploy-staging` job after `build-and-push`
- `~/srv/autoseo/deploy.sh` on the staging box (new)
- GitHub repo secrets: `STAGING_SSH_HOST`, `STAGING_SSH_USER`, `STAGING_SSH_KEY`

**`deploy.sh` shape:**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /srv/autoseo
docker compose pull autoseo
# Snapshot the current image tag for rollback
docker tag ghcr.io/<owner>/autoseo:staging ghcr.io/<owner>/autoseo:previous || true
docker compose up -d --no-deps autoseo
# Health gate
for i in {1..30}; do
  if curl -fsS http://localhost:8080/api/health/ready >/dev/null; then
    echo "deploy OK"; exit 0
  fi
  sleep 2
done
echo "deploy FAILED — rolling back"
docker tag ghcr.io/<owner>/autoseo:previous ghcr.io/<owner>/autoseo:staging
docker compose up -d --no-deps autoseo
exit 1
```

**Workflow step:**

```yaml
- name: Deploy to staging
  uses: appleboy/ssh-action@v1
  with:
    host: ${{ secrets.STAGING_SSH_HOST }}
    username: ${{ secrets.STAGING_SSH_USER }}
    key: ${{ secrets.STAGING_SSH_KEY }}
    script: bash /srv/autoseo/deploy.sh
```

**Acceptance:**
- [ ] Pushing a commit to main triggers an end-to-end deploy that lands within ~10 minutes
- [ ] Failed health gate triggers automatic rollback (forced bad image confirms)
- [ ] `cloudflared` tunnel URL serves the new build immediately

---

## 2.4 Rollback playbook

Even with auto-rollback in deploy.sh, sometimes humans need to roll back from
a bad release deliberately.

**Files touched:**
- `docs/RUNBOOK-rollback.md` (new)

**Content outline:**
- "How to roll back manually" — SSH to staging/prod, `docker tag previous → staging`, `docker compose up -d`
- "How to roll forward" — re-tag and re-deploy
- "When NOT to roll back" — if a DB migration ran (schema change), rollback may corrupt — must run a counter-migration

---

# Phase 3 — Production host bring-up

**Goal**: Stand up a production host (separate from Tris's dev box).
Mirror the staging setup, with TLS, tighter security, and tag-based deploys.

## 3.1 Pick + provision host

**Decision needed:**

| Option | Cost | Notes |
|---|---|---|
| Hetzner Cloud CCX23 (8 vCPU, 32GB, dedicated) | ~$50/mo | Good ffmpeg perf, EU |
| Fly.io machines (8x shared, 16GB) | ~$60/mo | App platform, slightly more abstraction |
| DigitalOcean Premium Intel 8c/16GB | ~$80/mo | Familiar, US |
| Hetzner dedicated AX41 (6c Ryzen, 64GB) | ~$40/mo + setup | Best perf/$ but bare-metal ops |

Recommendation: **Hetzner Cloud CCX23**, US or EU based on Tris's audience.
Can swap up/down without reprovision.

**Acceptance:**
- [ ] Host reachable via SSH on port 22 (or non-standard port — bonus security)
- [ ] Docker + docker-compose installed
- [ ] Non-root user `autoseo` with sudo for `docker` commands

---

## 3.2 Production secrets + .env

**Files touched (on the prod host, not in git):**
- `/srv/autoseo/.env`

**Procedure:**
1. SSH to prod host as `autoseo` user
2. `cd /srv/autoseo`
3. `cp .env.example .env`
4. Paste production values (different Groq key, different R2 bucket if applicable, different DASHBOARD_TOKEN)
5. `chmod 600 .env`
6. Document the rotation procedure in `docs/SECRETS.md`

**Acceptance:**
- [ ] `/srv/autoseo/.env` exists, 0600 mode, owned by `autoseo`
- [ ] `docker compose config` resolves without warnings
- [ ] No secrets in any GitHub repo or Action log

---

## 3.3 Caddy as TLS terminator (replaces cloudflared for prod, optional)

For prod, a dedicated reverse proxy with auto-TLS via Let's Encrypt is more
robust than relying on cloudflared. Cloudflared is fine as a backup tunnel
for dev/staging.

**Files touched:**
- `/srv/autoseo/Caddyfile` (on prod host)
- `docker-compose.yml` — add a `caddy` service (only in prod compose override)

**Caddyfile:**

```caddy
autoseo.example.com {
  reverse_proxy autoseo:8080
  encode gzip
  header {
    Strict-Transport-Security "max-age=63072000"
    X-Content-Type-Options nosniff
    X-Frame-Options DENY
  }
}
```

**Acceptance:**
- [ ] `https://autoseo.example.com/api/health/live` returns 200 with valid Let's Encrypt cert
- [ ] HTTP redirects to HTTPS
- [ ] `curl -I https://autoseo.example.com` shows `Strict-Transport-Security`

---

## 3.4 Cookie Secure flag for prod

The dashboard's `autoseo_token` cookie currently lacks `Secure`. Over HTTPS
that's a leak.

**Files touched:**
- `autoseo-dashboard/src/lib/auth.ts` — set Secure when `window.location.protocol === "https:"`

**Diff:**

```diff
-document.cookie = `autoseo_token=${token}; SameSite=Lax; Max-Age=2592000; path=/`
+const secure = typeof window !== "undefined" && window.location.protocol === "https:" ? "; Secure" : ""
+document.cookie = `autoseo_token=${token}; SameSite=Lax; Max-Age=2592000; path=/${secure}`
```

**Acceptance:**
- [ ] Cookie set over HTTPS includes `Secure` (verify in DevTools → Application → Cookies)
- [ ] Cookie set over HTTP (localhost dev) omits `Secure`

---

## 3.5 Pin android-agent commit in compose

Today docker-compose references `../android-agent` with no version pin. A
re-clone could grab a breaking commit.

**Files touched:**
- `docker-compose.yml` — switch from local build to a pinned image (CI side) OR document the pinned SHA in a `ANDROID_AGENT_REF` file

**Acceptance:**
- [ ] `ANDROID_AGENT_REF` file exists with a known-good commit
- [ ] CI/deploy uses that SHA when building the browser_worker image (or skips it for prod if browser-posting isn't enabled there)

---

## 3.6 Production release workflow (tag-based)

**Files touched:**
- `.github/workflows/release-prod.yml` (new)

**Trigger:** `git tag v*.*.* && git push --tags`

**Workflow:**
1. Reuse the staging image build (no rebuild — same artifact promotes up)
2. Re-tag `ghcr.io/<owner>/autoseo:sha-<sha>` as `:vX.Y.Z` and `:production`
3. GitHub Environment `production` requires manual approval (configure in
   repo Settings → Environments)
4. SSH to prod, pull, `compose up`, health-check, rollback on failure

**Acceptance:**
- [ ] `git tag v1.0.0 && git push --tags` triggers the workflow
- [ ] Approval gate in GitHub UI blocks until a reviewer approves
- [ ] After approval, prod deploy completes in <5 minutes
- [ ] `https://autoseo.example.com/api/system` reflects the new version

---

# Phase 4 — Observability + safety

**Goal**: When something goes wrong in prod, we know quickly. When a disk
dies, we can restore from backup.

## 4.1 Nightly backups

**Files touched:**
- `/srv/autoseo/scripts/backup.sh` (new, on prod host)
- Crontab entry: `0 3 * * * /srv/autoseo/scripts/backup.sh`

**Procedure per night:**
1. `sqlite3 /srv/autoseo/work/clipper.db ".backup /tmp/clipper.db.bak"` — atomic snapshot
2. Tar `/srv/autoseo/work/clipper.db.bak` + `/srv/autoseo/work/config.json` + (optional) recent clip metadata
3. Upload to a dedicated R2 bucket `autoseo-backups` with date prefix
4. Retention: keep 30 daily + 12 monthly + indefinite annual

**Acceptance:**
- [ ] Backup runs at 03:00 prod-local time
- [ ] R2 bucket has at least 7 days of backups after the first week
- [ ] Restoration runbook tested: down a test container, restore the backup, verify clip count

---

## 4.2 `/api/metrics` Prometheus endpoint

Surface counters + gauges for things ops cares about:

- `autoseo_jobs_total{status="done|failed|pending"}` — counter
- `autoseo_clips_total` — counter
- `autoseo_active_jobs` — gauge
- `autoseo_stt_requests_total{provider="groq|openai"}` — counter
- `autoseo_render_duration_seconds{variant="9x16|1x1|16x9"}` — histogram
- `autoseo_cost_cents_total` — counter

**Files touched:**
- `src/api/metrics.rs` (new) — Prometheus encoding via `prometheus` crate
- `src/api/mod.rs` — route `/api/metrics`

**Acceptance:**
- [ ] `curl https://autoseo.example.com/api/metrics` returns a Prometheus-format scrape
- [ ] A test Prometheus container can scrape and graph the metrics

---

## 4.3 Alerting (lightweight)

Self-hosted alerting can wait. Start with **uptime + cost alarms** via a
free external probe:

- UptimeRobot or Better Uptime monitoring `https://autoseo.example.com/api/health/ready` every 5 minutes
- Email/SMS on >2 consecutive failures
- Optionally: a daily cost report via a Cloudflare Worker that hits `/api/cost` and emails if >$N

**Acceptance:**
- [ ] An external probe reports green for /api/health/ready
- [ ] A simulated outage (stop the container) triggers an alert within 15 minutes

---

## 4.4 Crash logging / error tracking

Today, errors live in `tracing` output. For prod, send `ERROR`-level events
to a sink:

- Option A: Loki via the docker logging driver (self-hosted)
- Option B: Sentry via the `sentry-tracing` crate (managed, free tier OK)

Recommendation: **B** for simplicity. Sentry's free tier covers a single-app
project comfortably.

**Files touched:**
- `Cargo.toml` — add `sentry-tracing`
- `src/main.rs` — init Sentry when `SENTRY_DSN` env is set; pass-through if not

**Acceptance:**
- [ ] Errors emitted via `tracing::error!` appear in the Sentry project within 30s
- [ ] PII scrubbing configured (no API keys leak via spans)

---

# Open questions / decisions needed

Items where we need a human to pick before the work proceeds.

| # | Question | Blocks | Default if not answered |
|---|---|---|---|
| Q1 | Which production host? Hetzner / Fly / DO / dedicated? | Phase 3.1 | Hetzner Cloud CCX23 |
| Q2 | Monorepo or pinned dashboard SHA? | Phase 2.1 | Pinned SHA via `DASHBOARD_REF` file |
| Q3 | Bake models into image or warm volume on first deploy? | Phase 1.3 | Warm volume (smaller image) |
| Q4 | Caddy or keep cloudflared for prod TLS? | Phase 3.3 | Caddy (TLS local + cloudflared as backup tunnel) |
| Q5 | Sentry (managed) or Loki (self-hosted) for error tracking? | Phase 4.4 | Sentry (managed) |
| Q6 | Domain for production? | Phase 3.3 | TBD by user |
| Q7 | Do we keep the `android-agent` sibling required for the autoseo image, or make it optional? | Phase 3.5 | Optional — pin the SHA but skip building it if `BROWSER_POSTING_ENABLED=false` |

---

# Decisions made (ADR-style log)

Append entries here when a decision is made. Format:

```
## ADR-N: <title>
Date: <YYYY-MM-DD>
Status: accepted | superseded
Context: <one paragraph>
Decision: <one paragraph>
Consequences: <bullet points>
```

### ADR-1: Hybrid host model (staging on dev box, prod on separate machine)
Date: 2026-05-25
Status: accepted
Context: Tris's dev box (32-core / 128GB) has been serving as the autoseo
host via `nohup`. We want production isolation without scrapping that
hardware investment.
Decision: Keep the dev box as the staging environment. Provision a separate,
smaller production host. Both run the same Docker image; only `.env`
differs.
Consequences:
- Staging gets every push to main automatically
- Production gets tagged releases with manual approval
- We can break staging without affecting live work
- Dev work continues to happen against staging
- One extra host to maintain

### ADR-2: Bundle dashboard into autoseo image
Date: 2026-05-25
Status: accepted
Context: Dashboard is a static SPA. Could deploy separately (S3 + CloudFront)
or bundled into autoseo. Bundling means one container, one version, one
rollback target. Separate deploy needs CORS + WS-cross-origin work.
Decision: Bundle dashboard into autoseo's image at build time. Use a
`DASHBOARD_REF` SHA pin in the autoseo repo to control which dashboard
commit gets bundled.
Consequences:
- One artifact, one deploy
- Dashboard release cadence ties to autoseo's
- Image size grows by ~3MB (negligible)
- CORS is moot — same origin

---

# Glossary

- **Staging** — the environment that gets every main-branch commit auto-deployed. Currently Tris's 32-core dev box.
- **Production** — the environment that gets manually-approved tagged releases. To be provisioned.
- **GHCR** — GitHub Container Registry (`ghcr.io`), the image host.
- **Image tag** — version label on a Docker image. Naming convention: `:sha-<short>`, `:staging`, `:production`, `:vX.Y.Z`, `:previous`, `:latest`.
- **Health gate** — automated `/api/health/ready` poll the deploy script does post-restart. 60s timeout, auto-rollback on failure.
- **Graceful drain** — the period after SIGTERM during which the worker finishes its in-flight job and the HTTP server finishes active requests. ≤5 minutes.

---

# References

- Existing artifacts in autoseo:
  - [Dockerfile](../Dockerfile)
  - [docker-compose.yml](../docker-compose.yml)
  - [.github/workflows/ci.yml](../.github/workflows/ci.yml)
  - [DEV.md](../DEV.md)
- Existing artifacts in autoseo-dashboard:
  - `package.json`
  - `vite.config.ts`
  - `tsconfig.app.json`
- Related workstreams in this repo:
  - [dashboard-mock-data-workstream.md](dashboard-mock-data-workstream.md) — frontend data-realness backlog
- External docs to read before starting Phase 3:
  - [Hetzner Cloud docs](https://docs.hetzner.com/cloud/)
  - [Docker Compose production guidance](https://docs.docker.com/compose/production/)
  - [Caddy auto-TLS](https://caddyserver.com/docs/automatic-https)

---

# Changelog

Maintain a one-line entry per change to this document. The most recent
audit-of-state above should match the date of the last entry.

| Date | Change | By |
|---|---|---|
| 2026-05-25 | Initial draft. Inventory of current state + Phase 0–4 plan. | Tris + agent |
