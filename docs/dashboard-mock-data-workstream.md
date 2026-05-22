# Dashboard mock-data → live-data workstream

A self-contained backlog for replacing every hardcoded number, fake list, and
no-op button on the [autoseo-dashboard](https://github.com/t3chnicallyinclined/autoseo-dashboard)
frontend with real data from the autoseo Rust backend. Audit performed
2026-05-21. Pick any batch and execute — each task is scoped to be independent.

## Two repos in play

| Repo | Path on dev box | Purpose |
|---|---|---|
| `autoseo` | `~/projects/autoseo` | Rust backend, axum API at `:8080`, WS at `/ws`, background worker, this doc |
| `autoseo-dashboard` | `~/projects/autoseo-dashboard` | React 19 + Vite frontend, builds to `dist/`. Autoseo serves it via a symlink at `autoseo/dashboard/dist`. |

To run the dev stack, see [DEV.md](../DEV.md).

## Current state recap (what's already live)

These pages render entirely real data and need no further work:
- **Clips.tsx** — full clip schema including variants, social copy, posts
- **Jobs.tsx** — list + filters from real DB
- **Library.tsx** — episodes + shows
- **Settings.tsx** — wired to `/api/config` (PATCH + PUT alias both accepted)
- **SetupWizard.tsx** — same config endpoint

These existing real endpoints can be used freely by new dashboard code:
- `GET /api/health`
- `GET/PATCH/PUT /api/config` and `POST /api/config/test/{service}`
- `GET /api/jobs` · `POST /api/jobs` · `GET /api/jobs/{id}` · `POST /api/jobs/{id}/retry|cancel|rerun` · `DELETE /api/jobs/{id}[?purge=true]`
- `GET /api/clips` · `GET /api/clips/{id}` · `POST /api/clips/bulk`
- `POST /api/clips/{id}/approve|veto|post`
- `GET /api/pipeline/status` — projects the most-recent job onto 8 pipeline stages
- `GET /api/cost` — sums `jobs.cost_cents`
- WebSocket at `/ws` — emits `job_update`, `job_complete`, `job_failed`. Event schema lives in [src/events.rs](../src/events.rs); the dashboard's matching consumer is `src/contexts/WebSocketContext.tsx`.

These endpoints exist as **stubs** today (return `[]` or canned values) and
are the lowest-hanging fruit to make real:
- `GET /api/shows`, `GET /api/episodes`, `GET /api/agents`
- `GET /api/trends`, `GET /api/analytics`
- `GET /api/platforms` — already half-real (reads env to infer connection status)

All stubs live in [src/api/stubs.rs](../src/api/stubs.rs).

## Audit summary — what's still mock

| Page | Tier | Mock data | Dead buttons |
|---|---|---|---|
| Dashboard | A | 4 stat cards, 3 sparkline arrays, "2 jobs in flight" text | "Live Pipeline" refresh |
| Pipeline | A | `stageDescriptions` const with 8 fake per-stage stats | "Refresh", per-row "Retry"/"View" |
| Analytics | A | 4 KPI cards, `ctrData`, `watchData` | Date-range buttons |
| SocialCopy | A | `platformCopy` const overrides real `clip.social` | "Save", "Regenerate" |
| Ranker | A/C | `scoreDistData`, `accuracyData` computable; `featureImportance`, `vlmRerank` need real ML | — |
| Trends | B | live topics fed by stub; "Trend-Clip Correlation" table hardcoded | "Refresh All" |
| Agents | C | `recentTasks` hardcoded | — |
| Schedule | C | `scheduled` array hardcoded | "Play now", "Cancel" |
| Shows | — | (data live) | "Add Show", per-show "Settings" |
| Platforms | — | (data live) | "Settings"/"Configure"/"Reconnect"/"Disconnect" — needs OAuth per platform |
| Jobs | — | (live) | "Ingest Media", "Retry Failed", per-row "View"/"Retry" |

Tier definitions:
- **A** — pure frontend; the backend already returns enough data, the page just isn't using it
- **B** — small new Rust endpoint or stub→real conversion
- **C** — new feature, needs schema/scheduler/ML work

---

# Batch 1 — Tier A: frontend-only wins

Estimated: 1–2 hours. One dashboard rebuild ships all of these. **No Rust
changes required.** All file paths are in `~/projects/autoseo-dashboard/`.

## 1.1 Dashboard.tsx stat cards + sparklines

**File:** `src/pages/Dashboard.tsx`

- Lines 84–86 — `sparkJobs`, `sparkViews`, `sparkCtr` hardcoded arrays. Compute from real data: bucket `jobsQuery.data` by day → counts, walk `clipsQuery.data[].views` over time, walk `clipsQuery.data[].ctr`.
- Line 231 — "Active Jobs" hardcoded "2" → use `jobsQuery.data.filter(j => j.status !== 'done' && j.status !== 'failed' && j.status !== 'cancelled').length` (same pattern already applied to `Pipeline.tsx:89`).
- Line 232 — "Clips This Week: 108" → `clipsQuery.data.filter(c => Date.now() - new Date(c.created).getTime() < 7*86400_000).length`.
- Line 233 — "Posts Published: 47" → sum `c.platforms[*]` where status === 'posted' across `clipsQuery.data`.
- Line 235 — "Avg CTR: 12.4%" → average `c.ctr` for clips where `c.views > 0`.
- Line 249 — "2 jobs in flight" text — reuse `activeCount` from the new pattern.

**Acceptance:** every stat card reflects the actual DB state. With 0 jobs and
0 clips, cards show 0 / 0% / no spark.

## 1.2 SocialCopy.tsx — use `clip.social` instead of mock

**File:** `src/pages/SocialCopy.tsx`

- Lines 11–27 — `platformCopy` const is a hardcoded object with sample YouTube/Bluesky/LinkedIn copy. Delete this and read from `selectedClip.social` (the `Clip` type already has a `social` field; backend already populates it via [src/clipper.rs](../src/clipper.rs) social-copy generator).
- Verify the dashboard's `Clip` type in `src/api/types.ts` includes the `social` shape. If it doesn't, extend the type to match the Rust `SocialCopy` struct in [src/social_copy.rs](../src/social_copy.rs).

**Acceptance:** when a clip's `social_copy` was generated during the run, its
real text shows up; when it wasn't, fields show empty (not the fake "Wait for
THIS clip..." sample text).

## 1.3 Pipeline.tsx — kill the fake stage descriptions

**File:** `src/pages/Pipeline.tsx`

Lines 11–25 — `stageDescriptions` const has baked strings like
`"Polled 12 emails, found 1 new attachment"` and `"12,480 tokens processed"`.

Two acceptable resolutions:
- **(easy)** Replace each `stats` string with `""` (empty). The architecture
  diagram still renders without per-stage detail.
- **(better)** Derive per-stage stats from the currently-running job. The data
  isn't perfect today, but reasonable approximations:
  - `ingest` → `job.created`
  - `transcribe` → `job.duration` if `status >= transcribed`
  - `rank` → `clipsQuery.data.filter(c => c.episodeId === jobId).length` candidates
  - `render` → `clip_renders` count (would need a new endpoint or include in `/api/jobs/{id}`)
  - `post` → posts where `status === 'posted'` for this job's clips

I'd ship the easy version first; the better version is a Tier-B follow-up.

## 1.4 Analytics.tsx KPI cards

**File:** `src/pages/Analytics.tsx`

- Lines 100–118 — 4 KPI cards (`Avg Watch %`, `Engagement`, `Rev. Equiv.`, `Cost/Clip`) hardcoded. Derive:
  - `Avg Watch %` → `mean(c.watchPct for c in clips if c.views > 0)`
  - `Engagement` → `sum(c.views for c in clips)` or similar; pick a definition and document
  - `Rev. Equiv.` → leave at 0 or remove the card until there's a real revenue source
  - `Cost/Clip` → `costData.total / clips.length` from `useCostData()` + `useClips()`
- Lines 30–35 (`ctrData`), 37–42 (`watchData`) — defer to Tier B; needs benchmarks endpoint.

## 1.5 Ranker.tsx histograms (the doable half)

**File:** `src/pages/Ranker.tsx`

- Lines 13–24 — `scoreDistData` (histogram of LLM scores) → compute from
  `clipsQuery.data.map(c => c.llmScore)` bucketed into 10-point bins.
- Lines 35–43 — `accuracyData` (per-episode trend) → group clips by episodeId,
  compute avg of `c.vlmScore - c.llmScore` per episode and treat as the delta.
- Lines 26–33 (`featureImportance`) and 45–51 (`vlmRerank`) — leave for Tier C.

## 1.6 Wire every "Refresh" button

Search the dashboard for `<Button.*Refresh|RefreshCw` and add `onClick`
handlers that call `queryClient.invalidateQueries({ queryKey: [...] })` for
the relevant query. Touched files: at least `Pipeline.tsx:91`, `Trends.tsx:32`,
likely `Dashboard.tsx` and `Analytics.tsx`.

## 1.7 Acceptance + ship

```bash
cd ~/projects/autoseo-dashboard
npm run build        # one rebuild ships all of 1.1–1.6
```

The autoseo binary's symlink picks up the new dist on the next page reload;
no Rust restart needed. Verify in browser: every Tier-A page now reflects
real state, including the zero-data case (clean install / after `DELETE`).

---

# Batch 2 — Tier B: turn stubs into real endpoints

Estimated: 2–4 hours. All changes in `~/projects/autoseo/src/api/stubs.rs`
and adjacent storage methods. One Rust rebuild + a fresh `nohup` of the
binary; no dashboard rebuild needed (existing hooks already wired).

## 2.1 `GET /api/trends` — return real trend rows

**File:** [src/api/stubs.rs](../src/api/stubs.rs) (the `trends` handler)

Today returns `{gdelt: [], reddit: [], google: []}`.

The `trends` table is already populated by the pollers in
[src/context/](../src/context/) (`gdelt.rs`, `reddit.rs`, `google_trends.rs`)
when their respective env flags are set. Query:

```sql
SELECT source, topic_id, label, score
FROM trends
WHERE fetched_at >= ?1  -- last 24h
ORDER BY fetched_at DESC
LIMIT 100
```

Group rows by `source` into the three keys the dashboard expects. Field shapes
to match the dashboard's `TrendingTopics` type in
`autoseo-dashboard/src/api/types.ts`:

```ts
interface TrendingTopics {
  gdelt: { topic, score, sources, tone, matched }[]
  reddit: { title, subreddit, score, comments }[]
  google: { term, volume, related }[]
}
```

Current Rust `trends` schema has `source/topic_id/label/score/fetched_at`. We
don't store `sources`/`tone`/`matched`/`comments`/`volume` per source — pick
sensible defaults (0/empty) until extractors are extended.

**Acceptance:** with the pollers off, returns empty arrays (not stub fixtures
either). With `GOOGLE_TRENDS_ENABLED=true` and at least one tick fired, real
topics appear.

## 2.2 `GET /api/analytics` — aggregate from clips + analytics tables

**File:** [src/api/stubs.rs](../src/api/stubs.rs) (the `analytics` handler)

Today returns `{views: [], topClips: []}`.

Build:
- `views` = time-series of `SUM(views)` from `analytics` table grouped by day
- `topClips` = top 10 by `views` joined back to `clips` for `hook` and `id`

Schema reference: [src/storage.rs](../src/storage.rs) `analytics` table has
`(clip_id, platform, fetched_at, views, ctr, watch_pct)`.

## 2.3 `GET /api/agents` — at least surface pipeline stage timings

**File:** [src/api/stubs.rs](../src/api/stubs.rs) (the `agents` handler)

There's no concept of "agents" in the backend today. Either:
- **(a)** Define each pipeline stage as an "agent" and return a synthetic list with last-seen-active timestamps derived from job logs. Cheap, gives the page something to render.
- **(b)** Ship the Agents page as a "future" placeholder with a clear "not implemented" empty-state.

Recommended: (a). The dashboard's `Agent` type wants `{ id, name, role, color, status, currentTask, elapsed, skills, tasksCompleted, avgDuration, successRate }`. Map each of the 8 pipeline stages to an agent; populate from existing data where possible.

## 2.4 `GET /api/trends/correlation` — new endpoint

**File:** [src/api/stubs.rs](../src/api/stubs.rs) — add a new route + handler.

Join `clips.trend_match` against the `trends` table; group by trend label;
return `{ trend, sources, matched_clip_count, avg_views, lift_pct }`.

`lift_pct` definition: `mean(views | trend_match=X) / mean(views | trend_match IS NULL) - 1`.

## 2.5 Ship

```bash
cd ~/projects/autoseo
cargo build --release
# kill + restart the running binary (see DEV.md), wait until current
# job (if any) completes first
```

Frontend doesn't need rebuilding — the hooks were already wired; they just
got `[]`-shaped responses before.

---

# Batch 3 — Tier C: real new features

These each warrant their own sub-spec; ship in separate PRs.

## 3.1 Schedule (queue + scheduler loop)

**What's needed:** new `schedule` table, scheduler loop alongside the worker,
4 endpoints (`GET /api/schedule`, `POST /api/schedule`, `POST /api/schedule/{id}/execute`, `DELETE /api/schedule/{id}`), wire the Schedule.tsx mock array to live data.

**Where to start:** define the schema first; everything else flows from it. A
minimal row: `(id, clip_id, platform, scheduled_at, status, executed_at, error)`. The scheduler loop can be ~50 lines in [src/worker.rs](../src/worker.rs).

## 3.2 Ranker analysis page

`featureImportance` and `vlmRerank` impact need an offline analysis pass. Two
options:
- **(a)** Compute on demand on each `GET /api/ranker/analysis` call. Slow at scale but accurate.
- **(b)** Materialize a `ranker_analysis` row at end of each job; serve from cache.

Specific metrics:
- Feature importance: regress (or correlate) the final `blended` score against each per-window feature in the manifest's `scores.reasoning` JSON
- VLM impact: for each clip, compute `vlm - llm` rank delta; aggregate

## 3.3 Agents (real version)

If 2.3's "pipeline stages as agents" is unsatisfying, add a real `agent_tasks`
table that captures every stage entry/exit, and surface it as the agents
table.

## 3.4 Platforms — OAuth flows

Each platform's "Configure" / "Reconnect" buttons need OAuth handling. Lots of per-platform variation. Defer until the dashboard is otherwise solid.

---

# How to run the dev stack

See [DEV.md](../DEV.md). TL;DR:

```bash
cd ~/projects/autoseo
set -a; source .env; set +a
# add any feature flags
export CLIP_TOP_K=5 AST_ENABLED=true VLM_RERANK_ENABLED=true
export MODE=server RUST_LOG=info
nohup ./target/release/autoseo > /tmp/autoseo.log 2>&1 &
disown
~/.local/bin/cloudflared tunnel --url http://localhost:8080  # optional public URL
```

Each batch above can be done locally without the tunnel; tunnel only needed
for testing on another device.

# Notes for the next session

- The autoseo Rust repo currently has substantial uncommitted work from
  2026-05-21 (items 8–11, the new job endpoints, ASD chain wiring, etc.).
  Status: 27 changed/new entries; see `git status` for the list.
- The autoseo-dashboard repo also has uncommitted local edits to Settings,
  SetupWizard, Pipeline, Clips, types, client, hooks, WebSocketContext. **Do
  not clobber** without rebasing your work first.
- `~/projects/autoseo/work/config.json` has masked secret values (`sk-••••…`).
  Real secrets live in `~/projects/autoseo/.env`; the launch script in DEV.md
  sources `.env` first so env wins over the masked config values.
- SCRFD ASD inference panics with the current default model (`cromsc/scrfd-10g`
  output tensor shape mismatch). Either find a SCRFD ONNX matching the
  inference assumptions in [src/face_detect.rs:168-200](../src/face_detect.rs#L168-L200)
  or rewrite the inference to introspect tensor shapes at runtime. Tracking
  separately from this workstream.
