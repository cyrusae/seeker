# seeker v2 prototype

Job-search pipeline: structured feeds + manual submissions → local-model
scoring → selective paid-model escalation → human review with pitch/concerns
→ shortlist/applied tracking. Runs nightly, unattended, for multiple
applicants against their own profiles. Application materials (resume, cover
letter) are drafted in a separate chat session, not by this pipeline — that
gave better results and this stayed out of the token budget.

No agent loops, no cloud services. Every LLM call is single-shot and bounded;
cost is linear in jobs processed and capped by `MONTHLY_BUDGET_USD`.

## Architecture

```
sources (10 adapters — see Sources — or manual paste-or-URL)
   └→ jobs table (SQLite, status machine)
        pending_eval → [score: local Qwen] → [escalate: Haiku, if score ≥ threshold]
        → pending_user_review → (you shortlist/reject in web UI)
        shortlisted → applied ✓ / archived (tracked, not LLM-generated)

Alongside each cycle: closure detection (board-diff + out-of-band liveness
probe, see below) flags active jobs that quietly disappeared or expired.
```

Role → provider routing (`extract`/`score`/`escalate`) is all in `.env`
(`ROLE_SCORE=local`, `ROLE_SCORE=openrouter:...`, etc.), so different roles
can hit different providers in the same run. Two independent fallback layers
protect an unattended run: paid calls check the monthly budget first and drop
to local if it's exhausted; separately, any remote call that fails outright
(network error, 5xx, empty completion) retries once against local before
raising, so "OpenRouter/Anthropic happened to be down at 2am" degrades
quality instead of erroring out the whole cycle.

## Quickstart (laptop)

```sh
cd prototype
uv sync
cp .env.example .env        # fill in keys; leave Adzuna empty to start
                            # with manual submissions only

# Profiles: run docs/interview_spec.md through a chat session, save the JSON,
# or start from the example:
cp profiles/cyrus.example.json profiles/yourname.json   # then edit
uv run python cli.py import-profile profiles/yourname.json

uv run uvicorn app.main:app --reload
# open http://localhost:8000
```

- **Submit** — paste a job posting (or URL) from your phone; extraction + eval
  run automatically. This is the LinkedIn/Indeed path: paste text, don't scrape.
- **Review** — score, pitch ("why this one"), concerns, scam flag. Shortlist
  good matches to work from; reject with a note (notes feed the tuning report).
- **Jobs** — every job across every status, filterable, with bulk actions
  (re-evaluate, shortlist, dismiss selected) and per-job retry for anything
  that errored out mid-pipeline.
- **Companies** — the same jobs regrouped by employer instead of by status,
  with a cached LLM-generated diff summary when the same company has multiple
  postings, so you can see at a glance what's actually different between them.
- **Applications** — the shortlist you're actually working: copy the posting
  into a chat session to tailor materials, then mark **Applied ✓** or
  **Archive** if you change your mind. No LLM calls happen on this page.
- **Profiles** — upload a new/revised profile JSON, browse versions, pull a
  tuning report per applicant, all from the browser (no shell access needed
  once it's on the homelab).
- **Usage** — month-to-date spend vs. budget, per-role token counts, errors,
  last-cycle timing.
- **Status** — live run/stop controls, a progress bar during long eval passes
  (`X / Y evaluated this run`), recent source-adapter failures (rate limits,
  broken feeds), and the job-error list. The thing to check first if
  something seems off — see **Notifications** below for how it also reaches
  you without needing this tab open.

Nightly ingest+eval runs at `PIPELINE_HOUR`; a separate liveness probe runs
`PIPELINE_HOUR + 6`. "Run pipeline now" / "Stop pipeline" controls exist on
Usage and Status for out-of-band runs.

## Profile lifecycle

1. First profile: paste `docs/interview_spec.md` into any chat model, get JSON,
   `cli.py import-profile`.
2. Use the system; shortlist / reject-with-feedback / mark applied.
3. `uv run python cli.py tuning-report yourname > report.md`, take that back to a chat
   session, import the revised JSON. Versions are kept.
4. After a re-import, expand "re-score against updated profiles…" (Review page)
   or select jobs on the **Jobs** page → **Re-evaluate selected** to refresh
   old evals.

### What's in the tuning report

`cli.py tuning-report <applicant>` emits Markdown containing: the current
profile JSON; a score-distribution table by outcome; and every reviewed job
(APPLIED / SHORTLISTED / ARCHIVED-AFTER-INTEREST / REJECTED) with the model's score, pitch, concerns,
and your rejection feedback — i.e. both positive and negative examples, so the
refinement session can see where the model's scores disagreed with your
decisions. It deliberately includes only titles/pitches, never full posting
text, so it stays paste-sized no matter how long it runs.

### Updating profiles once it's on the homelab

`profiles/` is volume-mounted into the container, so from the homelab:

```sh
# copy the new profile JSON in (or edit in place), then:
docker compose exec seeker uv run python cli.py import-profile profiles/yourname.json
docker compose exec seeker uv run python cli.py tuning-report yourname > report.md
```

No restart needed — evals always read the latest imported version. (From your
laptop: `scp` the JSON to the homelab first, or run the whole thing over ssh.)

## Jev gate (in progress)

A cheaper, more inspectable "worth a look?" gate, being validated in shadow
before it replaces the DeepSeek score. `app/gate.py` asks Jev (a System One
model, via OpenRouter) one batch of narrow typed questions per job, built from
the profile: one per hard filter, anti-criterion and soft preference, plus role
fit, qualifications, seniority, thin posting, scam risk and passed deadline.
Policy is code: dealbreakers (plain-string `anti_criteria`/`hard_filters.other`
items, or `"severity": "dealbreaker"`) veto; everything else feeds a small
learned logistic blend whose output is the gate score (≈ chance you'd keep it).

- `GATE_MODE=shadow` scores every evaluated job alongside DeepSeek
  (`jobs.gate_score`, shown as "Jev NN" on cards); DeepSeek still decides.
- `python cli.py gate-backtest` scores your past decisions and writes
  `data/gate_backtest.md` (fair AUC vs DeepSeek, per-item firing rates).
- **Profiles → Gate** (or `cli.py gate-fit`) refits the blend from your
  decisions and previews what would change; nothing applies until you click
  Apply, and old versions can be rolled back. The nav shows a "refit" badge
  after ~30 new decisions. All Jev answers are cached in `gate_evals`, so
  refits are free unless the profile changed.
- `SPOT_CHECKS_PER_RUN` random below-the-floor jobs per applicant per cycle
  are surfaced as 🎲 spot checks, so the gate's training data isn't limited to
  jobs it already liked.

## Homelab migration

```sh
rsync -a prototype/ homelab:seeker/ && ssh homelab 'cd seeker && docker compose up -d --build'
```

SQLite + `data/` volume + `.env` is the entire state. Point `LOCAL_BASE_URL`
at wherever Ollama lives. Put it behind Tailscale/LAN — there's no auth.

## Sources

| adapter | auth | coverage | profile config key |
|---|---|---|---|
| `adzuna` | free API keys (.env) | general aggregator, all fields | `queries: [{what, where, max_days_old}]` |
| `usajobs` | free API key (.env) | US federal (lots of clerical/data entry, some remote) | `queries: [{keyword, location, remote}]` |
| `greenhouse` | none | tech/startup boards | `boards: [slug], title_include: […]` |
| `lever` | none | tech/startup boards | `companies: [slug], title_include: […]` |
| `ashby` | none | tech/startup boards | `orgs: [slug], title_include: […]` |
| `jibe` | none | iCIMS Talent Cloud "powered by Jibe" sites (Costco, REI, …) | `sites: [{host, company}], location, stretch_miles, title_include` |
| `workday` | none | Workday-hosted careers sites (unofficial CXS endpoints) | `tenants: [{tenant, shard, site, company, search, location_include}]` |
| `icims` | none (HTML scrape) | iCIMS hosted portals — one extra request per job for full text | `portals: [{host, company, keywords}]` |
| `govjobs` | none (HTML scrape) | governmentjobs.com/NEOGOV public-sector boards | `agencies: [{agency, company}], title_include, location_include` |
| `remoteok` | none | remote jobs (tech-leaning) | `search: [terms]` (required — unfiltered feed is a flood) |
| manual | — | anything (Indeed/LinkedIn: paste text or a URL) | Submit page |

`greenhouse`/`lever`/`ashby`/`jibe`/`remoteok` are "full enumerators" — each
run returns the board's complete current listing, which is what makes
board-diff closure detection possible for them (see below). The rest are
keyword/search APIs or per-job scrapers, where an absent job just means it
aged out of the search window, not that it closed.

## Closure detection

Two independent mechanisms flag active jobs (pending review or shortlisted,
not yet applied) that quietly went away, both with a 2-strike grace before
flagging so one missed poll doesn't false-positive:

- **Board-diff**, every ingest cycle: for full-enumerator sources, a stored
  job whose dedupe key is absent from a fresh, successful listing is
  genuinely gone. Reappearing on a later poll clears the flag.
- **Liveness probe**, daily at `PIPELINE_HOUR + 6`: for everything else
  (search APIs, per-job scrapers, manual submissions), one GET per active
  job's URL, checked for a 404/410 or a curated "no longer accepting
  applications" phrase. Network errors / 403 / rate-limits are treated as
  inconclusive, not closed.

Flagged jobs surface on Review for you to dismiss (mark "still applying
anyway") or let go; dismissal is permanent (`closure_dismissed`) so it won't
re-flag on the next sweep.

## Notifications

Off-screen visibility for a pipeline that's meant to run unattended — see
`app/notify.py`. Both are optional (env var unset = silent no-op) and stack
with the in-app Status tab:

- **Discord webhook** (`DISCORD_WEBHOOK_URL`) — human-readable pings: pipeline
  finished (with ingested/evaluated/closed counts), stopped early, crashed, or
  an LLM call fell back to local mid-run.
- **healthchecks.io** — dead-man's-switch heartbeats, one check each for the
  nightly pipeline (`HEALTHCHECKS_PIPELINE_URL`) and the liveness probe
  (`HEALTHCHECKS_LIVENESS_URL`), separate so one job's silence can't mask the
  other's. This is the one failure mode a "notify on error" webhook can't
  cover: the job not running at all.
- Falls back to a macOS notification (`osascript`) when neither is
  configured, so local dev keeps behaving the way it always did.

## Known scope cuts (deliberate)

- Anthropic Batches API (50% off) not used yet — everything is sync calls.
- No in-pipeline resume/cover-letter generation — a chat session, iterated
  interactively, produced better materials for less cost. The pipeline stops
  at shortlist/applied tracking; drafting happens outside it.
- No auth yet — see Homelab migration below.

## Description enrichment (Adzuna et al.)

Adzuna's API returns truncated descriptions (their limitation — full text sits
behind their `redirect_url`). Before scoring, any job whose stored text is
short (<1200 chars) and has a URL gets a full-text fetch + extraction; the
enriched text is persisted so the review card shows it too. If the fetch fails
(JS-walled board, login page), the scorer is told the text is a truncated
preview and applies the thin-posting rule: score capped at 45, missing info
named in concerns. Re-evaluating a job repeats the fetch, so jobs evaluated
before this existed can be fixed via Jobs → select → Re-evaluate.
