# Jev gate: status, decisions, and what's next

The project in one sentence: **replace the DeepSeek "score every job"
call with Jev, a cheaper model that returns typed judgments, as the "is this
worth looking at?" gate, and only pay a talky model to write a gloss for jobs
above a (lower) threshold.**

Started 2026-10-01. This file is the running record; update the Status and
Next steps sections when something moves.

---

## Status (as of 2026-10-01)

| Piece | State |
|---|---|
| Review-queue freshness (`last_seen_at`, stale view/status, age sort) | **Done**, live |
| Jev gate module, backtest, `gate-fit`, Gate page | **Done** |
| `GATE_MODE=shadow`: new jobs get a Jev score next to DeepSeek's | **On** (DeepSeek still decides) |
| Spot checks (3 random skipped jobs/applicant/cycle) | **On** |
| Gate model in use | **Built-in default (v0).** Candidate v2 is sitting unapplied on the Gate page; it was fit before the profile review, so don't apply it. |
| Profile full reviews (Cyrus, Martin) | **Next: in progress on your side** |
| Live mode (gate decides status, gloss step, gloss-on-open) | **Not built** |

## Next steps

1. **Profile full reviews.** Profiles → download tuning report → chat session
   with `docs/interview_spec.md` in **Full review** mode. Bring the
   per-criterion table; see "Evidence" below for what it already shows.
   Martin's review should be done with Martin.
2. **Import, then refit.** Import the new profiles, then Profiles → Gate →
   Preview refit with "first ask Jev" ticked (~$0.40; past decisions get
   answers under the new wording). Check "Dealbreakers that fire on jobs you
   kept" (should be near zero now) and the AUC. Apply.
3. **Shadow period.** Review normally for 2–3 weeks or ~100+ decisions,
   including the 🎲 spot checks. Refit when the nav shows the "refit" badge.
4. **Go/no-go for live mode.** Re-run `cli.py gate-backtest` and check:
   - Jev's AUC ≥ DeepSeek's on the fair comparison (jobs DeepSeek scored),
     for **each** applicant, not just overall.
   - Review floor keeps ≥ ~97% of kept jobs that weren't vetoed.
   - No dealbreaker fires on more than ~5% of kept jobs.
   - Spot checks: you rarely keep one (if you often do, the floor is too high).
5. **Build live mode** (see "Live mode design" below).

## Evidence so far

From two backtests over ~1,165 past decisions (151 kept, ~1,015 rejected),
plus a ~300-job sample of auto-skipped jobs and the 466-job backlog.

- **Cost:** Jev costs ~$0.00025/job vs DeepSeek's ~$0.0008. The state (the
  profile plus the posting) is billed once per request; each extra question
  adds ~28 tokens, so one question per profile item is nearly free.
- **Ranking quality (AUC, fair comparison: only jobs DeepSeek actually
  scored, n=466):**

  | | Jev | DeepSeek |
  |---|---|---|
  | all | 0.701 | 0.712 |
  | cyrus | 0.736 | 0.715 |
  | martin | **0.626** | 0.761 |

  With "over 4 years of experience" (Cyrus) and "regional or operational
  travel" (Martin) changed from dealbreakers to penalties: **0.736 vs 0.712**
  overall, and the share of kept jobs that reach review goes from 80% to 93%.
- **The unfair comparison to avoid:** 700 of the rejects were never scored by
  DeepSeek (bulk rejects, rejects after a rescore). Counting those as score 0
  makes DeepSeek look like 0.91. Always compare on the scored subset (the
  backtest report does).
- **Signal by judgment** (per-feature AUC): qualifications 0.86, role match
  0.84, seniority 0.81, soft preferences 0.74, thin posting barely matters,
  scam ≈ none in this data.
- **Criteria that misbehave** (from the tuning report's per-criterion table):
  - Cyrus, "over 4 years of experience" (dealbreaker): fires on 10–12% of
    kept jobs. You treat it as a stretch.
  - Martin, "regional or operational travel" (dealbreaker): fires about
    equally on kept and rejected jobs (19% vs 15%), so it has no signal.
  - Cyrus, "data engineering" (*high*): fires on more rejects (31%) than kept
    jobs (22%).
  - Cyrus, "interesting problems / generalist scope" (*high*): fires on 97% of
    kept and 83% of rejected jobs, so it doesn't separate anything.
  - Cyrus, "T-SQL stored procedures" (*high*): appears in ~2% of postings.
  - Cyrus, "relatively low (1–3) years of experience" (*medium*): the best
    separator (42% kept vs 10% rejected).
  - The grad-school schedule condition is in Cyrus's profile twice (hard
    filter and anti-criterion).
- **Wording matters.** The first backtest had the grad-school items firing
  on 12–14% of kept jobs, because ordinary day jobs read as "incompatible with
  evening classes". Adding "ordinary job features and nice-to-haves don't
  count" to every violation question dropped that to 0–1%.
- **Martin is the weak spot.** He has ~26 kept jobs, too few to tell whether
  his questions are badly worded or there's just too little data. Revisit
  after his profile review and the shadow period.

## Decisions (and why)

- **Jev via OpenRouter, not the TypeSafe API.** One credit pool to track.
  Endpoint: `POST https://openrouter.ai/api/v1/systemone`, model
  `~typesafe/jev-latest`, same request format as TypeSafe's own API, and the
  response includes the real `usage.cost`. Configured as
  `ROLE_GATE=openrouter:~typesafe/jev-latest`.
- **The gloss model writes text only; it never overrides the gate score.**
  The gate is the single source of truth for ranking.
- **One talky tier** (MiniMax, the current escalate model) for glosses. No
  separate DeepSeek-for-mid-scores tier.
- **The gloss threshold is a quality target:** set so ~95% of jobs you'd keep
  get a gloss (`GATE_GLOSS_RECALL`). `MONTHLY_BUDGET_USD` is the backstop. A
  gloss costs ~$0.004, so glossing more is cheap.
- **Prose on every card: undecided.** The plan is **gloss-on-open**: below the
  threshold, the card shows a code-written "why" plus a "write gloss" button.
  If you click it on most cards, lower the threshold to the review floor.
  Gloss the whole backlog up front when live mode lands.
- **Dealbreakers are declared, not learned.** Rare dealbreakers (crypto,
  clearance) almost never appear in your decisions, so no fit can learn them.
  A dealbreaker answered at p ≥ 0.8 (or a clear scam rating) vetoes the job.
  Profile items can be marked `"severity": "penalty"`, and penalties feed
  the fit instead.
- **Weights are learned, but refits are manual** with a preview. Nightly
  automatic refits would let the ranking drift unseen. Thresholds are picked
  from kept jobs that weren't vetoed (otherwise one misfiring dealbreaker drags
  them to the floor). A feature with too little data keeps its previous or
  default weight instead of being fit to 0.
- **Spot checks** fight label bias. Once the gate picks what you see, your
  decisions only cover jobs it liked; random below-the-floor jobs keep some
  labels independent of the gate.
- **"Kept" means applied, shortlisted or archived.** Archived means
  shortlisted then withdrawn, so it still counts as "worth a look". This adds
  some noise (e.g. a Cisco ML job you dropped over its experience ask).
  Rejects whose note is just "closed" are excluded.
- **Freshness belongs to the review queue, not the score.** `last_seen_at` is
  refreshed when a source lists the job again or its direct URL checks out
  (Adzuna landing pages don't count). The stale view holds jobs flagged
  closed, past a stated deadline (read by Jev), or unseen for
  `STALE_AFTER_DAYS`. A separate `stale` status keeps dismissals out of the
  Applications archive and out of tuning.

## How it works (code map)

- `app/gate.py`: builds the request and policy.
  - **State:** applicant summary/skills/education/work history/framing notes,
    the whole `criteria` block, the posting, and `today`.
  - **Questions:** one Noul per hard filter, anti-criterion and soft
    preference (IDs come from the item's text), plus Scores for role match,
    qualifications, seniority and scam, and Nouls for thin posting and
    passed deadline.
  - **Cache:** answers are stored in `gate_evals`, keyed by a hash of
    state + questions + model.
  - **`combine()`:** vetoes first, then the learned logistic blend over
    `role, qualified, seniority, soft, penalty, thin` plus a per-applicant
    bias. Score = probability × 100.
  - **Model versions** live in `gate_models` (candidate, active, retired).
- `app/gatefit.py`: collects labels, fits (5-fold CV), picks thresholds,
  builds the preview (AUC old vs new, queue moves, recall table, flagged
  dealbreakers, warnings), applies, rolls back, and decides when to nudge.
- `app/backtest.py`: `cli.py gate-backtest` writes `data/gate_backtest.md`.
- `app/pipeline.py`: `shadow_gate()` runs after each DeepSeek eval when
  `GATE_MODE=shadow` (stores `jobs.gate_score / gate_req / gate_json`).
  `pick_spot_checks()` runs after each full cycle.
- `app/report.py`: the tuning report, including the per-criterion firing
  table from cached answers.
- UI: Profiles → Gate box, `/gate` (preview / apply / discard / history),
  "Jev NN" on review cards (hover shows reasons), the 🎲 spot-check note, the
  ⏰ deadline-passed badge, and the "refit" badge on the nav.

## Operating it

| When | Do |
|---|---|
| Nav shows **refit** (30+ new decisions, or a profile changed) | Profiles → Gate → Preview refit → read warnings and flags → Apply |
| After importing a profile | Same, with "first ask Jev" ticked (~$0.40) |
| Jev changes version (warning on the Gate page) | `cli.py gate-backtest` first, then refit |
| You edit `.env` | Restart the server (or touch a `.py` file under `--reload`); settings are read once at startup. Docker: `docker compose up -d`, not `restart` |
| You want a fresh look at gate vs DeepSeek | `cli.py gate-backtest` (cached answers are free; only new jobs cost) |

CLI: `gate-backtest [--skipped N] [--no-backlog] [--limit N]`,
`gate-fit [--evaluate-missing] [--apply]`.

## Gotchas

- **Any profile edit invalidates the cached answers** for that applicant
  (even a summary tweak), because the whole profile is in the request.
  The cost is a re-ask, never wrong answers. `base_resume_md` and `contact`
  aren't sent, so edits to them are free.
- **Profile edits take effect on vetoes immediately, before any refit.** In
  shadow mode that only changes "Jev NN" numbers. In live mode it would hide
  jobs; see the live-mode design.
- **Soft preferences are averaged by weight**, so adding a broad preference
  shifts every job's score until the next refit.
- **Decisions count equally regardless of age.** If your preferences shift,
  old decisions pull the fit toward the old you. Add time-weighting once there
  are enough recent decisions.
- **Labels come from DeepSeek's review pool.** Decisions mostly cover jobs
  DeepSeek showed you (score ≥ 50). Spot checks gradually fix this.
- `data/seeker.db.pre-gate-backup` is a DB copy from before the first gate
  run. Delete it once you're comfortable.

## Live mode design (not built yet)

- `GATE_MODE=live`: after the gate runs, set the status from its thresholds:
  `< review_min` → skipped, `≥ review_min` → review, `≥ gloss_min` → also
  gloss. Jobs vetoed by a dealbreaker are skipped.
- **Gloss step:** a `ROLE_GLOSS` role (defaults to the escalate model) writes
  pitch/concerns, given Jev's findings so the prose explains the gate's
  verdict instead of second-guessing it. Text only.
- **Below the gloss threshold:** a code-written "why" from `top_reasons()`
  (already used in the backtest report and the hover text) plus a "write
  gloss" button.
- **Fallback:** if Jev fails or the budget is exhausted, fall back to the
  DeepSeek score path for that job.
- **Profile-change safety:** after a profile import, either keep the
  previous gate active until a refit preview has been applied, or flag new
  vetoes for review instead of skipping them silently.
- **Cleanup:** retire the separate escalate step, rename the "escalated"
  badge to "glossed", and update the Architecture diagram in the README.
