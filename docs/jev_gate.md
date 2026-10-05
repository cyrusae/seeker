# Jev gate: status, decisions, and what's next

The project in one sentence: **replace the DeepSeek "score every job"
call with Jev, a cheaper model that returns typed judgments, as the "is this
worth looking at?" gate, and only pay a talky model to write a gloss for jobs
above a (lower) threshold.**

Started 2026-10-01. This file is the running record; update the Status and
Next steps sections when something moves.

For the full reasoning, the math, and every number in one narrative (e.g. for
explaining the project to someone else), see
[`gate_case_study.md`](gate_case_study.md).

---

## Status (as of 2026-10-01)

| Piece | State |
|---|---|
| Review-queue freshness (`last_seen_at`, stale view/status, age sort) | **Done**, live |
| Jev gate module, backtest, `gate-fit`, Gate page | **Done** |
| `GATE_MODE=shadow`: new jobs get a Jev score next to DeepSeek's | **On** (DeepSeek still decides) |
| Spot checks (3 random skipped jobs/applicant/cycle) | **On** |
| Cyrus profile | **v14** (2026-10-02): v13 + the domain-expertise item changed from dealbreaker to penalty. Reports for v12→v13 and v13→v14 are saved in Gate lab → Profile changes |
| Gate model in use | **v13** (2026-10-05, 1,299 decisions): gloss ≥ 37.1, review ≥ 24.1; dealbreaker close calls (p 0.5–0.8) now count as penalties; 8 of 188 kept jobs vetoed. Fair comparison: Cyrus Jev 0.775 vs DeepSeek 0.707 |
| Shadow scores | Current for all jobs with answers under the current profiles (~1,630 jobs). Status → Backfill refreshes after any profile change |
| Martin profile | **v11** (2026-10-05), refit done. Fair comparison now Jev **0.746** vs DeepSeek 0.793 (was 0.594 vs 0.771 on v9). Still behind DeepSeek, but the gap is down from 0.18 to 0.05. The agency postings he keeps rejecting score ~4 in the gate; they reach him because DeepSeek decides his queue |
| Live mode (gate decides status, gloss step, gloss-on-open) | **Not built** |

## Next steps

1. **Martin's full review** (Cyrus's is done). Profiles → download his
   tuning report → chat session with `docs/interview_spec.md` in **Full
   review** mode, with Martin. Start with his travel dealbreaker (no signal;
   see Evidence).
2. **Import, then refit.** Profiles → Gate → Preview refit with "first ask
   Jev" ticked (past decisions get answers under the new wording). Check
   "Dealbreakers that fire on jobs you kept" and that the AUC isn't worse than
   v3's. Apply. Also look at which of the 18 vetoed kept jobs remain, and why.
3. **Shadow period.** Review normally for 2–3 weeks or ~100+ decisions,
   including the 🎲 spot checks. Refit when the nav shows the "refit" badge.
4. **Go/no-go for live mode.** Re-run `cli.py gate-backtest` and check:
   - Jev's AUC ≥ DeepSeek's on the fair comparison (jobs DeepSeek scored),
     for **each** applicant, not just overall.
   - Review floor keeps ≥ ~97% of kept jobs that weren't vetoed.
   - No dealbreaker fires on more than ~5% of kept jobs.
   - Spot checks: you rarely keep one (if you often do, the floor is too high).
5. **Build live mode** (see "Live mode design" below), switchable **per
   applicant**, so one applicant can go live while the other stays on
   DeepSeek until their profile and numbers are ready.

**Resolved 2026-10-01:** Cyrus's citizenship dealbreaker was firing on any
"US citizen required". Fixed in profile v13; it now fires on 0 kept jobs and
still catches an explicit dual-citizen exclusion (p = 0.98 on a synthetic
test). The Gate page now flags any dealbreaker that fires at least as often on
kept jobs as on rejected ones. Details in the case study, round 4.

## Evidence so far

Figures in this section come from the profiles as they were before the
reviews (Cyrus v10, Martin v9). From two backtests over ~1,165 past decisions (151 kept, ~1,015 rejected),
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
- **Dealbreaker close calls count as penalties** (p 0.5–0.8 feeds the penalty
  feature). A probable dealbreaker used to add nothing to the score.
- **No "strong penalty" tier (for now).** Considered for "only if it's
  really, really good" items (Martin's 8 AM start), with a fixed weight
  around −1.2 to −2.0. Not built: on the data, the agency postings it was
  meant for already score ~4, and −2.0 felt too strong. Revisit if a
  penalty item keeps letting through jobs that should have dropped.
- **Dealbreakers are declared, not learned.** Rare dealbreakers (crypto,
  clearance) almost never appear in your decisions, so no fit can learn them.
  A dealbreaker answered at p ≥ 0.8 (or a clear scam rating) vetoes the job.
  Profile items can be marked `"severity": "penalty"`, and penalties feed
  the fit instead.
- **After a profile change, refit even if the preview says the current model
  scores marginally higher.** The current model's AUC is partly measured on
  decisions it was trained on (flattered); the candidate's is cross-validated.
  And changing which items are penalties or preferences shifts every score,
  so the old thresholds go stale (v13→v14 pushed 29 jobs from gloss to review
  until the refit).
- **Weights are learned, but refits are manual** with a preview. Nightly
  automatic refits would let the ranking drift unseen. Thresholds are picked
  from kept jobs that weren't vetoed (otherwise one misfiring dealbreaker drags
  them to the floor). A feature with too little data keeps its previous or
  default weight instead of being fit to 0.
- **Spot checks** fight label bias. Once the gate picks what you see, your
  decisions only cover jobs it liked; random below-the-floor jobs keep some
  labels independent of the gate.
- **Withdrawing from the shortlist has two meanings.** **Archive** = "changed
  my mind": the gate counts it as kept (it was worth a look), while the tuning
  report counts it as a miss (the profile didn't predict the final call).
  Tested: counting archives as kept, excluded, or rejected gives AUC 0.926 /
  0.921 / 0.917 on the same test, so this choice doesn't matter much. **✗ Doesn't
  actually fit** (Applications, or the Gate lab job page) = "I misread it":
  recorded as a reject for both. Use it when the gate catches your mistake.
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
- `app/gatelab.py` (**Gate lab** tab, visible while "Jev shadow scores" is
  on): troubleshooting views built only from stored answers and the model's
  numbers, never generated text.
  - **Job inspector** (`/gate/job/<id>`, the "why?" link on cards): plain-words
    reasons, the score arithmetic term by term, every question's answer, and
    what changed across profile versions for that job.
  - **"Find evidence"**: one Jev call (~$0.0001–0.0003) asks which line of the
    posting is behind an answer. Cached in `gate_evidence` by posting text +
    question wording, so repeats and unrelated profile edits are free.
  - **Criteria** (`/gate/criteria`): how often each criterion fires on kept /
    rejected / other jobs at an adjustable threshold, ⚑ flags, and a
    drill-down to every job an item fired on.
  - **Profile changes** (`/gate/changes`): what a profile version changed
    (criteria diff, per-item firing before/after, jobs whose outcome changed,
    cross-validated AUC before/after, re-ask cost), scored with the same gate
    model on both sides. A report is saved automatically when a refit is
    applied, or on demand. Profile imports accept an optional note ("what
    this version is trying to fix"), which is shown on the report.
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
| A job's Jev score looks wrong | Turn on "Jev shadow scores", click **why?** on the card, and use **find evidence** on the answer that fired |
| You changed a profile and want to know what it did | Gate lab → Profile changes (pick the two versions); save the report |
| You want a fresh look at gate vs DeepSeek | `cli.py gate-backtest` (cached answers are free; only new jobs cost) |

CLI: `gate-backtest [--skipped N] [--no-backlog] [--limit N]`,
`gate-fit [--evaluate-missing] [--apply]`.

## Gotchas

- **The live server runs from this working folder with `--reload`.** Saving
  a `.py` file (including on an unmerged branch) reloads the server you and
  Martin are using. On 2026-10-05 an edit triggered a reload whose shutdown
  got stuck, and the server stopped answering until the old worker was
  killed (`kill -9 <worker pid>`; the reloader then starts a fresh one).
  Safer setups: start the server with `--timeout-graceful-shutdown 10`, so a
  stuck shutdown can't hang it, and/or run it from a separate checkout of
  `main` (see "Separate live checkout" below).
- **The two exclusion lists read in opposite directions.**
  `hard_filters.other` = requirements ("No Sunday work"); `anti_criteria` =
  jobs to rule out ("Position is per diem"). A describe-the-job item in the
  wrong list is inverted. The Gate lab flags exclusions that apply to most
  jobs (⚠).

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

## Separate live checkout (proposed, not done)

Keep editing and testing in this folder, and run the server from a second
checkout that only ever has merged `main`:

```sh
cd ~/GitHere/seeker && git -C prototype worktree add ../live-checkout live
# start the server from ../live-checkout/prototype with DATA_DIR pointing at
# this folder's data/ (absolute path) and --timeout-graceful-shutdown 10
# deploy: git -C ~/GitHere/seeker/live-checkout merge --ff-only main
```

The same split (dev here, live elsewhere, `git pull` to deploy) is how the
homelab move should work.

## Live mode design (not built yet)

- **Per-applicant rollout:** e.g. `GATE_LIVE_APPLICANTS=cyrus`. Everyone not
  listed stays on the DeepSeek path (with shadow scoring). Go/no-go criteria
  are applied per applicant.
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
