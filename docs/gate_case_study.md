# Case study: replacing an LLM scoring call with a typed-judgment gate

A record of how seeker's "is this job worth a look?" gate was rebuilt in
October 2026: the reasoning, the math, and every number along the way,
including the ones that didn't go our way. Written so it can be retold
accurately later, e.g. as an interview example. For the operational status
and next steps, see [`jev_gate.md`](jev_gate.md).

---

## 1. The problem

seeker ingests a few hundred job postings a night for two applicants. Each
posting that survives a keyword prefilter was scored by a general LLM
(DeepSeek via OpenRouter). The LLM read the applicant profile and the posting
and returned JSON: a 0–100 score, a pitch, concerns, and a scam rating.
Above 65 it was re-scored by a second model (MiniMax); below 50 it was skipped.

What the data showed about that setup:

| Measure | Value |
|---|---|
| Scoring calls | 10,717 calls, **$8.68** (~$0.00081/job, ~7.1k input tokens each) |
| Escalation calls | 607 calls, $2.27 (~$0.0037 each) |
| Jobs in `error` status (mostly JSON parse or truncation failures) | 118 |
| Mean score of jobs the applicant rejected vs kept | **62 vs 67**: weak separation |
| Kept jobs scored < 65 (never escalated) | 39 of 149 (26%) |

Goals: a gate that **ranks at least as well**, costs less, never fails on
output parsing, and produces **inspectable reasons** (why a job was ranked
low) that can feed back into the applicant's profile. Prose should only be
paid for on jobs that clear the gate.

## 2. The tool: a System One model

Jev (TypeSafe, called through OpenRouter's `/v1/systemone` endpoint) doesn't
generate text. Given a *state* (JSON) and a batch of *questions*, it returns
typed answers with probabilities:

- **Noul** (yes/no): returns p ∈ [0, 1], the probability the statement holds.
- **Score** (ordered levels 0…n, each described in words): returns a
  probability per level, p₀…pₙ, and a weighted position
  score = Σₖ k·pₖ.

All questions in a request see the same state and are answered in parallel.

**Billing probe.** Before designing the questions, the same ~1.6k-token state
was sent with 1, 2, 5 and 10 questions: input tokens were 1,651 / 1,679 /
1,763 / 1,903. **The state is billed once; each extra question costs ~28
tokens.** That made the design choice easy: one narrow question per profile
item (about 30 per job) costs barely more than one broad question.

## 3. Turning a profile into questions

Each job gets one request. The **state** holds the applicant (summary,
skills, education, work history, framing notes), the profile's `criteria`
block, the posting (title, company, location, salary, description, first-seen
date), and today's date.

The **questions** are generated from the profile:

| Source | Question type | Example |
|---|---|---|
| each hard filter (location, salary floor, other) | Noul: "violates…" | "The job conflicts with: *must be compatible with evening classes*" |
| each anti-criterion | Noul: "falls into…" | "The job falls into: *requires a security clearance*" |
| each soft preference | Noul: "offers…" | "The job offers: *NLP or multilingual work*" |
| target roles | Score, 4 levels | unrelated → same field, different work → adjacent → the role itself |
| qualifications | Score, 4 levels | lacks most required → meets essentially all |
| seniority | Score, 4 levels | far above → stretch → fits → overqualified |
| scam risk | Score, 4 levels | identifiable employer → … → clear scam markers |
| thin posting | Noul | "too thin to judge duties, schedule, location or seniority" |
| deadline passed | Noul | "states a deadline that is before `today`" |

Violation questions carry explicit true/false definitions. After the first
backtest, the false case was extended to say *ordinary job features (a
standard full-time day schedule, local travel) and "nice to have" lines don't
count* (see §6).

## 4. The math

### 4.1 Features

Per-question answers are collapsed into a fixed set of features, so the same
model works across applicants and profile versions with different question
lists:

- **role**, **qualified**: Score position normalized to [0, 1]:
  f = (Σₖ k·pₖ) / n, with n = 3.
- **seniority**: an expectation over a *non-monotonic* value per level,
  because overqualified beats a big stretch but a good fit beats both:
  f = Σₖ pₖ·vₖ, with v = (0, 0.5, 1.0, 0.7).
- **soft**: weighted mean of the preference Nouls, weights high = 3,
  medium = 2, low = 1:
  f = Σᵢ wᵢpᵢ / Σᵢ wᵢ.
- **penalty**: probability that at least one penalty-severity item applies,
  treating items as independent:
  f = 1 − Πⱼ (1 − pⱼ).
- **thin**: the Noul probability.

### 4.2 Vetoes (declared policy, not learned)

A job is vetoed if any **dealbreaker** (hard filter, or an anti-criterion not
marked `"severity": "penalty"`) has p ≥ 0.8, or if P(scam level 3) ≥ 0.5.
A vetoed job's score is capped at 2.

*Why not learn these too:* the rare dealbreakers (crypto, security
clearance, driver's license) almost never appear among the decisions, because
upstream filters already removed most such jobs. No fit can learn a weight
for something it never sees, so these rules have to be stated.

### 4.3 Score

A logistic model over the features, plus a per-applicant bias:

  z = b + Σₖ wₖ·fₖ + b_applicant
  score = 100 · σ(z),  σ(z) = 1 / (1 + e⁻ᶻ)

The score reads as "approximate probability this applicant would keep the
job".

### 4.4 Fitting (`gate-fit`)

- **Labels:** kept = applied, shortlisted or archived (archived means
  shortlisted then withdrawn, so still "worth a look"); rejected = rejected,
  except rejects whose note says the posting closed (those aren't fit
  judgments).
- **Training set:** vetoed jobs are excluded, because their score is fixed
  by the veto and the blend shouldn't learn from them.
- **Loss:** class-weighted logistic loss with L2 regularization (λ = 0.01,
  bias not penalized). Kept jobs are ~1 in 7, so each class gets weight
  n / (2·n_class) to stop the majority class dominating.
- **Optimizer:** plain batch gradient descent, learning rate 0.3, 3,000
  iterations. The data is ~1,200 rows × ~8 columns, so no library is needed.
- **Features with too little data:** a feature that is ≥ 0.05 on fewer than
  10 training rows isn't fit. It's held at its previous weight (or the
  built-in default) and entered as a fixed offset to z. This was added after
  "penalty" was fit to exactly 0 simply because no penalty items existed yet
  (§6).
- **Honest evaluation:** 5-fold cross-validation. Every job's score used for
  AUC and threshold selection comes from a model that didn't train on it.
- **Refits are manual**, with a preview: old vs new AUC, how many unreviewed
  jobs would change bucket, a recall table, and dealbreakers that fired on
  kept jobs. Nothing applies until approved, and older versions can be
  restored.

### 4.5 Ranking metric: AUC

AUC = the probability that a randomly chosen kept job outranks a randomly
chosen rejected one (0.5 = coin flip, 1.0 = perfect). It's computed with the
Mann–Whitney rank-sum, ties averaged:

  AUC = (R_kept − n_kept(n_kept + 1)/2) / (n_kept · n_rejected)

where R_kept is the sum of the kept jobs' ranks.

**Uncertainty.** The Hanley–McNeil approximation gives the standard error of
a single AUC. Because both scorers rank the *same* jobs, the difference
between them is tested with a **paired bootstrap**: resample jobs with
replacement 400 times, recompute both AUCs, and take the 2.5th–97.5th
percentiles of the difference.

### 4.6 Thresholds

Two thresholds are picked from **recall targets**, not set by hand:

- **gloss_min:** keep 95% of kept jobs (they get a written gloss).
- **review_min:** keep 99% of kept jobs (they reach the review queue).

Method: sort the cross-validated scores of kept, non-vetoed jobs in
descending order. The threshold for target r is the score at position
⌈r·n⌉, floored to 0.1. Vetoed kept jobs are excluded on purpose: including
them let one misfiring dealbreaker drag both thresholds to the veto floor (§6).
Their cost is reported separately.

## 5. Making the comparison fair

The first comparison made DeepSeek look far better than it was: **AUC 0.911**
vs Jev's 0.835. The reason was that **700 of the ~1,015 rejects had never been
scored by DeepSeek** (bulk rejects, rejects made after a rescore cleared the
score). Treating a missing score as 0 handed DeepSeek hundreds of free correct
rankings.

The fair comparison only uses jobs both scorers actually scored. A remaining
bias can't be removed: every labeled job had already passed DeepSeek's ≥ 50
review floor, so the labels come from DeepSeek's own pool. To start correcting
this, three random below-the-floor jobs per applicant per night are surfaced
for review as **spot checks**, so future labels aren't chosen by the scorer
being evaluated.

## 6. Timeline and numbers

**Round 1: hand-written formula.** score = 100 × weighted fit ×
(1 − worst hard-filter p) × (1 − worst anti-criterion p) × …

- Fair comparison: DeepSeek 0.726, Jev learned blend 0.723, **both combined
  0.741**. A tie, and the two carry partly different signal.
- Per-feature AUC: qualifications 0.864, role 0.837, seniority 0.814, soft
  preferences 0.735, scam 0.55 (not informative here), thin 0.54.
- **Diagnosis:** the multiplicative penalty treated every anti-criterion as a
  proportional veto, and some questions misfired on jobs the applicant kept:
  - "over 4 years of experience": 20% of kept jobs (61% of rejects)
  - the evening-classes schedule items: 12–14% of kept jobs, because
    ordinary day jobs read as "incompatible"
  - "no regional or operational travel": 23% of kept vs 27% of rejected,
    so no signal at all

**Round 2: changes made.**
1. Reworded the false case on every violation question.
2. Added severity tiers (dealbreaker or penalty) and the p ≥ 0.8 veto.
3. Replaced the hand formula with the learned blend.
4. Thresholds from recall targets.

Results:
- The evening-class items dropped to **0–1%** of kept jobs, and "over 4
  years" to 12%. The travel item didn't improve (19% kept vs 15% rejected).
- Dealbreakers still vetoed **19% of kept jobs**, which collapsed both
  thresholds to 2.0. **Fix:** thresholds are now chosen from non-vetoed kept
  jobs, and veto losses are reported separately.
- Fair AUC with default weights: Jev 0.701 vs DeepSeek 0.712 (Cyrus 0.736 vs
  0.715; Martin 0.626 vs 0.761).
- **What-if,** with the two misfiring dealbreakers changed to penalties:
  kept jobs lost to vetoes 29 → 10; fair AUC **0.736 vs 0.712**; AUC over all
  decisions 0.824 → 0.875; share of kept jobs reaching review 80% → 93%.
- **Bug found:** the fit set the "penalty" weight to exactly 0, because no
  profile had penalty items yet. **Fix:** features with too little data are
  held at their previous or default weight (§4.4).

**Round 3: profile review, then refit (current, model v3).** The applicant
reviewed their profile against a report showing how often each criterion
fired on kept vs rejected jobs. They downgraded and reworded items, then
refit.

| | Value |
|---|---|
| Decisions | 1,257 (177 kept), 5-fold CV |
| Weights | bias −2.16, role +1.29, qualified +1.48, seniority +1.41, soft +0.47, penalty **−0.58** (now learned), thin −1.60; applicant bias Cyrus +0.52, Martin −0.53 |
| Thresholds | gloss ≥ 39.2, review ≥ 29.3 |
| Kept jobs vetoed | 18 of 177 (10%), down from 29–31 |
| At the review floor | 89% of all kept jobs reach review (≈99% of non-vetoed); 68% of rejects are culled |
| AUC, all decisions | 0.855 |

Fair comparison against DeepSeek, with Hanley–McNeil standard errors and
paired-bootstrap 95% intervals for the difference:

| | Kept / rejected | Jev | DeepSeek | Difference (95% CI) |
|---|---|---|---|---|
| all | 177 / 380 | 0.728 ± 0.024 | 0.704 ± 0.025 | +0.025 [−0.040, +0.090] |
| Cyrus | 151 / 284 | 0.750 ± 0.026 | 0.704 ± 0.027 | +0.046 [−0.015, +0.121] |
| Martin | 26 / 96 | 0.594 ± 0.065 | 0.771 ± 0.058 | **−0.176 [−0.349, −0.002]** |

**Reading:** overall and for Cyrus, Jev is ahead but **not significantly**,
so call it a tie at about a third of the cost. For Martin, Jev is
**significantly worse**: his profile hadn't been reviewed yet, and he has
only 26 kept jobs. That's the open problem.

**Total Jev spend for the whole project** (every backtest, refit and re-ask):
**$1.37** over 4,809 calls.

## 7. What changed besides the score

- **No parse failures.** The output is typed, so the 118-error class of
  failure disappears.
- **Inspectable reasons.** Every score decomposes into named answers ("ruled
  out: requires security clearance"; "concern: over 4 years of experience";
  "for: NLP domain"). These drive hover text on cards and the code-written
  "why" for jobs without a gloss.
- **A feedback loop into the profile.** The tuning report now shows, per
  criterion, how often it fires on kept vs rejected jobs. That turned vague
  profile edits into evidence-based ones: Cyrus's *high*-weight "data
  engineering" preference fired more on rejects (31%) than on kept jobs
  (22%); "interesting problems" fired on 97% of kept and 83% of rejected jobs,
  so it separated nothing; the *medium* "1–3 years of experience" preference
  was the best separator (42% vs 10%).
- **Cost structure.** Jev costs ~$0.00025/job vs ~$0.0008. The planned live
  setup spends ~$0.004 per prose gloss only above the gloss threshold, chosen
  so 95% of jobs the applicant would keep get one.

## 8. Lessons

1. **Check the comparison before believing it.** The first headline number
   (0.911 vs 0.835) was an artifact of missing scores counted as zeros.
2. **Declared rules and learned weights do different jobs.** Rare
   dealbreakers can't be learned from data that upstream filters already
   cleaned; frequent, fuzzy preferences shouldn't be hand-weighted. Splitting
   them (veto layer plus learned blend) fixed both problems.
3. **Wording is most of the precision.** One clarifying sentence on every
   violation question cut a misfire rate from 12–14% to 0–1%.
4. **Watch for degenerate fits.** A weight of exactly 0 meant "no data", not
   "doesn't matter"; thresholds of 2.0 meant "vetoes ate the recall budget",
   not "everything passes".
5. **Report uncertainty.** With ~150 positives, AUC differences of ±0.03 are
   noise. Saying "tie at a third of the cost, worse for one applicant" is the
   accurate claim.
6. **Selection bias is structural.** Labels come from the old scorer's pool.
   Spot checks are the cheap fix, which only works going forward.

## 9. One-paragraph version

> My job-search pipeline scored every posting with a general LLM that
> returned a 0–100 score and prose; it cost ~$0.0008/job, failed to parse
> ~1% of the time, and separated jobs I kept from ones I rejected only weakly.
> I replaced the score with a "System One" model that answers ~30 narrow typed
> questions per job (one per profile criterion, plus role, qualification and
> seniority fit), combined with declared dealbreaker vetoes and a small
> logistic model learned from ~1,250 of my past decisions with 5-fold
> cross-validation. Thresholds are chosen from recall targets instead of set
> by hand. Along the way I caught an unfair baseline (700 unscored rejects
> counted as zeros), fixed questions that misfired on ordinary job features
> (misfire rate 12–14% → 0–1%), and used per-criterion firing rates to rewrite
> my own profile. Result: ranking quality statistically tied with the old
> scorer overall (AUC 0.728 vs 0.704, 95% CI on the difference −0.04 to
> +0.09), significantly worse for one applicant still under review, at about
> a third of the cost, with no parse failures and a human-readable reason for
> every score. Total model spend for the whole evaluation: $1.37.
