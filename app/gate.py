"""Jev gate: "is this job worth a look?" as typed judgments, not prose.

One Jev request per job (via OpenRouter's System One endpoint) asks a batch of
narrow questions built from the applicant's profile — one per hard filter,
anti-criterion and soft preference, plus role fit, qualification, seniority,
posting thinness, scam risk and a passed-deadline check. All questions see the
same state and run in parallel; the state is billed once, each extra question
costs ~30 tokens, so splitting the profile finely is nearly free.

Policy lives in `combine()`, not in the model: it turns the raw answers into
features and a 0–100 gate score. Raw answers are cached in `gate_evals`, so
changing weights or thresholds never needs another model call.

Nothing here touches job status — the backtest and (later) the pipeline
decide what to do with the score.
"""
import hashlib
import json
import time
from datetime import datetime, timezone

import httpx

from . import db
from .config import settings
from .llm import BudgetExceeded, LLMError

_BASE_URLS = {"openrouter": "https://openrouter.ai/api"}

# Applicant fields Jev sees. base_resume_md repeats work_history at 2x the
# tokens; contact details are irrelevant to fit.
APPLICANT_KEYS = ("summary", "skills", "education", "work_history", "framing_notes")
# Jev allows 32k tokens of state + longest question. ~4 chars/token; the
# profile takes ~2-3k tokens, so this leaves ample room.
MAX_DESC_CHARS = 60_000
PREF_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS gate_evals (
    job_id TEXT NOT NULL,
    req_hash TEXT NOT NULL,      -- hash of state (minus today) + questions + model alias
    applicant_id TEXT NOT NULL,
    profile_version INTEGER,
    model TEXT NOT NULL,         -- concrete model version that answered
    answers_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,     -- question id -> {kind, label, weight}
    input_tokens INTEGER,
    cost_usd REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, req_hash)
);
"""


def init_db():
    with db.connect() as conn:
        conn.executescript(SCHEMA)


# --- request construction ----------------------------------------------------

def build_state(profile: dict, job) -> dict:
    p = profile.get("profile") or {}
    desc = job["description"] or ""
    if len(desc) > MAX_DESC_CHARS:
        desc = desc[:MAX_DESC_CHARS] + "\n[...truncated]"
    return {
        "applicant": {k: p[k] for k in APPLICANT_KEYS if k in p},
        "criteria": profile.get("criteria") or {},
        "job": {
            "title": job["title"], "company": job["company"],
            "location": job["location"], "salary": job["salary"],
            "first_seen": (job["created_at"] or "")[:10],
            "description": desc,
        },
    }


def _noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions}
    if true and false:
        q["criteria"] = {"true": true, "false": false}
    return q


def _score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def build_questions(profile: dict) -> tuple[dict, dict]:
    """Questions for this applicant, plus meta describing each one so combine()
    and the UI know what a question id means. Fixed ids for the core judgments;
    indexed ids for profile-list items."""
    crit = profile.get("criteria") or {}
    hard = crit.get("hard_filters") or {}
    qs: dict = {}
    meta: dict = {}

    def add(qid, q, kind, label, weight=None):
        qs[qid] = q
        meta[qid] = {"kind": kind, "label": label, "weight": weight}

    no_info = "It fits, or the posting doesn't say enough to tell"
    if hard.get("location"):
        add("hard_location", _noul(
            f'The job\'s work location or remote/hybrid/on-site arrangement violates '
            f'the applicant\'s location requirement: "{hard["location"]}"',
            "The posting's location or work arrangement clearly falls outside the requirement",
            no_info), "hard", f"location: {hard['location']}")
    if hard.get("salary_floor"):
        add("hard_salary", _noul(
            f'The posting states pay, and even the top of its range is below the '
            f'applicant\'s floor of {hard["salary_floor"]}',
            "Stated pay is clearly below the floor",
            "Pay meets the floor, or no pay is stated"), "hard",
            f"salary floor {hard['salary_floor']}")
    for i, r in enumerate(hard.get("other") or []):
        add(f"hard_other_{i}", _noul(
            f'The job conflicts with this requirement of the applicant\'s: "{r}"',
            "The posting clearly states something that conflicts with the requirement",
            "No conflict, or the posting doesn't say enough to tell"), "hard", r)
    for i, a in enumerate(crit.get("anti_criteria") or []):
        add(f"anti_{i}", _noul(
            f'The job falls into this category the applicant has ruled out: "{a}"',
            "The job's actual work or requirements fall into the excluded category",
            "It doesn't, or the overlap is only superficial (a shared keyword "
            "without the substance)"), "anti", a)
    for i, sp in enumerate(crit.get("soft_preferences") or []):
        want = sp.get("want") if isinstance(sp, dict) else str(sp)
        weight = PREF_WEIGHTS.get((sp.get("weight") if isinstance(sp, dict) else "") or "", 2.0)
        add(f"soft_{i}", _noul(
            f'The job offers this, which the applicant wants: "{want}"'), "soft", want, weight)

    add("role", _score(
        "How closely does the main day-to-day work of the job in `job` match the "
        "roles listed in `criteria.target_roles`?", [
            "Unrelated to every target role",
            "Same broad field, but the main work differs from every target role",
            "A hybrid or adjacent role that overlaps substantially with a target role",
            "The main work is one of the target roles",
        ]), "core", "role match")
    add("qualified", _score(
        "How well does the applicant's background in `applicant` meet the job's "
        "REQUIRED qualifications (not the nice-to-haves)?", [
            "Lacks most of the required qualifications",
            "Meets some required qualifications but lacks several important ones",
            "Meets most required qualifications, with minor gaps",
            "Meets essentially all required qualifications",
        ]), "core", "meets requirements")
    add("seniority", _score(
        "How does the experience and seniority the job asks for compare to the "
        "applicant's background in `applicant`?", [
            "Requires far more experience or seniority than the applicant has "
            "(e.g., senior, lead or manager level for someone early in this field)",
            "Asks for somewhat more experience than the applicant has: a stretch, but plausible",
            "The experience and seniority asked for fit the applicant's background",
            "Clearly below the applicant's level: the applicant would be overqualified",
        ]), "core", "seniority fit")
    add("thin", _noul(
        "The text in `job.description` is too thin to judge the job's actual duties, "
        "schedule, location or seniority",
        "Only a preamble, a truncated preview or a few generic lines; the real "
        "duties and requirements aren't described",
        "The duties and requirements are described well enough to judge"),
        "flag", "posting too thin to judge")
    add("scam", _score(
        "How likely is this posting to be a scam? Consider the applicant's notes in "
        "`criteria.scam_wariness`.", [
            "An identifiable employer with a real product or service and an ordinary hiring process",
            "Probably legitimate, but the employer or process is vague or hard to verify",
            "Several warning signs: unverifiable company, pay far above market for "
            "low-skill remote work, personal email contacts",
            "Clear scam markers: up-front fees, equipment or check schemes, reshipping, "
            "crypto or wire transfers",
        ]), "flag", "scam risk")
    add("deadline_passed", _noul(
        "The posting states an application deadline or closing date, and that date "
        "is before `today`",
        "A stated deadline or closing date has already passed",
        "No deadline is stated, or the stated deadline is today or later"),
        "flag", "stated deadline has passed")
    return qs, meta


# --- transport ----------------------------------------------------------------

def _req_hash(state: dict, questions: dict, model: str) -> str:
    blob = json.dumps({"s": state, "q": questions, "m": model}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _post(state: dict, questions: dict) -> dict:
    target = settings.role("gate")
    if target.provider not in _BASE_URLS:
        raise LLMError(f"ROLE_GATE provider '{target.provider}' not supported "
                       f"(expected one of {sorted(_BASE_URLS)}).")
    if not settings.openrouter_api_key:
        raise LLMError("OPENROUTER_API_KEY not set but ROLE_GATE routes to openrouter.")
    with db.connect() as conn:
        if db.monthly_spend(conn) >= settings.monthly_budget_usd:
            raise BudgetExceeded(f"Monthly budget ${settings.monthly_budget_usd:.2f} "
                                 "reached; refusing gate call.")
    body = {"model": target.model, "state": state, "questions": questions}
    delay = 1.0
    for attempt in range(5):
        try:
            r = httpx.post(_BASE_URLS[target.provider] + "/v1/systemone", json=body,
                           headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                           timeout=60)
        except httpx.TransportError as e:
            err = str(e)
        else:
            if r.status_code == 200:
                data = r.json()
                if "answers" not in data:
                    raise LLMError(f"gate response missing answers: {str(data)[:300]}")
                return data
            if r.status_code not in (429, 500, 502, 503, 504, 529):
                raise LLMError(f"gate call failed {r.status_code}: {r.text[:300]}")
            err = f"{r.status_code}: {r.text[:200]}"
        if attempt < 4:
            time.sleep(delay)
            delay *= 2
    raise LLMError(f"gate call failed after retries ({err})")


def evaluate(job, profile: dict, profile_version: int | None = None,
             use_cache: bool = True) -> dict:
    """Ask Jev about one job. Returns {answers, meta, model, cached}. Results are
    cached by request content, so re-runs with an unchanged profile and posting
    are free. `today` is left out of the cache key: only the deadline question
    depends on it, and a cached answer from an earlier day errs toward 'open'."""
    state = build_state(profile, job)
    questions, meta = build_questions(profile)
    target = settings.role("gate")
    h = _req_hash(state, questions, target.model)
    if use_cache:
        with db.connect() as conn:
            row = conn.execute("SELECT model, answers_json, meta_json FROM gate_evals "
                               "WHERE job_id=? AND req_hash=?", (job["id"], h)).fetchone()
        if row:
            return {"answers": json.loads(row["answers_json"]),
                    "meta": json.loads(row["meta_json"]), "model": row["model"], "cached": True}
    today = datetime.now(timezone.utc).date().isoformat()
    data = _post({**state, "today": today}, questions)
    usage = data.get("usage") or {}
    in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    cost = float(usage.get("cost") or 0.0)
    model = data.get("model") or target.model
    db.log_usage("gate", target.provider, model, in_tok, out_tok, cost)
    with db.connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO gate_evals (job_id, req_hash, applicant_id, "
            "profile_version, model, answers_json, meta_json, input_tokens, cost_usd, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job["id"], h, job["applicant_id"], profile_version, model,
             json.dumps(data["answers"]), json.dumps(meta), in_tok, cost, db.now()))
    return {"answers": data["answers"], "meta": meta, "model": model, "cached": False}


# --- policy -------------------------------------------------------------------

# How much each seniority level is worth: overqualified beats a big stretch,
# but a good fit beats both.
SENIORITY_VALUE = [0.0, 0.5, 1.0, 0.7]
SCAM_LABELS = ["none", "low", "medium", "high"]
# Starting weights for the fit blend — expected to be replaced by what the
# backtest learns from your actual shortlist/reject decisions.
FIT_WEIGHTS = {"role": 0.40, "qualified": 0.25, "seniority": 0.15, "soft": 0.20}


def _expect(ans: dict, values: list[float]) -> float:
    return sum(float(p) * values[int(k)] for k, p in ans["probabilities"].items())


def _norm_score(ans: dict) -> float:
    n = len(ans["probabilities"]) - 1
    return float(ans["score"]) / n if n else 0.0


def features(answers: dict, meta: dict) -> dict:
    """Collapse per-question answers into a fixed feature set, comparable across
    applicants whose profiles produce different question lists."""
    def nouls(kind):
        return [(qid, float(answers[qid]["noul"])) for qid, m in meta.items()
                if m["kind"] == kind and qid in answers]
    hard, anti, soft = nouls("hard"), nouls("anti"), nouls("soft")
    sw = [(meta[q]["weight"] or 2.0, p) for q, p in soft]
    return {
        "role": _norm_score(answers["role"]),
        "qualified": _norm_score(answers["qualified"]),
        "seniority": _expect(answers["seniority"], SENIORITY_VALUE),
        "soft": (sum(w * p for w, p in sw) / sum(w for w, _ in sw)) if sw else 0.5,
        "hard_max": max((p for _, p in hard), default=0.0),
        "anti_max": max((p for _, p in anti), default=0.0),
        "thin": float(answers["thin"]["noul"]),
        "scam": _norm_score(answers["scam"]),
        "deadline_passed": float(answers["deadline_passed"]["noul"]),
    }


def combine(answers: dict, meta: dict) -> dict:
    """Gate score 0–100 from the raw answers. Fit is a weighted blend; hard
    filters and anti-criteria scale it down by how likely they're violated
    (so a 0.9 violation all but zeroes it, a 0.1 barely matters); a thin
    posting is capped at 45, mirroring the old prompt's rule."""
    f = features(answers, meta)
    fit = sum(FIT_WEIGHTS[k] * f[k] for k in FIT_WEIGHTS)
    score = 100 * fit * (1 - f["hard_max"]) * (1 - f["anti_max"])
    score *= 1 - 0.8 * f["deadline_passed"]
    score *= 1 - 0.8 * max(0.0, f["scam"] - 0.34) / 0.66  # only medium/high bite
    if f["thin"] > 0.5:
        score = min(score, 45)
    scam_probs = answers["scam"]["probabilities"]
    scam_label = SCAM_LABELS[int(max(scam_probs, key=lambda k: float(scam_probs[k])))]
    return {"gate_score": round(score, 1), "features": f, "scam_risk": scam_label}


def top_reasons(answers: dict, meta: dict, n: int = 3) -> tuple[list[str], list[str]]:
    """(for, against): the most decisive answers in plain words, for a card that
    has no written gloss. Nouls above 0.5 only; ordered by probability."""
    pros, cons = [], []
    for qid, m in meta.items():
        a = answers.get(qid)
        if not a or a.get("type") != "noul":
            continue
        p = float(a["noul"])
        if p < 0.5:
            continue
        if m["kind"] == "soft":
            pros.append((p * (m["weight"] or 2.0), m["label"]))
        elif m["kind"] in ("hard", "anti", "flag"):
            cons.append((p, m["label"]))
    for qid, label in (("role", "role match"), ("qualified", "meets requirements")):
        if qid in answers:
            legend = answers[qid].get("legend") or {}
            top = max(answers[qid]["probabilities"], key=lambda k: float(answers[qid]["probabilities"][k]))
            (pros if int(top) >= 2 else cons).append((1.0, f"{label}: {legend.get(top, top)}"))
    pros.sort(reverse=True)
    cons.sort(reverse=True)
    return [t for _, t in pros[:n]], [t for _, t in cons[:n]]
