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
import math
import time
from datetime import datetime, timezone

import httpx

from . import db
from .config import settings
from .llm import BudgetExceeded, LLMError

_BASE_URLS = {"openrouter": "https://openrouter.ai/api"}

# Applicant fields Jev sees. base_resume_md repeats work_history at 2x the
# tokens; contact details are irrelevant to fit. work_authorization (e.g.
# "US citizen (dual citizenship); no sponsorship needed") lets Jev judge
# citizenship/sponsorship requirements instead of assuming they're unmet.
APPLICANT_KEYS = ("summary", "skills", "education", "work_history", "framing_notes",
                  "work_authorization")
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
        conn.executescript(MODEL_SCHEMA)


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


def _item(entry) -> tuple[str, str]:
    """A hard-filter/anti-criterion entry is either a plain string (a
    dealbreaker, the default) or {"text": ..., "severity": "dealbreaker" |
    "penalty"}. Penalties count against fit; dealbreakers veto."""
    if isinstance(entry, dict):
        sev = entry.get("severity", "dealbreaker")
        return str(entry.get("text", "")), sev if sev in ("dealbreaker", "penalty") else "dealbreaker"
    return str(entry), "dealbreaker"


def _qid(prefix: str, text: str) -> str:
    """Stable id from the item's text, so editing or reordering a profile list
    doesn't silently reassign one item's history to another."""
    return f"{prefix}_{hashlib.sha1(text.encode()).hexdigest()[:8]}"


# Shared guidance for every "does the job violate X?" question. Written after
# the first backtest: these items fired on jobs the applicant went on to keep
# because ordinary job features (a full-time day schedule, a "5+ years
# preferred" line, local travel between sites) were read as violations.
_VIOLATION_FALSE = (
    "It doesn't, the posting doesn't say enough to tell, or the overlap is only "
    "superficial. Ordinary features of a typical job (a standard full-time "
    "weekday daytime schedule, occasional local travel between nearby sites) "
    "don't count unless the requirement names them, and a stated preference or "
    "'nice to have' is not a requirement.")


def build_questions(profile: dict) -> tuple[dict, dict]:
    """Questions for this applicant, plus meta describing each one so the
    policy and the UI know what a question id means. meta[qid] =
    {kind: hard|anti|soft|core|flag, label, weight, severity}."""
    crit = profile.get("criteria") or {}
    hard = crit.get("hard_filters") or {}
    qs: dict = {}
    meta: dict = {}

    def add(qid, q, kind, label, weight=None, severity=None):
        qs[qid] = q
        meta[qid] = {"kind": kind, "label": label, "weight": weight, "severity": severity}

    if hard.get("location"):
        add("hard_location", _noul(
            f'The job\'s work location or remote/hybrid/on-site arrangement violates '
            f'the applicant\'s location requirement: "{hard["location"]}"',
            "The posting's location or work arrangement clearly falls outside the requirement",
            "The location fits the requirement, or the posting doesn't say enough to tell"),
            "hard", f"location: {hard['location']}", severity="dealbreaker")
    if hard.get("salary_floor"):
        add("hard_salary", _noul(
            f'The posting states pay, and even the top of its range is below the '
            f'applicant\'s floor of {hard["salary_floor"]}',
            "Stated pay is clearly below the floor",
            "Pay meets the floor, or no pay is stated"), "hard",
            f"salary floor {hard['salary_floor']}", severity="dealbreaker")
    for entry in hard.get("other") or []:
        text, sev = _item(entry)
        add(_qid("hard", text), _noul(
            f'The job conflicts with this requirement of the applicant\'s: "{text}"',
            "The posting explicitly states or clearly implies something that "
            "conflicts with the requirement", _VIOLATION_FALSE), "hard", text, severity=sev)
    for entry in crit.get("anti_criteria") or []:
        text, sev = _item(entry)
        add(_qid("anti", text), _noul(
            f'The job falls into this category the applicant has ruled out: "{text}"',
            "The posting explicitly states or clearly implies that the job's actual "
            "work or requirements fall into the excluded category", _VIOLATION_FALSE),
            "anti", text, severity=sev)
    for sp in crit.get("soft_preferences") or []:
        want = sp.get("want") if isinstance(sp, dict) else str(sp)
        weight = PREF_WEIGHTS.get((sp.get("weight") if isinstance(sp, dict) else "") or "", 2.0)
        add(_qid("soft", want), _noul(
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


def request_hash(job, profile: dict) -> str:
    state = build_state(profile, job)
    questions, _ = build_questions(profile)
    return _req_hash(state, questions, settings.role("gate").model)


def lookup(job, profile: dict) -> dict | None:
    """Cached answers for this job under this profile, or None (no model call)."""
    h = request_hash(job, profile)
    with db.connect() as conn:
        row = conn.execute("SELECT model, answers_json, meta_json FROM gate_evals "
                           "WHERE job_id=? AND req_hash=?", (job["id"], h)).fetchone()
    if not row:
        return None
    return {"answers": json.loads(row["answers_json"]), "meta": json.loads(row["meta_json"]),
            "model": row["model"], "cached": True, "req_hash": h}


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
                    "meta": json.loads(row["meta_json"]), "model": row["model"],
                    "cached": True, "req_hash": h}
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
    return {"answers": data["answers"], "meta": meta, "model": model, "cached": False,
            "req_hash": h}


# --- policy -------------------------------------------------------------------
#
# Two layers, deliberately separate:
#   1. Vetoes — declared, not learned. A dealbreaker (hard filter or anti-
#      criterion with severity "dealbreaker") answered at p >= VETO_P, or a
#      high scam rating, caps the score at VETO_CAP. Rare dealbreakers almost
#      never appear in your decisions, so no fit could learn how much they
#      matter; they have to be stated.
#   2. Fit blend — learned. A logistic model over a handful of aggregate
#      features, refit from your decisions by gatefit. The gate score is its
#      probability x 100: "how likely you'd keep this one".
# Deadline-passed is neither: it's a freshness signal (see main._freshness).

SENIORITY_VALUE = [0.0, 0.5, 1.0, 0.7]  # far above, stretch, fits, overqualified
SCAM_LABELS = ["none", "low", "medium", "high"]
VETO_P = 0.8
VETO_CAP = 2.0
FIT_FEATURES = ["role", "qualified", "seniority", "soft", "penalty", "thin"]

# Used until the first `gate-fit` is applied: roughly the weights the first
# backtest learned from 1,165 decisions (Oct 2026).
DEFAULT_MODEL = {
    "version": 0,
    "weights": {"bias": -1.6, "role": 1.8, "qualified": 1.4, "seniority": 1.3,
                "soft": 0.5, "penalty": -0.6, "thin": -1.4},
    "applicant_bias": {},
    "thresholds": {"gloss_min": 40.0, "review_min": 20.0},
}

MODEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS gate_models (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,          -- candidate | active | retired | discarded
    weights_json TEXT NOT NULL,    -- {weights, applicant_bias}
    thresholds_json TEXT NOT NULL, -- {gloss_min, review_min}
    stats_json TEXT NOT NULL,      -- preview: AUCs, recall table, queue moves, flags
    jev_model TEXT,
    labels_through TEXT            -- latest reviewed_at among the decisions used
);
"""


def _expect(ans: dict, values: list[float]) -> float:
    return sum(float(p) * values[int(k)] for k, p in ans["probabilities"].items())


def _norm_score(ans: dict) -> float:
    n = len(ans["probabilities"]) - 1
    return float(ans["score"]) / n if n else 0.0


def _nouls(answers, meta, kinds, severity=None):
    return [(qid, float(answers[qid]["noul"])) for qid, m in meta.items()
            if m["kind"] in kinds and qid in answers
            and (severity is None or (m.get("severity") or "dealbreaker") == severity)]


def features(answers: dict, meta: dict) -> dict:
    """Collapse per-question answers into a fixed feature set, comparable across
    applicants (and profile versions) whose question lists differ."""
    soft = _nouls(answers, meta, ("soft",))
    sw = [(meta[q]["weight"] or 2.0, p) for q, p in soft]
    pen = _nouls(answers, meta, ("hard", "anti"), "penalty")
    none_fire = 1.0
    for _, p in pen:
        none_fire *= 1 - p
    return {
        "role": _norm_score(answers["role"]),
        "qualified": _norm_score(answers["qualified"]),
        "seniority": _expect(answers["seniority"], SENIORITY_VALUE),
        "soft": (sum(w * p for w, p in sw) / sum(w for w, _ in sw)) if sw else 0.5,
        "penalty": 1 - none_fire,  # P(at least one penalty item applies)
        "thin": float(answers["thin"]["noul"]),
    }


def vetoes(answers: dict, meta: dict) -> list[str]:
    """Labels of the dealbreakers this job trips (empty = no veto)."""
    out = [meta[q]["label"] for q, p in
           _nouls(answers, meta, ("hard", "anti"), "dealbreaker") if p >= VETO_P]
    if "scam" in answers and float(answers["scam"]["probabilities"].get("3", 0)) >= 0.5:
        out.append("clear scam markers")
    return out


def raw_logit(f: dict, model: dict, applicant_id: str | None = None) -> float:
    w = model["weights"]
    z = w.get("bias", 0.0) + sum(w.get(k, 0.0) * f[k] for k in FIT_FEATURES)
    return z + model.get("applicant_bias", {}).get(applicant_id or "", 0.0)


def combine(answers: dict, meta: dict, model: dict | None = None,
            applicant_id: str | None = None) -> dict:
    model = model or active_model()
    f = features(answers, meta)
    z = raw_logit(f, model, applicant_id)
    score = 100 / (1 + math.exp(-max(-30, min(30, z))))
    vetoed = vetoes(answers, meta)
    if vetoed:
        score = min(score, VETO_CAP)
    scam_probs = answers["scam"]["probabilities"]
    return {
        "gate_score": round(score, 1), "features": f, "vetoes": vetoed,
        "scam_risk": SCAM_LABELS[int(max(scam_probs, key=lambda k: float(scam_probs[k])))],
        "deadline_passed": float(answers.get("deadline_passed", {}).get("noul", 0.0)),
    }


# --- model store -----------------------------------------------------------------

def _row_to_model(r) -> dict:
    w = json.loads(r["weights_json"])
    return {"version": r["version"], "weights": w["weights"],
            "applicant_bias": w.get("applicant_bias", {}),
            "thresholds": json.loads(r["thresholds_json"]),
            "stats": json.loads(r["stats_json"]), "status": r["status"],
            "created_at": r["created_at"], "jev_model": r["jev_model"],
            "labels_through": r["labels_through"]}


def active_model() -> dict:
    with db.connect() as conn:
        conn.executescript(MODEL_SCHEMA)
        r = conn.execute("SELECT * FROM gate_models WHERE status='active' "
                         "ORDER BY version DESC LIMIT 1").fetchone()
    return _row_to_model(r) if r else DEFAULT_MODEL


def get_model(version: int) -> dict | None:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM gate_models WHERE version=?", (version,)).fetchone()
    return _row_to_model(r) if r else None


def list_models(limit: int = 10) -> list[dict]:
    with db.connect() as conn:
        conn.executescript(MODEL_SCHEMA)
        rows = conn.execute("SELECT * FROM gate_models WHERE status != 'discarded' "
                            "ORDER BY version DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_model(r) for r in rows]


def save_candidate(weights: dict, applicant_bias: dict, thresholds: dict,
                   stats: dict, jev_model: str | None, labels_through: str | None) -> int:
    with db.connect() as conn:
        conn.executescript(MODEL_SCHEMA)
        # one pending candidate at a time — a new preview replaces the old one
        conn.execute("UPDATE gate_models SET status='discarded' WHERE status='candidate'")
        cur = conn.execute(
            "INSERT INTO gate_models (created_at, status, weights_json, thresholds_json, "
            "stats_json, jev_model, labels_through) VALUES (?,?,?,?,?,?,?)",
            (db.now(), "candidate",
             json.dumps({"weights": weights, "applicant_bias": applicant_bias}),
             json.dumps(thresholds), json.dumps(stats), jev_model, labels_through))
        return cur.lastrowid


def activate(version: int) -> None:
    """Make `version` the live model (apply a candidate, or roll back)."""
    with db.connect() as conn:
        if not conn.execute("SELECT 1 FROM gate_models WHERE version=? AND status != "
                            "'discarded'", (version,)).fetchone():
            raise ValueError(f"no gate model v{version}")
        conn.execute("UPDATE gate_models SET status='retired' WHERE status='active'")
        conn.execute("UPDATE gate_models SET status='active' WHERE version=?", (version,))


def discard(version: int) -> None:
    with db.connect() as conn:
        conn.execute("UPDATE gate_models SET status='discarded' WHERE version=? "
                     "AND status='candidate'", (version,))


def top_reasons(answers: dict, meta: dict, n: int = 3) -> tuple[list[str], list[str]]:
    """(for, against): the most decisive answers in plain words, for a card that
    has no written gloss. Nouls above 0.5 only; ordered by strength."""
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
        elif m["kind"] in ("hard", "anti"):
            tag = "ruled out" if (m.get("severity") or "dealbreaker") == "dealbreaker" \
                and p >= VETO_P else "concern"
            cons.append((p + (1 if tag == "ruled out" else 0), f"{tag}: {m['label']}"))
        elif m["kind"] == "flag":
            cons.append((p, m["label"]))
    for qid, label in (("role", "role match"), ("qualified", "meets requirements")):
        if qid in answers:
            legend = answers[qid].get("legend") or {}
            probs = answers[qid]["probabilities"]
            top = max(probs, key=lambda k: float(probs[k]))
            (pros if int(top) >= 2 else cons).append((1.0, f"{label}: {legend.get(top, top)}"))
    pros.sort(reverse=True)
    cons.sort(reverse=True)
    return [t for _, t in pros[:n]], [t for _, t in cons[:n]]
