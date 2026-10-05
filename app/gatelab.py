"""Gate lab: troubleshooting views for the Jev gate.

Everything here is explained from stored data — cached Jev answers, the
active model's weights, profile versions — never by a language model writing
an after-the-fact story. Every sentence in an explanation corresponds to a
number on the same page. The one paid action is the on-demand evidence
lookup, which is cached.

  inspect(job_id)            "why this score": breakdown, every answer, history
  find_evidence(...)         which posting line made an item fire (cached)
  criteria_view(...)         what each criterion fires on, per applicant
  compute_change_report(...) what a profile version change did; snapshots
"""
import hashlib
import json
import math
import random
import re
from collections import defaultdict

from rapidfuzz import fuzz

from . import db, gate
from .config import settings
from .gatefit import POSITIVE, auc, closure_note, fit_logistic

FEATURE_LABELS = {
    "role": "role match", "qualified": "meets requirements", "seniority": "seniority fit",
    "soft": "preferences (weighted average)", "penalty": "penalty items (chance any applies)",
    "thin": "thin posting",
}
SOFT_HIT = 0.5      # a preference / penalty "counts" from here
NEAR_MISS = 0.5     # dealbreakers between this and VETO_P are "close calls"

SCHEMA = """
CREATE TABLE IF NOT EXISTS gate_evidence (
    key TEXT PRIMARY KEY,          -- hash(posting text + question wording)
    job_id TEXT NOT NULL,
    qid TEXT NOT NULL,
    result_json TEXT NOT NULL,     -- {exists, lines: [{n, text, p}], truncated}
    model TEXT, cost_usd REAL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gate_change_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id TEXT NOT NULL,
    from_version INTEGER NOT NULL,
    to_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    report_json TEXT NOT NULL
);
"""


def init_db():
    gate.init_db()
    with db.connect() as conn:
        conn.executescript(SCHEMA)


# --- shared helpers ---------------------------------------------------------------

def bucket(score: float, vetoed: bool, th: dict) -> str:
    if vetoed:
        return "skip"
    return "gloss" if score >= th["gloss_min"] else "review" if score >= th["review_min"] else "skip"


def ds_bucket(j) -> str | None:
    """What the DeepSeek path did with this job, in the same vocabulary."""
    if j["score"] is None:
        return None
    if j["score"] < settings.review_min_score:
        return "skip"
    if j["escalated"] or j["score"] >= settings.escalate_min_score:
        return "gloss"
    return "review"


def label_of(j) -> int | None:
    """1 kept, 0 rejected (fit judgment), None otherwise."""
    if j["status"] in POSITIVE:
        return 1
    if j["status"] == "rejected" and not closure_note(j["feedback"]):
        return 0
    return None


def _answers_for(job_id: str) -> list[dict]:
    """Every distinct answer set stored for a job, oldest first."""
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM gate_evals WHERE job_id=? ORDER BY created_at",
                            (job_id,)).fetchall()
    return [{"req_hash": r["req_hash"], "profile_version": r["profile_version"],
             "model": r["model"], "created_at": r["created_at"],
             "answers": json.loads(r["answers_json"]), "meta": json.loads(r["meta_json"])}
            for r in rows]


def latest_by_version(applicant_id: str, version: int) -> dict:
    """{job_id: answer set} using each job's newest answers under `version`."""
    out = {}
    with db.connect() as conn:
        for r in conn.execute("SELECT job_id, answers_json, meta_json, created_at FROM gate_evals "
                              "WHERE applicant_id=? AND profile_version=? ORDER BY created_at",
                              (applicant_id, version)):
            out[r["job_id"]] = {"answers": json.loads(r["answers_json"]),
                                "meta": json.loads(r["meta_json"])}
    return out


def _p(ans: dict) -> float:
    return float(ans.get("noul", 0.0))


def _top_level(ans: dict) -> tuple[str, float]:
    probs = ans["probabilities"]
    k = max(probs, key=lambda x: float(probs[x]))
    return (ans.get("legend") or {}).get(k, k), float(probs[k])


# --- job inspector ------------------------------------------------------------------

def explain(answers: dict, meta: dict, model: dict, applicant_id: str) -> dict:
    """Score breakdown + per-question table + plain-words sentences for one
    answer set under one model."""
    c = gate.combine(answers, meta, model, applicant_id)
    f = c["features"]
    w = model["weights"]
    rows = [{"label": "starting point (bias)", "value": None, "weight": None,
             "contrib": w.get("bias", 0.0)}]
    for k in gate.FIT_FEATURES:
        rows.append({"key": k, "label": FEATURE_LABELS[k], "value": f[k],
                     "weight": w.get(k, 0.0), "contrib": w.get(k, 0.0) * f[k]})
    ab = model.get("applicant_bias", {}).get(applicant_id, 0.0)
    rows.append({"label": f"applicant bias ({applicant_id})", "value": None, "weight": None,
                 "contrib": ab})
    logit = sum(r["contrib"] for r in rows)
    prob = 100 / (1 + math.exp(-max(-30, min(30, logit))))

    # every question
    qrows = []
    for q, m in meta.items():
        a = answers.get(q)
        if a is None:
            continue
        row = {"qid": q, "kind": m["kind"], "label": m["label"],
               "severity": m.get("severity"), "weight": m.get("weight")}
        if a.get("type") == "noul":
            p = _p(a)
            row["p"] = p
            if m["kind"] in ("hard", "anti"):
                sev = m.get("severity") or "dealbreaker"
                row["severity"] = sev
                row["status"] = ("VETO" if sev == "dealbreaker" and p >= gate.VETO_P else
                                 "close call" if sev == "dealbreaker" and p >= NEAR_MISS else
                                 "counts" if sev == "penalty" and p >= SOFT_HIT else "")
                row["order"] = (0 if row["status"] == "VETO" else 1, -p)
            elif m["kind"] == "soft":
                row["status"] = "matched" if p >= SOFT_HIT else ""
                row["order"] = (3, -p * (m.get("weight") or 2.0))
            else:
                row["status"] = "flag" if p >= SOFT_HIT else ""
                row["order"] = (2, -p)
        else:  # score
            probs = a["probabilities"]
            legend = a.get("legend") or {}
            row["levels"] = [{"label": legend.get(k, k), "p": float(probs[k])}
                             for k in sorted(probs, key=int)]
            row["top"], row["top_p"] = _top_level(a)
            row["order"] = (2, 0)
        qrows.append(row)
    qrows.sort(key=lambda r: r["order"])

    th = model["thresholds"]
    vetoed = bool(c["vetoes"])
    b = bucket(c["gate_score"], vetoed, th)
    words = []
    db_rows = [r for r in qrows if r["kind"] in ("hard", "anti")
               and (r.get("severity") or "dealbreaker") == "dealbreaker"]
    if vetoed:
        for r in db_rows:
            if r["status"] == "VETO":
                words.append(f"Vetoed: Jev is {r['p']:.0%} sure of “{r['label']}”, a "
                             f"dealbreaker (veto at {gate.VETO_P:.0%}), so the score is capped "
                             f"at {gate.VETO_CAP:g}.")
        if "clear scam markers" in c["vetoes"]:
            words.append("Vetoed: Jev rates the posting as showing clear scam markers.")
    else:
        closest = max(db_rows, key=lambda r: r["p"], default=None)
        words.append("No dealbreaker fired." + (
            f" The closest was “{closest['label']}” at {closest['p']:.0%} "
            f"(veto at {gate.VETO_P:.0%})." if closest and closest["p"] >= 0.2 else ""))
    contrib = sorted([r for r in rows if r.get("key")], key=lambda r: r["contrib"])

    def describe(r):
        k = r["key"]
        if k in ("role", "qualified", "seniority"):
            top, tp = _top_level(answers[k])
            return f"{r['label']}: “{top}” ({tp:.0%})"
        if k == "soft":
            hits = [x["label"] for x in qrows if x["kind"] == "soft" and x.get("status")]
            return f"{r['label']} {r['value']:.2f}" + (f": {len(hits)} matched" if hits else "")
        if k == "penalty":
            hits = [x["label"] for x in qrows if x.get("status") == "counts"]
            return f"{r['label']}: " + ("; ".join(hits) if hits else f"{r['value']:.2f}")
        return f"{r['label']} ({r['value']:.0%})"
    lifts = [describe(r) for r in reversed(contrib) if r["contrib"] > 0.15][:2]
    drags = [describe(r) for r in contrib if r["contrib"] < -0.15][:2]
    if lifts:
        words.append("Biggest lifts: " + "; ".join(lifts) + ".")
    if drags:
        words.append("Biggest drags: " + "; ".join(drags) + ".")
    if not vetoed:
        words.append(
            f"Score {c['gate_score']:.0f} is {'above' if c['gate_score'] >= th['review_min'] else 'below'} "
            f"the review floor ({th['review_min']}) and "
            f"{'above' if c['gate_score'] >= th['gloss_min'] else 'below'} the gloss threshold "
            f"({th['gloss_min']}), so Jev would {b}.")
    if c["deadline_passed"] >= 0.8:
        words.append(f"Jev reads a stated application deadline that has passed "
                     f"({c['deadline_passed']:.0%}); the job shows in Review's stale view.")
    return {"breakdown": rows, "logit": logit, "prob": prob, "score": c["gate_score"],
            "vetoes": c["vetoes"], "bucket": b, "questions": qrows, "words": words,
            "scam_risk": c["scam_risk"], "thresholds": th}


def _diff_answers(old: dict, new: dict) -> list[dict]:
    """Questions whose answers moved meaningfully between two answer sets.
    Items match by their text (ids are text hashes); core questions by id."""
    def index(s):
        out = {}
        for q, m in s["meta"].items():
            key = (m["kind"], m["label"]) if m["kind"] in ("hard", "anti", "soft") else q
            out[key] = (q, m)
        return out
    io, inew = index(old), index(new)
    changes = []
    for key in sorted(set(io) | set(inew), key=str):
        if key not in io:
            q, m = inew[key]
            changes.append({"label": m["label"], "kind": m["kind"], "change": "added",
                            "after": _fmt(new["answers"].get(q))})
        elif key not in inew:
            q, m = io[key]
            changes.append({"label": m["label"], "kind": m["kind"], "change": "removed",
                            "before": _fmt(old["answers"].get(q))})
        else:
            (qo, m), (qn, _) = io[key], inew[key]
            a, b = old["answers"].get(qo), new["answers"].get(qn)
            if a is None or b is None:
                continue
            va, vb = _val(a), _val(b)
            if abs(va - vb) >= 0.15:
                changes.append({"label": m["label"], "kind": m["kind"], "change": "moved",
                                "before": _fmt(a), "after": _fmt(b)})
    return changes


def _val(a: dict) -> float:
    if a.get("type") == "noul":
        return _p(a)
    n = len(a["probabilities"]) - 1
    return float(a["score"]) / n if n else 0.0


def _fmt(a: dict | None) -> str:
    if a is None:
        return "–"
    if a.get("type") == "noul":
        return f"{_p(a):.2f}"
    top, tp = _top_level(a)
    return f"{top} ({tp:.0%})"


def inspect(job_id: str) -> dict | None:
    init_db()
    with db.connect() as conn:
        j = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if j is None:
            return None
        profile = db.latest_profile(conn, j["applicant_id"])
        pv = conn.execute("SELECT MAX(version) v FROM profiles WHERE applicant_id=?",
                          (j["applicant_id"],)).fetchone()["v"]
    model = gate.active_model()
    current = gate.lookup(j, profile) if profile else None
    history = _answers_for(job_id)
    shown = current or (history[-1] if history else None)
    out = {"job": j, "model": model, "profile_version": pv, "current": bool(current),
           "shown_version": shown.get("profile_version") if shown else None,
           "ds_bucket": ds_bucket(j), "label": label_of(j)}
    if shown:
        out["x"] = explain(shown["answers"], shown["meta"], model, j["applicant_id"])
        out["evidence"] = cached_evidence(j, shown["meta"]) if current else {}
    # history: one entry per distinct answer set, with the diff to the previous
    hist = []
    for i, h in enumerate(history):
        c = gate.combine(h["answers"], h["meta"], model, j["applicant_id"])
        hist.append({"profile_version": h["profile_version"], "created_at": h["created_at"],
                     "score": c["gate_score"], "vetoes": c["vetoes"],
                     "changes": _diff_answers(history[i - 1], h) if i else []})
    out["history"] = hist
    return out


# --- evidence lookup ------------------------------------------------------------------

MAX_LINES = 250  # Choice allows 255 options; one is "none"


def posting_lines(desc: str) -> tuple[list[str], bool]:
    """Split a posting into short, citable lines: by line breaks, then long
    lines by sentence. Drops fragments too short to carry meaning."""
    out = []
    for raw in (desc or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•*-])", raw) if len(raw) > 240 else [raw]
        out += [p.strip()[:400] for p in parts if len(p.strip()) >= 12]
    return out[:MAX_LINES], len(out) > MAX_LINES


def _evidence_key(desc: str, instructions: str) -> str:
    return hashlib.sha256((desc or "").encode() + b"\x00" + instructions.encode()).hexdigest()[:32]


def cached_evidence(j, meta: dict) -> dict:
    """{qid: result} for whatever has been looked up before (no calls)."""
    profile_qs = {}
    with db.connect() as conn:
        profile = db.latest_profile(conn, j["applicant_id"])
    qs, _ = gate.build_questions(profile)
    keys = {_evidence_key(j["description"], qs[q]["instructions"]): q for q in meta if q in qs}
    if not keys:
        return {}
    marks = ",".join("?" * len(keys))
    with db.connect() as conn:
        for r in conn.execute(f"SELECT key, result_json FROM gate_evidence WHERE key IN ({marks})",
                              list(keys)):
            profile_qs[keys[r["key"]]] = json.loads(r["result_json"])
    return profile_qs


def find_evidence(job_id: str, qid: str) -> dict:
    """Which line of the posting most directly shows what question `qid`
    asks about. Cached by posting text + question wording, so asking again
    (or after an unrelated profile edit) is free."""
    init_db()
    with db.connect() as conn:
        j = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        profile = db.latest_profile(conn, j["applicant_id"])
    qs, meta = gate.build_questions(profile)
    if qid not in qs:
        raise ValueError("that question isn't in the current profile")
    instr = qs[qid]["instructions"]
    key = _evidence_key(j["description"], instr)
    with db.connect() as conn:
        row = conn.execute("SELECT result_json FROM gate_evidence WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row["result_json"])
    lines, truncated = posting_lines(j["description"])
    if not lines:
        raise ValueError("this job has no posting text to search")
    ids = [f"L{i:03d}" for i in range(len(lines))]
    state = {"job": {"title": j["title"], "company": j["company"]},
             "posting": "\n".join(f"{i}| {t}" for i, t in zip(ids, lines))}
    statement = instr
    questions = {
        "where": {"type": "choice", "instructions":
                  f"Which line of `posting` most directly shows this statement is true: "
                  f"{statement} Choose \"none\" if no line shows it.",
                  "criteria": {**{i: None for i in ids},
                               "none": "No line of the posting shows this"}},
        "exists": {"type": "noul", "instructions":
                   f"Some line of `posting` directly shows this statement is true: {statement}"},
    }
    data = gate._post(state, questions)
    usage = data.get("usage") or {}
    cost = float(usage.get("cost") or 0.0)
    db.log_usage("gate-evidence", settings.role("gate").provider, data.get("model", "?"),
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0), cost)
    probs = data["answers"]["where"]["probabilities"]
    top = sorted(probs.items(), key=lambda kv: -float(kv[1]))[:3]
    result = {"exists": float(data["answers"]["exists"]["noul"]), "truncated": truncated,
              "lines": [{"n": k, "text": lines[int(k[1:])] if k != "none" else None,
                         "p": float(p)} for k, p in top if float(p) >= 0.05]}
    with db.connect() as conn:
        conn.execute("INSERT OR REPLACE INTO gate_evidence (key, job_id, qid, result_json, "
                     "model, cost_usd, created_at) VALUES (?,?,?,?,?,?,?)",
                     (key, job_id, qid, json.dumps(result), data.get("model"), cost, db.now()))
    return result


# --- criteria view ------------------------------------------------------------------------

def criteria_view(applicant_id: str, threshold: float = 0.8, item: str | None = None) -> dict:
    """What every criterion fires on, across jobs with answers under the
    applicant's current profile version. With `item`, the jobs it fired on."""
    init_db()
    with db.connect() as conn:
        profile = db.latest_profile(conn, applicant_id)
        pv = conn.execute("SELECT MAX(version) v FROM profiles WHERE applicant_id=?",
                          (applicant_id,)).fetchone()["v"]
        jobs = {r["id"]: r for r in conn.execute("SELECT * FROM jobs WHERE applicant_id=?",
                                                 (applicant_id,))}
    sets = latest_by_version(applicant_id, pv)
    _, meta = gate.build_questions(profile)
    items = [(q, m) for q, m in meta.items() if m["kind"] in ("hard", "anti", "soft")] + \
            [(q, meta[q]) for q in ("thin", "deadline_passed") if q in meta]
    groups = {"kept": 0, "rejected": 0, "other": 0}
    counts = defaultdict(lambda: {"kept": 0, "rejected": 0, "other": 0})
    applies = defaultdict(int)  # p >= 0.5 across all jobs, for the "fires on most jobs" check
    fired_jobs = []
    for jid, s in sets.items():
        j = jobs.get(jid)
        if j is None:
            continue
        lab = label_of(j)
        g = "kept" if lab == 1 else "rejected" if lab == 0 else "other"
        groups[g] += 1
        for q, m in items:
            a = s["answers"].get(q)
            if a is None:
                continue
            p = _p(a)
            counts[q][g] += p >= threshold
            applies[q] += p >= 0.5
            if item == q and p >= max(0.0, threshold - 0.3):
                fired_jobs.append({"job": j, "p": p, "fired": p >= threshold, "group": g})
    rows = []
    for q, m in items:
        c = counts[q]
        sev = (m.get("severity") or "dealbreaker") if m["kind"] in ("hard", "anti") else None
        kr = c["kept"] / max(groups["kept"], 1)
        rr = c["rejected"] / max(groups["rejected"], 1)
        flag = (sev == "dealbreaker" and c["kept"] >= 2 and (kr >= 0.05 or kr >= rr))
        share = applies[q] / max(len(sets), 1)
        broad = (gate.broad_candidate(m) and len(sets) >= gate.BROAD_MIN_JOBS
                 and share > gate.BROAD_SHARE)
        rows.append({"qid": q, "kind": m["kind"], "label": m["label"], "severity": sev,
                     "weight": m.get("weight"), **c, "kept_rate": kr, "rejected_rate": rr,
                     "flag": flag, "broad": broad, "applies_share": share})
    order = {"hard": 0, "anti": 1, "soft": 2, "flag": 3}
    rows.sort(key=lambda r: (order[r["kind"]], -r["kept_rate"]))
    fired_jobs.sort(key=lambda x: -x["p"])
    return {"applicant": applicant_id, "version": pv, "threshold": threshold, "rows": rows,
            "groups": groups, "n_jobs": len(sets), "item": item,
            "item_label": meta[item]["label"] if item in meta else None, "fired_jobs": fired_jobs}


# --- change reports ------------------------------------------------------------------------

def _versions_with_answers(applicant_id: str) -> list[int]:
    with db.connect() as conn:
        return [r["v"] for r in conn.execute(
            "SELECT DISTINCT profile_version v FROM gate_evals WHERE applicant_id=? "
            "AND profile_version IS NOT NULL ORDER BY v", (applicant_id,))]


def _profile_version(applicant_id: str, v: int) -> tuple[dict, str, str | None]:
    with db.connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(profiles)")}
        r = conn.execute("SELECT json, created_at" + (", note" if "note" in cols else "") +
                         " FROM profiles WHERE applicant_id=? AND version=?",
                         (applicant_id, v)).fetchone()
    return json.loads(r["json"]), r["created_at"], (r["note"] if "note" in r.keys() else None)


def _criteria_diff(old: dict, new: dict) -> dict:
    _, mo = gate.build_questions(old)
    _, mn = gate.build_questions(new)
    io = {(m["kind"], m["label"]): m for m in mo.values() if m["kind"] in ("hard", "anti", "soft")}
    inn = {(m["kind"], m["label"]): m for m in mn.values() if m["kind"] in ("hard", "anti", "soft")}
    removed = [k for k in io if k not in inn]
    added = [k for k in inn if k not in io]
    reworded = []
    for k in list(removed):
        best = max((a for a in added if a[0] == k[0]),
                   key=lambda a: fuzz.token_set_ratio(k[1], a[1]), default=None)
        if best and fuzz.token_set_ratio(k[1], best[1]) >= 60:
            reworded.append({"kind": k[0], "before": k[1], "after": best[1]})
            removed.remove(k)
            added.remove(best)
    changed = []
    for k in set(io) & set(inn):
        a, b = io[k], inn[k]
        if (a.get("severity"), a.get("weight")) != (b.get("severity"), b.get("weight")):
            changed.append({"kind": k[0], "label": k[1],
                            "before": a.get("severity") or a.get("weight"),
                            "after": b.get("severity") or b.get("weight")})
    other = []
    for section, ko, kn in (("profile", old.get("profile", {}), new.get("profile", {})),
                            ("criteria", old.get("criteria", {}), new.get("criteria", {}))):
        for key in sorted(set(ko) | set(kn)):
            if key in ("anti_criteria", "soft_preferences", "hard_filters", "base_resume_md",
                       "contact"):
                continue
            if ko.get(key) != kn.get(key):
                other.append(f"{section}.{key}" + (" (added)" if key not in ko else
                                                   " (removed)" if key not in kn else ""))
    ho, hn = old.get("criteria", {}).get("hard_filters", {}), new.get("criteria", {}).get("hard_filters", {})
    for key in ("location", "salary_floor"):
        if ho.get(key) != hn.get(key):
            other.append(f"hard_filters.{key}")
    return {"removed": [{"kind": k[0], "label": k[1]} for k in removed],
            "added": [{"kind": k[0], "label": k[1]} for k in added],
            "reworded": reworded, "changed": changed, "other": other}


def _cv_auc(xs: list[list[float]], ys: list[int], vetoed: list[bool]) -> float | None:
    """Cross-validated AUC the way the gate actually scores: the blend is fit
    on jobs no dealbreaker vetoed, and vetoed jobs get the veto cap."""
    keep = [i for i in range(len(xs)) if not vetoed[i]]
    yk = [ys[i] for i in keep]
    if sum(yk) < 5 or len(yk) - sum(yk) < 5:
        return None
    idx = keep[:]
    random.Random(1).shuffle(idx)
    score = {}
    for k in range(5):
        te = set(idx[k::5])
        w = fit_logistic([xs[i] for i in idx if i not in te], [ys[i] for i in idx if i not in te],
                         iters=1500)
        for i in te:
            z = w[0] + sum(a * b for a, b in zip(w[1:], xs[i]))
            score[i] = 100 / (1 + math.exp(-max(-30, min(30, z))))
    return auc([(score.get(i, gate.VETO_CAP), ys[i]) for i in range(len(xs))])


def compute_change_report(applicant_id: str, from_v: int, to_v: int) -> dict:
    """What changed between two profile versions, measured on jobs that have
    Jev answers under both. Scores use the active gate model for both sides,
    so differences come from the answers, not the weights."""
    init_db()
    old, _, _ = _profile_version(applicant_id, from_v)
    new, new_at, note = _profile_version(applicant_id, to_v)
    before, after = latest_by_version(applicant_id, from_v), latest_by_version(applicant_id, to_v)
    with db.connect() as conn:
        jobs = {r["id"]: r for r in conn.execute("SELECT * FROM jobs WHERE applicant_id=?",
                                                 (applicant_id,))}
        cost = conn.execute("SELECT COALESCE(SUM(cost_usd),0) c, COUNT(*) n FROM gate_evals "
                            "WHERE applicant_id=? AND profile_version=?",
                            (applicant_id, to_v)).fetchone()
    model = gate.active_model()
    common = [jid for jid in before if jid in after and jid in jobs]
    diff = _criteria_diff(old, new)

    # per item firing, before vs after, on decided jobs answered under both
    def rates(sets, label, kind):
        k = r = kn = rn = 0
        for jid in common:
            lab = label_of(jobs[jid])
            if lab is None:
                continue
            s = sets[jid]
            q = next((q for q, m in s["meta"].items() if m["label"] == label and m["kind"] == kind), None)
            if q is None:
                continue
            hit = _p(s["answers"][q]) >= (gate.VETO_P if kind in ("hard", "anti") else SOFT_HIT)
            if lab:
                k += hit; kn += 1
            else:
                r += hit; rn += 1
        return {"kept": k, "kept_n": kn, "rejected": r, "rejected_n": rn} if kn + rn else None
    item_rows = []
    _, mn = gate.build_questions(new)
    for m in mn.values():
        if m["kind"] not in ("hard", "anti", "soft"):
            continue
        prev_label = next((x["before"] for x in diff["reworded"] if x["after"] == m["label"]), m["label"])
        item_rows.append({"kind": m["kind"], "label": m["label"],
                          "reworded_from": prev_label if prev_label != m["label"] else None,
                          "severity": m.get("severity"),
                          "before": rates(before, prev_label, m["kind"]),
                          "after": rates(after, m["label"], m["kind"])})
    for x in diff["removed"]:
        item_rows.append({"kind": x["kind"], "label": x["label"], "removed": True,
                          "before": rates(before, x["label"], x["kind"]), "after": None})

    # outcomes that changed
    moved = []
    trans = defaultdict(int)
    feats_b, feats_a, ys, vb, va, jb, ja = [], [], [], [], [], {}, {}
    th = model["thresholds"]
    for jid in common:
        j = jobs[jid]
        cb = gate.combine(before[jid]["answers"], before[jid]["meta"], model, applicant_id)
        ca = gate.combine(after[jid]["answers"], after[jid]["meta"], model, applicant_id)
        bb, ba = bucket(cb["gate_score"], bool(cb["vetoes"]), th), bucket(ca["gate_score"], bool(ca["vetoes"]), th)
        if bb != ba or bool(cb["vetoes"]) != bool(ca["vetoes"]):
            tag = lambda b_, c_: f"{b_} (vetoed)" if c_["vetoes"] else b_
            trans[f"{tag(bb, cb)} → {tag(ba, ca)}"] += 1
            moved.append({"id": jid, "title": j["title"], "company": j["company"],
                          "status": j["status"], "before": round(cb["gate_score"], 1),
                          "after": round(ca["gate_score"], 1), "bucket_before": bb,
                          "bucket_after": ba, "vetoes_before": cb["vetoes"],
                          "vetoes_after": ca["vetoes"]})
        lab = label_of(j)
        if lab is not None:
            feats_b.append([cb["features"][k] for k in gate.FIT_FEATURES])
            feats_a.append([ca["features"][k] for k in gate.FIT_FEATURES])
            ys.append(lab)
            vb.append(bool(cb["vetoes"]))
            va.append(bool(ca["vetoes"]))
            jb[jid], ja[jid] = cb, ca
    moved.sort(key=lambda m: (m["status"] not in POSITIVE, -abs(m["after"] - m["before"])))
    decided = [jid for jid in common if label_of(jobs[jid]) is not None]
    kept_vetoed = {"before": sum(1 for jid in decided if label_of(jobs[jid]) and jb[jid]["vetoes"]),
                   "after": sum(1 for jid in decided if label_of(jobs[jid]) and ja[jid]["vetoes"])}
    fair = [jid for jid in decided if jobs[jid]["score"] is not None]
    report = {
        "applicant": applicant_id, "from_version": from_v, "to_version": to_v,
        "to_created_at": new_at, "note": note, "computed_at": db.now(),
        "gate_model": model["version"], "criteria": diff, "items": item_rows,
        "coverage": {"common": len(common), "decided": len(decided),
                     "only_before": len(before) - len(common), "only_after": len(after) - len(common)},
        "kept_vetoed": kept_vetoed,
        "auc": {"before_cv": _cv_auc(feats_b, ys, vb), "after_cv": _cv_auc(feats_a, ys, va),
                "deepseek_fair": auc([(jobs[j]["score"], label_of(jobs[j])) for j in fair]),
                "n_fair": len(fair)},
        "transitions": dict(trans), "moved": moved[:60], "n_moved": len(moved),
        "reask_cost": round(cost["c"], 4), "reask_calls": cost["n"],
    }
    return report


def save_report(report: dict) -> int:
    init_db()
    with db.connect() as conn:
        cur = conn.execute("INSERT INTO gate_change_reports (applicant_id, from_version, "
                           "to_version, created_at, report_json) VALUES (?,?,?,?,?)",
                           (report["applicant"], report["from_version"], report["to_version"],
                            db.now(), json.dumps(report)))
        return cur.lastrowid


def list_reports(applicant_id: str | None = None) -> list[dict]:
    init_db()
    with db.connect() as conn:
        q = "SELECT id, applicant_id, from_version, to_version, created_at FROM gate_change_reports"
        args = []
        if applicant_id:
            q += " WHERE applicant_id=?"
            args.append(applicant_id)
        return [dict(r) for r in conn.execute(q + " ORDER BY id DESC", args)]


def get_report(report_id: int) -> dict | None:
    with db.connect() as conn:
        r = conn.execute("SELECT report_json FROM gate_change_reports WHERE id=?",
                         (report_id,)).fetchone()
    return json.loads(r["report_json"]) if r else None


def default_pair(applicant_id: str) -> tuple[int, int] | None:
    """The two most recent profile versions that both have Jev answers."""
    vs = _versions_with_answers(applicant_id)
    return (vs[-2], vs[-1]) if len(vs) >= 2 else None


def snapshot_pending() -> list[int]:
    """Called when a refit is applied: for each applicant whose newest
    answered profile version has no change report yet, save one against the
    previous answered version. Returns new report ids."""
    init_db()
    ids = []
    with db.connect() as conn:
        apps = [r["applicant_id"] for r in conn.execute("SELECT DISTINCT applicant_id FROM gate_evals")]
    for a in apps:
        pair = default_pair(a)
        if not pair:
            continue
        with db.connect() as conn:
            done = conn.execute("SELECT 1 FROM gate_change_reports WHERE applicant_id=? AND "
                                "from_version=? AND to_version=?", (a, *pair)).fetchone()
        if not done:
            ids.append(save_report(compute_change_report(a, *pair)))
    return ids


# --- lab index -------------------------------------------------------------------------------

def overview(applicant_id: str | None, q: str = "") -> dict:
    """Starting points: vetoed jobs you kept or are reviewing, the sharpest
    disagreements with DeepSeek (skip vs surfaced), and a title search."""
    init_db()
    model = gate.active_model()
    th = model["thresholds"]
    with db.connect() as conn:
        sql, args = "SELECT * FROM jobs WHERE gate_score IS NOT NULL", []
        if applicant_id:
            sql += " AND applicant_id=?"
            args.append(applicant_id)
        rows = conn.execute(sql, args).fetchall()
        found = []
        if q:
            sql2, a2 = "SELECT * FROM jobs WHERE (title LIKE ? OR company LIKE ?)", [f"%{q}%", f"%{q}%"]
            if applicant_id:
                sql2 += " AND applicant_id=?"
                a2.append(applicant_id)
            found = conn.execute(sql2 + " ORDER BY created_at DESC LIMIT 40", a2).fetchall()
    vetoed, sharp = [], []
    for j in rows:
        info = json.loads(j["gate_json"] or "{}")
        vt = bool(info.get("vetoes"))
        jb = bucket(j["gate_score"], vt, th)
        if vt and j["status"] in ("pending_user_review", "shortlisted", "applied", "archived"):
            vetoed.append({"job": j, "vetoes": info.get("vetoes")})
        d = ds_bucket(j)
        if d and (d == "skip") != (jb == "skip"):
            sharp.append({"job": j, "ds": d, "jev": jb})
    status_rank = {"shortlisted": 0, "applied": 0, "archived": 1, "pending_user_review": 2, "rejected": 3}
    sharp.sort(key=lambda x: (status_rank.get(x["job"]["status"], 4), -(x["job"]["gate_score"] or 0)))
    return {"vetoed": vetoed, "sharp": sharp[:40], "n_sharp": len(sharp), "found": found, "q": q}
