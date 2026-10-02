"""gate-fit: relearn how the gate weighs Jev's answers, from your decisions.

Never calls Jev for anything that's already cached and never changes the
questions — it refits the small logistic blend in gate.py (role, qualified,
seniority, soft, penalty, thin + a per-applicant bias), re-derives the gloss
and review thresholds for the recall targets, and saves the result as a
*candidate* with a preview of what would change. Nothing takes effect until
you apply it; applied versions are kept so rollback is one click.

Dealbreaker vetoes are deliberately outside the fit (see gate.py); the
preview instead flags dealbreakers that keep firing on jobs you kept, as
input for your next profile revision.
"""
import json
import math
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import db, gate
from .config import settings

POSITIVE = ("applied", "shortlisted", "archived")
MIN_NEW_DECISIONS = 30
MIN_FEATURE_ROWS = 10  # training rows where a feature is >= 0.05 before it's learned


def _held_weight(feature: str, current: dict) -> float:
    """Weight to hold an unlearnable feature at: the current model's, unless
    that's ~0 because it was never learned either, then the built-in default."""
    w = current["weights"].get(feature, 0.0)
    if abs(w) < 1e-3:
        w = gate.DEFAULT_MODEL["weights"].get(feature, 0.0)
    return w
_FIT_LOCK = threading.Lock()


def closure_note(feedback: str | None) -> bool:
    """A reject whose note says the posting died isn't a fit judgment."""
    f = (feedback or "").lower()
    return any(w in f for w in ("closed", "no longer", "expired", "filled", "dead link", "404"))


# --- logistic regression (tiny, dependency-free) ------------------------------

def fit_logistic(X: list[list[float]], y: list[int], l2: float = 0.01,
                 iters: int = 3000, lr: float = 0.3,
                 offset: list[float] | None = None) -> list[float]:
    """Batch gradient descent, class-weighted so ~1:7 positives aren't drowned
    out. Features are ~0-1 already. `offset` is a fixed per-row addition to
    the logit (contributions of features held at a fixed weight). Returns
    [bias, w1..wn]."""
    n, d = len(X), len(X[0])
    pos = sum(y)
    wpos, wneg = n / (2 * max(pos, 1)), n / (2 * max(n - pos, 1))
    w = [0.0] * (d + 1)
    off = offset or [0.0] * n
    for _ in range(iters):
        g = [0.0] * (d + 1)
        for xi, yi, oi in zip(X, y, off):
            z = oi + w[0] + sum(a * b for a, b in zip(w[1:], xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = (p - yi) * (wpos if yi else wneg)
            g[0] += e
            for j in range(d):
                g[j + 1] += e * xi[j]
        for j in range(d + 1):
            w[j] -= lr * (g[j] / n + (l2 * w[j] if j else 0))
    return w


def auc(pairs: list[tuple[float, int]]) -> float | None:
    """P(random positive outranks random negative), ties count half."""
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda t: t[0])
    r_pos, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        r_pos += avg * sum(1 for k in range(i, j + 1) if ranked[k][1] == 1)
        i = j + 1
    return (r_pos - pos * (pos + 1) / 2) / (pos * neg)


def _sigmoid100(z: float) -> float:
    return 100 / (1 + math.exp(-max(-30, min(30, z))))


# --- labels ---------------------------------------------------------------------

def _decided_jobs(conn):
    marks = ",".join("?" * len(POSITIVE))
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE status IN ({marks}, 'rejected') "
        "AND description IS NOT NULL", POSITIVE).fetchall()
    return [(r, 1 if r["status"] in POSITIVE else 0) for r in rows
            if not (r["status"] == "rejected" and closure_note(r["feedback"]))]


def _profiles(conn):
    out = {}
    for p in db.all_applicants(conn):
        if p:
            v = conn.execute("SELECT MAX(version) v FROM profiles WHERE applicant_id=?",
                             (p["applicant_id"],)).fetchone()["v"]
            out[p["applicant_id"]] = (p, v)
    return out


def missing_count() -> int:
    """Decisions with no cached answers under the current profile (each would
    cost one Jev call, ~$0.00025)."""
    with db.connect() as conn:
        jobs, profs = _decided_jobs(conn), _profiles(conn)
    return sum(1 for r, _ in jobs if r["applicant_id"] in profs
               and gate.lookup(r, profs[r["applicant_id"]][0]) is None)


def collect(evaluate_missing: bool, workers: int = 8, progress=None) -> tuple[list, int]:
    """[(job, y, ev)] for every decision with answers under the current
    profile; evaluates missing ones first if asked. Returns (labels, n_missing)."""
    gate.init_db()
    with db.connect() as conn:
        jobs, profs = _decided_jobs(conn), _profiles(conn)
    jobs = [(r, y) for r, y in jobs if r["applicant_id"] in profs]
    have, todo = [], []
    for r, y in jobs:
        ev = gate.lookup(r, profs[r["applicant_id"]][0])
        (have.append((r, y, ev)) if ev else todo.append((r, y)))
    if evaluate_missing and todo:
        if progress:
            progress(f"evaluating {len(todo)} decisions with no current answers…")
        def one(item):
            r, y = item
            p, v = profs[r["applicant_id"]]
            return r, y, gate.evaluate(r, p, v)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r, y, ev in pool.map(one, todo):
                have.append((r, y, ev))
        todo = []
    return have, len(todo)


# --- fit + preview ----------------------------------------------------------------

def _bucket(score: float, th: dict) -> str:
    if score >= th["gloss_min"]:
        return "gloss"
    return "review" if score >= th["review_min"] else "skip"


def _threshold_for(scores_pos: list[float], recall: float) -> float:
    """Highest threshold that keeps at least `recall` of positives."""
    s = sorted(scores_pos, reverse=True)
    if not s:
        return 0.0
    k = max(1, math.ceil(recall * len(s)))
    return math.floor(s[k - 1] * 10) / 10


def fit(evaluate_missing: bool = False, progress=None) -> int:
    """Fit a candidate model and save it with its preview. Returns its version."""
    with _FIT_LOCK:
        labels, n_missing = collect(evaluate_missing, progress=progress)
        current = gate.active_model()
        applicants = sorted({r["applicant_id"] for r, _, _ in labels})

        rows = []
        for r, y, ev in labels:
            f = gate.features(ev["answers"], ev["meta"])
            rows.append({"job": r, "y": y, "ev": ev, "f": f,
                         "vetoes": gate.vetoes(ev["answers"], ev["meta"])})
        # Vetoed jobs are capped regardless of the blend; leave them out of
        # the fit so the blend learns from jobs it actually decides.
        train = [x for x in rows if not x["vetoes"]]
        if sum(x["y"] for x in train) < 10 or sum(1 - x["y"] for x in train) < 10:
            raise ValueError("not enough decisions to fit (need 10+ kept and 10+ rejected)")

        # A feature the data barely exercises can't be learned: the fit would
        # just write ~0 (e.g. "penalty" before any profile item is marked a
        # penalty). Hold those at the current model's weight instead — or the
        # built-in default if the current one never learned it either — so
        # adding penalty items later doesn't silently do nothing.
        held = {k: _held_weight(k, current) for k in gate.FIT_FEATURES
                if sum(1 for x in train if x["f"][k] >= 0.05) < MIN_FEATURE_ROWS}
        learn = [k for k in gate.FIT_FEATURES if k not in held]

        def vec(x):
            return [x["f"][k] for k in learn] + \
                   [1.0 if x["job"]["applicant_id"] == a else 0.0 for a in applicants]

        X, y = [vec(x) for x in train], [x["y"] for x in train]
        off = [sum(wk * x["f"][k] for k, wk in held.items()) for x in train]
        # 5-fold CV predictions: honest scores for AUC and threshold picking
        idx = list(range(len(train)))
        random.Random(1).shuffle(idx)
        cv = [0.0] * len(train)
        for k in range(5):
            test = set(idx[k::5])
            w = fit_logistic([X[i] for i in idx if i not in test],
                             [y[i] for i in idx if i not in test],
                             offset=[off[i] for i in idx if i not in test])
            for i in test:
                cv[i] = off[i] + w[0] + sum(a * b for a, b in zip(w[1:], X[i]))
        w = fit_logistic(X, y, offset=off)
        nf = len(learn)
        learned = dict(zip(learn, w[1:1 + nf]))
        weights = {"bias": round(w[0], 4),
                   **{k: round(learned[k] if k in learned else held[k], 4)
                      for k in gate.FIT_FEATURES}}
        applicant_bias = {a: round(v, 4) for a, v in zip(applicants, w[1 + nf:])}

        # Candidate scores for every labeled job: CV score for trained jobs,
        # veto cap for vetoed ones. Current model scored the same way.
        cand = {}
        for x, z in zip(train, cv):
            cand[x["job"]["id"]] = _sigmoid100(z)
        for x in rows:
            if x["vetoes"]:
                cand[x["job"]["id"]] = gate.VETO_CAP
        cur = {x["job"]["id"]: gate.combine(x["ev"]["answers"], x["ev"]["meta"], current,
                                            x["job"]["applicant_id"])["gate_score"]
               for x in rows}
        # Thresholds come from kept jobs the blend actually decides. Vetoed
        # ones score VETO_CAP whatever the threshold, so including them would
        # let one misfiring dealbreaker drag both thresholds to the floor;
        # their cost is reported separately (veto_loss + dealbreaker flags).
        pos_scores = [cand[x["job"]["id"]] for x in rows if x["y"] and not x["vetoes"]]
        thresholds = {"gloss_min": _threshold_for(pos_scores, settings.gate_gloss_recall),
                      "review_min": _threshold_for(pos_scores, settings.gate_review_recall)}

        stats = _preview(rows, cand, cur, current, thresholds, weights, applicant_bias)
        stats.update({
            "n_labels": len(rows), "n_pos": sum(x["y"] for x in rows),
            "n_vetoed": len(rows) - len(train), "n_missing": n_missing,
            "veto_loss": sum(1 for x in rows if x["y"] and x["vetoes"]),
            "held_features": {k: round(v, 4) for k, v in held.items()},
            "auc_candidate": auc([(cand[x["job"]["id"]], x["y"]) for x in rows]),
            "auc_current": auc([(cur[x["job"]["id"]], x["y"]) for x in rows]),
            "current_version": current["version"],
            "recall_targets": {"gloss": settings.gate_gloss_recall,
                               "review": settings.gate_review_recall},
        })
        stats["warnings"] = _warnings(rows, stats, current)
        jev_models = sorted({x["ev"]["model"] for x in rows})
        through = max((x["job"]["reviewed_at"] or "" for x in rows), default=None)
        return gate.save_candidate(weights, applicant_bias, thresholds, stats,
                                   ",".join(jev_models), through)


def _preview(rows, cand, cur, current, thresholds, weights, applicant_bias) -> dict:
    pos = [x for x in rows if x["y"]]
    neg = [x for x in rows if not x["y"]]
    pct = lambda a, b: round(100 * a / b) if b else None

    def at(scores, th):
        return {"recall": pct(sum(scores[x["job"]["id"]] >= th for x in pos), len(pos)),
                "culled": pct(sum(scores[x["job"]["id"]] < th for x in neg), len(neg))}
    table = [{"t": t, **at(cand, t)} for t in (5, 10, 15, 20, 30, 40, 50, 60, 70)]

    # What would move in the unreviewed pool (jobs the gate has scored).
    cand_model = {"weights": weights, "applicant_bias": applicant_bias}
    moves: dict = defaultdict(int)
    with db.connect() as conn:
        live = conn.execute(
            "SELECT j.id, j.applicant_id, g.answers_json a, g.meta_json m FROM jobs j "
            "JOIN gate_evals g ON g.job_id=j.id AND g.req_hash=j.gate_req "
            "WHERE j.status IN ('pending_user_review','skipped')").fetchall()
    for r in live:
        a, m = json.loads(r["a"]), json.loads(r["m"])
        old = _bucket(gate.combine(a, m, current, r["applicant_id"])["gate_score"],
                      current["thresholds"])
        new = _bucket(gate.combine(a, m, cand_model, r["applicant_id"])["gate_score"],
                      thresholds)
        if old != new:
            moves[f"{old}→{new}"] += 1

    # Dealbreakers that fire on jobs you kept: candidates for "penalty".
    fire: dict = defaultdict(lambda: {"kept": 0, "rejected": 0})
    kept_n: dict = defaultdict(int)
    rej_n: dict = defaultdict(int)
    for x in rows:
        aid = x["job"]["applicant_id"]
        kept_n[aid] += x["y"]
        rej_n[aid] += 1 - x["y"]
        for q, mt in x["ev"]["meta"].items():
            if mt["kind"] in ("hard", "anti") and (mt.get("severity") or "dealbreaker") == \
                    "dealbreaker" and float(x["ev"]["answers"].get(q, {}).get("noul", 0)) >= gate.VETO_P:
                fire[(aid, mt["label"])]["kept" if x["y"] else "rejected"] += 1
    # Flag a dealbreaker that vetoes 5%+ of kept jobs, or one that fires at
    # least as often on kept jobs as on rejected ones: then it isn't
    # separating anything, just losing jobs you want (e.g. a qualifier like
    # "without dual citizenship" being ignored, so it fires on any
    # "US citizen required").
    flags = [{"applicant": aid, "item": label, **c,
              "kept_pct": pct(c["kept"], kept_n[aid]),
              "rejected_pct": pct(c["rejected"], rej_n[aid])}
             for (aid, label), c in fire.items()
             if c["kept"] >= 2 and (c["kept"] >= 0.05 * kept_n[aid]
                                    or c["kept"] / max(kept_n[aid], 1)
                                    >= c["rejected"] / max(rej_n[aid], 1))]
    flags.sort(key=lambda f: -f["kept"])
    return {"table": table, "moves": dict(moves), "n_unreviewed_scored": len(live),
            "dealbreaker_flags": flags}


def _warnings(rows, stats, current) -> list[str]:
    out = []
    new = new_decisions(current)
    if current["version"] and new < MIN_NEW_DECISIONS:
        out.append(f"Only {new} new decisions since v{current['version']} — a refit "
                   "this soon mostly fits noise.")
    a_new, a_cur = stats["auc_candidate"], stats["auc_current"]
    if a_new is not None and a_cur is not None and a_new < a_cur - 0.005:
        out.append(f"The candidate ranks your decisions worse than the current model "
                   f"(AUC {a_new:.3f} vs {a_cur:.3f}).")
    models = {x["ev"]["model"] for x in rows}
    if len(models) > 1:
        out.append("Answers come from more than one Jev version (" + ", ".join(sorted(models))
                   + "). Re-run the backtest so they're consistent before trusting a fit.")
    elif current.get("jev_model") and models and current["jev_model"] not in models:
        out.append(f"Jev changed since the current model was fit ({current['jev_model']} → "
                   f"{next(iter(models))}). Re-run the backtest before refitting.")
    for k, v in stats.get("held_features", {}).items():
        out.append(f"Too few decisions involve '{k}' to learn its weight, so it's held "
                   f"at {v:+.2f} (from the current or default model) rather than fit.")
    if stats["veto_loss"]:
        out.append(f"{stats['veto_loss']} of {stats['n_pos']} kept jobs "
                   f"({100 * stats['veto_loss'] / max(stats['n_pos'], 1):.0f}%) are vetoed by a "
                   "dealbreaker, so they'd never reach you whatever the thresholds. See the "
                   "flagged dealbreakers below.")
    if stats["n_missing"]:
        out.append(f"{stats['n_missing']} decisions had no answers under the current "
                   "profile and were left out.")
    return out


# --- apply / nudge -----------------------------------------------------------------

def apply(version: int) -> int:
    """Activate a model and rescore every gate-scored job from cached answers
    (no model calls). Returns the number of jobs rescored. Statuses are not
    touched here — which status a score maps to is the pipeline's call."""
    gate.activate(version)
    model = gate.active_model()
    # A refit after a profile change is the natural moment to record what the
    # change did; snapshot a change report for any applicant missing one.
    try:
        from . import gatelab
        gatelab.snapshot_pending()
    except Exception as e:  # noqa: BLE001 — never block an apply on a report
        print(f"[gatelab] change-report snapshot failed: {e}")
    n = 0
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT j.id, j.applicant_id, j.gate_json, g.answers_json a, g.meta_json m "
            "FROM jobs j JOIN gate_evals g ON g.job_id=j.id AND g.req_hash=j.gate_req").fetchall()
        for r in rows:
            c = gate.combine(json.loads(r["a"]), json.loads(r["m"]), model, r["applicant_id"])
            summary = {**json.loads(r["gate_json"] or "{}"), "model_version": model["version"],
                       "vetoes": c["vetoes"], "scam_risk": c["scam_risk"]}
            conn.execute("UPDATE jobs SET gate_score=?, gate_json=? WHERE id=?",
                          (c["gate_score"], json.dumps(summary), r["id"]))
            n += 1
    return n


def new_decisions(model: dict | None = None) -> int:
    """Fit-eligible decisions made since `model` (default: active) was fit."""
    model = model or gate.active_model()
    since = model.get("labels_through") or ""
    marks = ",".join("?" * len(POSITIVE))
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT status, feedback FROM jobs WHERE status IN ({marks}, 'rejected') "
            "AND reviewed_at > ?", (*POSITIVE, since)).fetchall()
    return sum(1 for r in rows if not (r["status"] == "rejected" and closure_note(r["feedback"])))


def profiles_changed_since_fit(model: dict | None = None) -> list[str]:
    """Applicants whose latest profile version is newer than the active
    model's fit. A new profile changes the questions, so the weights,
    thresholds and cached answers are all out of date until a refit."""
    model = model or gate.active_model()
    if not model.get("version"):
        return []  # never fit: the decision count drives the first nudge
    with db.connect() as conn:
        rows = conn.execute("SELECT applicant_id, MAX(created_at) t FROM profiles "
                            "GROUP BY applicant_id").fetchall()
    return [r["applicant_id"] for r in rows if r["t"] > model["created_at"]]


def nudge() -> str | None:
    """Why it's time to refit (one line), or None."""
    try:
        model = gate.active_model()
        changed = profiles_changed_since_fit(model)
        n = new_decisions(model)
    except Exception:  # noqa: BLE001 — a nudge must never break a page
        return None
    reasons = []
    if changed:
        reasons.append(f"profile changed since the last gate fit ({', '.join(changed)})")
    if n >= MIN_NEW_DECISIONS:
        reasons.append(f"{n} new decisions since the last gate fit")
    return "; ".join(reasons).capitalize() if reasons else None


# --- background run for the web UI --------------------------------------------------

def start_background(evaluate_missing: bool) -> bool:
    if _FIT_LOCK.locked():
        return False

    def run():
        db.set_meta("gate_fit_status", json.dumps({"running": True, "msg": "fitting…"}))
        try:
            v = fit(evaluate_missing, progress=lambda m: db.set_meta(
                "gate_fit_status", json.dumps({"running": True, "msg": m})))
            db.set_meta("gate_fit_status", json.dumps({"running": False, "version": v}))
        except Exception as e:  # noqa: BLE001 — surface on the page
            db.set_meta("gate_fit_status", json.dumps({"running": False, "error": str(e)[:300]}))
    threading.Thread(target=run, daemon=True).start()
    return True


def status() -> dict:
    raw = db.get_meta("gate_fit_status")
    return json.loads(raw) if raw else {"running": False}
