"""Gate backtest: would the Jev gate have ranked jobs the way you did?

Runs the gate over every job you've made a decision on (positives: applied,
shortlisted, archived — "this was worth a look"; negatives: rejected), plus an
unlabeled sample of auto-skipped jobs and optionally the current review
backlog. Compares against the stored DeepSeek score and writes a Markdown
report: ranking quality (AUC), recall-vs-threshold, a weight fit learned from
your decisions, and the worst misses to eyeball.

Gate answers are cached (gate_evals), so re-running after a weight change
costs nothing; only new or changed jobs/profiles call the model.
"""
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import db, gate

POSITIVE = ("applied", "shortlisted", "archived")


def _auc(pairs: list[tuple[float, int]]) -> float | None:
    """P(random positive outranks random negative), ties count half."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda t: t[0])
    # average ranks for ties (Mann-Whitney U)
    ranks, i = {}, 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(ranks[k] for k, (_, y) in enumerate(ranked) if y == 1)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _fit_logistic(X: list[list[float]], y: list[int], l2: float = 0.01,
                  iters: int = 3000, lr: float = 0.3) -> list[float]:
    """Plain batch gradient descent; features are already ~0–1. Returns
    [bias, w1..wn]. Class-weighted so 1:7 positives don't get drowned out."""
    n, d = len(X), len(X[0])
    pos = sum(y)
    wpos, wneg = n / (2 * max(pos, 1)), n / (2 * max(n - pos, 1))
    w = [0.0] * (d + 1)
    for _ in range(iters):
        g = [0.0] * (d + 1)
        for xi, yi in zip(X, y):
            z = w[0] + sum(a * b for a, b in zip(w[1:], xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = (p - yi) * (wpos if yi else wneg)
            g[0] += e
            for j in range(d):
                g[j + 1] += e * xi[j]
        for j in range(d + 1):
            w[j] -= lr * (g[j] / n + (l2 * w[j] if j else 0))
    return w


def _predict(w, x):
    return w[0] + sum(a * b for a, b in zip(w[1:], x))


def _closure_note(feedback: str | None) -> bool:
    f = (feedback or "").lower()
    return any(w in f for w in ("closed", "no longer", "expired", "filled", "dead link", "404"))


def _select(conn, skipped_sample: int, include_backlog: bool, seed: int):
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE status IN ({','.join('?' * len(POSITIVE))}, 'rejected') "
        "AND description IS NOT NULL", POSITIVE).fetchall()
    # A reject whose note is just "closed" says the posting died, not that it
    # was a bad fit — leave those out rather than count them as negatives.
    jobs = [(r, 1 if r["status"] in POSITIVE else 0) for r in rows
            if not (r["status"] == "rejected" and _closure_note(r["feedback"]))]
    skipped = conn.execute("SELECT * FROM jobs WHERE status='skipped' "
                           "AND description IS NOT NULL").fetchall()
    random.Random(seed).shuffle(skipped := list(skipped))
    jobs += [(r, None) for r in skipped[:skipped_sample]]
    if include_backlog:
        jobs += [(r, None) for r in conn.execute(
            "SELECT * FROM jobs WHERE status='pending_user_review'").fetchall()]
    return jobs


def run(skipped_sample: int = 300, include_backlog: bool = True, workers: int = 8,
        limit: int | None = None, seed: int = 7, progress=print) -> str:
    gate.init_db()
    with db.connect() as conn:
        jobs = _select(conn, skipped_sample, include_backlog, seed)
        profiles, versions = {}, {}
        for r, _ in jobs:
            aid = r["applicant_id"]
            if aid not in profiles:
                profiles[aid] = db.latest_profile(conn, aid)
                v = conn.execute("SELECT MAX(version) v FROM profiles WHERE applicant_id=?",
                                 (aid,)).fetchone()
                versions[aid] = v["v"] if v else None
    jobs = [(r, y) for r, y in jobs if profiles.get(r["applicant_id"])]
    if limit:
        random.Random(seed).shuffle(jobs)
        jobs = jobs[:limit]

    results, errors, fresh = [], [], 0
    progress(f"[backtest] {len(jobs)} jobs "
             f"({sum(1 for _, y in jobs if y == 1)} positive, "
             f"{sum(1 for _, y in jobs if y == 0)} rejected, "
             f"{sum(1 for _, y in jobs if y is None)} unlabeled)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(gate.evaluate, r, profiles[r["applicant_id"]],
                            versions[r["applicant_id"]]): (r, y) for r, y in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r, y = futs[fut]
            try:
                ev = fut.result()
            except Exception as e:  # noqa: BLE001 — record and keep going
                errors.append((r, str(e)[:200]))
                if isinstance(e, gate.BudgetExceeded):
                    for f in futs:
                        f.cancel()
                    break
                continue
            fresh += not ev["cached"]
            c = gate.combine(ev["answers"], ev["meta"])
            results.append({"job": r, "y": y, "ev": ev, **c})
            if i % 50 == 0:
                progress(f"[backtest] {i}/{len(jobs)}")
    return _report(results, errors, fresh)


FEATS = ["role", "qualified", "seniority", "soft", "hard_max", "anti_max",
         "thin", "scam", "deadline_passed"]


def _report(results: list[dict], errors: list, fresh: int) -> str:
    lab = [r for r in results if r["y"] is not None]
    pos = [r for r in lab if r["y"] == 1]
    unl = [r for r in results if r["y"] is None]
    out = ["# Jev gate backtest\n",
           f"{len(results)} jobs evaluated ({fresh} new model calls, rest cached), "
           f"{len(errors)} errors. Labeled: {len(pos)} worth a look "
           f"(applied/shortlisted/archived), {len(lab) - len(pos)} rejected. "
           f"Unlabeled: {len(unl)} (skipped sample + current backlog).\n"]
    models = sorted({r["ev"]["model"] for r in results})
    out.append(f"Model: {', '.join(models)}\n")

    # 1. ranking quality
    out.append("## Ranking quality (AUC: chance a kept job outranks a rejected one)\n")
    out.append("| applicant | n | Jev gate | DeepSeek score |\n|---|---|---|---|")
    groups = {"all": lab}
    for r in lab:
        groups.setdefault(r["job"]["applicant_id"], []).append(r)
    for name, rs in groups.items():
        g = _auc([(r["gate_score"], r["y"]) for r in rs])
        d = _auc([(r["job"]["score"] or 0, r["y"]) for r in rs])
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"
        out.append(f"| {name} | {len(rs)} | {fmt(g)} | {fmt(d)} |")
    out.append("\n0.5 = coin flip, 1.0 = perfect. Caveat: every labeled job already "
               "passed DeepSeek's ≥50 review floor, so this measures ranking *within* "
               "that pool; the skipped sample below checks what's outside it.\n")

    # 2. per-feature signal
    out.append("## Which judgments carry signal (per-feature AUC)\n")
    out.append("| feature | AUC | mean (kept) | mean (rejected) |\n|---|---|---|---|")
    for f in FEATS:
        a = _auc([(r["features"][f], r["y"]) for r in lab])
        mp = sum(r["features"][f] for r in pos) / max(len(pos), 1)
        neg = [r for r in lab if r["y"] == 0]
        mn = sum(r["features"][f] for r in neg) / max(len(neg), 1)
        out.append(f"| {f} | {a:.3f} | {mp:.2f} | {mn:.2f} |" if a is not None else f"| {f} | - | | |")
    out.append("\nAUC below 0.5 means higher values go with *rejection* "
               "(expected for hard/anti/thin/scam/deadline).\n")

    # 3. learned weights (5-fold CV so the AUC isn't just memorization)
    if len(pos) >= 10 and len(lab) - len(pos) >= 10:
        X = [[r["features"][f] for f in FEATS] for r in lab]
        y = [r["y"] for r in lab]
        idx = list(range(len(lab)))
        random.Random(1).shuffle(idx)
        cv_pairs = []
        for k in range(5):
            test = set(idx[k::5])
            w = _fit_logistic([X[i] for i in idx if i not in test],
                              [y[i] for i in idx if i not in test])
            cv_pairs += [(_predict(w, X[i]), y[i]) for i in test]
        w = _fit_logistic(X, y)
        out.append("## Weights learned from your decisions (logistic regression)\n")
        out.append(f"Cross-validated AUC: **{_auc(cv_pairs):.3f}** "
                   "(compare to the hand-set gate above).\n")
        out.append("| feature | weight |\n|---|---|")
        out.append(f"| (bias) | {w[0]:+.2f} |")
        for f, wi in zip(FEATS, w[1:]):
            out.append(f"| {f} | {wi:+.2f} |")
        out.append("")

    # 4. threshold table
    out.append("## Threshold → what you'd see\n")
    out.append("For a gloss threshold T: share of kept jobs at/above T (recall), share of "
               "rejects below T (culled), share of skipped-sample jobs at/above T (newly "
               "surfaced).\n")
    skipped = [r for r in unl if r["job"]["status"] == "skipped"]
    neg = [r for r in lab if r["y"] == 0]
    out.append("| T | recall (kept) | rejects culled | skipped sample surfaced |\n|---|---|---|---|")
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
    for t in range(5, 85, 5):
        out.append(f"| {t} | {pct(sum(r['gate_score'] >= t for r in pos), len(pos))} | "
                   f"{pct(sum(r['gate_score'] < t for r in neg), len(neg))} | "
                   f"{pct(sum(r['gate_score'] >= t for r in skipped), len(skipped))} |")
    ds = sum((r["job"]["score"] or 0) >= 65 for r in pos)
    out.append(f"\nFor reference, DeepSeek's escalation threshold (65) glossed "
               f"{pct(ds, len(pos))} of kept jobs.\n")
    ps = sorted(r["gate_score"] for r in pos)
    for target in (0.90, 0.95, 1.0):
        if ps:
            t = ps[max(0, math.ceil(len(ps) * (1 - target)) - 1)] if target < 1 else ps[0]
            out.append(f"- Highest threshold keeping {target:.0%} of kept jobs: **{t:.1f}**")
    out.append("")

    # 5. eyeball lists
    def line(r):
        pros, cons = gate.top_reasons(r["ev"]["answers"], r["ev"]["meta"])
        j = r["job"]
        return (f"- **{r['gate_score']:.0f}** (DeepSeek {j['score']}) {j['title']} — "
                f"{j['company']} [{j['status']}]\n  - for: {'; '.join(pros) or '—'}\n"
                f"  - against: {'; '.join(cons) or '—'}"
                + (f"\n  - your feedback: {j['feedback']}" if j["feedback"] else ""))
    out.append("## Kept jobs the gate scored lowest (misses)\n")
    out += [line(r) for r in sorted(pos, key=lambda r: r["gate_score"])[:15]]
    out.append("\n## Rejected jobs the gate scored highest\n")
    out += [line(r) for r in sorted(neg, key=lambda r: -r["gate_score"])[:15]]
    out.append("\n## Skipped jobs the gate would have surfaced (DeepSeek said < 50)\n")
    out += [line(r) for r in sorted(skipped, key=lambda r: -r["gate_score"])[:15]]
    if errors:
        out.append("\n## Errors\n")
        out += [f"- {r['id']} {r['title']}: {e}" for r, e in errors[:20]]
    return "\n".join(out)

