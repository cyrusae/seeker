"""Gate backtest: how would the Jev gate have ranked jobs you've decided on?

Runs the gate over every decision (kept = applied/shortlisted/archived,
rejected minus "closed" notes), plus an unlabeled sample of auto-skipped jobs
and optionally the review backlog, filling the gate_evals cache. Reports
ranking quality against DeepSeek, which judgments carry signal, how often each
hard filter / anti-criterion fires on kept vs rejected jobs, and the worst
misses to eyeball.

Weights and thresholds are `gate-fit`'s job; run it after this (it reuses the
cache, so it's free).
"""
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import db, gate
from .gatefit import POSITIVE, auc, closure_note


def _select(conn, skipped_sample: int, include_backlog: bool, seed: int):
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE status IN ({','.join('?' * len(POSITIVE))}, 'rejected') "
        "AND description IS NOT NULL", POSITIVE).fetchall()
    jobs = [(r, 1 if r["status"] in POSITIVE else 0) for r in rows
            if not (r["status"] == "rejected" and closure_note(r["feedback"]))]
    skipped = list(conn.execute("SELECT * FROM jobs WHERE status='skipped' "
                                "AND description IS NOT NULL").fetchall())
    random.Random(seed).shuffle(skipped)
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

    model = gate.active_model()
    results, errors, fresh = [], [], 0
    progress(f"[backtest] {len(jobs)} jobs "
             f"({sum(1 for _, y in jobs if y == 1)} kept, "
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
            c = gate.combine(ev["answers"], ev["meta"], model, r["applicant_id"])
            results.append({"job": r, "y": y, "ev": ev, **c})
            if i % 100 == 0:
                progress(f"[backtest] {i}/{len(jobs)}")
    return _report(results, errors, fresh, model)


def _report(results, errors, fresh, model) -> str:
    lab = [r for r in results if r["y"] is not None]
    pos = [r for r in lab if r["y"] == 1]
    neg = [r for r in lab if r["y"] == 0]
    skipped = [r for r in results if r["y"] is None and r["job"]["status"] == "skipped"]
    fmt = lambda v: f"{v:.3f}" if v is not None else "-"
    out = ["# Jev gate backtest\n",
           f"{len(results)} jobs ({fresh} new model calls, rest cached), {len(errors)} errors. "
           f"Labeled: {len(pos)} kept, {len(neg)} rejected. Scored with gate model "
           f"v{model['version']}{' (built-in default)' if not model['version'] else ''}.\n",
           f"Jev: {', '.join(sorted({r['ev']['model'] for r in results}))}\n"]

    out.append("## Ranking quality (AUC: chance a kept job outranks a rejected one)\n")
    out.append("Compared only on jobs DeepSeek actually scored; jobs rejected before "
               "any eval would otherwise count as free wins for it.\n")
    out.append("| applicant | n | Jev gate | DeepSeek |\n|---|---|---|---|")
    groups = defaultdict(list)
    for r in lab:
        if r["job"]["score"] is not None:
            groups["all"].append(r)
            groups[r["job"]["applicant_id"]].append(r)
    for name, rs in groups.items():
        out.append(f"| {name} | {len(rs)} | {fmt(auc([(r['gate_score'], r['y']) for r in rs]))} "
                   f"| {fmt(auc([(r['job']['score'], r['y']) for r in rs]))} |")
    out.append(f"\nAll {len(lab)} decisions (incl. never-scored rejects): Jev gate "
               f"{fmt(auc([(r['gate_score'], r['y']) for r in lab]))}.\n")

    out.append("## Which judgments carry signal (per-feature AUC)\n")
    out.append("| feature | AUC | mean (kept) | mean (rejected) |\n|---|---|---|---|")
    for f in gate.FIT_FEATURES:
        a = auc([(r["features"][f], r["y"]) for r in lab])
        mp = sum(r["features"][f] for r in pos) / max(len(pos), 1)
        mn = sum(r["features"][f] for r in neg) / max(len(neg), 1)
        out.append(f"| {f} | {fmt(a)} | {mp:.2f} | {mn:.2f} |")
    vk = sum(bool(r["vetoes"]) for r in pos)
    vr = sum(bool(r["vetoes"]) for r in neg)
    out.append(f"\nVetoed by a dealbreaker: {vk} kept jobs "
               f"({100 * vk / max(len(pos), 1):.0f}%), {vr} rejected "
               f"({100 * vr / max(len(neg), 1):.0f}%).\n")

    out.append("## Hard filters & anti-criteria: how often each fires (p ≥ "
               f"{gate.VETO_P})\n")
    out.append("A dealbreaker that fires on jobs you kept is either worded too broadly "
               "or isn't really a dealbreaker; consider `\"severity\": \"penalty\"`.\n")
    out.append("| applicant | severity | kept | rejected | item |\n|---|---|---|---|---|")
    stat = defaultdict(lambda: [0, 0, 0, 0, "", ""])
    for r in lab:
        aid = r["job"]["applicant_id"]
        for q, m in r["ev"]["meta"].items():
            if m["kind"] not in ("hard", "anti"):
                continue
            s = stat[(aid, q)]
            s[4], s[5] = m["label"], m.get("severity") or "dealbreaker"
            hit = float(r["ev"]["answers"][q]["noul"]) >= gate.VETO_P
            if r["y"]:
                s[0] += hit; s[1] += 1
            else:
                s[2] += hit; s[3] += 1
    for (aid, _), (kh, kn, rh, rn, label, sev) in sorted(
            stat.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        out.append(f"| {aid} | {sev} | {100 * kh / max(kn, 1):.0f}% | "
                   f"{100 * rh / max(rn, 1):.0f}% | {label[:110]} |")

    def line(r):
        pros, cons = gate.top_reasons(r["ev"]["answers"], r["ev"]["meta"])
        j = r["job"]
        return (f"- **{r['gate_score']:.0f}** (DeepSeek {j['score']}) {j['title']} — "
                f"{j['company']} [{j['status']}]\n  - for: {'; '.join(pros) or '—'}\n"
                f"  - against: {'; '.join(cons) or '—'}"
                + (f"\n  - your feedback: {j['feedback']}" if j["feedback"] else ""))
    out.append("\n## Kept jobs the gate scored lowest (misses)\n")
    out += [line(r) for r in sorted(pos, key=lambda r: r["gate_score"])[:15]]
    out.append("\n## Rejected jobs the gate scored highest\n")
    out += [line(r) for r in sorted(neg, key=lambda r: -r["gate_score"])[:15]]
    out.append("\n## Skipped jobs the gate would have surfaced (DeepSeek said < 50)\n")
    out += [line(r) for r in sorted(skipped, key=lambda r: -r["gate_score"])[:15]]
    if errors:
        out.append("\n## Errors\n")
        out += [f"- {r['id']} {r['title']}: {e}" for r, e in errors[:20]]
    out.append("\nNext: `python cli.py gate-fit` (or Profiles → Gate) to learn weights "
               "and thresholds from these answers; it's free, everything is cached.")
    return "\n".join(out)
