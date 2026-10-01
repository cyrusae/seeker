"""Tuning report: Markdown evidence for a profile-refinement session.

Take this plus docs/interview_spec.md to a chat session to produce the next
profile version. Shared by `cli.py tuning-report` and the Profiles page.

Built to be read by a model in one sitting, so it leads with aggregates and
lists only the decisions that teach something: where your call disagreed with
the scorers, and every reject you left a note on. Agreements are counted, not
listed (listing all of them made reports run to 40k+ tokens of noise).
"""
import json
from collections import defaultdict

from . import db

REQUIRED_PROFILE_KEYS = {"applicant_id", "name", "profile", "criteria"}
POSITIVE = ("shortlisted", "applied", "archived")
MAX_LISTED = 30  # per disagreement section


def _decision(status: str) -> str:
    return {"applied": "APPLIED", "shortlisted": "SHORTLISTED",
            # archived = shortlisted first, then withdrawn — a late change of
            # mind, a stronger miss signal than a reject
            "archived": "ARCHIVED-AFTER-INTEREST"}.get(status, "REJECTED")


def _gate_evidence(applicant_id: str, profile: dict, jobs) -> tuple[dict, list[str]]:
    """Per-job gate view (score, reasons) for the decided jobs, plus a Markdown
    table of how often each hard filter / anti-criterion / preference fires on
    kept vs rejected jobs. Uses cached answers under the CURRENT profile only;
    never calls the model. Returns ({job_id: view}, lines)."""
    try:
        from . import gate
        from .gatefit import closure_note
    except Exception:  # noqa: BLE001 — the report must work without the gate
        return {}, []
    views, stat = {}, defaultdict(lambda: {"kept": 0, "kn": 0, "rej": 0, "rn": 0})
    meta_seen = {}
    model = gate.active_model()
    for j in jobs:
        if j["status"] == "rejected" and closure_note(j["feedback"]):
            continue
        ev = gate.lookup(j, profile)
        if not ev:
            continue
        c = gate.combine(ev["answers"], ev["meta"], model, applicant_id)
        pros, cons = gate.top_reasons(ev["answers"], ev["meta"])
        views[j["id"]] = {"score": c["gate_score"], "for": pros, "against": cons,
                          "vetoes": c["vetoes"]}
        kept = j["status"] in POSITIVE
        for q, m in ev["meta"].items():
            if m["kind"] not in ("hard", "anti", "soft"):
                continue
            meta_seen[q] = m
            hit = float(ev["answers"][q]["noul"]) >= (0.5 if m["kind"] == "soft" else gate.VETO_P)
            s = stat[q]
            if kept:
                s["kept"] += hit; s["kn"] += 1
            else:
                s["rej"] += hit; s["rn"] += 1
    if not views:
        return {}, ["_No gate answers cached for this profile yet — run "
                    "`cli.py gate-backtest` to include per-item evidence._\n"]
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
    lines = [f"Jev answered each criterion as a yes/no question about each posting "
             f"({len(views)} decided jobs). A useful dealbreaker fires on rejects and "
             "almost never on kept jobs; one that fires on kept jobs is worded too "
             "broadly or isn't really a dealbreaker. A useful preference fires more "
             "on kept jobs than rejected ones.\n",
             "| kind | severity/weight | fires on kept | fires on rejected | item |",
             "|---|---|---|---|---|"]
    order = {"hard": 0, "anti": 1, "soft": 2}
    for q, s in sorted(stat.items(), key=lambda kv: (order[meta_seen[kv[0]]["kind"]],
                                                     -kv[1]["kept"] / max(kv[1]["kn"], 1))):
        m = meta_seen[q]
        tag = m.get("severity") or "dealbreaker" if m["kind"] != "soft" else \
            {3.0: "high", 2.0: "medium", 1.0: "low"}.get(m["weight"], "?")
        lines.append(f"| {m['kind']} | {tag} | {pct(s['kept'], s['kn'])} | "
                     f"{pct(s['rej'], s['rn'])} | {m['label']} |")
    lines.append("")
    return views, lines


def tuning_report(applicant_id: str) -> str:
    reviewed = "('rejected','shortlisted','applied','archived')"
    # A job archived because its posting closed carries no match-quality signal
    # — it must not read as an ARCHIVED-AFTER-INTEREST miss. Only manual-close
    # ever sets closed_at on an archived row (the sweep skips archived jobs), so
    # this uniquely excludes closure-archives while keeping real withdrawals.
    not_closed = "NOT (status='archived' AND closed_at IS NOT NULL)"
    with db.connect() as conn:
        profile = db.latest_profile(conn, applicant_id)
        vrow = conn.execute(
            "SELECT version, created_at FROM profiles WHERE applicant_id=? "
            "ORDER BY version DESC LIMIT 1", (applicant_id,)).fetchone()
        version = vrow["version"] if vrow else None
        since = vrow["created_at"] if vrow else None
        # Decisions on the CURRENT profile version are the ones it's
        # responsible for; older ones were signal for a profile that no
        # longer exists.
        jobs = conn.execute(
            "SELECT * FROM jobs WHERE applicant_id=? AND status IN "
            f"{reviewed} AND {not_closed} AND reviewed_at >= ? ORDER BY reviewed_at",
            (applicant_id, since or "")).fetchall()
        prior = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE applicant_id=? AND status IN "
            f"{reviewed} AND {not_closed} AND reviewed_at < ?",
            (applicant_id, since or "")).fetchone()["n"]
        dist = conn.execute(
            "SELECT status, COUNT(*) n, AVG(score) avg_score FROM jobs "
            f"WHERE applicant_id=? AND {not_closed} AND reviewed_at >= ? GROUP BY status",
            (applicant_id, since or "")).fetchall()
        stale_n = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE applicant_id=? AND status='stale' "
            "AND reviewed_at >= ?", (applicant_id, since or "")).fetchone()["n"]

    from .config import settings
    views, gate_lines = _gate_evidence(applicant_id, profile or {}, jobs)

    out = [f"# Profile tuning report: {applicant_id}\n"]
    if version is not None:
        out.append(f"Scope: profile **v{version}** (uploaded {since}) and the "
                   f"{len(jobs)} decisions made since. {prior} earlier decision(s) on prior "
                   f"versions are excluded. {stale_n} job(s) dismissed as stale/closed are "
                   "excluded too — a dead posting says nothing about fit.\n")
    out.append("## Current profile\n```json")
    out.append(json.dumps(profile, indent=1))
    out.append("```\n\n## Outcomes (this version)\n")
    out.append("| status | count | avg DeepSeek score |\n|---|---|---|")
    for r in dist:
        avg = f"{r['avg_score']:.0f}" if r["avg_score"] is not None else "-"
        out.append(f"| {r['status']} | {r['n']} | {avg} |")
    out.append("\n## How each criterion behaves in practice\n")
    out += gate_lines

    def high(j):
        """Did the scorers think this was a strong match? (either one)"""
        v = views.get(j["id"])
        ds = j["score"] is not None and j["score"] >= settings.escalate_min_score
        return ds or (v is not None and v["score"] >= 50)

    def low(j):
        v = views.get(j["id"])
        ds_low = j["score"] is None or j["score"] < settings.review_min_score + 10
        return ds_low and (v is None or v["score"] < 30)

    def entry(j):
        v = views.get(j["id"])
        lines = [f"### [{_decision(j['status'])}] {j['title']} — {j['company']}",
                 f"- scores: DeepSeek {j['score'] if j['score'] is not None else '–'}, "
                 f"Jev gate {v['score']:.0f}" if v else
                 f"- scores: DeepSeek {j['score'] if j['score'] is not None else '–'}"]
        if j["pitch"]:
            lines.append(f"- DeepSeek pitch: {j['pitch']}")
        if j["concerns"]:
            lines.append(f"- DeepSeek concerns: {j['concerns']}")
        if v:
            if v["vetoes"]:
                lines.append(f"- **vetoed by dealbreaker:** {'; '.join(v['vetoes'])}")
            lines.append(f"- Jev for: {'; '.join(v['for']) or '—'}")
            lines.append(f"- Jev against: {'; '.join(v['against']) or '—'}")
        if j["feedback"]:
            lines.append(f"- **your feedback:** {j['feedback']}")
        if j["notes"]:
            lines.append(f"- your note: {j['notes']}")
        return "\n".join(lines) + "\n"

    kept = [j for j in jobs if j["status"] in POSITIVE]
    rejected = [j for j in jobs if j["status"] == "rejected"]
    kept_low = [j for j in kept if low(j) or (views.get(j["id"]) or {}).get("vetoes")]
    rej_high = [j for j in rejected if high(j)]
    rej_high.sort(key=lambda j: -max(j["score"] or 0, (views.get(j["id"]) or {}).get("score", 0)))
    listed = {j["id"] for j in kept_low} | {j["id"] for j in rej_high[:MAX_LISTED]}
    noted = [j for j in rejected if j["feedback"] and j["id"] not in listed]

    out.append("## Kept jobs the scorers undervalued (most important)\n")
    out.append("You kept these, but a scorer rated them low or a dealbreaker vetoed them. "
               "Each is a sign the profile is missing something you value or "
               "rules out something you'd accept.\n")
    out += [entry(j) for j in kept_low[:MAX_LISTED]] or ["_None._\n"]
    out.append("## Rejected jobs the scorers rated highly\n")
    out.append("The profile made these look good; you passed. Your feedback says why; "
               "look for patterns the profile doesn't express yet.\n")
    out += [entry(j) for j in rej_high[:MAX_LISTED]] or ["_None._\n"]
    if len(rej_high) > MAX_LISTED:
        out.append(f"_…and {len(rej_high) - MAX_LISTED} more like these, not listed._\n")
    out.append("## Other rejects with your feedback\n")
    out += [f"- {j['title']} — {j['company']} (DeepSeek {j['score'] if j['score'] is not None else '–'}): "
            f"{j['feedback']}" for j in noted] or ["_None._"]
    rest = len(jobs) - len(listed) - len(noted)
    out.append(f"\n{rest} other decision(s) agreed with the scorers and are not listed.\n")

    out.append("## Instructions for the refinement session\n")
    out.append("Paste this whole report into a chat session along with "
               "docs/interview_spec.md and follow its **Full review** or **Tune** "
               "mode. Ground every change in evidence above: criteria that fire on "
               "kept jobs, preferences that don't separate kept from rejected, and "
               "patterns in the undervalued and rejected-anyway sections.")
    return "\n".join(out)
