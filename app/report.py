"""Tuning report: Markdown dump of eval outcomes + human feedback.

Take this plus docs/interview_spec.md to a chat session to produce the next
profile version. Shared by `cli.py tuning-report` and the Profiles page.
"""
import json

from . import db

REQUIRED_PROFILE_KEYS = {"applicant_id", "name", "profile", "criteria"}


def tuning_report(applicant_id: str) -> str:
    reviewed = ("('approved','rejected','draft_created','shortlisted',"
                "'applied','archived')")
    # A job archived because its posting closed carries no match-quality signal
    # — it must not read as an ARCHIVED-AFTER-INTEREST miss. Only manual-close
    # ever sets closed_at on an archived row (the sweep skips archived jobs), so
    # this uniquely excludes closure-archives while keeping real withdrawals.
    not_closed = "NOT (status='archived' AND closed_at IS NOT NULL)"
    with db.connect() as conn:
        profile = db.latest_profile(conn, applicant_id)
        # Restrict to iterations on the CURRENT profile version: only jobs
        # reviewed since it was uploaded reflect decisions the live profile
        # is responsible for. (No per-job profile-version stamp exists, so
        # the upload timestamp is the cutoff.) Older decisions were tuning
        # signal for a profile that no longer exists — including them bloats
        # the report and muddies the disagreements worth acting on.
        vrow = conn.execute(
            "SELECT version, created_at FROM profiles WHERE applicant_id=? "
            "ORDER BY version DESC LIMIT 1", (applicant_id,)).fetchone()
        version = vrow["version"] if vrow else None
        since = vrow["created_at"] if vrow else None

        jobs = conn.execute(
            "SELECT * FROM jobs WHERE applicant_id=? AND status IN "
            f"{reviewed} AND {not_closed} AND reviewed_at >= ? "
            "ORDER BY reviewed_at",
            (applicant_id, since or "")).fetchall()
        # decisions that predate the current version, shown only as a count
        prior = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE applicant_id=? AND status IN "
            f"{reviewed} AND {not_closed} AND reviewed_at < ?",
            (applicant_id, since or "")).fetchone()["n"]
        dist = conn.execute(
            "SELECT status, COUNT(*) n, AVG(score) avg_score FROM jobs "
            f"WHERE applicant_id=? AND {not_closed} AND reviewed_at >= ? GROUP BY status",
            (applicant_id, since or "")).fetchall()

    out = [f"# Profile tuning report: {applicant_id}\n"]
    if version is not None:
        out.append(f"Scope: profile **v{version}** (uploaded {since}) and "
                   f"the {len(jobs)} decisions made since. "
                   f"{prior} earlier decision(s) on prior versions are excluded.\n")
    out.append("## Current profile\n```json")
    out.append(json.dumps(profile, indent=1))
    out.append("```\n\n## Score distribution by outcome (this version)\n")
    out.append("| status | count | avg score |\n|---|---|---|")
    for r in dist:
        avg = f"{r['avg_score']:.0f}" if r["avg_score"] is not None else "-"
        out.append(f"| {r['status']} | {r['n']} | {avg} |")
    out.append("\n## Reviewed jobs since this version (model score vs. your decision)\n")
    for j in jobs:
        decision = ("APPLIED" if j["status"] == "applied"
                    else "APPROVED" if j["status"] in ("approved", "draft_created")
                    else "SHORTLISTED" if j["status"] == "shortlisted"
                    # archived = shortlisted/drafted first, then withdrawn — a
                    # late change of mind, stronger miss signal than a reject
                    else "ARCHIVED-AFTER-INTEREST" if j["status"] == "archived"
                    else "REJECTED")
        out.append(f"### [{decision}] {j['title']} — {j['company']} (scored {j['score']})")
        out.append(f"- pitch: {j['pitch']}")
        out.append(f"- concerns: {j['concerns']}")
        if j["feedback"]:
            out.append(f"- **your feedback:** {j['feedback']}")
        out.append("")
    out.append("## Instructions for the refinement session\n")
    out.append("Paste this whole report into a chat session along with "
               "docs/interview_spec.md. Ask for a revised profile JSON that better "
               "predicts the APPROVED/REJECTED decisions above — especially where "
               "the model's score disagreed with the human decision.")
    return "\n".join(out)
