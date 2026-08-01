"""Group listings by company and compare two of them.

Two features that go together: the Companies view buckets an applicant's jobs
by (normalized) employer so near-identical postings sit next to each other, and
`compare_pair` explains — in a few lines — what actually differs between any two
of them. The field-level comparison (title/location/salary/score) is rendered
free/instantly client-side; this module owns the two things that need Python:
the rapidfuzz similarity number and the cheap-tier LLM semantic summary, the
latter cached per pair so re-opening it costs nothing.
"""
import hashlib

from rapidfuzz import fuzz

from . import db, llm
from .dedupe import _norm, _desc_sim

# Status buckets for the Companies view filter. "review" = jobs ready for a
# human decision; "shortlist" = jobs you've confirmed interest in; "applied"
# (already submitted) and "archived" (withdrawn) are opt-in layers. Pre-review
# noise (pending_eval, filtered dupes, auto-skipped, plain rejects) never
# appears here.
_REVIEW = ("pending_user_review",)
_SHORTLIST = ("shortlisted",)
_APPLIED = ("applied",)
_ARCHIVED = ("archived",)


def statuses_for(view: str, include_archived: bool,
                 include_applied: bool = False) -> tuple[str, ...]:
    """Resolve the Companies filter (view + applied/archived toggles) to a
    status tuple."""
    result = _SHORTLIST if view == "shortlist" else (_REVIEW + _SHORTLIST)
    if include_applied:
        result += _APPLIED
    if include_archived:
        result += _ARCHIVED
    return result

_SUMMARY_SYSTEM = (
    "You compare two job postings, usually from the same employer, for a "
    "job-seeker who finds it hard to hold the difference in their head. "
    "Be concise and concrete. In 2-4 short lines, cover only what MATTERS: "
    "is it the same core role? what differs (seniority, required skills, "
    "team/scope, location/remote, comp)? Ignore boilerplate (EEO statements, "
    "benefits blurbs, company mission). End with a one-line verdict: are these "
    "two genuine openings, the same opening reworded, or a repost? "
    "Plain text, no headings, no preamble."
)


def _sig(a: dict, b: dict) -> str:
    """Signature over both descriptions so an edited/re-fetched posting
    invalidates a stale cached summary."""
    material = (a.get("description") or "") + "\x00" + (b.get("description") or "")
    return hashlib.sha1(material.encode()).hexdigest()


def company_groups(conn, applicant_id: str, statuses=None) -> list[dict]:
    """Buckets an applicant's listed jobs by normalized company name. Each group
    carries its rows (newest first) and a set of similar_to pairs so the view
    can flag the near-duplicates seeker already detected at ingest. `statuses`
    filters which rows appear (defaults to review + shortlist)."""
    statuses = tuple(statuses) if statuses is not None else (_REVIEW + _SHORTLIST)
    marks = ",".join("?" * len(statuses))
    q = (f"SELECT * FROM jobs WHERE status IN ({marks})"
         + (" AND applicant_id=?" if applicant_id else ""))
    args = list(statuses) + ([applicant_id] if applicant_id else [])
    rows = [dict(r) for r in conn.execute(q, args)]

    buckets: dict[str, dict] = {}
    for r in rows:
        key = _norm(r.get("company")) or "\x00unknown"
        g = buckets.setdefault(key, {"company": r.get("company") or "(unknown)", "jobs": []})
        g["jobs"].append(r)

    groups = list(buckets.values())
    for g in groups:
        g["jobs"].sort(key=lambda j: j.get("created_at") or "", reverse=True)
        ids = {j["id"] for j in g["jobs"]}
        # ids that seeker linked as near-duplicates to another row in this bucket
        g["similar_ids"] = {
            j["id"] for j in g["jobs"] if j.get("similar_to") in ids
        } | {
            j["similar_to"] for j in g["jobs"] if j.get("similar_to") in ids
        }
    # Multi-listing companies first (that's where comparison matters), then by size.
    groups.sort(key=lambda g: (len(g["jobs"]) == 1, -len(g["jobs"]), g["company"].lower()))
    return groups


def _load(conn, job_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(r) if r else None


def compare_pair(conn, id_a: str, id_b: str, refresh: bool = False) -> dict:
    """Returns {similarity, summary, cached, error?} for two jobs. The similarity
    is free (rapidfuzz); the summary is a cached cheap-tier LLM call, recomputed
    only when missing, stale, or explicitly refreshed."""
    a, b = _load(conn, id_a), _load(conn, id_b)
    if not a or not b:
        return {"error": "one of the listings no longer exists"}

    title_sim = fuzz.token_set_ratio(_norm(a.get("title")), _norm(b.get("title")))
    desc_sim = _desc_sim(a.get("description"), b.get("description"))
    similarity = round((title_sim + desc_sim) / 2)

    lo, hi = sorted((id_a, id_b))
    sig = _sig(*((a, b) if id_a == lo else (b, a)))
    if not refresh:
        row = conn.execute(
            "SELECT summary, sig FROM comparisons WHERE job_a=? AND job_b=?", (lo, hi)
        ).fetchone()
        if row and row["sig"] == sig:
            return {"similarity": similarity, "summary": row["summary"], "cached": True}

    user_msg = (
        f"POSTING A — {a.get('title')} ({a.get('company') or '?'}, "
        f"{a.get('location') or '?'})\n{(a.get('description') or '').strip()}\n\n"
        f"POSTING B — {b.get('title')} ({b.get('company') or '?'}, "
        f"{b.get('location') or '?'})\n{(b.get('description') or '').strip()}"
    )
    try:
        summary = llm.complete("score", _SUMMARY_SYSTEM, user_msg, max_tokens=500).strip()
    except llm.LLMError as e:
        return {"similarity": similarity, "error": str(e)}

    conn.execute(
        "INSERT INTO comparisons (id, job_a, job_b, sig, summary, created_at) "
        "VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(job_a, job_b) DO UPDATE SET sig=excluded.sig, "
        "summary=excluded.summary, created_at=excluded.created_at",
        (db.new_id(), lo, hi, sig, summary, db.now()),
    )
    return {"similarity": similarity, "summary": summary, "cached": False}
