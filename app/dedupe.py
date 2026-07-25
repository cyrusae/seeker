"""Fuzzy cross-source duplicate detection, run at ingest.

Per-source dedupe keys can't catch the same posting arriving via Adzuna AND
the employer's own board, so each new job is fuzzy-compared against the
applicant's existing rows:

  EXACT   — same listing (title + description effectively identical).
            Ingest parks it as status='filtered' ("duplicate of ..."), so it
            keeps a dedupe key (scrapers skip re-fetching it) but never costs
            an eval. If the older copy came from an aggregator and the new one
            is direct-from-board, the older row is upgraded in place first.
  SIMILAR — same employer + near-same title, or highly similar text, with
            meaningful differences (e.g. two openings, different locations).
            Proceeds through eval normally, flagged via jobs.similar_to.

Comparison is local string similarity (rapidfuzz) — no LLM cost. Scoped per
applicant on purpose: both applicants ingesting one posting is intended.
"""
import re

from rapidfuzz import fuzz

# Tunable thresholds (0-100 similarity).
EXACT_TITLE = 95
EXACT_DESC = 90
SIMILAR_TITLE = 85
SIMILAR_COMPANY = 80
SIMILAR_DESC = 88

DESC_CMP_CHARS = 1500  # Adzuna truncates; compare prefixes of equal length
_norm_re = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str | None) -> str:
    return _norm_re.sub(" ", (s or "").lower()).strip()


def _desc_sim(a: str | None, b: str | None) -> float:
    a, b = _norm(a)[:DESC_CMP_CHARS], _norm(b)[:DESC_CMP_CHARS]
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return fuzz.token_set_ratio(a[:n], b[:n])


def load_corpus(conn, applicant_id: str) -> list[dict]:
    """Existing rows to compare incoming jobs against. Excludes rows that are
    themselves parked duplicates so chains always point at the original."""
    return [dict(r) for r in conn.execute(
        """SELECT id, title, company, location, description, url, source
           FROM jobs WHERE applicant_id=? AND similar_to IS NULL
           AND NOT (status='filtered' AND error LIKE 'duplicate of%')""",
        (applicant_id,))]


def find_match(job: dict, corpus: list[dict]) -> tuple[str, dict] | None:
    """Returns ("exact"|"similar", existing_row) for the strongest match."""
    title = _norm(job.get("title"))
    company = _norm(job.get("company"))
    if not title:
        return None
    best: tuple[str, float, dict] | None = None
    for row in corpus:
        t_sim = fuzz.token_set_ratio(title, _norm(row["title"]))
        if t_sim < 70:
            continue  # cheap gate before touching descriptions
        c_sim = fuzz.token_set_ratio(company, _norm(row["company"])) if company else 0
        d_sim = _desc_sim(job.get("description"), row["description"])
        # Same company + effectively the same title is one requisition, even
        # when the description drifts below EXACT_DESC — which is exactly what
        # a role posted across dozens of locations does (identical title, but
        # per-city boilerplate knocks desc similarity down). Collapse it as
        # exact so we don't eval the same job 30 times; ingest unions the
        # locations onto the survivor.
        same_req = t_sim >= EXACT_TITLE and c_sim >= SIMILAR_COMPANY
        if (t_sim >= EXACT_TITLE and d_sim >= EXACT_DESC) or same_req:
            kind, score = "exact", t_sim + d_sim
        elif (t_sim >= SIMILAR_TITLE and c_sim >= SIMILAR_COMPANY) or d_sim >= SIMILAR_DESC:
            kind, score = "similar", (t_sim + d_sim) / 2
        else:
            continue
        # exact beats similar; then higher score wins
        rank = (kind == "exact", score)
        if best is None or rank > (best[0] == "exact", best[1]):
            best = (kind, score, row)
    return (best[0], best[2]) if best else None


def prefer_direct(conn, existing: dict, job: dict):
    """When the stored copy is an aggregator redirect and the duplicate came
    straight from the employer's board, upgrade the stored row in place:
    direct apply URL, untruncated description, and salary if we lacked one."""
    old_url, new_url = existing.get("url") or "", job.get("url") or ""
    if "adzuna" in old_url and new_url and "adzuna" not in new_url:
        conn.execute(
            """UPDATE jobs SET url=?, description=?,
               salary=COALESCE(salary, ?) WHERE id=?""",
            (new_url, job.get("description"), job.get("salary"), existing["id"]))


_MORE_RE = re.compile(r"\s*\(\+(\d+) more\)$")


def merge_location(conn, existing_id: str, new_loc: str | None):
    """Location-clone collapse: keep a compact union of locations on the
    surviving row (' / '-joined, capped at 6 with a '(+N more)' tally) so the
    reviewer still sees everywhere a role is posted. Reads the survivor's
    current value from the DB each call so unions accumulate within a run."""
    new_loc = (new_loc or "").strip()
    if not new_loc:
        return
    row = conn.execute("SELECT location FROM jobs WHERE id=?", (existing_id,)).fetchone()
    cur = ((row["location"] if row else "") or "").strip()
    m = _MORE_RE.search(cur)
    overflow = int(m.group(1)) if m else 0
    if m:
        cur = cur[:m.start()]
    parts = [p.strip() for p in cur.split(" / ") if p.strip()]
    if new_loc in parts:
        return
    if len(parts) < 6:
        parts.append(new_loc)
    else:
        overflow += 1
    shown = " / ".join(parts) + (f" (+{overflow} more)" if overflow else "")
    conn.execute("UPDATE jobs SET location=? WHERE id=?", (shown, existing_id))
