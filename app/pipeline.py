"""The batch pipeline: ingest → score (local) → escalate (paid, selective)
→ human review → write (paid) → render.

Every LLM call here is single-shot and bounded: one job + one profile in,
one JSON or document out. No loops, no accumulating context.
"""
import json
import re
import threading
import time

import httpx

from . import db, llm, render, textclean

SCORE_SYSTEM = """You are a job-match evaluator. You receive an applicant profile
(JSON) and one job posting. Score the match honestly — a mediocre match should
score mediocre. Apply the profile's hard filters strictly (a hard-filter failure
caps the score at 20) and its anti-criteria (e.g. excluded title patterns).

Also assess scam risk. Red flags: pay-to-start, fees, crypto/wire transfers,
"equipment check" schemes, vague or unverifiable company, wildly above-market
pay for low-skill remote work (data entry is a high-scam category).

If the posting text is too thin to assess actual duties, schedule, location,
or seniority (e.g. truncated to a preamble), cap the score at 45 and name the
missing information in concerns. Never rate a job highly on title alone.

Output JSON (respect the word limits — output will be cut off if too long):
{
  "score": 0-100,
  "pitch": "why this is/isn't worth this applicant's time — max 50 words",
  "concerns": "gaps, risks, unknowns — max 30 words",
  "scam_risk": "none" | "low" | "medium" | "high",
  "criteria": {"<criterion name>": "assessment, max 12 words", ...}
}"""

WRITE_SYSTEM = """You are a professional resume and cover-letter writer.
You receive an applicant profile (JSON), a job posting, and an evaluator's pitch.

Rules:
- NEVER invent credentials, employers, dates, or skills not in the profile.
- If the profile has "base_resume_md", treat it as the authoritative source
  document: tailor it (reorder, reword, trim, re-emphasize for this job) rather
  than writing from scratch. Structured fields supplement it.
- If the profile has a "contact" block, put it in the resume header; otherwise
  leave a clearly-marked placeholder line like [CONTACT INFO].
- Reframe honestly: emphasize the profile's real experience in the language of
  this job's domain. Use any "framing_notes" in the profile.
- A REVIEWER NOTE, if present, was written by the applicant when approving this
  job. Treat it as authoritative supplemental profile information — it may add
  real experience or context missing from the profile, and it overrides the
  evaluator's pitch/concerns where they conflict.
- Resume: reverse-chronological, tight bullet points, one page of content.
- Cover letter: specific to this company and role, references why the match is
  genuine (use the pitch), no generic filler, ~250 words.

Output exactly two Markdown documents separated by marker lines, and nothing
else — no JSON, no commentary before, between, or after:

===RESUME===
(the complete resume in Markdown)
===COVER LETTER===
(the complete cover letter in Markdown)"""


def _kw_matches(kw: str, t: str) -> bool:
    """True if `kw` appears in lowercased title `t` as a whole token, not a
    substring — so 'V' (meaning 'very senior') hits "Senior V Eng" but not
    "Every Developer". Boundaries are anchored only on a keyword's alphanumeric
    edges, so symbol-bearing tokens like 'C++' or '.NET' still match."""
    k = kw.lower()
    if not k:
        return False
    left = r"(?<![a-z0-9])" if k[0].isalnum() else ""
    right = r"(?![a-z0-9])" if k[-1].isalnum() else ""
    return re.search(left + re.escape(k) + right, t) is not None


def _company_matches(company: str | None, scopes, groups: dict) -> bool:
    """True if `company` token-matches any name in `scopes`. Each scope is
    either a company_groups alias (expanded to its list) or a literal company
    name. Source company strings vary wildly ('anthropic', 'Microsoft
    Corporation', 'Amazon Web Services, Inc.'), so a scope name matches as a
    whole token inside the lowercased company string — 'amazon' hits every
    Amazon entity, but 'meta' won't fire on 'Metadata Corp'."""
    c = (company or "").lower()
    if not c:
        return False
    names: list[str] = []
    for s in scopes:
        names.extend(groups.get(s, [s]))
    return any(_kw_matches(n, c) for n in names)


def _rule_hit(title_l: str, rule: dict, company: str | None, groups: dict) -> str | None:
    """A title_exclude_rule fires when every token in `all` is present AND at
    least one token in `any` is present (each clause optional, order-independent,
    token-boundary matched) — and, if `at` is set, the company is in scope.
    Returns a short reason label, or None."""
    at = rule.get("at")
    if at is not None:
        scopes = [at] if isinstance(at, str) else list(at)
        if not _company_matches(company, scopes, groups):
            return None
    alls = rule.get("all") or []
    anys = rule.get("any") or []
    if not alls and not anys:
        return None
    if alls and not all(_kw_matches(k, title_l) for k in alls):
        return None
    if anys and not any(_kw_matches(k, title_l) for k in anys):
        return None
    label = "+".join(alls) if alls else "/".join(anys)
    return f"{label} @ {company}" if at is not None else label


def _excluded_kw(title: str | None, profile: dict, company: str | None = None) -> str | None:
    """Title gate, run before any LLM call. Returns a human-readable reason the
    job was excluded, or None. Two config shapes under profile['prefilter']:
      title_exclude:       flat list of global single token/phrase excludes.
      title_exclude_rules: [{all:[...] | any:[...], at: alias | [companies]}]
                           — multi-token patterns, optionally company-scoped;
                           `company_groups` maps an alias to a list of names."""
    pf = profile.get("prefilter") or {}
    t = (title or "").lower()
    for kw in pf.get("title_exclude") or []:
        if _kw_matches(kw, t):
            return f"title matched exclude keyword '{kw}'"
    groups = pf.get("company_groups") or {}
    for rule in pf.get("title_exclude_rules") or []:
        hit = _rule_hit(t, rule, company, groups)
        if hit:
            return f"title matched exclude rule: {hit}"
    return None


# A job must fail its closure check this many consecutive times before it's
# flagged — one miss is often a transient board hiccup or CDN blip.
CLOSE_AFTER_MISSES = 2
# Statuses worth tracking for closure: the ones the applicant is acting on.
# Not 'applied' (already submitted — closure is just history) or dead states.
_ACTIVE_STATUSES = ("pending_user_review", "shortlisted", "approved", "draft_created")


def _mark(conn, row, closed: bool, reason: str) -> bool:
    """Apply one closure observation with 2-strike grace. `row` needs id,
    missed_count, closed_at. A "still open" observation clears any prior flag;
    a "closed" one increments the miss counter and flags once it crosses the
    grace threshold. Returns True iff this call newly flagged the row."""
    if not closed:
        if row["missed_count"] or row["closed_at"]:  # recovered — un-flag
            conn.execute("UPDATE jobs SET missed_count=0, closed_at=NULL, "
                         "closed_reason=NULL WHERE id=?", (row["id"],))
        return False
    n = (row["missed_count"] or 0) + 1
    if n >= CLOSE_AFTER_MISSES and not row["closed_at"]:
        conn.execute("UPDATE jobs SET missed_count=?, closed_at=?, "
                     "closed_reason=? WHERE id=?", (n, db.now(), reason, row["id"]))
        return True
    conn.execute("UPDATE jobs SET missed_count=? WHERE id=?", (n, row["id"]))
    return False


def _sweep_closed(conn, applicant_id: str, scan) -> int:
    """Board-diff closure: flag active jobs whose full-enumerator board no
    longer lists them. Only judges jobs from enumerators that actually ran this
    cycle; reappearance clears the flag. Returns the count newly marked closed."""
    if not scan.swept_sources:
        return 0
    marks = ",".join("?" * len(scan.swept_sources))
    rows = conn.execute(
        f"SELECT id, dedupe_key, missed_count, closed_at FROM jobs "
        f"WHERE applicant_id=? AND poll_source IN ({marks}) "
        f"AND status IN {_ACTIVE_STATUSES} AND closure_dismissed=0",
        (applicant_id, *scan.swept_sources)).fetchall()
    return sum(_mark(conn, r, r["dedupe_key"] not in scan.enumerated_keys, "absent")
               for r in rows)


def ingest_all() -> tuple[int, int]:
    """Poll all configured sources for all applicants. Returns
    (new-job count, newly-closed count). Fuzzy-detected duplicates are stored
    but not counted as new.

    Deliberately opens a short-lived connection per phase rather than one
    connection for the whole function: fetch_for_profile does many sequential
    network calls (adzuna alone can be dozens of paced requests per profile),
    and sqlite3 holds an implicit write transaction open from the first INSERT
    until the connection's `with` block exits. Wrapping all of that network
    I/O in one connection held the write lock for the entire fetch, so any
    concurrent writer (a web UI click, the liveness probe) blocked past the
    10s busy_timeout and died with "database is locked"."""
    from . import dedupe
    from .sources import fetch_for_profile
    added = closed = 0
    with db.connect() as conn:
        applicants = db.all_applicants(conn)
    for profile in applicants:
        if _STOP.is_set():
            break
        applicant_id = profile["applicant_id"]
        with db.connect() as conn:
            seen = frozenset(r["dedupe_key"] for r in conn.execute(
                "SELECT dedupe_key FROM jobs WHERE applicant_id=?", (applicant_id,)))
            corpus = dedupe.load_corpus(conn, applicant_id)
        scan = fetch_for_profile(profile, seen)  # network I/O, no db lock held
        with db.connect() as conn:
            for job in scan:
                if _STOP.is_set():
                    break
                # safety net for adapters that don't clean (plain-text APIs
                # still carry entities); idempotent for the ones that do
                job = {**job, "description": textclean.clean(job.get("description"))}
                kw = _excluded_kw(job.get("title"), profile, job.get("company"))
                if kw:
                    # Keep the row (dedupe key stops nightly re-adds; visible
                    # and re-evaluable on the Jobs page) but never eval it.
                    job = {**job, "status": "filtered", "error": kw}
                else:
                    match = dedupe.find_match(job, corpus)
                    if match:
                        kind, existing = match
                        if kind == "exact":
                            # Same listing from another source: park it (the
                            # kept dedupe key stops nightly re-fetches) and
                            # upgrade the original if this copy is more direct.
                            dedupe.prefer_direct(conn, existing, job)
                            dedupe.merge_location(conn, existing["id"], job.get("location"))
                            job = {**job, "status": "filtered",
                                   "error": f"duplicate of {existing['id']} "
                                            f"({existing['title']})"[:300]}
                        else:
                            job = {**job, "similar_to": existing["id"]}
                jid = db.insert_job(conn, applicant_id=applicant_id,
                                    source="poll", **job)
                if jid:
                    if job.get("status") != "filtered":
                        added += 1
                        # New legit jobs join the corpus so intra-run dupes
                        # (same posting from two sources in one cycle) match.
                        if not job.get("similar_to"):
                            corpus.append({"id": jid, "title": job.get("title"),
                                           "company": job.get("company"),
                                           "location": job.get("location"),
                                           "description": job.get("description"),
                                           "url": job.get("url"), "source": "poll"})
            closed += _sweep_closed(conn, applicant_id, scan)
    return added, closed


ENRICH_MIN_CHARS = 1200  # descriptions shorter than this get a full-text fetch


def _maybe_enrich(job) -> str:
    """Aggregator descriptions (Adzuna) are truncated by their API. If the
    stored text is thin and we have a URL, fetch the real posting and extract
    the full description. Persist it so review shows the full text too."""
    desc = job["description"] or ""
    needs = "[Note: Adzuna truncates" in desc or len(desc) < ENRICH_MIN_CHARS
    if not (needs and job["url"]):
        return desc
    from .sources import manual
    try:
        enriched = manual.extract(manual.fetch_url(job["url"]), job["url"])
        new_desc = textclean.clean(enriched.get("description"))
        if len(new_desc) > len(desc):
            with db.connect() as conn:
                conn.execute(
                    "UPDATE jobs SET description=?, salary=COALESCE(salary, ?), "
                    "location=COALESCE(location, ?) WHERE id=?",
                    (new_desc, enriched.get("salary"), enriched.get("location"), job["id"]))
            return new_desc
    except Exception as e:
        # Not fatal — tell the scorer the text is incomplete so the thin-posting
        # rule kicks in, and persist the note for the review card.
        note = f"\n\n[Full posting text could not be fetched ({str(e)[:120]}); this may be a truncated preview.]"
        if note not in desc:
            desc += note
            with db.connect() as conn:
                conn.execute("UPDATE jobs SET description=? WHERE id=?", (desc, job["id"]))
    return desc


def evaluate_job(job_id: str):
    with db.connect() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or job["status"] != "pending_eval":
            return
        profile = db.latest_profile(conn, job["applicant_id"])
    if profile is None:
        return

    # Same keyword gate as ingest — catches jobs queued before the profile's
    # prefilter existed, at zero token cost.
    kw = _excluded_kw(job["title"], profile, job["company"])
    if kw:
        with db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='filtered', error=? WHERE id=? AND status='pending_eval'",
                (kw, job_id))
        return

    description = _maybe_enrich(job)
    user_msg = (
        f"APPLICANT PROFILE:\n{json.dumps(profile, indent=1)}\n\n"
        f"JOB POSTING:\nTitle: {job['title']}\nCompany: {job['company']}\n"
        f"Location: {job['location']}\nSalary: {job['salary']}\n\n{description}"
    )
    try:
        ev = llm.complete_json("score", SCORE_SYSTEM, user_msg, max_tokens=2500)
        escalated = 0
        from .config import settings
        esc, sc = settings.role("escalate"), settings.role("score")
        # Only escalate if it's actually a different judge (provider OR model).
        if int(ev.get("score", 0)) >= settings.escalate_min_score and \
                (esc.provider, esc.model) != (sc.provider, sc.model):
            try:
                ev = llm.complete_json("escalate", SCORE_SYSTEM, user_msg, max_tokens=2500)
                escalated = 1
            except llm.BudgetExceeded:
                pass  # keep the local eval
        score = int(ev.get("score", 0))
        # Below the review floor → auto-skip: kept in the DB (visible on the
        # Jobs page, re-evaluable) but not surfaced for human review.
        status = "pending_user_review" if score >= settings.review_min_score else "skipped"
        with db.connect() as conn:
            # status guard: if the user rejected/deleted this job while the
            # eval was in flight, their decision wins — don't overwrite it.
            conn.execute(
                """UPDATE jobs SET status=?, score=?, pitch=?,
                   concerns=?, scam_risk=?, eval_json=?, escalated=?
                   WHERE id=? AND status='pending_eval'""",
                (status, score, ev.get("pitch"), ev.get("concerns"),
                 ev.get("scam_risk"), json.dumps(ev), escalated, job_id),
            )
    except Exception as e:
        with db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='error', error=? WHERE id=? AND status='pending_eval'",
                (str(e)[:500], job_id))


def prefilter_pending() -> int:
    """Sweep the whole pending_eval queue through the title keyword gate in one
    pass and mark the matches 'filtered' en masse — before any LLM call. Without
    this the (microsecond) keyword check runs inside evaluate_job, so a keyword
    reject waits its turn behind multi-second evals ahead of it. Loads each
    applicant's profile once, not once per job. Returns the count filtered."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, applicant_id, title, company FROM jobs WHERE status='pending_eval'").fetchall()
        profiles: dict = {}
        to_filter = []
        for r in rows:
            aid = r["applicant_id"]
            if aid not in profiles:
                profiles[aid] = db.latest_profile(conn, aid)
            prof = profiles[aid]
            if prof is None:
                continue
            kw = _excluded_kw(r["title"], prof, r["company"])
            if kw:
                to_filter.append((kw, r["id"]))
        conn.executemany(
            "UPDATE jobs SET status='filtered', error=? WHERE id=? AND status='pending_eval'",
            to_filter)
    return len(to_filter)


def evaluate_pending() -> int:
    # Blocking acquire: if a full cycle is mid-run this waits for it rather than
    # opening a competing writer loop. Reentrant, so run_full_cycle's own call
    # (same thread, lock already held) passes straight through.
    with _RUN_LOCK:
        # Callers other than run_full_cycle (startup resume, the bulk-eval
        # endpoint, a manual rescore) invoke this directly, so RUN_STATUS
        # otherwise never leaves "idle" for them — the Status tab would show
        # nothing happening while a real eval pass grinds through the queue.
        standalone = not RUN_STATUS["running"]
        if standalone:
            RUN_STATUS.update(running=True, phase="evaluating", started_at=db.now(),
                               finished_at=None, error=None)
        # Cheap rejects first, in bulk, so the LLM loop only ever sees jobs that
        # actually need an eval. evaluate_job keeps its own gate as a safety net
        # (single-job reevaluate skips this path).
        prefilter_pending()
        with db.connect() as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM jobs WHERE status='pending_eval'")]
        RUN_STATUS.update(eval_total=len(ids), eval_done=0)
        done = 0
        for jid in ids:
            if _STOP.is_set():
                break
            evaluate_job(jid)
            done += 1
            RUN_STATUS["eval_done"] = done
        if standalone:
            RUN_STATUS.update(running=False, evaluated=done, finished_at=db.now(),
                               phase="stopped" if _STOP.is_set() else "done")
    return done


def generate_documents(job_id: str):
    """Called on approval. Writes resume + cover letter, renders HTML."""
    with db.connect() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            return
        profile = db.latest_profile(conn, job["applicant_id"])

    user_msg = (
        f"APPLICANT PROFILE:\n{json.dumps(profile, indent=1)}\n\n"
        f"JOB POSTING:\nTitle: {job['title']}\nCompany: {job['company']}\n"
        f"Location: {job['location']}\n\n{job['description']}\n\n"
        f"EVALUATOR'S PITCH: {job['pitch']}\nCONCERNS: {job['concerns']}"
    )
    if job["notes"]:
        user_msg += f"\nREVIEWER NOTE: {job['notes']}"
    try:
        text = llm.complete("write", WRITE_SYSTEM, user_msg, max_tokens=6000)
        m = re.search(r"===\s*RESUME\s*===\s*(.*?)\s*===\s*COVER\s*LETTER\s*===\s*(.*)",
                      text, re.DOTALL | re.IGNORECASE)
        if not m:
            raise llm.LLMError(
                f"write output missing ===RESUME===/===COVER LETTER=== markers. "
                f"Tail: ...{text[-200:]}")
        docs = {"resume_md": m.group(1).strip(), "cover_letter_md": m.group(2).strip()}
        with db.connect() as conn:
            for kind in ("resume", "cover_letter"):
                md = docs.get(f"{kind}_md")
                if not md:
                    continue
                path_md, path_html = render.save(job, kind, md)
                conn.execute(
                    "INSERT INTO documents (id, job_id, kind, path_md, path_html, created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (db.new_id(), job_id, kind, path_md, path_html, db.now()),
                )
            conn.execute("UPDATE jobs SET status='draft_created' WHERE id=?", (job_id,))
    except Exception as e:
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='error', error=? WHERE id=?", (str(e)[:500], job_id))


SCAFFOLD_SYSTEM = """You prepare a job-application FRAMING OUTLINE — not a resume, \
not a cover letter. The applicant drafts the actual materials themselves (often
iterating in a chat session); your outline is their reference and starting
structure. Be specific to THIS job and THIS profile; never invent experience.

Output a Markdown outline with exactly these sections:
## Lead with
The 2-4 profile items most relevant to this posting, and why each maps to it.
## Framing per requirement
For each significant requirement in the posting: the matching profile evidence
and how to phrase it (honor the profile's framing_notes).
## Gaps and mitigation
Requirements the profile doesn't cover, and honest ways to address or defuse each.
## Keywords
Terms from the posting worth echoing verbatim in a resume/cover letter.
## Cover letter angle
2-3 sentences: the single strongest narrative connecting applicant to role.
## Open questions
Anything the applicant should verify or decide before applying.

If a REVIEWER NOTE is present, treat it as authoritative supplemental profile
information — it overrides the evaluator's pitch/concerns where they conflict."""


def generate_scaffold(job_id: str):
    """Cheap alternative to generate_documents: a framing outline the applicant
    drafts from manually. Job stays shortlisted; the outline attaches as a doc."""
    with db.connect() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            return
        profile = db.latest_profile(conn, job["applicant_id"])

    user_msg = (
        f"APPLICANT PROFILE:\n{json.dumps(profile, indent=1)}\n\n"
        f"JOB POSTING:\nTitle: {job['title']}\nCompany: {job['company']}\n"
        f"Location: {job['location']}\n\n{job['description']}\n\n"
        f"EVALUATOR'S PITCH: {job['pitch']}\nCONCERNS: {job['concerns']}"
    )
    if job["notes"]:
        user_msg += f"\nREVIEWER NOTE: {job['notes']}"
    try:
        # Generous budget: reasoning models burn thinking tokens against it.
        md = llm.complete("scaffold", SCAFFOLD_SYSTEM, user_msg, max_tokens=6000)
        md = re.sub(r"<think>.*?</think>", "", md, flags=re.DOTALL).strip()
        path_md, path_html = render.save(job, "scaffold", md)
        with db.connect() as conn:
            # Regenerating replaces the old outline instead of stacking copies.
            conn.execute("DELETE FROM documents WHERE job_id=? AND kind='scaffold'", (job_id,))
            conn.execute(
                "INSERT INTO documents (id, job_id, kind, path_md, path_html, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (db.new_id(), job_id, "scaffold", path_md, path_html, db.now()),
            )
            conn.execute("UPDATE jobs SET error=NULL WHERE id=?", (job_id,))
    except Exception as e:
        # Stay shortlisted so the card doesn't vanish; surface the error there.
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET error=? WHERE id=?",
                         (f"scaffold: {str(e)[:400]}", job_id))


# Progress of the current/last full cycle, read by GET /run/status. Single
# process + GIL means plain dict updates are safe enough for a status readout.
# eval_total/eval_done drive the progress meter on the Status tab — set at the
# start of a batch eval and incremented per job, since that's the slow,
# one-LLM-call-at-a-time phase where "is this actually moving" matters most.
RUN_STATUS = {"running": False, "phase": "idle", "started_at": None,
              "finished_at": None, "ingested": None, "evaluated": None,
              "closed": None, "error": None, "eval_total": None, "eval_done": None}

# Only one writer-heavy batch operation runs at a time. Reentrant so a
# run_full_cycle holding the lock can still call evaluate_pending in the same
# thread; a *different* thread (rescore, cron, a second click) waits its turn
# instead of racing another writer into a "database is locked".
_RUN_LOCK = threading.RLock()
# Cooperative cancel. The batch loops check it at each item boundary; endpoints
# that intentionally start work clear it first (see main.clear_stop()).
_STOP = threading.Event()


def request_stop():
    """Ask the in-flight run/eval loop to stop at the next item boundary."""
    _STOP.set()


def clear_stop():
    _STOP.clear()


def is_stopping() -> bool:
    return _STOP.is_set()


def run_full_cycle() -> dict:
    # Non-blocking: a second trigger (double-click, cron landing on a manual run)
    # just bows out rather than stacking a second writer loop.
    if not _RUN_LOCK.acquire(blocking=False):
        return {"skipped": "a run is already in progress"}
    RUN_STATUS.update(running=True, phase="ingesting", started_at=db.now(),
                      finished_at=None, ingested=None, evaluated=None,
                      closed=None, error=None)
    try:
        added, closed = ingest_all()
        RUN_STATUS.update(ingested=added, closed=closed)
        if _STOP.is_set():
            RUN_STATUS["phase"] = "stopped"
            from . import notify
            notify.notify("seeker: pipeline stopped",
                           f"stopped early — {added} ingested, {closed} closed, evaluation not reached")
            notify.ping_pipeline(ok=True)  # stopped by request still counts as "it ran"
            return {"ingested": added, "closed": closed, "stopped": True}
        RUN_STATUS["phase"] = "evaluating"
        RUN_STATUS["evaluated"] = evaluated = evaluate_pending()
        RUN_STATUS["phase"] = "stopped" if _STOP.is_set() else "done"
        db.set_meta("last_cycle", db.now())
        from . import notify
        if _STOP.is_set():
            notify.notify("seeker: pipeline stopped",
                           f"stopped early — {added} ingested, {evaluated} evaluated, {closed} closed")
        else:
            notify.notify("seeker: pipeline finished",
                           f"{added} ingested · {evaluated} evaluated · {closed} closed")
        notify.ping_pipeline(ok=True)
        return {"ingested": added, "evaluated": evaluated, "closed": closed,
                "stopped": _STOP.is_set()}
    except Exception as e:
        RUN_STATUS.update(phase="error", error=str(e)[:300])
        from . import notify
        notify.notify("seeker: pipeline failed", str(e)[:300])
        notify.ping_pipeline(ok=False)
        raise
    finally:
        RUN_STATUS.update(running=False, finished_at=db.now())
        _RUN_LOCK.release()


# --- URL liveness probe ------------------------------------------------------
# Covers the sources board-diff can't judge: search APIs (adzuna, usajobs,
# workday), per-job scrapers (icims, govjobs), and manual submissions — plus
# older enumerator jobs stored before poll_source existed (they have NULL and
# so miss the board-diff path). Disjoint from _sweep_closed, which owns rows
# whose poll_source is a full enumerator, so the two never fight over a row.
_PROBE_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
# Curated closure phrases (lowercased). Kept conservative: a match only ever
# *flags* for human review, never auto-archives, and still needs the 2-strike
# grace, so an occasional false hit costs a glance, not a lost application.
_CLOSED_PHRASES = (
    "no longer accepting applications", "no longer accepting application",
    "this job is no longer available", "this position is no longer available",
    "position has been filled", "this position has been filled",
    "posting is no longer active", "this posting is no longer active",
    "job posting has expired", "this job has expired", "posting has closed",
    "this position is closed", "job you are looking for is no longer",
    "the job you requested was not found", "requisition is closed",
)


def _liveness_verdict(url: str) -> tuple[bool | None, str]:
    """GET the posting. Returns (closed?, reason). closed=None means we couldn't
    tell (network error, 403/429/5xx) — the caller must leave state untouched
    rather than count it as a miss. GET, not HEAD: many ATS mishandle HEAD, and
    we need the body for the closure-phrase check anyway."""
    try:
        r = httpx.get(url, headers={"User-Agent": _PROBE_UA}, timeout=15,
                      follow_redirects=True)
    except Exception:
        return None, ""          # couldn't reach it — inconclusive
    if r.status_code in (404, 410):
        return True, str(r.status_code)
    if r.status_code >= 400:
        return None, ""          # blocked / rate-limited / server error
    body = r.text.lower()
    if any(p in body for p in _CLOSED_PHRASES):
        return True, "expired-text"
    return False, ""


def probe_liveness() -> int:
    """Daily out-of-band check for the non-enumerator active jobs. One GET each,
    politeness-throttled; the acting-on set is small (tens of rows). Returns the
    count newly flagged closed."""
    from . import notify
    from .sources import FULL_ENUMERATORS
    try:
        marks = ",".join("?" * len(FULL_ENUMERATORS))
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT id, url, missed_count, closed_at FROM jobs "
                f"WHERE url IS NOT NULL AND url != '' AND status IN {_ACTIVE_STATUSES} "
                f"AND closure_dismissed=0 "
                f"AND (poll_source IS NULL OR poll_source NOT IN ({marks}))",
                tuple(FULL_ENUMERATORS)).fetchall()
        closed = 0
        for r in rows:
            verdict, reason = _liveness_verdict(r["url"])
            if verdict is None:
                continue             # inconclusive — don't touch missed_count
            with db.connect() as conn:
                closed += _mark(conn, r, verdict, reason)
            time.sleep(0.5)          # politeness: their site, our schedule
    except Exception:
        notify.ping_liveness(ok=False)
        raise
    if closed:
        print(f"[seeker] liveness probe flagged {closed} job(s) closed")
    notify.ping_liveness(ok=True)
    return closed
