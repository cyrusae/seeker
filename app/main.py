"""FastAPI app: review dashboard, manual submission, applications, usage."""
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import json

from apscheduler.events import EVENT_JOB_ERROR
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import compare, db, notify, pipeline, report
from .config import settings
from .sources import manual

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


def _money(s):
    """Comma-group any 4+ digit numbers in a salary string; drop '.0' cents.
    '55000.0–72000.0 (adzuna est.)' → '55,000–72,000 (adzuna est.)'"""
    if not s:
        return s

    def fmt(m):
        int_part, _, dec = m.group(0).partition(".")
        out = f"{int(int_part):,}"
        if dec.strip("0"):
            out += f".{dec}"
        return out

    return re.sub(r"\d{4,}(?:\.\d+)?", fmt, str(s))


templates.env.filters["money"] = _money


def _days_ago(ts: str | None) -> int | None:
    """Whole days between an ISO timestamp and now (None if missing/garbled)."""
    if not ts:
        return None
    try:
        then = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - then).days)


templates.env.filters["days_ago"] = _days_ago
templates.env.filters["gate_info"] = lambda j: _gate_info(j)


_THRESH_CACHE: dict = {"at": 0.0, "t": None}


def _gate_thresholds() -> dict:
    """Active gate thresholds, cached briefly so a page of cards doesn't hit
    the DB once per card."""
    import time as _t
    if _THRESH_CACHE["t"] is None or _t.time() - _THRESH_CACHE["at"] > 30:
        from . import gate
        _THRESH_CACHE.update(at=_t.time(), t=gate.active_model()["thresholds"])
    return _THRESH_CACHE["t"]


def _gate_compare(j) -> dict | None:
    """What DeepSeek did vs what the Jev gate would do, as decisions rather
    than raw numbers: the two scores live on different scales (DeepSeek's is a
    0-100 judgment, Jev's a calibrated probability x 100), so a point
    difference means little; a different decision is what matters."""
    if j["gate_score"] is None:
        return None
    th = _gate_thresholds()
    info = _gate_info(j)
    gs = j["gate_score"]
    if info.get("vetoes"):
        jev = "skip"
    else:
        jev = "gloss" if gs >= th["gloss_min"] else "review" if gs >= th["review_min"] else "skip"
    ds = None
    if j["score"] is not None:
        if j["score"] < settings.review_min_score:
            ds = "skip"
        elif j["escalated"] or j["score"] >= settings.escalate_min_score:
            ds = "gloss"
        else:
            ds = "review"
    return {"jev": jev, "ds": ds, "agree": ds is None or ds == jev, "score": gs,
            "vetoes": info.get("vetoes") or [], "for": info.get("for") or [],
            "against": info.get("against") or []}


templates.env.filters["gate_cmp"] = _gate_compare


def _show_gate(request: Request) -> bool:
    return request.cookies.get("show_gate") == "1"


def _gate_nudge():
    from . import gatefit
    return gatefit.nudge()


templates.env.globals["gate_nudge"] = _gate_nudge


def _gate_info(j) -> dict:
    try:
        return json.loads(j["gate_json"] or "{}")
    except (ValueError, TypeError):
        return {}


def _deadline_passed(j) -> bool:
    """The Jev gate read a stated application deadline that's already past."""
    return _gate_info(j).get("deadline_passed", 0) >= 0.8


def _freshness(j) -> str:
    """Review-queue bucket: 'stale' (flagged closed, a stated deadline has
    passed, or no source has listed it in STALE_AFTER_DAYS), 'recent' (listed
    within RECENT_SEEN_DAYS), else 'aging'. last_seen_at falls back to
    created_at for safety."""
    if j["closed_at"] or _deadline_passed(j):
        return "stale"
    seen = _days_ago(j["last_seen_at"] or j["created_at"])
    if seen is None:
        return "aging"
    if seen >= settings.stale_after_days:
        return "stale"
    return "recent" if seen < settings.recent_seen_days else "aging"


def _resume_interrupted():
    """A restart (or --reload picking up a file edit) kills in-flight background
    tasks. Pick the queue back up: everything still waiting for an eval."""
    with db.connect() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE status='pending_eval'").fetchone()["n"]
    if pending:
        print(f"[seeker] resuming interrupted work: {pending} pending evals")
        pipeline.evaluate_pending()


# If the daily cron slot was missed entirely (laptop powered off at 2am, not
# just asleep), catch up on the next startup when the last full cycle is older
# than this. Well under 24h so one skipped night still triggers a make-up run.
CATCHUP_AFTER_HOURS = 20


def _catch_up_if_overdue():
    """Cover the gap the cron can't: a machine that was *off* at the scheduled
    hour never records a misfire, so APScheduler just waits for the next slot.
    On startup, if we haven't completed a cycle recently, run one now."""
    last = db.get_meta("last_cycle")
    if last:
        try:
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(last)).total_seconds() / 3600
            if age_h < CATCHUP_AFTER_HOURS:
                return
        except ValueError:
            pass  # unparseable marker — treat as overdue
    print(f"[seeker] last full cycle was {last or 'never'} — running catch-up")
    pipeline.run_full_cycle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    sched = BackgroundScheduler()
    # misfire_grace_time=None: if the box was merely asleep at the slot, run the
    # job whenever the scheduler thread resumes rather than silently dropping it
    # (APScheduler's default 1s grace discards anything late). coalesce collapses
    # several missed firings into a single make-up run.
    sched.add_job(pipeline.run_full_cycle, "cron", hour=settings.pipeline_hour,
                  minute=0, misfire_grace_time=None, coalesce=True)
    # URL liveness probe on its own daily slot, offset 6h so it doesn't pile
    # external requests onto the ingest+eval cycle. Board-diff closure already
    # runs inside run_full_cycle; this covers the sources it can't judge.
    sched.add_job(pipeline.probe_liveness, "cron",
                  hour=(settings.pipeline_hour + 6) % 24, minute=0,
                  misfire_grace_time=None, coalesce=True)
    # Desktop notification for any scheduled job that crashes outright — the
    # net that catches surprises pipeline.run_full_cycle's own try/except
    # doesn't (e.g. a probe_liveness failure), so a 2am cron death doesn't go
    # unnoticed just because no one had a terminal open to see the traceback.
    def _on_job_error(event):
        exc = getattr(event, "exception", None)
        notify.notify(f"seeker: {event.job_id} crashed", str(exc)[:300])
    sched.add_listener(_on_job_error, EVENT_JOB_ERROR)
    sched.start()
    # One thread, run in sequence — never two writer loops at once. (Running
    # resume and catch-up concurrently is what caused the "database is locked"
    # storm: two pipelines hammering SQLite from different threads.)
    threading.Thread(target=_startup_tasks, daemon=True).start()
    yield
    sched.shutdown(wait=False)


def _startup_tasks():
    _resume_interrupted()
    _catch_up_if_overdue()


app = FastAPI(title="seeker v2", lifespan=lifespan)


def _applicants(conn):
    return [p["applicant_id"] for p in db.all_applicants(conn) if p]


def _sticky_applicant(request: Request, applicant: str) -> str:
    """The applicant filter persists via cookie until explicitly changed.
    An explicit ?applicant= (even empty = "all") wins over the cookie."""
    if "applicant" in request.query_params:
        return applicant
    return request.cookies.get("applicant", "")


def _remember_applicant(request: Request, resp, applicant: str):
    if "applicant" in request.query_params:
        resp.set_cookie("applicant", applicant, max_age=365 * 24 * 3600)
    return resp


REVIEW_SORTS = {"fit": "best fit first", "newest": "newest first", "oldest": "oldest first"}


@app.get("/")
def review(request: Request, applicant: str = "", view: str = "", sort: str = "fit"):
    applicant = _sticky_applicant(request, applicant)
    with db.connect() as conn:
        # self-join pulls the near-duplicate's title/status for the ⚠ line
        q = """SELECT j.*, s.title similar_title, s.status similar_status
               FROM jobs j LEFT JOIN jobs s ON s.id = j.similar_to
               WHERE j.status='pending_user_review'"""
        args = []
        if applicant:
            q += " AND j.applicant_id=?"
            args.append(applicant)
        jobs = conn.execute(q + " ORDER BY j.score DESC", args).fetchall()
        # queue counts follow the applicant filter ("" = all applicants)
        cq, cargs = "SELECT status, COUNT(*) n FROM jobs", []
        if applicant:
            cq += " WHERE applicant_id=?"
            cargs.append(applicant)
        counts = {r["status"]: r["n"] for r in
                  conn.execute(cq + " GROUP BY status", cargs)}
        names = _applicants(conn)
    # Split the queue by how recently a source last showed the posting, so a
    # backlog built up over weeks unattended doesn't bury live jobs under
    # ones that have probably been filled. Score order is kept within each.
    if sort not in REVIEW_SORTS:
        sort = "fit"
    if sort != "fit":  # age = first seen; SQL already gave score order
        jobs = sorted(jobs, key=lambda j: j["created_at"] or "", reverse=(sort == "newest"))
    buckets = {"recent": [], "aging": [], "stale": []}
    for j in jobs:
        buckets[_freshness(j)].append(j)
    rd, sd = settings.recent_seen_days, settings.stale_after_days
    if view == "stale":
        sections = [
            ("Likely closed", "The board stopped listing these, the posting URL says "
             "they're gone, or the posting's own application deadline has passed.",
             [j for j in buckets["stale"] if j["closed_at"] or _deadline_passed(j)]),
            (f"Not seen in {sd}+ days", "No closure signal, but no source has listed "
             "these in a long time. Check the posting before spending time on one.",
             [j for j in buckets["stale"] if not (j["closed_at"] or _deadline_passed(j))]),
        ]
    else:
        view = ""
        sections = [
            ("Seen recently", f"A source listed these within the last {rd} days.",
             buckets["recent"]),
            ("Not seen recently", f"Last listed {rd}–{sd} days ago. May still be "
             "open, especially from search sources that only re-list what matches "
             "today's query.", buckets["aging"]),
        ]
    resp = templates.TemplateResponse(request, "review.html", {
        "sections": sections, "view": view, "stale_count": len(buckets["stale"]),
        "sort": sort, "sorts": REVIEW_SORTS, "show_gate": _show_gate(request),
        "live_count": len(buckets["recent"]) + len(buckets["aging"]),
        "counts": counts, "applicants": names, "selected": applicant,
    })
    return _remember_applicant(request, resp, applicant)


@app.post("/jobs/{job_id}/reject")
def reject(job_id: str, feedback: str = Form(""), redirect: str = Form("/")):
    with db.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='rejected', feedback=?, reviewed_at=? WHERE id=?",
            (feedback, db.now(), job_id))
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/archive")
def archive(job_id: str, feedback: str = Form(""), redirect: str = Form("/applications")):
    """Withdraw a job from the applications flow (shortlisted, then changed
    mind). Same negative tuning signal as reject, separate status so the
    Applications archived tab isn't flooded with ordinary Review rejects."""
    with db.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='archived', feedback=COALESCE(NULLIF(?, ''), feedback), "
            "reviewed_at=? WHERE id=?",
            (feedback.strip(), db.now(), job_id))
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/eval")
def eval_now(job_id: str, background: BackgroundTasks):
    background.add_task(pipeline.evaluate_job, job_id)
    return RedirectResponse("/submit?msg=Evaluating…", status_code=303)


@app.get("/submit")
def submit_form(request: Request, msg: str = ""):
    with db.connect() as conn:
        names = _applicants(conn)
        recent = conn.execute(
            "SELECT * FROM jobs WHERE source='manual' ORDER BY created_at DESC LIMIT 15"
        ).fetchall()
    return templates.TemplateResponse(request, "submit.html", {
        "applicants": names, "msg": msg, "recent": recent,
    })


@app.post("/submit")
def submit(background: BackgroundTasks, applicant_id: str = Form(...),
           url: str = Form(""), text: str = Form("")):
    url, text = url.strip(), text.strip()
    if not url and not text:
        return RedirectResponse("/submit?msg=Provide a URL or pasted text.", status_code=303)

    # Insert a visible placeholder first — a hung or failed extraction should
    # show up as a status on the submit page, never vanish.
    with db.connect() as conn:
        jid = db.insert_job(conn, applicant_id=applicant_id, source="manual",
                            dedupe_key=db.new_id(), url=url or None,
                            title="(extracting…)", description=(text or url)[:2000],
                            status="pending_extract")

    def process():
        try:
            raw = text or manual.fetch_url(url)
            job = manual.extract(raw, url or None)
            with db.connect() as conn:
                dupe = conn.execute(
                    "SELECT id FROM jobs WHERE applicant_id=? AND dedupe_key=? AND id<>?",
                    (applicant_id, job["dedupe_key"], jid)).fetchone()
                if dupe:
                    conn.execute(
                        "UPDATE jobs SET status='error', title=?, error=? WHERE id=?",
                        (job["title"], f"duplicate of existing job {dupe['id']}", jid))
                    return
                conn.execute(
                    """UPDATE jobs SET url=?, title=?, company=?, location=?, salary=?,
                       description=?, dedupe_key=?, status='pending_eval' WHERE id=?""",
                    (job["url"], job["title"], job["company"], job["location"],
                     job["salary"], job["description"], job["dedupe_key"], jid))
            pipeline.evaluate_job(jid)
        except Exception as e:
            with db.connect() as conn:
                conn.execute("UPDATE jobs SET status='error', error=? WHERE id=?",
                             (str(e)[:500], jid))

    background.add_task(process)
    return RedirectResponse("/submit?msg=Queued — extraction and eval running.", status_code=303)


REEVAL_OK = {"pending_user_review", "rejected", "shortlisted", "error", "pending_eval",
             "skipped", "filtered", "archived", "stale"}


@app.get("/jobs")
def jobs_page(request: Request, applicant: str = "", status: str = "", disagree: int = 0):
    applicant = _sticky_applicant(request, applicant)
    show_gate = _show_gate(request)
    q = "SELECT * FROM jobs WHERE 1=1"
    args = []
    if show_gate and disagree:
        q += " AND gate_score IS NOT NULL"
    if applicant:
        q += " AND applicant_id=?"
        args.append(applicant)
    if status:
        q += " AND status=?"
        args.append(status)
    with db.connect() as conn:
        jobs = conn.execute(q + " ORDER BY created_at DESC LIMIT 500", args).fetchall()
        names = _applicants(conn)
        statuses = [r["status"] for r in conn.execute("SELECT DISTINCT status FROM jobs")]
    if show_gate and disagree:
        jobs = [j for j in jobs if not (_gate_compare(j) or {"agree": True})["agree"]]
    resp = templates.TemplateResponse(request, "jobs.html", {
        "jobs": jobs, "applicants": names, "selected": applicant,
        "statuses": sorted(statuses), "status_sel": status,
        "show_gate": show_gate, "disagree": bool(disagree),
    })
    return _remember_applicant(request, resp, applicant)


@app.get("/companies")
def companies_page(request: Request, applicant: str = "", view: str = "active",
                   arch: str = "", appl: str = ""):
    applicant = _sticky_applicant(request, applicant)
    if view not in ("active", "shortlist"):
        view = "active"
    include_archived, include_applied = bool(arch), bool(appl)
    statuses = compare.statuses_for(view, include_archived, include_applied)
    with db.connect() as conn:
        groups = compare.company_groups(conn, applicant, statuses)
        names = _applicants(conn)
    resp = templates.TemplateResponse(request, "companies.html", {
        "groups": groups, "applicants": names, "selected": applicant,
        "view": view, "include_archived": include_archived,
        "include_applied": include_applied,
    })
    return _remember_applicant(request, resp, applicant)


@app.get("/jobs/{job_id}/card")
def job_card(request: Request, job_id: str, applicant: str = ""):
    with db.connect() as conn:
        j = conn.execute(
            """SELECT j.*, s.title similar_title, s.status similar_status
               FROM jobs j LEFT JOIN jobs s ON s.id = j.similar_to
               WHERE j.id=?""", (job_id,)).fetchone()
        if not j:
            return PlainTextResponse("not found", status_code=404)
    return templates.TemplateResponse(request, "_jobcard.html", {
        "j": j, "applicant": applicant,
    })


@app.post("/compare")
def compare_now(job_a: str = Form(...), job_b: str = Form(...),
                refresh: str = Form("")):
    if job_a == job_b:
        return JSONResponse({"error": "pick two different listings"}, status_code=400)
    with db.connect() as conn:
        result = compare.compare_pair(conn, job_a, job_b, refresh=bool(refresh))
    return JSONResponse(result)


@app.post("/jobs/bulk")
def jobs_bulk(background: BackgroundTasks, action: str = Form(...),
              job_ids: list[str] = Form([]), redirect: str = Form("/jobs")):
    if not job_ids:
        return RedirectResponse(redirect, status_code=303)
    marks = ",".join("?" * len(job_ids))
    with db.connect() as conn:
        if action == "delete":
            conn.execute(f"DELETE FROM documents WHERE job_id IN ({marks})", job_ids)
            conn.execute(f"DELETE FROM jobs WHERE id IN ({marks})", job_ids)
        elif action == "reject":
            # Unlike delete, this keeps the row (and its dedupe key), so polled
            # jobs won't come back on the next ingest. Never touches applied jobs.
            conn.execute(
                f"""UPDATE jobs SET status='rejected', reviewed_at=? WHERE id IN ({marks})
                    AND status IN ('pending_extract','pending_eval','pending_user_review',
                                   'skipped','shortlisted','error','filtered','archived','stale')""",
                [db.now(), *job_ids])
        elif action == "reevaluate":
            conn.execute(
                f"""UPDATE jobs SET status='pending_eval', error=NULL, score=NULL,
                    pitch=NULL, concerns=NULL, scam_risk=NULL, eval_json=NULL, escalated=0
                    WHERE id IN ({marks}) AND status IN
                    ('pending_user_review','rejected','shortlisted','error','pending_eval',
                     'skipped','filtered','archived','stale')""",
                job_ids)
        elif action == "restore":
            # Undo a stale dismissal: you've checked and it's still open, so
            # it counts as a fresh sighting (otherwise its age would drop it
            # straight back into the stale view).
            conn.execute(
                f"""UPDATE jobs SET status='pending_user_review', reviewed_at=NULL,
                    closed_at=NULL, closed_reason=NULL, missed_count=0,
                    closure_dismissed=1, last_seen_at=?
                    WHERE id IN ({marks}) AND status='stale'""",
                [db.now(), *job_ids])
    if action == "reevaluate":
        pipeline.clear_stop()  # explicit re-evaluate overrides a prior stop
        background.add_task(pipeline.evaluate_pending)
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/reevaluate")
def reevaluate(job_id: str, background: BackgroundTasks, redirect: str = Form("/jobs")):
    """Fresh eval against the CURRENT profile — re-fetches thin descriptions too."""
    with db.connect() as conn:
        job = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or job["status"] not in REEVAL_OK:
            return RedirectResponse(redirect, status_code=303)
        conn.execute(
            "UPDATE jobs SET status='pending_eval', error=NULL, score=NULL, pitch=NULL, "
            "concerns=NULL, scam_risk=NULL, eval_json=NULL, escalated=0 WHERE id=?",
            (job_id,))
    background.add_task(pipeline.evaluate_job, job_id)
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/shortlist")
def shortlist(job_id: str, redirect: str = Form("/"), note: str = Form("")):
    """Confirmed-interested. Counts as a positive signal in the tuning report.
    A note entered now is kept for later reference (e.g. context to bring
    into a chat session when drafting materials)."""
    with db.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='shortlisted', reviewed_at=?, error=NULL, "
            "notes=COALESCE(NULLIF(?, ''), notes) WHERE id=?",
            (db.now(), note.strip(), job_id))
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/note")
def save_note(job_id: str, redirect: str = Form("/applications?tab=shortlist"),
              note: str = Form("")):
    """Persist an annotation without moving the job out of its zone — e.g.
    'closes July 18th'. Unlike the zone-changing buttons, this writes the note
    verbatim (including clearing it), since editing is the whole point."""
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET notes=? WHERE id=?", (note.strip(), job_id))
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/unflag")
def unflag_closed(job_id: str, redirect: str = Form("/applications?tab=shortlist")):
    """Human override: the posting is actually still open (the probe was wrong).
    Clear the closure flag AND set closure_dismissed so neither the board-diff
    sweep nor the URL probe re-flags this job."""
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET closed_at=NULL, closed_reason=NULL, "
                     "missed_count=0, closure_dismissed=1, last_seen_at=? WHERE id=?",
                     (db.now(), job_id))
    return RedirectResponse(redirect, status_code=303)


def _set_stale(conn, job_ids: list[str], reason: str) -> None:
    """Move review-queue jobs to status 'stale': gone or too old to bother with.
    Its own status (not 'archived' or 'rejected') so it stays out of the
    Applications archived tab and out of profile tuning — a dead posting says
    nothing about match quality. Keeps any existing closure reason; otherwise
    records why it was dismissed ('manual' = you saw it's gone, 'stale' = old)."""
    if not job_ids:
        return
    now = db.now()
    marks = ",".join("?" * len(job_ids))
    conn.execute(
        f"""UPDATE jobs SET status='stale', reviewed_at=?,
            closed_at=COALESCE(closed_at, ?), closed_reason=COALESCE(closed_reason, ?),
            closure_dismissed=1
            WHERE id IN ({marks}) AND status IN ('pending_user_review','skipped')""",
        [now, now, reason, *job_ids])


@app.post("/jobs/{job_id}/stale")
def mark_stale(job_id: str, reason: str = Form("manual"), redirect: str = Form("/")):
    with db.connect() as conn:
        _set_stale(conn, [job_id], "stale" if reason == "stale" else "manual")
    return RedirectResponse(redirect, status_code=303)


@app.post("/review/stale")
def bulk_stale(job_ids: list[str] = Form([]), redirect: str = Form("/?view=stale")):
    with db.connect() as conn:
        _set_stale(conn, job_ids, "stale")
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/close")
def mark_closed(job_id: str, redirect: str = Form("/")):
    """Human override: the posting is gone (dead link, or you just know). Archive
    it out of the active flow AND stamp the closure marker; closure_dismissed
    stops the board-diff sweep 'recovering' it if the stale listing lingers.
    Unlike an ordinary archive this is NOT a match-quality miss — a closed job
    says nothing about the profile — so the tuning report skips archived+closed
    jobs (report.py). The '↺ Not closed' undo clears the flag like any other."""
    now = db.now()
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET status='archived', reviewed_at=?, closed_at=?, "
                     "closed_reason='manual', closure_dismissed=1 WHERE id=?",
                     (now, now, job_id))
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/applied")
def mark_applied(job_id: str, redirect: str = Form("/"), note: str = Form("")):
    """Applied outside the pipeline (shared portal resume, claude.ai draft...).
    Records the strongest positive tuning signal without a write call."""
    with db.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='applied', reviewed_at=?, "
            "notes=COALESCE(NULLIF(?, ''), notes) WHERE id=?",
            (db.now(), note.strip(), job_id))
    return RedirectResponse(redirect, status_code=303)


@app.post("/rescore")
def rescore(background: BackgroundTasks, applicant: str = Form(""),
            include_skipped: str = Form(""), include_filtered: str = Form("")):
    """Throw unreviewed jobs back to the eval queue — for after a profile
    re-import. Reviewed/drafted jobs are never touched. Optionally includes
    auto-skipped jobs (their skip was decided under the old profile) and
    keyword-filtered jobs (whose title_exclude may have changed). Re-queuing
    filtered jobs is cheap: the keyword gate re-runs before any LLM call, so
    still-excluded titles bounce straight back at zero token cost."""
    included = ["pending_user_review"]
    if include_skipped:
        included.append("skipped")
    if include_filtered:
        included.append("filtered")
    statuses = "(" + ",".join(f"'{s}'" for s in included) + ")"
    q = ("UPDATE jobs SET status='pending_eval', error=NULL, score=NULL, pitch=NULL, "
         "concerns=NULL, scam_risk=NULL, eval_json=NULL, escalated=0 "
         f"WHERE status IN {statuses}")
    args = []
    if applicant:
        q += " AND applicant_id=?"
        args.append(applicant)
    with db.connect() as conn:
        conn.execute(q, args)
    pipeline.clear_stop()  # explicit re-score overrides a prior stop
    background.add_task(pipeline.evaluate_pending)
    return RedirectResponse(f"/?applicant={applicant}", status_code=303)


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, background: BackgroundTasks, redirect: str = Form("/usage")):
    """Retry an errored job by sending it back through eval."""
    with db.connect() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or job["status"] != "error":
            return RedirectResponse(redirect, status_code=303)
        conn.execute(
            "UPDATE jobs SET status='pending_eval', error=NULL, score=NULL, "
            "pitch=NULL, concerns=NULL, scam_risk=NULL, eval_json=NULL, escalated=0 "
            "WHERE id=?", (job_id,))
    background.add_task(pipeline.evaluate_job, job_id)
    return RedirectResponse(redirect, status_code=303)


@app.post("/jobs/{job_id}/delete")
def delete_job(job_id: str, redirect: str = Form("/submit")):
    """Remove a job (and any documents) from the record entirely."""
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return RedirectResponse(redirect, status_code=303)


@app.post("/errors/clear")
def clear_errors(redirect: str = Form("/usage")):
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE job_id IN (SELECT id FROM jobs WHERE status='error')")
        conn.execute("DELETE FROM jobs WHERE status='error'")
    return RedirectResponse(redirect, status_code=303)


# Applications-page tabs → the job status behind each. "archived" = shortlisted
# then withdrawn (changed my mind), which keeps it as negative signal for tuning.
# Tab order mirrors the application lifecycle: shortlist → applied.
APPLICATION_TABS = {"shortlist": "shortlisted",
                     # "archived" is its own status (not 'rejected') so the tab shows
                     # only jobs withdrawn from this flow, not every Review reject.
                     "applied": "applied", "archived": "archived"}


@app.get("/applications")
def applications(request: Request, applicant: str = "", tab: str = "shortlist",
                  closed: str = ""):
    applicant = _sticky_applicant(request, applicant)
    if tab not in APPLICATION_TABS:
        tab = "shortlist"
    q = """SELECT id, created_at, title, company, pitch, notes, feedback,
                  applicant_id, url, description, error, closed_at, closed_reason
           FROM jobs WHERE status=?"""
    args: list = [APPLICATION_TABS[tab]]
    cq = "SELECT status s, COUNT(*) n FROM jobs WHERE status IN ('applied','shortlisted','archived')"
    cargs: list = []
    # count of flagged-closed jobs within the current tab (drives the filter chip)
    clq = "SELECT COUNT(*) n FROM jobs WHERE status=? AND closed_at IS NOT NULL"
    clargs: list = [APPLICATION_TABS[tab]]
    if applicant:
        q += " AND applicant_id=?"
        args.append(applicant)
        cq += " AND applicant_id=?"
        cargs.append(applicant)
        clq += " AND applicant_id=?"
        clargs.append(applicant)
    if closed:
        q += " AND closed_at IS NOT NULL"
    with db.connect() as conn:
        rows = conn.execute(q + " ORDER BY created_at DESC", args).fetchall()
        by_status = {r["s"]: r["n"] for r in conn.execute(cq + " GROUP BY status", cargs)}
        closed_count = conn.execute(clq, clargs).fetchone()["n"]
        names = _applicants(conn)
    tab_counts = {t: by_status.get(s, 0) for t, s in APPLICATION_TABS.items()}
    resp = templates.TemplateResponse(request, "applications.html", {
        "jobs": rows, "applicants": names, "selected": applicant,
        "tab": tab, "tab_counts": tab_counts,
        "closed_count": closed_count, "closed_filter": bool(closed)})
    return _remember_applicant(request, resp, applicant)


@app.get("/profiles")
def profiles_page(request: Request, msg: str = ""):
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT applicant_id, MAX(version) version, MAX(created_at) updated
               FROM profiles GROUP BY applicant_id ORDER BY applicant_id""").fetchall()
        profs = [{**dict(r), "name": (db.latest_profile(conn, r["applicant_id"]) or {}).get("name")}
                 for r in rows]
    from . import gate, gatefit
    gate.init_db()
    return templates.TemplateResponse(request, "profiles.html", {
        "profiles": profs, "msg": msg, "gate_model": gate.active_model(),
        "gate_new": gatefit.new_decisions(), "gate_min_new": gatefit.MIN_NEW_DECISIONS,
        "gate_mode": settings.gate_mode})


@app.get("/prefs/gate")
def toggle_gate(on: int = 0, back: str = "/"):
    """Show/hide Jev shadow scores on Review and Jobs (a per-browser cookie)."""
    if not back.startswith("/") or back.startswith("//"):
        back = "/"  # same-site paths only
    resp = RedirectResponse(back, status_code=303)
    resp.set_cookie("show_gate", "1" if on else "0", max_age=365 * 24 * 3600)
    return resp


@app.post("/gate/backfill")
def gate_backfill(background: BackgroundTasks):
    if not pipeline.BACKFILL_STATUS["running"]:
        background.add_task(pipeline.backfill_shadow)
    return RedirectResponse("/status", status_code=303)


@app.get("/gate")
def gate_page(request: Request):
    """Gate model: refit preview (candidate), apply/discard, version history."""
    from . import gate, gatefit
    gate.init_db()
    models = gate.list_models(15)
    cand = next((m for m in models if m["status"] == "candidate"), None)
    return templates.TemplateResponse(request, "gate.html", {
        "active": gate.active_model(), "candidate": cand,
        "history": [m for m in models if m["status"] in ("active", "retired")],
        "fit_status": gatefit.status(), "new": gatefit.new_decisions(),
        "missing": gatefit.missing_count(), "features": gate.FIT_FEATURES})


@app.post("/gate/fit")
def gate_fit(evaluate_missing: str = Form("")):
    from . import gatefit
    gatefit.start_background(bool(evaluate_missing))
    return RedirectResponse("/gate", status_code=303)


@app.post("/gate/{version}/apply")
def gate_apply(version: int):
    from . import gatefit
    n = gatefit.apply(version)
    return RedirectResponse(f"/profiles?msg=Gate model v{version} applied; "
                            f"{n} gate scores updated.", status_code=303)


@app.post("/gate/{version}/discard")
def gate_discard(version: int):
    from . import gate
    gate.discard(version)
    return RedirectResponse("/gate", status_code=303)


@app.post("/profiles/upload")
def profiles_upload(file: UploadFile = File(None), text: str = Form("")):
    """Import a new profile version: upload the JSON file or paste it.
    Same validation as `cli.py import-profile`."""
    try:
        raw = text.strip() or (file.file.read().decode("utf-8") if file else "")
        if not raw:
            raise ValueError("no file or pasted JSON provided")
        profile = json.loads(raw)
        missing = report.REQUIRED_PROFILE_KEYS - profile.keys()
        if missing:
            raise ValueError(f"profile missing required keys: {sorted(missing)} "
                             "(see docs/interview_spec.md)")
        version = db.import_profile(profile)
        msg = (f"Imported {profile['applicant_id']} as version {version}. "
               "Unreviewed jobs still carry old-profile scores — use Rescore "
               "on the Review page to re-queue them.")
    except (ValueError, json.JSONDecodeError) as e:
        msg = f"Import failed: {e}"
    return RedirectResponse(f"/profiles?msg={msg}", status_code=303)


@app.get("/profiles/{applicant_id}/report")
def profiles_report(applicant_id: str):
    """Tuning report download — same output as `cli.py tuning-report`."""
    with db.connect() as conn:
        if db.latest_profile(conn, applicant_id) is None:
            return PlainTextResponse("unknown applicant", status_code=404)
    md = report.tuning_report(applicant_id)
    fname = f"tuning-report-{applicant_id}-{db.now()[:10]}.md"
    return PlainTextResponse(md, headers={
        "Content-Disposition": f'attachment; filename="{fname}"'})


def _cycle_status() -> dict:
    last_cycle = db.get_meta("last_cycle")
    cycle_age_h = None
    if last_cycle:
        try:
            cycle_age_h = (datetime.now(timezone.utc)
                           - datetime.fromisoformat(last_cycle)).total_seconds() / 3600
        except ValueError:
            pass
    # Stale past a full day's slot = the nightly cron didn't land (asleep/off).
    cycle_overdue = cycle_age_h is None or cycle_age_h >= 24
    return {"last_cycle": last_cycle, "cycle_age_h": cycle_age_h, "cycle_overdue": cycle_overdue}


@app.get("/usage")
def usage(request: Request):
    with db.connect() as conn:
        spend = db.monthly_spend(conn)
        rows = conn.execute("""
            SELECT substr(ts,1,7) month, role, provider, model,
                   SUM(input_tokens) in_tok, SUM(output_tokens) out_tok,
                   ROUND(SUM(cost_usd), 4) cost
            FROM usage_log GROUP BY month, role, provider, model
            ORDER BY month DESC, cost DESC""").fetchall()
        errors = conn.execute(
            "SELECT * FROM jobs WHERE status='error' ORDER BY created_at DESC LIMIT 20").fetchall()
    return templates.TemplateResponse(request, "usage.html", {
        "rows": rows, "spend": spend, "budget": settings.monthly_budget_usd, "errors": errors,
        "run_status": pipeline.RUN_STATUS, "pipeline_hour": settings.pipeline_hour,
        **_cycle_status(),
    })


def _shadow_stats() -> dict:
    """Is shadow mode working, and how often does it agree with DeepSeek?
    Coverage: of jobs added since shadow mode first scored anything, how many
    DeepSeek-evaluated ones also got a Jev score (older jobs never will unless
    backfilled). Agreement: over every job with both scores, on decisions
    (skip/review/gloss) rather than raw numbers."""
    since = db.get_meta("shadow_since")
    with db.connect() as conn:
        recent = conn.execute("SELECT * FROM jobs WHERE created_at >= ? AND score IS NOT NULL",
                              (since,)).fetchall() if since else []
        both = conn.execute("SELECT * FROM jobs WHERE score IS NOT NULL "
                            "AND gate_score IS NOT NULL").fetchall()
        unscored = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE status IN ('pending_user_review','shortlisted') "
            "AND gate_score IS NULL").fetchone()["n"]
        last = conn.execute("SELECT MAX(ts) t FROM usage_log WHERE role='gate'").fetchone()["t"]
    cmp = [c for c in (_gate_compare(j) for j in both) if c and c["ds"]]
    moves: dict = {}
    for c in cmp:
        if not c["agree"]:
            k = f"DeepSeek {c['ds']} → Jev {c['jev']}"
            moves[k] = moves.get(k, 0) + 1
    from . import gate
    # jobs scored before their applicant's latest profile version
    with db.connect() as conn:
        profile_changed = conn.execute(
            "SELECT COUNT(*) n FROM jobs j JOIN (SELECT applicant_id, MAX(created_at) t "
            "FROM profiles GROUP BY applicant_id) p ON p.applicant_id=j.applicant_id "
            "JOIN gate_evals g ON g.job_id=j.id AND g.req_hash=j.gate_req "
            "WHERE j.gate_score IS NOT NULL AND g.created_at < p.t").fetchone()["n"]
    return {"mode": settings.gate_mode, "model": gate.active_model()["version"],
            "profile_changed": profile_changed,
            "since": since, "evaluated_new": len(recent),
            "scored_new": sum(1 for j in recent if j["gate_score"] is not None),
            "agree": sum(c["agree"] for c in cmp), "compared": len(cmp), "moves": moves,
            "unscored_active": unscored, "last_call": last,
            "failures": list(reversed(pipeline.GATE_FAILURES)),
            "backfill": dict(pipeline.BACKFILL_STATUS)}


@app.get("/status")
def status(request: Request):
    from .sources import RECENT_FAILURES
    with db.connect() as conn:
        errors = conn.execute(
            "SELECT * FROM jobs WHERE status='error' ORDER BY created_at DESC LIMIT 20").fetchall()
    return templates.TemplateResponse(request, "status.html", {
        "run_status": pipeline.RUN_STATUS, "pipeline_hour": settings.pipeline_hour,
        "source_failures": list(reversed(RECENT_FAILURES)), "errors": errors,
        "shadow": _shadow_stats(),
        **_cycle_status(),
    })


@app.post("/run")
def run_now(background: BackgroundTasks, redirect: str = Form("/usage")):
    pipeline.clear_stop()  # a fresh, explicit run overrides a prior stop
    background.add_task(pipeline.run_full_cycle)
    return RedirectResponse(redirect, status_code=303)


@app.post("/pipeline/stop")
def pipeline_stop(redirect: str = Form("/usage")):
    """Ask the running ingest/eval loop to stop at the next job boundary. The
    current job finishes; nothing half-written. Jobs not yet evaluated stay
    pending_eval and resume on the next run (or a manual re-trigger)."""
    pipeline.request_stop()
    return RedirectResponse(redirect, status_code=303)


@app.post("/probe")
def probe_now():
    """Manually run the URL liveness probe (normally the daily cron). Synchronous
    — the acting-on set is small — and returns a summary so a live test shows
    something even under the 2-strike grace: `provisionally_missing` is dead
    URLs seen once (not yet flagged), `newly_closed` is those that crossed it."""
    newly = pipeline.probe_liveness()
    with db.connect() as conn:
        flagged = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE closed_at IS NOT NULL").fetchone()["n"]
        pending = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE closed_at IS NULL AND missed_count>0"
        ).fetchone()["n"]
    return {"newly_closed": newly, "total_flagged_closed": flagged,
            "provisionally_missing": pending}


@app.get("/run/status")
def run_status():
    """Progress of the current/last pipeline run, polled by the Jobs page.
    `pending` counts stragglers too (e.g. evals resumed after a restart)."""
    with db.connect() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) n FROM jobs WHERE status IN ('pending_eval','pending_extract')"
        ).fetchone()["n"]
    return {**pipeline.RUN_STATUS, "pending": pending}
