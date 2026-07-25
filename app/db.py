"""SQLite storage. One file, WAL mode, no ORM."""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    applicant_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    applicant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    url TEXT,
    title TEXT,
    company TEXT,
    location TEXT,
    salary TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending_eval',
    -- pending_extract | pending_eval | filtered | pending_user_review | skipped
    -- | approved | rejected | shortlisted | draft_created | applied | error
    -- | archived (was shortlisted/drafted, then withdrawn by the user)
    score INTEGER,
    pitch TEXT,
    concerns TEXT,
    scam_risk TEXT,
    eval_json TEXT,
    escalated INTEGER DEFAULT 0,
    feedback TEXT,
    notes TEXT,   -- reviewer note passed to the drafting agent
    similar_to TEXT,  -- id of a near-duplicate listing found at ingest (flag, not drop)
    poll_source TEXT,  -- originating adapter (greenhouse, lever, ...); NULL for manual
    closed_at TEXT,    -- set when the posting vanished from its board (see closure sweep)
    closed_reason TEXT,
    missed_count INTEGER NOT NULL DEFAULT 0,  -- consecutive polls a board-listed job was absent
    closure_dismissed INTEGER NOT NULL DEFAULT 0,  -- human said "still open" — stop auto-flagging
    error TEXT,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    UNIQUE(applicant_id, dedupe_key)
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,           -- resume | cover_letter
    path_md TEXT NOT NULL,
    path_html TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_log (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS comparisons (
    id TEXT PRIMARY KEY,
    job_a TEXT NOT NULL,          -- the pair, always stored a<b so lookup is order-free
    job_b TEXT NOT NULL,
    sig TEXT NOT NULL,            -- hash of both descriptions; mismatch => cache is stale
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_a, job_b)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def get_meta(key: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, value))


def connect() -> sqlite3.Connection:
    os.makedirs(settings.data_dir, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Default busy timeout is 0: any lock contention raises "database is locked"
    # immediately. WAL keeps readers non-blocking, but two writers still collide;
    # wait up to 10s for the lock to clear instead of crashing the worker.
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
        # lightweight migrations for columns added after first release
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        if "notes" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN notes TEXT")
        if "similar_to" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN similar_to TEXT")
        if "poll_source" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN poll_source TEXT")
            conn.execute("ALTER TABLE jobs ADD COLUMN closed_at TEXT")
            conn.execute("ALTER TABLE jobs ADD COLUMN closed_reason TEXT")
            conn.execute("ALTER TABLE jobs ADD COLUMN missed_count INTEGER NOT NULL DEFAULT 0")
        if "closure_dismissed" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN closure_dismissed INTEGER NOT NULL DEFAULT 0")


# --- profiles ---------------------------------------------------------------

def import_profile(profile: dict) -> int:
    """Store a new version of an applicant profile. Returns the version."""
    applicant_id = profile["applicant_id"]
    with connect() as conn:
        row = conn.execute(
            "SELECT MAX(version) v FROM profiles WHERE applicant_id=?", (applicant_id,)
        ).fetchone()
        version = (row["v"] or 0) + 1
        conn.execute(
            "INSERT INTO profiles (id, applicant_id, version, json, created_at) VALUES (?,?,?,?,?)",
            (new_id(), applicant_id, version, json.dumps(profile, sort_keys=True), now()),
        )
    return version


def latest_profile(conn, applicant_id: str) -> dict | None:
    row = conn.execute(
        "SELECT json FROM profiles WHERE applicant_id=? ORDER BY version DESC LIMIT 1",
        (applicant_id,),
    ).fetchone()
    return json.loads(row["json"]) if row else None


def all_applicants(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT applicant_id, MAX(version) FROM profiles GROUP BY applicant_id"
    ).fetchall()
    return [latest_profile(conn, r["applicant_id"]) for r in rows]


# --- jobs -------------------------------------------------------------------

def insert_job(conn, **f) -> str | None:
    """Insert a job; returns id, or None if it's a duplicate for that applicant."""
    jid = new_id()
    try:
        conn.execute(
            """INSERT INTO jobs (id, applicant_id, source, dedupe_key, url, title,
                 company, location, salary, description, status, similar_to,
                 poll_source, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                jid, f["applicant_id"], f["source"], f["dedupe_key"],
                f.get("url"), f.get("title"), f.get("company"), f.get("location"),
                f.get("salary"), f.get("description"),
                f.get("status", "pending_eval"), f.get("similar_to"),
                # _source is the adapter name tagged on by fetch_for_profile
                f.get("_source"), f.get("error"), now(),
            ),
        )
        return jid
    except sqlite3.IntegrityError:
        return None


def monthly_spend(conn) -> float:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) c FROM usage_log WHERE ts LIKE ?",
        (month + "%",),
    ).fetchone()
    return row["c"]


def log_usage(role: str, provider: str, model: str, in_tok: int, out_tok: int, cost: float):
    with connect() as conn:
        conn.execute(
            "INSERT INTO usage_log (id, ts, role, provider, model, input_tokens, output_tokens, cost_usd)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (new_id(), now(), role, provider, model, in_tok, out_tok, cost),
        )
