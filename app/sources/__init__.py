"""Source adapters. Each adapter yields normalized job dicts:

    {dedupe_key, url, title, company, location, salary, description}

Adapters are pure ingestion: they never score or filter beyond their query.
Which adapters run for an applicant is driven by the profile's `sources` block.
"""
import inspect
from collections import deque
from datetime import datetime, timezone

from . import adzuna, ashby, govjobs, greenhouse, icims, jibe, lever, remoteok, usajobs, workday

# In-memory ring buffer of recent adapter failures (rate limits, timeouts,
# broken feeds). These are swallowed per-adapter so one bad source doesn't
# kill the run, which also means they'd otherwise only ever be a stdout print
# lost the moment nobody's watching the terminal. Surfaced on the Status tab.
# Same durability tier as pipeline.RUN_STATUS: in-memory, reset on restart,
# good enough for a single-process deployment.
RECENT_FAILURES: deque = deque(maxlen=20)

REGISTRY = {
    "adzuna": adzuna.fetch,
    "ashby": ashby.fetch,
    "govjobs": govjobs.fetch,
    "greenhouse": greenhouse.fetch,
    "icims": icims.fetch,
    "jibe": jibe.fetch,
    "lever": lever.fetch,
    "remoteok": remoteok.fetch,
    "usajobs": usajobs.fetch,
    "workday": workday.fetch,
}


# Adapters that return the COMPLETE current set of open postings per board/org/
# company/feed each run and filter client-side. For these, a stored job whose
# key is absent from a successful fetch is genuinely gone (closed/filled) — the
# basis for closure detection. Excluded: keyword/search APIs (adzuna, usajobs,
# workday) where absence just means "aged out of the recency window / dropped
# in ranking", and per-job scrapers (icims, govjobs) that use `seen` to SKIP
# re-listing known jobs, so absence there means "we didn't look".
FULL_ENUMERATORS = frozenset({"greenhouse", "ashby", "lever", "jibe", "remoteok"})


class Scan(list):
    """The jobs found this run, plus closure bookkeeping. Subclasses list so
    existing `for job in fetch_for_profile(...)` iteration is unchanged.

    enumerated_keys: every dedupe key seen from a full-enumerator source that
      completed without error — the "still open" set.
    swept_sources: which full-enumerator adapters actually ran this cycle, so
      the closure sweep only judges jobs whose board we truly re-listed.
    """
    enumerated_keys: frozenset
    swept_sources: frozenset


def fetch_for_profile(profile: dict, seen: frozenset = frozenset()) -> "Scan":
    """`seen` = dedupe keys already stored for this applicant. Adapters that
    declare a `seen` parameter (scrapers paying one request per description)
    use it to skip known jobs; API adapters ignore it — dedupe-on-insert is
    enough when listing is one request total.

    Each returned job is tagged with `_source` (its adapter name), persisted as
    jobs.poll_source so the closure sweep knows which board a job came from."""
    scan = Scan()
    enumerated_keys: set = set()
    swept_sources: set = set()
    for name, cfg in (profile.get("sources") or {}).items():
        fetcher = REGISTRY.get(name)
        if fetcher is None:
            continue
        try:
            if "seen" in inspect.signature(fetcher).parameters:
                found = fetcher(cfg, seen=seen)
            else:
                found = fetcher(cfg)
        except Exception as e:  # a broken source shouldn't kill the run
            print(f"[sources] {name} failed for {profile.get('applicant_id')}: {e}")
            RECENT_FAILURES.append({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": name, "applicant_id": profile.get("applicant_id"),
                "error": str(e)[:300],
            })
            continue
        for j in found:
            j["_source"] = name
        scan.extend(found)
        if name in FULL_ENUMERATORS:
            swept_sources.add(name)
            enumerated_keys.update(j["dedupe_key"] for j in found)
    scan.enumerated_keys = frozenset(enumerated_keys)
    scan.swept_sources = frozenset(swept_sources)
    return scan
