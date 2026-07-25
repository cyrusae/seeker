"""Jibe (iCIMS Talent Cloud) careers-site adapter — clean JSON, no auth.

Sites "powered by Jibe" (careers.costco.com, www.rei.jobs, ...) expose
GET https://{host}/api/jobs with location/stretch filtering and full
descriptions inline, so no per-job detail fetches are needed.

Profile config example:
  "jibe": {"sites": [{"host": "careers.costco.com", "company": "Costco Wholesale"},
                     {"host": "www.rei.jobs", "company": "REI Co-op"}],
           "location": "Seattle, WA", "stretch_miles": 25,
           "title_include": ["data engineer", "administrative"]}

The keyword param is ignored server-side (Costco returns all 26k jobs
regardless), so `title_include` does the narrowing client-side.
"""
import hashlib

import httpx

from ..textclean import clean as _detag

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; seeker-jobsearch)"}
LIMIT = 50
MAX_PAGES = 40  # 2000 postings per site within the radius — plenty


def fetch(cfg: dict, seen: frozenset = frozenset()) -> list[dict]:
    include = [s.lower() for s in cfg.get("title_include", [])]
    out, done = [], set()
    for site in cfg.get("sites", []):
        host = site["host"]
        for page in range(1, MAX_PAGES + 1):
            params = {"page": page, "limit": LIMIT}
            if cfg.get("location"):
                params.update({"location": cfg["location"],
                               "stretch": cfg.get("stretch_miles", 25),
                               "stretchUnit": "MILES"})
            r = httpx.get(f"https://{host}/api/jobs", params=params,
                          headers=HEADERS, timeout=30)
            r.raise_for_status()
            jobs = r.json().get("jobs") or []
            if not jobs:
                break
            for item in jobs:
                d = item.get("data") or {}
                title = d.get("title") or ""
                if include and not any(s in title.lower() for s in include):
                    continue
                key = hashlib.sha1(
                    f"jibe:{host}:{d.get('req_id') or d.get('slug')}".encode()).hexdigest()
                if key in done or key in seen:
                    continue
                done.add(key)
                out.append({
                    "dedupe_key": key,
                    "url": d.get("apply_url") or f"https://{host}/jobs/{d.get('slug')}",
                    "title": title,
                    "company": site.get("company", host),
                    "location": d.get("full_location") or d.get("short_location"),
                    "salary": None,
                    "description": _detag(d.get("description", ""))[:15000],
                })
            if len(jobs) < LIMIT:
                break
    return out
