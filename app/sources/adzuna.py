"""Adzuna adapter. Free tier: https://developer.adzuna.com/

Profile config example:
  "sources": {"adzuna": {"queries": [
      {"what": "data entry", "where": "Seattle, WA", "max_days_old": 7},
      {"what": "administrative assistant", "where": "Seattle, WA"}
  ]}}

Note: Adzuna descriptions are truncated; the eval prompt is told this and the
original URL is always kept for the human review step.
"""
import hashlib
import time

import httpx

from ..config import settings

BASE = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"

# Free-tier Adzuna throttles well under one request/second. A profile with a
# dozen+ "what" queries (and multiple profiles per run) otherwise fires them
# in a tight loop, tripping 429s and provoking 503s; pace requests and retry
# a transient 429/503 once before giving up on that query.
_REQUEST_GAP_S = 1.2
_RETRY_STATUSES = (429, 503)


def _get(url, params):
    for attempt in range(2):
        r = httpx.get(url, params=params, timeout=30)
        if r.status_code in _RETRY_STATUSES and attempt == 0:
            time.sleep(3)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def fetch(cfg: dict) -> list[dict]:
    if not (settings.adzuna_app_id and settings.adzuna_app_key):
        print("[adzuna] skipped: ADZUNA_APP_ID/KEY not set")
        return []
    out = []
    for i, q in enumerate(cfg.get("queries", [])):
        if i:
            time.sleep(_REQUEST_GAP_S)
        params = {
            "app_id": settings.adzuna_app_id,
            "app_key": settings.adzuna_app_key,
            "results_per_page": q.get("results_per_page", 50),
            "what": q.get("what", ""),
            "content-type": "application/json",
        }
        # Queries without an explicit "where" search the shared household
        # location with a radius covering the whole metro (SEARCH_WHERE /
        # SEARCH_DISTANCE_KM). Explicit "where" (e.g. "United States" for
        # remote sweeps) still wins.
        if q.get("where"):
            params["where"] = q["where"]
            if q.get("distance_km"):
                params["distance"] = q["distance_km"]
        else:
            params["where"] = settings.search_where
            params["distance"] = q.get("distance_km", settings.search_distance_km)
        if q.get("max_days_old"):
            params["max_days_old"] = q["max_days_old"]
        r = _get(BASE.format(country=settings.adzuna_country), params)
        for item in r.json().get("results", []):
            url = item.get("redirect_url") or ""
            salary = None
            if item.get("salary_min") or item.get("salary_max"):
                salary = f"{item.get('salary_min') or '?'}–{item.get('salary_max') or '?'} (adzuna est.)"
            out.append({
                "dedupe_key": hashlib.sha1((item.get("id") or url).encode()).hexdigest(),
                "url": url,
                "title": item.get("title"),
                "company": (item.get("company") or {}).get("display_name"),
                "location": (item.get("location") or {}).get("display_name"),
                "salary": salary,
                "description": (item.get("description") or "") + "\n\n[Note: Adzuna truncates descriptions; see URL for full posting.]",
            })
    return out
