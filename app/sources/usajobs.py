"""USAJobs adapter — official US federal jobs API.

Free key: https://developer.usajobs.gov/apirequest/ (requires the registered
email sent as User-Agent). Set USAJOBS_API_KEY and USAJOBS_EMAIL in .env.

Profile config:
  "sources": {"usajobs": {"queries": [
      {"keyword": "data entry", "location": "Seattle, Washington"},
      {"keyword": "office automation clerk", "remote": true}
  ]}}
"""
import hashlib

import httpx

from ..config import settings

URL = "https://data.usajobs.gov/api/search"


def fetch(cfg: dict) -> list[dict]:
    if not (settings.usajobs_api_key and settings.usajobs_email):
        print("[usajobs] skipped: USAJOBS_API_KEY/EMAIL not set")
        return []
    headers = {
        "Authorization-Key": settings.usajobs_api_key,
        "User-Agent": settings.usajobs_email,
    }
    out = []
    for q in cfg.get("queries", []):
        params = {"ResultsPerPage": q.get("results_per_page", 50)}
        if q.get("keyword"):
            params["Keyword"] = q["keyword"]
        if q.get("location"):
            params["LocationName"] = q["location"]
        if q.get("remote"):
            params["RemoteIndicator"] = "True"
        r = httpx.get(URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        items = r.json().get("SearchResult", {}).get("SearchResultItems", [])
        for item in items:
            d = item.get("MatchedObjectDescriptor", {})
            pay = (d.get("PositionRemuneration") or [{}])[0]
            salary = None
            if pay.get("MinimumRange"):
                salary = (f"{pay.get('MinimumRange')}–{pay.get('MaximumRange')} "
                          f"per {(pay.get('RateIntervalCode') or '').lower()}")
            details = (d.get("UserArea") or {}).get("Details") or {}
            desc = "\n\n".join(filter(None, [
                details.get("JobSummary"),
                "QUALIFICATIONS: " + d.get("QualificationSummary", "")
                if d.get("QualificationSummary") else None,
            ]))
            out.append({
                "dedupe_key": hashlib.sha1(
                    f"usajobs:{item.get('MatchedObjectId')}".encode()).hexdigest(),
                "url": d.get("PositionURI"),
                "title": d.get("PositionTitle"),
                "company": d.get("OrganizationName"),
                "location": d.get("PositionLocationDisplay"),
                "salary": salary,
                "description": desc[:15000],
            })
    return out
