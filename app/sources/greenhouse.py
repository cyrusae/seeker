"""Greenhouse public board adapter (no auth).

Profile config example:
  "sources": {"greenhouse": {"boards": ["anthropic", "stripe"],
                             "title_include": ["engineer", "developer"]}}

`title_include` and `location_include` are optional cheap pre-filters
(case-insensitive substrings) so a big board doesn't flood the eval queue —
e.g. location_include: ["seattle", "bellevue", "remote", "united states"].
"""
import hashlib
import httpx

from ..textclean import clean as _detag

BASE = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"


def fetch(cfg: dict) -> list[dict]:
    out = []
    include = [s.lower() for s in cfg.get("title_include", [])]
    loc_include = [s.lower() for s in cfg.get("location_include", [])]
    for board in cfg.get("boards", []):
        r = httpx.get(BASE.format(board=board), timeout=30)
        if r.status_code == 404:
            print(f"[greenhouse] board '{board}' not found")
            continue
        r.raise_for_status()
        for item in r.json().get("jobs", []):
            title = item.get("title") or ""
            if include and not any(s in title.lower() for s in include):
                continue
            loc = ((item.get("location") or {}).get("name") or "").lower()
            if loc_include and not any(s in loc for s in loc_include):
                continue
            url = item.get("absolute_url") or ""
            out.append({
                "dedupe_key": hashlib.sha1(f"gh:{board}:{item.get('id')}".encode()).hexdigest(),
                "url": url,
                "title": title,
                "company": board,
                "location": (item.get("location") or {}).get("name"),
                "salary": None,
                "description": _detag(item.get("content", ""))[:15000],
            })
    return out
