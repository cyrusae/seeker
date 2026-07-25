"""Ashby public job-board adapter (no auth).

Profile config:
  "sources": {"ashby": {"orgs": ["org-slug"], "title_include": ["…"]}}
"""
import hashlib

import httpx

BASE = "https://api.ashbyhq.com/posting-api/job-board/{org}"


def fetch(cfg: dict) -> list[dict]:
    out = []
    include = [s.lower() for s in cfg.get("title_include", [])]
    loc_include = [s.lower() for s in cfg.get("location_include", [])]
    for org in cfg.get("orgs", []):
        r = httpx.get(BASE.format(org=org), timeout=30)
        if r.status_code == 404:
            print(f"[ashby] org '{org}' not found")
            continue
        r.raise_for_status()
        for item in r.json().get("jobs", []):
            if not item.get("isListed", True):
                continue
            title = item.get("title") or ""
            if include and not any(s in title.lower() for s in include):
                continue
            loc = ("remote — " if item.get("isRemote") else "") + (item.get("location") or "").lower()
            if loc_include and not any(s in loc for s in loc_include):
                continue
            out.append({
                "dedupe_key": hashlib.sha1(f"ashby:{org}:{item.get('id')}".encode()).hexdigest(),
                "url": item.get("jobUrl") or item.get("applyUrl"),
                "title": title,
                "company": org,
                "location": ("Remote — " if item.get("isRemote") else "") + (item.get("location") or ""),
                "salary": None,
                "description": (item.get("descriptionPlain") or "")[:15000],
            })
    return out
