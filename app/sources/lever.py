"""Lever public postings adapter (no auth).

Profile config:
  "sources": {"lever": {"companies": ["company-slug"], "title_include": ["…"]}}
"""
import hashlib

import httpx

BASE = "https://api.lever.co/v0/postings/{company}?mode=json"


def fetch(cfg: dict) -> list[dict]:
    out = []
    include = [s.lower() for s in cfg.get("title_include", [])]
    loc_include = [s.lower() for s in cfg.get("location_include", [])]
    for company in cfg.get("companies", []):
        r = httpx.get(BASE.format(company=company), timeout=30)
        if r.status_code == 404:
            print(f"[lever] board '{company}' not found")
            continue
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):  # {"ok": false, ...}
            print(f"[lever] board '{company}': {data.get('error')}")
            continue
        for item in data:
            title = item.get("text") or ""
            if include and not any(s in title.lower() for s in include):
                continue
            cats = item.get("categories") or {}
            loc = (cats.get("location") or "").lower()
            if loc_include and not any(s in loc for s in loc_include):
                continue
            out.append({
                "dedupe_key": hashlib.sha1(f"lever:{company}:{item.get('id')}".encode()).hexdigest(),
                "url": item.get("hostedUrl"),
                "title": title,
                "company": company,
                "location": cats.get("location"),
                "salary": None,
                "description": (item.get("descriptionPlain") or "")[:15000],
            })
    return out
