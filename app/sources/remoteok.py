"""RemoteOK adapter (no auth; one global feed, filtered client-side).

Profile config:
  "sources": {"remoteok": {"search": ["data entry", "customer support"]}}

Terms: RemoteOK asks that reusers link back to the listing URL — we always
store and surface the original URL, which satisfies that for personal use.
"""
import hashlib
import re

import httpx

URL = "https://remoteok.com/api"
TAG = re.compile(r"<[^>]+>")


def fetch(cfg: dict) -> list[dict]:
    terms = [s.lower() for s in cfg.get("search", [])]
    if not terms:
        return []  # the global feed unfiltered would flood the queue
    r = httpx.get(URL, headers={"User-Agent": "seeker-personal-job-search"}, timeout=30)
    r.raise_for_status()
    out = []
    for item in r.json():
        if not isinstance(item, dict) or "id" not in item:
            continue  # first element is a legal notice
        hay = " ".join([
            item.get("position") or "", " ".join(item.get("tags") or []),
            (item.get("description") or "")[:2000],
        ]).lower()
        if not any(t in hay for t in terms):
            continue
        salary = None
        if item.get("salary_min") or item.get("salary_max"):
            salary = f"{item.get('salary_min') or '?'}–{item.get('salary_max') or '?'}"
        desc = TAG.sub(" ", item.get("description") or "")
        out.append({
            "dedupe_key": hashlib.sha1(f"remoteok:{item['id']}".encode()).hexdigest(),
            "url": item.get("url"),
            "title": item.get("position"),
            "company": item.get("company"),
            "location": item.get("location") or "Remote",
            "salary": salary,
            "description": re.sub(r"\s+", " ", desc).strip()[:15000],
        })
    return out
