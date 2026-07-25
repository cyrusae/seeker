"""Workday hosted-careers adapter (unofficial but stable CXS JSON endpoints).

Profile config example:
  "sources": {"workday": {"tenants": [{
      "tenant": "nordstrom", "shard": "wd501", "site": "nordstrom_careers",
      "company": "Nordstrom",
      "search": ["data engineer", "administrative assistant"],
      "location_include": ["seattle", "remote"]
  }],
  "title_include": ["…"]}}

`search` is full-text (a "data entry" search also returns roles that merely
mention data entry), so the optional top-level `title_include` filter is worth
setting when the eval queue matters — it runs before the per-job detail fetch.

Finding tenant/shard/site: the company careers page links to
https://{tenant}.{shard}.myworkdayjobs.com/{site}(/...). A wrong site name
answers 422, a wrong tenant 404/no-DNS.

Each `search` term is a server-side searchText query, paginated. The listing
carries title/location only, so descriptions cost one extra request per NEW
job — the `seen` set skips jobs ingested on earlier runs.
"""
import hashlib
import time

import httpx

from ..textclean import clean as _detag

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; seeker-jobsearch)",
           "Accept": "application/json", "Content-Type": "application/json"}
PAGE = 20          # max the endpoint allows per request (400 above this)
MAX_PER_SEARCH = 200  # sanity cap per search term per tenant


def _base(t: dict) -> str:
    return (f"https://{t['tenant']}.{t['shard']}.myworkdayjobs.com"
            f"/wday/cxs/{t['tenant']}/{t['site']}")


def _postings(t: dict, search: str):
    """Yield jobPostings dicts for one search term, paginated."""
    offset = 0
    while offset < MAX_PER_SEARCH:
        r = httpx.post(f"{_base(t)}/jobs", headers=HEADERS, timeout=30,
                       json={"appliedFacets": {}, "limit": PAGE,
                             "offset": offset, "searchText": search})
        r.raise_for_status()
        batch = r.json().get("jobPostings") or []
        yield from batch
        if len(batch) < PAGE:
            return
        offset += PAGE


def _detail(t: dict, external_path: str) -> dict:
    r = httpx.get(_base(t) + external_path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("jobPostingInfo") or {}


def fetch(cfg: dict, seen: frozenset = frozenset()) -> list[dict]:
    include = [s.lower() for s in cfg.get("title_include", [])]
    out, done = [], set()
    for t in cfg.get("tenants", []):
        loc_include = [s.lower() for s in t.get("location_include", [])]
        for search in t.get("search") or [""]:
            for item in _postings(t, search):
                path = item.get("externalPath") or ""
                # the requisition id is the stable identity; externalPath
                # embeds the (mutable) title
                req = (item.get("bulletFields") or [path])[0]
                key = hashlib.sha1(
                    f"workday:{t['tenant']}:{req}".encode()).hexdigest()
                if key in done or key in seen or not path:
                    continue
                title = (item.get("title") or "").lower()
                if include and not any(s in title for s in include):
                    continue
                loc = (item.get("locationsText") or "").lower()
                if loc_include and not any(s in loc for s in loc_include):
                    continue
                done.add(key)
                time.sleep(0.5)  # politeness: their site, our schedule
                info = _detail(t, path)
                out.append({
                    "dedupe_key": key,
                    "url": info.get("externalUrl")
                           or f"https://{t['tenant']}.{t['shard']}.myworkdayjobs.com/{t['site']}{path}",
                    "title": item.get("title"),
                    "company": t.get("company", t["tenant"]),
                    "location": info.get("location") or item.get("locationsText"),
                    "salary": None,  # WA law puts ranges in the description
                    "description": _detag(info.get("jobDescription", ""))[:15000],
                })
    return out
