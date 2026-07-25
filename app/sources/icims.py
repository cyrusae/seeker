"""iCIMS hosted-portal adapter (HTML scraping — iCIMS has no public JSON API).

Profile config example:
  "sources": {"icims": {"portals": [{
      "host": "careers-fhcrc.icims.com",
      "company": "Fred Hutchinson Cancer Center",
      "keywords": ["administrative", "data entry", "coordinator"]
  }]}}

Each keyword runs a server-side search; an empty/missing keywords list pulls
the whole board. Search pages only carry title/type/location, so the full
description needs one extra request per job — the `seen` set (dedupe keys
already in the DB) lets us skip that fetch for jobs ingested on a prior run.
"""
import hashlib
import re
import time

import httpx

from ..textclean import clean

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; seeker-jobsearch)"}
CARD = re.compile(r'class="iCIMS_JobCardItem"(.*?)</li>', re.DOTALL)
LINK = re.compile(r'href="(https://[^"]+/jobs/(\d+)/[^"]+)"')
FIELD = re.compile(
    r'<dt class="iCIMS_JobHeaderField">(.*?)</dt>\s*'
    r'<dd class="iCIMS_JobHeaderData">(.*?)</dd>', re.DOTALL)
TITLE = re.compile(r"<h3\s*>(.*?)</h3>", re.DOTALL)
MAX_PAGES = 5  # per keyword; a page holds 100 jobs


def _detag(html: str) -> str:
    """One-line fields (titles, header data)."""
    return clean(html, oneline=True)


def _dedupe_key(host: str, job_id: str) -> str:
    return hashlib.sha1(f"icims:{host}:{job_id}".encode()).hexdigest()


def _description(url: str) -> str:
    """One extra request per new job: the search page has no description."""
    # in_iframe=1 gets the server-rendered content; without it the page is a
    # JS shell with no job text. Stored/user-facing URLs stay clean.
    r = httpx.get(url + "?in_iframe=1", headers=HEADERS, timeout=30,
                  follow_redirects=True)
    r.raise_for_status()
    html = r.text
    start = html.find("iCIMS_JobContent")
    if start == -1:
        return ""
    start = html.find(">", start) + 1
    end = html.find("iCIMS_JobOptions", start)
    return clean(html[start : end if end != -1 else None])[:15000]


def _search_pages(host: str, keyword: str):
    """Yield search-result HTML pages until results run out."""
    for page in range(MAX_PAGES):
        params = {"ss": "1", "in_iframe": "1", "pr": str(page)}
        if keyword:
            params["searchKeyword"] = keyword
        r = httpx.get(f"https://{host}/jobs/search", params=params,
                      headers=HEADERS, timeout=30)
        r.raise_for_status()
        if "iCIMS_JobCardItem" not in r.text:
            return
        yield r.text


def fetch(cfg: dict, seen: frozenset = frozenset()) -> list[dict]:
    out, done = [], set()
    for portal in cfg.get("portals", []):
        host = portal["host"]
        for keyword in portal.get("keywords") or [""]:
            for page_html in _search_pages(host, keyword):
                for card in CARD.finditer(page_html):
                    link = LINK.search(card.group(1))
                    if not link:
                        continue
                    url, job_id = link.group(1).split("?")[0], link.group(2)
                    key = _dedupe_key(host, job_id)
                    # keywords overlap and reruns re-list everything; the
                    # detail fetch is the expensive part, so gate it hard
                    if key in done or key in seen:
                        continue
                    done.add(key)
                    fields = {_detag(k): _detag(v)
                              for k, v in FIELD.findall(card.group(1))}
                    title_m = TITLE.search(card.group(1))
                    time.sleep(0.5)  # politeness: their site, our schedule
                    out.append({
                        "dedupe_key": key,
                        "url": url,
                        "title": _detag(title_m.group(1)) if title_m else None,
                        "company": portal.get("company", host),
                        "location": fields.get("Location"),
                        "salary": None,  # WA law puts ranges in the description
                        "description": _description(url),
                    })
    return out
