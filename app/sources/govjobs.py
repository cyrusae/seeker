"""governmentjobs.com (NEOGOV) adapter — public-sector boards, HTML scraping.

Profile config example:
  "sources": {"govjobs": {"agencies": [
      {"agency": "seattle", "company": "City of Seattle"},
      {"agency": "kingcounty", "company": "King County"}
    ],
    "title_include": ["administrative", "data entry", "clerk"],
    "location_include": ["seattle", "king county"]}}

`location_include` matters for statewide boards (agency "washington" is all of
WA State government) — it's checked before the per-job detail fetch.

The agency slug is the path segment in governmentjobs.com/careers/{agency}.
The server ignores keyword params on this endpoint, so we paginate the whole
board (10 jobs/page) and filter titles locally — that costs one request per
10 postings, which is nothing for agency boards. Full descriptions cost one
request per NEW job that passes the title filter; the `seen` set skips jobs
ingested on earlier runs. Titles/locations/salaries come from the list page.
"""
import hashlib
import re
import time

import httpx

from ..textclean import clean

# The list endpoint only returns the jobs fragment when called as XHR;
# detail pages are the opposite — they 404 with the XHR header set.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; seeker-jobsearch)"}
XHR_HEADERS = {**HEADERS, "X-Requested-With": "XMLHttpRequest"}
LIST_URL = "https://www.governmentjobs.com/careers/home/index"
ITEM = re.compile(r'class="list-item" data-job-id="(\d+)"(.*?)(?=class="list-item" data-job-id="|$)',
                  re.DOTALL)
LINK = re.compile(r'<a[^>]*class="item-details-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
META_LI = re.compile(r"<li[^>]*>(.*?)</li>", re.DOTALL)
SNIPPET = re.compile(r'class="list-entry">\s*(.*?)</div>', re.DOTALL)
MAX_PAGES = 50  # 500 postings per agency, plenty for municipal boards


def _detag(html: str) -> str:
    """One-line fields (titles, meta list items)."""
    return clean(html, oneline=True)


def _description(url: str) -> str:
    """Detail pages are server-rendered; the posting body is the active tab."""
    r = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    r.raise_for_status()
    html = r.text
    start = html.find('class="tab-pane active fr-view"')
    if start == -1:
        return ""
    start = html.find(">", start) + 1
    end = html.find('class="job-details-questions', start)
    return clean(html[start : end if end != -1 else start + 60000])[:15000]


def _pages(agency: str):
    for page in range(1, MAX_PAGES + 1):
        r = httpx.get(LIST_URL, params={"agency": agency, "page": page},
                      headers=XHR_HEADERS, timeout=30)
        r.raise_for_status()
        items = list(ITEM.finditer(r.text))
        if not items:
            return
        yield items
        if len(items) < 10:  # short page = last page
            return
        time.sleep(0.5)  # politeness: their site, our schedule


def fetch(cfg: dict, seen: frozenset = frozenset()) -> list[dict]:
    include = [s.lower() for s in cfg.get("title_include", [])]
    loc_include = [s.lower() for s in cfg.get("location_include", [])]
    out, done = [], set()
    for a in cfg.get("agencies", []):
        agency = a["agency"]
        for items in _pages(agency):
            for m in items:
                job_id, body = m.group(1), m.group(2)
                key = hashlib.sha1(f"govjobs:{agency}:{job_id}".encode()).hexdigest()
                if key in done or key in seen:
                    continue
                link = LINK.search(body)
                if not link:
                    continue
                title = _detag(link.group(2))
                if include and not any(s in title.lower() for s in include):
                    continue
                url = "https://www.governmentjobs.com" + link.group(1)
                lis = [_detag(x) for x in META_LI.findall(body)]
                location = lis[0] if lis else None
                if loc_include and not any(s in (location or "").lower()
                                           for s in loc_include):
                    continue
                done.add(key)
                # second <li> reads "Classified..., Full-Time - $60.55 - $70.40 Hourly"
                salary = None
                if len(lis) > 1 and "$" in lis[1]:
                    salary = lis[1][lis[1].find("$"):]
                snippet = SNIPPET.search(body)
                time.sleep(0.5)
                try:
                    desc = _description(url)
                except Exception:
                    desc = ""
                out.append({
                    "dedupe_key": key,
                    "url": url,
                    "title": title,
                    "company": a.get("company", agency),
                    "location": location,
                    "salary": salary,
                    "description": desc or (_detag(snippet.group(1)) if snippet else ""),
                })
    return out
