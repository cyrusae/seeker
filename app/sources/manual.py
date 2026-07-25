"""Manual submission: paste text (always works) or a URL (best-effort fetch).

Pasted text / fetched pages go through the `extract` role to structure them.
LinkedIn URLs usually won't render without login — paste the text instead.
"""
import hashlib

import httpx

from .. import llm
from ..textclean import clean

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
BROWSER_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
EXTRACT_SYSTEM = """You extract structured job posting data from raw text.
Output JSON with keys: title, company, location, salary (string or null),
description (the full posting text, cleaned up but not summarized).
If the text does not appear to be a job posting, set "title" to null."""


def fetch_url(url: str) -> str:
    r = httpx.get(url, headers=BROWSER_HEADERS, timeout=30, follow_redirects=True)
    r.raise_for_status()
    text = clean(r.text)
    if len(text) < 200:
        raise ValueError("Fetched page had almost no text (login wall or JS-rendered). Paste the posting text instead.")
    return text[:30000]


def extract(raw_text: str, url: str | None) -> dict:
    data = llm.complete_json("extract", EXTRACT_SYSTEM, raw_text[:30000], max_tokens=4000)
    if not data.get("title"):
        raise ValueError("Extractor did not find a job posting in that text.")
    key_material = url or (data["title"] + (data.get("company") or "") + raw_text[:500])
    return {
        "dedupe_key": hashlib.sha1(key_material.encode()).hexdigest(),
        "url": url,
        "title": data.get("title"),
        "company": data.get("company"),
        "location": data.get("location"),
        "salary": data.get("salary"),
        "description": clean(data.get("description")) or raw_text[:15000],
    }
