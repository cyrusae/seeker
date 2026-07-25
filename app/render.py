"""Markdown → HTML rendering for deliverables.

HTML is print-styled so browser print-to-PDF produces a clean one-pager.
(pandoc/typst can slot in later if you want native PDF.)
"""
import os
import re

import markdown as md_lib

from .config import settings

PRINT_CSS = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 46rem;
       margin: 2rem auto; line-height: 1.4; color: #1a1a1a; padding: 0 1rem; }
h1 { font-size: 1.5rem; margin-bottom: 0.1rem; }
h2 { font-size: 1.05rem; border-bottom: 1px solid #999; margin-top: 1.1rem; }
ul { margin-top: 0.2rem; } li { margin-bottom: 0.15rem; }
@media print { body { margin: 0 auto; } }
"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "x").lower()).strip("-")[:40]


def save(job, kind: str, md_text: str) -> tuple[str, str]:
    docs_dir = os.path.join(settings.data_dir, "docs", job["applicant_id"])
    os.makedirs(docs_dir, exist_ok=True)
    base = f"{_slug(job['company'])}-{_slug(job['title'])}-{job['id']}-{kind}"
    path_md = os.path.join(docs_dir, base + ".md")
    path_html = os.path.join(docs_dir, base + ".html")
    with open(path_md, "w") as f:
        f.write(md_text)
    body = md_lib.markdown(md_text, extensions=["extra"])
    with open(path_html, "w") as f:
        f.write(f"<!doctype html><meta charset='utf-8'><title>{base}</title>"
                f"<style>{PRINT_CSS}</style>\n{body}")
    return path_md, path_html
