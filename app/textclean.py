"""Shared posting-text cleanup — one place for the HTML/CSS junk heuristics.

Job descriptions arrive in three flavors of dirty:
  1. Entity-encoded HTML (Greenhouse ships `&lt;div&gt;` — tag strippers see no
     tags at all, so the escaped markup lands in the DB as visible text).
  2. Real HTML where tag-stripping worked but entities (`&amp;`, `&nbsp;`) and
     Word-export CSS residue (`mso-…`, `font-family: …;`) survived as text.
  3. Plain text that only needs entity + whitespace normalization.

`clean()` handles all three, is idempotent (safe to re-run over stored rows),
and keeps paragraph/bullet structure — nicer to read, and list structure is
signal for the eval models. Junk removal happens BEFORE the 15k truncation in
adapters so markup overhead doesn't eat the real content budget.
"""
import html as _html
import re

_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1\s*>", re.I | re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_LI = re.compile(r"<li\b[^>]*>", re.I)
_BLOCK = re.compile(
    r"</?(?:p|div|ul|ol|li|table|tbody|tr|h[1-6]|section|article|blockquote)\b[^>]*>"
    r"|<br\s*/?>", re.I)
# bounded so an unescaped stray "<" can't swallow half a posting (bound is
# generous: long hrefs + inline styles routinely push tags past 300 chars)
_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,1500}>")
# CSS declarations that leak through as text ("font-family: Calibri;", Word's
# "mso-fareast-…"). Property whitelist keeps prose like "Note: …" safe.
_CSS = re.compile(
    r"\b(?:mso-[\w-]+|font(?:-(?:family|size|weight|style|variant|stretch))?|"
    r"line-height|letter-spacing|text-(?:align|indent|decoration|autospace|transform)|"
    r"margin(?:-(?:top|bottom|left|right))?|padding(?:-(?:top|bottom|left|right))?|"
    r"background(?:-color)?|vertical-align|white-space|word-break|overflow-wrap|"
    r"tab-stops|caret-color|border(?:-[\w-]+)?|box-sizing|orphans|widows|"
    r"text-size-adjust|-webkit-[\w-]+|color)\s*:\s*[^;{}<>\n]{1,120};?", re.I)
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
_HSPACE = re.compile(r"[ \t\r\f\v]+")
_NL_PAD = re.compile(r" ?\n ?")
_BULLET_RUN = re.compile(r"(?:• *){2,}")
_EMPTY_BULLET = re.compile(r"\n• *(?=\n|$)")
_MANY_NL = re.compile(r"\n{3,}")


def clean(raw: str | None, oneline: bool = False) -> str:
    s = raw or ""
    if not s:
        return ""
    # peel entity-encoding layers until stable (Greenhouse double-encodes)
    for _ in range(3):
        u = _html.unescape(s)
        if u == s:
            break
        s = u
    s = _SCRIPT.sub(" ", s)
    s = _COMMENT.sub(" ", s)
    s = _LI.sub("\n• ", s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = _CSS.sub(" ", s)
    s = _TAG.sub(" ", s)  # again: CSS-stripping can shrink oversized tags into range
    s = s.replace("\xa0", " ").translate(_INVISIBLE)
    s = _HSPACE.sub(" ", s)
    s = _NL_PAD.sub("\n", s)
    s = _BULLET_RUN.sub("• ", s)
    s = _EMPTY_BULLET.sub("", s)
    s = _MANY_NL.sub("\n\n", s)
    s = s.strip()
    if oneline:
        s = re.sub(r"\s+", " ", s)
    return s
