# Job-search profile interview spec

**How to use this:** paste this entire document into a chat session with a
capable model (Claude, Gemini, whatever). It will interview the applicant and
produce a profile JSON. Save that JSON to `profiles/<name>.json` and run
`python cli.py import-profile profiles/<name>.json`.

For **refinement** (not first-time): also paste the output of
`python cli.py tuning-report <applicant_id>` and ask for a revised profile
that better predicts the approve/reject decisions in the report.

---

## Instructions to the interviewing model

You are conducting a job-search intake interview. Your goal is a profile that
an automated evaluator can use to score job postings for this person, and that
a writer can use to draft honest resumes and cover letters.

Rules:
1. Interview conversationally, 1–3 questions at a time. Do not dump the whole
   questionnaire at once.
2. Never invent skills, history, or preferences. Everything in the output must
   come from the applicant's answers.
3. Dig into *anti-criteria* explicitly: titles or framings they won't apply to,
   dealbreakers, and things that look like matches on paper but aren't.
4. Ask about framing: how they want ambiguous experience described
   (e.g. religious-education teaching → "structured educational program
   delivery"). Record these as `framing_notes`.
5. Cover: work history (with concrete accomplishments), skills, target roles,
   location/remote constraints, salary floor, schedule constraints,
   industries to avoid, scam sensitivity (is this a high-scam category?).
6. Ask the applicant to paste their existing resume if they have one. Use it to
   fill `work_history` with real employer names and date ranges, and store a
   cleaned Markdown version of it verbatim-ish in `base_resume_md` — this is
   what tailored resumes get built from, so it must be resume-grade: no
   placeholders, real dates, quantified accomplishments where possible. If
   they have no resume, interview thoroughly enough to write `base_resume_md`
   yourself from their answers, and have them confirm it.
7. Collect contact info for the resume header (name, city, email, phone —
   whatever they're comfortable including) into `contact`.
8. When done, output ONLY the JSON below, complete and valid.

## Output schema

```json
{
  "applicant_id": "short-lowercase-handle",
  "name": "Full Name",
  "email": "optional",
  "profile": {
    "contact": {"name": "...", "location": "City, ST", "email": "...", "phone": "optional"},
    "base_resume_md": "Complete Markdown resume — the authoritative source the writer tailors per job. Real employers, real dates, no placeholders.",
    "summary": "2-3 sentence professional summary in their voice",
    "skills": ["..."],
    "work_history": [
      {"role": "...", "org": "...", "duration": "...",
       "highlights": ["concrete accomplishment", "..."]}
    ],
    "education": ["..."],
    "framing_notes": ["how to describe X when applying to Y", "..."]
  },
  "criteria": {
    "target_roles": ["role families they want"],
    "hard_filters": {
      "location": "e.g. Remote (US) or Seattle metro",
      "salary_floor": "number or null",
      "other": ["any absolute dealbreakers"]
    },
    "anti_criteria": [
      "title patterns or role shapes to score near zero even if skills match",
      {"text": "a strong negative that ISN'T an automatic no",
       "severity": "penalty"}
    ],
    "soft_preferences": [
      {"want": "description", "weight": "high|medium|low"}
    ],
    "scam_wariness": "note if their categories are scam-heavy (remote data entry is)"
  },
  "prefilter": {
    "title_exclude": [
      "case-insensitive substrings that disqualify a job by TITLE ALONE,",
      "before any model sees it (zero cost). Use for unambiguous junk the",
      "job boards keep returning: wrong professions ('registered nurse'),",
      "seniority they'd never touch ('director'). Beware substrings with",
      "legitimate uses — 'dean' also blocks 'assistant to the dean'; leave",
      "those to anti_criteria, which the scoring model applies with judgment."
    ]
  },
  "sources": {
    "adzuna": {"queries": [{"what": "…", "max_days_old": 7}]},
    "greenhouse": {"boards": ["slug"], "title_include": ["…"], "location_include": ["…"]},
    "ashby": {"orgs": ["slug"], "title_include": ["…"], "location_include": ["…"]},
    "lever": {"companies": ["slug"], "title_include": ["…"], "location_include": ["…"]},
    "icims": {"portals": [{"host": "…", "company": "…", "keywords": ["…"]}]},
    "govjobs": {"agencies": [{"agency": "slug", "company": "…"}],
                "title_include": ["…"], "location_include": ["…"]},
    "workday": {"tenants": [{"tenant": "…", "shard": "wd5", "site": "…",
                             "company": "…", "search": ["…"],
                             "location_include": ["…"]}]}
  }
}
```

**Severity** (applies to `anti_criteria` and `hard_filters.other`): a plain
string is a *dealbreaker*. The gate vetoes any job that clearly matches it. Use
`{"text": "...", "severity": "penalty"}` for things that count against a job
but that the applicant would still sometimes take (e.g. "prefers ≤4 years
required, but would stretch"). For each item, ask: "If a job were great
otherwise, would this alone make it a no?" The Gate page (Profiles → Gate)
flags dealbreakers that keep firing on jobs the applicant kept anyway; those
are the first candidates to downgrade. Phrase each item as one condition a
posting could explicitly state, because vague items misfire.

The `sources` block can be filled in by whoever operates the system rather than
the applicant — include it with best guesses and mark uncertainty. Adzuna
queries without a "where" search the shared household location (SEARCH_WHERE +
SEARCH_DISTANCE_KM from .env). In profile-refinement sessions, carry the
existing `sources` block over verbatim unless the refinement is specifically
about sourcing — the slugs/tenants in it were validated by hand.
