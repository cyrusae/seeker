#!/usr/bin/env python3
"""seeker CLI: profile import, pipeline runs, tuning report.

  python cli.py import-profile profiles/yourname.json
  python cli.py run                 # ingest + evaluate everything pending
  python cli.py tuning-report yourname > report.md
"""
import argparse
import json
import sys

from app import db, pipeline, report


def cmd_import(path: str):
    with open(path) as f:
        profile = json.load(f)
    missing = report.REQUIRED_PROFILE_KEYS - profile.keys()
    if missing:
        sys.exit(f"Profile missing required keys: {sorted(missing)} "
                 f"(see docs/interview_spec.md for the schema)")
    db.init_db()
    version = db.import_profile(profile)
    print(f"Imported {profile['applicant_id']} as version {version}.")


def cmd_run():
    db.init_db()
    result = pipeline.run_full_cycle()
    print(json.dumps(result))


def cmd_tuning_report(applicant_id: str):
    """Markdown dump of eval outcomes + your feedback — take this plus the
    current profile back to a chat session to produce the next profile version.
    Also downloadable from the web UI's Profiles page."""
    db.init_db()
    print(report.tuning_report(applicant_id))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("import-profile"); sp.add_argument("path")
    sub.add_parser("run")
    sp = sub.add_parser("tuning-report"); sp.add_argument("applicant_id")
    args = p.parse_args()
    if args.cmd == "import-profile":
        cmd_import(args.path)
    elif args.cmd == "run":
        cmd_run()
    elif args.cmd == "tuning-report":
        cmd_tuning_report(args.applicant_id)


if __name__ == "__main__":
    main()
