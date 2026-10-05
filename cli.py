#!/usr/bin/env python3
"""seeker CLI: profile import, pipeline runs, tuning report.

  python cli.py import-profile profiles/yourname.json
  python cli.py run                 # ingest + evaluate everything pending
  python cli.py tuning-report yourname > report.md
  python cli.py gate-backtest       # score past decisions with the Jev gate
  python cli.py gate-fit [--apply]  # relearn gate weights from your decisions
"""
import argparse
import json
import sys

from app import db, pipeline, report


def cmd_import(path: str, note: str | None = None):
    with open(path) as f:
        profile = json.load(f)
    missing = report.REQUIRED_PROFILE_KEYS - profile.keys()
    if missing:
        sys.exit(f"Profile missing required keys: {sorted(missing)} "
                 f"(see docs/interview_spec.md for the schema)")
    db.init_db()
    version = db.import_profile(profile, note)
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


def cmd_gate_backtest(args):
    """Run the Jev gate over everything you've already decided on and report
    how well it ranks vs. the current DeepSeek score. Answers are cached, so
    re-runs are free unless jobs or profiles changed."""
    from app import backtest
    db.init_db()
    md = backtest.run(skipped_sample=args.skipped, include_backlog=not args.no_backlog,
                      workers=args.workers, limit=args.limit,
                      progress=lambda m: print(m, file=sys.stderr))
    with open(args.out, "w") as f:
        f.write(md)
    print(f"wrote {args.out}", file=sys.stderr)


def cmd_gate_fit(args):
    """Fit a candidate gate model from your decisions and print its preview.
    --apply activates it (same as Apply on Profiles → Gate)."""
    from app import gatefit
    db.init_db()
    v = gatefit.fit(evaluate_missing=args.evaluate_missing,
                    progress=lambda m: print(m, file=sys.stderr))
    from app import gate
    m = gate.get_model(v)
    st = m["stats"]
    print(f"candidate v{v}: {st['n_labels']} decisions ({st['n_pos']} kept, "
          f"{st['n_vetoed']} vetoed)")
    print(f"AUC candidate {st['auc_candidate']:.3f} vs current v{st['current_version']} "
          f"{st['auc_current']:.3f}")
    print("weights:", json.dumps(m["weights"]), "| applicant bias:", json.dumps(m["applicant_bias"]))
    print("thresholds:", json.dumps(m["thresholds"]))
    print("unreviewed jobs that would move:", json.dumps(st["moves"]) or "none")
    for f in st["dealbreaker_flags"]:
        print(f"  ⚑ {f['applicant']}: dealbreaker fired on {f['kept']} kept jobs "
              f"({f['kept_pct']}%) vs {f['rejected']} rejected ({f['rejected_pct']}%) "
              f"— {f['item'][:90]}")
    for b in st.get("broad_flags", []):
        print(f"  ⚠ {b['applicant']}: {b['kind']} item applies to {b['share']}% of decided jobs "
              f"(inverted or misfiled?) — {b['item'][:90]}")
    for w in st["warnings"]:
        print("  ! " + w)
    if args.apply:
        n = gatefit.apply(v)
        print(f"applied v{v}; rescored {n} jobs")
    else:
        print(f"not applied — rerun with --apply, or use Profiles → Gate")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("import-profile"); sp.add_argument("path")
    sp.add_argument("--note", help="what this version is trying to fix (shown in gate change reports)")
    sub.add_parser("run")
    sp = sub.add_parser("tuning-report"); sp.add_argument("applicant_id")
    sp = sub.add_parser("gate-backtest")
    sp.add_argument("--skipped", type=int, default=300, help="auto-skipped jobs to sample")
    sp.add_argument("--no-backlog", action="store_true", help="leave out the review backlog")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--limit", type=int, help="evaluate at most N jobs (smoke test)")
    sp.add_argument("--out", default="data/gate_backtest.md")
    sp = sub.add_parser("gate-fit")
    sp.add_argument("--apply", action="store_true", help="activate the fitted model")
    sp.add_argument("--evaluate-missing", action="store_true",
                    help="call Jev for decisions with no cached answers (~$0.00025 each)")
    args = p.parse_args()
    if args.cmd == "import-profile":
        cmd_import(args.path, args.note)
    elif args.cmd == "run":
        cmd_run()
    elif args.cmd == "tuning-report":
        cmd_tuning_report(args.applicant_id)
    elif args.cmd == "gate-backtest":
        cmd_gate_backtest(args)
    elif args.cmd == "gate-fit":
        cmd_gate_fit(args)


if __name__ == "__main__":
    main()
