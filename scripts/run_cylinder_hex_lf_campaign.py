#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_lf_workflow import prepare_lf_campaign, run_lf_campaign


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the restartable WP5 ADBSat geometry campaign")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--design-manifest", required=True)
    prepare.add_argument("--source-samples", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--n-samples", type=int, default=256)
    prepare.add_argument("--adbsat-runtime-dir", default="ADBSat-PyVersion")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_lf_campaign(
            args.design_manifest, args.source_samples, args.config,
            n_samples=args.n_samples, adbsat_runtime_dir=args.adbsat_runtime_dir,
        )
    else:
        result = run_lf_campaign(args.config, args.output_dir, execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
