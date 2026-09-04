#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_local_validation import build_dsmc_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Report paired final DSMC-vs-TPMC validation")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--finalists", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    result = build_dsmc_validation_report(
        args.bundle, args.finalists, args.output_dir,
        bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
