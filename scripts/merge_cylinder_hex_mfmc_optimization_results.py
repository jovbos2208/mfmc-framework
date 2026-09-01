#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_mfmc_optimization import merge_geometry_mfmc_tpmc_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge TPMC optimization results and existing Sentman controls")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--sentman-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = merge_geometry_mfmc_tpmc_results(
        args.bundle,
        args.suite,
        args.run_root,
        args.sentman_results,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
