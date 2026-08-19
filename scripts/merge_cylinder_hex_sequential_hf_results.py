#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_mf_surrogate import merge_sequential_hf_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge collected sequential DSMC results into the WP5 paired bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = merge_sequential_hf_results(args.bundle, args.run_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
