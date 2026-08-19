#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_round2 import select_round2_geometries


def main() -> int:
    parser = argparse.ArgumentParser(description="Select new WP5 training geometries for round 2")
    parser.add_argument("--design-manifest", required=True)
    parser.add_argument("--sentman-metrics", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--surrogate-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--round-number", type=int, default=2)
    args = parser.parse_args()
    result = select_round2_geometries(
        args.design_manifest, args.sentman_metrics, args.bundle,
        args.surrogate_manifest, args.output, count=args.count, round_number=args.round_number,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
