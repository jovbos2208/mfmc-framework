#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_mfmc_optimization import select_geometry_mfmc_optimization_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the next surrogate-free TPMC/Sentman MFMC optimization batch")
    parser.add_argument("--design-manifest", required=True)
    parser.add_argument("--sentman-metrics", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--mfmc-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--mean-objective-weight", type=float, default=0.5)
    parser.add_argument("--std-objective-weight", type=float, default=0.5)
    args = parser.parse_args()
    result = select_geometry_mfmc_optimization_batch(
        args.design_manifest,
        args.sentman_metrics,
        args.bundle,
        args.mfmc_metrics,
        args.output,
        count=args.count,
        iteration=args.iteration,
        mean_objective_weight=args.mean_objective_weight,
        std_objective_weight=args.std_objective_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
