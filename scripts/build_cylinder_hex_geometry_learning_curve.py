#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_learning_curve import build_geometry_learning_curve


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the WP5 geometry-held-out surrogate learning curve")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-mf-relative-improvement", type=float, default=0.01)
    parser.add_argument("--target-geometry-rmse", type=float, default=1.0e-4)
    parser.add_argument("--minimum-geometry-count", type=int, default=12)
    args = parser.parse_args()
    result = build_geometry_learning_curve(
        args.manifests,
        args.output_dir,
        minimum_mf_relative_improvement=args.minimum_mf_relative_improvement,
        target_geometry_rmse=args.target_geometry_rmse,
        minimum_geometry_count=args.minimum_geometry_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
