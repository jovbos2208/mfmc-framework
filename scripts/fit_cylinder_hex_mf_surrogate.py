#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_mf_surrogate import fit_geometry_multifidelity_surrogate


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the WP5 DSMC-target geometry/uncertainty surrogate")
    parser.add_argument("--bundle", required=True, help="Paired DSMC/TPMC WP5 bundle JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--q-norm", type=float, default=0.75)
    parser.add_argument("--max-interaction", type=int, default=2)
    parser.add_argument("--target-hf-per-geometry", type=int, default=10)
    parser.add_argument("--acquisition-geometry-count", type=int, default=3)
    args = parser.parse_args()
    result = fit_geometry_multifidelity_surrogate(
        args.bundle,
        args.output_dir,
        degree=args.degree,
        q_norm=args.q_norm,
        max_interaction=args.max_interaction,
        target_hf_per_geometry=args.target_hf_per_geometry,
        acquisition_geometry_count=args.acquisition_geometry_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
