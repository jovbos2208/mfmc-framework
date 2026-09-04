#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_metric_gp import fit_geometry_metric_gp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit heteroskedastic geometry GPs for TPMC mean/std/q95 drag area"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=500)
    parser.add_argument("--optimizer-restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    result = fit_geometry_metric_gp(
        args.bundle,
        args.output_dir,
        bootstrap_count=args.bootstrap_count,
        optimizer_restarts=args.optimizer_restarts,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
