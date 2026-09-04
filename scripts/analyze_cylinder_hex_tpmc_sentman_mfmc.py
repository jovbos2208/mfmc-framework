#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_mfmc_optimization import analyze_geometry_mfmc_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate geometry drag mean/std with target=TPMC and control=Sentman"
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-hf-equivalent", type=float, default=20.0)
    parser.add_argument("--pilot-count", type=int, default=5)
    parser.add_argument(
        "--target-run-count",
        type=int,
        help="Use a fixed target-run budget (20 for the control-node workflow)",
    )
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    parser.add_argument("--mean-objective-weight", type=float, default=0.5)
    parser.add_argument("--std-objective-weight", type=float, default=0.5)
    parser.add_argument("--target-cost-override", type=float)
    parser.add_argument("--control-cost-override", type=float)
    parser.add_argument("--minimum-abs-control-correlation", type=float, default=0.5)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--improvement-probability-threshold", type=float, default=0.95)
    args = parser.parse_args()
    result = analyze_geometry_mfmc_bundle(
        args.bundle,
        args.output_dir,
        budget_hf_equivalent=args.budget_hf_equivalent,
        pilot_count=args.pilot_count,
        target_run_count=args.target_run_count,
        crossfit_folds=args.crossfit_folds,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        mean_objective_weight=args.mean_objective_weight,
        std_objective_weight=args.std_objective_weight,
        target_cost_override=args.target_cost_override,
        control_cost_override=args.control_cost_override,
        minimum_abs_control_correlation=args.minimum_abs_control_correlation,
        confidence_level=args.confidence_level,
        improvement_probability_threshold=args.improvement_probability_threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
