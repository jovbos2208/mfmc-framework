from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .production import production_status, run_production
from .sensitivity import run_field_sensitivity
from .synthetic import run_synthetic
from .workflow import (
    allocation_sweep,
    benchmark,
    field_benchmark,
    field_estimate,
    field_pilot,
    field_pod,
    inspect,
    optimal_allocation,
    pilot,
    prepare_field_snapshots,
    prepare_snapshots,
    report,
    run_all,
)


def parser():
    p=argparse.ArgumentParser(description="Field-aware MFMC allocation and matrix-free POD for PICLAS surface loads"); sub=p.add_subparsers(dest="command",required=True)
    for command in ("inspect","prepare-field-snapshots","field-pilot","optimal-allocation","field-estimate","field-pod","field-benchmark","prepare-snapshots","pilot","allocation-sweep","benchmark","report","run-all"):
        sp=sub.add_parser(command); sp.add_argument("--config",required=True)
    sp=sub.add_parser("synthetic"); sp.add_argument("--config",required=False); sp.add_argument("--output",default="paper_postprocessed/mfpod_surface_loads/synthetic_validation")
    sp=sub.add_parser("production", help="Run/resume pilot, field allocation, solver production, and analysis")
    sp.add_argument("--config", required=True)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--stop-after", choices=("plan", "pilot", "allocation", "production", "analysis"), default="analysis")
    sp=sub.add_parser("production-status", help="Show archive counts and restartable production state")
    sp.add_argument("--config", required=True)
    sp=sub.add_parser(
        "sensitivity",
        help="Postprocess existing production pools over m0, permutations, and reference sizes",
    )
    sp.add_argument("--config", required=True)
    sp.add_argument("--results-dir")
    sp.add_argument(
        "--minimum-targets",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8, 10, 12, 16, 20],
    )
    sp.add_argument(
        "--reference-sizes",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50],
    )
    sp.add_argument("--repetitions", type=int, default=30)
    sp.add_argument("--random-seed", type=int, default=20260727)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=="synthetic": result=run_synthetic(Path(args.output)); print(json.dumps({"output":args.output,"methods":result["methods"]},indent=2)); return 0
    if args.command=="production":
        cfg=load_config(args.config); result=run_production(cfg,resume=args.resume,dry_run=args.dry_run,stop_after=args.stop_after); print(json.dumps(result,default=str,indent=2)); return 0
    if args.command=="production-status":
        cfg=load_config(args.config); print(json.dumps(production_status(cfg),default=str,indent=2)); return 0
    if args.command=="sensitivity":
        cfg=load_config(args.config)
        result=run_field_sensitivity(
            cfg,
            results_dir=args.results_dir,
            minimum_targets=args.minimum_targets,
            reference_sizes=args.reference_sizes,
            repetitions=args.repetitions,
            random_seed=args.random_seed,
        )
        print(json.dumps(result,default=str,indent=2)); return 0
    cfg=load_config(args.config); funcs={"inspect":inspect,"prepare-field-snapshots":prepare_field_snapshots,"field-pilot":field_pilot,"optimal-allocation":optimal_allocation,"field-estimate":field_estimate,"field-pod":field_pod,"field-benchmark":field_benchmark,"prepare-snapshots":prepare_snapshots,"pilot":pilot,"allocation-sweep":allocation_sweep,"benchmark":benchmark,"report":report,"run-all":run_all}; result=funcs[args.command](cfg); print(json.dumps(result if isinstance(result,dict) else {"result":result},default=str,indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
