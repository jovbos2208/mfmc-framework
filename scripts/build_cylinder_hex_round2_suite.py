#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_round2 import build_round2_piclas_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Build L1 DSMC/TPMC workflows for WP5 round 2")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--design-manifest", required=True)
    parser.add_argument("--lf-config", required=True)
    parser.add_argument("--output-root", default="piclas/geometry/cylinder_hex_wp5/L1")
    parser.add_argument("--config-output-dir", default="configs/studies/cylinder_hex_wp5_round2")
    parser.add_argument("--suite-output", default="outputs/cylinder_hex/wp5_round2/suite.json")
    parser.add_argument("--base-config", default="configs/studies/cylinder_hex_piclas_adapter_l1_5seeds.json")
    parser.add_argument("--n-dsmc", type=int, default=5)
    parser.add_argument("--n-tpmc", type=int, default=90)
    parser.add_argument("--gmsh", default="gmsh")
    parser.add_argument("--pyhope", default="pyhope")
    parser.add_argument("--round-number", type=int, default=2)
    parser.add_argument("--mpi-procs", type=int, default=128)
    args = parser.parse_args()
    result = build_round2_piclas_suite(
        args.selection, args.design_manifest, args.lf_config,
        output_root=args.output_root,
        config_output_dir=args.config_output_dir,
        suite_output_json=args.suite_output,
        base_config_json=args.base_config,
        n_dsmc=args.n_dsmc,
        n_tpmc=args.n_tpmc,
        gmsh_executable=args.gmsh,
        pyhope_executable=args.pyhope,
        round_number=args.round_number,
        mpi_procs=args.mpi_procs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
