#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.selected_hf_geometry import build_selected_hf_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Build L1 meshes and PICLas configs for selected WP5 HF geometries")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--design-manifest", required=True)
    parser.add_argument("--lf-config", required=True)
    parser.add_argument("--output-root", default="piclas/geometry/cylinder_hex_wp5/L1")
    parser.add_argument("--config-output-dir", default="configs/studies/cylinder_hex_wp5_initial_hf")
    parser.add_argument("--base-config", default="configs/studies/cylinder_hex_piclas_adapter_l1_5seeds.json")
    parser.add_argument("--hf-samples-per-geometry", type=int, default=5)
    parser.add_argument("--gmsh", default="gmsh")
    parser.add_argument("--pyhope", default="pyhope")
    args = parser.parse_args()
    summary = build_selected_hf_suite(
        args.selection,
        args.design_manifest,
        args.lf_config,
        output_root=args.output_root,
        config_output_dir=args.config_output_dir,
        base_config_json=args.base_config,
        hf_samples_per_geometry=args.hf_samples_per_geometry,
        gmsh_executable=args.gmsh,
        pyhope_executable=args.pyhope,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
