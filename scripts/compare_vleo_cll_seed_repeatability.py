#!/usr/bin/env python3
"""Compare fixed-anchor PICLas repeatability across two VLEO geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_audit", type=Path)
    parser.add_argument("candidate_audit", type=Path)
    parser.add_argument("--baseline-area", type=float, required=True)
    parser.add_argument("--candidate-area", type=float, required=True)
    parser.add_argument("--planning-n", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_audit.read_text(encoding="utf-8"))
    vb = float(baseline["pooled"]["within_anchor_variance_c_d"])
    vc = float(candidate["pooled"]["within_anchor_variance_c_d"])
    # Each audit contains three groups of five: 3*(5-1)=12 pooled df.
    df_b = sum(int(baseline[k]["n"]) - 1 for k in ("low", "mid", "high"))
    df_c = sum(int(candidate[k]["n"]) - 1 for k in ("low", "mid", "high"))
    v_pool = (df_b * vb + df_c * vc) / (df_b + df_c)
    per_state_drag_sd = float(
        np.sqrt(vb * args.baseline_area**2 + vc * args.candidate_area**2)
    )
    report = {
        "schema_version": 1,
        "baseline": {"pooled_sd_c_d": float(np.sqrt(vb)), "df": df_b},
        "candidate": {"pooled_sd_c_d": float(np.sqrt(vc)), "df": df_c},
        "candidate_to_baseline_variance_ratio": vc / vb,
        "candidate_to_baseline_sd_ratio": float(np.sqrt(vc / vb)),
        "cross_geometry_pooled_sd_c_d": float(np.sqrt(v_pool)),
        "conservative_independent_seed_difference_sd_drag_area_m2": per_state_drag_sd,
        "planning_n": args.planning_n,
        "three_se_mean_difference_resolution_m2": float(
            3.0 * per_state_drag_sd / np.sqrt(args.planning_n)
        ),
        "interpretation": [
            "Particle noise is small at both geometries but geometry-dependent.",
            "The three-SE value is a provisional numerical resolution diagnostic, not a locked inferential threshold.",
            "Independent seeds make the difference-noise calculation conservative relative to beneficial common random numbers."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
