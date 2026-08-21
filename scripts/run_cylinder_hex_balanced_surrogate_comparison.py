#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_learning_curve import build_geometry_learning_curve
from mfmc_campaign.geometry_metric_gp import fit_geometry_metric_gp
from mfmc_campaign.geometry_mf_surrogate import fit_geometry_multifidelity_surrogate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the balanced 6/9/12 PCE learning curve and the 12-geometry metric-GP fallback"
    )
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Paired bundle in increasing geometry-count order; pass exactly three times",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-lf-per-geometry", type=int, default=90)
    parser.add_argument("--training-hf-per-geometry", type=int, default=5)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--q-norm", type=float, default=0.75)
    parser.add_argument("--max-interaction", type=int, default=2)
    parser.add_argument("--target-geometry-rmse", type=float, default=1.0e-4)
    parser.add_argument("--bootstrap-count", type=int, default=500)
    parser.add_argument("--optimizer-restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if len(args.bundle) != 3:
        parser.error("--bundle must be supplied exactly three times (6, 9 and 12 geometries)")

    root = Path(args.output_dir).resolve()
    reference_bundle = Path(args.bundle[-1]).resolve()
    pce_manifests: list[str] = []
    pce_results = []
    for bundle in args.bundle:
        payload = json.loads(Path(bundle).resolve().read_text(encoding="utf-8"))
        n_geometries = len(payload["selected_geometry_ids"])
        result = fit_geometry_multifidelity_surrogate(
            bundle,
            root / f"pce_{n_geometries:02d}_geometries",
            degree=args.degree,
            q_norm=args.q_norm,
            max_interaction=args.max_interaction,
            target_geometry_rmse=args.target_geometry_rmse,
            training_lf_per_geometry=args.training_lf_per_geometry,
            training_hf_per_geometry=args.training_hf_per_geometry,
            balance_reference_bundle_json=reference_bundle,
        )
        pce_results.append(result)
        pce_manifests.append(result["manifest_json"])

    curve = build_geometry_learning_curve(
        pce_manifests,
        root / "balanced_pce_learning_curve",
        target_geometry_rmse=args.target_geometry_rmse,
        require_balanced_training=True,
    )
    gp = fit_geometry_metric_gp(
        reference_bundle,
        root / "geometry_metric_gp_12",
        bootstrap_count=args.bootstrap_count,
        optimizer_restarts=args.optimizer_restarts,
        seed=args.seed,
    )
    gp_source_eligible = pce_results[-1]["model_selection"]["selected_surrogate"] == "lf_pce"
    if curve["status"] == "target_met":
        decision = "joint_uncertainty_geometry_pce"
    elif gp_source_eligible and gp["status"] == "geometry_metric_gp_ready_for_optimization_candidate":
        decision = "two_stage_geometry_metric_gp"
    else:
        decision = "more_geometry_acquisition_required"
    summary = {
        "schema_version": 1,
        "status": decision,
        "balance_reference_bundle": str(reference_bundle),
        "training_lf_per_geometry": args.training_lf_per_geometry,
        "training_hf_per_geometry": args.training_hf_per_geometry,
        "pce_manifests": pce_manifests,
        "pce_learning_curve_manifest": curve["manifest_json"],
        "geometry_metric_gp_manifest": gp["manifest_json"],
        "geometry_metric_gp_source_eligible": gp_source_eligible,
        "decision_rule": [
            "Use joint PCE if the balanced 12-geometry held-out RMSE target is met.",
            "Otherwise use metric GP only if all metric-specific LOO relative-RMSE targets are met.",
            "Otherwise acquire additional geometries; do not add DSMC samples per existing geometry first.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "balanced_surrogate_comparison.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "manifest_json": str(manifest_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
