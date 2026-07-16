#!/usr/bin/env python3
"""Audit available Cube/GOCE surface archives and generate paper-ready evidence.

The script reports absent fields rather than synthesizing them.  It is intended
to be run from the clean Framework checkout while pointing at the research
workspace that contains ``paper_postprocessed``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfmc_campaign.field_mfpod.allocation import AllocationOptions, optimize_allocation


CASES = {
    "Cube-300km": "cube_300km",
    "GOCE-244km": "goce_244km",
}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _scalar(npz, name, default=np.nan):
    return float(np.asarray(npz[name]).reshape(-1)[0]) if name in npz.files else default


def audit_case(root: Path, case: str, folder: str) -> dict:
    base = root / "paper_postprocessed" / "field_inputs" / folder
    paths = {name: base / f"{name}_surface_loads.npz" for name in ("DSMC", "TPMC", "SENTMAN")}
    row = {"case": case, "sentman_archive": paths["SENTMAN"].exists()}
    if not paths["DSMC"].exists() or not paths["TPMC"].exists():
        row["ready_for_paired_audit"] = False
        return row
    with np.load(paths["DSMC"], allow_pickle=False) as h, np.load(paths["TPMC"], allow_pickle=False) as l:
        h_ids = np.asarray(h["sample_id"]).astype(str)
        l_ids = np.asarray(l["sample_id"]).astype(str)
        common = sorted(set(h_ids) & set(l_ids))
        h_lookup = {value: i for i, value in enumerate(h_ids)}
        l_lookup = {value: i for i, value in enumerate(l_ids)}
        hi = np.asarray([h_lookup[value] for value in common])
        li = np.asarray([l_lookup[value] for value in common])
        h_force = np.asarray(h["force_per_area"], dtype=float)[hi]
        l_force = np.asarray(l["force_per_area"], dtype=float)[li]
        area_h = np.asarray(h["face_area"], dtype=float)
        area_l = np.asarray(l["face_area"], dtype=float)
        q_h = np.asarray(h["q_inf"], dtype=float).reshape(-1)[hi]
        q_l = np.asarray(l["q_inf"], dtype=float).reshape(-1)[li]
        aref_h = np.asarray(h["A_ref_per_sample"] if "A_ref_per_sample" in h.files else np.full(len(h_ids), _scalar(h, "A_ref")), dtype=float).reshape(-1)[hi]
        aref_l = np.asarray(l["A_ref_per_sample"] if "A_ref_per_sample" in l.files else np.full(len(l_ids), _scalar(l, "A_ref")), dtype=float).reshape(-1)[li]
        z_h = h_force / q_h[:, None, None] * np.sqrt(area_h[None, :, None] / aref_h[:, None, None])
        z_l = l_force / q_l[:, None, None] * np.sqrt(area_l[None, :, None] / aref_l[:, None, None])
        energy_h = np.sum(z_h * z_h, axis=(1, 2))
        energy_l = np.sum(z_l * z_l, axis=(1, 2))
        cd_corr = np.nan
        if "C_D" in h.files and "C_D" in l.files:
            cd_corr = float(np.corrcoef(np.asarray(h["C_D"])[hi], np.asarray(l["C_D"])[li])[0, 1])
        preferred = ("coordinate_frame", "cpu_hours", "face_normal", "hardware", "reference_point")
        row.update({
            "ready_for_paired_audit": True,
            "n_dsmc": len(h_ids),
            "n_tpmc": len(l_ids),
            "n_paired": len(common),
            "ordered_ids_equal": bool(np.array_equal(h_ids, l_ids)),
            "duplicate_dsmc_ids": int(len(h_ids) - len(set(h_ids))),
            "duplicate_tpmc_ids": int(len(l_ids) - len(set(l_ids))),
            "n_faces": int(area_h.size),
            "face_areas_equal": bool(np.allclose(area_h, area_l, rtol=1e-10, atol=0.0)),
            "face_centers_equal": bool("face_center" in h.files and "face_center" in l.files and np.allclose(h["face_center"], l["face_center"], atol=1e-12)),
            "a_ref_dsmc_min": float(np.min(aref_h)),
            "a_ref_dsmc_max": float(np.max(aref_h)),
            "a_ref_tpmc_min": float(np.min(aref_l)),
            "a_ref_tpmc_max": float(np.max(aref_l)),
            "q_inf_dsmc_min": float(np.min(q_h)),
            "q_inf_dsmc_max": float(np.max(q_h)),
            "q_inf_tpmc_min": float(np.min(q_l)),
            "q_inf_tpmc_max": float(np.max(q_l)),
            "field_energy_correlation": float(np.corrcoef(energy_h, energy_l)[0, 1]),
            "cd_correlation": cd_corr,
            "missing_preferred_dsmc": [name for name in preferred if name not in h.files],
            "missing_preferred_tpmc": [name for name in preferred if name not in l.files],
            "normalization": "force_per_area/q_inf with sqrt(face_area/A_ref) weighting",
            "coordinate_frame_status": "implicit body-fixed assumption" if "coordinate_frame" not in h.files else str(np.asarray(h["coordinate_frame"]).reshape(-1)[0]),
        })
    summary = root / "paper_postprocessed" / "field_pod_mfmc" / case / "summary.json"
    if summary.exists():
        content = json.loads(summary.read_text(encoding="utf-8"))
        projection = content.get("projection_summary", {})
        row.update({
            "projection_residual_mean": projection.get("mean_projection_residual"),
            "projection_residual_max": projection.get("max_projection_residual"),
            "cd_projection_rmse": projection.get("cd_projection_rmse"),
            "cd_variance_projection_loss": projection.get("cd_variance_projection_loss"),
        })
    metrics = root / "paper_postprocessed" / "field_pod_mfmc" / case / "comparison_metrics.csv"
    if metrics.exists():
        with metrics.open(encoding="utf-8") as handle:
            values = list(csv.DictReader(handle))
        budget10 = [entry for entry in values if float(entry["budget"]) == 10.0]
        for key in ("gain_mu", "gain_cov", "principal_angle_max_mfmc_rad", "principal_angle_max_dsmc_only_rad"):
            if budget10:
                row[f"budget10_median_{key}"] = float(np.median([float(entry[key]) for entry in budget10]))
    return row


def synthetic_allocations() -> tuple[list[dict], dict]:
    rng = np.random.default_rng(4401)
    h = rng.normal(size=400)
    t = h + 0.20 * rng.normal(size=400)
    s = 0.65 * h + 0.30 * rng.normal(size=400)
    responses = {"DSMC": np.column_stack([h, h * h]), "TPMC": np.column_stack([t, t * t]), "SENTMAN": np.column_stack([s, s * s])}
    costs = {"DSMC": 1.0, "TPMC": 0.10, "SENTMAN": 0.01}
    rows = []
    robust_result = None
    for budget in (6.0, 10.0, 14.0):
        common = dict(
            budget=budget,
            minimum_target=2,
            minimum_counts={"TPMC": 2},
            min_ratios={"TPMC": 1.0},
            max_ratios={"TPMC": 10.0},
            maximum_counts={"DSMC": 14, "TPMC": 80, "SENTMAN": 120},
            random_seed=4401,
        )
        for mode in ("enumeration", "continuous_round", "greedy"):
            result = optimize_allocation(responses, costs, AllocationOptions(mode=mode, **common))
            rows.append({"budget": budget, "method": mode, **result.counts, "cost": result.total_cost, "objective": result.objective})
        robust_result = optimize_allocation(
            responses,
            costs,
            AllocationOptions(
                mode="bootstrap_robust",
                bootstrap_repeats=200,
                robust_quantile=.9,
                max_enumeration_candidates=20000,
                **common,
            ),
        )
        rows.append({"budget": budget, "method": "bootstrap_robust", **robust_result.counts, "cost": robust_result.total_cost, "objective": robust_result.objective})
    return rows, robust_result.as_dict()


def write_outputs(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    audits = [audit_case(root, case, folder) for case, folder in CASES.items()]
    allocations, robust = synthetic_allocations()
    (output / "data_readiness.json").write_text(json.dumps(_jsonable({"cases": audits}), indent=2), encoding="utf-8")
    (output / "robust_allocation_diagnostics.json").write_text(json.dumps(_jsonable(robust), indent=2), encoding="utf-8")
    for filename, rows in (("data_readiness.csv", audits), ("synthetic_allocation_comparison.csv", allocations)):
        fields = sorted({key for row in rows for key in row})
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for method in ("enumeration", "continuous_round", "greedy", "bootstrap_robust"):
        selected = [row for row in allocations if row["method"] == method]
        ax.plot([row["budget"] for row in selected], [row["objective"] for row in selected], marker="o", label=method.replace("_", " "))
    ax.set_xlabel("Budget (DSMC-equivalent cost units)")
    ax.set_ylabel("Modeled normalized feature variance")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "synthetic_allocation_objective.pdf")
    fig.savefig(output / "synthetic_allocation_objective.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    names = [row["case"] for row in audits]
    means = [row.get("projection_residual_mean", np.nan) for row in audits]
    maxima = [row.get("projection_residual_max", np.nan) for row in audits]
    x = np.arange(len(names)); width = .36
    ax.bar(x - width / 2, means, width, label="mean")
    ax.bar(x + width / 2, maxima, width, label="maximum")
    ax.axhline(.10, color="black", linestyle="--", linewidth=.8, label="mean warning threshold")
    ax.set_xticks(x, names)
    ax.set_ylabel("TPMC-basis projection residual")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "case_projection_residuals.pdf")
    fig.savefig(output / "case_projection_residuals.png", dpi=220)
    plt.close(fig)
    return {"cases": audits, "allocation_rows": len(allocations), "output": str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_jsonable(write_outputs(args.project_root.resolve(), args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
