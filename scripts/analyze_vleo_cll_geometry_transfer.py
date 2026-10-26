#!/usr/bin/env python3
"""Audit CLL control-variate transfer from the baseline to one paired geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_cd(root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [row for row in read_rows(root / "model_evaluations.csv") if row["qoi"] == "C_D"]
    models: dict[str, dict[str, float]] = {}
    for row in rows:
        models.setdefault(row["model_id"], {})[row["sample_id"]] = float(row["value"])
    hf_name = next(name for name in models if name.startswith("PICLas"))
    lf_name = next(name for name in models if name.startswith("ADBSat"))
    ids = sorted(models[hf_name])
    if ids != sorted(models[lf_name]):
        raise ValueError(f"Unpaired model evaluations in {root}")
    return ids, np.asarray([models[hf_name][key] for key in ids]), np.asarray([models[lf_name][key] for key in ids])


def sample_contract(root: Path) -> dict[str, tuple[float, float]]:
    rows = read_rows(root / "sample_inputs.csv")
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        if row["qoi"] == "C_D" and row["model_id"].startswith("PICLas"):
            out[row["sample_id"]] = (
                float(row["input__cll_alpha_n"]),
                float(row["input__cll_sigma_t"]),
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("transfer", type=Path)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--transfer-manifest", type=Path, required=True)
    parser.add_argument("--baseline-lf-recovery", type=Path, default=None)
    args = parser.parse_args()
    base_ids, base_hf, base_lf = paired_cd(args.baseline)
    transfer_ids, transfer_hf, transfer_lf = paired_cd(args.transfer)
    baseline_lf_source = "baseline model_evaluations.csv"
    if args.baseline_lf_recovery is not None:
        recovery_rows = read_rows(args.baseline_lf_recovery)
        recovery = {row["sample_id"]: float(row["C_D"]) for row in recovery_rows}
        if sorted(recovery) != base_ids:
            raise ValueError("Recovered baseline LF samples do not match the baseline HF sample IDs")
        base_lf = np.asarray([recovery[key] for key in base_ids], dtype=float)
        baseline_lf_source = str(args.baseline_lf_recovery)
    if base_ids != transfer_ids or sample_contract(args.baseline) != sample_contract(args.transfer):
        raise ValueError("Baseline and transfer geometry do not use the exact same ordered CLL samples")
    base_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    transfer_manifest = json.loads(args.transfer_manifest.read_text(encoding="utf-8"))
    base_area = 0.5 * float(base_manifest["validation"]["surface_area_m2"])
    transfer_area = 0.5 * float(transfer_manifest["validation"]["surface_area_m2"])
    base_drag = base_hf * base_area
    transfer_drag = transfer_hf * transfer_area
    delta_hf = transfer_drag - base_drag
    delta_lf = transfer_lf * transfer_area - base_lf * base_area
    base_rho = float(np.corrcoef(base_hf, base_lf)[0, 1])
    transfer_rho = float(np.corrcoef(transfer_hf, transfer_lf)[0, 1])
    base_mean_ratio = float(base_lf.mean() / base_hf.mean())
    transfer_mean_ratio = float(transfer_lf.mean() / transfer_hf.mean())
    lf_provenance_gate = bool(
        base_rho >= 0.95
        and transfer_rho >= 0.95
        and 0.8 <= base_mean_ratio <= 1.2
        and 0.8 <= transfer_mean_ratio <= 1.2
    )
    rng = np.random.default_rng(20261023)
    indices = rng.integers(0, len(base_ids), size=(50_000, len(base_ids)))
    mean_ratio_boot = transfer_drag[indices].mean(axis=1) / base_drag[indices].mean(axis=1)
    sd_ratio_boot = transfer_drag[indices].std(axis=1, ddof=1) / base_drag[indices].std(axis=1, ddof=1)
    delta_mean_boot = delta_hf[indices].mean(axis=1)
    delta_hf_boot = delta_hf[indices]
    delta_lf_boot = delta_lf[indices]
    delta_hf_centered = delta_hf_boot - delta_hf_boot.mean(axis=1, keepdims=True)
    delta_lf_centered = delta_lf_boot - delta_lf_boot.mean(axis=1, keepdims=True)
    delta_rho_boot = np.sum(delta_hf_centered * delta_lf_centered, axis=1) / np.sqrt(
        np.sum(delta_hf_centered ** 2, axis=1) * np.sum(delta_lf_centered ** 2, axis=1)
    )
    delta_rho_boot = delta_rho_boot[np.isfinite(delta_rho_boot)]
    report = {
        "n_pairs": len(base_ids),
        "baseline_lf_source": baseline_lf_source,
        "baseline_hf_lf_rho": base_rho,
        "transfer_hf_lf_rho": transfer_rho,
        "baseline_lf_hf_mean_ratio": base_mean_ratio,
        "transfer_lf_hf_mean_ratio": transfer_mean_ratio,
        "lf_provenance_gate_passed": lf_provenance_gate,
        "geometry_delta_hf_lf_rho": (
            float(np.corrcoef(delta_hf, delta_lf)[0, 1]) if lf_provenance_gate else None
        ),
        "geometry_delta_hf_lf_rho_bootstrap_95": (
            [float(value) for value in np.quantile(delta_rho_boot, [0.025, 0.975])]
            if lf_provenance_gate else None
        ),
        "hf_baseline_transfer_rho": float(np.corrcoef(base_drag, transfer_drag)[0, 1]),
        "baseline_drag_area_mean_m2": float(base_drag.mean()),
        "transfer_drag_area_mean_m2": float(transfer_drag.mean()),
        "mean_drag_area_ratio_transfer_to_baseline": float(transfer_drag.mean() / base_drag.mean()),
        "mean_drag_area_ratio_bootstrap_95": [float(value) for value in np.quantile(mean_ratio_boot, [0.025, 0.975])],
        "baseline_drag_area_sd_m2": float(base_drag.std(ddof=1)),
        "transfer_drag_area_sd_m2": float(transfer_drag.std(ddof=1)),
        "sd_drag_area_ratio_transfer_to_baseline": float(transfer_drag.std(ddof=1) / base_drag.std(ddof=1)),
        "sd_drag_area_ratio_bootstrap_95": [float(value) for value in np.quantile(sd_ratio_boot, [0.025, 0.975])],
        "hf_geometry_delta_mean_m2": float(delta_hf.mean()),
        "hf_geometry_delta_mean_bootstrap_95_m2": [float(value) for value in np.quantile(delta_mean_boot, [0.025, 0.975])],
        "hf_geometry_delta_sd_m2": float(delta_hf.std(ddof=1)),
        "lf_geometry_delta_rmse_m2": (
            float(np.sqrt(np.mean((delta_hf - delta_lf) ** 2))) if lf_provenance_gate else None
        ),
        "baseline_reference_area_m2": base_area,
        "transfer_reference_area_m2": transfer_area,
        "interpretation": (
            "HF geometry comparison is valid, but LF geometry-transfer metrics are suppressed because the model-provenance gate failed."
            if not lf_provenance_gate
            else "HF and LF geometry-transfer metrics passed the numerical provenance gate."
        ),
    }
    output = args.transfer / "phase3_geometry_transfer_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
