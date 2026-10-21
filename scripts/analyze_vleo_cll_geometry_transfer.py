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
    args = parser.parse_args()
    base_ids, base_hf, base_lf = paired_cd(args.baseline)
    transfer_ids, transfer_hf, transfer_lf = paired_cd(args.transfer)
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
    report = {
        "n_pairs": len(base_ids),
        "baseline_hf_lf_rho": float(np.corrcoef(base_hf, base_lf)[0, 1]),
        "transfer_hf_lf_rho": float(np.corrcoef(transfer_hf, transfer_lf)[0, 1]),
        "geometry_delta_hf_lf_rho": float(np.corrcoef(delta_hf, delta_lf)[0, 1]),
        "baseline_drag_area_mean_m2": float(base_drag.mean()),
        "transfer_drag_area_mean_m2": float(transfer_drag.mean()),
        "mean_drag_area_ratio_transfer_to_baseline": float(transfer_drag.mean() / base_drag.mean()),
        "baseline_drag_area_sd_m2": float(base_drag.std(ddof=1)),
        "transfer_drag_area_sd_m2": float(transfer_drag.std(ddof=1)),
        "sd_drag_area_ratio_transfer_to_baseline": float(transfer_drag.std(ddof=1) / base_drag.std(ddof=1)),
        "hf_geometry_delta_mean_m2": float(delta_hf.mean()),
        "hf_geometry_delta_sd_m2": float(delta_hf.std(ddof=1)),
        "lf_geometry_delta_rmse_m2": float(np.sqrt(np.mean((delta_hf - delta_lf) ** 2))),
        "baseline_reference_area_m2": base_area,
        "transfer_reference_area_m2": transfer_area,
    }
    output = args.transfer / "phase3_geometry_transfer_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
