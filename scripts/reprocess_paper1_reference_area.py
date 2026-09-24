#!/usr/bin/env python3
"""Re-normalize legacy Paper-1 PICLas coefficients without changing raw archives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LINEAR_QOIS = {"C_D", "C_L", "C_Y", "C_Mx", "C_My", "C_Mz"}
SQUARED_QOIS = {"C_D2", "C_L2", "C_Y2"}


def _surface_archive(framework_root: Path, hf_mesh: str) -> Path:
    mesh = Path(hf_mesh)
    name = mesh.name
    if not name.endswith("_mesh.h5"):
        raise ValueError(f"Unsupported PICLas mesh name: {hf_mesh}")
    return framework_root / "piclas" / mesh.with_name(name.removesuffix("_mesh.h5") + ".surface.npz")


def reprocess_case(case_dir: Path, framework_root: Path) -> dict:
    snapshot_path = case_dir / "config_snapshot.json"
    source_path = case_dir / "model_evaluations.csv"
    if not snapshot_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(f"Missing config_snapshot.json or model_evaluations.csv in {case_dir}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config = snapshot.get("config", snapshot)
    geometries = config.get("geometries", [])
    if len(geometries) != 1:
        raise ValueError(f"Expected exactly one geometry in {snapshot_path}")
    metadata = geometries[0].get("metadata", {})
    target_area = float(metadata["reference_area_m2"])
    surface_path = _surface_archive(framework_root, str(metadata["hf_mesh"]))
    with np.load(surface_path) as surface:
        legacy_area = 0.5 * float(np.asarray(surface["triangle_area"], dtype=float).sum())
    factor = legacy_area / target_area

    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "model_id" not in fieldnames or "qoi" not in fieldnames or "value" not in fieldnames:
        raise ValueError(f"Required columns missing in {source_path}")

    corrected_count = 0
    for row in rows:
        if not str(row["model_id"]).startswith("PICLas"):
            continue
        qoi = str(row["qoi"])
        exponent = 1 if qoi in LINEAR_QOIS else 2 if qoi in SQUARED_QOIS else 0
        if exponent:
            row["value"] = repr(float(row["value"]) * factor**exponent)
            corrected_count += 1

    output_path = case_dir / "model_evaluations_reference_area_corrected.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    backward_case = "backward" in case_dir.name.lower()
    report = {
        "source_csv": source_path.name,
        "corrected_csv": output_path.name,
        "surface_archive": str(surface_path.relative_to(framework_root)),
        "legacy_reference_area_m2": legacy_area,
        "configured_reference_area_m2": target_area,
        "piclas_coefficient_factor": factor,
        "corrected_value_count": corrected_count,
        "raw_archive_modified": False,
        "requires_solver_rerun": backward_case,
        "note": (
            "Backward shell PICLas values remain invalid because the legacy run injected particles "
            "at the wrong domain boundary; rerun with piclas_inflow_boundary_index=2."
            if backward_case
            else "PICLas values were deterministically re-normalized; solver rerun is not required."
        ),
    }
    report_path = case_dir / "reference_area_correction.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"case": case_dir.name, **report}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Directory containing Paper-1 verification case directories.",
    )
    args = parser.parse_args()
    framework_root = Path(__file__).resolve().parents[1]
    case_dirs = sorted(
        path.parent
        for path in args.campaign_root.glob("*/model_evaluations.csv")
        if (path.parent / "config_snapshot.json").is_file()
    )
    if not case_dirs:
        raise SystemExit(f"No verification cases found below {args.campaign_root}")
    reports = [reprocess_case(path, framework_root) for path in case_dirs]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
