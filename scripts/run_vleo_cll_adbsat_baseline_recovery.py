#!/usr/bin/env python3
"""Recompute only the cheap baseline ADBSat-CLL branch from archived pilot samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--sample-inputs",
        type=Path,
        default=Path("outputs/vleo_cll/cylinder_hex_pilot_20/sample_inputs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vleo_cll/baseline_adbsat_local_recovery"),
    )
    args = parser.parse_args()
    root = args.framework_root.resolve()
    source = args.sample_inputs if args.sample_inputs.is_absolute() else root / args.sample_inputs
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    with source.open(newline="", encoding="utf-8") as handle:
        raw = [
            row for row in csv.DictReader(handle)
            if row["qoi"] == "C_D" and row["model_id"].startswith("PICLas")
        ]
    raw.sort(key=lambda row: int(row["sample_index"]))
    if len(raw) != 20 or len({row["sample_id"] for row in raw}) != 20:
        raise ValueError("Expected exactly 20 unique baseline C_D sample rows")

    adbsat = root / "ADBSat-PyVersion"
    sys.path.insert(0, str(adbsat))
    import simulate  # type: ignore

    records = []
    for index, row in enumerate(raw):
        payload = {
            "environment_model": "csv",
            "geometry_id": "cylinder_hex_scale_0p1_l1",
            "lf_model": "cylinder_hex_scale_0p1",
            "flow_zero_direction": [1.0, 0.0, 0.0],
            "aos_deg": 0.0,
            "aoa_deg": 0.0,
            "adbsat_aos_offset_deg": 0.0,
            "cll_alpha_n": float(row["input__cll_alpha_n"]),
            "cll_sigma_t": float(row["input__cll_sigma_t"]),
            "wall_temperature_k": 300.0,
            "relative_speed_mps": 7718.371,
            "atmosphere_row": [
                2.65669645265334e-11, 0.0, 0.0, 1.0e15, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 961.0,
            ],
            "write_input_audit": True,
            "write_panel_surface_field": False,
            "write_mat": False,
            "surface_sample_id": row["sample_id"],
        }
        coeffs, runtime_ms = simulate.run_simulation(
            "CLL", 920000 + index, 313.0, 0.0, str(adbsat), env_payload_path=payload
        )
        c_d = float(coeffs["C_D"] if isinstance(coeffs, dict) else coeffs[0])
        records.append(
            {
                "sample_id": row["sample_id"],
                "sample_index": int(row["sample_index"]),
                "cll_alpha_n": payload["cll_alpha_n"],
                "cll_sigma_t": payload["cll_sigma_t"],
                "C_D": c_d,
                "C_D2": c_d ** 2,
                "runtime_ms": float(runtime_ms),
                "run_id": 920000 + index,
            }
        )

    csv_path = output / "adbsat_baseline_recovery.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema_version": 1,
        "source_sample_inputs": str(source),
        "source_sample_inputs_sha256": sha256(source),
        "geometry_model": "cylinder_hex_scale_0p1",
        "reference_area_convention": "A_total/2",
        "n_samples": len(records),
        "mean_c_d": sum(row["C_D"] for row in records) / len(records),
        "csv": str(csv_path),
    }
    manifest_path = output / "adbsat_baseline_recovery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
