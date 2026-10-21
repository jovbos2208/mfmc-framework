#!/usr/bin/env python3
"""Run one deterministic ADBSat-CLL anchor and archive force diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/vleo_cll/adbsat_anchor_audit"))
    parser.add_argument("--run-id", type=int, default=910001)
    parser.add_argument("--alpha-n", type=float, default=0.5)
    parser.add_argument("--sigma-t", type=float, default=0.5)
    args = parser.parse_args()

    root = args.framework_root.resolve()
    adbsat = root / "ADBSat-PyVersion"
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(adbsat))
    import simulate  # type: ignore

    payload = {
        "environment_model": "csv",
        "geometry_id": "cylinder_hex_scale_0p1_l1",
        "lf_model": "cylinder_hex_scale_0p1",
        "flow_zero_direction": [1.0, 0.0, 0.0],
        "aos_deg": 0.0,
        "aoa_deg": 0.0,
        "adbsat_aos_offset_deg": 0.0,
        "cll_alpha_n": float(args.alpha_n),
        "cll_sigma_t": float(args.sigma_t),
        "wall_temperature_k": 300.0,
        "relative_speed_mps": 7718.371,
        "atmosphere_row": [
            2.65669645265334e-11,
            0.0,
            0.0,
            1.0e15,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            961.0,
        ],
        # Intentionally no reference_area_m2: exercise A_total/2 fallback.
        "write_input_audit": True,
        "write_panel_surface_field": True,
        "write_mat": False,
        "surface_sample_id": "cll_anchor_alpha0p5_sigma0p5",
    }
    coeffs, runtime_ms = simulate.run_simulation(
        "CLL", args.run_id, 313.0, 0.0, str(adbsat), env_payload_path=payload
    )

    result_root = adbsat / "MFMC_Jobs_CLL"
    audit_path = result_root / f"input_audit_CLL_{args.run_id}.json"
    field_path = result_root / "surface_fields" / f"CLL_{args.run_id}.npz"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_path = root / "outputs/vleo_geometry/cylinder_hex_scale_0p1/cylinder_hex_scale_0p1.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_area_total = float(manifest["validation"]["surface_area_m2"])
    with np.load(field_path, allow_pickle=False) as field:
        area = np.asarray(field["panel_area"], dtype=float)
        flow = np.asarray(field["u_hat_inf"], dtype=float).reshape(3)
        area_ref = float(np.asarray(field["A_ref"], dtype=float).reshape(-1)[0])
        cd = float(np.asarray(field["C_D"], dtype=float).reshape(-1)[0])
        normal_cd_signed = float(np.asarray(field["normal_C_D_signed"], dtype=float).reshape(-1)[0])
        tangential_cd_signed = float(np.asarray(field["tangential_C_D_signed"], dtype=float).reshape(-1)[0])
        signed_cd_sum = normal_cd_signed + tangential_cd_signed
        if not np.isclose(abs(signed_cd_sum), cd, rtol=1.0e-10, atol=1.0e-12):
            raise RuntimeError(
                "ADBSat wind-axis force decomposition does not reconstruct C_D: "
                f"abs({normal_cd_signed} + {tangential_cd_signed}) != {cd}"
            )
        projected_area = 0.5 * float(np.sum(np.abs(np.sum(field["panel_normal"] * flow, axis=1)) * area))
        actual_area_total = float(area.sum())
        if not np.isclose(actual_area_total, expected_area_total, rtol=1.0e-6, atol=1.0e-12):
            raise RuntimeError(
                "ADBSat loaded the wrong or incorrectly scaled geometry: "
                f"surface area is {actual_area_total:.16g} m^2, but the canonical "
                f"geometry manifest requires {expected_area_total:.16g} m^2. "
                f"Resolved geometry model: {audit.get('geometry_model', 'unknown')!r}."
            )
        report = {
            "run_id": args.run_id,
            "alpha_n": args.alpha_n,
            "sigma_t": args.sigma_t,
            "reference_area_was_injected": "reference_area_m2" in payload,
            "geometry_model": audit.get("geometry_model"),
            "geometry_manifest": str(manifest_path),
            "expected_area_total_m2": expected_area_total,
            "area_total_m2": actual_area_total,
            "area_ref_m2": area_ref,
            "half_total_area_m2": 0.5 * float(area.sum()),
            "projected_area_m2": projected_area,
            "u_hat_inf": flow.tolist(),
            "c_d_half_total": cd,
            "drag_area_m2": cd * area_ref,
            "c_d_projected": cd * area_ref / projected_area,
            "normal_c_d_contribution_signed": normal_cd_signed,
            "tangential_c_d_contribution_signed": tangential_cd_signed,
            "shadowed_panel_count": int(np.asarray(field["shadowed_panel_count"]).reshape(-1)[0]),
            "panel_count": int(area.size),
            "runtime_ms": float(runtime_ms),
            "input_audit": audit,
            "raw_input_audit": str(audit_path),
            "raw_surface_field": str(field_path),
        }
    report_path = output / "adbsat_cll_anchor_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(json.dumps({key: report[key] for key in (
        "area_ref_m2", "half_total_area_m2", "projected_area_m2", "u_hat_inf",
        "c_d_half_total", "c_d_projected", "drag_area_m2",
        "normal_c_d_contribution_signed", "tangential_c_d_contribution_signed",
        "shadowed_panel_count", "panel_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
