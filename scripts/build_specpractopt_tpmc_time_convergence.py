#!/usr/bin/env python3
"""Build a fixed-condition PICLas TPMC end-time convergence suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_END_TIMES_S = (2.5e-5, 5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4)
DEFAULT_SEEDS = tuple(20260930 + index for index in range(5))
MANUAL_TIMESTEP_S = 1.0e-7
SAMPLING_FRACTION = 0.25


def _time_token(value: float) -> str:
    return f"tend_{value:.1e}".replace(".", "p").replace("+", "")


def build_workflow_config(t_end_s: float, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict:
    if not math.isfinite(t_end_s) or t_end_s <= 0.0:
        raise ValueError("t_end_s must be finite and positive")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct random seeds are required")
    time_steps = int(round(t_end_s / MANUAL_TIMESTEP_S))
    if not math.isclose(time_steps * MANUAL_TIMESTEP_S, t_end_s, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("Each end time must be an integer multiple of the manual timestep")
    sampling_iterations = max(1, int(round(SAMPLING_FRACTION * time_steps)))
    token = _time_token(t_end_s)
    atmosphere_row = [
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
    ]
    samples = [
        {
            "database_index": 0,
            "aos_deg": 45.0,
            "aoa_deg": 0.0,
            "energy_accommodation": 0.5,
            "wall_temperature_k": 300.0,
            "random_seed": int(seed),
        }
        for seed in seeds
    ]
    return {
        "schema_version": 1,
        "similarity": {
            "linear_scale": 1.0,
            "density_scale": 1.0,
            "basis": "Unscaled SpecPractOpt geometry and fixed physical environment",
        },
        "convergence_level": {
            "id": token,
            "t_end_s": float(t_end_s),
            "manual_timestep_s": MANUAL_TIMESTEP_S,
            "time_steps": time_steps,
            "sampling_iterations": sampling_iterations,
            "sampling_fraction": SAMPLING_FRACTION,
        },
        "adapter": {
            "model_id": "PICLas_TPMC",
            "fidelity": "hf",
            "available_qois": ["C_D", "C_L"],
            "kwargs": {
                "simulator_module": "PICLas_prandtl",
                "piclas_dir": "piclas",
                "update_dir": "update_parameter_file",
                "update_script": "python update_parameter.py",
                "piclas_mode": "tpmc",
                "collision_mode": 0,
                "calc_quality_factors": False,
                "mpi_procs": 64,
                "submission_group_size": len(seeds),
                "payload_defaults": {
                    "momentum_accommodation": 1.0,
                    "t_end_s": float(t_end_s),
                    "manual_timestep_s": MANUAL_TIMESTEP_S,
                    "sampling_iterations": sampling_iterations,
                },
            },
        },
        "request": {
            "study_id": "specpractopt_cylinder_hex_tpmc_time_convergence",
            "cell_id": token,
            "qois": ["C_D", "C_L"],
            "seed": int(seeds[0]),
            "geometry": {
                "id": "SpecPractOpt_cylinder_hex",
                "name": "SpecPractOpt cylinder hex",
                "characteristic_length": 0.35902899,
                "geometry_class": "cylinder",
                "metadata": {
                    "hf_mesh": "SpecPractOpt_cylinder_hex_mesh.h5",
                    "piclas_object_boundary_name": "OBJ",
                    "reference_area_m2": 0.2054622883746301,
                },
            },
            "regime": {
                "id": "AO_1E15_FIXED_AOS_P045_AOA_P000",
                "label": "Fixed pure atomic oxygen at 1e15 m^-3; AoS=45 deg; AoA=0 deg",
                "descriptors": {
                    "altitude_km": 313.0,
                    "number_density_m3": 1.0e15,
                    "freestream_temperature": 961.0,
                    "relative_speed_mps": 7718.371,
                    "surface_state": "nominal",
                },
            },
            "active_source_blocks": [],
            "sample_ids": [f"seed-{seed}" for seed in seeds],
            "samples": samples,
            "metadata": {
                "environment_model": "csv",
                "atmosphere_row": atmosphere_row,
                "relative_speed_mps": 7718.371,
                "flow_zero_direction": [1.0, 0.0, 0.0],
                "wind_enu_mps": [0.0, 0.0, 0.0],
                "use_winds": False,
                "apply_wind_to_speed": False,
                "aos_deg": 45.0,
                "aoa_deg": 0.0,
                "surface_state": "nominal",
                "case_name": f"specpractopt_cylinder_hex_tpmc_{token}",
            },
        },
    }


def build_suite(output_dir: Path, end_times_s: tuple[float, ...], seeds: tuple[int, ...]) -> dict:
    if sorted(set(end_times_s)) != list(end_times_s):
        raise ValueError("End times must be unique and strictly increasing")
    output_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path.cwd().resolve()
    levels = []
    for t_end_s in end_times_s:
        config = build_workflow_config(t_end_s, seeds)
        token = config["convergence_level"]["id"]
        config_path = output_dir / f"{token}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        try:
            portable_config_path = str(config_path.resolve().relative_to(repository_root))
        except ValueError:
            portable_config_path = str(config_path.resolve())
        levels.append(
            {
                **config["convergence_level"],
                "workflow_config": portable_config_path,
                "seeds": list(seeds),
            }
        )
    suite = {
        "schema_version": 1,
        "study_id": "specpractopt_cylinder_hex_tpmc_time_convergence",
        "purpose": "Numerical convergence of fixed-condition TPMC coefficients with simulation end time",
        "common_random_numbers": True,
        "levels": levels,
    }
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return {"suite_path": str(suite_path.resolve()), **suite}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/studies/specpractopt/tpmc_time_convergence"),
    )
    parser.add_argument("--end-times", type=float, nargs="+", default=list(DEFAULT_END_TIMES_S))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    result = build_suite(args.output_dir.resolve(), tuple(args.end_times), tuple(args.seeds))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
