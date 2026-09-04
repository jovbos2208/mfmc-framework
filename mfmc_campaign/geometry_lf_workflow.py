from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .adapters import LegacyADBSatAdapter, make_request


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_lf_campaign(
    design_manifest: str | Path,
    source_sample_inputs_csv: str | Path,
    output_config: str | Path,
    *,
    n_samples: int = 256,
    adbsat_runtime_dir: str | Path = "ADBSat-PyVersion",
) -> Dict[str, Any]:
    design_path = Path(design_manifest).resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    with Path(source_sample_inputs_csv).resolve().open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    preferred = [row for row in source_rows if row.get("qoi") == "C_D" and row.get("phase") == "prod_lf_full"]
    if not preferred:
        preferred = [row for row in source_rows if row.get("qoi") == "C_D"] or source_rows
    unique: Dict[str, Dict[str, Any]] = {}
    for row in preferred:
        key = row.get("sample_fingerprint") or row.get("sample_id") or str(len(unique))
        unique.setdefault(key, row)
    if len(unique) < n_samples:
        raise ValueError(f"Only {len(unique)} unique archived uncertainty samples are available; requested {n_samples}")
    ordered = sorted(unique.values(), key=lambda row: (row.get("sample_fingerprint", ""), row.get("sample_id", "")))
    # Even spacing avoids taking one contiguous repetition/budget block while
    # retaining the exact archived physical draws and common random numbers.
    indices = np.linspace(0, len(ordered) - 1, n_samples, dtype=int)
    samples: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    for position, index in enumerate(indices):
        row = ordered[int(index)]
        sample = {
            key.removeprefix("input__"): float(value)
            for key, value in row.items()
            if key.startswith("input__") and value not in (None, "")
        }
        sample["database_index"] = 0
        sample["density_scale"] = float(design["density_scale_required_for_kn_similarity"])
        samples.append(sample)
        sample_ids.append(f"wp1-crn-{position:04d}")
    config = {
        "schema_version": 1,
        "study_id": "vleo_cylinder_hex_wp5_lf",
        "design_manifest": str(design_path),
        "source_sample_inputs_csv": str(Path(source_sample_inputs_csv).resolve()),
        "adbsat_runtime_dir": str(Path(adbsat_runtime_dir).resolve()),
        "method": "Sentman",
        "qois": ["C_D"],
        "sample_ids": sample_ids,
        "samples": samples,
        "regime": {
            "id": "CUBE_300KM",
            "label": "Cylinder-hex representative circular orbit at 300 km",
            "descriptors": {
                "altitude_km": 300.0,
                "datetime_utc": "2013-10-21T00:00:00Z",
                "lat_deg": 0.0,
                "lon_deg": 0.0,
                "f107": 130.0,
                "f107a": 130.0,
                "ap": 18.0,
                "surface_state": "proxy_bounded",
            },
        },
        "metadata": {
            "environment_model": "pymsis_hwm14",
            "flow_zero_direction": [1.0, 0.0, 0.0],
            "density_scale": float(design["density_scale_required_for_kn_similarity"]),
            "use_winds": True,
            "apply_wind_to_speed": True,
        },
        "provenance": {
            "uncertainty_draws": "exact archived WP1 samples, deterministically thinned",
            "common_random_numbers_across_geometries": True,
            "validation_policy": "validation-role geometries are not evaluated during LF training or HF acquisition",
        },
    }
    target = Path(output_config).resolve()
    _write_json(target, config)
    return {"config": str(target), "n_samples": n_samples, "n_geometries": design["n_designs"]}


def _stage_geometry(design_root: Path, row: Mapping[str, Any], runtime: Path) -> None:
    geometry_id = str(row["geometry_id"])
    manifest_path = design_root / str(row["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_dir = manifest_path.parent
    targets = (
        (source_dir / manifest["assets"]["adbsat_obj"], runtime / "inou" / "obj_files" / f"{geometry_id}.obj"),
        (source_dir / manifest["assets"]["adbsat_mat"], runtime / "inou" / "models" / f"{geometry_id}.mat"),
    )
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def lf_campaign_preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    design_path = Path(str(config["design_manifest"]))
    runtime = Path(str(config["adbsat_runtime_dir"]))
    checks = {
        "design_manifest": design_path.is_file(),
        "adbsat_simulate": (runtime / "simulate.py").is_file(),
        "adbsat_obj_dir": (runtime / "inou" / "obj_files").is_dir(),
        "adbsat_model_dir": (runtime / "inou" / "models").is_dir(),
        "samples_present": bool(config.get("samples")),
    }
    return {"ready": all(checks.values()), "checks": checks, "n_samples": len(config.get("samples", []))}


def run_lf_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    execute: bool,
    simulator_module: str | None = None,
) -> Dict[str, Any]:
    config = json.loads(Path(config_path).resolve().read_text(encoding="utf-8"))
    design_path = Path(config["design_manifest"]).resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    report = lf_campaign_preflight(config)
    if not report["ready"]:
        raise RuntimeError(f"LF campaign preflight failed: {report['checks']}")
    target = Path(output_dir).resolve()
    if not execute:
        return {"status": "dry_run", "output_dir": str(target), "n_geometries": design["n_designs"], **report}
    runtime = Path(config["adbsat_runtime_dir"]).resolve()
    selected_simulator_module = str(
        simulator_module or config.get("simulator_module", "ADBSat")
    ).strip()
    if not selected_simulator_module:
        raise ValueError("ADBSat simulator_module must be non-empty")
    adapter = LegacyADBSatAdapter(
        "Sentman", str(config["method"]), list(config["qois"]),
        {"simulator_module": selected_simulator_module, "base_dir": str(runtime)},
    )
    state_path = target / "lf_campaign_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"schema_version": 1, "completed": []}
    completed = set(state["completed"])
    all_rows: List[Dict[str, Any]] = []
    results_csv = target / "lf_results.csv"
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
    training_designs = [row for row in design["designs"] if row["eligible_for_model_fitting"]]
    for row in training_designs:
        geometry_id = row["geometry_id"]
        if geometry_id in completed:
            continue
        _stage_geometry(design_path.parent, row, runtime)
        geometry = {
            "id": geometry_id,
            "name": geometry_id,
            "characteristic_length": 0.1,
            "geometry_class": "parametric_cylinder_hex",
            "metadata": {"lf_model": geometry_id, "reference_area_m2": row["reference_area_m2"]},
        }
        request = make_request(
            study_id=config["study_id"], cell_id=f"lf_{geometry_id}", model_id="Sentman", fidelity="lf",
            qois=list(config["qois"]), geometry=geometry, regime=dict(config["regime"]),
            active_source_blocks=[], sample_ids=list(config["sample_ids"]), samples=list(config["samples"]),
            seed=20260818, metadata=dict(config["metadata"]),
        )
        result = adapter.evaluate(request)
        area = float(row["reference_area_m2"])
        for sample_id, cd, cost in zip(result.sample_ids, result.values_by_qoi["C_D"], result.costs):
            all_rows.append({
                "geometry_id": geometry_id, "role": row["role"], "sample_id": sample_id,
                "C_D": float(cd), "drag_area_m2": float(cd) * area,
                "reference_area_m2": area,
                "reference_area_convention": "canonical_manifest_area",
                "cost_cpu_hours": float(cost),
            })
        target.mkdir(parents=True, exist_ok=True)
        with results_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
        completed.add(geometry_id)
        state["completed"] = sorted(completed)
        _write_json(state_path, state)
    metrics: List[Dict[str, Any]] = []
    for row in training_designs:
        values = np.asarray([float(item["drag_area_m2"]) for item in all_rows if item["geometry_id"] == row["geometry_id"]])
        metrics.append({
            "geometry_id": row["geometry_id"], "role": row["role"], "n": len(values),
            "mean_drag": float(np.mean(values)), "std_drag": float(np.std(values, ddof=1)),
            "q95_drag": float(np.quantile(values, 0.95)),
            "reference_area_convention": "canonical_manifest_area",
        })
    metrics_csv = target / "lf_robust_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    return {"status": "completed", "results_csv": str(results_csv), "metrics_csv": str(metrics_csv), "completed": len(completed)}
