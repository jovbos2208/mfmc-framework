from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .geometry_design import VARIABLES
from .parametric_geometry import CylinderHexSpec, build_geometry_assets, generate_cylinder_hex, validate_surface


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _point_pareto_indices(values: np.ndarray) -> list[int]:
    return [
        index for index in range(len(values))
        if not np.any(np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1))
    ]


def generate_local_refinement_manifest(
    design_manifest_json: str | Path,
    mfmc_details_json: str | Path,
    output_json: str | Path,
    *,
    count: int = 4,
    normalized_radius: float = 0.08,
    seed: int = 20260901,
) -> Dict[str, Any]:
    """Generate bounded, validated local candidates without fitting a geometry surrogate."""
    if count < 1 or not 0.0 < normalized_radius <= 0.25:
        raise ValueError("count must be positive and normalized_radius must be in (0, 0.25]")
    design = _read_json(design_manifest_json)
    details = _read_json(mfmc_details_json)
    rows = list(details["geometries"])
    design_by_id = {str(row["geometry_id"]): row for row in design["designs"]}
    missing = [row["geometry_id"] for row in rows if row["geometry_id"] not in design_by_id]
    if missing:
        raise ValueError(f"MFMC geometries are absent from the design manifest: {missing}")
    objectives = np.asarray([[float(row["mean_drag"]), float(row["std_drag"])] for row in rows])
    span = np.ptp(objectives, axis=0)
    span[span <= 1.0e-30] = 1.0
    normalized_objectives = (objectives - np.min(objectives, axis=0)) / span
    pareto = _point_pareto_indices(objectives)
    center_indices = [
        int(np.argmin(objectives[:, 0])),
        int(np.argmin(np.mean(normalized_objectives, axis=1))),
        min(pareto, key=lambda index: (float(np.linalg.norm(normalized_objectives[index])), index)),
    ]
    center_roles = ("nominal_mean_optimum", "robust_optimum", "pareto_knee")
    centers: list[tuple[int, str]] = []
    for index, role in zip(center_indices, center_roles):
        if all(existing[0] != index for existing in centers):
            centers.append((index, role))
    bounds = np.asarray([design["bounds"][name] for name in VARIABLES], dtype=float)
    existing_points = np.asarray([
        [float(row[f"normalized_{name}"]) for name in VARIABLES] for row in design["designs"]
    ])
    rng = np.random.default_rng(seed)
    directions = np.vstack([np.eye(len(VARIABLES)), -np.eye(len(VARIABLES))])
    rng.shuffle(directions)
    accepted: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    sequence = 0
    while len(accepted) < count and sequence < max(64, count * 20):
        center_index, center_role = centers[sequence % len(centers)]
        center_id = str(rows[center_index]["geometry_id"])
        center = np.asarray([
            float(design_by_id[center_id][f"normalized_{name}"]) for name in VARIABLES
        ])
        direction = directions[(sequence // len(centers)) % len(directions)]
        shell = 1 + (sequence // (len(centers) * len(directions)))
        point = np.clip(center + normalized_radius * shell * direction, 0.0, 1.0)
        sequence += 1
        reason = None
        if np.min(np.linalg.norm(existing_points - point, axis=1)) <= 1.0e-10:
            reason = "duplicates_existing_design"
        elif accepted and min(
            np.linalg.norm(np.asarray(row["normalized_parameters"]) - point) for row in accepted
        ) <= 1.0e-10:
            reason = "duplicates_local_candidate"
        physical = bounds[:, 0] + point * (bounds[:, 1] - bounds[:, 0])
        spec = CylinderHexSpec(**dict(zip(VARIABLES, map(float, physical))))
        validation: Mapping[str, Any] = {}
        if reason is None:
            try:
                surface, _derived = generate_cylinder_hex(spec)
                validation = validate_surface(surface, spec)
                if not validation["valid"]:
                    reason = "surface_validation_failed"
            except ValueError as exc:
                reason = f"invalid_geometry:{exc}"
        if reason is not None:
            rejected.append({"center_geometry_id": center_id, "reason": reason})
            continue
        geometry_id = f"cylinder_hex_local_{len(accepted):03d}"
        accepted.append({
            "geometry_id": geometry_id,
            "role": "local_optimization_candidate",
            "eligible_for_model_fitting": True,
            "center_geometry_id": center_id,
            "center_role": center_role,
            "normalized_parameters": [float(value) for value in point],
            "parameters": {name: float(value) for name, value in zip(VARIABLES, physical)},
            "validation": validation,
            **{name: float(value) for name, value in zip(VARIABLES, physical)},
            **{f"normalized_{name}": float(value) for name, value in zip(VARIABLES, point)},
        })
    if len(accepted) != count:
        raise ValueError(f"Could generate only {len(accepted)} of {count} valid, unique local candidates")
    target = Path(output_json).resolve()
    uniform_scale = float(design.get("uniform_linear_scale_factor", 0.1))
    for candidate in accepted:
        geometry_id = str(candidate["geometry_id"])
        specification = CylinderHexSpec(**candidate["parameters"]).scaled(uniform_scale)
        asset_dir = target.parent / "designs" / geometry_id
        assets = build_geometry_assets(
            asset_dir, specification, design_id=geometry_id,
            uniform_scale_factor=uniform_scale, body_mesh_size_m=0.003, farfield_mesh_size_m=0.015,
        )
        manifest_path = Path(assets["manifest_json"])
        candidate.update({
            "manifest_path": str(manifest_path.relative_to(target.parent)),
            "design_fingerprint": assets["design_fingerprint"],
            "mesh_fingerprint": assets["mesh_fingerprint"],
            "reference_area_m2": float(assets["reference_area"]["value_m2"]),
        })
    result = {
        "schema_version": 1,
        "method": "bounded_local_pattern_search_without_geometry_surrogate",
        "seed": int(seed),
        "normalized_radius": float(normalized_radius),
        "variables": list(VARIABLES),
        "bounds": design["bounds"],
        "source_centers": [{"geometry_id": rows[index]["geometry_id"], "role": role} for index, role in centers],
        "validation_geometry_ids_excluded": sorted(map(str, design["validation_geometry_ids"])),
        "candidates": accepted,
        "designs": accepted,
        "baseline_geometry_id": accepted[0]["geometry_id"],
        "validation_geometry_ids": [],
        "uniform_linear_scale_factor": uniform_scale,
        "rejected_candidates": rejected,
        "evaluation_contract": {
            "target_model": "PICLas_TPMC",
            "control_model": "Sentman",
            "budget_hf_equivalent_per_geometry": 20.0,
            "mpi_processes": 64,
            "dsmc_during_optimization": False,
        },
    }
    target = _write_json(target, result)
    return {**result, "output_json": str(target)}


def select_dsmc_finalists(
    design_manifest_json: str | Path,
    mfmc_details_json: str | Path,
    output_json: str | Path,
    *,
    maximum_finalists: int = 5,
    baseline_geometry_id: str | None = None,
) -> Dict[str, Any]:
    if not 3 <= maximum_finalists <= 5:
        raise ValueError("maximum_finalists must be between three and five")
    design = _read_json(design_manifest_json)
    rows = list(_read_json(mfmc_details_json)["geometries"])
    by_id = {str(row["geometry_id"]): row for row in rows}
    baseline_id = str(
        baseline_geometry_id
        if baseline_geometry_id is not None
        else design["baseline_geometry_id"]
    )
    if baseline_id not in by_id:
        raise ValueError("Baseline must have a completed optimization estimate before finalization")
    objectives = np.asarray([[float(row["mean_drag"]), float(row["std_drag"])] for row in rows])
    span = np.ptp(objectives, axis=0)
    span[span <= 1.0e-30] = 1.0
    scaled = (objectives - np.min(objectives, axis=0)) / span
    pareto = _point_pareto_indices(objectives)
    requested = [
        (baseline_id, "baseline"),
        (str(rows[int(np.argmin(objectives[:, 0]))]["geometry_id"]), "minimum_mean_drag"),
        (str(rows[int(np.argmin(np.mean(scaled, axis=1)))]["geometry_id"]), "robust_optimum"),
        (str(rows[min(pareto, key=lambda index: (float(np.linalg.norm(scaled[index])), index))]["geometry_id"]), "pareto_knee"),
    ]
    if pareto:
        edge = max(pareto, key=lambda index: (float(np.linalg.norm(scaled[index])), -index))
        requested.append((str(rows[edge]["geometry_id"]), "pareto_edge"))
    finalists: list[Dict[str, Any]] = []
    for geometry_id, role in requested:
        if geometry_id not in {row["geometry_id"] for row in finalists} and len(finalists) < maximum_finalists:
            finalists.append({"geometry_id": geometry_id, "validation_role": role})
    result = {
        "schema_version": 1,
        "optimization_closed": True,
        "dsmc_results_must_not_feed_optimization": True,
        "mpi_processes": 64,
        "common_random_numbers_required": True,
        "finalists": finalists,
    }
    target = _write_json(output_json, result)
    return {**result, "output_json": str(target)}


def build_dsmc_validation_report(
    bundle_json: str | Path,
    finalists_json: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_repeats: int = 2000,
    seed: int = 20260901,
) -> Dict[str, Any]:
    bundle = _read_json(bundle_json)
    plan = _read_json(finalists_json)
    evaluations = bundle["evaluations"]
    output_rows: list[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for finalist in plan["finalists"]:
        geometry_id = str(finalist["geometry_id"])
        model_rows: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        for model in ("PICLas_TPMC", "PICLas_DSMC"):
            model_rows[model] = {
                str(row["canonical_sample_id"]): row for row in evaluations
                if str(row["geometry_id"]) == geometry_id and str(row["model_id"]) == model
            }
        common = sorted(set(model_rows["PICLas_TPMC"]) & set(model_rows["PICLas_DSMC"]))
        if len(common) < 2:
            raise ValueError(f"{geometry_id} needs at least two paired TPMC/DSMC CRN states")
        tpmc = np.asarray([float(model_rows["PICLas_TPMC"][sample]["drag_area_m2"]) for sample in common])
        dsmc = np.asarray([float(model_rows["PICLas_DSMC"][sample]["drag_area_m2"]) for sample in common])
        mean_delta = np.empty(bootstrap_repeats)
        std_delta = np.empty(bootstrap_repeats)
        for repeat in range(bootstrap_repeats):
            indices = rng.integers(0, len(common), len(common))
            mean_delta[repeat] = np.mean(dsmc[indices]) - np.mean(tpmc[indices])
            std_delta[repeat] = np.std(dsmc[indices], ddof=1) - np.std(tpmc[indices], ddof=1)
        output_rows.append({
            **finalist,
            "n_common_crn": len(common),
            "common_sample_ids": common,
            "tpmc_mean_drag": float(np.mean(tpmc)),
            "dsmc_mean_drag": float(np.mean(dsmc)),
            "mean_dsmc_minus_tpmc": float(np.mean(dsmc) - np.mean(tpmc)),
            "mean_difference_ci95": [float(value) for value in np.quantile(mean_delta, [0.025, 0.975])],
            "tpmc_std_drag": float(np.std(tpmc, ddof=1)),
            "dsmc_std_drag": float(np.std(dsmc, ddof=1)),
            "std_dsmc_minus_tpmc": float(np.std(dsmc, ddof=1) - np.std(tpmc, ddof=1)),
            "std_difference_ci95": [float(value) for value in np.quantile(std_delta, [0.025, 0.975])],
        })
    tpmc_rank = [row["geometry_id"] for row in sorted(output_rows, key=lambda row: (row["tpmc_mean_drag"], row["geometry_id"]))]
    dsmc_rank = [row["geometry_id"] for row in sorted(output_rows, key=lambda row: (row["dsmc_mean_drag"], row["geometry_id"]))]
    tpmc_std_rank = [row["geometry_id"] for row in sorted(output_rows, key=lambda row: (row["tpmc_std_drag"], row["geometry_id"]))]
    dsmc_std_rank = [row["geometry_id"] for row in sorted(output_rows, key=lambda row: (row["dsmc_std_drag"], row["geometry_id"]))]
    tpmc_objectives = np.asarray([[row["tpmc_mean_drag"], row["tpmc_std_drag"]] for row in output_rows])
    dsmc_objectives = np.asarray([[row["dsmc_mean_drag"], row["dsmc_std_drag"]] for row in output_rows])
    tpmc_span = np.ptp(tpmc_objectives, axis=0)
    dsmc_span = np.ptp(dsmc_objectives, axis=0)
    tpmc_span[tpmc_span <= 1.0e-30] = 1.0
    dsmc_span[dsmc_span <= 1.0e-30] = 1.0
    tpmc_score = np.mean((tpmc_objectives - np.min(tpmc_objectives, axis=0)) / tpmc_span, axis=1)
    dsmc_score = np.mean((dsmc_objectives - np.min(dsmc_objectives, axis=0)) / dsmc_span, axis=1)
    tpmc_robust_rank = [output_rows[index]["geometry_id"] for index in np.argsort(tpmc_score)]
    dsmc_robust_rank = [output_rows[index]["geometry_id"] for index in np.argsort(dsmc_score)]
    result = {
        "schema_version": 1,
        "status": "tpmc_optimum_confirmed" if tpmc_robust_rank[0] == dsmc_robust_rank[0] else "tpmc_optimum_not_confirmed",
        "optimization_updated_from_dsmc": False,
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_seed": int(seed),
        "tpmc_mean_rank": tpmc_rank,
        "dsmc_mean_rank": dsmc_rank,
        "tpmc_std_rank": tpmc_std_rank,
        "dsmc_std_rank": dsmc_std_rank,
        "tpmc_robust_rank": tpmc_robust_rank,
        "dsmc_robust_rank": dsmc_robust_rank,
        "rank_stability": {
            "mean": tpmc_rank == dsmc_rank,
            "std": tpmc_std_rank == dsmc_std_rank,
            "robust": tpmc_robust_rank == dsmc_robust_rank,
        },
        "geometries": output_rows,
    }
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = _write_json(root / "dsmc_validation_report.json", result)
    csv_path = root / "dsmc_validation_report.csv"
    flat_rows = [{key: value for key, value in row.items() if key != "common_sample_ids"} for row in output_rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return {**result, "report_json": str(json_path), "report_csv": str(csv_path)}
