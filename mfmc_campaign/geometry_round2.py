from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .geometry_design import VARIABLES
from .parametric_geometry import (
    CylinderHexSpec,
    build_geometry_assets,
    build_gmsh_exterior_mesh,
    build_piclas_hdf5_mesh,
    validate_piclas_hdf5_mesh,
)
from .selected_hf_geometry import L1_CONTROLS, _workflow_config, select_common_hf_samples
from .sparse_pce import SparsePCEModel


def _read_csv(path: str | Path) -> list[Dict[str, str]]:
    with Path(path).resolve().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scale(values: np.ndarray, *, smaller_is_better: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    span = float(np.ptp(array))
    scaled = np.zeros_like(array) if span <= 1.0e-14 else (array - np.min(array)) / span
    return 1.0 - scaled if smaller_is_better else scaled


def _model_inputs(
    input_names: Sequence[str], design: Mapping[str, Any], sample: Mapping[str, Any]
) -> list[float]:
    values: list[float] = []
    for name in input_names:
        if name.startswith("geometry__"):
            values.append(float(design[f"normalized_{name.removeprefix('geometry__')}"]))
        elif name.startswith("input__"):
            values.append(float(sample[name.removeprefix("input__")]))
        else:
            raise KeyError(f"Unsupported WP5 surrogate input: {name}")
    return values


def select_round2_geometries(
    design_manifest_json: str | Path,
    sentman_metrics_csv: str | Path,
    paired_bundle_json: str | Path,
    surrogate_manifest_json: str | Path,
    output_json: str | Path,
    *,
    count: int = 3,
    round_number: int = 2,
) -> Dict[str, Any]:
    """Select new training geometries from objective, coverage and disagreement diagnostics."""
    if round_number < 2:
        raise ValueError("Sequential geometry acquisition starts at round 2")
    design = json.loads(Path(design_manifest_json).resolve().read_text(encoding="utf-8"))
    bundle = json.loads(Path(paired_bundle_json).resolve().read_text(encoding="utf-8"))
    surrogate = json.loads(Path(surrogate_manifest_json).resolve().read_text(encoding="utf-8"))
    sentman = {row["geometry_id"]: row for row in _read_csv(sentman_metrics_csv)}
    current_ids = set(str(value) for value in bundle["selected_geometry_ids"])
    validation_ids = set(str(value) for value in design["validation_geometry_ids"])
    candidates = [
        row for row in design["designs"]
        if bool(row["eligible_for_model_fitting"])
        and row["geometry_id"] not in current_ids
        and row["geometry_id"] not in validation_ids
    ]
    if count < 1 or count > len(candidates):
        raise ValueError("Round-2 geometry count is outside the available training candidates")
    missing_sentman = [row["geometry_id"] for row in candidates if row["geometry_id"] not in sentman]
    if missing_sentman:
        raise ValueError(f"Sentman metrics are missing for candidates: {missing_sentman[:10]}")
    input_names = list(surrogate["input_names"])
    tpmc_model = SparsePCEModel.read_json(surrogate["models"]["tpmc"])
    delta_model = SparsePCEModel.read_json(surrogate["models"]["dsmc_minus_tpmc"])
    model_selection = dict(surrogate.get("model_selection", {}))
    selected_surrogate = str(model_selection.get("selected_surrogate", ""))
    if selected_surrogate not in {"lf_pce", "mf_pce"}:
        cv = {row["method"]: row for row in surrogate.get("geometry_held_out_summary", [])}
        lf_rmse = float(cv["lf_pce"]["mean_geometry_rmse"])
        mf_rmse = float(cv["mf_pce"]["mean_geometry_rmse"])
        selected_surrogate = "mf_pce" if (lf_rmse - mf_rmse) / lf_rmse >= 0.01 else "lf_pce"
    samples = bundle["uncertainty_samples"]
    sample_ids = sorted(samples)
    diagnostics: list[Dict[str, Any]] = []
    for row in candidates:
        x = np.asarray([_model_inputs(input_names, row, samples[sample_id]) for sample_id in sample_ids])
        prediction = tpmc_model.predict(x)
        if selected_surrogate == "mf_pce":
            prediction = prediction + delta_model.predict(x)
        diagnostics.append({
            "geometry_id": row["geometry_id"],
            "predicted_mean_drag": float(np.mean(prediction)),
            "predicted_std_drag": float(np.std(prediction, ddof=1)),
            "predicted_q95_drag": float(np.quantile(prediction, 0.95)),
            "sentman_mean_drag": float(sentman[row["geometry_id"]]["mean_drag"]),
            "sentman_std_drag": float(sentman[row["geometry_id"]]["std_drag"]),
            "sentman_q95_drag": float(sentman[row["geometry_id"]]["q95_drag"]),
        })
    candidate_points = np.asarray(
        [[float(row[f"normalized_{name}"]) for name in VARIABLES] for row in candidates]
    )
    existing_rows = [row for row in design["designs"] if row["geometry_id"] in current_ids]
    existing_points = np.asarray(
        [[float(row[f"normalized_{name}"]) for name in VARIABLES] for row in existing_rows]
    )
    predicted_q95 = np.asarray([row["predicted_q95_drag"] for row in diagnostics])
    sentman_q95 = np.asarray([row["sentman_q95_drag"] for row in diagnostics])
    disagreement = np.abs(_scale(predicted_q95) - _scale(sentman_q95))
    selected: list[int] = []
    bases: list[str] = []

    def add(index: int, basis: str) -> None:
        if index not in selected and len(selected) < count:
            selected.append(index)
            bases.append(basis)

    add(int(np.argmin(predicted_q95)), "predicted_robust_objective")
    reference = np.vstack([existing_points, candidate_points[selected]])
    distances = np.min(np.linalg.norm(candidate_points[:, None, :] - reference[None, :, :], axis=2), axis=1)
    distances[selected] = -np.inf
    add(int(np.argmax(distances)), "geometry_space_filling")
    available = [index for index in range(len(candidates)) if index not in selected]
    if available:
        add(max(available, key=lambda index: (float(disagreement[index]), -index)), "tpmc_sentman_disagreement")
    while len(selected) < count:
        reference = np.vstack([existing_points, candidate_points[selected]])
        distances = np.min(np.linalg.norm(candidate_points[:, None, :] - reference[None, :, :], axis=2), axis=1)
        desirability = _scale(predicted_q95, smaller_is_better=True)
        score = 0.5 * _scale(distances) + 0.3 * desirability + 0.2 * _scale(disagreement)
        score[selected] = -np.inf
        add(int(np.argmax(score)), "combined_objective_coverage_disagreement")

    rows = []
    by_id = {row["geometry_id"]: row for row in diagnostics}
    for order, (index, basis) in enumerate(zip(selected, bases)):
        geometry_id = candidates[index]["geometry_id"]
        rows.append({
            "selection_order": order,
            "selection_basis": basis,
            "geometry_id": geometry_id,
            "geometry_min_distance_to_existing": float(
                np.min(np.linalg.norm(candidate_points[index] - existing_points, axis=1))
            ),
            "normalized_tpmc_sentman_q95_disagreement": float(disagreement[index]),
            **by_id[geometry_id],
        })
    result = {
        "schema_version": 1,
        "round": int(round_number),
        "count": len(rows),
        "selected": rows,
        "previous_training_geometry_ids": sorted(current_ids),
        "untouched_validation_geometry_ids": sorted(validation_ids),
        "candidate_diagnostics": diagnostics,
        "surrogate_used_for_acquisition": selected_surrogate,
    }
    target = Path(output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "output_json": str(target)}


def _relative_to_repository(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError as exc:
        raise ValueError(f"Round-2 artifact must be inside repository: {path}") from exc


def _stage_generated_geometry(source_manifest_path: Path, geometry_dir: Path) -> Dict[str, Any]:
    """Copy an already generated non-analytical surface without recreating its nodes."""
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for asset_name in (
        "adbsat_obj",
        "adbsat_mat",
        "meshing_surface_stl",
        "canonical_surface_npz",
        "gmsh_exterior_geo",
    ):
        relative = source["assets"].get(asset_name)
        if not relative:
            raise ValueError(f"Generated geometry is missing required asset {asset_name}")
        origin = source_manifest_path.parent / str(relative)
        destination = geometry_dir / origin.name
        if origin.resolve() != destination.resolve():
            shutil.copy2(origin, destination)
        source["assets"][asset_name] = destination.name
    source["assets"]["piclas_volume_mesh"] = None
    source["assets"]["piclas_mesh_status"] = "requires independent Gmsh/PyHOPE volume-mesh stage"
    target = geometry_dir / source_manifest_path.name
    target.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**source, "manifest_json": str(target)}


def build_round2_piclas_suite(
    selection_json: str | Path,
    design_manifest_json: str | Path,
    lf_config_json: str | Path,
    *,
    output_root: str | Path = "piclas/geometry/cylinder_hex_wp5/L1",
    config_output_dir: str | Path = "configs/studies/cylinder_hex_wp5_round2",
    suite_output_json: str | Path = "outputs/cylinder_hex/wp5_round2/suite.json",
    base_config_json: str | Path = "configs/studies/cylinder_hex_piclas_adapter_l1_5seeds.json",
    n_dsmc: int = 5,
    n_tpmc: int = 90,
    gmsh_executable: str = "gmsh",
    pyhope_executable: str = "pyhope",
    round_number: int = 2,
    mpi_procs: int = 64,
    simulator_module: str | None = None,
) -> Dict[str, Any]:
    if round_number < 2:
        raise ValueError("Sequential geometry acquisition starts at round 2")
    if mpi_procs < 1:
        raise ValueError("mpi_procs must be positive")
    if simulator_module is not None and not simulator_module.strip():
        raise ValueError("simulator_module must be non-empty when provided")
    if n_dsmc < 0 or n_tpmc < 1 or n_tpmc < n_dsmc:
        raise ValueError("Sequential acquisition requires 0 <= n_dsmc <= n_tpmc and n_tpmc >= 1")
    selection = json.loads(Path(selection_json).resolve().read_text(encoding="utf-8"))
    design_path = Path(design_manifest_json).resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    lf_config = json.loads(Path(lf_config_json).resolve().read_text(encoding="utf-8"))
    base = json.loads(Path(base_config_json).resolve().read_text(encoding="utf-8"))
    validation = set(selection["untouched_validation_geometry_ids"])
    selected_ids = [row["geometry_id"] for row in selection["selected"]]
    if validation.intersection(selected_ids):
        raise ValueError("Sequential selection contains an untouched validation geometry")
    design_rows = {row["geometry_id"]: row for row in design["designs"]}
    sample_indices = select_common_hf_samples(lf_config["samples"], n_tpmc)
    dsmc_indices = sample_indices[:n_dsmc]
    repository_root = Path.cwd().resolve()
    piclas_root = (repository_root / "piclas").resolve()
    mesh_root = Path(output_root).resolve()
    config_root = Path(config_output_dir).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    for selection_row in selection["selected"]:
        geometry_id = selection_row["geometry_id"]
        design_row = design_rows[geometry_id]
        source_manifest_path = design_path.parent / design_row["manifest_path"]
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        geometry_dir = mesh_root / geometry_id
        if source_manifest.get("parameterization") == "symmetric_surface_control_nodes":
            manifest = _stage_generated_geometry(source_manifest_path, geometry_dir)
        else:
            manifest = build_geometry_assets(
                geometry_dir,
                CylinderHexSpec(**source_manifest["specification"]),
                design_id=geometry_id,
                uniform_scale_factor=float(source_manifest["provenance"]["uniform_linear_scale_factor"]),
                **L1_CONTROLS,
            )
        manifest_path = Path(manifest["manifest_json"])
        mesh_path = geometry_dir / f"{geometry_id}_mesh.h5"
        if mesh_path.is_file():
            h5 = validate_piclas_hdf5_mesh(mesh_path)
            if not h5["valid"]:
                raise ValueError(f"Existing PICLas mesh is invalid: {mesh_path}")
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            current["assets"]["piclas_volume_mesh"] = mesh_path.name
            manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
            gmsh = current.get("gmsh_exterior_mesh", {}).get("validation", {})
        else:
            gmsh = build_gmsh_exterior_mesh(manifest_path, gmsh_executable=gmsh_executable)
            h5 = build_piclas_hdf5_mesh(manifest_path, pyhope_executable=pyhope_executable)
            mesh_path = Path(h5["mesh_path"])
        hdf5_area = h5.get("x_projected_reference_area_m2")
        if hdf5_area is None:
            raise ValueError(f"HDF5 mesh validation did not provide a reference area: {mesh_path}")
        manifest_area = float(design_row["reference_area_m2"])
        if not np.isclose(float(hdf5_area), manifest_area, rtol=1.0e-6, atol=1.0e-14):
            raise ValueError(
                f"HDF5 reference area mismatch for {geometry_id}: {hdf5_area} != {manifest_area}"
            )
        mesh_reference = _relative_to_repository(mesh_path, piclas_root)
        common = _workflow_config(
            base,
            geometry_id=geometry_id,
            mesh_reference=mesh_reference,
            reference_area_m2=float(design_row["reference_area_m2"]),
            lf_config=lf_config,
            sample_indices=sample_indices,
        )
        study_id = f"vleo_cylinder_hex_wp5_round{round_number}_{mpi_procs}proc"
        common["adapter"]["kwargs"]["mpi_procs"] = int(mpi_procs)
        if simulator_module is not None:
            common["adapter"]["kwargs"]["simulator_module"] = simulator_module
        common["request"]["study_id"] = study_id
        tpmc = deepcopy(common)
        tpmc["adapter"]["model_id"] = "PICLas_TPMC"
        tpmc["adapter"]["fidelity"] = "lf"
        tpmc["adapter"]["kwargs"]["piclas_mode"] = "tpmc"
        tpmc["adapter"]["kwargs"]["submission_group_size"] = 10
        tpmc["request"]["cell_id"] = f"round{round_number}_tpmc_{geometry_id}_{mpi_procs}proc"
        tpmc["request"]["sample_ids"] = [f"wp1-crn-{index:04d}" for index in sample_indices]
        tpmc["request"]["samples"] = []
        for index in sample_indices:
            sample = deepcopy(lf_config["samples"][index])
            sample["random_seed"] = 20260900 + int(index)
            tpmc["request"]["samples"].append(sample)
        tpmc["request"]["metadata"]["case_name"] = f"{geometry_id}_l1_round{round_number}_tpmc_{mpi_procs}proc"
        tpmc_path = config_root / f"{geometry_id}_tpmc.json"
        tpmc_path.write_text(json.dumps(tpmc, indent=2, sort_keys=True) + "\n")
        suite_row = {
            "selection_order": selection_row["selection_order"],
            "selection_basis": selection_row["selection_basis"],
            "geometry_id": geometry_id,
            "design": design_row,
            "manifest_json": _relative_to_repository(manifest_path, repository_root),
            "mesh_path": _relative_to_repository(mesh_path, repository_root),
            "mesh_reference": mesh_reference,
            "reference_area_m2": float(design_row["reference_area_m2"]),
            "hdf5_reference_area_m2": float(hdf5_area),
            "n_tetrahedra": gmsh.get("n_tetrahedra"),
            "n_hexahedra": h5.get("nElems"),
            "hdf5_fingerprint": h5.get("mesh_fingerprint"),
            "tpmc_workflow_config": _relative_to_repository(tpmc_path, repository_root),
        }
        if n_dsmc:
            dsmc = deepcopy(common)
            dsmc["request"]["cell_id"] = f"round{round_number}_dsmc_{geometry_id}_{mpi_procs}proc"
            dsmc["request"]["sample_ids"] = [f"hf-crn-{index:04d}" for index in dsmc_indices]
            dsmc["request"]["samples"] = []
            for index in dsmc_indices:
                sample = deepcopy(lf_config["samples"][index])
                sample["random_seed"] = 20260900 + int(index)
                dsmc["request"]["samples"].append(sample)
            dsmc["request"]["metadata"]["case_name"] = (
                f"{geometry_id}_l1_round{round_number}_dsmc_{mpi_procs}proc"
            )
            dsmc_path = config_root / f"{geometry_id}_dsmc.json"
            dsmc_path.write_text(json.dumps(dsmc, indent=2, sort_keys=True) + "\n")
            suite_row["dsmc_workflow_config"] = _relative_to_repository(dsmc_path, repository_root)
        rows.append(suite_row)
    suite = {
        "schema_version": 1,
        "study_id": f"vleo_cylinder_hex_wp5_round{round_number}_{mpi_procs}proc",
        "round": int(round_number),
        "mpi_procs": int(mpi_procs),
        "simulator_module": str(
            simulator_module
            if simulator_module is not None
            else base.get("adapter", {}).get("kwargs", {}).get("simulator_module", "PICLas")
        ),
        "mesh_level": "L1",
        "n_dsmc_per_geometry": n_dsmc,
        "n_tpmc_per_geometry": n_tpmc,
        "total_dsmc_runs": len(rows) * n_dsmc,
        "total_tpmc_runs": len(rows) * n_tpmc,
        "common_dsmc_sample_indices": dsmc_indices,
        "common_tpmc_sample_indices": sample_indices,
        "untouched_validation_geometry_ids": sorted(validation),
        "geometries": rows,
    }
    target = Path(suite_output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    return {"suite_manifest": str(target), **suite}


def merge_round2_results(
    bundle_json: str | Path,
    suite_json: str | Path,
    run_root: str | Path,
    output_json: str | Path,
) -> Dict[str, Any]:
    bundle = json.loads(Path(bundle_json).resolve().read_text(encoding="utf-8"))
    suite = json.loads(Path(suite_json).resolve().read_text(encoding="utf-8"))
    root = Path(run_root).resolve()
    evaluations = list(bundle["evaluations"])
    positions = {
        (str(row["geometry_id"]), str(row["model_id"]), str(row["canonical_sample_id"])): index
        for index, row in enumerate(evaluations)
    }
    round_number = int(suite.get("round", 2))
    new_rows = 0
    replaced_rows = 0
    for geometry in suite["geometries"]:
        geometry_id = geometry["geometry_id"]
        bundle["geometries"][geometry_id] = {
            "design": geometry["design"],
            f"round{round_number}_suite": {
                key: geometry[key]
                for key in (
                    "geometry_id", "manifest_json", "mesh_path", "mesh_reference",
                    "reference_area_m2", "n_tetrahedra", "n_hexahedra", "hdf5_fingerprint",
                    "selection_order", "selection_basis",
                )
            },
        }
        area = float(geometry["reference_area_m2"])
        for model_id, fidelity in (("PICLas_DSMC", "hf"), ("PICLas_TPMC", "lf")):
            result_path = root / geometry_id / model_id / "piclas_results.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing collected round-2 result: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "collected":
                raise ValueError(f"Round-2 result is not collected: {result_path}")
            sample_ids = result["sample_ids"]
            cd_values = result["values_by_qoi"]["C_D"]
            costs = result["costs_cpu_hours"]
            if not (len(sample_ids) == len(cd_values) == len(costs)):
                raise ValueError(f"Inconsistent result lengths: {result_path}")
            for sample_id, cd, cost in zip(sample_ids, cd_values, costs):
                canonical_id = str(sample_id).replace("hf-crn-", "wp1-crn-")
                if canonical_id not in bundle["uncertainty_samples"]:
                    raise ValueError(f"Unknown round-2 uncertainty sample: {sample_id}")
                value = float(cd)
                row = {
                    "C_D": value,
                    "C_D2": value * value,
                    "canonical_sample_id": canonical_id,
                    "cost_cpu_hours": float(cost),
                    "drag_area_m2": value * area,
                    "drag_area_m4": (value * area) ** 2,
                    "fidelity": fidelity,
                    "geometry_id": geometry_id,
                    "model_id": model_id,
                    "reference_area_m2": area,
                    "sample_id": str(sample_id),
                }
                key = (geometry_id, model_id, canonical_id)
                if key in positions:
                    evaluations[positions[key]] = row
                    replaced_rows += 1
                else:
                    positions[key] = len(evaluations)
                    evaluations.append(row)
                    new_rows += 1
    evaluations.sort(key=lambda row: (row["geometry_id"], row["model_id"], row["canonical_sample_id"]))
    bundle["evaluations"] = evaluations
    bundle["selected_geometry_ids"] = sorted(bundle["geometries"])
    counts: Dict[str, int] = {}
    for row in evaluations:
        key = f"{row['geometry_id']}/{row['model_id']}"
        counts[key] = counts.get(key, 0) + 1
    bundle["counts"] = dict(sorted(counts.items()))
    bundle["study_id"] = f"vleo_cylinder_hex_wp5_paired_after_round{round_number}"
    target = Path(output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return {
        "schema_version": 1,
        "output_json": str(target),
        "new_rows": new_rows,
        "replaced_rows": replaced_rows,
        "total_evaluations": len(evaluations),
        "counts": bundle["counts"],
    }
