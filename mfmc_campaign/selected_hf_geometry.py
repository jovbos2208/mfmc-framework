from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .parametric_geometry import (
    CylinderHexSpec,
    build_geometry_assets,
    build_gmsh_exterior_mesh,
    build_piclas_hdf5_mesh,
    validate_piclas_hdf5_mesh,
)


L1_CONTROLS = {"body_mesh_size_m": 0.030, "farfield_mesh_size_m": 0.150}


def select_common_hf_samples(samples: Sequence[Mapping[str, Any]], count: int) -> List[int]:
    if count < 1 or count > len(samples):
        raise ValueError("HF sample count must be positive and no larger than the LF sample pool")
    excluded = {"database_index", "random_seed", "seed", "operations.seed", "density_scale"}
    numeric_names = sorted(
        key
        for key in set().union(*(sample.keys() for sample in samples))
        if key not in excluded
        and all(isinstance(sample.get(key), (int, float)) for sample in samples)
    )
    if not numeric_names:
        return list(range(count))
    values = np.asarray([[float(sample[name]) for name in numeric_names] for sample in samples])
    lower = np.min(values, axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0.0] = 1.0
    points = (values - lower) / span
    center = np.full(points.shape[1], 0.5)
    first = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    selected = [first]
    while len(selected) < count:
        remaining = [index for index in range(len(points)) if index not in selected]
        selected.append(
            max(
                remaining,
                key=lambda index: (
                    float(np.min(np.linalg.norm(points[index] - points[selected], axis=1))),
                    -index,
                ),
            )
        )
    return selected


def _workflow_config(
    base: Mapping[str, Any],
    *,
    geometry_id: str,
    mesh_reference: str,
    reference_area_m2: float,
    lf_config: Mapping[str, Any],
    sample_indices: Sequence[int],
) -> Dict[str, Any]:
    config = deepcopy(dict(base))
    request = config["request"]
    request["study_id"] = "vleo_cylinder_hex_wp5_initial_hf"
    request["cell_id"] = f"initial_hf_{geometry_id}"
    request["qois"] = ["C_D", "C_L", "C_Mz"]
    request["geometry"] = {
        "id": geometry_id,
        "name": geometry_id,
        "characteristic_length": 0.1,
        "geometry_class": "parametric_cylinder_hex",
        "tags": ["wp5_initial_hf", "l1_production_mesh", "uniform_scale_0p1"],
        "metadata": {
            "hf_mesh": mesh_reference,
            "piclas_object_boundary_name": "CYLINDER_HEX",
            "reference_area_m2": float(reference_area_m2),
        },
    }
    request["regime"] = deepcopy(lf_config["regime"])
    request["active_source_blocks"] = [
        "environment.density",
        "environment.composition",
        "environment.temperature",
        "environment.winds",
        "attitude.dispersion",
        "gsi.energy_accommodation",
        "gsi.surface_temperature",
    ]
    request["sample_ids"] = [f"hf-crn-{index:04d}" for index in sample_indices]
    request["samples"] = []
    for index in sample_indices:
        sample = deepcopy(lf_config["samples"][index])
        sample["random_seed"] = 20260900 + int(index)
        request["samples"].append(sample)
    request["seed"] = 20260818
    metadata = deepcopy(lf_config["metadata"])
    metadata.update(
        {
            "case_name": f"{geometry_id}_l1_initial_hf",
            "piclas_object_boundary_name": "CYLINDER_HEX",
            "flow_zero_direction": [1.0, 0.0, 0.0],
        }
    )
    request["metadata"] = metadata
    return config


def build_selected_hf_suite(
    selection_json: str | Path,
    design_manifest_json: str | Path,
    lf_config_json: str | Path,
    *,
    output_root: str | Path = "piclas/geometry/cylinder_hex_wp5/L1",
    config_output_dir: str | Path = "configs/studies/cylinder_hex_wp5_initial_hf",
    base_config_json: str | Path = "configs/studies/cylinder_hex_piclas_adapter_l1_5seeds.json",
    hf_samples_per_geometry: int = 5,
    gmsh_executable: str = "gmsh",
    pyhope_executable: str = "pyhope",
) -> Dict[str, Any]:
    selection = json.loads(Path(selection_json).resolve().read_text(encoding="utf-8"))
    design_path = Path(design_manifest_json).resolve()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    lf_config = json.loads(Path(lf_config_json).resolve().read_text(encoding="utf-8"))
    base_config = json.loads(Path(base_config_json).resolve().read_text(encoding="utf-8"))
    selected_ids = [str(row["geometry_id"]) for row in selection["selected"]]
    validation_ids = set(selection.get("untouched_validation_geometry_ids", []))
    if validation_ids.intersection(selected_ids):
        raise ValueError("HF selection contains an untouched validation geometry")
    design_rows = {str(row["geometry_id"]): row for row in design["designs"]}
    missing = [geometry_id for geometry_id in selected_ids if geometry_id not in design_rows]
    if missing:
        raise ValueError(f"Selected geometries are absent from the design manifest: {missing}")
    sample_indices = select_common_hf_samples(lf_config["samples"], hf_samples_per_geometry)
    repository_root = Path.cwd().resolve()
    piclas_root = (repository_root / "piclas").resolve()
    mesh_root = Path(output_root).resolve()
    config_root = Path(config_output_dir).resolve()
    mesh_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    for selection_row in selection["selected"]:
        geometry_id = str(selection_row["geometry_id"])
        design_row = design_rows[geometry_id]
        source_manifest_path = design_path.parent / str(design_row["manifest_path"])
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        spec = CylinderHexSpec(**source_manifest["specification"])
        geometry_dir = mesh_root / geometry_id
        manifest = build_geometry_assets(
            geometry_dir,
            spec,
            design_id=geometry_id,
            uniform_scale_factor=float(source_manifest["provenance"]["uniform_linear_scale_factor"]),
            **L1_CONTROLS,
        )
        manifest_path = Path(manifest["manifest_json"])
        mesh_path = geometry_dir / f"{geometry_id}_mesh.h5"
        if mesh_path.is_file():
            h5_summary = validate_piclas_hdf5_mesh(mesh_path)
            if not h5_summary["valid"]:
                raise ValueError(f"Existing PICLas mesh is invalid: {mesh_path}")
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_manifest["assets"]["piclas_volume_mesh"] = mesh_path.name
            manifest_path.write_text(json.dumps(current_manifest, indent=2, sort_keys=True) + "\n")
            gmsh_summary = current_manifest.get("gmsh_exterior_mesh", {}).get("validation", {})
        else:
            gmsh_summary = build_gmsh_exterior_mesh(
                manifest_path, gmsh_executable=gmsh_executable
            )
            h5_summary = build_piclas_hdf5_mesh(
                manifest_path, pyhope_executable=pyhope_executable
            )
            mesh_path = Path(h5_summary["mesh_path"])
        try:
            mesh_reference = str(mesh_path.relative_to(piclas_root))
        except ValueError as exc:
            raise ValueError("HF mesh output root must be below the repository piclas directory") from exc
        workflow = _workflow_config(
            base_config,
            geometry_id=geometry_id,
            mesh_reference=mesh_reference,
            reference_area_m2=float(design_row["reference_area_m2"]),
            lf_config=lf_config,
            sample_indices=sample_indices,
        )
        workflow_path = config_root / f"{geometry_id}.json"
        workflow_path.write_text(json.dumps(workflow, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(
            {
                "selection_order": int(selection_row["selection_order"]),
                "selection_basis": selection_row["selection_basis"],
                "geometry_id": geometry_id,
                "manifest_json": str(manifest_path.relative_to(repository_root)),
                "mesh_path": str(mesh_path.relative_to(repository_root)),
                "mesh_reference": mesh_reference,
                "workflow_config": str(workflow_path.relative_to(repository_root)),
                "reference_area_m2": float(design_row["reference_area_m2"]),
                "n_tetrahedra": gmsh_summary.get("n_tetrahedra"),
                "n_hexahedra": h5_summary.get("nElems"),
                "hdf5_fingerprint": h5_summary.get("mesh_fingerprint"),
                "scaled_jacobian_histogram": h5_summary.get("scaled_jacobian_histogram", {}),
            }
        )
    suite = {
        "schema_version": 1,
        "study_id": "vleo_cylinder_hex_wp5_initial_hf",
        "mesh_level": "L1",
        "mesh_controls": L1_CONTROLS,
        "hf_samples_per_geometry": hf_samples_per_geometry,
        "total_hf_runs": len(summaries) * hf_samples_per_geometry,
        "common_lf_sample_indices": sample_indices,
        "common_hf_sample_ids": [f"hf-crn-{index:04d}" for index in sample_indices],
        "untouched_validation_geometry_ids": sorted(validation_ids),
        "geometries": summaries,
    }
    suite_path = mesh_root.parent / "initial_hf_suite.json"
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"suite_manifest": str(suite_path), **suite}


def build_sequential_hf_suite(
    initial_suite_json: str | Path,
    acquisition_json: str | Path,
    paired_bundle_json: str | Path,
    *,
    config_output_dir: str | Path = "configs/studies/cylinder_hex_wp5_sequential_hf_round1",
    suite_output_json: str | Path = "outputs/cylinder_hex/wp5_sequential_hf_round1/suite.json",
) -> Dict[str, Any]:
    """Reuse validated L1 meshes and build the next nested DSMC acquisition suite."""
    initial = json.loads(Path(initial_suite_json).resolve().read_text(encoding="utf-8"))
    acquisition = json.loads(Path(acquisition_json).resolve().read_text(encoding="utf-8"))
    bundle = json.loads(Path(paired_bundle_json).resolve().read_text(encoding="utf-8"))
    initial_by_geometry = {str(row["geometry_id"]): row for row in initial["geometries"]}
    uncertainty_samples = bundle["uncertainty_samples"]
    repository_root = Path.cwd().resolve()
    config_root = Path(config_output_dir).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    common_ids: list[str] | None = None
    for acquisition_row in acquisition["geometries"]:
        geometry_id = str(acquisition_row["geometry_id"])
        if geometry_id not in initial_by_geometry:
            raise ValueError(f"Acquisition geometry is absent from the initial HF suite: {geometry_id}")
        canonical_ids = [str(value) for value in acquisition_row["selected_canonical_sample_ids"]]
        if common_ids is None:
            common_ids = canonical_ids
        elif canonical_ids != common_ids:
            raise ValueError("Sequential acquisition must retain common random numbers across geometries")
        missing = [sample_id for sample_id in canonical_ids if sample_id not in uncertainty_samples]
        if missing:
            raise ValueError(f"Acquisition samples are absent from the paired bundle: {missing}")
        initial_row = initial_by_geometry[geometry_id]
        base_path = repository_root / str(initial_row["workflow_config"])
        config = json.loads(base_path.read_text(encoding="utf-8"))
        request = config["request"]
        request["study_id"] = "vleo_cylinder_hex_wp5_sequential_hf_round1"
        request["cell_id"] = f"sequential_hf_round1_{geometry_id}"
        request["geometry"]["tags"] = [
            "wp5_sequential_hf_round1", "l1_production_mesh", "uniform_scale_0p1"
        ]
        request["sample_ids"] = [sample_id.replace("wp1-crn-", "hf-crn-") for sample_id in canonical_ids]
        request["samples"] = []
        for sample_id in canonical_ids:
            sample = deepcopy(uncertainty_samples[sample_id])
            index = int(sample_id.rsplit("-", 1)[1])
            sample["random_seed"] = 20260900 + index
            request["samples"].append(sample)
        request["metadata"]["case_name"] = f"{geometry_id}_l1_sequential_hf_round1"
        workflow_path = config_root / f"{geometry_id}.json"
        workflow_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({
            **initial_row,
            "workflow_config": str(workflow_path.relative_to(repository_root)),
            "acquisition_round": 1,
            "canonical_sample_ids": canonical_ids,
            "hf_sample_ids": list(request["sample_ids"]),
        })
    suite = {
        "schema_version": 1,
        "study_id": "vleo_cylinder_hex_wp5_sequential_hf_round1",
        "mesh_level": initial.get("mesh_level", "L1"),
        "hf_samples_per_geometry": len(common_ids or []),
        "total_hf_runs": len(rows) * len(common_ids or []),
        "common_canonical_sample_ids": common_ids or [],
        "untouched_validation_geometry_ids": initial.get("untouched_validation_geometry_ids", []),
        "geometries": rows,
    }
    target = Path(suite_output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"suite_manifest": str(target), **suite}
