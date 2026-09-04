from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .parametric_geometry import (
    CylinderHexSpec,
    ParametricGeometryError,
    SurfaceMesh,
    generate_cylinder_hex,
    projected_reference_area,
    surface_mesh_from_points,
    validate_surface,
    write_adbsat_mat,
    write_gmsh_exterior_geo,
    write_obj,
    write_stl,
)


CONTROL_NODE_GENERATOR = "mfmc-cylinder-hex-symmetric-control-nodes"
CONTROL_NODE_GENERATOR_VERSION = "1.0.0"
CONTROL_NODE_VARIABLES = (
    "nose_station_shift",
    "tail_station_shift",
    "nose_width_log_scale",
    "nose_height_log_scale",
    "shoulder_width_log_scale",
    "shoulder_height_log_scale",
    "tail_shoulder_width_log_scale",
    "tail_shoulder_height_log_scale",
    "tail_width_log_scale",
    "tail_height_log_scale",
)


@dataclass(frozen=True)
class SymmetricControlNodeDesignSpace:
    nose_station_shift: tuple[float, float] = (-0.08, 0.08)
    tail_station_shift: tuple[float, float] = (-0.08, 0.08)
    nose_width_log_scale: tuple[float, float] = (-0.25, 0.25)
    nose_height_log_scale: tuple[float, float] = (-0.25, 0.25)
    shoulder_width_log_scale: tuple[float, float] = (-0.20, 0.20)
    shoulder_height_log_scale: tuple[float, float] = (-0.20, 0.20)
    tail_shoulder_width_log_scale: tuple[float, float] = (-0.20, 0.20)
    tail_shoulder_height_log_scale: tuple[float, float] = (-0.20, 0.20)
    tail_width_log_scale: tuple[float, float] = (-0.25, 0.25)
    tail_height_log_scale: tuple[float, float] = (-0.25, 0.25)

    def bounds(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in CONTROL_NODE_VARIABLES], dtype=float)

    def validate(self) -> None:
        bounds = self.bounds()
        if bounds.shape != (len(CONTROL_NODE_VARIABLES), 2) or not np.all(np.isfinite(bounds)):
            raise ParametricGeometryError("control-node bounds must be finite lower/upper pairs")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ParametricGeometryError("every control-node lower bound must be below its upper bound")


def _signed_volume(mesh: SurfaceMesh) -> float:
    xyz = mesh.points[mesh.triangles]
    return float(np.sum(np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2]))) / 6.0)


def _parameter_array(parameters: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(parameters, Mapping):
        missing = [name for name in CONTROL_NODE_VARIABLES if name not in parameters]
        if missing:
            raise ParametricGeometryError(f"missing control-node parameters: {missing}")
        values = np.asarray([parameters[name] for name in CONTROL_NODE_VARIABLES], dtype=float)
    else:
        values = np.asarray(parameters, dtype=float)
    if values.shape != (len(CONTROL_NODE_VARIABLES),) or not np.all(np.isfinite(values)):
        raise ParametricGeometryError("control-node parameters must contain ten finite values")
    return values


def deform_cylinder_hex_control_nodes(
    baseline_spec: CylinderHexSpec,
    parameters: Mapping[str, float] | Sequence[float],
    *,
    design_space: SymmetricControlNodeDesignSpace | None = None,
) -> tuple[SurfaceMesh, Dict[str, Any]]:
    """Move four symmetric surface rings and restore the prescribed body volume."""
    if int(baseline_spec.axial_segments_per_taper) != 1:
        raise ParametricGeometryError("control-node deformation requires axial_segments_per_taper=1")
    space = design_space or SymmetricControlNodeDesignSpace()
    space.validate()
    values = _parameter_array(parameters)
    bounds = space.bounds()
    if np.any(values < bounds[:, 0] - 1.0e-12) or np.any(values > bounds[:, 1] + 1.0e-12):
        raise ParametricGeometryError("control-node parameters lie outside the configured bounds")

    baseline, baseline_derived = generate_cylinder_hex(baseline_spec)
    points = np.array(baseline.points, copy=True)
    ring_size = 8
    ring_count = (len(points) - 2) // ring_size
    if ring_count != 4:
        raise ParametricGeometryError(f"expected four control rings, found {ring_count}")
    length = float(baseline_spec.total_length_m)
    nose_x = baseline_spec.nose_length_fraction * length + values[0] * length
    tail_x = length * (1.0 - baseline_spec.tail_length_fraction) + values[1] * length
    if nose_x <= 0.02 * length or tail_x >= 0.98 * length or tail_x - nose_x < 0.12 * length:
        raise ParametricGeometryError("moved axial control stations leave an invalid center section")
    ring_x = (0.0, nose_x, tail_x, length)
    transverse = np.exp(values[2:].reshape(4, 2))
    before: list[list[list[float]]] = []
    for ring in range(ring_count):
        start = ring * ring_size
        stop = start + ring_size
        before.append(points[start:stop].tolist())
        points[start:stop, 0] = ring_x[ring]
        points[start:stop, 1] *= transverse[ring, 0]
        points[start:stop, 2] *= transverse[ring, 1]
    points[-2] = [0.0, 0.0, 0.0]
    points[-1] = [length, 0.0, 0.0]

    provisional = surface_mesh_from_points(points, baseline.triangles)
    volume = _signed_volume(provisional)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ParametricGeometryError("control-node deformation produced non-positive volume")
    volume_scale = float(np.sqrt(baseline_spec.target_volume_m3 / volume))
    points[:, 1:] *= volume_scale
    mesh = surface_mesh_from_points(points, baseline.triangles)
    reference_area = projected_reference_area(mesh, axis=0)
    validation = validate_surface(mesh, baseline_spec)
    # Independent ring motion can intentionally create a mildly non-convex
    # shoulder.  Ring order, positive transverse scales, manifoldness and
    # non-degenerate faces certify this restricted deformation family without
    # incorrectly requiring global convexity.
    validation["checks"]["ordered_positive_ring_deformation"] = bool(
        np.all(np.diff(ring_x) > 0.0) and np.all(transverse > 0.0)
    )
    validation["convexity_required"] = False
    validation["valid"] = all(
        passed
        for name, passed in validation["checks"].items()
        if name != "convex_no_self_intersection_certificate"
    )
    if not validation["valid"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise ParametricGeometryError(f"control-node geometry failed validation: {failed}")
    normalized = (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    derived = {
        **baseline_derived,
        "width_m": float(np.ptp(mesh.points[:, 1])),
        "height_m": float(np.ptp(mesh.points[:, 2])),
        "maximum_cross_section_area_m2": reference_area,
        "center_length_m": float(tail_x - nose_x),
        "control_node_volume_scale": volume_scale,
    }
    return mesh, {
        "parameters": {name: float(value) for name, value in zip(CONTROL_NODE_VARIABLES, values)},
        "normalized_parameters": [float(value) for value in normalized],
        "bounds": {name: list(getattr(space, name)) for name in CONTROL_NODE_VARIABLES},
        "ring_x_m": [float(value) for value in ring_x],
        "transverse_log_scales": values[2:].reshape(4, 2).tolist(),
        "volume_normalization_scale": volume_scale,
        "control_nodes_before": before,
        "control_nodes_after": [
            mesh.points[ring * ring_size : (ring + 1) * ring_size].tolist()
            for ring in range(ring_count)
        ],
        "derived_geometry": derived,
        "validation": validation,
    }


def _design_fingerprint(spec: CylinderHexSpec, parameters: Mapping[str, float]) -> str:
    payload = {
        "generator": CONTROL_NODE_GENERATOR,
        "generator_version": CONTROL_NODE_GENERATOR_VERSION,
        "baseline_specification": asdict(spec),
        "parameters": {name: float(parameters[name]) for name in CONTROL_NODE_VARIABLES},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_control_node_geometry_assets(
    output_dir: str | Path,
    baseline_spec: CylinderHexSpec,
    parameters: Mapping[str, float] | Sequence[float],
    *,
    design_id: str | None = None,
    design_space: SymmetricControlNodeDesignSpace | None = None,
    uniform_scale_factor: float = 1.0,
    body_mesh_size_m: float = 0.06,
    farfield_mesh_size_m: float = 0.30,
) -> Dict[str, Any]:
    values = _parameter_array(parameters)
    parameter_map = {name: float(value) for name, value in zip(CONTROL_NODE_VARIABLES, values)}
    mesh, deformation = deform_cylinder_hex_control_nodes(
        baseline_spec, parameter_map, design_space=design_space
    )
    derived = deformation["derived_geometry"]
    fingerprint = _design_fingerprint(baseline_spec, parameter_map)
    geometry_id = design_id or f"cylinder_hex_control_{fingerprint[:12]}"
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    obj_path = target / f"{geometry_id}.obj"
    stl_path = target / f"{geometry_id}.stl"
    mat_path = target / f"{geometry_id}.mat"
    surface_path = target / f"{geometry_id}.surface.npz"
    gmsh_path = target / f"{geometry_id}.exterior.geo"
    manifest_path = target / f"{geometry_id}.manifest.json"
    write_obj(obj_path, mesh)
    write_stl(stl_path, mesh)
    write_adbsat_mat(mat_path, mesh, float(derived["maximum_cross_section_area_m2"]))
    write_gmsh_exterior_geo(
        gmsh_path, mesh, baseline_spec,
        body_mesh_size_m=body_mesh_size_m, farfield_mesh_size_m=farfield_mesh_size_m,
    )
    np.savez_compressed(
        surface_path,
        points=mesh.points,
        triangles=mesh.triangles,
        triangle_area=mesh.triangle_area,
        triangle_normal=mesh.triangle_normal,
        triangle_center=mesh.triangle_center,
        mesh_fingerprint=np.asarray([mesh.mesh_fingerprint]),
        design_fingerprint=np.asarray([fingerprint]),
    )
    manifest = {
        "schema_version": 1,
        "generator": CONTROL_NODE_GENERATOR,
        "generator_version": CONTROL_NODE_GENERATOR_VERSION,
        "geometry_id": geometry_id,
        "design_fingerprint": fingerprint,
        "mesh_fingerprint": mesh.mesh_fingerprint,
        "specification": asdict(baseline_spec),
        "baseline_specification": asdict(baseline_spec),
        "parameterization": "symmetric_surface_control_nodes",
        "control_node_deformation": deformation,
        "derived_geometry": derived,
        "reference_area": {
            "value_m2": float(derived["maximum_cross_section_area_m2"]),
            "convention": "two-sided surface projection normal to positive body x-axis",
            "optimization_quantity": "C_D_times_reference_area_m2",
        },
        "coordinate_system": {
            "units": "m", "body_axis": "+x nose-to-tail", "width_axis": "y",
            "height_axis": "z", "origin": "nose cap center",
        },
        "validation": deformation["validation"],
        "assets": {
            "adbsat_obj": obj_path.name,
            "adbsat_mat": mat_path.name,
            "meshing_surface_stl": stl_path.name,
            "canonical_surface_npz": surface_path.name,
            "gmsh_exterior_geo": gmsh_path.name,
            "piclas_volume_mesh": None,
            "piclas_mesh_status": "requires independent Gmsh/PyHOPE volume-mesh stage",
        },
        "campaign_geometry_descriptor": {
            "id": geometry_id,
            "name": geometry_id,
            "characteristic_length": float(baseline_spec.total_length_m),
            "geometry_class": "cylinder_hex_symmetric_control_nodes",
            "tags": ["control_nodes", "fixed_volume", "symmetric", "generated_from_scratch"],
            "metadata": {"reference_area_m2": float(derived["maximum_cross_section_area_m2"]), **parameter_map},
        },
        "provenance": {
            "source_geometry": "analytical_baseline_with_symmetric_control_node_deformation",
            "external_mesh_reused": False,
            "uniform_linear_scale_factor": float(uniform_scale_factor),
        },
        "gmsh_mesh_controls": {
            "body_mesh_size_m": float(body_mesh_size_m),
            "farfield_mesh_size_m": float(farfield_mesh_size_m),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_json": str(manifest_path)}


def generate_control_node_refinement_manifest(
    output_json: str | Path,
    *,
    center_parameters: Mapping[str, float] | Sequence[float] | None = None,
    baseline_spec: CylinderHexSpec | None = None,
    design_space: SymmetricControlNodeDesignSpace | None = None,
    count: int = 6,
    normalized_radius: float = 0.12,
    iteration: int = 1,
    uniform_scale_factor: float = 0.1,
    include_center: bool = False,
) -> Dict[str, Any]:
    """Create a deterministic trust-region poll around one symmetric control-node design."""
    if count < 1:
        raise ValueError("control-node candidate count must be positive")
    if not 0.0 < normalized_radius <= 0.5:
        raise ValueError("normalized control-node radius must be in (0, 0.5]")
    space = design_space or SymmetricControlNodeDesignSpace()
    space.validate()
    bounds = space.bounds()
    center = np.zeros(len(CONTROL_NODE_VARIABLES)) if center_parameters is None else _parameter_array(center_parameters)
    center_normalized = (center - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    directions = np.vstack([np.eye(len(center)), -np.eye(len(center))])
    directions = np.roll(directions, shift=(int(iteration) * 3) % len(directions), axis=0)
    target = Path(output_json).resolve()
    scaled = (baseline_spec or CylinderHexSpec(axial_segments_per_taper=1)).scaled(uniform_scale_factor)
    accepted: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    poll_points: list[tuple[int, np.ndarray, str]] = []
    if include_center:
        poll_points.append((-1, center_normalized, "control_node_baseline"))
    poll_points.extend(
        (index, np.clip(center_normalized + normalized_radius * direction, 0.0, 1.0),
         "symmetric_control_node_trust_region_poll")
        for index, direction in enumerate(directions)
    )
    for direction_index, point, selection_basis in poll_points:
        if len(accepted) >= count:
            break
        values = bounds[:, 0] + point * (bounds[:, 1] - bounds[:, 0])
        geometry_id = f"cylinder_hex_control_i{int(iteration):03d}_{len(accepted):03d}"
        try:
            assets = build_control_node_geometry_assets(
                target.parent / "designs" / geometry_id, scaled, values,
                design_id=geometry_id, design_space=space,
                uniform_scale_factor=uniform_scale_factor,
                body_mesh_size_m=0.003, farfield_mesh_size_m=0.015,
            )
        except ValueError as exc:
            rejected.append({"direction_index": direction_index, "reason": str(exc)})
            continue
        parameter_map = {name: float(value) for name, value in zip(CONTROL_NODE_VARIABLES, values)}
        accepted.append({
            "geometry_id": geometry_id,
            "role": "control_node_optimization_candidate",
            "eligible_for_model_fitting": True,
            "selection_basis": selection_basis,
            "parameters": parameter_map,
            "normalized_parameters": [float(value) for value in point],
            **parameter_map,
            **{f"normalized_{name}": float(value) for name, value in zip(CONTROL_NODE_VARIABLES, point)},
            "manifest_path": str(Path(assets["manifest_json"]).relative_to(target.parent)),
            "design_fingerprint": assets["design_fingerprint"],
            "mesh_fingerprint": assets["mesh_fingerprint"],
            "reference_area_m2": float(assets["reference_area"]["value_m2"]),
            "validation": assets["validation"],
        })
    if len(accepted) != count:
        raise ValueError(f"could generate only {len(accepted)} of {count} valid control-node candidates")
    result = {
        "schema_version": 1,
        "method": "symmetric_control_node_trust_region_poll",
        "parameterization": "symmetric_surface_control_nodes",
        "iteration": int(iteration),
        "normalized_radius": float(normalized_radius),
        "variables": list(CONTROL_NODE_VARIABLES),
        "bounds": {name: list(getattr(space, name)) for name in CONTROL_NODE_VARIABLES},
        "center_parameters": {name: float(value) for name, value in zip(CONTROL_NODE_VARIABLES, center)},
        "center_normalized_parameters": [float(value) for value in center_normalized],
        "center_included": bool(include_center),
        "candidates": accepted,
        "designs": accepted,
        "n_designs": len(accepted),
        "baseline_geometry_id": accepted[0]["geometry_id"],
        "validation_geometry_ids": [],
        "uniform_linear_scale_factor": float(uniform_scale_factor),
        "density_scale_required_for_kn_similarity": float(1.0 / uniform_scale_factor),
        "rejected_candidates": rejected,
        "evaluation_contract": {
            "target_model": "PICLas_TPMC", "control_model": "Sentman",
            "tpmc_runs_per_geometry": 20, "common_random_numbers": True,
            "dsmc_during_optimization": False,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "output_json": str(target)}
