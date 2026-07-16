from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import importlib
import inspect
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .adbsat_surface_archive import export_adbsat_surface_archive
from .adbsat_surface_mapping import load_surface_mapping
from .piclas_surface_archive import export_piclas_surface_archive
from .types import EvaluationRequest, EvaluationResult, GeometryDescriptor, RegimeDescriptor

_PYMSIS_ROW_CACHE: Dict[str, List[float]] = {}
_EARTH_MU_M3PS2 = 3.986004418e14
_EARTH_RADIUS_M = 6378137.0
_AMU_KG = 1.66053906660e-27
_PYMSIS_SPECIES_MASS_KG = np.asarray(
    [28.0134, 31.998, 15.999, 4.002602, 1.00794, 39.948, 14.0067, 15.999, 30.006],
    dtype=float,
) * _AMU_KG
_PICLAS_PAYLOAD_DEFAULT_KEYS = {
    "manual_timestep_s",
    "manual_time_step_s",
    "time_step_s",
    "timestep_s",
    "t_end_s",
    "tend_s",
    "simulation_end_time_s",
    "t_end_scale",
    "tend_scale",
    "simulation_end_time_scale",
    "macro_particle_factor_scale",
    "mpf_scale",
    "macro_particle_factor",
    "mpf_override",
    "sampling_iterations",
    "part_iteration_for_macro_val",
    "macro_sampling_iterations",
    "octree_part_num_node",
    "octree_node_particles",
    "octree_part_num_node_min",
    "octree_node_particles_min",
    "domain_side_length_m",
    "domain_size_m",
    "domain_side_m",
    "piclas_domain_side_m",
    "particles_mpi_weight",
    "mpi_particle_weight",
}
_PICLAS_MPF_OVERRIDE_KEYS = {
    "macro_particle_factor_scale",
    "mpf_scale",
    "macro_particle_factor",
    "mpf_override",
}
_PICLAS_TEND_OVERRIDE_KEYS = {
    "t_end_s",
    "tend_s",
    "simulation_end_time_s",
    "t_end_scale",
    "tend_scale",
    "simulation_end_time_scale",
}


def _slug_for_path(value: Any) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _validate_atmosphere_row(row: Any, context: str) -> np.ndarray:
    arr = np.asarray(row, dtype=float).reshape(-1)
    if arr.size < 11:
        raise ValueError(f"Atmosphere row has {arr.size} entries, expected at least 11; {context}")
    arr = arr[:11]
    if not np.isfinite(arr[10]) or arr[10] <= 0.0:
        raise ValueError(f"Atmosphere temperature is non-positive; {context}; row={arr.tolist()}")
    species = arr[1:10].copy()
    species[~np.isfinite(species)] = 0.0
    species = np.clip(species, 0.0, None)
    arr[1:10] = species
    if not np.isfinite(arr[0]) or arr[0] <= 0.0:
        reconstructed_mass_density = float(np.dot(species, _PYMSIS_SPECIES_MASS_KG))
        if not np.isfinite(reconstructed_mass_density) or reconstructed_mass_density <= 0.0:
            raise ValueError(f"Atmosphere mass density cannot be reconstructed; {context}; row={arr.tolist()}")
        arr[0] = reconstructed_mass_density
    return arr


def _sanitize_payload_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_payload_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_payload_value(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return 0.0 if np.isnan(value) else float(value)
    if isinstance(value, float):
        return 0.0 if np.isnan(value) else value
    return value


def _strip_sample_piclas_numerical_controls(payload: Dict[str, Any]) -> None:
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        return
    for key in _PICLAS_PAYLOAD_DEFAULT_KEYS:
        sample.pop(key, None)


def _first_present(sample: Dict[str, Any], metadata: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _resolve_attitude(sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
    aos_base = _to_float(
        _first_present(
            sample,
            metadata,
            ["aos_deg", "aos", "beta_deg", "attitude_aos_deg", "attitude.aos"],
            metadata.get("aos_deg", 0.0),
        ),
        0.0,
    )
    aoa_base = _to_float(
        _first_present(
            sample,
            metadata,
            ["aoa_deg", "aoa", "alpha_deg", "attitude_aoa_deg", "attitude.aoa"],
            metadata.get("aoa_deg", 0.0),
        ),
        0.0,
    )

    jitter_block = _first_present(sample, metadata, ["attitude_jitter", "jitter", "attitude.jitter"], None)
    jitter_scalar = _to_float(
        _first_present(
            sample,
            metadata,
            ["attitude_jitter_deg", "jitter_deg", "attitude.jitter_deg"],
            0.0,
        ),
        0.0,
    )
    jitter_aoa = _to_float(
        _first_present(
            sample,
            metadata,
            ["jitter_aoa_deg", "attitude_jitter_aoa_deg"],
            None,
        ),
        None,
    )
    jitter_aos = _to_float(
        _first_present(
            sample,
            metadata,
            ["jitter_aos_deg", "attitude_jitter_aos_deg"],
            None,
        ),
        None,
    )

    if isinstance(jitter_block, dict):
        if jitter_aos is None:
            jitter_aos = _to_float(jitter_block.get("aos_deg", jitter_block.get("aos")), None)
        if jitter_aoa is None:
            jitter_aoa = _to_float(jitter_block.get("aoa_deg", jitter_block.get("aoa")), None)
        if jitter_scalar == 0.0:
            jitter_scalar = _to_float(jitter_block.get("deg", jitter_block.get("value")), 0.0)

    jitter_aos = jitter_scalar if jitter_aos is None else jitter_aos
    jitter_aoa = jitter_scalar if jitter_aoa is None else jitter_aoa

    return {
        "aos_deg": float((aos_base or 0.0) + (jitter_aos or 0.0)),
        "aoa_deg": float((aoa_base or 0.0) + (jitter_aoa or 0.0)),
        "jitter_aos_deg": float(jitter_aos or 0.0),
        "jitter_aoa_deg": float(jitter_aoa or 0.0),
        "jitter_scalar_deg": float(jitter_scalar or 0.0),
    }


def _flow_zero_direction_from_payload(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    value = _first_present(
        payload,
        {},
        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
        None,
    )
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) != 3:
            return None
        value = parts
    try:
        vec = np.asarray(value, dtype=float).reshape(-1)[:3]
    except Exception:
        return None
    if vec.size < 3:
        return None
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vec / norm


def _flow_unit_from_angles(aos_deg: float, aoa_deg: float, flow_zero_direction: Any = None) -> np.ndarray:
    zero_payload = {"flow_zero_direction": flow_zero_direction}
    zero_dir = _flow_zero_direction_from_payload(zero_payload) if flow_zero_direction is not None else None
    if zero_dir is not None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        horizontal_zero = zero_dir - float(np.dot(zero_dir, up)) * up
        if float(np.linalg.norm(horizontal_zero)) <= 1e-12:
            horizontal_zero = np.array([1.0, 0.0, 0.0], dtype=float)
        horizontal_zero = horizontal_zero / (np.linalg.norm(horizontal_zero) + 1e-16)
        side = np.cross(horizontal_zero, up)
        side = side / (np.linalg.norm(side) + 1e-16)

        aos_rad = np.radians(float(aos_deg))
        aoa_rad = np.radians(float(aoa_deg))
        horizontal = np.cos(aos_rad) * horizontal_zero + np.sin(aos_rad) * side
        flow = np.cos(aoa_rad) * horizontal + np.sin(aoa_rad) * up
        return flow / (np.linalg.norm(flow) + 1e-16)

    aos_rad = np.radians(float(aos_deg))
    aoa_rad = np.radians(float(aoa_deg))
    cos_aoa = float(np.cos(aoa_rad))
    return np.array(
        [
            -np.sin(aos_rad) * cos_aoa,
            np.cos(aos_rad) * cos_aoa,
            np.sin(aoa_rad),
        ],
        dtype=float,
    )


def _angles_from_flow_vector(flow_vec: Any, flow_zero_direction: Any = None) -> Optional[Dict[str, float]]:
    try:
        vec = np.asarray(flow_vec, dtype=float).reshape(-1)
    except Exception:
        return None
    if vec.size < 3:
        return None
    vec = np.nan_to_num(vec[:3], nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    unit = vec / norm
    zero_payload = {"flow_zero_direction": flow_zero_direction}
    zero_dir = _flow_zero_direction_from_payload(zero_payload) if flow_zero_direction is not None else None
    if zero_dir is not None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        horizontal_zero = zero_dir - float(np.dot(zero_dir, up)) * up
        if float(np.linalg.norm(horizontal_zero)) <= 1e-12:
            horizontal_zero = np.array([1.0, 0.0, 0.0], dtype=float)
        horizontal_zero = horizontal_zero / (np.linalg.norm(horizontal_zero) + 1e-16)
        side = np.cross(horizontal_zero, up)
        side = side / (np.linalg.norm(side) + 1e-16)
        aoa_deg = float(np.degrees(np.arcsin(np.clip(float(np.dot(unit, up)), -1.0, 1.0))))
        horizontal = unit - float(np.dot(unit, up)) * up
        if float(np.linalg.norm(horizontal)) <= 1e-12:
            aos_deg = 0.0
        else:
            horizontal = horizontal / (np.linalg.norm(horizontal) + 1e-16)
            aos_deg = float(np.degrees(np.arctan2(float(np.dot(horizontal, side)), float(np.dot(horizontal, horizontal_zero)))))
        return {
            "aoa_deg": aoa_deg,
            "aos_deg": aos_deg,
            "relative_flow_speed_mps": norm,
        }

    aoa_deg = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))
    aos_deg = float(np.degrees(np.arctan2(-unit[0], unit[1])))
    return {
        "aoa_deg": aoa_deg,
        "aos_deg": aos_deg,
        "relative_flow_speed_mps": norm,
    }


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = np.radians(float(angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle = np.radians(float(angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _piclas_flow_dir(aos_deg: Any, aoa_deg: Any) -> np.ndarray:
    flow = (_rotation_matrix_z(float(_to_float(aos_deg, 0.0) or 0.0)) @ _rotation_matrix_x(float(_to_float(aoa_deg, 0.0) or 0.0))) @ np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    )
    return flow / (np.linalg.norm(flow) + 1e-16)


def _resolve_hf_mesh_path(mesh_name: Any) -> Optional[str]:
    if mesh_name in {None, ""}:
        return None
    mesh_path = str(mesh_name)
    candidates = [mesh_path]
    if not os.path.isabs(mesh_path):
        candidates.extend(
            [
                os.path.abspath(mesh_path),
                os.path.abspath(os.path.join("piclas", mesh_path)),
                os.path.abspath(os.path.join("PICLas", mesh_path)),
            ]
        )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _half_total_reference_area_from_mesh(mesh_path: str) -> Optional[float]:
    try:
        import pyvista as pv  # type: ignore
    except Exception:
        return None

    try:
        dataset = pv.read(mesh_path)
    except Exception:
        return None

    blocks = list(dataset) if hasattr(dataset, "__iter__") and not hasattr(dataset, "n_cells") else [dataset]
    total_area = 0.0
    for block in blocks:
        if block is None or not hasattr(block, "n_cells"):
            continue
        for cell_idx in range(int(block.n_cells)):
            points = np.asarray(block.get_cell(cell_idx).points, dtype=float)
            if points.shape[0] < 3:
                continue
            area_vector = np.zeros(3, dtype=float)
            origin = points[0]
            for point_idx in range(1, points.shape[0] - 1):
                area_vector += 0.5 * np.cross(points[point_idx] - origin, points[point_idx + 1] - origin)
            total_area += float(np.linalg.norm(area_vector))
    if not np.isfinite(total_area) or total_area <= 0.0:
        return None
    return float(max(0.5 * total_area, 1e-12))


def _candidate_hopr_boundary_vtus(mesh_path: str) -> List[str]:
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    cwd = os.getcwd()
    stem = os.path.splitext(os.path.basename(mesh_path))[0]
    roots = [stem]
    if stem.endswith("_mesh"):
        roots.append(stem[: -len("_mesh")])

    dirs = [mesh_dir, cwd]
    for root in roots:
        dirs.extend(path for path in glob.glob(os.path.join(cwd, f"{root}*")) if os.path.isdir(path))

    candidates: List[str] = []
    seen = set()
    for directory in dirs:
        for root in roots:
            path = os.path.join(directory, f"{root}_Debugmesh_BC.vtu")
            if path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                candidates.append(path)
    return candidates


def _half_total_reference_area_from_hopr_boundary_vtu(vtu_path: str, obj_bc_ids: List[int]) -> Optional[float]:
    try:
        import pyvista as pv  # type: ignore
    except Exception:
        return None

    try:
        mesh = pv.read(vtu_path)
        with_area = mesh.compute_cell_sizes(length=False, volume=False)
        areas = np.asarray(with_area.cell_data["Area"], dtype=float)
    except Exception:
        return None

    obj_ids = {int(value) for value in obj_bc_ids}
    if not obj_ids or areas.size != int(getattr(mesh, "n_cells", 0)):
        return None

    total_area = 0.0
    if "BCIndex" in mesh.cell_data:
        bc_index = np.asarray(mesh.cell_data["BCIndex"]).reshape(-1)
        if bc_index.size != areas.size:
            return None
        mask = np.asarray([int(round(float(value))) in obj_ids for value in bc_index], dtype=bool)
        total_area = float(np.sum(areas[mask]))
    elif "BCIndex" in mesh.point_data:
        bc_index = np.asarray(mesh.point_data["BCIndex"]).reshape(-1)
        for cell_idx in range(int(mesh.n_cells)):
            point_ids = mesh.get_cell(cell_idx).point_ids
            if not point_ids:
                continue
            values = bc_index[np.asarray(point_ids, dtype=int)]
            if values.size == 0:
                continue
            rounded = np.rint(values).astype(int)
            if np.all(rounded == rounded[0]) and int(rounded[0]) in obj_ids:
                total_area += float(areas[cell_idx])
    else:
        return None

    if not np.isfinite(total_area) or total_area <= 0.0:
        return None
    return float(max(0.5 * total_area, 1e-12))


def _piclas_h5_half_total_reference_area(mesh_path: str) -> Optional[float]:
    try:
        import h5py  # type: ignore
    except Exception:
        return None

    face_nodes = {
        1: [0, 1, 2, 3],
        2: [4, 5, 6, 7],
        3: [0, 1, 5, 4],
        4: [1, 2, 6, 5],
        5: [2, 3, 7, 6],
        6: [3, 0, 4, 7],
    }

    def resolve_side(side_info: np.ndarray, side_id: int) -> Optional[np.ndarray]:
        current = abs(int(side_id))
        seen = set()
        for _ in range(32):
            if current <= 0 or current > len(side_info) or current in seen:
                return None
            seen.add(current)
            row = side_info[current - 1]
            face_id = int(row[3]) // 10
            if int(row[2]) > 0 and face_id in face_nodes:
                return row
            current = abs(int(row[1]))
        return None

    try:
        with h5py.File(mesh_path, "r") as handle:
            nodes = np.asarray(handle["NodeCoords"], dtype=float)
            elem_info = np.asarray(handle["ElemInfo"], dtype=int)
            side_info = np.asarray(handle["SideInfo"], dtype=int)
            bc_names = [bytes(name).decode("utf-8", errors="ignore").strip("\x00").strip() for name in handle["BCNames"][:]]
    except Exception:
        return None

    mesh_boundary_name = os.path.basename(mesh_path).split("_", 1)[0].upper()
    object_boundary_names = {"OBJ"}
    if mesh_boundary_name:
        object_boundary_names.add(mesh_boundary_name)
    obj_bc_ids = [
        idx + 1
        for idx, name in enumerate(bc_names)
        if name.upper() in object_boundary_names
    ]
    if not obj_bc_ids:
        return None

    for boundary_vtu in _candidate_hopr_boundary_vtus(mesh_path):
        area = _half_total_reference_area_from_hopr_boundary_vtu(boundary_vtu, obj_bc_ids)
        if area is not None:
            return area

    total_area = 0.0
    for boundary_row in side_info[np.isin(side_info[:, 4], obj_bc_ids)]:
        side_row = resolve_side(side_info, int(boundary_row[1]))
        if side_row is None:
            continue
        elem_idx = int(side_row[2]) - 1
        face_id = int(side_row[3]) // 10
        if elem_idx < 0 or elem_idx >= len(elem_info):
            continue
        start, stop = int(elem_info[elem_idx, 4]), int(elem_info[elem_idx, 5])
        elem_nodes = nodes[start:stop]
        if elem_nodes.shape[0] < 8:
            continue
        points = elem_nodes[face_nodes[face_id]]
        area_vector = np.zeros(3, dtype=float)
        origin = points[0]
        for point_idx in range(1, points.shape[0] - 1):
            area_vector += 0.5 * np.cross(points[point_idx] - origin, points[point_idx + 1] - origin)
        total_area += float(np.linalg.norm(area_vector))

    if not np.isfinite(total_area) or total_area <= 0.0:
        return None
    return float(max(0.5 * total_area, 1e-12))


def _attach_piclas_reference_area(payload: Dict[str, Any]) -> None:
    explicit = _first_present(
        payload,
        {},
        ["reference_area_m2", "piclas_reference_area_m2", "area_ref_m2", "A_ref"],
        None,
    )
    explicit_value = _to_float(explicit, None)
    if explicit_value is not None and np.isfinite(explicit_value) and explicit_value > 0.0:
        payload["reference_area_m2"] = float(explicit_value)
        payload["piclas_reference_area_m2"] = float(explicit_value)
        payload.setdefault("reference_area_source", "explicit_payload")
        return

    mesh_path = _resolve_hf_mesh_path(payload.get("hf_mesh"))
    if mesh_path is None:
        return
    area = None
    if mesh_path.lower().endswith(".h5"):
        area = _piclas_h5_half_total_reference_area(mesh_path)
        if area is None:
            raise RuntimeError(
                f"Could not derive PICLas reference_area_m2 from HDF5 mesh '{mesh_path}'. "
                "Install h5py in the runtime environment and verify the mesh contains "
                "NodeCoords, ElemInfo, SideInfo, and BCNames with an OBJ boundary "
                "or a mesh-named object boundary such as CUBE."
            )
    else:
        area = _half_total_reference_area_from_mesh(mesh_path)
    if area is None:
        return
    payload["reference_area_m2"] = float(area)
    payload["piclas_reference_area_m2"] = float(area)
    payload["reference_area_source"] = "piclas_half_total_hf_mesh"
    payload["reference_area_mesh"] = mesh_path


def _estimate_freestream_speed_mps(altitude_km: Any) -> float:
    altitude_m = max(0.0, float(_to_float(altitude_km, 0.0) or 0.0) * 1000.0)
    return float(np.sqrt(_EARTH_MU_M3PS2 / (_EARTH_RADIUS_M + altitude_m)))


def _resolve_effective_attitude_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    wind_vec = payload.get("wind_enu_mps")
    try:
        wind = np.asarray(wind_vec, dtype=float).reshape(-1)[:3] if wind_vec is not None else None
    except Exception:
        wind = None
    if wind is None or wind.size < 3:
        return None
    wind = np.nan_to_num(wind, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.linalg.norm(wind)) <= 1e-12:
        return None

    nominal_aos = float(_to_float(payload.get("nominal_aos_deg", payload.get("aos_deg", 0.0)), 0.0) or 0.0)
    nominal_aoa = float(_to_float(payload.get("nominal_aoa_deg", payload.get("aoa_deg", 0.0)), 0.0) or 0.0)
    flow_speed = _to_float(payload.get("flow_speed_mps", payload.get("freestream_speed_mps")), None)
    if flow_speed is None or not np.isfinite(flow_speed) or flow_speed <= 0:
        flow_speed = _estimate_freestream_speed_mps(payload.get("altitude_km", 0.0))

    flow_zero_direction = payload.get(
        "flow_zero_direction",
        payload.get("flow_zero_direction_xyz", payload.get("zero_flow_direction", payload.get("zero_flow_direction_xyz"))),
    )
    rel_flow = _flow_unit_from_angles(nominal_aos, nominal_aoa, flow_zero_direction) * float(flow_speed) - wind
    return _angles_from_flow_vector(rel_flow, flow_zero_direction)


def _resolve_space_weather(sample: Dict[str, Any], metadata: Dict[str, Any], regime: RegimeDescriptor) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Regime-level defaults (lowest priority)
    regime_sw = regime.descriptors.get("space_weather")
    if isinstance(regime_sw, dict):
        out.update(regime_sw)
    for k in ["f107", "f107a", "ap", "aps", "kp", "dst"]:
        if k in regime.descriptors:
            out[k] = regime.descriptors.get(k)

    # Metadata defaults and sample overrides
    meta_sw = metadata.get("space_weather")
    if isinstance(meta_sw, dict):
        out.update(meta_sw)
    sample_sw = sample.get("space_weather")
    if isinstance(sample_sw, dict):
        out.update(sample_sw)

    aliases = {
        "f107": ["f107", "space_weather_f107"],
        "f107a": ["f107a", "space_weather_f107a"],
        "ap": ["ap", "daily_ap", "ap_daily", "space_weather_ap"],
        "aps": ["aps", "ap_vector", "ap_history", "ap_3h", "ap3h", "space_weather_aps"],
        "kp": ["kp", "kp_index", "space_weather_kp"],
        "dst": ["dst", "dst_index", "space_weather_dst"],
    }
    for key, names in aliases.items():
        value = _first_present(sample, metadata, names, None)
        if value is not None:
            out[key] = value

    solar_state = str(regime.descriptors.get("solar_activity_state", "")).strip().lower()
    geom_state = str(regime.descriptors.get("geomagnetic_activity_state", "")).strip().lower()
    solar_to_f107 = {
        "quiet": 70.0,
        "low": 90.0,
        "moderate": 130.0,
        "elevated": 150.0,
        "high": 190.0,
        "storm": 220.0,
        "disturbed": 170.0,
    }
    geom_to_ap = {
        "quiet": 4.0,
        "low": 6.0,
        "moderate": 12.0,
        "elevated": 18.0,
        "high": 24.0,
        "disturbed": 30.0,
        "storm": 50.0,
    }

    if "f107" not in out and solar_state in solar_to_f107:
        out["f107"] = solar_to_f107[solar_state]
    if "f107a" not in out and "f107" in out:
        out["f107a"] = out["f107"]
    if "ap" not in out and geom_state in geom_to_ap:
        out["ap"] = geom_to_ap[geom_state]

    return out


def _space_weather_value_from_payload(payload: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    sample_sw = sample.get("space_weather", {})
    sample_sw = sample_sw if isinstance(sample_sw, dict) else {}
    payload_sw = payload.get("space_weather", {})
    payload_sw = payload_sw if isinstance(payload_sw, dict) else {}

    for key in keys:
        if key in sample_sw:
            return sample_sw[key]
        if key in sample:
            return sample[key]
        if key in payload_sw:
            return payload_sw[key]
        if key in payload:
            return payload[key]
    return default


def _space_weather_numeric_from_payload(payload: Dict[str, Any], keys: List[str], default: Optional[float]) -> Optional[float]:
    value = _space_weather_value_from_payload(payload, keys, default)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return default


def _space_weather_ap_vector_from_payload(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    raw = _space_weather_value_from_payload(payload, ["aps", "ap_vector", "ap_history", "ap_3h", "ap3h"], None)
    if raw is None:
        ap_scalar = _space_weather_numeric_from_payload(payload, ["ap", "ap_daily", "daily_ap"], None)
        if ap_scalar is None:
            return None
        return np.full(7, float(ap_scalar), dtype=float)

    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.size == 1:
        return np.full(7, float(arr[0]), dtype=float)
    if arr.size < 7:
        arr = np.concatenate([arr, np.full(7 - arr.size, float(arr[-1]), dtype=float)])
    if arr.size > 7:
        arr = arr[:7]
    return arr


def _resolve_pymsis_callable(ps: Any) -> Any:
    candidates = [
        getattr(ps, "calculate", None),
        getattr(ps, "run", None),
        getattr(getattr(ps, "msis", None), "calculate", None),
        getattr(getattr(ps, "msis", None), "run", None),
    ]
    try:
        msis_mod = importlib.import_module("pymsis.msis")
    except Exception:
        msis_mod = None
    if msis_mod is not None:
        candidates.extend([getattr(msis_mod, "calculate", None), getattr(msis_mod, "run", None)])
    for candidate in candidates:
        if callable(candidate):
            return candidate
    available = sorted(name for name in dir(ps) if not name.startswith("_"))
    raise AttributeError(f"pymsis callable not found; available top-level attributes: {available}")


def _sample_pymsis_row_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    try:
        import pymsis as ps  # type: ignore
    except Exception as exc:
        raise ImportError("pymsis is required for environment_model='pymsis_hwm14'") from exc

    altitude_km = float(payload.get("altitude_km", 0.0))
    lat_deg = float(payload.get("lat_deg", 0.0))
    lon_deg = float(payload.get("lon_deg", 0.0))
    datetime_utc = str(payload.get("datetime_utc", "2006-01-01T00:00"))
    dt64 = np.datetime64(datetime_utc)

    f107 = _space_weather_numeric_from_payload(payload, ["f107"], None)
    f107a = _space_weather_numeric_from_payload(payload, ["f107a"], f107)
    aps = _space_weather_ap_vector_from_payload(payload)

    calculate_args = (
        np.array([dt64]),
        np.array([lon_deg], dtype=float),
        np.array([lat_deg], dtype=float),
        np.array([altitude_km], dtype=float),
    )
    kwargs: Dict[str, Any] = {}
    if f107 is not None:
        kwargs["f107s"] = np.array([float(f107)], dtype=float)
    if f107a is not None:
        kwargs["f107as"] = np.array([float(f107a)], dtype=float)
    if aps is not None:
        kwargs["aps"] = np.asarray(aps, dtype=float).reshape(1, -1)

    calculate = _resolve_pymsis_callable(ps)
    try:
        out = calculate(*calculate_args, **kwargs) if kwargs else calculate(*calculate_args)
    except Exception:
        out = calculate(*calculate_args)
    row = np.asarray(out, dtype=float).reshape(-1, np.asarray(out).shape[-1])[0]
    if row.shape[0] < 11:
        raise ValueError(f"Unexpected pymsis output shape: {row.shape}")
    context = (
        f"pymsis payload altitude_km={altitude_km}, lat_deg={lat_deg}, lon_deg={lon_deg}, "
        f"datetime_utc={datetime_utc}, f107={f107}, f107a={f107a}, aps={aps.tolist() if aps is not None else None}"
    )
    return _validate_atmosphere_row(row[:11], context)


def _ensure_shared_atmosphere_row(payload: Dict[str, Any]) -> None:
    if str(payload.get("environment_model", "csv")) != "pymsis_hwm14":
        return
    if "atmosphere_row" in payload:
        return

    cache_key_payload = {
        "datetime_utc": payload.get("datetime_utc"),
        "lat_deg": payload.get("lat_deg"),
        "lon_deg": payload.get("lon_deg"),
        "altitude_km": payload.get("altitude_km"),
        "f107": payload.get("f107"),
        "f107a": payload.get("f107a"),
        "aps": payload.get("aps"),
        "ap": payload.get("ap"),
    }
    cache_key = json.dumps(cache_key_payload, sort_keys=True, default=str)

    cached = _PYMSIS_ROW_CACHE.get(cache_key)
    if cached is None:
        row = _sample_pymsis_row_from_payload(payload)
        cached = [float(v) for v in _validate_atmosphere_row(row, f"cache_key={cache_key}")]
        _PYMSIS_ROW_CACHE[cache_key] = cached
    else:
        cached = [float(v) for v in _validate_atmosphere_row(cached, f"cache_key={cache_key}")]

    payload["atmosphere_row"] = list(cached)
    payload["atmosphere_source"] = "pymsis_shared"


def _build_environment_payload(
    sample: Dict[str, Any],
    regime: RegimeDescriptor,
    metadata: Dict[str, Any],
    geometry: Optional[GeometryDescriptor] = None,
    *,
    attach_piclas_reference_area: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "environment_model": str(metadata.get("environment_model", metadata.get("env_model", "csv"))),
        "altitude_km": regime.descriptors.get("altitude_km"),
        "database_index": sample.get("database_index", 0),
        "sample": dict(sample),
        "regime_descriptors": dict(regime.descriptors),
    }
    for k in ["datetime_utc", "lat_deg", "lon_deg", "local_time_class", "orbit_class", "atmosphere_row"]:
        if k in regime.descriptors:
            payload[k] = regime.descriptors.get(k)
    if geometry is not None:
        payload["geometry_id"] = geometry.geometry_id
        payload["geometry_name"] = geometry.name
        payload["geometry_class"] = geometry.geometry_class
        if isinstance(geometry.metadata, dict):
            if "lf_model" in geometry.metadata:
                payload["lf_model"] = geometry.metadata.get("lf_model")
            if "hf_mesh" in geometry.metadata:
                payload["hf_mesh"] = geometry.metadata.get("hf_mesh")
            for key in _PICLAS_MPF_OVERRIDE_KEYS:
                if key in geometry.metadata:
                    payload[key] = geometry.metadata[key]

    # Allow metadata-level defaults and sample-level overrides for common env inputs.
    for key in [
        "altitude_km",
        "datetime_utc",
        "lat_deg",
        "lon_deg",
        "local_solar_time_h",
        "relative_speed_mps",
        "flow_speed_mps",
        "freestream_speed_mps",
        "f107",
        "f107a",
        "ap",
        "aps",
        "kp",
        "dst",
        "space_weather",
        "density_scale",
        "density_state_scale",
        "density_state_temperature_offset_k",
        "density_state_composition_shift",
        "temperature_scale",
        "temperature_offset_k",
        "composition_shift",
        "composition_delta",
        "wind_enu_mps",
        "use_winds",
        "apply_wind_to_speed",
        "flow_zero_direction",
        "flow_zero_direction_xyz",
        "zero_flow_direction",
        "zero_flow_direction_xyz",
        "adbsat_aos_offset_deg",
        "adbsat_aos_offset",
        "energy_accommodation",
        "alpha",
        "alphaN",
        "sigmaN",
        "sigmaT",
        "surface_temperature_k",
        "wall_temperature_k",
        "surface_state",
        "trans_accommodation",
        "momentum_accommodation",
        "vib_accommodation",
        "rot_accommodation",
        "aos_deg",
        "aoa_deg",
        "attitude_jitter_deg",
        "jitter_deg",
        "jitter_aos_deg",
        "jitter_aoa_deg",
        "geometry_id",
        "geometry_name",
        "geometry_class",
        "lf_model",
        "hf_mesh",
        "domain_side_length_m",
        "domain_size_m",
        "domain_side_m",
        "piclas_domain_side_m",
        "reference_area_m2",
        "piclas_reference_area_m2",
        "area_ref_m2",
        "A_ref",
        "atmosphere_row",
        "trajectory_index",
    ]:
        if key in metadata:
            payload[key] = metadata[key]
        if key in sample:
            payload[key] = sample[key]

    for key in _PICLAS_MPF_OVERRIDE_KEYS:
        if key in metadata:
            payload[key] = metadata[key]

    # Resolve per-sample attitude so AoS/AoA uncertainties are carried end-to-end.
    attitude = _resolve_attitude(sample, metadata)
    payload["attitude"] = {
        "aos_deg": attitude["aos_deg"],
        "aoa_deg": attitude["aoa_deg"],
        "jitter_aos_deg": attitude["jitter_aos_deg"],
        "jitter_aoa_deg": attitude["jitter_aoa_deg"],
    }
    payload["aos_deg"] = attitude["aos_deg"]
    payload["aoa_deg"] = attitude["aoa_deg"]
    payload["jitter_aos_deg"] = attitude["jitter_aos_deg"]
    payload["jitter_aoa_deg"] = attitude["jitter_aoa_deg"]
    payload["attitude_jitter_deg"] = attitude["jitter_scalar_deg"]

    # Compose vector wind payload from scalar components if provided.
    wind_e = _first_present(sample, metadata, ["wind_east_mps", "wind_e_mps"], None)
    wind_n = _first_present(sample, metadata, ["wind_north_mps", "wind_n_mps"], None)
    wind_u = _first_present(sample, metadata, ["wind_up_mps", "wind_u_mps"], None)
    if wind_e is not None or wind_n is not None or wind_u is not None:
        payload["wind_enu_mps"] = [
            float(_to_float(wind_e, 0.0) or 0.0),
            float(_to_float(wind_n, 0.0) or 0.0),
            float(_to_float(wind_u, 0.0) or 0.0),
        ]

    # Surface-state defaults are taken from regime descriptors unless explicitly overridden.
    payload.setdefault("surface_state", regime.descriptors.get("surface_state"))

    # Canonical space-weather packet used by pymsis/hwm14 forcing.
    sw = _resolve_space_weather(sample, metadata, regime)
    if sw:
        payload["space_weather"] = sw
        for key in ["f107", "f107a", "ap", "aps", "kp", "dst"]:
            if key in sw:
                payload[key] = sw[key]

    payload = _sanitize_payload_value(payload)

    payload["nominal_aos_deg"] = float(_to_float(payload.get("aos_deg", 0.0), 0.0) or 0.0)
    payload["nominal_aoa_deg"] = float(_to_float(payload.get("aoa_deg", 0.0), 0.0) or 0.0)
    effective_attitude = _resolve_effective_attitude_from_payload(payload)
    if effective_attitude is not None:
        payload["effective_aos_deg"] = effective_attitude["aos_deg"]
        payload["effective_aoa_deg"] = effective_attitude["aoa_deg"]
        payload["relative_flow_speed_mps"] = effective_attitude["relative_flow_speed_mps"]
        payload["aos_deg"] = effective_attitude["aos_deg"]
        payload["aoa_deg"] = effective_attitude["aoa_deg"]
        attitude_block = payload.get("attitude")
        if isinstance(attitude_block, dict):
            attitude_block["nominal_aos_deg"] = payload["nominal_aos_deg"]
            attitude_block["nominal_aoa_deg"] = payload["nominal_aoa_deg"]
            attitude_block["effective_aos_deg"] = payload["effective_aos_deg"]
            attitude_block["effective_aoa_deg"] = payload["effective_aoa_deg"]

    if attach_piclas_reference_area:
        _attach_piclas_reference_area(payload)

    # For pymsis-driven runs, pin one shared atmosphere sample into the payload
    # so LF/HF use identical density/temperature inputs for correlation estimates.
    _ensure_shared_atmosphere_row(payload)

    return payload


class BaseModelAdapter:
    def __init__(self, model_id: str, fidelity: str, available_qois: List[str]):
        self.model_id = model_id
        self.fidelity = fidelity
        self.available_qois = set(available_qois)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        raise NotImplementedError


class MockModelAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, fidelity: str, available_qois: List[str]):
        super().__init__(model_id=model_id, fidelity=fidelity, available_qois=available_qois)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        rng = np.random.default_rng(request.seed)
        values_by_qoi: Dict[str, List[float]] = {q: [] for q in request.qois}
        costs: List[float] = []

        model_bias = (abs(hash(self.model_id)) % 1000) / 1e6
        fidelity_noise = 4e-5 if self.fidelity == "hf" else 2e-4
        cost_base = 5.0 if self.fidelity == "hf" else 0.15

        for sample in request.samples:
            numeric_vals = [float(v) for v in sample.values() if isinstance(v, (int, float, np.integer, np.floating))]
            x = float(np.mean(numeric_vals)) if numeric_vals else 0.0

            cd = 0.45 + 2e-4 * x + model_bias + rng.normal(0.0, fidelity_noise)
            cl = 0.01 * np.sin(0.001 * x) + rng.normal(0.0, fidelity_noise * 0.5)
            cmx = 0.001 * np.cos(0.0015 * x) + rng.normal(0.0, fidelity_noise * 0.2)
            cmy = 0.001 * np.sin(0.0012 * x) + rng.normal(0.0, fidelity_noise * 0.2)
            cmz = 0.001 * np.sin(0.0018 * x) + rng.normal(0.0, fidelity_noise * 0.2)

            qoi_map = {
                "C_D": float(cd),
                "C_D2": float(cd * cd),
                "C_L": float(cl),
                "C_L2": float(cl * cl),
                "C_Y": float(cmy),
                "C_Y2": float(cmy * cmy),
                "C_Mx": float(cmx),
                "C_My": float(cmy),
                "C_Mz": float(cmz),
                "C_Mz2": float(cmz * cmz),
            }
            for q in request.qois:
                values_by_qoi[q].append(qoi_map.get(q, float("nan")))

            costs.append(float(cost_base + abs(rng.normal(0.0, 0.02 * cost_base))))

        return EvaluationResult(values_by_qoi=values_by_qoi, costs=costs, sample_ids=request.sample_ids)


class LegacyPiclasAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, available_qois: List[str], kwargs: Dict[str, Any], fidelity: str = "hf"):
        super().__init__(model_id=model_id, fidelity=fidelity, available_qois=available_qois)

        sim_kwargs = dict(kwargs)
        simulator_module = str(sim_kwargs.pop("simulator_module", "PICLas"))
        payload_defaults = sim_kwargs.pop("payload_defaults", sim_kwargs.pop("environment_payload_defaults", {}))
        if payload_defaults is None:
            payload_defaults = {}
        if not isinstance(payload_defaults, dict):
            raise TypeError(f"PICLas payload_defaults must be a mapping for model '{model_id}'")
        self.payload_defaults = dict(payload_defaults)
        for key in list(sim_kwargs):
            if key in _PICLAS_PAYLOAD_DEFAULT_KEYS:
                self.payload_defaults.setdefault(key, sim_kwargs.pop(key))
        self.surface_archive_config = sim_kwargs.pop(
            "surface_archive",
            sim_kwargs.pop("field_surface_archive", {}),
        )
        if self.surface_archive_config is None:
            self.surface_archive_config = {}
        if not isinstance(self.surface_archive_config, dict):
            raise TypeError(f"surface_archive config must be a mapping for model '{model_id}'")
        self.piclas_mode = str(sim_kwargs.get("piclas_mode", self.payload_defaults.get("piclas_mode", ""))).strip().lower()
        if self.piclas_mode in {"tpmc", "collisionless", "free_molecular", "free-molecular"}:
            self.payload_defaults.setdefault("piclas_mode", "tpmc")
            self.payload_defaults.setdefault("t_end_s", 1.0e-4)
            self.payload_defaults.setdefault("sampling_iterations", 250)
            # One TPMC SLURM job processes ten cases sequentially by default.
            sim_kwargs.setdefault("submission_group_size", 10)
        else:
            self.payload_defaults.setdefault("piclas_mode", "dsmc")
            self.payload_defaults.setdefault("t_end_s", 1.0e-3)
            self.payload_defaults.setdefault("sampling_iterations", 2500)
            # DSMC cases are independent SLURM jobs by default.
            sim_kwargs.setdefault("submission_group_size", 1)
        PiclasSimulator = getattr(importlib.import_module(simulator_module), "PiclasSimulator")
        try:
            signature = inspect.signature(PiclasSimulator.__init__)
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if not accepts_kwargs and "piclas_mode" not in signature.parameters:
                sim_kwargs.pop("piclas_mode", None)
        except (TypeError, ValueError):
            pass
        batch_size = sim_kwargs.pop("submission_batch_size", None)
        self.submission_batch_size = max(1, int(batch_size)) if batch_size is not None else None
        self.sim = PiclasSimulator(**sim_kwargs)

    def _prepare_batch_request(self, request: EvaluationRequest):
        altitude = int(request.regime.descriptors["altitude_km"])
        aos = float(request.metadata.get("aos_deg", 0.0))
        aoa = float(request.metadata.get("aoa_deg", 0.0))
        indices = [int(s.get("database_index", 0)) for s in request.samples]
        env_model = str(request.metadata.get("environment_model", request.metadata.get("env_model", "csv")))

        payload_dir = tempfile.mkdtemp(prefix="mfmc_piclas_env_")
        env_payload_paths: List[str] = []
        aos_values: List[float] = []
        aoa_values: List[float] = []
        random_seeds: List[int] = []
        for pos, sample in enumerate(request.samples):
            payload = _build_environment_payload(sample, request.regime, request.metadata, request.geometry)
            _strip_sample_piclas_numerical_controls(payload)
            for key, value in getattr(self, "payload_defaults", {}).items():
                payload.setdefault(key, _sanitize_payload_value(value))
            payload["environment_model"] = env_model
            aos_values.append(float(payload.get("aos_deg", aos)))
            aoa_values.append(float(payload.get("aoa_deg", aoa)))
            sample_seed = sample.get("operations.seed", sample.get("seed", sample.get("random_seed", None)))
            if sample_seed is None:
                sample_seed = int((int(request.seed) + pos + 1) % (2**31 - 1))
            try:
                seed_value = int(sample_seed)
            except Exception:
                seed_value = int((int(request.seed) + pos + 1) % (2**31 - 1))
            random_seeds.append(seed_value)
            payload_path = os.path.join(payload_dir, f"env_{pos}.json")
            with open(payload_path, "w", encoding="utf-8") as pf:
                json.dump(payload, pf)
            env_payload_paths.append(payload_path)

        return altitude, aos, aoa, indices, env_model, payload_dir, env_payload_paths, aos_values, aoa_values, random_seeds

    def submit(self, request: EvaluationRequest):
        altitude, aos, _aoa, indices, env_model, payload_dir, env_payload_paths, aos_values, aoa_values, random_seeds = self._prepare_batch_request(request)
        try:
            if hasattr(self.sim, "submit_batch_jobs"):
                batch_handle = self.sim.submit_batch_jobs(
                    altitude,
                    aos,
                    indices,
                    env_payload_paths=env_payload_paths,
                    env_model=env_model,
                    aos_values=aos_values,
                    aoa_values=aoa_values,
                    random_seeds=random_seeds,
                    geometry_id=request.geometry.geometry_id,
                    geometry_mesh=request.metadata.get("hf_mesh", request.geometry.metadata.get("hf_mesh")),
                    flow_zero_direction=request.metadata.get(
                        "flow_zero_direction",
                        request.metadata.get(
                            "flow_zero_direction_xyz",
                            request.metadata.get("zero_flow_direction", request.metadata.get("zero_flow_direction_xyz")),
                        ),
                    ),
                )
            else:
                qoi_values, cpu_hours_list = self.sim.run_batch_qois(
                    altitude,
                    aos,
                    indices,
                    int(request.seed),
                    requested_qois=list(request.qois),
                    env_payload_paths=env_payload_paths,
                    env_model=env_model,
                    aos_values=aos_values,
                    aoa_values=aoa_values,
                    geometry_id=request.geometry.geometry_id,
                    geometry_mesh=request.metadata.get("hf_mesh", request.geometry.metadata.get("hf_mesh")),
                )
                batch_handle = {
                    "_completed_result": EvaluationResult(
                        values_by_qoi={q: list(np.asarray(qoi_values.get(q, []), dtype=float)) for q in request.qois},
                        costs=list(np.asarray(cpu_hours_list, dtype=float)),
                        sample_ids=list(request.sample_ids),
                    )
                }
        finally:
            shutil.rmtree(payload_dir, ignore_errors=True)

        batch_handle["requested_qois"] = list(request.qois)
        batch_handle["sample_ids"] = list(request.sample_ids)
        batch_handle["random_seed"] = int(request.seed)
        batch_handle["surface_archive_context"] = {
            "study_id": request.study_id,
            "cell_id": request.cell_id,
            "model_id": request.model_id,
            "adapter_model_id": self.model_id,
            "adapter_fidelity": self.fidelity,
            "piclas_mode": getattr(self, "piclas_mode", ""),
            "geometry_id": request.geometry.geometry_id,
            "regime_id": request.regime.regime_id,
            "case_name": request.metadata.get(
                "case_name",
                request.metadata.get(
                    "case_id",
                    f"{request.geometry.geometry_id}-{request.regime.regime_id}",
                ),
            ),
        }
        return batch_handle

    def wait(self, batch_handle):
        self.sim.wait_for_batch_jobs(batch_handle, max_retries=2)
        return batch_handle

    def submit_postprocessing(self, batch_handles, random_seed=None, wait_for_completion=False):
        handles = [batch_handles] if isinstance(batch_handles, dict) else list(batch_handles)
        if random_seed is None:
            random_seed = int(handles[0].get("random_seed", 0)) if handles else 0
        return self.sim.submit_batch_postprocessing(
            handles,
            int(random_seed),
            wait_for_completion=wait_for_completion,
        )

    def wait_postprocessing(self, postprocess_handle):
        self.sim.wait_for_postprocessing(postprocess_handle)
        return postprocess_handle

    def collect_outputs(self, batch_handle) -> EvaluationResult:
        requested_qois = list(batch_handle.get("requested_qois", []))
        qoi_values, cpu_hours_list = self.sim.collect_batch_results(
            batch_handle,
            requested_qois=requested_qois,
        )
        metadata: Dict[str, Any] = {}
        archive_summary = self._maybe_export_surface_archive(batch_handle)
        if archive_summary:
            metadata["surface_archive"] = archive_summary
        return EvaluationResult(
            values_by_qoi={q: list(np.asarray(qoi_values.get(q, []), dtype=float)) for q in requested_qois},
            costs=list(np.asarray(cpu_hours_list, dtype=float)),
            sample_ids=list(batch_handle.get("sample_ids", [])),
            metadata=metadata,
        )

    def _surface_archive_fidelity_label(self) -> str:
        cfg = getattr(self, "surface_archive_config", {})
        explicit = cfg.get("fidelity")
        if explicit:
            return str(explicit).upper()
        if getattr(self, "piclas_mode", "") in {"tpmc", "collisionless", "free_molecular", "free-molecular"}:
            return "TPMC"
        return "DSMC"

    def _surface_archive_output_path(self, batch_handle: Dict[str, Any], fidelity_label: str) -> Path:
        cfg = getattr(self, "surface_archive_config", {})
        if cfg.get("path"):
            return Path(str(cfg["path"]))
        context = batch_handle.get("surface_archive_context", {})
        case_name = str(cfg.get("case_name", context.get("case_name", "piclas_case")))
        output_dir = Path(str(cfg.get("output_dir", "paper_postprocessed/field_inputs"))) / _slug_for_path(case_name)
        filename = str(cfg.get("filename", f"{fidelity_label}_surface_loads.npz"))
        return output_dir / filename

    def _maybe_export_surface_archive(self, batch_handle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cfg = getattr(self, "surface_archive_config", {})
        if not bool(cfg.get("enabled", False)):
            return None
        context = batch_handle.get("surface_archive_context", {})
        fidelity_label = self._surface_archive_fidelity_label()
        output_path = self._surface_archive_output_path(batch_handle, fidelity_label)
        case_name = str(cfg.get("case_name", context.get("case_name", "piclas_case")))
        summary = export_piclas_surface_archive(
            self.sim,
            batch_handle,
            output_path,
            fidelity=fidelity_label,
            model_id=str(cfg.get("model_id", self.model_id)),
            case_name=case_name,
            geometry_id=str(cfg.get("geometry_id", context.get("geometry_id", ""))),
            regime_id=str(cfg.get("regime_id", context.get("regime_id", ""))),
            append=bool(cfg.get("append", True)),
        )
        print(
            "[surface-archive] "
            f"{self.model_id} wrote {summary.get('n_samples')} samples to {summary.get('path')}",
            flush=True,
        )
        return summary

    def collect(self, batch_handle) -> EvaluationResult:
        if "_completed_result" in batch_handle:
            return batch_handle["_completed_result"]
        self.wait(batch_handle)
        postprocess_handle = self.submit_postprocessing(
            batch_handle,
            random_seed=batch_handle.get("random_seed", 0),
            wait_for_completion=True,
        )
        self.wait_postprocessing(postprocess_handle)
        return self.collect_outputs(batch_handle)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        batch_size = getattr(self, "submission_batch_size", None) or max(len(request.samples), 1)
        values_by_qoi = {q: [] for q in request.qois}
        costs_list: List[float] = []

        for start in range(0, len(request.samples), batch_size):
            stop = min(start + batch_size, len(request.samples))
            chunk_request = EvaluationRequest(
                study_id=request.study_id,
                cell_id=request.cell_id,
                model_id=request.model_id,
                fidelity=request.fidelity,
                qois=list(request.qois),
                geometry=request.geometry,
                regime=request.regime,
                active_source_blocks=list(request.active_source_blocks),
                sample_ids=list(request.sample_ids[start:stop]),
                samples=list(request.samples[start:stop]),
                seed=request.seed,
                metadata=dict(request.metadata),
            )
            result = self.collect(self.submit(chunk_request))
            costs_list.extend(list(np.asarray(result.costs, dtype=float)))
            for q in request.qois:
                arr = np.asarray(result.values_by_qoi.get(q, []), dtype=float)
                if q == "C_D2" and arr.size == 0 and "C_D" in result.values_by_qoi:
                    cd = np.asarray(result.values_by_qoi.get("C_D", []), dtype=float)
                    arr = cd * cd
                if q == "C_L2" and arr.size == 0 and "C_L" in result.values_by_qoi:
                    cl = np.asarray(result.values_by_qoi.get("C_L", []), dtype=float)
                    arr = cl * cl
                if q == "C_Mz2" and arr.size == 0 and "C_Mz" in result.values_by_qoi:
                    cmz = np.asarray(result.values_by_qoi.get("C_Mz", []), dtype=float)
                    arr = cmz * cmz
                if q == "C_Y2" and arr.size == 0 and "C_Y" in result.values_by_qoi:
                    cy = np.asarray(result.values_by_qoi.get("C_Y", []), dtype=float)
                    arr = cy * cy
                values_by_qoi[q].extend(arr.tolist())

        return EvaluationResult(values_by_qoi=values_by_qoi, costs=costs_list, sample_ids=request.sample_ids)


class LegacyADBSatAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, method: str, available_qois: List[str], kwargs: Dict[str, Any]):
        super().__init__(model_id=model_id, fidelity="lf", available_qois=available_qois)

        sim_kwargs = dict(kwargs)
        simulator_module = str(sim_kwargs.pop("simulator_module", "ADBSat"))
        self.surface_archive_config = sim_kwargs.pop(
            "surface_archive",
            sim_kwargs.pop("field_surface_archive", {}),
        )
        if self.surface_archive_config is None:
            self.surface_archive_config = {}
        if not isinstance(self.surface_archive_config, dict):
            raise TypeError(f"surface_archive config must be a mapping for model '{model_id}'")
        self.surface_mapping = None
        if bool(self.surface_archive_config.get("enabled", False)):
            mapping_path = self.surface_archive_config.get("mapping_path")
            if not mapping_path:
                raise ValueError(f"ADBSat surface_archive.mapping_path is required for model '{model_id}'")
            self.surface_mapping = load_surface_mapping(mapping_path)
        self.method = method
        missing_retries = sim_kwargs.pop("missing_results_retries", sim_kwargs.pop("results_missing_retries", 1))
        self.missing_results_retries = max(0, int(missing_retries))
        ADBSatSimulator = getattr(importlib.import_module(simulator_module), "ADBSatSimulator")
        self.sim = ADBSatSimulator(method=method, **sim_kwargs)

    def _adbsat_payload(self, sample, request, env_model: str, sample_id: str) -> Dict[str, Any]:
        payload = _build_environment_payload(
            sample,
            request.regime,
            request.metadata,
            request.geometry,
            attach_piclas_reference_area=False,
        )
        payload["environment_model"] = env_model
        payload.setdefault("write_input_audit", False)
        payload.setdefault("write_mat", False)
        if self.surface_mapping is not None:
            payload["write_panel_surface_field"] = True
            payload["surface_sample_id"] = str(sample_id)
            payload["surface_mesh_fingerprint"] = self.surface_mapping.mesh_fingerprint
        # ADBSat/simulate.py reads all required values from top-level payload
        # keys. Dropping these large duplicate blocks keeps the batch file small.
        payload.pop("sample", None)
        payload.pop("regime_descriptors", None)
        return payload

    def _collect_adbsat_results(self, request: EvaluationRequest, run_ids: List[int]) -> EvaluationResult:
        values_by_qoi = {q: [] for q in request.qois}
        if hasattr(self.sim, "analyze_simulation_results_qois"):
            qoi_data, costs, ret_idx = self.sim.analyze_simulation_results_qois(run_ids, requested_qois=request.qois)
            costs = np.asarray(costs, dtype=float)
            ret_idx = np.asarray(ret_idx, dtype=int)
            ordered_costs = []
            for run_id in run_ids:
                pos = np.where(ret_idx == int(run_id))[0]
                if pos.size:
                    ordered_costs.append(float(costs[pos[0]]))
                else:
                    ordered_costs.append(float("nan"))

            for q in request.qois:
                arr = np.asarray(qoi_data.get(q, []), dtype=float)
                idx_to_val = {int(i): float(v) for i, v in zip(ret_idx, arr)}
                ordered_vals = [idx_to_val.get(int(run_id), float("nan")) for run_id in run_ids]
                if q == "C_D2" and len(arr) == 0 and "C_D" in qoi_data:
                    cd_idx_to_val = {int(i): float(v) for i, v in zip(ret_idx, np.asarray(qoi_data["C_D"], dtype=float))}
                    ordered_vals = [
                        (cd_idx_to_val[int(run_id)] ** 2)
                        if int(run_id) in cd_idx_to_val and np.isfinite(cd_idx_to_val[int(run_id)])
                        else float("nan")
                        for run_id in run_ids
                    ]
                missing = [
                    int(run_id) for run_id, value in zip(run_ids, ordered_vals)
                    if not np.isfinite(float(value))
                ]
                if missing:
                    raise ValueError(
                        f"ADBSat {self.method} returned non-finite or missing values for qoi={q}; "
                        f"missing run_ids={missing[:10]}{'...' if len(missing) > 10 else ''}"
                    )
                values_by_qoi[q] = ordered_vals
        else:
            raw = self.sim.analyze_simulation_results(run_ids)
            if not isinstance(raw, tuple):
                raise TypeError("ADBSat adapter expected tuple output")

            if len(raw) == 3:
                vals, costs, returned_indices = raw
                ret_idx = np.asarray(returned_indices, dtype=int)
            elif len(raw) == 2:
                vals, costs = raw
                ret_idx = np.asarray(run_ids[: len(vals)], dtype=int)
            else:
                raise ValueError("Unexpected ADBSat return signature")

            vals = np.asarray(vals, dtype=float)
            costs = np.asarray(costs, dtype=float)

            idx_to_pair = {int(i): (float(v), float(c)) for i, v, c in zip(ret_idx, vals, costs)}
            ordered_vals, ordered_costs = [], []
            for run_id in run_ids:
                v, c = idx_to_pair.get(int(run_id), (float("nan"), float("nan")))
                ordered_vals.append(v)
                ordered_costs.append(c)

            for cd in ordered_vals:
                qoi_map = {
                    "C_D": float(cd),
                    "C_D2": float(cd * cd) if not np.isnan(cd) else float("nan"),
                    "C_L": float("nan"),
                    "C_L2": float("nan"),
                    "C_Y": float("nan"),
                    "C_Y2": float("nan"),
                    "C_Mx": float("nan"),
                    "C_My": float("nan"),
                    "C_Mz": float("nan"),
                    "C_Mz2": float("nan"),
                }
                for q in request.qois:
                    values_by_qoi[q].append(qoi_map.get(q, float("nan")))

        metadata: Dict[str, Any] = {}
        if self.surface_mapping is not None:
            cfg = self.surface_archive_config
            case_name = str(
                cfg.get(
                    "case_name",
                    request.metadata.get("case_name", f"{request.geometry.geometry_id}-{request.regime.regime_id}"),
                )
            )
            if cfg.get("path"):
                output_path = Path(str(cfg["path"]))
            else:
                output_dir = Path(str(cfg.get("output_dir", "paper_postprocessed/field_inputs"))) / _slug_for_path(case_name)
                output_path = output_dir / str(cfg.get("filename", "SENTMAN_surface_loads.npz"))
            summary = export_adbsat_surface_archive(
                result_dir=os.path.join(self.sim.base_dir, f"MFMC_Jobs_{self.method}"),
                method=self.method,
                run_ids=run_ids,
                sample_ids=request.sample_ids,
                mapping_path=str(cfg["mapping_path"]),
                output_path=output_path,
                case_name=case_name,
                geometry_id=str(cfg.get("geometry_id", request.geometry.geometry_id)),
                regime_id=str(cfg.get("regime_id", request.regime.regime_id)),
                model_id=str(cfg.get("model_id", self.model_id)),
                append=bool(cfg.get("append", True)),
            )
            metadata["surface_archive"] = summary
            print(
                "[surface-archive] "
                f"{self.model_id} wrote {summary.get('n_samples')} samples to {summary.get('path')}",
                flush=True,
            )
        return EvaluationResult(
            values_by_qoi=values_by_qoi,
            costs=ordered_costs,
            sample_ids=request.sample_ids,
            metadata=metadata,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        altitude = int(request.regime.descriptors["altitude_km"])
        aos = int(float(request.metadata.get("aos_deg", 0)))
        database_indices = [int(s.get("database_index", 0)) for s in request.samples]
        run_ids = list(range(len(request.samples)))
        env_model = str(request.metadata.get("environment_model", request.metadata.get("env_model", "csv")))

        job_subdir = os.path.join(self.sim.base_dir, f"MFMC_Jobs_{self.method}")
        os.makedirs(job_subdir, exist_ok=True)
        payloads = []
        for pos, db_idx in enumerate(database_indices):
            sample = request.samples[pos] if pos < len(request.samples) else {"database_index": db_idx}
            payloads.append(self._adbsat_payload(sample, request, env_model, request.sample_ids[pos]))
        payloads_path = os.path.join(
            job_subdir,
            f"payloads_{int(request.seed)}_{os.getpid()}_{id(request)}.pkl",
        )
        with open(payloads_path, "wb") as pf:
            pickle.dump(payloads, pf, protocol=pickle.HIGHEST_PROTOCOL)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=job_subdir, suffix=".txt") as f:
            for pos, _db_idx in enumerate(database_indices):
                f.write(f"{self.method} {run_ids[pos]} @payload_pickle {payloads_path} {pos}\n")
            input_file = f.name

        for attempt in range(self.missing_results_retries + 1):
            job_id, _ = self.sim.queue_simulation_job(altitude, aos, input_file)
            self.sim._wait_for_job_completion(job_id)
            try:
                return self._collect_adbsat_results(request, run_ids)
            except FileNotFoundError as exc:
                if attempt >= self.missing_results_retries:
                    raise
                print(
                    f"ADBSat {self.method} results missing after job {job_id}: {exc}. "
                    f"Retrying job ({attempt + 1}/{self.missing_results_retries}).",
                    flush=True,
                )

        raise RuntimeError(f"ADBSat {self.method} retry loop exited unexpectedly")


class LegacyRaytracerAdapter(BaseModelAdapter):
    def __init__(self, model_id: str, available_qois: List[str], kwargs: Dict[str, Any]):
        super().__init__(model_id=model_id, fidelity="lf", available_qois=available_qois)
        from RayTracer import RaytracerSimulator

        self.sim = RaytracerSimulator(**kwargs)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        altitude = int(request.regime.descriptors["altitude_km"])
        aos = int(float(request.metadata.get("aos_deg", 0)))
        indices = [int(s.get("database_index", 0)) for s in request.samples]

        job_ids = self.sim.queue_simulation_job(altitude, aos, indices)
        self.sim._wait_for_job_completion(job_ids)
        raw = self.sim.analyze_simulation_results(indices)

        if not isinstance(raw, tuple):
            raise TypeError("Raytracer adapter expected tuple output")

        if len(raw) >= 2:
            vals = np.asarray(raw[0], dtype=float)
            costs = np.asarray(raw[1], dtype=float)
        else:
            raise ValueError("Unexpected Raytracer return signature")

        ret_idx = np.asarray(indices[: len(vals)], dtype=int)
        if len(raw) >= 3:
            try:
                candidate = np.asarray(raw[2], dtype=int)
                if candidate.shape[0] == vals.shape[0]:
                    ret_idx = candidate
            except Exception:
                pass

        idx_to_pair = {int(i): (float(v), float(c)) for i, v, c in zip(ret_idx, vals, costs)}
        ordered_vals, ordered_costs = [], []
        for idx in indices:
            v, c = idx_to_pair.get(int(idx), (float("nan"), float("nan")))
            ordered_vals.append(v)
            ordered_costs.append(c)

        values_by_qoi = {q: [] for q in request.qois}
        for cd in ordered_vals:
            qoi_map = {
                "C_D": float(cd),
                "C_D2": float(cd * cd) if not np.isnan(cd) else float("nan"),
                "C_L": float("nan"),
                "C_L2": float("nan"),
                "C_Y": float("nan"),
                "C_Y2": float("nan"),
                "C_Mx": float("nan"),
                "C_My": float("nan"),
                "C_Mz": float("nan"),
                "C_Mz2": float("nan"),
            }
            for q in request.qois:
                values_by_qoi[q].append(qoi_map.get(q, float("nan")))

        return EvaluationResult(values_by_qoi=values_by_qoi, costs=ordered_costs, sample_ids=request.sample_ids)


@dataclass
class AdapterRegistry:
    hf: BaseModelAdapter
    lfs: Dict[str, BaseModelAdapter]

    def get(self, model_id: str) -> BaseModelAdapter:
        if self.hf.model_id == model_id:
            return self.hf
        if model_id in self.lfs:
            return self.lfs[model_id]
        raise KeyError(f"Unknown model id '{model_id}'")


def _available_qois(model_id: str, model_qoi_map: Dict[str, Any], fallback: List[str]) -> List[str]:
    vals = model_qoi_map.get(model_id)
    if vals is None:
        return list(fallback)
    return [str(v) for v in vals]


def build_adapter_registry(config: Dict[str, Any]) -> AdapterRegistry:
    backend = str(config.get("execution", {}).get("backend", "mock"))
    model_qoi_map = config.get("models", {}).get("available_qois", {})
    requested_direct = [q for q in config.get("qois", {}).get("direct", []) if isinstance(q, str)]

    hf_cfg = config.get("models", {}).get("hf", {})
    hf_id = str(hf_cfg.get("id", "hf"))
    hf_qois = _available_qois(hf_id, model_qoi_map, requested_direct)

    if backend == "mock":
        hf = MockModelAdapter(hf_id, "hf", hf_qois)
        lfs: Dict[str, BaseModelAdapter] = {}
        for lf_cfg in config.get("models", {}).get("lf", []):
            lf_id = str(lf_cfg.get("id", "lf"))
            lf_qois = _available_qois(lf_id, model_qoi_map, requested_direct)
            lfs[lf_id] = MockModelAdapter(lf_id, "lf", lf_qois)
        return AdapterRegistry(hf=hf, lfs=lfs)

    if backend == "legacy_slurm":
        hf_kind = str(hf_cfg.get("kind", "legacy_piclas"))
        if hf_kind != "legacy_piclas":
            raise ValueError(f"Unsupported HF kind '{hf_kind}' for legacy backend")

        hf = LegacyPiclasAdapter(
            model_id=hf_id,
            available_qois=hf_qois,
            kwargs=hf_cfg.get("kwargs", {}),
            fidelity="hf",
        )

        lfs: Dict[str, BaseModelAdapter] = {}
        for lf_cfg in config.get("models", {}).get("lf", []):
            lf_id = str(lf_cfg.get("id", "lf"))
            kind = str(lf_cfg.get("kind", "legacy_adbsat"))
            lf_qois = _available_qois(lf_id, model_qoi_map, requested_direct)
            if kind == "legacy_adbsat":
                method = str(lf_cfg.get("method", lf_id))
                lfs[lf_id] = LegacyADBSatAdapter(
                    model_id=lf_id,
                    method=method,
                    available_qois=lf_qois,
                    kwargs=lf_cfg.get("kwargs", {}),
                )
            elif kind in {"legacy_piclas", "legacy_piclas_tpmc"}:
                kwargs = dict(lf_cfg.get("kwargs", {}))
                if kind == "legacy_piclas_tpmc":
                    kwargs.setdefault("piclas_mode", "tpmc")
                lfs[lf_id] = LegacyPiclasAdapter(
                    model_id=lf_id,
                    available_qois=lf_qois,
                    kwargs=kwargs,
                    fidelity="lf",
                )
            elif kind == "legacy_raytracer":
                lfs[lf_id] = LegacyRaytracerAdapter(
                    model_id=lf_id,
                    available_qois=lf_qois,
                    kwargs=lf_cfg.get("kwargs", {}),
                )
            else:
                raise ValueError(f"Unsupported LF kind '{kind}'")

        return AdapterRegistry(hf=hf, lfs=lfs)

    raise ValueError(f"Unsupported execution backend '{backend}'")


def make_request(
    study_id: str,
    cell_id: str,
    model_id: str,
    fidelity: str,
    qois: List[str],
    geometry: Dict[str, Any],
    regime: Dict[str, Any],
    active_source_blocks: List[str],
    sample_ids: List[str],
    samples: List[Dict[str, Any]],
    seed: int,
    metadata: Dict[str, Any],
) -> EvaluationRequest:
    geom = GeometryDescriptor(
        geometry_id=str(geometry.get("id", geometry.get("name", "geometry"))),
        name=str(geometry.get("name", geometry.get("id", "geometry"))),
        characteristic_length=geometry.get("characteristic_length"),
        geometry_class=geometry.get("geometry_class"),
        tags=list(geometry.get("tags", [])),
        metadata=dict(geometry.get("metadata", {})),
    )
    reg = RegimeDescriptor(
        regime_id=str(regime.get("id", regime.get("label", "regime"))),
        label=str(regime.get("label", regime.get("id", "regime"))),
        descriptors=dict(regime.get("descriptors", {})),
        metadata=dict(regime.get("metadata", {})),
    )

    return EvaluationRequest(
        study_id=study_id,
        cell_id=cell_id,
        model_id=model_id,
        fidelity=fidelity,
        qois=qois,
        geometry=geom,
        regime=reg,
        active_source_blocks=list(active_source_blocks),
        sample_ids=list(sample_ids),
        samples=samples,
        seed=seed,
        metadata=metadata,
    )
