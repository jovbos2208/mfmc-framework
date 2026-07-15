import argparse
import importlib
import json
import os
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ADBSatConstants import ConstantsData


SURFACE_STATE_PRESETS: Dict[str, Dict[str, float]] = {
    # Nominal keeps backward-compatible defaults.
    "nominal": {
        "wall_temperature_k": 300.0,
        "trans_accommodation": 0.9,
        "momentum_accommodation": 0.81,
        "vib_accommodation": 1.0,
        "rot_accommodation": 1.0,
    },
    "clean": {
        "wall_temperature_k": 295.0,
        "trans_accommodation": 0.85,
        "momentum_accommodation": 0.72,
        "vib_accommodation": 1.0,
        "rot_accommodation": 1.0,
    },
    "rough": {
        "wall_temperature_k": 320.0,
        "trans_accommodation": 0.96,
        "momentum_accommodation": 0.92,
        "vib_accommodation": 1.0,
        "rot_accommodation": 1.0,
    },
    "oxidized": {
        "wall_temperature_k": 330.0,
        "trans_accommodation": 0.98,
        "momentum_accommodation": 0.96,
        "vib_accommodation": 1.0,
        "rot_accommodation": 1.0,
    },
    "contaminated": {
        "wall_temperature_k": 300.0,
        "trans_accommodation": 0.78,
        "momentum_accommodation": 0.62,
        "vib_accommodation": 1.0,
        "rot_accommodation": 1.0,
    },
}

_PROJECT_NAME_ALIASES: Dict[str, str] = {
    "CUBE": "Cube",
    "SOAR": "SOAR",
    "GOCE": "GOCE",
    "CHAMP": "CHAMP",
}


def rotation_matrix_z(aos_deg):
    aos_rad = np.radians(aos_deg)
    return np.array(
        [
            [np.cos(aos_rad), -np.sin(aos_rad), 0],
            [np.sin(aos_rad), np.cos(aos_rad), 0],
            [0, 0, 1],
        ]
    )


def rotation_matrix_x(aoa_deg):
    aoa_rad = np.radians(aoa_deg)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(aoa_rad), -np.sin(aoa_rad)],
            [0, np.sin(aoa_rad), np.cos(aoa_rad)],
        ]
    )


def _load_payload(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _replace_nans_with_zero(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    out[np.isnan(out)] = 0.0
    return out


def _first_numeric(payload: Dict[str, Any], sample: Dict[str, Any], keys, default: float) -> float:
    for k in keys:
        if k in sample:
            try:
                return float(sample[k])
            except Exception:
                pass
        if k in payload:
            try:
                return float(payload[k])
            except Exception:
                pass
    return float(default)


def _payload_value(payload: Dict[str, Any], keys, default):
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    for k in keys:
        if k in sample:
            return sample[k]
        if k in payload:
            return payload[k]
    return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _surface_state_defaults(payload: Dict[str, Any]) -> Dict[str, float]:
    state = _payload_value(payload, ["surface_state", "gsi_surface_state"], None)
    if isinstance(state, dict):
        out: Dict[str, float] = {}
        for key, value in state.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        return out
    if state is None:
        return {}
    key = str(state).strip().lower()
    return dict(SURFACE_STATE_PRESETS.get(key, {}))


def _resolve_piclas_surface_parameters(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Project LF-specific GSI inputs into the canonical HF PICLas wall model.

    PICLas exposes one wall-interaction formulation with translational and momentum
    accommodation coefficients. We therefore treat:
      - Sentman-like `energy_accommodation` as the translational axis,
      - CLL-like `sigmaT` as the translational axis,
      - CLL-like `momentum_accommodation` as the momentum axis.

    Explicit PICLas keys (`trans_accommodation`, `momentum_accommodation`) take
    precedence. When only the translational axis is specified, momentum falls back
    to the same default curve already used by the surface presets.
    """
    surface_defaults = _surface_state_defaults(payload)

    wall_temp = float(
        _payload_value(
            payload,
            ["wall_temperature_k", "surface_temperature_k", "Tw"],
            surface_defaults.get("wall_temperature_k", 300.0),
        )
    )

    trans_default = surface_defaults.get("trans_accommodation", 0.9)
    trans_acc = _clip01(
        float(
            _payload_value(
                payload,
                ["trans_accommodation", "energy_accommodation", "alpha", "sigmaT"],
                trans_default,
            )
        )
    )

    momentum_default = surface_defaults.get("momentum_accommodation", trans_acc * trans_acc)
    momentum_acc = _clip01(
        float(
            _payload_value(
                payload,
                ["momentum_accommodation", "sigmaN", "alphaN"],
                momentum_default,
            )
        )
    )

    vib_acc = _clip01(
        float(_payload_value(payload, ["vib_accommodation"], surface_defaults.get("vib_accommodation", 1.0)))
    )
    rot_acc = _clip01(
        float(_payload_value(payload, ["rot_accommodation"], surface_defaults.get("rot_accommodation", 1.0)))
    )

    return {
        "wall_temperature_k": wall_temp,
        "trans_accommodation": trans_acc,
        "momentum_accommodation": momentum_acc,
        "vib_accommodation": vib_acc,
        "rot_accommodation": rot_acc,
    }


def _resolve_piclas_numerical_controls(payload: Dict[str, Any], computed_mpf: float) -> Dict[str, float | int]:
    t_end_s = float(
        _payload_value(
            payload,
            ["t_end_s", "tend_s", "simulation_end_time_s"],
            float("nan"),
        )
    )
    t_end_scale = float(
        _payload_value(
            payload,
            ["t_end_scale", "tend_scale", "simulation_end_time_scale"],
            float("nan"),
        )
    )
    manual_timestep_s = float(
        _payload_value(
            payload,
            ["manual_timestep_s", "manual_time_step_s", "time_step_s", "timestep_s"],
            1.0e-7,
        )
    )
    macro_particle_factor_scale = float(
        _payload_value(
            payload,
            ["macro_particle_factor_scale", "mpf_scale"],
            1.0,
        )
    )
    macro_particle_factor = float(
        _payload_value(
            payload,
            ["macro_particle_factor", "mpf_override"],
            computed_mpf * macro_particle_factor_scale,
        )
    )
    sampling_iterations = int(
        _payload_value(
            payload,
            ["sampling_iterations", "part_iteration_for_macro_val", "macro_sampling_iterations"],
            2500,
        )
    )
    octree_part_num_node = int(
        _payload_value(
            payload,
            ["octree_part_num_node", "octree_node_particles"],
            80,
        )
    )
    octree_part_num_node_min = int(
        _payload_value(
            payload,
            ["octree_part_num_node_min", "octree_node_particles_min"],
            max(1, int(round(0.75 * octree_part_num_node))),
        )
    )
    particles_mpi_weight = int(
        _payload_value(
            payload,
            ["particles_mpi_weight", "mpi_particle_weight"],
            1000,
        )
    )

    return {
        "t_end_s": float(t_end_s) if np.isfinite(t_end_s) and t_end_s > 0.0 else float("nan"),
        "t_end_scale": float(t_end_scale) if np.isfinite(t_end_scale) and t_end_scale > 0.0 else float("nan"),
        "manual_timestep_s": max(float(manual_timestep_s), 1.0e-12),
        "macro_particle_factor": max(float(macro_particle_factor), 1.0),
        "sampling_iterations": max(int(sampling_iterations), 1),
        "octree_part_num_node": max(int(octree_part_num_node), 1),
        "octree_part_num_node_min": max(int(octree_part_num_node_min), 1),
        "particles_mpi_weight": max(int(particles_mpi_weight), 1),
    }


def _space_weather_value(payload: Dict[str, Any], keys, default):
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    sample_sw = sample.get("space_weather", {})
    sample_sw = sample_sw if isinstance(sample_sw, dict) else {}
    payload_sw = payload.get("space_weather", {})
    payload_sw = payload_sw if isinstance(payload_sw, dict) else {}

    for k in keys:
        if k in sample_sw:
            return sample_sw[k]
        if k in sample:
            return sample[k]
        if k in payload_sw:
            return payload_sw[k]
        if k in payload:
            return payload[k]
    return default


def _space_weather_numeric(payload: Dict[str, Any], keys, default: Optional[float]) -> Optional[float]:
    value = _space_weather_value(payload, keys, default)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return default


def _space_weather_ap_vector(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    raw = _space_weather_value(payload, ["aps", "ap_vector", "ap_history", "ap_3h", "ap3h"], None)
    if raw is None:
        ap_scalar = _space_weather_numeric(payload, ["ap", "ap_daily", "daily_ap"], None)
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
        pad = np.full(7 - arr.size, float(arr[-1]), dtype=float)
        arr = np.concatenate([arr, pad])
    if arr.size > 7:
        arr = arr[:7]
    return arr


def _resolve_pymsis_callable(ps: Any):
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


def _resolve_attitude(payload: Dict[str, Any], default_aos_deg: float) -> tuple[float, float]:
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}

    aos_explicit = any(k in sample or k in payload for k in ["aos_deg", "aos", "beta_deg"])
    aoa_explicit = any(k in sample or k in payload for k in ["aoa_deg", "aoa", "alpha_deg"])

    aos = _first_numeric(payload, sample, ["aos_deg", "aos", "beta_deg"], float(default_aos_deg))
    aoa = _first_numeric(payload, sample, ["aoa_deg", "aoa", "alpha_deg"], 0.0)

    jitter_scalar = _first_numeric(payload, sample, ["attitude_jitter_deg", "jitter_deg"], 0.0)
    jitter_aos = _first_numeric(payload, sample, ["jitter_aos_deg", "attitude_jitter_aos_deg"], jitter_scalar)
    jitter_aoa = _first_numeric(payload, sample, ["jitter_aoa_deg", "attitude_jitter_aoa_deg"], jitter_scalar)

    if not aos_explicit:
        aos += jitter_aos
    if not aoa_explicit:
        aoa += jitter_aoa
    return float(aoa), float(aos)


def _flow_zero_direction_from_payload(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    value = _payload_value(
        payload,
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
    vec = _replace_nans_with_zero(vec)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vec / norm


def _flow_unit_from_angles(aos_deg: float, aoa_deg: float, flow_zero_direction=None) -> np.ndarray:
    zero_dir = None
    if flow_zero_direction is not None:
        zero_dir = _flow_zero_direction_from_payload({"flow_zero_direction": flow_zero_direction})
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
    # Convention: AoA=0, AoS=0 -> flow along -X.
    return np.array(
        [
            -np.cos(aos_rad) * cos_aoa,
            np.sin(aos_rad) * cos_aoa,
            np.sin(aoa_rad),
        ],
        dtype=float,
    )


def _angles_from_flow_vector(flow_vec, flow_zero_direction=None) -> Optional[tuple[float, float, float]]:
    try:
        vec = np.asarray(flow_vec, dtype=float).reshape(-1)
    except Exception:
        return None
    if vec.size < 3:
        return None
    vec = _replace_nans_with_zero(vec[:3])
    speed = float(np.linalg.norm(vec))
    if not np.isfinite(speed) or speed <= 1e-12:
        return None
    unit = vec / speed
    zero_dir = None
    if flow_zero_direction is not None:
        zero_dir = _flow_zero_direction_from_payload({"flow_zero_direction": flow_zero_direction})
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
        return aoa_deg, aos_deg, speed

    aoa_deg = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))
    aos_deg = float(np.degrees(np.arctan2(unit[1], -unit[0])))
    return aoa_deg, aos_deg, speed


def _resolve_nominal_attitude(payload: Dict[str, Any], default_aos_deg: float) -> tuple[float, float]:
    nominal_aoa = _payload_value(payload, ["nominal_aoa_deg"], None)
    nominal_aos = _payload_value(payload, ["nominal_aos_deg"], None)
    if nominal_aoa is not None or nominal_aos is not None:
        try:
            aoa_deg = float(nominal_aoa if nominal_aoa is not None else 0.0)
        except Exception:
            aoa_deg = 0.0
        try:
            aos_deg = float(nominal_aos if nominal_aos is not None else default_aos_deg)
        except Exception:
            aos_deg = float(default_aos_deg)
        return aoa_deg, aos_deg
    return _resolve_attitude(payload, default_aos_deg)


def _resolve_effective_attitude(payload: Dict[str, Any], default_aos_deg: float, flow_speed_mps: float, altitude_km: float):
    explicit_aoa = _payload_value(payload, ["effective_aoa_deg"], None)
    explicit_aos = _payload_value(payload, ["effective_aos_deg"], None)
    explicit_speed = _payload_value(payload, ["relative_flow_speed_mps"], None)
    if explicit_aoa is not None or explicit_aos is not None:
        try:
            aoa_deg = float(explicit_aoa if explicit_aoa is not None else 0.0)
        except Exception:
            aoa_deg = 0.0
        try:
            aos_deg = float(explicit_aos if explicit_aos is not None else default_aos_deg)
        except Exception:
            aos_deg = float(default_aos_deg)
        try:
            rel_speed = float(explicit_speed if explicit_speed is not None else flow_speed_mps)
        except Exception:
            rel_speed = float(flow_speed_mps)
        return aoa_deg, aos_deg, rel_speed

    nominal_aoa_deg, nominal_aos_deg = _resolve_nominal_attitude(payload, default_aos_deg)
    wind_vec = payload.get("wind_enu_mps")
    if wind_vec is None:
        wind_vec = _sample_hwm14_wind(payload, altitude_km)
    try:
        wind = np.asarray(wind_vec, dtype=float).reshape(-1)[:3] if wind_vec is not None else None
    except Exception:
        wind = None
    if wind is None or wind.size < 3:
        return nominal_aoa_deg, nominal_aos_deg, float(flow_speed_mps)
    wind = _replace_nans_with_zero(wind)
    if float(np.linalg.norm(wind)) <= 1e-12:
        return nominal_aoa_deg, nominal_aos_deg, float(flow_speed_mps)

    flow_zero_direction = _payload_value(
        payload,
        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
        None,
    )
    rel = _flow_unit_from_angles(nominal_aos_deg, nominal_aoa_deg, flow_zero_direction) * float(flow_speed_mps) - wind
    resolved = _angles_from_flow_vector(rel, flow_zero_direction)
    if resolved is None:
        return nominal_aoa_deg, nominal_aos_deg, float(flow_speed_mps)
    return resolved


def _resolve_boundary3_source_name(payload: Dict[str, Any]) -> str:
    project_name = _resolve_project_name(payload)
    mesh_file = _resolve_mesh_file(payload)
    if project_name.upper() == "CUBE" or mesh_file.strip().lower() == "cube_mesh.h5":
        return "CUBE"
    return "OBJ"


def _geometry_candidates(payload: Dict[str, Any]) -> list[Any]:
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    return [
        sample.get("geometry_id"),
        payload.get("geometry_id"),
        sample.get("geometry_name"),
        payload.get("geometry_name"),
        sample.get("hf_mesh"),
        payload.get("hf_mesh"),
    ]


def _normalize_project_name(candidate: Any) -> Optional[str]:
    if candidate is None:
        return None
    token = os.path.basename(str(candidate)).strip()
    if not token:
        return None
    lower = token.lower()
    if lower.endswith("_mesh.h5"):
        token = token[:-8]
    elif lower.endswith(".h5"):
        token = token[:-3]
    if not token:
        return None
    return _PROJECT_NAME_ALIASES.get(token.upper(), token)


def _resolve_project_name(payload: Dict[str, Any]) -> str:
    for cand in _geometry_candidates(payload):
        project_name = _normalize_project_name(cand)
        if project_name:
            return project_name
    return "Cube"


def _resolve_mesh_file(payload: Dict[str, Any]) -> str:
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    for cand in [sample.get("hf_mesh"), payload.get("hf_mesh")]:
        if cand is None:
            continue
        mesh_name = os.path.basename(str(cand)).strip()
        if mesh_name:
            return mesh_name

    project_name = _resolve_project_name(payload)
    return f"{project_name}_mesh.h5"


def _load_atmosphere_from_csv(gn: int, idx: int) -> np.ndarray:
    requested_gn = int(gn)
    default_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ADBSat-PyVersion", "atmos_data")
    )
    atmos_dir = os.path.abspath(os.environ.get("MFMC_ATMOS_DATA", default_dir))
    csv_path = os.path.join(atmos_dir, f"database_{requested_gn}km.csv")
    if not os.path.exists(csv_path):
        available = []
        for name in os.listdir(atmos_dir):
            m = re.fullmatch(r"database_(\d+)km\.csv", name)
            if m:
                available.append(int(m.group(1)))
        if not available:
            raise FileNotFoundError(
                f"No atmosphere CSV files found in '{atmos_dir}' "
                "(expected names like database_200km.csv)."
            )
        nearest = min(available, key=lambda x: abs(x - requested_gn))
        csv_path = os.path.join(atmos_dir, f"database_{nearest}km.csv")
        print(
            f"[WARN] Atmosphere CSV for {requested_gn} km not found. "
            f"Using nearest available table: {nearest} km."
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Atmosphere database is empty: {csv_path}")
    database_idx = int(idx) % len(df)
    return df.iloc[database_idx].to_numpy(dtype=float)


def _sample_pymsis_row(payload: Dict[str, Any], gn: int) -> np.ndarray:
    try:
        import pymsis as ps  # type: ignore
    except Exception as exc:
        raise ImportError("pymsis is required for environment_model='pymsis_hwm14'") from exc

    altitude_km = float(payload.get("altitude_km", gn))
    lat_deg = float(payload.get("lat_deg", 0.0))
    lon_deg = float(payload.get("lon_deg", 0.0))
    datetime_utc = str(payload.get("datetime_utc", "2006-01-01T00:00"))
    dt64 = np.datetime64(datetime_utc)
    f107 = _space_weather_numeric(payload, ["f107"], None)
    f107a = _space_weather_numeric(payload, ["f107a"], f107)
    aps = _space_weather_ap_vector(payload)

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
    return _replace_nans_with_zero(row[:11])


def _datetime_to_doy_seconds(datetime_utc: str) -> tuple[int, int, float]:
    dt = np.datetime64(datetime_utc, "s")
    year = int(str(dt)[:4])
    year_start = np.datetime64(f"{year}-01-01T00:00:00", "s")
    delta_s = int((dt - year_start).astype("timedelta64[s]").astype(int))
    doy = delta_s // 86400 + 1
    sec = float(delta_s % 86400)
    return year, doy, sec


def _sample_hwm14_wind(payload: Dict[str, Any], altitude_km: float):
    if not bool(payload.get("use_winds", False)):
        return None
    try:
        import hwm14  # type: ignore
    except Exception:
        return None

    lat = float(payload.get("lat_deg", 0.0))
    lon = float(payload.get("lon_deg", 0.0))
    datetime_utc = str(payload.get("datetime_utc", "2006-01-01T00:00"))
    f107 = float(_space_weather_numeric(payload, ["f107"], 150.0))
    f107a = float(_space_weather_numeric(payload, ["f107a"], f107))
    ap = float(_space_weather_numeric(payload, ["ap", "ap_daily", "daily_ap"], 4.0))
    year, doy, sec = _datetime_to_doy_seconds(datetime_utc)

    # Best-effort support for common python wrappers.
    for name in ("hwm14", "run", "wind"):
        fn = getattr(hwm14, name, None)
        if callable(fn):
            try:
                out = fn(year, doy, sec, altitude_km, lat, lon, f107a, f107, ap)
                arr = np.asarray(out, dtype=float).reshape(-1)
                if arr.size >= 2:
                    return [float(arr[0]), float(arr[1]), float(arr[2]) if arr.size >= 3 else 0.0]
            except Exception:
                continue
    return None


def _apply_atmosphere_perturbations(atmosphere: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
    row = _replace_nans_with_zero(np.asarray(atmosphere, dtype=float))
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}

    density_scale = _first_numeric(payload, sample, ["density_scale"], 1.0)
    density_state_scale = _first_numeric(payload, sample, ["density_state_scale"], 1.0)
    row[:10] = np.clip(row[:10] * density_scale * density_state_scale, 0.0, None)

    temp_scale = _first_numeric(payload, sample, ["temperature_scale"], 1.0)
    temp_offset = _first_numeric(payload, sample, ["temperature_offset_k", "temp_offset_k"], 0.0)
    density_state_temp_offset = _first_numeric(payload, sample, ["density_state_temperature_offset_k"], 0.0)
    row[10] = max(1.0, row[10] * temp_scale + temp_offset + density_state_temp_offset)

    composition_shift = _first_numeric(payload, sample, ["composition_shift"], 0.0)
    composition_shift += _first_numeric(payload, sample, ["density_state_composition_shift"], 0.0)
    if abs(composition_shift) > 0:
        # Light-touch skew: O up, N2 down.
        row[1] = max(0.0, row[1] * (1.0 - composition_shift))  # N2
        row[3] = max(0.0, row[3] * (1.0 + composition_shift))  # O

    composition_delta = sample.get("composition_delta", payload.get("composition_delta"))
    if isinstance(composition_delta, dict):
        species_map = {"N2": 1, "O2": 2, "O": 3, "HE": 4, "H": 5, "AR": 6, "N": 7, "AO": 8, "NO": 9}
        for species, delta in composition_delta.items():
            key = str(species).upper()
            if key in species_map:
                try:
                    dval = float(delta)
                except Exception:
                    continue
                row[species_map[key]] = max(0.0, row[species_map[key]] * (1.0 + dval))
    return _replace_nans_with_zero(row)


def _atmosphere_from_payload(payload: Dict[str, Any], gn: int) -> Optional[np.ndarray]:
    if not payload:
        return None

    if "atmosphere_row" in payload:
        row = np.asarray(payload["atmosphere_row"], dtype=float).reshape(-1)
        if row.shape[0] >= 11:
            return _replace_nans_with_zero(row[:11])

    if "rho" in payload and "Tinf" in payload:
        rho = _replace_nans_with_zero(np.asarray(payload["rho"], dtype=float).reshape(-1))
        if rho.shape[0] >= 9:
            he = float(rho[0])
            o = float(rho[1])
            n2 = float(rho[2])
            o2 = float(rho[3])
            ar = float(rho[4]) if rho.shape[0] > 4 else 0.0
            h = float(rho[5]) if rho.shape[0] > 5 else 0.0
            n = float(rho[6]) if rho.shape[0] > 6 else 0.0
            ao = float(rho[7]) if rho.shape[0] > 7 else 0.0
            no = float(rho[8]) if rho.shape[0] > 8 else 0.0
            mass_density = float(payload.get("mass_density", rho[9] if rho.shape[0] > 9 else max(np.sum(rho[:9]), 0.0)))
            temp = float(payload["Tinf"])
            return _replace_nans_with_zero(np.array([mass_density, n2, o2, o, he, h, ar, n, ao, no, temp], dtype=float))

    env_model = str(payload.get("environment_model", "csv"))
    if env_model == "pymsis_hwm14":
        return _sample_pymsis_row(payload, gn)

    return None


def _resolve_atmosphere(gn: int, idx: int, payload: Dict[str, Any], env_model: str) -> np.ndarray:
    if payload:
        payload = dict(payload)
        payload.setdefault("environment_model", env_model)
        try:
            row = _atmosphere_from_payload(payload, gn)
        except Exception:
            row = None
        if row is not None:
            return _apply_atmosphere_perturbations(row, payload)

    if env_model == "pymsis_hwm14":
        try:
            row = _sample_pymsis_row(payload or {}, gn)
            return _apply_atmosphere_perturbations(row, payload or {})
        except Exception:
            pass

    row = _load_atmosphere_from_csv(gn, idx)
    return _apply_atmosphere_perturbations(row, payload or {})


def _resolve_macro_particle_factor(rho: np.ndarray) -> float:
    """
    Enforce:
        number_density * 0.2^3 / MPF = 1e6
    -> MPF = number_density * 0.2^3 / 1e6
    """
    number_density = float(np.sum(np.asarray(rho[:9], dtype=float)))
    mpf = number_density * (0.2 ** 3) / 1.0e6
    if not np.isfinite(mpf) or mpf <= 0:
        return 1.0
    return float(mpf)


def update_ini_from_csv(
    gn,
    aos,
    idx,
    ini_path,
    random_seed=None,
    env_payload_path=None,
    env_model="csv",
    geometry_id=None,
    geometry_name=None,
    geometry_mesh=None,
    debug_print=False,
    debug_json=None,
):
    gn = int(gn)
    payload = _load_payload(env_payload_path)
    if geometry_id is not None:
        payload["geometry_id"] = geometry_id
    if geometry_name is not None:
        payload["geometry_name"] = geometry_name
    if geometry_mesh is not None:
        payload["hf_mesh"] = geometry_mesh
    if payload and env_model == "csv":
        env_model = str(payload.get("environment_model", env_model))

    atmosphere = _resolve_atmosphere(gn, int(idx), payload, str(env_model))

    rho = np.array(
        [
            atmosphere[4],
            atmosphere[3],
            atmosphere[1],
            atmosphere[2],
            atmosphere[6],
            atmosphere[5],
            atmosphere[7],
            atmosphere[8],
            atmosphere[9],
            atmosphere[0],
        ],
        dtype=float,
    )
    rho = _replace_nans_with_zero(rho)
    mpf = _resolve_macro_particle_factor(rho)

    Tinf = float(atmosphere[10])
    surface_params = _resolve_piclas_surface_parameters(payload)
    numerical_controls = _resolve_piclas_numerical_controls(payload, mpf)
    wall_temp = float(surface_params["wall_temperature_k"])
    trans_acc = float(surface_params["trans_accommodation"])
    momentum_acc = float(surface_params["momentum_accommodation"])
    vib_acc = float(surface_params["vib_accommodation"])
    rot_acc = float(surface_params["rot_accommodation"])
    mpf = float(numerical_controls["macro_particle_factor"])
    h = gn * 1e3
    constants = ConstantsData()

    total_density = max(np.sum(rho[:9]), 1e-30)
    mmean = (
        rho[0] * constants.mHe
        + rho[1] * constants.mO
        + rho[2] * constants.mN2
        + rho[3] * constants.mO2
        + rho[4] * constants.mAr
        + rho[5] * constants.mH
        + rho[6] * constants.mN
        + rho[7] * constants.mAnO
        + rho[8] * constants.mNO
    ) / total_density

    vinf = np.sqrt(constants.mu_E / (constants.R_E + h))
    vth = np.sqrt(2 * constants.kb * Tinf / (mmean / constants.NA / 1000))
    aoa_deg, aos_deg, rel_flow_speed = _resolve_effective_attitude(payload, float(aos), vinf, float(gn))
    if bool(payload.get("apply_wind_to_speed", False)):
        vinf = float(rel_flow_speed)

    dyn_p = np.array([0.5 * rho[-1] * vinf**2], dtype=float)
    np.savetxt("dyn_p.txt", dyn_p)

    mesh_file = _resolve_mesh_file(payload)
    project_name = _resolve_project_name(payload)
    boundary3_source_name = _resolve_boundary3_source_name(payload)
    debug_payload = {
        "geometry_id": payload.get("geometry_id"),
        "geometry_name": payload.get("geometry_name"),
        "hf_mesh": payload.get("hf_mesh"),
        "resolved_mesh_file": mesh_file,
        "resolved_project_name": project_name,
        "resolved_boundary3_source_name": boundary3_source_name,
        "env_model": env_model,
        "env_payload_path": env_payload_path,
        "ini_path": ini_path,
    }

    debug_values = {
        **debug_payload,
        "regime_and_attitude": {
            "height_km": float(gn),
            "input_aos_deg": float(aos),
            "resolved_aos_deg": float(aos_deg),
            "resolved_aoa_deg": float(aoa_deg),
            "relative_flow_speed_mps": float(rel_flow_speed),
            "flow_zero_direction": _payload_value(
                payload,
                ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
                None,
            ),
            "flow_unit_vector_xyz": [
                float(v_in_comp)
                for v_in_comp in _flow_unit_from_angles(
                    aos_deg,
                    aoa_deg,
                    _payload_value(
                        payload,
                        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
                        None,
                    ),
                )
            ],
        },
        "atmosphere": {
            "atmosphere_row_raw": [float(x) for x in atmosphere.tolist()],
            "rho_vector_reordered": [float(x) for x in rho.tolist()],
            "freestream_temperature_K": float(Tinf),
            "total_number_density": float(total_density),
            "mean_particle_mass_kg": float(mmean),
        },
        "dynamics": {
            "orbital_speed_vinf_mps": float(vinf),
            "thermal_speed_vth_mps": float(vth),
            "dynamic_pressure_Pa": float(dyn_p[0]),
            "dynamic_pressure_convention": "0.5*rho_mass*v_rel^2",
        },
        "surface_parameters": {k: float(v) for k, v in surface_params.items()},
        "numerical_controls": {
            **{k: (int(v) if isinstance(v, int) else float(v)) for k, v in numerical_controls.items()},
            "computed_macro_particle_factor_from_rho": float(_resolve_macro_particle_factor(rho)),
            "effective_macro_particle_factor_used": float(mpf),
        },
        "execution_environment": {
            "env_model": str(env_model),
            "use_winds": bool(payload.get("use_winds", False)),
            "apply_wind_to_speed": bool(payload.get("apply_wind_to_speed", False)),
            "f107": _space_weather_numeric(payload, ["f107"], None),
            "f107a": _space_weather_numeric(payload, ["f107a"], None),
            "ap": _space_weather_numeric(payload, ["ap", "ap_daily", "daily_ap"], None),
            "datetime_utc": payload.get("datetime_utc"),
            "lat_deg": payload.get("lat_deg"),
            "lon_deg": payload.get("lon_deg"),
        },
    }

    if debug_print:
        print("[update_parameter DEBUG] Resolved inputs and injected values:")
        print(json.dumps(debug_values, indent=2, sort_keys=True))
    if debug_json:
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump(debug_values, f, indent=2, sort_keys=True)
    flow_zero_direction = _payload_value(
        payload,
        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
        None,
    )
    v_in = _flow_unit_from_angles(aos_deg, aoa_deg, flow_zero_direction)

    with open(ini_path, "r", encoding="utf-8") as file:
        ini_lines = file.readlines()

    updated_lines = []
    t_end_override = float(numerical_controls["t_end_s"])
    t_end_scale = float(numerical_controls["t_end_scale"])
    species_pattern = re.compile(r"Part-Species(\d+)-")
    seed_base = None if random_seed is None else max(1, int(random_seed) % (2**31 - 1))
    seed_1 = seed_base
    seed_2 = None
    if seed_base is not None:
        seed_2 = ((seed_base * 1103515245 + 12345) % (2**31 - 1)) or 2
        if seed_2 == seed_1:
            seed_2 = ((seed_1 + 1) % (2**31 - 1)) or 2

    for line in ini_lines:
        match = species_pattern.match(line)
        if match:
            species_index = int(match.group(1))
            if "Init1-PartDensity" in line:
                line = (
                    f"Part-Species{species_index}-Init1-PartDensity = {rho[species_index-1]:.5E}  "
                    "! Number density [1/m³] (real particles)\n"
                )
            elif "Surfaceflux1-PartDensity" in line:
                line = (
                    f"Part-Species{species_index}-Surfaceflux1-PartDensity = {rho[species_index-1]:.5E}  "
                    "! Number density [1/m³] (real particles)\n"
                )

        if line.lstrip().startswith("MeshFile"):
            line = f"MeshFile = {mesh_file}  ! (relative) path to meshfile\n"
        elif line.lstrip().startswith("TEnd"):
            t_end_value = t_end_override
            if not np.isfinite(t_end_value):
                try:
                    base_t_end = float(line.split("=", 1)[1].split("!", 1)[0].strip())
                except Exception:
                    base_t_end = float("nan")
                if np.isfinite(base_t_end) and np.isfinite(t_end_scale):
                    t_end_value = float(base_t_end * t_end_scale)
            if np.isfinite(t_end_value) and t_end_value > 0.0:
                line = f"TEnd                  = {t_end_value:.6E}      ! End time [s] of the simulation\n"
        elif line.lstrip().startswith("ProjectName"):
            line = f"ProjectName     = {project_name}    ! Name of the current simulation\n"
        elif "Init1-MWTemperatureIC" in line:
            line = f"Part-Species$-Init1-MWTemperatureIC = {Tinf:.0f}  ! Temperature [K] for Maxwell distribution\n"
        elif "Part-Boundary3-SourceName" in line:
            line = f"Part-Boundary3-SourceName  = {boundary3_source_name}\n"
        elif "Part-Boundary3-WallTemp" in line:
            line = f"Part-Boundary3-WallTemp    = {wall_temp:.6g}         ! Wall temperature [K] of reflective particle boundary [$].\n"
        elif "Part-Boundary3-TransACC" in line:
            line = f"Part-Boundary3-TransACC    = {trans_acc:.6g}           ! Translation accommodation coefficient of reflective particle boundary [$].\n"
        elif "Part-Boundary3-MomentumACC" in line:
            line = f"Part-Boundary3-MomentumACC = {momentum_acc:.6g}           ! Momentum accommodation coefficient of reflective particle boundary [$].\n"
        elif "Part-Boundary3-VibACC" in line:
            line = f"Part-Boundary3-VibACC      = {vib_acc:.6g}           ! Vibrational accommodation coefficient of reflective particle boundary [$].\n"
        elif "Part-Boundary3-RotACC" in line:
            line = f"Part-Boundary3-RotACC      = {rot_acc:.6g}           ! Rotational accommodation coefficient of reflective particle boundary [$].\n"
        elif "MacroParticleFactor" in line:
            line = (
                f"Part-Species$-MacroParticleFactor = {mpf:.5E} "
                "! Particle weighting factor: number of simulation particles per real particle for species [$]\n"
            )
        elif line.lstrip().startswith("ManualTimeStep"):
            line = (
                f"ManualTimeStep        = {float(numerical_controls['manual_timestep_s']):.6E}  "
                "! Manual timestep [s]\n"
            )
        elif line.lstrip().startswith("Particles-MPIWeight"):
            line = (
                f"Particles-MPIWeight                      = {int(numerical_controls['particles_mpi_weight'])}  "
                "! Define weight of particles for elem loads.\n"
            )
        elif line.lstrip().startswith("Part-IterationForMacroVal"):
            line = (
                f"Part-IterationForMacroVal         = {int(numerical_controls['sampling_iterations'])}    "
                "! Set number of iterations used for sampling                                      (Can not be enabled together with Part-TimeFracForSampling)\n"
            )
        elif line.lstrip().startswith("Particles-OctreePartNumNodeMin"):
            line = (
                f"Particles-OctreePartNumNodeMin     = {int(numerical_controls['octree_part_num_node_min'])}  "
                "! Allow grid division until the minimum number of particles in a subcell is above OctreePartNumNodeMin\n"
            )
        elif line.lstrip().startswith("Particles-OctreePartNumNode"):
            line = (
                f"Particles-OctreePartNumNode        = {int(numerical_controls['octree_part_num_node'])}  "
                "! Resolve grid until the maximum number of particles in a subcell equals OctreePartNumNode\n"
            )
        elif line.lstrip().startswith("Part-NumberOfRandomSeeds") and seed_base is not None:
            line = "Part-NumberOfRandomSeeds = 2  ! Number of Seeds for Random Number Generator\n"
        elif line.lstrip().startswith("Particles-RandomSeed1") and seed_1 is not None:
            line = f"Particles-RandomSeed1    = {int(seed_1)}\n"
        elif line.lstrip().startswith("Particles-RandomSeed2") and seed_2 is not None:
            line = f"Particles-RandomSeed2    = {int(seed_2)}\n"
        elif "Surfaceflux1-MWTemperatureIC" in line:
            line = f"Part-Species$-Surfaceflux1-MWTemperatureIC = {Tinf:.0f}  ! Temperature [K] for Maxwell distribution\n"
        elif "Init1-VeloIC" in line:
            line = f"Part-Species$-Init1-VeloIC = {vinf:.3f}  ! Velocity magnitude [m/s]\n"
        elif "Surfaceflux1-VeloIC" in line:
            line = f"Part-Species$-Surfaceflux1-VeloIC = {vinf:.3f}  ! Velocity magnitude [m/s]\n"
        elif "Init1-VeloVecIC" in line:
            line = (
                f"Part-Species$-Init1-VeloVecIC = (/{v_in[0]:.6f},{v_in[1]:.6f},{v_in[2]:.6f}/)  "
                "! Velocity magnitude [m/s]\n"
            )
        elif "Surfaceflux1-VeloVecIC" in line:
            line = (
                f"Part-Species$-Surfaceflux1-VeloVecIC = (/{v_in[0]:.6f},{v_in[1]:.6f},{v_in[2]:.6f}/)  "
                "! Velocity magnitude [m/s]\n"
            )

        updated_lines.append(line)

    with open(ini_path, "w", encoding="utf-8") as file:
        file.writelines(updated_lines)

    print(f"Die Datei {ini_path} wurde erfolgreich aktualisiert.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update PICLas parameter.ini from atmosphere inputs.")
    parser.add_argument("height_km", type=float, help="Altitude in km")
    parser.add_argument("aos_deg", type=float, help="Angle of sideslip in deg")
    parser.add_argument("database_index", type=int, help="Database row index")
    parser.add_argument("ini_file", type=str, help="INI file path")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional PICLas particle RNG base seed")
    parser.add_argument("--env-payload", type=str, default=None, help="Optional JSON payload for environment inputs")
    parser.add_argument(
        "--env-model",
        type=str,
        default="csv",
        choices=["csv", "pymsis_hwm14"],
        help="Environment source model",
    )
    parser.add_argument("--geometry-id", type=str, default=None, help="Optional geometry identifier")
    parser.add_argument("--geometry-name", type=str, default=None, help="Optional geometry display name")
    parser.add_argument("--geometry-mesh", type=str, default=None, help="Optional geometry mesh filename")
    parser.add_argument("--debug-print", action="store_true", help="Print resolved geometry debug information")
    parser.add_argument("--debug-json", type=str, default=None, help="Write resolved geometry debug JSON")
    args = parser.parse_args()
    update_ini_from_csv(
        args.height_km,
        args.aos_deg,
        args.database_index,
        args.ini_file,
        random_seed=args.random_seed,
        env_payload_path=args.env_payload,
        env_model=args.env_model,
        geometry_id=args.geometry_id,
        geometry_name=args.geometry_name,
        geometry_mesh=args.geometry_mesh,
        debug_print=args.debug_print,
        debug_json=args.debug_json,
    )
