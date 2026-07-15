import os
import json
import pickle
import numpy as np
import pandas as pd
import sys
import shutil
import time
try:
    from mat2vtu import mat2vtu
except ImportError:
    def mat2vtu(*args, **kwargs):
        return None
from scipy.io import loadmat
from calc.environment import environment
from calc.calc_coeff import calc_coeff
from postpro.plot_surfq import plot_surfq
from calc.ADBSatConstants import ConstantsData
import multiprocessing
import re
import inspect


SURFACE_STATE_PRESETS = {
    "nominal": {"alpha": 0.9, "alphaN": 0.9, "momentum_accommodation": 0.81, "sigmaN": 0.9, "sigmaT": 0.7, "Tw": 300.0, "sol_cR": 0.15, "sol_cD": 0.25},
    "clean": {"alpha": 0.85, "alphaN": 0.85, "momentum_accommodation": 0.72, "sigmaN": 0.75, "sigmaT": 0.6, "Tw": 295.0, "sol_cR": 0.12, "sol_cD": 0.22},
    "rough": {"alpha": 0.96, "alphaN": 0.96, "momentum_accommodation": 0.92, "sigmaN": 0.94, "sigmaT": 0.85, "Tw": 320.0, "sol_cR": 0.18, "sol_cD": 0.30},
    "oxidized": {"alpha": 0.98, "alphaN": 0.98, "momentum_accommodation": 0.96, "sigmaN": 0.97, "sigmaT": 0.9, "Tw": 330.0, "sol_cR": 0.2, "sol_cD": 0.33},
    "contaminated": {"alpha": 0.78, "alphaN": 0.78, "momentum_accommodation": 0.62, "sigmaN": 0.65, "sigmaT": 0.5, "Tw": 300.0, "sol_cR": 0.1, "sol_cD": 0.2},
}

GEOMETRY_MODEL_MAP = {
    "CUBE": "Cube",
    "SOAR": "SOAR",
    "GOCE": "GOCE",
    "CHAMP": "CHAMP",
    "OPT_SAT": "Opt_Sat",
    "TRIPLE_CUBE": "Triple_Cube",
    "CUBESAT": "CubeSat",
    "SPECPRACTOPT_CYLINDER_HEX": "SpecPractOpt_cylinder_hex",
    "SPECPRACTOPT_CYLINDERHEX": "SpecPractOpt_cylinder_hex",
    "SPECPRACTOPT_CYLINDER": "SpecPractOpt_cylinder_hex",
}

GEOMETRY_MODEL_FALLBACK = {
    "SOAR": "Opt_Sat",
    "GOCE": "Triple_Cube",
    "CHAMP": "CubeSat",
}


def _finite_summary(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return f"{name}=<empty>"
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"{name}=all-nonfinite shape={arr.shape}"
    return (
        f"{name}: shape={arr.shape}, min={float(np.min(finite)):.6g}, "
        f"max={float(np.max(finite)):.6g}, finite={finite.size}/{arr.size}"
    )


def _require_finite(name, value, context):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite ADBSat input {name}; {context}; {_finite_summary(name, value)}")


def _validate_environment_inputs(inparam, context):
    for key in ("rho", "vinf", "vth", "s", "Tinf", "mmean", "Tw", "alpha"):
        if key in inparam:
            _require_finite(key, inparam[key], context)
    if "rho" in inparam:
        rho = np.asarray(inparam["rho"], dtype=float)
        if np.any(rho < 0.0):
            raise ValueError(f"Negative density in ADBSat input; {context}; {_finite_summary('rho', inparam['rho'])}")
        if rho.size >= 10 and rho[9] <= 0.0:
            raise ValueError(f"Non-positive mass density in ADBSat input; {context}; {_finite_summary('rho', inparam['rho'])}")
        if rho.size >= 9 and np.sum(rho[:9]) <= 0.0:
            raise ValueError(f"Non-positive species density sum in ADBSat input; {context}; {_finite_summary('rho', inparam['rho'])}")
    for key in ("vinf", "vth", "s", "Tinf", "mmean"):
        if key in inparam and np.any(np.asarray(inparam[key], dtype=float) <= 0.0):
            raise ValueError(f"Non-positive ADBSat input {key}; {context}; {_finite_summary(key, inparam[key])}")


def _validate_coefficients(coeffs, context):
    if not isinstance(coeffs, dict):
        _require_finite("C_D", coeffs, context)
        return
    bad = []
    for key in ("C_D", "C_L", "C_Y", "C_Mx", "C_My", "C_Mz"):
        if key not in coeffs:
            continue
        value = coeffs[key]
        if not np.isfinite(float(value)):
            bad.append(f"{key}={value}")
    if bad:
        raise ValueError(f"Non-finite ADBSat output coefficients; {context}; {', '.join(bad)}")


def _write_panel_surface_field(coeffs, payload, result_dir, gsi_model, run_id):
    if not bool(payload.get("write_panel_surface_field", False)):
        return None
    required = (
        "panel_force_per_area",
        "panel_area",
        "panel_center",
        "panel_normal",
        "u_hat_inf",
        "q_inf",
        "AreaRef",
    )
    missing = [key for key in required if key not in coeffs]
    if missing:
        raise RuntimeError(f"ADBSat panel surface field is missing: {missing}")
    output_dir = os.path.join(result_dir, "surface_fields")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{gsi_model}_{int(run_id)}.npz")
    np.savez_compressed(
        output_path,
        sample_id=np.asarray([str(payload.get("surface_sample_id", run_id))]),
        model_id=np.asarray([str(gsi_model)]),
        mesh_fingerprint=np.asarray([str(payload.get("surface_mesh_fingerprint", ""))]),
        panel_force_per_area=np.asarray(coeffs["panel_force_per_area"], dtype=float),
        panel_traction_over_q=np.asarray(coeffs["panel_traction_over_q"], dtype=float),
        panel_area=np.asarray(coeffs["panel_area"], dtype=float),
        panel_center=np.asarray(coeffs["panel_center"], dtype=float),
        panel_normal=np.asarray(coeffs["panel_normal"], dtype=float),
        q_inf=np.asarray([float(coeffs["q_inf"])]),
        A_ref=np.asarray([float(coeffs["AreaRef"])]),
        u_hat_inf=np.asarray(coeffs["u_hat_inf"], dtype=float),
        C_D=np.asarray([float(coeffs["C_D"])]),
        C_L=np.asarray([float(coeffs["C_L"])]),
        C_Y=np.asarray([float(coeffs["C_Y"])]),
    )
    return output_path


def _loaded_mat_string(mat_data, key):
    if key not in mat_data:
        return ""
    values = np.asarray(mat_data[key]).reshape(-1)
    if values.size == 0:
        return ""
    return str(values[0]).strip()


def _payload_value(payload, keys, default):
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    for k in keys:
        if k in sample:
            return sample[k]
        if k in payload:
            return payload[k]
    return default


def _as_panel_array(value, n_elems, default):
    if value is None:
        return np.full(n_elems, float(default), dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == n_elems:
            return arr
        if arr.size == 1:
            return np.full(n_elems, float(arr[0]), dtype=float)
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return np.full(n_elems, v, dtype=float)


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _panel_summary(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "size": int(arr.size),
            "finite": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "size": int(arr.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _assert_panel_binding(name, actual, expected, n_elems, default, context):
    expected_arr = _as_panel_array(expected, n_elems, default)
    actual_arr = np.asarray(actual, dtype=float).reshape(-1)
    if actual_arr.size != expected_arr.size:
        raise ValueError(
            f"ADBSat input binding failed for {name}: "
            f"actual_size={actual_arr.size}, expected_size={expected_arr.size}; {context}"
        )
    diff = np.nanmax(np.abs(actual_arr - expected_arr)) if actual_arr.size else 0.0
    if not np.isfinite(diff) or float(diff) > 1e-12:
        raise ValueError(
            f"ADBSat input binding failed for {name}: max_abs_diff={diff}; "
            f"actual={_panel_summary(actual_arr)}, expected={_panel_summary(expected_arr)}; {context}"
        )


def _write_input_audit(path, payload):
    audit_path = os.path.join(path, f"input_audit_{payload['gsi_model']}_{payload['run_id']}.json")
    tmp_path = f"{audit_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
    os.replace(tmp_path, audit_path)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _payload_has_key(payload, keys):
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    for k in keys:
        if k in sample or k in payload:
            return True
    return False


def _canonical_geometry_token(value):
    if value is None:
        return ""
    token = str(value).strip().upper()
    return token.replace("-", "_").replace(" ", "_")


def _geometry_assets_exist(adbsat_path, model_name):
    obj_path = os.path.join(adbsat_path, "inou", "obj_files", f"{model_name}.obj")
    mat_path = os.path.join(adbsat_path, "inou", "models", f"{model_name}.mat")
    return os.path.exists(obj_path) and os.path.exists(mat_path)


def _resolve_geometry_model(payload, adbsat_path, default_model="Cube"):
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    candidates = [
        sample.get("geometry_id"),
        payload.get("geometry_id"),
        sample.get("geometry_name"),
        payload.get("geometry_name"),
        sample.get("lf_model"),
        payload.get("lf_model"),
    ]

    for cand in candidates:
        key = _canonical_geometry_token(cand)
        if not key:
            continue
        model_name = GEOMETRY_MODEL_MAP.get(key)
        if not model_name:
            continue
        if _geometry_assets_exist(adbsat_path, model_name):
            return model_name
        fallback = GEOMETRY_MODEL_FALLBACK.get(key)
        if fallback and _geometry_assets_exist(adbsat_path, fallback):
            return fallback

    if _geometry_assets_exist(adbsat_path, default_model):
        return default_model
    return "Cube"


def _surface_state_defaults(payload):
    state = _payload_value(payload, ["surface_state", "gsi_surface_state"], None)
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    if state is None:
        return {}
    return dict(SURFACE_STATE_PRESETS.get(str(state).strip().lower(), {}))


def _resolve_surface_parameters(payload):
    """
    Resolve a shared wall/GSI convention from the campaign payload.

    PICLas maps energy_accommodation to TransACC. For ADBSat this is best
    represented by Sentman alpha and CLL alphaN. For PICLasMaxwell, TransACC
    and MomentumACC are kept separate to mirror PICLas' reflective wall model.
    PICLas MomentumACC is not the same quantity as CLL sigmaT, so sigmaT
    remains an independent CLL parameter unless explicitly supplied.
    """
    defaults = _surface_state_defaults(payload)

    tw = float(
        _payload_value(
            payload,
            ["Tw", "wall_temperature_k", "surface_temperature_k"],
            defaults.get("Tw", 300.0),
        )
    )

    trans_default = defaults.get("alpha", defaults.get("alphaN", 0.9))
    trans_acc = _clip01(
        _payload_value(
            payload,
            ["trans_accommodation", "energy_accommodation", "alpha"],
            trans_default,
        )
    )

    momentum_default = defaults.get("momentum_accommodation", defaults.get("MomentumACC", trans_acc * trans_acc))
    momentum_acc = _clip01(_payload_value(payload, ["momentum_accommodation"], momentum_default))

    alpha_n = _clip01(_payload_value(payload, ["alphaN", "alpha_n"], trans_acc))

    sigma_n = _clip01(_payload_value(payload, ["sigmaN", "sigma_n"], defaults.get("sigmaN", 0.9)))
    sigma_t = _clip01(_payload_value(payload, ["sigmaT", "sigma_t"], defaults.get("sigmaT", 0.7)))

    return {
        "alpha": trans_acc,
        "trans_accommodation": trans_acc,
        "momentum_accommodation": momentum_acc,
        "alphaN": alpha_n,
        "sigmaN": sigma_n,
        "sigmaT": sigma_t,
        "Tw": tw,
        "sol_cR": float(_payload_value(payload, ["sol_cR", "solar_reflectivity"], defaults.get("sol_cR", 0.15))),
        "sol_cD": float(_payload_value(payload, ["sol_cD", "solar_diffusivity"], defaults.get("sol_cD", 0.25))),
    }


def _resolve_attitude(payload, default_aos_deg):
    aos_explicit = _payload_has_key(payload, ["aos_deg", "aos", "beta_deg"])
    aoa_explicit = _payload_has_key(payload, ["aoa_deg", "aoa", "alpha_deg"])

    aos = _payload_value(payload, ["aos_deg", "aos", "beta_deg"], default_aos_deg)
    aoa = _payload_value(payload, ["aoa_deg", "aoa", "alpha_deg"], 0.0)
    try:
        aos = float(aos)
    except Exception:
        aos = float(default_aos_deg)
    try:
        aoa = float(aoa)
    except Exception:
        aoa = 0.0

    jitter_scalar = _payload_value(payload, ["attitude_jitter_deg", "jitter_deg"], 0.0)
    jitter_aos = _payload_value(payload, ["jitter_aos_deg", "attitude_jitter_aos_deg"], jitter_scalar)
    jitter_aoa = _payload_value(payload, ["jitter_aoa_deg", "attitude_jitter_aoa_deg"], jitter_scalar)
    try:
        jitter_aos = float(jitter_aos)
    except Exception:
        jitter_aos = 0.0
    try:
        jitter_aoa = float(jitter_aoa)
    except Exception:
        jitter_aoa = 0.0

    if not aos_explicit:
        aos += jitter_aos
    if not aoa_explicit:
        aoa += jitter_aoa
    return float(aoa), float(aos)


def _flow_zero_direction_from_payload(payload):
    value = _payload_value(
        payload,
        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
        None,
    )
    if value is None:
        return None
    if isinstance(value, str):
        value = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
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


def _flow_unit_from_angles(aos_deg, aoa_deg, flow_zero_direction=None):
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
    return np.array(
        [
            -np.sin(aos_rad) * cos_aoa,
            np.cos(aos_rad) * cos_aoa,
            np.sin(aoa_rad),
        ],
        dtype=float,
    )


def _adbsat_calc_angles_from_flow_vector(flow_vec):
    try:
        vec = np.asarray(flow_vec, dtype=float).reshape(-1)[:3]
    except Exception:
        return 0.0, 0.0
    if vec.size < 3:
        return 0.0, 0.0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return 0.0, 0.0

    unit = vec / norm
    aos_rad = float(np.arcsin(np.clip(unit[1], -1.0, 1.0)))
    aoa_rad = float(np.arctan2(unit[2], -unit[0]))
    return float(np.degrees(aoa_rad)), float(np.degrees(aos_rad))


def _angles_from_flow_vector(flow_vec, flow_zero_direction=None):
    try:
        vec = np.asarray(flow_vec, dtype=float).reshape(-1)
    except Exception:
        return None
    if vec.size < 3:
        return None
    vec = np.nan_to_num(vec[:3], nan=0.0, posinf=0.0, neginf=0.0)
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
    aos_deg = float(np.degrees(np.arctan2(-unit[0], unit[1])))
    return aoa_deg, aos_deg, speed


def _resolve_nominal_attitude(payload, default_aos_deg):
    nominal_aoa = _payload_value(payload, ["nominal_aoa_deg"], None)
    nominal_aos = _payload_value(payload, ["nominal_aos_deg"], None)
    if nominal_aoa is not None or nominal_aos is not None:
        try:
            aoa = float(nominal_aoa if nominal_aoa is not None else 0.0)
        except Exception:
            aoa = 0.0
        try:
            aos = float(nominal_aos if nominal_aos is not None else default_aos_deg)
        except Exception:
            aos = float(default_aos_deg)
        return aoa, aos
    return _resolve_attitude(payload, default_aos_deg)


def _resolve_effective_attitude(payload, default_aos_deg, flow_speed_mps, altitude_km):
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
    use_winds = payload.get("use_winds", wind_vec is not None)
    if isinstance(use_winds, str):
        use_winds = use_winds.strip().lower() in {"1", "true", "yes", "on"}
    if wind_vec is None or not use_winds:
        return nominal_aoa_deg, nominal_aos_deg, float(flow_speed_mps)
    try:
        wind = np.asarray(wind_vec, dtype=float).reshape(-1)[:3] if wind_vec is not None else None
    except Exception:
        wind = None
    if wind is None or wind.size < 3:
        return nominal_aoa_deg, nominal_aos_deg, float(flow_speed_mps)
    wind = np.nan_to_num(wind, nan=0.0, posinf=0.0, neginf=0.0)
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


def _resolve_adbsat_aos_offset(payload):
    value = _payload_value(payload, ["adbsat_aos_offset_deg", "adbsat_aos_offset"], 90.0)
    try:
        return float(value)
    except Exception:
        return 90.0


def _load_atmosphere_database(atmos_path, altitude_km):
    requested_alt_km = int(round(float(altitude_km)))
    exact_path = os.path.join(atmos_path, f"database_{requested_alt_km}km.csv")
    if os.path.exists(exact_path):
        return pd.read_csv(exact_path), requested_alt_km, requested_alt_km

    available = []
    for name in os.listdir(atmos_path):
        match = re.fullmatch(r"database_(\d+)km\.csv", name)
        if match:
            available.append(int(match.group(1)))
    if not available:
        raise FileNotFoundError(
            f"No atmosphere database files found in '{atmos_path}' "
            f"(expected files like database_200km.csv)."
        )

    nearest_alt_km = min(available, key=lambda x: abs(x - requested_alt_km))
    nearest_path = os.path.join(atmos_path, f"database_{nearest_alt_km}km.csv")
    print(
        f"[WARN] Atmosphere database for {requested_alt_km} km not found. "
        f"Using nearest available table: {nearest_alt_km} km."
    )
    return pd.read_csv(nearest_path), requested_alt_km, nearest_alt_km


def run_simulation(
    gsi_model,
    run_id,
    altitude,
    aos_deg,
    adbsat_path,
    env_payload_path=None,
    verbose=False,
    delete_temp_files=False,
):
    """
    Runs a single simulation with altitude and AoS taken from command-line arguments.

    :param gsi_model: The GSI model name
    :param run_id: The run ID for the simulation
    :param altitude: The altitude in km
    :param aos_deg: The angle of sideslip in degrees
    :param adbsat_path: The base path for ADBSat data
    :param verbose: If True, additional details are printed
    :param delete_temp_files: If True, temporary files are deleted
    :return: Path to the output file
    """
    

    start = time.time()
    payload = {}
    payload_source = env_payload_path
    if isinstance(env_payload_path, dict):
        payload = dict(env_payload_path)
        payload_source = "<batch_payload>"
    elif env_payload_path:
        try:
            with open(env_payload_path, "r", encoding="utf-8") as f:
                p = json.load(f)
            if isinstance(p, dict):
                payload = p
        except Exception as exc:
            raise RuntimeError(f"Failed to read ADBSat payload JSON: {env_payload_path}") from exc

    # Define file paths
    mod_name = _resolve_geometry_model(payload, adbsat_path, default_model="Cube")
    mod_in = os.path.join(adbsat_path,'inou','obj_files',f'{mod_name}.obj')
    mod_out = os.path.join(adbsat_path,f'MFMC_Jobs_{gsi_model}')
    res_out = mod_out
    os.makedirs(res_out, exist_ok=True)
    atmos_path = os.path.join(adbsat_path, 'atmos_data')

    # Import model
    mod_out = os.path.join(adbsat_path,'inou','models',f'{mod_name}.mat')
    mesh = loadmat(mod_out)
    expected_mesh_fingerprint = str(payload.get("surface_mesh_fingerprint", "")).strip()
    if expected_mesh_fingerprint:
        actual_mesh_fingerprint = _loaded_mat_string(mesh, "mesh_fingerprint")
        if actual_mesh_fingerprint != expected_mesh_fingerprint:
            raise RuntimeError(
                "ADBSat MAT mesh fingerprint mismatch: "
                f"actual={actual_mesh_fingerprint or '<missing>'}, expected={expected_mesh_fingerprint}, path={mod_out}"
            )
    N_elems = np.shape(mesh['meshdata']['XData'][0, 0])[1]

    # Constants
    constants = ConstantsData()

    # Convert altitude to meters
    alt = altitude * 1e3


    # Fixed parameters for inclination and environment
    inc = 130  # Inclination in degrees
    env = {"h": alt}

    base_aos_deg = float(aos_deg)
    adbsat_aos_offset_deg = _resolve_adbsat_aos_offset(payload)
    aoa_deg = 0.0
    aos_deg = base_aos_deg + adbsat_aos_offset_deg

    # Model parameters
    shadow = True
    solar = True
    inparam = {
        "gsi_model": gsi_model,
        "alpha": 0.9 * np.ones(N_elems),
        "alphaN": 0.9 * np.ones(N_elems),
        "sigmaN": 0.9 * np.ones(N_elems),
        "sigmaT": 0.7 *  np.ones(N_elems),
        "Tw": 300,
        "sol_cR": 0.15,
        "sol_cD": 0.25
    }

    env_model = str(payload.get("environment_model", os.environ.get("MFMC_ENV_MODEL", "csv")))

    # Resolve sample-wise AoS/AoA (and optional jitter) from payload if present,
    # then correct them for any explicit wind in the payload.
    flow_speed_mps = np.sqrt(constants.mu_E / (constants.R_E + alt))
    flow_zero_direction = _payload_value(
        payload,
        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
        None,
    )
    aoa_sample, aos_sample, _ = _resolve_effective_attitude(payload, base_aos_deg, flow_speed_mps, altitude)
    if flow_zero_direction is not None:
        desired_flow = _flow_unit_from_angles(aos_sample, aoa_sample, flow_zero_direction)
        aoa_deg, aos_deg = _adbsat_calc_angles_from_flow_vector(desired_flow)
        adbsat_aos_offset_applied_deg = 0.0
    else:
        aoa_deg = float(aoa_sample)
        aos_deg = float(aos_sample + adbsat_aos_offset_deg)
        adbsat_aos_offset_applied_deg = float(adbsat_aos_offset_deg)

    # GSI / surface parameter overrides from payload (sample-level first).
    surface_params = _resolve_surface_parameters(payload)
    inparam["alpha"] = _as_panel_array(surface_params["alpha"], N_elems, 0.9)
    inparam["trans_accommodation"] = _as_panel_array(surface_params["trans_accommodation"], N_elems, 0.9)
    inparam["momentum_accommodation"] = _as_panel_array(surface_params["momentum_accommodation"], N_elems, 0.81)
    inparam["alphaN"] = _as_panel_array(surface_params["alphaN"], N_elems, 0.9)
    inparam["sigmaN"] = _as_panel_array(surface_params["sigmaN"], N_elems, 0.9)
    inparam["sigmaT"] = _as_panel_array(surface_params["sigmaT"], N_elems, 0.7)
    try:
        inparam["Tw"] = float(surface_params["Tw"])
    except Exception:
        inparam["Tw"] = 300.0
    try:
        inparam["sol_cR"] = float(surface_params["sol_cR"])
    except Exception:
        inparam["sol_cR"] = 0.15
    try:
        inparam["sol_cD"] = float(surface_params["sol_cD"])
    except Exception:
        inparam["sol_cD"] = 0.25
    reference_area_m2 = _payload_value(
        payload,
        ["reference_area_m2", "piclas_reference_area_m2", "area_ref_m2", "A_ref"],
        None,
    )
    try:
        reference_area_m2 = float(reference_area_m2)
    except Exception:
        reference_area_m2 = None
    if reference_area_m2 is not None and np.isfinite(reference_area_m2) and reference_area_m2 > 0.0:
        inparam["reference_area_m2"] = float(reference_area_m2)
        inparam["reference_area_source"] = str(payload.get("reference_area_source", "payload"))

    # Load atmospheric data only when needed. For pymsis/shared payload rows,
    # environment() can work directly from payload without a CSV table.
    has_payload_atmosphere = ("atmosphere_row" in payload) or ("rho" in payload and "Tinf" in payload)
    if str(env_model) == "csv" and not has_payload_atmosphere:
        database, _, _ = _load_atmosphere_database(atmos_path, altitude)
    else:
        database = None

    # Compute environmental properties
    inparam = environment(inparam, database, run_id, alt, env_payload=payload, env_model=env_model)
    context = (
        f"gsi_model={gsi_model}, run_id={run_id}, geometry={mod_name}, altitude_km={altitude}, "
        f"aoa_deg={aoa_deg}, aos_deg={aos_deg}, env_model={env_model}, payload={payload_source}"
    )
    _validate_environment_inputs(inparam, context)

    if _payload_has_key(payload, ["alpha", "energy_accommodation"]):
        _assert_panel_binding(
            "alpha",
            inparam["alpha"],
            surface_params["alpha"],
            N_elems,
            0.9,
            context,
        )
    if _payload_has_key(payload, ["alphaN", "alpha_n", "energy_accommodation"]):
        _assert_panel_binding(
            "alphaN",
            inparam["alphaN"],
            surface_params["alphaN"],
            N_elems,
            0.9,
            context,
        )
    if _payload_has_key(payload, ["sigmaT", "sigma_t"]):
        _assert_panel_binding(
            "sigmaT",
            inparam["sigmaT"],
            surface_params["sigmaT"],
            N_elems,
            0.7,
            context,
        )
    if _payload_has_key(payload, ["momentum_accommodation"]):
        _assert_panel_binding(
            "momentum_accommodation",
            inparam["momentum_accommodation"],
            surface_params["momentum_accommodation"],
            N_elems,
            0.81,
            context,
        )

    thermal_incident_temperature = 0.5 * float(inparam["s"]) ** 2 * float(inparam["Tinf"])
    write_input_audit = str(
        payload.get("write_input_audit", os.environ.get("ADBSAT_WRITE_INPUT_AUDIT", "1"))
    ).strip().lower() not in {"0", "false", "no", "off"}
    if write_input_audit:
        _write_input_audit(
            res_out,
            {
            "gsi_model": str(gsi_model),
            "run_id": int(run_id),
            "geometry_model": str(mod_name),
            "altitude_km": float(altitude),
            "base_aos_deg": float(base_aos_deg),
            "payload_effective_aoa_deg": float(aoa_sample),
            "payload_effective_aos_deg": float(aos_sample),
            "adbsat_calc_aoa_deg": float(aoa_deg),
            "adbsat_calc_aos_deg": float(aos_deg),
            "adbsat_aos_offset_deg": float(adbsat_aos_offset_deg),
            "adbsat_aos_offset_applied_deg": float(adbsat_aos_offset_applied_deg),
            "payload_flow_zero_direction": flow_zero_direction,
            "payload_path": str(payload_source),
            "payload_energy_accommodation": _payload_value(payload, ["energy_accommodation"], None),
            "payload_trans_accommodation": _payload_value(payload, ["trans_accommodation"], None),
            "payload_momentum_accommodation": _payload_value(payload, ["momentum_accommodation"], None),
            "payload_alpha": _payload_value(payload, ["alpha"], None),
            "payload_alphaN": _payload_value(payload, ["alphaN", "alpha_n"], None),
            "payload_sigmaT": _payload_value(payload, ["sigmaT", "sigma_t"], None),
            "payload_wall_temperature_k": _payload_value(payload, ["Tw", "wall_temperature_k", "surface_temperature_k"], None),
            "payload_reference_area_m2": _payload_value(
                payload,
                ["reference_area_m2", "piclas_reference_area_m2", "area_ref_m2", "A_ref"],
                None,
            ),
            "reference_area_m2": float(inparam["reference_area_m2"]) if "reference_area_m2" in inparam else None,
            "reference_area_source": inparam.get("reference_area_source"),
            "resolved_surface_parameters": surface_params,
            "alpha": _panel_summary(inparam["alpha"]),
            "trans_accommodation": _panel_summary(inparam["trans_accommodation"]),
            "momentum_accommodation": _panel_summary(inparam["momentum_accommodation"]),
            "alphaN": _panel_summary(inparam["alphaN"]),
            "sigmaN": _panel_summary(inparam["sigmaN"]),
            "sigmaT": _panel_summary(inparam["sigmaT"]),
            "Tw": float(inparam["Tw"]),
            "Tinf": float(inparam["Tinf"]),
            "s": float(inparam["s"]),
            "sentman_Ti": float(thermal_incident_temperature),
            "sentman_Tw_over_Ti": float(float(inparam["Tw"]) / thermal_incident_temperature)
            if thermal_incident_temperature > 0.0
            else None,
            "rho_mass_density": float(np.asarray(inparam["rho"], dtype=float).reshape(-1)[-1]),
            "rho_species_sum": float(np.sum(np.asarray(inparam["rho"], dtype=float).reshape(-1)[:9])),
            },
        )

    # Compute dynamic pressure
    dyn_p = 0.5 * inparam['rho'][-1] * inparam['vinf'] ** 2
    _require_finite("dynamic_pressure", dyn_p, context)
    print(dyn_p)

    # Compute aerodynamic coefficients
    calc_kwargs = {}
    try:
        signature_target = getattr(calc_coeff, "side_effect", None)
        if not callable(signature_target):
            signature_target = calc_coeff
        calc_signature = inspect.signature(signature_target)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in calc_signature.parameters.values()
        )
        if accepts_kwargs or "write_mat" in calc_signature.parameters:
            calc_kwargs["write_mat"] = str(
                payload.get("write_mat", os.environ.get("ADBSAT_WRITE_MAT", "1"))
            ).strip().lower() not in {"0", "false", "no", "off"}
        if accepts_kwargs or "return_details" in calc_signature.parameters:
            calc_kwargs["return_details"] = True
    except (TypeError, ValueError):
        pass
    coeffs = calc_coeff(
        mod_out,
        res_out,
        [np.radians(aoa_deg)],
        [np.radians(aos_deg)],
        inparam,
        shadow,
        solar,
        dyn_p,
        delete_temp_files,
        verbose,
        **calc_kwargs,
    )
    _validate_coefficients(coeffs, context)
    _write_panel_surface_field(coeffs, payload, res_out, gsi_model, run_id)

    
    end = time.time()
    runtime = (end - start)*1000

 
    return coeffs, runtime


def parallel_run(param_list, altitude, aos_deg, adbsat_path, num_workers=32):
    """
    Runs multiple simulations in parallel and stores results in a file with 3 columns: gsi_model, Fd, cpu_time.

    :param param_list: List of tuples containing (gsi_model, run_id)
    :param altitude: Altitude in km
    :param aos_deg: Angle of Sideslip in degrees
    :param adbsat_path: The base path for ADBSat
    :param num_workers: Number of parallel processes
    """
    job_args = []
    for item in param_list:
        gsi_model = item[0]
        run_id = item[1]
        item_altitude = altitude
        item_aos_deg = aos_deg
        env_payload_path = None

        if len(item) >= 5:
            item_altitude = float(item[2])
            item_aos_deg = float(item[3])
            env_payload_path = item[4]
        elif len(item) >= 3:
            env_payload_path = item[2]

        job_args.append((gsi_model, run_id, item_altitude, item_aos_deg, adbsat_path, env_payload_path))
    
    N_chunks = int(len(param_list)/(4*num_workers))
    if N_chunks <= 1:
        N_chunks = 1

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(run_simulation, job_args)

    # Speichere die Ergebnisse mit 3 Spalten (gsi_model, Fd, cpu_time)
    output_file = "all_results.txt"
    # Speichere die Ergebnisse mit 3 Spalten (gsi_model, idx, Fd_array, cpu_time)
    with open(output_file, "w") as f:
        f.write("gsi_model idx C_D C_L C_Y C_Mx C_My C_Mz cpu_time_ms\n")
        for item, (coeffs, cpu_time) in zip(param_list, results):
            gsi_model, run_id = item[0], item[1]
            if not isinstance(coeffs, dict):
                # Legacy fallback if calc_coeff returns only drag scalar.
                coeffs = {
                    "C_D": float(coeffs),
                    "C_L": float("nan"),
                    "C_Y": float("nan"),
                    "C_Mx": float("nan"),
                    "C_My": float("nan"),
                    "C_Mz": float("nan"),
                }
            f.write(
                f"{gsi_model} {run_id} "
                f"{coeffs.get('C_D', float('nan'))} "
                f"{coeffs.get('C_L', float('nan'))} "
                f"{coeffs.get('C_Y', float('nan'))} "
                f"{coeffs.get('C_Mx', float('nan'))} "
                f"{coeffs.get('C_My', float('nan'))} "
                f"{coeffs.get('C_Mz', float('nan'))} "
                f"{cpu_time:.2f}\n"
            )


    print(f"All simulations completed. Results saved in {output_file}")



def read_input_file(txt_file):
    """
    Reads the input text file containing gsi_model and run_id.

    :param txt_file: Path to the input text file
    :return: List of tuples in one of two formats:
             - (gsi_model, run_id, payload_path)
             - (gsi_model, run_id, altitude_km, aos_deg, payload_path)
    """
    out = []
    payload_cache = {}
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if len(parts) >= 5 and parts[2] in {"@payloads", "@payload_pickle"}:
                gsi_model = parts[0]
                run_id = parts[1]
                payload_file = parts[3]
                payload_index = int(parts[4])
                if payload_file not in payload_cache:
                    if parts[2] == "@payload_pickle":
                        with open(payload_file, "rb") as pf:
                            payload_data = pickle.load(pf)
                    else:
                        with open(payload_file, "r", encoding="utf-8") as pf:
                            payload_data = json.load(pf)
                    if not isinstance(payload_data, list):
                        raise ValueError(f"ADBSat payload batch must be a list: {payload_file}")
                    payload_cache[payload_file] = payload_data
                out.append((gsi_model, run_id, payload_cache[payload_file][payload_index]))
                continue
            if len(parts) >= 5:
                gsi_model = parts[0]
                run_id = parts[1]
                altitude_km = parts[2]
                aos_deg = parts[3]
                env_payload_path = parts[4]
                out.append((gsi_model, run_id, altitude_km, aos_deg, env_payload_path))
                continue

            gsi_model = parts[0]
            run_id = parts[1]
            env_payload_path = parts[2] if len(parts) >= 3 else None
            out.append((gsi_model, run_id, env_payload_path))
    return out


if __name__ == "__main__":
    # Check for correct command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python simulate.py <altitude> <AoS> <txt-file>")
        sys.exit(1)

    # Read command-line arguments
    altitude = int(sys.argv[1])  # Altitude in km
    aos_deg = int(sys.argv[2])  # AoS in degrees
    txt_file = sys.argv[3]  # Path to text file

    # Resolve ADBSat base path from environment or from this script location.
    adbsat_path = os.environ.get("ADBSAT_PATH", os.path.abspath(os.path.dirname(__file__)))

    # Read input file with gsi_model and run_id
    simulations = read_input_file(txt_file)

    # Run parallel simulations
    parallel_run(simulations, altitude, aos_deg, adbsat_path)
