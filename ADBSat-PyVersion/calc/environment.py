import importlib

import numpy as np

from .ADBSatConstants import ConstantsData

_AMU_KG = 1.66053906660e-27
_PYMSIS_SPECIES_MASS_KG = np.asarray(
    [28.0134, 31.998, 15.999, 4.002602, 1.00794, 39.948, 14.0067, 15.999, 30.006],
    dtype=float,
) * _AMU_KG


def _first_numeric(payload, sample, keys, default):
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


def _space_weather_value(payload, keys, default):
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


def _space_weather_numeric(payload, keys, default):
    val = _space_weather_value(payload, keys, default)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return default


def _space_weather_ap_vector(payload):
    raw = _space_weather_value(payload, ["aps", "ap_vector", "ap_history", "ap_3h", "ap3h"], None)
    if raw is None:
        ap = _space_weather_numeric(payload, ["ap", "ap_daily", "daily_ap"], None)
        if ap is None:
            return None
        return np.full(7, float(ap), dtype=float)

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


def _resolve_pymsis_callable(ps):
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


def _validate_atmosphere_row(row, context):
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


def _sample_pymsis_row(payload, h_m):
    try:
        import pymsis as ps  # type: ignore
    except Exception as exc:
        raise ImportError("pymsis is required for environment_model='pymsis_hwm14'") from exc

    altitude_km = float(payload.get("altitude_km", h_m / 1000.0))
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
    kwargs = {}
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


def _datetime_to_doy_seconds(datetime_utc):
    dt = np.datetime64(str(datetime_utc), "s")
    year = int(str(dt)[:4])
    year_start = np.datetime64(f"{year}-01-01T00:00:00", "s")
    delta_s = int((dt - year_start).astype("timedelta64[s]").astype(int))
    doy = delta_s // 86400 + 1
    sec = float(delta_s % 86400)
    return year, doy, sec


def _sample_hwm14_wind(payload, altitude_km):
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


def _payload_value(payload, keys, default):
    sample = payload.get("sample", {})
    sample = sample if isinstance(sample, dict) else {}
    for key in keys:
        if key in sample:
            return sample[key]
        if key in payload:
            return payload[key]
    return default


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


def _relative_flow_speed_from_payload(payload, nominal_speed_mps):
    explicit = payload.get("relative_flow_speed_mps")
    try:
        if explicit is not None:
            speed = float(explicit)
            if np.isfinite(speed) and speed > 0:
                return speed
    except Exception:
        pass

    nominal_aos = payload.get("nominal_aos_deg", payload.get("aos_deg", 0.0))
    nominal_aoa = payload.get("nominal_aoa_deg", payload.get("aoa_deg", 0.0))
    try:
        aos_deg = float(nominal_aos)
    except Exception:
        aos_deg = 0.0
    try:
        aoa_deg = float(nominal_aoa)
    except Exception:
        aoa_deg = 0.0

    wind_vec = payload.get("wind_enu_mps")
    if wind_vec is None:
        wind_vec = _sample_hwm14_wind(payload, float(payload.get("altitude_km", 0.0)))
    try:
        wind = np.asarray(wind_vec, dtype=float).reshape(-1)[:3] if wind_vec is not None else None
    except Exception:
        wind = None
    if wind is None or wind.size < 3:
        return float(nominal_speed_mps)
    wind = np.nan_to_num(wind, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.linalg.norm(wind)) <= 1e-12:
        return float(nominal_speed_mps)

    flow_zero_direction = _payload_value(
        payload,
        ["flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction", "zero_flow_direction_xyz"],
        None,
    )
    rel = _flow_unit_from_angles(aos_deg, aoa_deg, flow_zero_direction) * float(nominal_speed_mps) - wind
    speed = float(np.linalg.norm(rel))
    return speed if np.isfinite(speed) and speed > 0 else float(nominal_speed_mps)


def _atmosphere_from_payload(payload, h_m):
    if not payload:
        return None

    if "atmosphere_row" in payload:
        return _validate_atmosphere_row(payload["atmosphere_row"], "payload atmosphere_row")

    if "rho" in payload and "Tinf" in payload:
        rho = np.asarray(payload["rho"], dtype=float).reshape(-1)
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
            return np.array([mass_density, n2, o2, o, he, h, ar, n, ao, no, temp], dtype=float)

    if str(payload.get("environment_model", "csv")) == "pymsis_hwm14":
        return _sample_pymsis_row(payload, h_m)

    return None


def _apply_perturbations(atmosphere, payload):
    row = _validate_atmosphere_row(atmosphere, "before perturbations").copy()
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

    return _validate_atmosphere_row(row, "after perturbations")


def _resolve_atmosphere(database, idx, h_m, payload, env_model):
    if payload:
        payload = dict(payload)
        payload.setdefault("environment_model", env_model)
        row = _atmosphere_from_payload(payload, h_m)
        if row is not None:
            return _apply_perturbations(row, payload)

    if env_model == "pymsis_hwm14":
        row = _sample_pymsis_row(payload or {}, h_m)
        return _apply_perturbations(row, payload or {})

    if database is None:
        raise ValueError("No atmosphere database provided for CSV fallback")
    if len(database.index) == 0:
        raise ValueError("Atmosphere database is empty")
    row = database.iloc[int(idx) % len(database.index)].to_numpy(dtype=float)
    return _apply_perturbations(row, payload or {})


def environment(param_eq, database, idx, h, env_payload=None, env_model="csv"):
    constants = ConstantsData()
    payload = env_payload if isinstance(env_payload, dict) else {}
    model = str(payload.get("environment_model", env_model))

    atmosphere = _resolve_atmosphere(database, idx, h, payload, model)
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

    param_eq["Tinf"] = float(atmosphere[10])
    param_eq["rho"] = rho

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

    param_eq["mmean"] = mmean
    param_eq["massConc"] = rho[:8] / max(np.sum(rho[:8]), 1e-30)
    param_eq["Rmean"] = (constants.R / mmean) * 1000
    param_eq["vinf"] = np.sqrt(constants.mu_E / (constants.R_E + h))

    wind_vec = payload.get("wind_enu_mps")
    if wind_vec is None:
        wind_vec = _sample_hwm14_wind(payload, h / 1000.0)
    if isinstance(wind_vec, (list, tuple)) and len(wind_vec) >= 2 and bool(payload.get("apply_wind_to_speed", False)):
        param_eq["vinf"] = _relative_flow_speed_from_payload(payload, param_eq["vinf"])

    param_eq["vth"] = np.sqrt(2 * constants.kb * param_eq["Tinf"] / (mmean / constants.NA / 1000))
    param_eq["s"] = param_eq["vinf"] / max(param_eq["vth"], 1e-30)

    return param_eq
