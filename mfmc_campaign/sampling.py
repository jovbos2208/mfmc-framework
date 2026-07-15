from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_AMU_KG = 1.66053906660e-27
_SPECIES_MASS_KG = {
    "O": 15.999 * _AMU_KG,
    "N2": 28.0134 * _AMU_KG,
    "O2": 31.998 * _AMU_KG,
    "HE": 4.002602 * _AMU_KG,
}


@dataclass
class SamplingContext:
    regime_id: str
    active_source_blocks: Sequence[str]


def _apply_transform(value: float, transform: str) -> float:
    if transform == "log":
        return float(math.log(value))
    if transform == "exp":
        return float(math.exp(value))
    if transform == "square":
        return float(value * value)
    if transform == "sqrt":
        return float(math.sqrt(max(value, 0.0)))
    return float(value)


def _clip_bounds(value: Any, bounds: List[float]) -> Any:
    if value is None or bounds is None:
        return value
    lo, hi = bounds
    try:
        return max(lo, min(hi, value))
    except TypeError:
        return value


def _sample_from_distribution(rng: np.random.Generator, dist: Dict[str, Any], n: int) -> np.ndarray:
    kind = dist.get("kind", "fixed")
    params = dist.get("params", {})

    if kind == "fixed":
        value = params.get("value")
        return np.full(n, value)
    if kind == "uniform":
        return rng.uniform(float(params["low"]), float(params["high"]), size=n)
    if kind == "int_uniform":
        return rng.integers(int(params["low"]), int(params["high"]) + 1, size=n)
    if kind == "normal":
        return rng.normal(float(params["mean"]), float(params["std"]), size=n)
    if kind == "lognormal":
        return rng.lognormal(float(params["mean"]), float(params["sigma"]), size=n)
    if kind == "choice":
        values = params.get("values", [])
        probs = params.get("probabilities")
        return rng.choice(values, size=n, p=probs)
    if kind == "empirical":
        values = params.get("values", [])
        if not values:
            raise ValueError("Empirical distribution requires non-empty 'values'")
        return rng.choice(values, size=n, replace=True)

    raise ValueError(f"Unsupported distribution kind '{kind}'")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _first_float(row: Dict[str, Any], names: Sequence[str]) -> Optional[float]:
    for name in names:
        if name in row and row[name] not in {"", None}:
            value = _float_or_none(row[name])
            if value is not None:
                return value
    return None


def _first_text(row: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value not in {"", None}:
            return str(value)
    return None


def _composition_descriptor(o: float, n2: float, o2: float, he: float) -> str:
    return f"O={o:.6g},N2={n2:.6g},O2={o2:.6g},He={he:.6g}"


def _atmosphere_row_from_composition(
    *,
    density_kg_m3: float,
    temperature_k: float,
    x_o: float,
    x_n2: float,
    x_o2: float,
    x_he: float,
) -> List[float]:
    fractions = np.asarray([x_o, x_n2, x_o2, x_he], dtype=float)
    fractions = np.clip(np.nan_to_num(fractions, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = float(np.sum(fractions))
    if total <= 0.0:
        fractions = np.asarray([0.7, 0.27, 0.02, 0.01], dtype=float)
    else:
        fractions = fractions / total

    masses = np.asarray(
        [_SPECIES_MASS_KG["O"], _SPECIES_MASS_KG["N2"], _SPECIES_MASS_KG["O2"], _SPECIES_MASS_KG["HE"]],
        dtype=float,
    )
    mean_particle_mass = float(np.dot(fractions, masses))
    total_number_density = float(density_kg_m3) / max(mean_particle_mass, 1e-30)
    n_o, n_n2, n_o2, n_he = (fractions * total_number_density).tolist()
    return [
        float(density_kg_m3),
        float(n_n2),
        float(n_o2),
        float(n_o),
        float(n_he),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        float(temperature_k),
    ]


def _datetime_to_doy_seconds(datetime_utc: str) -> tuple[int, int, float]:
    dt = np.datetime64(datetime_utc, "s")
    year = int(str(dt)[:4])
    year_start = np.datetime64(f"{year}-01-01T00:00:00", "s")
    delta_s = int((dt - year_start).astype("timedelta64[s]").astype(int))
    doy = delta_s // 86400 + 1
    sec = float(delta_s % 86400)
    return year, doy, sec


def _sample_hwm14_wind(record: Dict[str, Any]) -> Optional[List[float]]:
    try:
        import hwm14  # type: ignore
    except Exception:
        hwm14 = None

    altitude_km = _float_or_none(record.get("altitude_km"))
    lat_deg = _float_or_none(record.get("lat_deg"))
    lon_deg = _float_or_none(record.get("lon_deg"))
    datetime_utc = record.get("datetime_utc")
    if altitude_km is None or lat_deg is None or lon_deg is None or datetime_utc is None:
        return None

    year, doy, sec = _datetime_to_doy_seconds(str(datetime_utc))
    f107 = float(record.get("f107", 150.0))
    f107a = float(record.get("f107a", f107))
    ap = float(record.get("ap", 4.0))

    if hwm14 is not None:
        for name in ("hwm14", "run", "wind"):
            fn = getattr(hwm14, name, None)
            if callable(fn):
                try:
                    out = fn(year, doy, sec, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
                    arr = np.asarray(out, dtype=float).reshape(-1)
                    if arr.size >= 2:
                        return [float(arr[0]), float(arr[1]), float(arr[2]) if arr.size >= 3 else 0.0]
                except Exception:
                    continue

    try:
        from pyhwm2014 import HWM14  # type: ignore
    except Exception:
        return None

    try:
        ut_hours = float(sec / 3600.0)
        model = HWM14(
            alt=float(altitude_km),
            altlim=[float(altitude_km), float(altitude_km)],
            altstp=1,
            year=int(year),
            day=int(doy),
            ut=ut_hours,
            glat=float(lat_deg),
            glon=float(lon_deg),
            ap=[-1, float(ap)],
            option=1,
            verbose=False,
        )
        zonal = np.asarray(getattr(model, "Uwind"), dtype=float).reshape(-1)
        meridional = np.asarray(getattr(model, "Vwind"), dtype=float).reshape(-1)
        if zonal.size and meridional.size:
            return [float(zonal[0]), float(meridional[0]), 0.0]
    except Exception:
        return None
    return None


def _trajectory_record_from_csv_row(row: Dict[str, Any], index: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    utc = _first_text(row, ["utc", "datetime_utc", "time", "timestamp"])
    lat = _first_float(row, ["geodetic_lat_deg", "lat_deg", "latitude_deg", "latitude"])
    lon = _first_float(row, ["geodetic_lon_deg", "lon_deg", "longitude_deg", "longitude"])
    altitude = _first_float(row, ["altitude_km", "alt_km", "height_km"])
    speed = _first_float(row, ["relative_speed_m_s", "relative_speed_mps", "flow_speed_mps", "freestream_speed_mps"])
    temperature = _first_float(row, ["temperature_K", "temperature_k", "freestream_temperature"])
    density = _first_float(row, ["density_kg_m3", "mass_density"])

    record: Dict[str, Any] = {
        "trajectory_index": int(index),
        "database_index": int(index),
    }
    if utc is not None:
        record["datetime_utc"] = utc
    if lat is not None:
        record["lat_deg"] = lat
    if lon is not None:
        record["lon_deg"] = lon
    if altitude is not None:
        record["altitude_km"] = altitude
    if speed is not None:
        record["relative_speed_mps"] = speed
        record["flow_speed_mps"] = speed
        record["freestream_speed_mps"] = speed
    if temperature is not None:
        record["freestream_temperature"] = temperature
    local_solar_time = _first_float(row, ["local_solar_time_h", "lst_h"])
    if local_solar_time is not None:
        record["local_solar_time_h"] = local_solar_time

    for key in ["mission", "arc_class", "atmosphere_model", "model_note"]:
        if row.get(key) not in {"", None}:
            record[key] = row[key]

    x_o = _first_float(row, ["x_o_fraction", "x_O_fraction", "o_fraction"])
    x_n2 = _first_float(row, ["x_n2_fraction", "x_N2_fraction", "n2_fraction"])
    x_o2 = _first_float(row, ["x_o2_fraction", "x_O2_fraction", "o2_fraction"])
    x_he = _first_float(row, ["x_he_fraction", "x_HE_fraction", "he_fraction"])
    if all(v is not None for v in [density, temperature, x_o, x_n2, x_o2, x_he]):
        record["density_kg_m3"] = float(density)  # type: ignore[arg-type]
        record["composition_descriptor"] = _composition_descriptor(float(x_o), float(x_n2), float(x_o2), float(x_he))
        if str(cfg.get("atmosphere", "from_csv")).lower() in {"from_csv", "csv", "precomputed"}:
            record["atmosphere_row"] = _atmosphere_row_from_composition(
                density_kg_m3=float(density),
                temperature_k=float(temperature),
                x_o=float(x_o),
                x_n2=float(x_n2),
                x_o2=float(x_o2),
                x_he=float(x_he),
            )

    wind = [
        _first_float(row, ["wind_east_mps", "wind_e_mps", "hwm14_wind_east_mps"]),
        _first_float(row, ["wind_north_mps", "wind_n_mps", "hwm14_wind_north_mps"]),
        _first_float(row, ["wind_up_mps", "wind_u_mps", "hwm14_wind_up_mps"]),
    ]
    if any(v is not None for v in wind):
        wind_vec = [float(v if v is not None else 0.0) for v in wind]
    else:
        wind_vec = None

    if wind_vec is not None:
        record["wind_east_mps"] = float(wind_vec[0])
        record["wind_north_mps"] = float(wind_vec[1])
        record["wind_up_mps"] = float(wind_vec[2]) if len(wind_vec) > 2 else 0.0
        record["wind_enu_mps"] = [record["wind_east_mps"], record["wind_north_mps"], record["wind_up_mps"]]

    return record


def _load_trajectory_records(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = cfg.get(
        "path",
        cfg.get(
            "csv",
            cfg.get("trajectory_csv", cfg.get("environment_csv")),
        ),
    )
    if not path:
        if bool(cfg.get("enabled", False)):
            raise ValueError(
                "sampling.trajectory.enabled is true, but no trajectory CSV path was provided. "
                "Set sampling.trajectory.path to the GOCE environment CSV."
            )
        return []
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"sampling.trajectory path not found: {path}")

    stride = max(1, int(cfg.get("stride", 1)))
    limit = cfg.get("limit")
    limit = int(limit) if limit is not None else None
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            if idx % stride != 0:
                continue
            records.append(_trajectory_record_from_csv_row(row, idx, cfg))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"sampling.trajectory produced no records from: {path}")
    return records


class InputModel:
    def __init__(
        self,
        variables: List[Dict[str, Any]],
        sampling: Dict[str, Any],
        regime_label_map: Optional[Dict[str, str]] = None,
    ):
        self.variables = variables
        self.sampling = sampling
        self.regime_label_map = dict(regime_label_map or {})
        trajectory_cfg = sampling.get("trajectory", {})
        self.trajectory_cfg = trajectory_cfg if isinstance(trajectory_cfg, dict) else {}
        self.trajectory_records = (
            _load_trajectory_records(self.trajectory_cfg)
            if bool(self.trajectory_cfg.get("enabled", False))
            else []
        )

    def _override_keys(self, regime_id: str) -> List[str]:
        keys = [regime_id]
        for label, rid in self.regime_label_map.items():
            if rid == regime_id and label not in keys:
                keys.append(label)
        return keys

    def _resolve_distribution(self, var: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
        overrides = var.get("regime_overrides", {})
        for key in self._override_keys(regime_id):
            if key in overrides:
                override = overrides[key]
                merged = dict(var.get("distribution", {}))
                if "distribution" in override:
                    merged.update(override.get("distribution", {}))
                return merged
        return dict(var.get("distribution", {}))

    def _resolve_baseline(self, var: Dict[str, Any], regime_id: str) -> Any:
        overrides = var.get("regime_overrides", {})
        for key in self._override_keys(regime_id):
            if key in overrides and "baseline" in overrides[key]:
                return overrides[key]["baseline"]
        return var.get("baseline")

    def sample(self, n: int, context: SamplingContext, rng: np.random.Generator) -> List[Dict[str, Any]]:
        method = self.sampling.get("method", "independent")
        if method == "blockwise_joint":
            rows = self._sample_blockwise_joint(n, context, rng)
        else:
            rows = self._sample_independent(n, context, rng)
        return self._apply_trajectory(rows, rng)

    def _trajectory_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        count = len(self.trajectory_records)
        mode = str(self.trajectory_cfg.get("sample", self.trajectory_cfg.get("strategy", "random"))).lower()
        if mode in {"sequential", "ordered"}:
            start = int(self.trajectory_cfg.get("start_index", 0))
            return (np.arange(start, start + n, dtype=int) % count).astype(int)
        if mode in {"without_replacement", "choice_without_replacement"} and n <= count:
            return rng.choice(count, size=n, replace=False).astype(int)
        return rng.choice(count, size=n, replace=True).astype(int)

    def _apply_trajectory(self, rows: List[Dict[str, Any]], rng: np.random.Generator) -> List[Dict[str, Any]]:
        if not self.trajectory_records:
            if bool(self.trajectory_cfg.get("enabled", False)):
                raise ValueError(
                    "sampling.trajectory.enabled is true, but no trajectory records were loaded. "
                    "Check sampling.trajectory.path and make sure the running code includes trajectory CSV loading."
                )
            return rows

        overwrite = bool(self.trajectory_cfg.get("overwrite", True))
        indices = self._trajectory_indices(len(rows), rng)
        for out, record_idx in zip(rows, indices):
            record = self.trajectory_records[int(record_idx)]
            if str(self.trajectory_cfg.get("wind_model", "")).lower() == "hwm14" and "wind_enu_mps" not in record:
                wind_vec = _sample_hwm14_wind(record)
                if wind_vec is None and bool(self.trajectory_cfg.get("require_hwm14", False)):
                    raise ImportError("hwm14 is required for sampling.trajectory.wind_model='hwm14'")
                if wind_vec is not None:
                    record["wind_east_mps"] = float(wind_vec[0])
                    record["wind_north_mps"] = float(wind_vec[1])
                    record["wind_up_mps"] = float(wind_vec[2]) if len(wind_vec) > 2 else 0.0
                    record["wind_enu_mps"] = [record["wind_east_mps"], record["wind_north_mps"], record["wind_up_mps"]]
            for key, value in record.items():
                if overwrite or key not in out:
                    out[key] = value
        return rows

    def _sample_independent(self, n: int, context: SamplingContext, rng: np.random.Generator) -> List[Dict[str, Any]]:
        out_rows = [dict() for _ in range(n)]
        active = set(context.active_source_blocks)

        for var in self.variables:
            var_name = str(var["name"])
            src = str(var["source_block"])
            bounds = var.get("bounds")
            transform = var.get("transform")

            if src in active:
                dist = self._resolve_distribution(var, context.regime_id)
                values = _sample_from_distribution(rng, dist, n)
            else:
                baseline = self._resolve_baseline(var, context.regime_id)
                values = np.full(n, baseline)

            for i in range(n):
                value = values[i]
                if transform and isinstance(value, (int, float, np.integer, np.floating)):
                    value = _apply_transform(float(value), str(transform))
                value = _clip_bounds(value, bounds)
                if isinstance(value, np.generic):
                    value = value.item()
                out_rows[i][var_name] = value

        return out_rows

    def _sample_blockwise_joint(self, n: int, context: SamplingContext, rng: np.random.Generator) -> List[Dict[str, Any]]:
        rows = self._sample_independent(n, context, rng)
        active = set(context.active_source_blocks)
        block_covariances = self.sampling.get("block_covariances", {})

        # Replace independent draws with joint MVN samples for blocks where covariance is defined.
        for block_name, block_cfg in block_covariances.items():
            if block_name not in active:
                continue

            var_names = block_cfg.get("variables", [])
            matrix = np.asarray(block_cfg.get("matrix", []), dtype=float)
            means = np.asarray(block_cfg.get("means", [0.0] * len(var_names)), dtype=float)
            if len(var_names) == 0 or matrix.size == 0:
                continue

            if matrix.shape != (len(var_names), len(var_names)):
                raise ValueError(f"Covariance matrix shape mismatch for block '{block_name}'")

            joint = rng.multivariate_normal(mean=means, cov=matrix, size=n)
            for i in range(n):
                for j, name in enumerate(var_names):
                    rows[i][name] = float(joint[i, j])

        return rows


def freeze_all_except(source_blocks: Sequence[str], selected: str) -> List[str]:
    if selected not in source_blocks:
        raise ValueError(f"Selected source block '{selected}' not part of source taxonomy")
    return [selected]
