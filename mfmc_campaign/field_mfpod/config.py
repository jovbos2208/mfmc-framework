from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from .models import MFPODError


@dataclass(frozen=True)
class MFPODConfig:
    path: Path
    raw: dict[str, Any]
    case_name: str
    geometry_id: str
    archives: dict[str, Path]
    output_dir: Path
    snapshot_type: str
    coordinate_frame: str
    centering_mode: str
    random_seed: int


def _resolve(config_path: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (config_path.parent / p).resolve()


def load_config(path: str | Path) -> MFPODConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict): raise MFPODError("MFPOD configuration must be a YAML mapping")
    if str(raw.get("high_fidelity", "DSMC")).upper() != "DSMC":
        raise MFPODError("The field-aware workflow requires high_fidelity=DSMC")
    controls = [str(name).upper() for name in raw.get("control_variates", [raw.get("low_fidelity", "TPMC")])]
    if len(controls) != len(set(controls)) or any(name not in {"TPMC", "SENTMAN"} for name in controls):
        raise MFPODError("control_variates must be a unique subset of [TPMC, SENTMAN]")
    representation = raw.get("field_representation", {}) or {}
    coordinate_frame = str(representation.get("coordinate_frame", raw.get("coordinate_frame", "body_fixed")))
    if coordinate_frame != "body_fixed":
        raise MFPODError("wind_aligned is unavailable until complete transformation metadata exist")
    centering_mode = str(representation.get("centering", raw.get("centering_mode", "pilot_dsmc_mean")))
    if centering_mode not in {"none", "pilot_dsmc_mean", "common_fixed_reference", "per_fidelity_pilot_mean"}:
        raise MFPODError("Unknown centering_mode")
    if representation and str(representation.get("quantity", "Total_ForcePerArea")) != "Total_ForcePerArea":
        raise MFPODError("The field-aware workflow currently requires quantity=Total_ForcePerArea")
    if representation and (not bool(representation.get("nondimensionalize", True)) or not bool(representation.get("area_weighted", True))):
        raise MFPODError("Full-field allocation requires nondimensionalized, area-weighted snapshots")
    allocation = raw.get("field_allocation", raw.get("allocation_optimization", {})) or {}
    constraints = raw.get("allocation_constraints", {}) or {}
    mean_weight = float(allocation.get("mean_weight", 0.25))
    second_weight = float(allocation.get("second_moment_weight", 0.75))
    if not isfinite(mean_weight) or not isfinite(second_weight) or mean_weight < 0.0 or second_weight < 0.0 or not abs(mean_weight + second_weight - 1.0) <= 1.0e-12:
        raise MFPODError("field allocation weights must be nonnegative and sum to one")
    if "budget_hf_equivalent" in constraints:
        budget_hf_equivalent = float(constraints["budget_hf_equivalent"])
        if not isfinite(budget_hf_equivalent) or budget_hf_equivalent <= 0.0:
            raise MFPODError(
                "allocation_constraints.budget_hf_equivalent must be finite and positive"
            )
    pod = raw.get("pod", {}) or {}
    if "eigensolver_tolerance" in pod and (not isfinite(float(pod["eigensolver_tolerance"])) or float(pod["eigensolver_tolerance"]) <= 0.0):
        raise MFPODError("pod.eigensolver_tolerance must be finite and positive")
    if "negative_eigenvalue_tolerance" in pod and (not isfinite(float(pod["negative_eigenvalue_tolerance"])) or float(pod["negative_eigenvalue_tolerance"]) < 0.0):
        raise MFPODError("pod.negative_eigenvalue_tolerance must be finite and nonnegative")
    production = raw.get("production", {}) or {}
    if bool(production.get("enabled", False)):
        for key in ("geometry", "regime", "variables", "models", "maximum_counts"):
            if not production.get(key):
                raise MFPODError(f"production.{key} is required when production.enabled=true")
        maximum = {str(k).upper(): int(v) for k, v in (production.get("maximum_counts", {}) or {}).items()}
        for name in ("DSMC", *controls):
            if maximum.get(name, 0) <= 0:
                raise MFPODError(f"production.maximum_counts.{name} must be positive")
    archives = {k.upper(): _resolve(config_path, v) for k, v in (raw.get("fidelity_archives") or {}).items()}
    if not archives and raw.get("input_root"):
        root = _resolve(config_path, raw["input_root"]); archives = {x: root / f"{x}_surface_loads.npz" for x in ("DSMC", *controls)}
    if bool(production.get("enabled", False)):
        missing_archives = [name for name in ("DSMC", *controls) if name not in archives]
        if missing_archives:
            raise MFPODError(f"fidelity_archives is missing production models {missing_archives}")
    case = str(raw.get("case_name", "Cube-300km"))
    output_root = _resolve(config_path, raw.get("output_root", "../../paper_postprocessed/mfpod_surface_loads"))
    return MFPODConfig(config_path, raw, case, str(raw.get("geometry_id", case)), archives, output_root / case, str(raw.get("snapshot_type", "full_traction")), coordinate_frame, centering_mode, int(raw.get("random_seed", 2202)))
