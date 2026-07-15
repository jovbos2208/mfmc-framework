from __future__ import annotations

from dataclasses import dataclass
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
    if raw.get("high_fidelity", "DSMC").upper() != "DSMC" or raw.get("low_fidelity", "TPMC").upper() != "TPMC":
        raise MFPODError("Primary two-fidelity workflow requires high_fidelity=DSMC and low_fidelity=TPMC")
    if raw.get("coordinate_frame", "body_fixed") != "body_fixed":
        raise MFPODError("wind_aligned is unavailable until complete transformation metadata exist")
    if raw.get("centering_mode", "per_fidelity_pilot_mean") not in {"none", "common_fixed_reference", "per_fidelity_pilot_mean"}:
        raise MFPODError("Unknown centering_mode")
    archives = {k.upper(): _resolve(config_path, v) for k, v in (raw.get("fidelity_archives") or {}).items()}
    if not archives and raw.get("input_root"):
        root = _resolve(config_path, raw["input_root"]); archives = {x: root / f"{x}_surface_loads.npz" for x in ("DSMC", "TPMC")}
    case = str(raw.get("case_name", "Cube-300km"))
    output_root = _resolve(config_path, raw.get("output_root", "../../paper_postprocessed/mfpod_surface_loads"))
    return MFPODConfig(config_path, raw, case, str(raw.get("geometry_id", case)), archives, output_root / case, str(raw.get("snapshot_type", "full_traction")), str(raw.get("coordinate_frame", "body_fixed")), str(raw.get("centering_mode", "per_fidelity_pilot_mean")), int(raw.get("random_seed", 2202)))
