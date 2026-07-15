from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


class MFPODError(RuntimeError):
    pass


@dataclass
class SurfaceGeometry:
    face_area: np.ndarray
    A_ref: float
    geometry_id: str
    coordinate_frame: str = "body_fixed"
    component_order: tuple[str, ...] = ("x", "y", "z")
    face_center: Optional[np.ndarray] = None
    face_normal: Optional[np.ndarray] = None
    reference_point: Optional[np.ndarray] = None

    @property
    def n_faces(self) -> int:
        return int(self.face_area.size)


@dataclass
class SurfaceSnapshotBatch:
    values: np.ndarray
    sample_ids: np.ndarray
    fidelity: str
    snapshot_type: str
    geometry: SurfaceGeometry
    q_inf: np.ndarray
    A_ref_per_sample: np.ndarray
    u_hat_inf: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PODResult:
    modes: np.ndarray
    eigenvalues: np.ndarray
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MFPODResult(PODResult):
    raw_eigenvalues: np.ndarray = field(default_factory=lambda: np.empty(0))
    corrected_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    hf_mc_replacements: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha: Any = 0.0


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
