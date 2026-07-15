from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


REQUIRED_DATA_FIELDS = [
    "force_per_area",
    "sample_id",
    "face_area",
    "A_ref",
    "q_inf",
    "u_hat_inf",
]
PREFERRED_DATA_FIELDS = ["face_center", "face_normal", "reference_point", "C_D"]
SUPPORTED_FIDELITIES = {"DSMC", "TPMC", "SENTMAN"}


class FieldPodMfmcError(RuntimeError):
    pass


@dataclass(frozen=True)
class TopologyTolerance:
    area_rtol: float = 1.0e-10
    center_atol: float = 1.0e-12
    normal_atol: float = 1.0e-12


@dataclass(frozen=True)
class ProjectionWarningThreshold:
    mean: float = 0.10
    max: float = 0.30


@dataclass(frozen=True)
class FieldPodMfmcConfig:
    case_name: str
    output_root: Path
    high_fidelity: str
    low_fidelity_basis_source: str
    snapshot_type: str
    basis_size_s: int
    pod_modes_r: int
    budgets: Tuple[float, ...]
    repeats: int
    random_seed: int
    include_pilot_cost: bool
    mfmc_weight_mode: str
    shared_weight_response: str
    psd_correction: str
    cd_reconstruction_tolerance: float
    topology_tolerance: TopologyTolerance
    projection_residual_warning_threshold: ProjectionWarningThreshold
    fidelity_archives: Dict[str, Path]
    hf_cost: float
    lf_cost: float
    mfmc_hf_fraction: float
    control_variates: Tuple[str, ...] = ()
    control_costs: Optional[Dict[str, float]] = None

    @property
    def output_dir(self) -> Path:
        return self.output_root / _slug(self.case_name)

    @property
    def resolved_control_variates(self) -> Tuple[str, ...]:
        controls = self.control_variates or (self.low_fidelity_basis_source,)
        return tuple(dict.fromkeys(str(value).upper() for value in controls))

    def cost_for(self, fidelity: str) -> float:
        key = str(fidelity).upper()
        if key == self.high_fidelity:
            return float(self.hf_cost)
        if self.control_costs and key in self.control_costs:
            return float(self.control_costs[key])
        if key == self.low_fidelity_basis_source:
            return float(self.lf_cost)
        raise FieldPodMfmcError(f"No cost configured for control variate {key}")


@dataclass
class SurfaceGeometry:
    face_area: np.ndarray
    A_ref: float
    face_center: Optional[np.ndarray] = None
    face_normal: Optional[np.ndarray] = None
    reference_point: Optional[np.ndarray] = None

    @property
    def n_faces(self) -> int:
        return int(self.face_area.size)


@dataclass
class SurfaceFieldArchive:
    fidelity: str
    sample_ids: np.ndarray
    force_per_area: np.ndarray
    q_inf: np.ndarray
    u_hat_inf: np.ndarray
    geometry: SurfaceGeometry
    A_ref_per_sample: np.ndarray
    scalar_cd: Optional[np.ndarray] = None
    source_path: Optional[Path] = None

    @property
    def n_samples(self) -> int:
        return int(self.force_per_area.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.force_per_area.shape[1])


@dataclass
class SnapshotMatrix:
    values: np.ndarray
    sample_ids: np.ndarray
    snapshot_type: str
    fidelity: str
    component_names: List[str]
    metadata: Dict[str, Any]


class SurfaceSnapshotAdapter:
    """Adapter interface for loading and mapping surface-load snapshots.

    Archives consumed here already live on the PICLAS reference topology.
    ADBSat panel fields are conservatively mapped during archive export.
    """

    def load_snapshot(self, sample_id: str, fidelity: str) -> np.ndarray:
        raise NotImplementedError

    def load_geometry(self, case_name: str) -> SurfaceGeometry:
        raise NotImplementedError

    def map_to_reference_surface(self, values: np.ndarray, fidelity: str) -> np.ndarray:
        if fidelity.upper() in SUPPORTED_FIDELITIES:
            return values
        raise NotImplementedError(
            f"Surface mapping for fidelity '{fidelity}' is not implemented. "
            "Panel data require an explicit tested mapping to PICLAS faces."
        )


def _slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _jsonable(value: Any) -> Any:
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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _resolve_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_field_config(path: str | os.PathLike[str]) -> FieldPodMfmcConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FieldPodMfmcError("Field POD/MFMC config root must be a mapping")

    archives: Dict[str, Path] = {}
    input_archives = raw.get("fidelity_archives", raw.get("input_archives", {}))
    if isinstance(input_archives, dict):
        for fidelity, archive_path in input_archives.items():
            archives[str(fidelity).upper()] = _resolve_path(config_path, archive_path)
    input_root = raw.get("input_root")
    if input_root and not archives:
        root = _resolve_path(config_path, input_root)
        for fidelity in raw.get("fidelities", ["DSMC", "TPMC"]):
            archives[str(fidelity).upper()] = root / f"{str(fidelity).upper()}_surface_loads.npz"

    output_root = _resolve_path(config_path, raw.get("output_root", "paper_postprocessed/field_pod_mfmc"))
    topo = raw.get("topology_tolerance", {}) or {}
    proj = raw.get("projection_residual_warning_threshold", {}) or {}
    costs = raw.get("costs", {}) or {}
    controls_raw = raw.get(
        "control_variates",
        raw.get("low_fidelity_models", [raw.get("low_fidelity_basis_source", "TPMC")]),
    )
    if isinstance(controls_raw, str):
        controls_raw = [controls_raw]
    controls = tuple(str(value).upper() for value in controls_raw)
    control_costs = {
        str(name).upper(): float(value)
        for name, value in costs.items()
        if str(name).upper() != str(raw.get("high_fidelity", "DSMC")).upper()
    }

    cfg = FieldPodMfmcConfig(
        case_name=str(raw.get("case_name", raw.get("case_id", "Cube-300km"))),
        output_root=output_root,
        high_fidelity=str(raw.get("high_fidelity", "DSMC")).upper(),
        low_fidelity_basis_source=str(raw.get("low_fidelity_basis_source", "TPMC")).upper(),
        snapshot_type=str(raw.get("snapshot_type", "full_traction")),
        basis_size_s=int(raw.get("basis_size_s", 20)),
        pod_modes_r=int(raw.get("pod_modes_r", 5)),
        budgets=tuple(float(v) for v in raw.get("budgets", [10.0, 20.0, 50.0])),
        repeats=int(raw.get("repeats", 10)),
        random_seed=int(raw.get("random_seed", 12345)),
        include_pilot_cost=bool(raw.get("include_pilot_cost", False)),
        mfmc_weight_mode=str(raw.get("mfmc_weight_mode", "shared_weights")),
        shared_weight_response=str(raw.get("shared_weight_response", "coefficient_norm")),
        psd_correction=str(raw.get("psd_correction", "none")),
        cd_reconstruction_tolerance=float(raw.get("cd_reconstruction_tolerance", 1.0e-6)),
        topology_tolerance=TopologyTolerance(
            area_rtol=float(topo.get("area_rtol", 1.0e-10)),
            center_atol=float(topo.get("center_atol", 1.0e-12)),
            normal_atol=float(topo.get("normal_atol", 1.0e-12)),
        ),
        projection_residual_warning_threshold=ProjectionWarningThreshold(
            mean=float(proj.get("mean", 0.10)),
            max=float(proj.get("max", 0.30)),
        ),
        fidelity_archives=archives,
        hf_cost=float(costs.get("DSMC", raw.get("hf_cost", 1.0))),
        lf_cost=float(costs.get("TPMC", raw.get("lf_cost", 0.1))),
        mfmc_hf_fraction=float(raw.get("mfmc_hf_fraction", 0.5)),
        control_variates=controls,
        control_costs=control_costs,
    )
    if cfg.include_pilot_cost:
        raise NotImplementedError("include_pilot_cost=true is not implemented for the first field demonstrator")
    if cfg.high_fidelity != "DSMC" or cfg.low_fidelity_basis_source != "TPMC":
        raise FieldPodMfmcError("Field POD/MFMC requires high_fidelity=DSMC and low_fidelity_basis_source=TPMC")
    if cfg.low_fidelity_basis_source not in cfg.resolved_control_variates:
        raise FieldPodMfmcError("low_fidelity_basis_source must also appear in control_variates")
    unsupported = sorted(set(cfg.resolved_control_variates) - SUPPORTED_FIDELITIES)
    if unsupported:
        raise FieldPodMfmcError(f"Unsupported field control variates: {unsupported}")
    for control in cfg.resolved_control_variates:
        if cfg.cost_for(control) <= 0.0:
            raise FieldPodMfmcError(f"Cost for {control} must be positive")
    if cfg.mfmc_weight_mode not in {"shared_weights", "entrywise_weights"}:
        raise FieldPodMfmcError("mfmc_weight_mode must be shared_weights or entrywise_weights")
    if cfg.mfmc_weight_mode == "entrywise_weights":
        raise NotImplementedError("entrywise_weights is experimental and not implemented in the first demonstrator")
    if cfg.psd_correction not in {"none", "eigenvalue_clip"}:
        raise FieldPodMfmcError("psd_correction must be none or eigenvalue_clip")
    if not (0.0 < cfg.mfmc_hf_fraction < 1.0):
        raise FieldPodMfmcError("mfmc_hf_fraction must lie between 0 and 1")
    return cfg


def _npz_has(npz: np.lib.npyio.NpzFile, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in npz.files:
            return name
    return None


def inspect_archive(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "present": [], "missing_required": REQUIRED_DATA_FIELDS}
    if path.suffix.lower() != ".npz":
        return {
            "path": str(path),
            "exists": True,
            "error": "Only NPZ surface-load archives are supported by the first demonstrator",
            "present": [],
            "missing_required": REQUIRED_DATA_FIELDS,
        }
    with np.load(path, allow_pickle=False) as npz:
        aliases = {
            "force_per_area": ["force_per_area", "forcePerArea", "Total_ForcePerArea"],
            "sample_id": ["sample_id", "sample_ids"],
            "face_area": ["face_area", "face_areas", "areas", "A_j"],
            "A_ref": ["A_ref", "reference_area_m2", "piclas_reference_area_m2"],
            "q_inf": ["q_inf", "dynamic_pressure", "dyn_p"],
            "u_hat_inf": ["u_hat_inf", "freestream_unit_vector", "flow_dir"],
            "face_center": ["face_center", "face_centers", "centers"],
            "face_normal": ["face_normal", "face_normals", "normals"],
            "reference_point": ["reference_point", "moment_reference_point"],
            "C_D": ["C_D", "cd", "scalar_cd"],
        }
        present = []
        missing = []
        preferred_missing = []
        shapes: Dict[str, Any] = {}
        for logical, names in aliases.items():
            found = _npz_has(npz, names)
            if found is not None:
                present.append(logical)
                shapes[logical] = list(np.asarray(npz[found]).shape)
            elif logical in REQUIRED_DATA_FIELDS:
                missing.append(logical)
            elif logical in PREFERRED_DATA_FIELDS:
                preferred_missing.append(logical)
    return {
        "path": str(path),
        "exists": True,
        "present": present,
        "missing_required": missing,
        "missing_preferred": preferred_missing,
        "shapes": shapes,
    }


def _read_array(npz: np.lib.npyio.NpzFile, logical: str, aliases: Sequence[str], required: bool = True) -> Optional[np.ndarray]:
    name = _npz_has(npz, aliases)
    if name is None:
        if required:
            raise FieldPodMfmcError(f"Missing required field '{logical}' in NPZ archive")
        return None
    return np.asarray(npz[name])


def load_surface_archive(path: Path, fidelity: str) -> SurfaceFieldArchive:
    if fidelity.upper() not in SUPPORTED_FIDELITIES:
        raise NotImplementedError(
            f"Fidelity '{fidelity}' is not supported in the first field demonstrator; "
            "panel mappings must be implemented explicitly."
        )
    if not path.exists():
        raise FileNotFoundError(f"Surface-load archive not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        force = np.asarray(
            _read_array(npz, "force_per_area", ["force_per_area", "forcePerArea", "Total_ForcePerArea"]),
            dtype=np.float64,
        )
        if force.ndim != 3 or force.shape[2] != 3:
            raise FieldPodMfmcError("force_per_area must have shape (n_samples, n_faces, 3)")

        raw_sample_ids = _read_array(npz, "sample_id", ["sample_id", "sample_ids"])
        sample_ids = np.asarray([str(v) for v in raw_sample_ids.reshape(-1)])
        if sample_ids.size != force.shape[0]:
            raise FieldPodMfmcError("sample_id length must match force_per_area n_samples")

        face_area = np.asarray(_read_array(npz, "face_area", ["face_area", "face_areas", "areas", "A_j"]), dtype=np.float64).reshape(-1)
        if face_area.size != force.shape[1]:
            raise FieldPodMfmcError("face_area length must match force_per_area n_faces")
        if np.any(~np.isfinite(face_area)) or np.any(face_area <= 0.0):
            raise FieldPodMfmcError("face_area must be finite and positive")

        A_ref_arr = np.asarray(_read_array(npz, "A_ref", ["A_ref", "reference_area_m2", "piclas_reference_area_m2"]), dtype=np.float64).reshape(-1)
        A_ref = float(A_ref_arr[0])
        if not np.isfinite(A_ref) or A_ref <= 0.0:
            raise FieldPodMfmcError("A_ref must be finite and positive")
        A_ref_per_sample_arr = _read_array(
            npz,
            "A_ref_per_sample",
            ["A_ref_per_sample", "reference_area_per_sample_m2", "piclas_reference_area_per_sample_m2"],
            required=False,
        )
        if A_ref_per_sample_arr is None:
            A_ref_per_sample = np.full(force.shape[0], A_ref, dtype=np.float64)
        else:
            A_ref_per_sample = np.asarray(A_ref_per_sample_arr, dtype=np.float64).reshape(-1)
            if A_ref_per_sample.size == 1:
                A_ref_per_sample = np.full(force.shape[0], float(A_ref_per_sample[0]), dtype=np.float64)
        if (
            A_ref_per_sample.size != force.shape[0]
            or np.any(~np.isfinite(A_ref_per_sample))
            or np.any(A_ref_per_sample <= 0.0)
        ):
            raise FieldPodMfmcError("A_ref_per_sample must be positive with length 1 or n_samples")

        q_inf = np.asarray(_read_array(npz, "q_inf", ["q_inf", "dynamic_pressure", "dyn_p"]), dtype=np.float64).reshape(-1)
        if q_inf.size == 1:
            q_inf = np.full(force.shape[0], float(q_inf[0]), dtype=np.float64)
        if q_inf.size != force.shape[0] or np.any(~np.isfinite(q_inf)) or np.any(q_inf <= 0.0):
            raise FieldPodMfmcError("q_inf must be positive with length 1 or n_samples")

        u_hat_inf = np.asarray(
            _read_array(npz, "u_hat_inf", ["u_hat_inf", "freestream_unit_vector", "flow_dir"]),
            dtype=np.float64,
        )
        if u_hat_inf.ndim == 1:
            u_hat_inf = np.tile(u_hat_inf.reshape(1, 3), (force.shape[0], 1))
        if u_hat_inf.shape != (force.shape[0], 3):
            raise FieldPodMfmcError("u_hat_inf must have shape (3,) or (n_samples, 3)")
        norms = np.linalg.norm(u_hat_inf, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-14):
            raise FieldPodMfmcError("u_hat_inf contains non-finite or zero vectors")
        u_hat_inf = u_hat_inf / norms[:, None]

        face_center = _read_array(npz, "face_center", ["face_center", "face_centers", "centers"], required=False)
        if face_center is not None:
            face_center = np.asarray(face_center, dtype=np.float64)
            if face_center.shape != (force.shape[1], 3):
                raise FieldPodMfmcError("face_center must have shape (n_faces, 3)")

        face_normal = _read_array(npz, "face_normal", ["face_normal", "face_normals", "normals"], required=False)
        if face_normal is not None:
            face_normal = np.asarray(face_normal, dtype=np.float64)
            if face_normal.shape != (force.shape[1], 3):
                raise FieldPodMfmcError("face_normal must have shape (n_faces, 3)")

        reference_point = _read_array(npz, "reference_point", ["reference_point", "moment_reference_point"], required=False)
        if reference_point is not None:
            reference_point = np.asarray(reference_point, dtype=np.float64).reshape(-1)[:3]

        cd = _read_array(npz, "C_D", ["C_D", "cd", "scalar_cd"], required=False)
        scalar_cd = None if cd is None else np.asarray(cd, dtype=np.float64).reshape(-1)
        if scalar_cd is not None and scalar_cd.size != force.shape[0]:
            raise FieldPodMfmcError("C_D length must match force_per_area n_samples")

    return SurfaceFieldArchive(
        fidelity=fidelity.upper(),
        sample_ids=sample_ids,
        force_per_area=force,
        q_inf=q_inf,
        u_hat_inf=u_hat_inf,
        geometry=SurfaceGeometry(
            face_area=face_area,
            A_ref=A_ref,
            face_center=face_center,
            face_normal=face_normal,
            reference_point=reference_point,
        ),
        A_ref_per_sample=A_ref_per_sample,
        scalar_cd=scalar_cd,
        source_path=path,
    )


def check_topology(
    reference: SurfaceGeometry,
    other: SurfaceGeometry,
    tolerance: TopologyTolerance,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "same_number_of_faces": reference.n_faces == other.n_faces,
        "same_face_areas": False,
        "same_face_centers": None,
        "same_face_normals": None,
        "identity_mapping_allowed": False,
    }
    if reference.n_faces != other.n_faces:
        report["reason"] = f"face count differs: {reference.n_faces} vs {other.n_faces}"
        return report
    report["same_face_areas"] = bool(np.allclose(reference.face_area, other.face_area, rtol=tolerance.area_rtol, atol=0.0))
    if reference.face_center is not None and other.face_center is not None:
        report["same_face_centers"] = bool(np.allclose(reference.face_center, other.face_center, rtol=0.0, atol=tolerance.center_atol))
    if reference.face_normal is not None and other.face_normal is not None:
        report["same_face_normals"] = bool(np.allclose(reference.face_normal, other.face_normal, rtol=0.0, atol=tolerance.normal_atol))
    report["identity_mapping_allowed"] = bool(
        report["same_number_of_faces"]
        and report["same_face_areas"]
        and report["same_face_centers"] is not False
        and report["same_face_normals"] is not False
    )
    if not report["identity_mapping_allowed"]:
        report["reason"] = "DSMC/TPMC topology mismatch; no non-identity mapping is implemented"
    return report


def build_snapshots(archive: SurfaceFieldArchive, snapshot_type: str = "full_traction") -> SnapshotMatrix:
    """Build dimensionless PICLAS surface-load snapshots.

    For ``full_traction`` the returned row for sample n is
    ``sqrt(A_j / A_ref[n]) * t_{j,c}^{(n)} / q_inf^{(n)}`` for all faces j and
    components c in x, y, z. If no per-sample reference area is present in the
    archive, ``A_ref[n]`` is the scalar ``A_ref``. The Euclidean norm is
    therefore the discrete area-weighted surface-integral norm of
    dimensionless traction.

    For ``drag_contribution`` the returned row is
    ``c_D,j = -(t_j dot u_hat_inf) A_j / (q_inf A_ref)`` and summing over
    faces reconstructs the scalar drag coefficient when the archived scalar
    uses the same sign and reference-area convention.
    """
    geom = archive.geometry
    force = np.asarray(archive.force_per_area, dtype=np.float64)
    q = archive.q_inf[:, None, None]
    A_ref_n = archive.A_ref_per_sample.reshape(archive.n_samples, 1, 1)
    if snapshot_type == "full_traction":
        weights = np.sqrt(geom.face_area.reshape(1, geom.n_faces, 1) / A_ref_n)
        values = (force / q) * weights
        flat = values.reshape(archive.n_samples, 3 * geom.n_faces)
        names = [f"face_{j}_{comp}" for j in range(geom.n_faces) for comp in ("tx", "ty", "tz")]
    elif snapshot_type == "drag_contribution":
        dot = np.einsum("nfc,nc->nf", force, archive.u_hat_inf)
        flat = -dot * geom.face_area.reshape(1, geom.n_faces) / (
            archive.q_inf[:, None] * archive.A_ref_per_sample[:, None]
        )
        names = [f"face_{j}_cD" for j in range(geom.n_faces)]
    elif snapshot_type in {"moment_contribution", "force_components_only"}:
        raise NotImplementedError(f"snapshot_type={snapshot_type!r} is reserved for a future implementation")
    else:
        raise FieldPodMfmcError(f"Unsupported snapshot_type: {snapshot_type}")
    return SnapshotMatrix(
        values=np.asarray(flat, dtype=np.float64),
        sample_ids=archive.sample_ids.copy(),
        snapshot_type=snapshot_type,
        fidelity=archive.fidelity,
        component_names=names,
        metadata={
            "n_samples": archive.n_samples,
            "n_faces": geom.n_faces,
            "A_ref": geom.A_ref,
            "A_ref_per_sample_min": float(np.min(archive.A_ref_per_sample)),
            "A_ref_per_sample_max": float(np.max(archive.A_ref_per_sample)),
            "A_ref_per_sample_varies": bool(not np.allclose(archive.A_ref_per_sample, geom.A_ref)),
            "area_weighting": "sqrt_area_over_Aref" if snapshot_type == "full_traction" else "area_over_Aref",
            "normalize_by_qinf": True,
            "source_path": archive.source_path,
        },
    )


def reconstruct_cd_from_full_traction(snapshot: SnapshotMatrix, archive: SurfaceFieldArchive) -> np.ndarray:
    if snapshot.snapshot_type != "full_traction":
        raise FieldPodMfmcError("C_D reconstruction from full_traction requires a full_traction snapshot")
    n_faces = archive.n_faces
    z = snapshot.values.reshape(snapshot.values.shape[0], n_faces, 3)
    A_ref_n = archive.A_ref_per_sample.reshape(snapshot.values.shape[0], 1, 1)
    weights = np.sqrt(archive.geometry.face_area.reshape(1, n_faces, 1) / A_ref_n)
    dimless_traction = z / weights
    return -np.einsum("nfc,nc,f->n", dimless_traction, archive.u_hat_inf, archive.geometry.face_area) / archive.A_ref_per_sample


def _align_cd_to_archive_convention(actual: np.ndarray, reconstructed: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    actual_arr = np.asarray(actual, dtype=float).reshape(-1)
    reconstructed_arr = np.asarray(reconstructed, dtype=float).reshape(-1)
    if actual_arr.shape != reconstructed_arr.shape:
        raise FieldPodMfmcError("C_D arrays must have matching shapes for convention alignment")
    candidates = {
        "signed": reconstructed_arr,
        "negated": -reconstructed_arr,
        "absolute": np.abs(reconstructed_arr),
    }
    errors = {
        name: float(np.max(np.abs(values - actual_arr))) if actual_arr.size else float("nan")
        for name, values in candidates.items()
    }
    finite = {name: value for name, value in errors.items() if np.isfinite(value)}
    convention = min(finite, key=finite.get) if finite else "signed"
    return candidates[convention], {
        "cd_reconstruction_convention": convention,
        "cd_reconstruction_max_abs_error": errors[convention],
        "cd_reconstruction_signed_max_abs_error": errors["signed"],
        "cd_reconstruction_negated_max_abs_error": errors["negated"],
        "cd_reconstruction_absolute_max_abs_error": errors["absolute"],
    }


def coupled_indices(hf_ids: Sequence[str], lf_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lf_lookup = {str(sample_id): idx for idx, sample_id in enumerate(lf_ids)}
    h_idx: List[int] = []
    l_idx: List[int] = []
    ids: List[str] = []
    for idx, sample_id in enumerate(hf_ids):
        key = str(sample_id)
        if key in lf_lookup:
            h_idx.append(idx)
            l_idx.append(lf_lookup[key])
            ids.append(key)
    return np.asarray(h_idx, dtype=int), np.asarray(l_idx, dtype=int), np.asarray(ids, dtype=str)


def coupled_indices_many(
    reference_ids: Sequence[str],
    control_ids: Dict[str, Sequence[str]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    lookups = {
        name: {str(sample_id): idx for idx, sample_id in enumerate(ids)}
        for name, ids in control_ids.items()
    }
    reference_idx: List[int] = []
    control_idx: Dict[str, List[int]] = {name: [] for name in control_ids}
    common_ids: List[str] = []
    for idx, sample_id in enumerate(reference_ids):
        key = str(sample_id)
        if all(key in lookup for lookup in lookups.values()):
            reference_idx.append(idx)
            common_ids.append(key)
            for name, lookup in lookups.items():
                control_idx[name].append(lookup[key])
    return (
        np.asarray(reference_idx, dtype=int),
        {name: np.asarray(values, dtype=int) for name, values in control_idx.items()},
        np.asarray(common_ids, dtype=str),
    )


def build_tpmc_basis(
    snapshots: SnapshotMatrix,
    basis_size: int,
    output_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_ref = np.mean(snapshots.values, axis=0)
    centered = snapshots.values - z_ref
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    s = min(int(basis_size), vt.shape[0])
    psi = vt[:s, :].T.copy()
    gram = psi.T @ psi
    if not np.allclose(gram, np.eye(s), atol=1.0e-10, rtol=1.0e-10):
        raise FieldPodMfmcError("TPMC POD basis failed orthonormality check")
    variance = singular_values**2
    total = float(np.sum(variance))
    evr = variance / total if total > 0.0 else np.zeros_like(variance)
    cum = np.cumsum(evr)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "z_ref.npz", z_ref=z_ref)
        np.savez_compressed(output_dir / "Psi_s.npz", Psi_s=psi)
        _write_csv(
            output_dir / "singular_values.csv",
            [
                {
                    "mode": i + 1,
                    "singular_value": singular_values[i],
                    "explained_variance_ratio": evr[i],
                    "cumulative_explained_variance_ratio": cum[i],
                }
                for i in range(singular_values.size)
            ],
            ["mode", "singular_value", "explained_variance_ratio", "cumulative_explained_variance_ratio"],
        )
        _write_json(
            output_dir / "basis_metadata.json",
            {
                "basis_source": snapshots.fidelity,
                "snapshot_type": snapshots.snapshot_type,
                "basis_size_s": s,
                "ambient_dimension": int(snapshots.values.shape[1]),
                "n_basis_snapshots": int(snapshots.values.shape[0]),
                "orthonormality_error_fro": float(np.linalg.norm(gram - np.eye(s), ord="fro")),
            },
        )
        _plot_vector(output_dir / "tpmc_basis_spectrum.png", singular_values, "TPMC POD singular values", "mode", "singular value")
        _plot_vector(output_dir / "tpmc_basis_cumulative_evr.png", cum, "TPMC POD cumulative EVR", "mode", "cumulative EVR")
    return z_ref, psi, singular_values, evr, cum


def projection_diagnostics(
    hf_snapshots: SnapshotMatrix,
    z_ref: np.ndarray,
    psi: np.ndarray,
    sample_ids: Sequence[str],
    thresholds: ProjectionWarningThreshold,
    output_dir: Optional[Path] = None,
    cd_actual: Optional[np.ndarray] = None,
    cd_projected: Optional[np.ndarray] = None,
    cd_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    centered = hf_snapshots.values - z_ref
    projected = (centered @ psi) @ psi.T
    denom = np.linalg.norm(centered, axis=1)
    numer = np.linalg.norm(centered - projected, axis=1)
    residual = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)
    summary = {
        "mean_projection_residual": float(np.mean(residual)),
        "median_projection_residual": float(np.median(residual)),
        "max_projection_residual": float(np.max(residual)) if residual.size else float("nan"),
        "p90_projection_residual": float(np.percentile(residual, 90)) if residual.size else float("nan"),
        "p95_projection_residual": float(np.percentile(residual, 95)) if residual.size else float("nan"),
        "warning_mean_threshold": thresholds.mean,
        "warning_max_threshold": thresholds.max,
        "mean_threshold_exceeded": bool(residual.size and float(np.mean(residual)) > thresholds.mean),
        "max_threshold_exceeded": bool(residual.size and float(np.max(residual)) > thresholds.max),
    }
    if cd_actual is not None and cd_projected is not None:
        delta = np.asarray(cd_projected, dtype=float) - np.asarray(cd_actual, dtype=float)
        summary["cd_projection_rmse"] = float(np.sqrt(np.mean(delta**2)))
        actual_var = float(np.var(cd_actual, ddof=1)) if len(cd_actual) > 1 else float("nan")
        proj_var = float(np.var(cd_projected, ddof=1)) if len(cd_projected) > 1 else float("nan")
        summary["cd_variance_projection_loss"] = (
            float((actual_var - proj_var) / actual_var) if np.isfinite(actual_var) and abs(actual_var) > 1.0e-14 else float("nan")
        )
    if cd_metadata:
        summary.update(cd_metadata)
    if output_dir is not None:
        rows = [
            {
                "sample_id": sample_ids[i],
                "projection_residual": residual[i],
                "C_D": "" if cd_actual is None else cd_actual[i],
                "C_D_projected": "" if cd_projected is None else cd_projected[i],
            }
            for i in range(residual.size)
        ]
        _write_csv(output_dir / "projection_residuals.csv", rows, ["sample_id", "projection_residual", "C_D", "C_D_projected"])
        _write_json(output_dir / "projection_summary.json", summary)
        _plot_vector(output_dir / "hf_projection_residuals.png", residual, "HF projection residuals", "coupled sample", "relative residual")
        _plot_hist(output_dir / "hf_projection_residual_histogram.png", residual, "HF projection residual histogram", "relative residual")
    return summary


def project_coefficients(snapshots: SnapshotMatrix, z_ref: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return (snapshots.values - z_ref) @ psi


def moment_matrix(coefficients: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coefficients.ndim != 2 or coefficients.shape[0] == 0:
        raise FieldPodMfmcError("Coefficient array must have shape (n_samples, s) with n_samples > 0")
    mu = np.mean(coefficients, axis=0)
    M = coefficients.T @ coefficients / float(coefficients.shape[0])
    M = 0.5 * (M + M.T)
    Sigma = M - np.outer(mu, mu)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, M, Sigma


def _response_for_shared_beta(b_h: np.ndarray, b_l: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "coefficient_norm":
        return np.linalg.norm(b_h, axis=1), np.linalg.norm(b_l, axis=1)
    if mode == "first_coefficient":
        return b_h[:, 0], b_l[:, 0]
    raise FieldPodMfmcError(f"Unsupported shared_weight_response: {mode}")


def _shared_beta(b_h: np.ndarray, b_l: np.ndarray, mode: str) -> float:
    y_h, y_l = _response_for_shared_beta(b_h, b_l, mode)
    if y_h.size < 2 or y_l.size < 2:
        return 0.0
    var_l = float(np.var(y_l, ddof=1))
    if not np.isfinite(var_l) or abs(var_l) < 1.0e-14:
        return 0.0
    cov = float(np.cov(y_h, y_l, ddof=1)[0, 1])
    return float(cov / var_l) if np.isfinite(cov) else 0.0


def _multi_shared_betas(
    b_h: np.ndarray,
    controls: Sequence[np.ndarray],
    mode: str,
) -> np.ndarray:
    if not controls:
        return np.asarray([], dtype=float)
    response_h = _response_for_shared_beta(b_h, controls[0], mode)[0]
    response_controls = np.column_stack(
        [_response_for_shared_beta(b_h, values, mode)[1] for values in controls]
    )
    if response_h.size < 2:
        return np.zeros(len(controls), dtype=float)
    centered_h = response_h - np.mean(response_h)
    centered_l = response_controls - np.mean(response_controls, axis=0)
    covariance_ll = centered_l.T @ centered_l / float(response_h.size - 1)
    covariance_lh = centered_l.T @ centered_h / float(response_h.size - 1)
    scale = max(float(np.trace(covariance_ll)), 1.0)
    ridge = np.finfo(float).eps * scale * max(1, len(controls))
    return np.linalg.pinv(covariance_ll + ridge * np.eye(len(controls))) @ covariance_lh


def multi_control_mfmc_moments(
    b_h_paired: np.ndarray,
    b_l_paired: Sequence[np.ndarray],
    b_l_full: Sequence[np.ndarray],
    control_names: Optional[Sequence[str]] = None,
    shared_weight_response: str = "coefficient_norm",
    psd_correction: str = "none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(b_l_paired) != len(b_l_full) or not b_l_paired:
        raise FieldPodMfmcError("At least one paired/full control-variate pair is required")
    names = list(control_names or [f"LF{i + 1}" for i in range(len(b_l_paired))])
    if len(names) != len(b_l_paired):
        raise FieldPodMfmcError("control_names length must match control arrays")
    n_paired = b_h_paired.shape[0]
    if any(values.shape[0] != n_paired for values in b_l_paired):
        raise FieldPodMfmcError("All control variates must share the paired HF sample set")
    betas = _multi_shared_betas(b_h_paired, b_l_paired, shared_weight_response)
    mu, M, _ = moment_matrix(b_h_paired)
    for beta, paired, full in zip(betas, b_l_paired, b_l_full):
        mu_paired, M_paired, _ = moment_matrix(paired)
        mu_full, M_full, _ = moment_matrix(full)
        mu -= float(beta) * (mu_paired - mu_full)
        M -= float(beta) * (M_paired - M_full)
    M = 0.5 * (M + M.T)
    Sigma = 0.5 * (M - np.outer(mu, mu) + (M - np.outer(mu, mu)).T)
    raw_eigs = np.linalg.eigvalsh(Sigma)
    diag: Dict[str, Any] = {
        "shared_betas": {name: float(beta) for name, beta in zip(names, betas)},
        "shared_weight_response": shared_weight_response,
        "min_eigenvalue_before_correction": float(np.min(raw_eigs)) if raw_eigs.size else float("nan"),
        "negative_eigenvalue_count_before_correction": int(np.sum(raw_eigs < -1.0e-12)),
        "psd_correction": psd_correction,
    }
    if psd_correction == "eigenvalue_clip":
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        Sigma = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
        Sigma = 0.5 * (Sigma + Sigma.T)
    elif psd_correction != "none":
        raise FieldPodMfmcError(f"Unsupported psd_correction: {psd_correction}")
    final_eigs = np.linalg.eigvalsh(Sigma)
    diag["min_eigenvalue_after_correction"] = float(np.min(final_eigs)) if final_eigs.size else float("nan")
    diag["negative_eigenvalue_count_after_correction"] = int(np.sum(final_eigs < -1.0e-12))
    return mu, M, Sigma, diag


def allocate_control_samples(
    budget: float,
    n_hf: int,
    hf_cost: float,
    control_costs: Sequence[float],
    control_pool_sizes: Sequence[int],
    importance: Sequence[float],
) -> List[int]:
    if not (len(control_costs) == len(control_pool_sizes) == len(importance)):
        raise FieldPodMfmcError("Control allocation inputs must have equal length")
    counts = [min(int(n_hf), int(size)) for size in control_pool_sizes]
    spent = float(n_hf) * float(hf_cost) + sum(count * float(cost) for count, cost in zip(counts, control_costs))
    remaining = max(0.0, float(budget) - spent)
    scores = [max(float(value), 1.0e-16) for value in importance]
    while True:
        candidates = [
            idx
            for idx, (count, size, cost) in enumerate(zip(counts, control_pool_sizes, control_costs))
            if count < int(size) and float(cost) <= remaining + 1.0e-14
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda idx: scores[idx] / (max(counts[idx], 1) * (counts[idx] + 1) * float(control_costs[idx])),
        )
        counts[best] += 1
        remaining -= float(control_costs[best])
    return counts


def shared_weight_mfmc_moments(
    b_h_paired: np.ndarray,
    b_l_paired: np.ndarray,
    b_l_full: np.ndarray,
    shared_weight_response: str = "coefficient_norm",
    psd_correction: str = "none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    beta = _shared_beta(b_h_paired, b_l_paired, shared_weight_response)
    mu_h, M_h, _ = moment_matrix(b_h_paired)
    mu_lp, M_lp, _ = moment_matrix(b_l_paired)
    mu_lf, M_lf, _ = moment_matrix(b_l_full)
    mu = mu_h - beta * (mu_lp - mu_lf)
    M = M_h - beta * (M_lp - M_lf)
    M = 0.5 * (M + M.T)
    Sigma = M - np.outer(mu, mu)
    Sigma = 0.5 * (Sigma + Sigma.T)
    raw_eigs = np.linalg.eigvalsh(Sigma)
    diag: Dict[str, Any] = {
        "shared_beta": beta,
        "shared_weight_response": shared_weight_response,
        "min_eigenvalue_before_correction": float(np.min(raw_eigs)) if raw_eigs.size else float("nan"),
        "negative_eigenvalue_count_before_correction": int(np.sum(raw_eigs < -1.0e-12)),
        "psd_correction": psd_correction,
    }
    if psd_correction == "eigenvalue_clip":
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        Sigma = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
        Sigma = 0.5 * (Sigma + Sigma.T)
    elif psd_correction != "none":
        raise FieldPodMfmcError(f"Unsupported psd_correction: {psd_correction}")
    final_eigs = np.linalg.eigvalsh(Sigma)
    diag["min_eigenvalue_after_correction"] = float(np.min(final_eigs)) if final_eigs.size else float("nan")
    diag["negative_eigenvalue_count_after_correction"] = int(np.sum(final_eigs < -1.0e-12))
    return mu, M, Sigma, diag


def pod_from_covariance(Sigma: np.ndarray, modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if modes is not None:
        eigvals = eigvals[:modes]
        eigvecs = eigvecs[:, :modes]
    total = float(np.sum(np.clip(eigvals, 0.0, None)))
    evr = np.clip(eigvals, 0.0, None) / total if total > 0.0 else np.zeros_like(eigvals)
    return eigvals, eigvecs, evr


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0))


def relative_norm_error(estimate: np.ndarray, reference: np.ndarray, ord: Any = None) -> float:
    denom = float(np.linalg.norm(reference, ord=ord))
    if denom <= 1.0e-14 or not np.isfinite(denom):
        return float("nan")
    return float(np.linalg.norm(estimate - reference, ord=ord) / denom)


def run_data_check(cfg: FieldPodMfmcConfig) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "case_name": cfg.case_name,
        "required_fields": REQUIRED_DATA_FIELDS,
        "preferred_fields": PREFERRED_DATA_FIELDS,
        "archives": {},
        "supported": False,
        "missing_required": [],
    }
    missing: List[str] = []
    required_fidelities = tuple(dict.fromkeys((cfg.high_fidelity,) + cfg.resolved_control_variates))
    for fidelity in required_fidelities:
        path = cfg.fidelity_archives.get(fidelity)
        if path is None:
            info = {"path": None, "exists": False, "missing_required": REQUIRED_DATA_FIELDS}
        else:
            info = inspect_archive(path)
        report["archives"][fidelity] = info
        for field in info.get("missing_required", []):
            missing.append(f"{fidelity}:{field}")

    if not missing:
        try:
            hf = load_surface_archive(cfg.fidelity_archives[cfg.high_fidelity], cfg.high_fidelity)
            controls = {
                name: load_surface_archive(cfg.fidelity_archives[name], name)
                for name in cfg.resolved_control_variates
            }
            h_idx, _, ids = coupled_indices_many(
                hf.sample_ids,
                {name: archive.sample_ids for name, archive in controls.items()},
            )
            topology_reports = {
                name: check_topology(hf.geometry, archive.geometry, cfg.topology_tolerance)
                for name, archive in controls.items()
            }
            report["n_coupled_samples"] = int(ids.size)
            report["n_faces"] = int(hf.n_faces)
            report["available_force_components"] = ["x", "y", "z"]
            report["topology_reports"] = topology_reports
            report["topology_report"] = topology_reports.get(cfg.low_fidelity_basis_source)
            report["cd_reconstruction_possible"] = bool(hf.scalar_cd is not None)
            topology_ok = all(value.get("identity_mapping_allowed", False) for value in topology_reports.values())
            report["supported"] = bool(ids.size > 0 and topology_ok)
            if ids.size == 0:
                missing.append("coupled DSMC/control-variate sample_id intersection")
            for name, topology in topology_reports.items():
                if not topology.get("identity_mapping_allowed", False):
                    missing.append(f"identical DSMC/{name} surface topology")
            if h_idx.size and hf.scalar_cd is not None:
                drag = build_snapshots(hf, "drag_contribution")
                _, cd_info = _align_cd_to_archive_convention(hf.scalar_cd[h_idx], np.sum(drag.values[h_idx], axis=1))
                report.update({f"drag_contribution_{key}": value for key, value in cd_info.items()})
        except Exception as exc:
            report["load_error"] = str(exc)
            missing.append(str(exc))

    report["missing_required"] = missing
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_dir / "data_availability_report.json", report)
    if missing:
        md = [
            f"# Missing Field POD/MFMC Data: {cfg.case_name}",
            "",
            "The field-level POD/MFMC workflow cannot run for this case yet.",
            "",
            "Missing or invalid required items:",
            *[f"- {item}" for item in missing],
            "",
            "Required NPZ archive fields per fidelity:",
            "- force_per_area: shape (n_samples, n_faces, 3), PICLAS Total_ForcePerArea",
            "- sample_id: coupled realization identifiers",
            "- face_area: shape (n_faces,)",
            "- A_ref: scalar reference area",
            "- q_inf: scalar or shape (n_samples,)",
            "- u_hat_inf: shape (3,) or (n_samples, 3)",
            "",
            "Preferred fields: face_center, face_normal, reference_point, C_D.",
        ]
        (cfg.output_dir / f"field_pod_mfmc_missing_data_{_slug(cfg.case_name)}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_field_workflow(cfg: FieldPodMfmcConfig) -> Dict[str, Any]:
    check = run_data_check(cfg)
    if check.get("missing_required"):
        raise FieldPodMfmcError(
            f"Field POD/MFMC data check failed for {cfg.case_name}; see {cfg.output_dir / 'data_availability_report.json'}"
        )

    out = cfg.output_dir
    hf = load_surface_archive(cfg.fidelity_archives[cfg.high_fidelity], cfg.high_fidelity)
    controls = {
        name: load_surface_archive(cfg.fidelity_archives[name], name)
        for name in cfg.resolved_control_variates
    }
    topology_reports = {
        name: check_topology(hf.geometry, archive.geometry, cfg.topology_tolerance)
        for name, archive in controls.items()
    }
    _write_json(out / "topology_report.json", {"controls": topology_reports})
    invalid_topology = [name for name, report in topology_reports.items() if not report.get("identity_mapping_allowed", False)]
    if invalid_topology:
        raise FieldPodMfmcError(f"DSMC/control topology mismatch for {invalid_topology}")

    hf_snap = build_snapshots(hf, cfg.snapshot_type)
    control_snapshots = {
        name: build_snapshots(archive, cfg.snapshot_type)
        for name, archive in controls.items()
    }
    h_idx, control_idx, ids = coupled_indices_many(
        hf_snap.sample_ids,
        {name: snapshot.sample_ids for name, snapshot in control_snapshots.items()},
    )
    if ids.size < 2:
        raise FieldPodMfmcError("At least two samples coupled across DSMC and all control variates are required")

    basis_snap = control_snapshots[cfg.low_fidelity_basis_source]
    z_ref, psi, singular_values, evr, cum = build_tpmc_basis(basis_snap, cfg.basis_size_s, out)
    b_h_all = project_coefficients(hf_snap, z_ref, psi)
    b_control_all = {
        name: project_coefficients(snapshot, z_ref, psi)
        for name, snapshot in control_snapshots.items()
    }
    b_h_c = b_h_all[h_idx]
    b_control_c = {
        name: b_control_all[name][control_idx[name]]
        for name in cfg.resolved_control_variates
    }
    np.savez_compressed(out / "b_coefficients_DSMC.npz", b=b_h_all, sample_id=hf_snap.sample_ids)
    for name, values in b_control_all.items():
        np.savez_compressed(
            out / f"b_coefficients_{name}.npz",
            b=values,
            sample_id=control_snapshots[name].sample_ids,
        )

    cd_actual = None
    cd_projected = None
    cd_projection_metadata: Dict[str, Any] = {}
    cd_convention = "signed"
    if hf.scalar_cd is not None and cfg.snapshot_type == "full_traction":
        cd_actual = hf.scalar_cd[h_idx]
        centered = hf_snap.values[h_idx] - z_ref
        projected_snap = SnapshotMatrix(
            values=z_ref + (centered @ psi) @ psi.T,
            sample_ids=ids,
            snapshot_type="full_traction",
            fidelity=hf.fidelity,
            component_names=hf_snap.component_names,
            metadata=hf_snap.metadata,
        )
        tmp_archive = SurfaceFieldArchive(
            fidelity=hf.fidelity,
            sample_ids=ids,
            force_per_area=hf.force_per_area[h_idx],
            q_inf=hf.q_inf[h_idx],
            u_hat_inf=hf.u_hat_inf[h_idx],
            geometry=hf.geometry,
            A_ref_per_sample=hf.A_ref_per_sample[h_idx],
            scalar_cd=cd_actual,
            source_path=hf.source_path,
        )
        cd_reconstructed = reconstruct_cd_from_full_traction(projected_snap, tmp_archive)
        cd_projected, cd_projection_metadata = _align_cd_to_archive_convention(cd_actual, cd_reconstructed)
        cd_convention = str(cd_projection_metadata.get("cd_reconstruction_convention", "signed"))

    proj_summary = projection_diagnostics(
        SnapshotMatrix(hf_snap.values[h_idx], ids, hf_snap.snapshot_type, hf_snap.fidelity, hf_snap.component_names, hf_snap.metadata),
        z_ref,
        psi,
        ids,
        cfg.projection_residual_warning_threshold,
        out,
        cd_actual=cd_actual,
        cd_projected=cd_projected,
        cd_metadata=cd_projection_metadata,
    )

    mu_ref, M_ref, Sigma_ref = moment_matrix(b_h_all)
    eig_ref, W_ref, evr_ref = pod_from_covariance(Sigma_ref)
    np.savez_compressed(out / "mu_b_ref.npz", mu_b_ref=mu_ref, M_b_ref=M_ref)
    np.savez_compressed(out / "Sigma_b_ref.npz", Sigma_b_ref=Sigma_ref)
    np.savez_compressed(out / "pod_ref.npz", eigenvalues=eig_ref, eigenvectors=W_ref, explained_variance_ratio=evr_ref)
    _write_json(
        out / "reference_metadata.json",
        {
            "case": cfg.case_name,
            "snapshot_type": cfg.snapshot_type,
            "basis_source": cfg.low_fidelity_basis_source,
            "basis_size": int(psi.shape[1]),
            "n_dsmc_samples": int(b_h_all.shape[0]),
            "random_seed": cfg.random_seed,
            "invalid_samples_removed": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation": "internal archived DSMC reference, not external physical truth",
        },
    )
    _plot_vector(out / "dsmc_reference_pod_spectrum.png", eig_ref, "Internal DSMC reference POD spectrum", "mode", "eigenvalue")

    metrics_rows: List[Dict[str, Any]] = []
    qoi_rows: List[Dict[str, Any]] = []
    mu_mfmc: List[np.ndarray] = []
    Sigma_mfmc: List[np.ndarray] = []
    mu_dsmc: List[np.ndarray] = []
    Sigma_dsmc: List[np.ndarray] = []
    psd_diags: List[Dict[str, Any]] = []
    qoi_functional_available, qoi_functional_reason = _cd_linear_functional_available(hf)
    if cd_convention == "absolute":
        qoi_functional_available = False
        qoi_functional_reason = "archived C_D matches abs(signed field functional); abs is not a linear reduced functional"
    qoi_sign = -1.0 if cd_convention == "negated" else 1.0

    rng = np.random.default_rng(cfg.random_seed)
    control_names = list(cfg.resolved_control_variates)
    control_costs = [cfg.cost_for(name) for name in control_names]
    pilot_betas = _multi_shared_betas(
        b_h_c,
        [b_control_c[name] for name in control_names],
        cfg.shared_weight_response,
    )
    control_importance = []
    for name, beta in zip(control_names, pilot_betas):
        response = _response_for_shared_beta(b_h_c, b_control_c[name], cfg.shared_weight_response)[1]
        variance = float(np.var(response, ddof=1)) if response.size > 1 else 0.0
        control_importance.append(float(beta * beta) * max(variance, 0.0))
    for budget in cfg.budgets:
        n_hf_mfmc = max(2, int(math.floor(float(budget) * cfg.mfmc_hf_fraction / cfg.hf_cost)))
        n_hf_only = max(2, int(math.floor(float(budget) / cfg.hf_cost)))
        affordable_paired = int(math.floor(float(budget) / (cfg.hf_cost + sum(control_costs))))
        if affordable_paired >= 2:
            n_hf_mfmc = min(n_hf_mfmc, affordable_paired)
        n_hf_mfmc = min(n_hf_mfmc, b_h_c.shape[0])
        n_hf_only = min(n_hf_only, b_h_all.shape[0])
        control_counts = allocate_control_samples(
            float(budget),
            n_hf_mfmc,
            cfg.hf_cost,
            control_costs,
            [b_control_all[name].shape[0] for name in control_names],
            control_importance,
        )

        for rep in range(cfg.repeats):
            h_pair_sel = rng.choice(b_h_c.shape[0], size=n_hf_mfmc, replace=False)
            control_full = []
            for name, count in zip(control_names, control_counts):
                selected = rng.choice(b_control_all[name].shape[0], size=count, replace=False)
                control_full.append(b_control_all[name][selected])
            dsmc_sel = rng.choice(b_h_all.shape[0], size=n_hf_only, replace=False)
            mu_m, M_m, Sig_m, diag = multi_control_mfmc_moments(
                b_h_c[h_pair_sel],
                [b_control_c[name][h_pair_sel] for name in control_names],
                control_full,
                control_names=control_names,
                shared_weight_response=cfg.shared_weight_response,
                psd_correction=cfg.psd_correction,
            )
            mu_d, _, Sig_d = moment_matrix(b_h_all[dsmc_sel])
            eig_m, W_m, _ = pod_from_covariance(Sig_m, cfg.pod_modes_r)
            eig_d, W_d, _ = pod_from_covariance(Sig_d, cfg.pod_modes_r)
            r = min(cfg.pod_modes_r, W_ref.shape[1], W_m.shape[1], W_d.shape[1])
            ang_m = principal_angles(W_ref[:, :r], W_m[:, :r])
            ang_d = principal_angles(W_ref[:, :r], W_d[:, :r])
            err_mu_m = relative_norm_error(mu_m, mu_ref)
            err_mu_d = relative_norm_error(mu_d, mu_ref)
            err_cov_m = relative_norm_error(Sig_m, Sigma_ref, ord="fro")
            err_cov_d = relative_norm_error(Sig_d, Sigma_ref, ord="fro")
            metric_row = {
                    "budget": budget,
                    "repeat": rep,
                    "n_hf_mfmc": n_hf_mfmc,
                    "n_lf_mfmc": control_counts[0],
                    "n_hf_dsmc_only": n_hf_only,
                    "err_mu_mfmc": err_mu_m,
                    "err_mu_dsmc_only": err_mu_d,
                    "err_cov_mfmc": err_cov_m,
                    "err_cov_dsmc_only": err_cov_d,
                    "gain_mu": _gain(err_mu_d, err_mu_m),
                    "gain_cov": _gain(err_cov_d, err_cov_m),
                    "principal_angle_max_mfmc_rad": float(np.max(ang_m)) if ang_m.size else float("nan"),
                    "principal_angle_max_dsmc_only_rad": float(np.max(ang_d)) if ang_d.size else float("nan"),
                    "lambda1_mfmc": eig_m[0] if eig_m.size else float("nan"),
                    "lambda1_dsmc_only": eig_d[0] if eig_d.size else float("nan"),
                    "lambda1_ref": eig_ref[0] if eig_ref.size else float("nan"),
                    "shared_beta": diag["shared_betas"][control_names[0]],
                }
            for name, count in zip(control_names, control_counts):
                metric_row[f"n_lf_{name}"] = count
                metric_row[f"shared_beta_{name}"] = diag["shared_betas"][name]
            metrics_rows.append(metric_row)
            if cd_actual is not None and cfg.snapshot_type == "full_traction" and qoi_functional_available:
                l_b = qoi_sign * _cd_linear_functional_in_basis(hf, psi)
                q_base = float(qoi_sign * _cd_linear_functional_in_basis_base(hf, z_ref))
                q_ref_mean = float(np.mean(hf.scalar_cd))
                q_ref_var = float(np.var(hf.scalar_cd, ddof=1))
                q_m_mean = float(q_base + l_b @ mu_m)
                q_d_mean = float(q_base + l_b @ mu_d)
                q_m_var = float(l_b @ Sig_m @ l_b)
                q_d_var = float(l_b @ Sig_d @ l_b)
                qoi_rows.append(
                    {
                        "budget": budget,
                        "repeat": rep,
                        "qoi": "C_D",
                        "mean_ref": q_ref_mean,
                        "mean_mfmc": q_m_mean,
                        "mean_dsmc_only": q_d_mean,
                        "var_ref": q_ref_var,
                        "var_mfmc": q_m_var,
                        "var_dsmc_only": q_d_var,
                        "err_Q_mean_mfmc": abs(q_m_mean - q_ref_mean) / max(abs(q_ref_mean), 1.0e-14),
                        "err_Q_mean_dsmc_only": abs(q_d_mean - q_ref_mean) / max(abs(q_ref_mean), 1.0e-14),
                        "err_Q_var_mfmc": abs(q_m_var - q_ref_var) / max(abs(q_ref_var), 1.0e-14),
                        "err_Q_var_dsmc_only": abs(q_d_var - q_ref_var) / max(abs(q_ref_var), 1.0e-14),
                    }
                )
            mu_mfmc.append(mu_m)
            Sigma_mfmc.append(Sig_m)
            mu_dsmc.append(mu_d)
            Sigma_dsmc.append(Sig_d)
            diag.update({"budget": budget, "repeat": rep})
            psd_diags.append(diag)

    _write_csv(out / "comparison_metrics.csv", metrics_rows, list(metrics_rows[0].keys()) if metrics_rows else [])
    if qoi_rows:
        _write_csv(out / "qoi_metrics.csv", qoi_rows, list(qoi_rows[0].keys()))
    elif hf.scalar_cd is not None and cfg.snapshot_type == "full_traction":
        _write_csv(
            out / "qoi_metrics.csv",
            [{"qoi": "C_D", "available": False, "reason": qoi_functional_reason}],
            ["qoi", "available", "reason"],
        )
        _write_json(
            out / "qoi_metrics_unavailable.json",
            {
                "qoi": "C_D",
                "reason": qoi_functional_reason,
                "cd_reconstruction_convention": cd_convention,
                "A_ref_per_sample_min": float(np.min(hf.A_ref_per_sample)),
                "A_ref_per_sample_max": float(np.max(hf.A_ref_per_sample)),
                "u_hat_inf_varies": bool(not np.allclose(hf.u_hat_inf, hf.u_hat_inf[0], rtol=0.0, atol=1.0e-12)),
            },
        )
    _write_json(out / "covariance_psd_diagnostics.json", {"diagnostics": psd_diags})
    np.savez_compressed(out / "mu_b_mfmc_by_budget.npz", values=np.asarray(mu_mfmc), budgets=np.asarray(cfg.budgets))
    np.savez_compressed(out / "Sigma_b_mfmc_by_budget.npz", values=np.asarray(Sigma_mfmc), budgets=np.asarray(cfg.budgets))
    np.savez_compressed(out / "mu_b_dsmc_only_by_budget.npz", values=np.asarray(mu_dsmc), budgets=np.asarray(cfg.budgets))
    np.savez_compressed(out / "Sigma_b_dsmc_only_by_budget.npz", values=np.asarray(Sigma_dsmc), budgets=np.asarray(cfg.budgets))
    if mu_mfmc:
        np.savez_compressed(out / "mu_b_mfmc.npz", mu_b_mfmc=mu_mfmc[-1])
        np.savez_compressed(out / "M_b_mfmc.npz", M_b_mfmc=M_m)
        np.savez_compressed(out / "Sigma_b_mfmc.npz", Sigma_b_mfmc=Sigma_mfmc[-1])
    _plot_metric(out / "covariance_error_vs_budget.png", metrics_rows, "err_cov", "Covariance error vs budget")
    _plot_metric(out / "mean_error_vs_budget.png", metrics_rows, "err_mu", "Mean error vs budget")
    _plot_angles(out / "principal_angles_vs_budget.png", metrics_rows)
    if qoi_rows:
        _plot_qoi(out / "qoi_mean_error_vs_budget.png", qoi_rows, "mean")
        _plot_qoi(out / "qoi_variance_error_vs_budget.png", qoi_rows, "var")

    summary = {
        "case_name": cfg.case_name,
        "output_dir": out,
        "n_coupled_samples": int(ids.size),
        "n_hf_samples": int(hf.n_samples),
        "n_lf_samples": int(controls[cfg.low_fidelity_basis_source].n_samples),
        "control_variates": control_names,
        "n_control_samples": {name: int(controls[name].n_samples) for name in control_names},
        "n_faces": int(hf.n_faces),
        "basis_size_s": int(psi.shape[1]),
        "projection_summary": proj_summary,
        "comparison_metrics_csv": out / "comparison_metrics.csv",
    }
    _write_json(out / "summary.json", summary)
    return summary


def _gain(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 1.0e-14:
        return float("nan")
    return float(num / den)


def _cd_linear_functional_available(archive: SurfaceFieldArchive) -> Tuple[bool, str]:
    if not np.allclose(archive.A_ref_per_sample, archive.geometry.A_ref, rtol=1.0e-12, atol=1.0e-14):
        return (
            False,
            "A_ref_per_sample varies; C_D is sample-dependent and is not represented by one fixed linear reduced functional",
        )
    if not np.allclose(archive.u_hat_inf, archive.u_hat_inf[0], rtol=0.0, atol=1.0e-12):
        return (
            False,
            "u_hat_inf varies; C_D is sample-dependent and is not represented by one fixed linear reduced functional",
        )
    return True, "C_D is represented by one fixed linear reduced functional"


def _cd_linear_functional_in_basis(archive: SurfaceFieldArchive, psi: np.ndarray) -> np.ndarray:
    n_faces = archive.n_faces
    weights = np.sqrt(archive.geometry.face_area / archive.geometry.A_ref)
    # z contains sqrt(A/Aref) * t/q, so C_D = sum_j -u dot z_j * sqrt(A/Aref).
    l_z = np.zeros(3 * n_faces, dtype=float)
    u = np.mean(archive.u_hat_inf, axis=0)
    for j in range(n_faces):
        l_z[3 * j : 3 * j + 3] = -u * weights[j]
    return psi.T @ l_z


def _cd_linear_functional_in_basis_base(archive: SurfaceFieldArchive, z_ref: np.ndarray) -> float:
    n_faces = archive.n_faces
    weights = np.sqrt(archive.geometry.face_area / archive.geometry.A_ref)
    u = np.mean(archive.u_hat_inf, axis=0)
    l_z = np.zeros(3 * n_faces, dtype=float)
    for j in range(n_faces):
        l_z[3 * j : 3 * j + 3] = -u * weights[j]
    return float(l_z @ z_ref)


def _plot_vector(path: Path, values: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(values, bins=min(20, max(5, int(np.sqrt(values.size)))), color="#4c78a8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mean_by_budget(rows: List[Dict[str, Any]], key: str, estimator: str) -> Tuple[np.ndarray, np.ndarray]:
    budgets = sorted({float(row["budget"]) for row in rows})
    vals = []
    col = f"{key}_{estimator}"
    for budget in budgets:
        arr = np.asarray([float(row[col]) for row in rows if float(row["budget"]) == budget], dtype=float)
        vals.append(float(np.nanmean(arr)))
    return np.asarray(budgets), np.asarray(vals)


def _plot_metric(path: Path, rows: List[Dict[str, Any]], prefix: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if not rows:
        return
    x_m, y_m = _mean_by_budget(rows, prefix, "mfmc")
    x_d, y_d = _mean_by_budget(rows, prefix, "dsmc_only")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(x_d, y_d, marker="o", label="DSMC-only")
    ax.plot(x_m, y_m, marker="s", label="MFMC")
    ax.set_title(title)
    ax.set_xlabel("HF-equivalent budget")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_angles(path: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    if not rows:
        return
    budgets = sorted({float(row["budget"]) for row in rows})
    mfmc = [np.nanmean([float(row["principal_angle_max_mfmc_rad"]) for row in rows if float(row["budget"]) == b]) for b in budgets]
    dsmc = [np.nanmean([float(row["principal_angle_max_dsmc_only_rad"]) for row in rows if float(row["budget"]) == b]) for b in budgets]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(budgets, dsmc, marker="o", label="DSMC-only")
    ax.plot(budgets, mfmc, marker="s", label="MFMC")
    ax.set_title("POD subspace principal angles vs budget")
    ax.set_xlabel("HF-equivalent budget")
    ax.set_ylabel("max principal angle [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_qoi(path: Path, rows: List[Dict[str, Any]], quantity: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    key_m = f"err_Q_{quantity}_mfmc"
    key_d = f"err_Q_{quantity}_dsmc_only"
    budgets = sorted({float(row["budget"]) for row in rows})
    mfmc = [np.nanmean([float(row[key_m]) for row in rows if float(row["budget"]) == b]) for b in budgets]
    dsmc = [np.nanmean([float(row[key_d]) for row in rows if float(row["budget"]) == b]) for b in budgets]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(budgets, dsmc, marker="o", label="DSMC-only")
    ax.plot(budgets, mfmc, marker="s", label="MFMC")
    ax.set_title(f"C_D {quantity} error vs budget")
    ax.set_xlabel("HF-equivalent budget")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_check_from_path(config_path: str) -> Dict[str, Any]:
    cfg = load_field_config(config_path)
    report = run_data_check(cfg)
    if report.get("missing_required"):
        raise FieldPodMfmcError(
            f"Missing required field data for {cfg.case_name}; see {cfg.output_dir / 'data_availability_report.json'}"
        )
    return report


def run_workflow_from_path(config_path: str) -> Dict[str, Any]:
    return run_field_workflow(load_field_config(config_path))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Field POD/MFMC demonstrator")
    parser.add_argument("command", choices=["check-field-data", "run-field-pod-mfmc"])
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "check-field-data":
        print(json.dumps(run_check_from_path(args.config), indent=2, default=str))
        return 0
    print(json.dumps(run_workflow_from_path(args.config), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
