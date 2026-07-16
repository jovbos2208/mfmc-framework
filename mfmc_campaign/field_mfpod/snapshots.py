from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..field_pod_mfmc import load_surface_archive
from .models import MFPODError, SurfaceGeometry, SurfaceSnapshotBatch, jsonable


class SurfaceSnapshotAdapter(ABC):
    @abstractmethod
    def load_geometry(self, case: str, fidelity: str) -> SurfaceGeometry: ...

    @abstractmethod
    def load_snapshot(self, case: str, sample_id: str, fidelity: str) -> np.ndarray: ...

    @abstractmethod
    def validate_topology(self, reference_geometry: SurfaceGeometry, candidate_geometry: SurfaceGeometry) -> dict[str, Any]: ...

    def map_to_reference_surface(self, snapshot: np.ndarray, mapping: Optional[Any] = None) -> np.ndarray:
        if mapping is not None:
            raise NotImplementedError("Non-identity surface mapping requires a validated adapter")
        return np.asarray(snapshot)


def _scalar_string(npz: np.lib.npyio.NpzFile, name: str, default: str) -> str:
    if name not in npz.files:
        return default
    return str(np.asarray(npz[name]).reshape(-1)[0])


def _component_order(npz: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    if "component_order" not in npz.files:
        return ("x", "y", "z")
    values = np.asarray(npz["component_order"]).reshape(-1)
    if values.size == 1:
        return tuple(part.strip() for part in str(values[0]).split(","))
    return tuple(str(value) for value in values)


class PICLASIdentitySurfaceAdapter(SurfaceSnapshotAdapter):
    """NPZ adapter that permits only identical, ordered PICLAS surfaces."""

    def __init__(self, archives: dict[str, str | Path], tolerances: Optional[dict[str, float]] = None):
        self.archives = {k.upper(): Path(v) for k, v in archives.items()}
        self.tolerances = tolerances or {}
        self._loaded: dict[str, Any] = {}

    def _archive(self, fidelity: str):
        key = fidelity.upper()
        if key not in self._loaded:
            self._loaded[key] = load_surface_archive(self.archives[key], key)
        return self._loaded[key]

    def load_geometry(self, case: str, fidelity: str) -> SurfaceGeometry:
        archive = self._archive(fidelity)
        with np.load(self.archives[fidelity.upper()], allow_pickle=False) as npz:
            geometry_id = _scalar_string(npz, "geometry_id", case)
            coordinate_frame = _scalar_string(npz, "coordinate_frame", "body_fixed")
            component_order = _component_order(npz)
        g = archive.geometry
        return SurfaceGeometry(
            face_area=g.face_area.copy(), A_ref=g.A_ref, geometry_id=geometry_id,
            coordinate_frame=coordinate_frame, component_order=component_order,
            face_center=None if g.face_center is None else g.face_center.copy(),
            face_normal=None if g.face_normal is None else g.face_normal.copy(),
            reference_point=None if g.reference_point is None else g.reference_point.copy(),
        )

    def load_snapshot(self, case: str, sample_id: str, fidelity: str) -> np.ndarray:
        archive = self._archive(fidelity)
        matches = np.flatnonzero(archive.sample_ids == str(sample_id))
        if matches.size != 1:
            raise MFPODError(f"Expected one {fidelity} sample_id={sample_id!r}; found {matches.size}")
        return np.asarray(archive.force_per_area[matches[0]], dtype=np.float64)

    def validate_topology(self, reference_geometry: SurfaceGeometry, candidate_geometry: SurfaceGeometry) -> dict[str, Any]:
        return validate_surface_topology(reference_geometry, candidate_geometry, self.tolerances)

    def load_batch(self, case: str, fidelity: str, snapshot_type: str = "full_traction") -> SurfaceSnapshotBatch:
        archive = self._archive(fidelity)
        geometry = self.load_geometry(case, fidelity)
        values = build_full_traction_snapshots(archive.force_per_area, geometry.face_area, archive.q_inf, archive.A_ref_per_sample)
        if snapshot_type == "drag_contribution":
            values = build_drag_contribution_snapshots(
                archive.force_per_area, geometry.face_area, archive.q_inf,
                archive.A_ref_per_sample, archive.u_hat_inf,
            )
        elif snapshot_type != "full_traction":
            raise MFPODError(f"Unsupported snapshot_type={snapshot_type!r}")
        return SurfaceSnapshotBatch(
            values=values, sample_ids=archive.sample_ids.copy(), fidelity=fidelity.upper(),
            snapshot_type=snapshot_type, geometry=geometry, q_inf=archive.q_inf.copy(),
            A_ref_per_sample=archive.A_ref_per_sample.copy(), u_hat_inf=archive.u_hat_inf.copy(),
            metadata={"normalization": "traction/q_inf", "area_weighting": "sqrt_area_over_Aref" if snapshot_type == "full_traction" else "area_over_Aref", "coordinate_frame": geometry.coordinate_frame},
        )


def validate_surface_topology(reference: SurfaceGeometry, candidate: SurfaceGeometry, tolerances: Optional[dict[str, float]] = None) -> dict[str, Any]:
    tol = tolerances or {}
    report = {
        "same_face_count": reference.n_faces == candidate.n_faces,
        "same_geometry_id": reference.geometry_id == candidate.geometry_id,
        "same_coordinate_frame": reference.coordinate_frame == candidate.coordinate_frame,
        "same_component_order": reference.component_order == candidate.component_order,
        "same_geometry_scale": False,
        "same_face_areas": False,
        "same_face_centers": None,
        "same_face_normals": None,
        "explicit_mapping": False,
    }
    if report["same_face_count"]:
        report["same_face_areas"] = bool(np.allclose(reference.face_area, candidate.face_area, rtol=tol.get("area_rtol", 1e-10), atol=0.0))
        report["same_geometry_scale"] = bool(np.isclose(reference.A_ref, candidate.A_ref, rtol=tol.get("area_rtol", 1e-10), atol=0.0))
        if reference.face_center is not None and candidate.face_center is not None:
            report["same_face_centers"] = bool(np.allclose(reference.face_center, candidate.face_center, rtol=0.0, atol=tol.get("center_atol", 1e-12)))
        if reference.face_normal is not None and candidate.face_normal is not None:
            report["same_face_normals"] = bool(np.allclose(reference.face_normal, candidate.face_normal, rtol=0.0, atol=tol.get("normal_atol", 1e-12)))
    report["identity_mapping_allowed"] = bool(
        report["same_face_count"] and report["same_geometry_id"] and report["same_coordinate_frame"]
        and report["same_component_order"] and report["same_geometry_scale"] and report["same_face_areas"]
        and report["same_face_centers"] is not False and report["same_face_normals"] is not False
    )
    if not report["identity_mapping_allowed"]:
        report["reason"] = "DSMC and TPMC are not the same ordered PICLAS surface Hilbert space"
    return report


def build_full_traction_snapshots(force_per_area: np.ndarray, face_area: np.ndarray, q_inf: np.ndarray, A_ref: np.ndarray | float) -> np.ndarray:
    force = np.asarray(force_per_area, dtype=np.float64)
    area = np.asarray(face_area, dtype=np.float64).reshape(-1)
    q = np.asarray(q_inf, dtype=np.float64).reshape(-1)
    aref = np.asarray(A_ref, dtype=np.float64).reshape(-1)
    if aref.size == 1:
        aref = np.full(force.shape[0], aref[0])
    if force.ndim != 3 or force.shape[1:] != (area.size, 3) or q.size != force.shape[0] or aref.size != force.shape[0]:
        raise MFPODError("Incompatible surface snapshot shapes")
    if np.any(area <= 0) or np.any(q <= 0) or np.any(aref <= 0):
        raise MFPODError("face areas, q_inf, and A_ref must be positive")
    weighted = force / q[:, None, None] * np.sqrt(area[None, :, None] / aref[:, None, None])
    return np.asarray(weighted.reshape(force.shape[0], -1), dtype=np.float64)


def build_drag_contribution_snapshots(force_per_area: np.ndarray, face_area: np.ndarray, q_inf: np.ndarray, A_ref: np.ndarray | float, u_hat_inf: np.ndarray) -> np.ndarray:
    force = np.asarray(force_per_area, dtype=np.float64)
    area = np.asarray(face_area, dtype=np.float64).reshape(-1)
    q = np.asarray(q_inf, dtype=np.float64).reshape(-1)
    aref = np.asarray(A_ref, dtype=np.float64).reshape(-1)
    if aref.size == 1:
        aref = np.full(force.shape[0], aref[0])
    u = np.asarray(u_hat_inf, dtype=np.float64)
    if u.ndim == 1:
        u = np.tile(u, (force.shape[0], 1))
    return -np.einsum("nfc,nc->nf", force, u) * area[None, :] / (q[:, None] * aref[:, None])


def drag_functional(geometry: SurfaceGeometry, u_hat_inf: np.ndarray) -> np.ndarray:
    u = np.asarray(u_hat_inf, dtype=float).reshape(3)
    return (-np.sqrt(geometry.face_area / geometry.A_ref)[:, None] * u[None, :]).reshape(-1)


def body_force_functionals(geometry: SurfaceGeometry) -> np.ndarray:
    """Columns map a full-traction snapshot to body-axis force coefficients."""
    scale = np.sqrt(geometry.face_area / geometry.A_ref)
    result = np.zeros((3 * geometry.n_faces, 3))
    for j in range(geometry.n_faces):
        result[3 * j:3 * j + 3, :] = np.eye(3) * scale[j]
    return result


def moment_functionals(geometry: SurfaceGeometry, reference_length: float) -> np.ndarray:
    if geometry.face_center is None or geometry.reference_point is None:
        raise MFPODError("Moment functionals require face centers and a moment reference point")
    if reference_length <= 0:
        raise MFPODError("reference_length must be positive")
    scale = np.sqrt(geometry.face_area / geometry.A_ref) / reference_length
    result = np.zeros((3 * geometry.n_faces, 3))
    for j, arm in enumerate(geometry.face_center - geometry.reference_point):
        cross = np.array([[0, -arm[2], arm[1]], [arm[2], 0, -arm[0]], [-arm[1], arm[0], 0]])
        result[3 * j:3 * j + 3, :] = (scale[j] * cross).T
    return result


def inspect_surface_data(case: str, archives: dict[str, str | Path], output_dir: Optional[Path] = None, tolerances: Optional[dict[str, float]] = None) -> dict[str, Any]:
    report: dict[str, Any] = {"case": case, "fidelities": {}, "missing_fields": [], "missing_preferred_fields": [], "invalid_samples": [], "duplicate_sample_ids": {}}
    required = {"force_per_area", "sample_id", "face_area", "A_ref", "q_inf", "u_hat_inf", "fidelity", "case_name", "geometry_id", "coordinate_frame", "component_order"}
    preferred = {"face_center", "face_normal", "reference_point", "C_D", "cpu_hours", "hardware"}
    batches = {}
    adapter = PICLASIdentitySurfaceAdapter(archives, tolerances)
    fidelities = ["DSMC", *sorted(str(name).upper() for name in archives if str(name).upper() != "DSMC")]
    for fidelity in fidelities:
        path = Path(archives.get(fidelity, ""))
        if not path.is_file():
            report["fidelities"][fidelity] = {"path": str(path), "exists": False, "n_snapshots": 0}
            report["missing_fields"].append(f"{fidelity}:archive")
            continue
        with np.load(path, allow_pickle=False) as npz:
            missing = sorted(required - set(npz.files))
            missing_preferred = sorted(preferred - set(npz.files))
            ids = [str(x) for x in np.asarray(npz["sample_id"]).reshape(-1)] if "sample_id" in npz else []
            dup = sorted({x for x in ids if ids.count(x) > 1})
            report["duplicate_sample_ids"][fidelity] = dup
            report["fidelities"][fidelity] = {"path": str(path), "exists": True, "n_snapshots": len(ids), "fields": sorted(npz.files), "missing": missing, "missing_preferred": missing_preferred}
            report["missing_fields"].extend(f"{fidelity}:{name}" for name in missing)
            report["missing_preferred_fields"].extend(f"{fidelity}:{name}" for name in missing_preferred)
        try:
            batches[fidelity] = adapter.load_batch(case, fidelity)
        except Exception as exc:
            report["invalid_samples"].append(f"{fidelity}: {exc}")
    if "DSMC" in batches and len(batches) == len(fidelities):
        id_sets = {name: set(batch.sample_ids.tolist()) for name, batch in batches.items()}
        paired_ids = set.intersection(*(id_sets[name] for name in fidelities))
        hf_ids = id_sets["DSMC"]
        report.update({
            "n_dsmc": len(hf_ids),
            "n_tpmc": len(id_sets.get("TPMC", set())),
            "n_sentman": len(id_sets.get("SENTMAN", set())),
            "n_paired": len(paired_ids),
            "n_additional_tpmc": len(id_sets.get("TPMC", set()) - hf_ids),
            "n_additional_sentman": len(id_sets.get("SENTMAN", set()) - hf_ids),
            "n_faces": batches["DSMC"].geometry.n_faces,
        })
        topology_by_model = {
            name: validate_surface_topology(batches["DSMC"].geometry, batches[name].geometry, tolerances)
            for name in fidelities[1:]
        }
        report["topology_by_model"] = topology_by_model
        report["topology"] = (
            topology_by_model["TPMC"]
            if list(topology_by_model) == ["TPMC"]
            else topology_by_model
        )
        report["topology_consistent"] = all(
            item["identity_mapping_allowed"] for item in topology_by_model.values()
        )
        report["force_functionals_available"] = True
        report["torque_functionals_available"] = batches["DSMC"].geometry.face_center is not None and batches["DSMC"].geometry.reference_point is not None
        report["feasible_maximum"] = {name: len(ids) for name, ids in id_sets.items()}
        report["global_cd_reconstructable"] = "C_D" in report["fidelities"]["DSMC"]["fields"]
        if report["global_cd_reconstructable"]:
            archive = adapter._archive("DSMC")
            reconstructed = np.sum(build_drag_contribution_snapshots(archive.force_per_area, archive.geometry.face_area, archive.q_inf, archive.A_ref_per_sample, archive.u_hat_inf), axis=1)
            actual = archive.scalar_cd
            candidates = {"signed": reconstructed, "negated": -reconstructed, "absolute": np.abs(reconstructed)}
            errors = {name: float(np.max(np.abs(value - actual))) for name, value in candidates.items()}
            convention = min(errors, key=errors.get)
            report["cd_reconstruction"] = {"convention": convention, "max_abs_error": errors[convention], "candidate_errors": errors}
    report["ready"] = not report["missing_fields"] and not report["invalid_samples"] and report.get("topology_consistent", False) and not any(report["duplicate_sample_ids"].values())
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data_availability_report.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
        lines = [f"# MFPOD data availability: {case}", "", f"Ready: **{report['ready']}**", "", f"DSMC: {report.get('n_dsmc', 0)}; TPMC: {report.get('n_tpmc', 0)}; Sentman: {report.get('n_sentman', 0)}; all-model paired: {report.get('n_paired', 0)}; additional TPMC: {report.get('n_additional_tpmc', 0)}; additional Sentman: {report.get('n_additional_sentman', 0)}."]
        if report["missing_fields"]: lines += ["", "Missing fields: " + ", ".join(report["missing_fields"])]
        if report["missing_preferred_fields"]: lines += ["", "Missing preferred fields: " + ", ".join(report["missing_preferred_fields"])]
        if report["invalid_samples"]: lines += ["", "Invalid: " + "; ".join(report["invalid_samples"])]
        (output_dir / "data_availability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
