from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .adbsat_surface_mapping import (
    ADBSatSurfaceMappingError,
    aggregate_panel_traction_to_reference,
    load_surface_mapping,
)
from .piclas_surface_archive import write_surface_archive_npz


def export_adbsat_surface_archive(
    *,
    result_dir: str | Path,
    method: str,
    run_ids: Sequence[int],
    sample_ids: Sequence[str],
    mapping_path: str | Path,
    output_path: str | Path,
    case_name: str,
    geometry_id: str = "",
    regime_id: str = "",
    model_id: str = "Sentman",
    append: bool = True,
) -> Dict[str, Any]:
    if len(run_ids) != len(sample_ids):
        raise ADBSatSurfaceMappingError("run_ids and sample_ids must have equal length")
    mapping = load_surface_mapping(mapping_path)
    root = Path(result_dir) / "surface_fields"
    force_rows = []
    q_rows = []
    aref_rows = []
    u_rows = []
    cd_rows = []
    source_rows = []
    for run_id, expected_sample_id in zip(run_ids, sample_ids):
        source = root / f"{method}_{int(run_id)}.npz"
        if not source.exists():
            raise FileNotFoundError(f"Missing ADBSat panel surface field: {source}")
        with np.load(source, allow_pickle=False) as data:
            fingerprint = str(np.asarray(data["mesh_fingerprint"]).reshape(-1)[0])
            if fingerprint and fingerprint != mapping.mesh_fingerprint:
                raise ADBSatSurfaceMappingError(
                    f"ADBSat mesh fingerprint mismatch for {source}: {fingerprint} != {mapping.mesh_fingerprint}"
                )
            panel_area = np.asarray(data["panel_area"], dtype=float).reshape(-1)
            if panel_area.shape != mapping.triangle_area.shape or not np.allclose(
                panel_area, mapping.triangle_area, rtol=1.0e-10, atol=1.0e-14
            ):
                raise ADBSatSurfaceMappingError(f"ADBSat panel areas do not match canonical mapping: {source}")
            for field, expected in (
                ("panel_center", mapping.triangle_center),
                ("panel_normal", mapping.triangle_normal),
            ):
                actual = np.asarray(data[field], dtype=float)
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-12):
                    raise ADBSatSurfaceMappingError(f"ADBSat {field} does not match canonical mapping: {source}")
            panel_force = np.asarray(data["panel_force_per_area"], dtype=float)
            reference_force = aggregate_panel_traction_to_reference(panel_force, mapping)
            q_inf = float(np.asarray(data["q_inf"]).reshape(-1)[0])
            A_ref = float(np.asarray(data["A_ref"]).reshape(-1)[0])
            u_hat = np.asarray(data["u_hat_inf"], dtype=float).reshape(3)
            u_hat /= np.linalg.norm(u_hat)
            cd = float(np.asarray(data["C_D"]).reshape(-1)[0])
            total_force = np.sum(reference_force * mapping.reference_face_area[:, None], axis=0)
            reconstructed_cd = abs(float(-np.dot(total_force, u_hat) / (q_inf * A_ref)))
            if not np.isclose(reconstructed_cd, cd, rtol=1.0e-10, atol=1.0e-12):
                raise ADBSatSurfaceMappingError(
                    f"Mapped ADBSat panel field does not reconstruct C_D for {source}: {reconstructed_cd} != {cd}"
                )
            force_rows.append(reference_force)
            q_rows.append(q_inf)
            aref_rows.append(A_ref)
            u_rows.append(u_hat)
            cd_rows.append(cd)
            archived_sample_id = str(np.asarray(data["sample_id"]).reshape(-1)[0])
            if archived_sample_id != str(expected_sample_id):
                raise ADBSatSurfaceMappingError(
                    f"ADBSat surface sample mismatch for run {run_id}: {archived_sample_id} != {expected_sample_id}"
                )
            source_rows.append(str(source))

    payload = {
        "force_per_area": np.asarray(force_rows, dtype=float),
        "sample_id": np.asarray([str(v) for v in sample_ids]),
        "face_area": mapping.reference_face_area,
        "face_center": mapping.reference_face_center,
        "face_normal": mapping.reference_face_normal,
        "A_ref": np.asarray([aref_rows[0]], dtype=float),
        "A_ref_per_sample": np.asarray(aref_rows, dtype=float),
        "q_inf": np.asarray(q_rows, dtype=float),
        "u_hat_inf": np.asarray(u_rows, dtype=float),
        "C_D": np.asarray(cd_rows, dtype=float),
        "job_subdir": np.asarray(source_rows),
        "fidelity": np.asarray(["SENTMAN"]),
        "model_id": np.asarray([str(model_id)]),
        "case_name": np.asarray([str(case_name)]),
        "geometry_id": np.asarray([str(geometry_id)]),
        "regime_id": np.asarray([str(regime_id)]),
        "mesh_fingerprint": np.asarray([mapping.mesh_fingerprint]),
        "surface_mapping_path": np.asarray([str(Path(mapping_path).resolve())]),
    }
    return write_surface_archive_npz(output_path, payload, append=append)
