from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class PiclasSurfaceArchiveError(RuntimeError):
    pass


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


def _write_manifest_csv(path: Path, manifest: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "fidelity",
        "model_id",
        "case_name",
        "geometry_id",
        "regime_id",
        "job_subdir",
        "q_inf",
        "A_ref",
        "C_D",
        "aos_deg",
        "aoa_deg",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _load_existing(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _merge_by_sample_id(existing: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return new

    old_ids = [str(v) for v in np.asarray(existing["sample_id"]).reshape(-1)]
    new_ids = [str(v) for v in np.asarray(new["sample_id"]).reshape(-1)]
    merged_order = list(old_ids)
    for sample_id in new_ids:
        if sample_id not in merged_order:
            merged_order.append(sample_id)

    old_lookup = {sample_id: idx for idx, sample_id in enumerate(old_ids)}
    new_lookup = {sample_id: idx for idx, sample_id in enumerate(new_ids)}
    row_fields = {
        "force_per_area",
        "q_inf",
        "u_hat_inf",
        "C_D",
        "aos_deg",
        "aoa_deg",
        "job_subdir",
        "A_ref_per_sample",
    }
    merged: Dict[str, Any] = {}
    for key, value in new.items():
        if key in row_fields:
            rows = []
            for sample_id in merged_order:
                if sample_id in new_lookup:
                    rows.append(np.asarray(new[key])[new_lookup[sample_id]])
                else:
                    rows.append(np.asarray(existing[key])[old_lookup[sample_id]])
            merged[key] = np.asarray(rows)
        elif key == "sample_id":
            merged[key] = np.asarray(merged_order, dtype=str)
        elif key == "A_ref":
            # The existing scalar PICLAS convention may use wind-projected
            # reference area, which changes under attitude dispersion. Keep one
            # scalar for first-demonstrator compatibility and preserve the
            # per-sample values in A_ref_per_sample.
            arr = np.asarray(value)
            merged[key] = np.asarray(existing[key]) if key in existing else arr
        else:
            old = np.asarray(existing.get(key)) if key in existing else None
            arr = np.asarray(value)
            if old is not None and old.size and arr.size and old.shape == arr.shape:
                if np.issubdtype(arr.dtype, np.number) and not np.allclose(old, arr, rtol=1.0e-10, atol=1.0e-12):
                    raise PiclasSurfaceArchiveError(f"Cannot merge surface archive: field '{key}' changed")
            merged[key] = arr

    for key, value in existing.items():
        if key not in merged and key not in row_fields:
            merged[key] = value
    return merged


def write_surface_archive_npz(
    path: str | os.PathLike[str],
    payload: Dict[str, Any],
    *,
    append: bool = True,
) -> Dict[str, Any]:
    """Write or append a PICLAS surface-load archive keyed by sample_id."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: np.asarray(value) for key, value in payload.items() if value is not None}
    if "sample_id" not in normalized or "force_per_area" not in normalized:
        raise PiclasSurfaceArchiveError("surface archive payload requires sample_id and force_per_area")
    merged = _merge_by_sample_id(_load_existing(out) if append else None, normalized)
    np.savez_compressed(out, **merged)
    sample_count = int(np.asarray(merged["sample_id"]).reshape(-1).size)
    summary = {
        "path": str(out),
        "n_samples": sample_count,
        "n_faces": int(np.asarray(merged["force_per_area"]).shape[1]),
        "fields": sorted(merged.keys()),
    }
    _write_json(out.with_suffix(".summary.json"), summary)

    manifest = []
    ids = [str(v) for v in np.asarray(merged["sample_id"]).reshape(-1)]
    for idx, sample_id in enumerate(ids):
        manifest.append(
            {
                "sample_id": sample_id,
                "fidelity": str(np.asarray(merged.get("fidelity", [""])).reshape(-1)[0]) if "fidelity" in merged else "",
                "model_id": str(np.asarray(merged.get("model_id", [""])).reshape(-1)[0]) if "model_id" in merged else "",
                "case_name": str(np.asarray(merged.get("case_name", [""])).reshape(-1)[0]) if "case_name" in merged else "",
                "geometry_id": str(np.asarray(merged.get("geometry_id", [""])).reshape(-1)[0]) if "geometry_id" in merged else "",
                "regime_id": str(np.asarray(merged.get("regime_id", [""])).reshape(-1)[0]) if "regime_id" in merged else "",
                "job_subdir": str(np.asarray(merged.get("job_subdir", [""] * len(ids))).reshape(-1)[idx]) if "job_subdir" in merged else "",
                "q_inf": float(np.asarray(merged["q_inf"]).reshape(-1)[idx]) if "q_inf" in merged else float("nan"),
                "A_ref": float(np.asarray(merged.get("A_ref_per_sample", [np.asarray(merged.get("A_ref", [np.nan])).reshape(-1)[0]])).reshape(-1)[idx if "A_ref_per_sample" in merged else 0]),
                "C_D": float(np.asarray(merged["C_D"]).reshape(-1)[idx]) if "C_D" in merged else float("nan"),
                "aos_deg": float(np.asarray(merged["aos_deg"]).reshape(-1)[idx]) if "aos_deg" in merged else float("nan"),
                "aoa_deg": float(np.asarray(merged["aoa_deg"]).reshape(-1)[idx]) if "aoa_deg" in merged else float("nan"),
            }
        )
    _write_manifest_csv(out.with_suffix(".manifest.csv"), manifest)
    return summary


def _module_helpers(sim: Any) -> Dict[str, Any]:
    module = importlib.import_module(sim.__class__.__module__)
    required = [
        "_force_frame_axes",
        "_extract_force_per_area_cell",
        "cell_areas_and_total",
        "wind_projected_reference_area",
    ]
    helpers = {}
    for name in required:
        if not hasattr(module, name):
            raise PiclasSurfaceArchiveError(f"Simulator module '{module.__name__}' lacks helper '{name}'")
        helpers[name] = getattr(module, name)
    return helpers


def _cell_normals(mesh: Any) -> Optional[np.ndarray]:
    try:
        with_normals = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        if "Normals" in with_normals.cell_data:
            normals = np.asarray(with_normals.cell_data["Normals"], dtype=float)
            if normals.ndim == 2 and normals.shape[1] == 3:
                norms = np.linalg.norm(normals, axis=1)
                return normals / (norms[:, None] + 1.0e-16)
    except Exception:
        return None
    return None


def collect_piclas_surface_archive_payload(
    sim: Any,
    batch_handle: Dict[str, Any],
    *,
    fidelity: str,
    model_id: str,
    case_name: str,
    geometry_id: str = "",
    regime_id: str = "",
) -> Dict[str, Any]:
    """Collect averaged PICLAS Total_ForcePerArea fields for one completed batch."""
    helpers = _module_helpers(sim)
    job_subdirs = list(batch_handle.get("job_subdirs", []))
    sample_ids = [str(v) for v in batch_handle.get("sample_ids", [])]
    aos_seq = np.asarray(batch_handle.get("aos_seq", [0.0] * len(job_subdirs)), dtype=float).reshape(-1)
    aoa_seq = np.asarray(batch_handle.get("aoa_seq", [0.0] * len(job_subdirs)), dtype=float).reshape(-1)
    flow_zero_direction = batch_handle.get("flow_zero_direction", getattr(sim, "flow_zero_direction", None))
    if len(sample_ids) != len(job_subdirs):
        raise PiclasSurfaceArchiveError("sample_ids and job_subdirs lengths differ; cannot build coupled surface archive")

    force_rows: List[np.ndarray] = []
    q_rows: List[float] = []
    u_rows: List[np.ndarray] = []
    cd_rows: List[float] = []
    aref_rows: List[float] = []
    centers_ref: Optional[np.ndarray] = None
    normals_ref: Optional[np.ndarray] = None
    areas_ref: Optional[np.ndarray] = None

    for idx, subdir in enumerate(job_subdirs):
        output_files = sim._output_vtu_files(subdir) if hasattr(sim, "_output_vtu_files") else []
        if not output_files:
            raise FileNotFoundError(f"No PICLAS output*.vtu files found in {subdir}")
        area_file = output_files[0]
        areas, _ = helpers["cell_areas_and_total"](area_file)
        areas = np.asarray(areas, dtype=float).reshape(-1)
        flow_dir, _, _ = helpers["_force_frame_axes"](
            float(aos_seq[idx]) if idx < aos_seq.size else 0.0,
            float(aoa_seq[idx]) if idx < aoa_seq.size else 0.0,
            flow_zero_direction,
        )
        A_ref = float(helpers["wind_projected_reference_area"](area_file, flow_dir, areas))
        dyn_p = float(np.loadtxt(os.path.join(subdir, "dyn_p.txt")))
        result_files = output_files[1:] if len(output_files) > 1 else output_files
        force_stack: List[np.ndarray] = []
        centers = None
        normals = None
        for path in result_files:
            module = importlib.import_module(sim.__class__.__module__)
            mesh = module.pv.read(path)
            force_pa = np.asarray(helpers["_extract_force_per_area_cell"](mesh), dtype=float)
            if force_pa.ndim != 2 or force_pa.shape[1] < 3:
                raise PiclasSurfaceArchiveError(f"Surface archive requires vector Total_ForcePerArea in {path}")
            force_pa = force_pa[:, :3]
            if force_pa.shape[0] != areas.size:
                raise PiclasSurfaceArchiveError(f"Cell count mismatch in {path}")
            force_stack.append(force_pa)
            if centers is None:
                centers = np.asarray(mesh.cell_centers().points, dtype=float)
                normals = _cell_normals(mesh)
        force_mean = np.mean(np.stack(force_stack, axis=0), axis=0)
        total_force = np.sum(force_mean * areas.reshape(-1, 1), axis=0)
        c_vec = total_force / (dyn_p * A_ref)
        cd = float(abs(-np.dot(c_vec, flow_dir)))

        if areas_ref is None:
            areas_ref = areas
            centers_ref = centers
            normals_ref = normals
        else:
            if areas.shape != areas_ref.shape or not np.allclose(areas, areas_ref, rtol=1.0e-10, atol=0.0):
                raise PiclasSurfaceArchiveError("PICLAS surface areas changed within one archive batch")
            if centers_ref is not None and centers is not None and not np.allclose(centers, centers_ref, rtol=0.0, atol=1.0e-12):
                raise PiclasSurfaceArchiveError("PICLAS face centers changed within one archive batch")

        force_rows.append(force_mean)
        q_rows.append(dyn_p)
        u_rows.append(np.asarray(flow_dir, dtype=float))
        cd_rows.append(cd)
        aref_rows.append(A_ref)

    if areas_ref is None:
        raise PiclasSurfaceArchiveError("No PICLAS surface fields collected")
    aref_arr = np.asarray(aref_rows, dtype=float)
    A_ref = float(aref_arr[0])
    payload: Dict[str, Any] = {
        "force_per_area": np.asarray(force_rows, dtype=float),
        "sample_id": np.asarray(sample_ids, dtype=str),
        "face_area": np.asarray(areas_ref, dtype=float),
        "A_ref": np.asarray([A_ref], dtype=float),
        "A_ref_per_sample": aref_arr,
        "q_inf": np.asarray(q_rows, dtype=float),
        "u_hat_inf": np.asarray(u_rows, dtype=float),
        "C_D": np.asarray(cd_rows, dtype=float),
        "aos_deg": np.asarray(aos_seq[: len(job_subdirs)], dtype=float),
        "aoa_deg": np.asarray(aoa_seq[: len(job_subdirs)], dtype=float),
        "job_subdir": np.asarray(job_subdirs, dtype=str),
        "fidelity": np.asarray([str(fidelity)], dtype=str),
        "model_id": np.asarray([str(model_id)], dtype=str),
        "case_name": np.asarray([str(case_name)], dtype=str),
        "geometry_id": np.asarray([str(geometry_id)], dtype=str),
        "regime_id": np.asarray([str(regime_id)], dtype=str),
    }
    if centers_ref is not None:
        payload["face_center"] = np.asarray(centers_ref, dtype=float)
    if normals_ref is not None:
        payload["face_normal"] = np.asarray(normals_ref, dtype=float)
    return payload


def export_piclas_surface_archive(
    sim: Any,
    batch_handle: Dict[str, Any],
    output_path: str | os.PathLike[str],
    *,
    fidelity: str,
    model_id: str,
    case_name: str,
    geometry_id: str = "",
    regime_id: str = "",
    append: bool = True,
) -> Dict[str, Any]:
    payload = collect_piclas_surface_archive_payload(
        sim,
        batch_handle,
        fidelity=fidelity,
        model_id=model_id,
        case_name=case_name,
        geometry_id=geometry_id,
        regime_id=regime_id,
    )
    return write_surface_archive_npz(output_path, payload, append=append)
