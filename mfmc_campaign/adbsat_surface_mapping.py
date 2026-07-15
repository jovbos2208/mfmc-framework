from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


class ADBSatSurfaceMappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalSurfaceMapping:
    source_vtu: str
    mesh_fingerprint: str
    points: np.ndarray
    triangles: np.ndarray
    triangle_to_reference_cell: np.ndarray
    triangle_area: np.ndarray
    triangle_center: np.ndarray
    triangle_normal: np.ndarray
    reference_face_area: np.ndarray
    reference_face_center: np.ndarray
    reference_face_normal: np.ndarray
    length_scale_to_m: float

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def n_reference_faces(self) -> int:
        return int(self.reference_face_area.size)


def _cell_point_ids(mesh: Any, cell_id: int) -> List[int]:
    cell = mesh.get_cell(cell_id)
    ids = [int(v) for v in np.asarray(cell.point_ids).reshape(-1)]
    if len(ids) < 3:
        raise ADBSatSurfaceMappingError(f"VTU cell {cell_id} has fewer than three vertices")
    return ids


def _fan_triangulation(point_ids: Sequence[int]) -> Iterable[Tuple[int, int, int]]:
    for idx in range(1, len(point_ids) - 1):
        yield int(point_ids[0]), int(point_ids[idx]), int(point_ids[idx + 1])


def _triangle_geometry(points: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = points[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(twice_area)) or np.any(twice_area <= 0.0):
        bad = np.where((~np.isfinite(twice_area)) | (twice_area <= 0.0))[0]
        raise ADBSatSurfaceMappingError(f"Degenerate generated triangles: {bad[:10].tolist()}")
    area = 0.5 * twice_area
    normal = cross / twice_area[:, None]
    center = np.mean(xyz, axis=1)
    return area, center, normal


def _reference_geometry(
    triangle_to_reference_cell: np.ndarray,
    triangle_area: np.ndarray,
    triangle_center: np.ndarray,
    triangle_normal: np.ndarray,
    n_reference_faces: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    area = np.zeros(n_reference_faces, dtype=float)
    center_sum = np.zeros((n_reference_faces, 3), dtype=float)
    normal_sum = np.zeros((n_reference_faces, 3), dtype=float)
    np.add.at(area, triangle_to_reference_cell, triangle_area)
    np.add.at(center_sum, triangle_to_reference_cell, triangle_area[:, None] * triangle_center)
    np.add.at(normal_sum, triangle_to_reference_cell, triangle_area[:, None] * triangle_normal)
    if np.any(area <= 0.0):
        raise ADBSatSurfaceMappingError("At least one VTU reference cell received no triangles")
    center = center_sum / area[:, None]
    normal_norm = np.linalg.norm(normal_sum, axis=1)
    if np.any(normal_norm <= 0.0):
        raise ADBSatSurfaceMappingError("At least one VTU reference cell has an undefined normal")
    normal = normal_sum / normal_norm[:, None]
    return area, center, normal


def _mesh_fingerprint(
    points: np.ndarray,
    triangles: np.ndarray,
    triangle_to_reference_cell: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for array in (points.astype("<f8", copy=False), triangles.astype("<i8", copy=False), triangle_to_reference_cell.astype("<i8", copy=False)):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_mapping_from_vtu(
    vtu_path: str | Path,
    *,
    length_scale_to_m: float = 1.0,
) -> CanonicalSurfaceMapping:
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - dependency error is environment-specific
        raise ADBSatSurfaceMappingError("pyvista is required for VTU surface conversion") from exc

    source = Path(vtu_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    scale = float(length_scale_to_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ADBSatSurfaceMappingError("length_scale_to_m must be finite and positive")

    mesh = pv.read(source)
    points = np.asarray(mesh.points, dtype=float) * scale
    if points.ndim != 2 or points.shape[1] != 3:
        raise ADBSatSurfaceMappingError("VTU points must have shape (n_points, 3)")
    triangles: List[Tuple[int, int, int]] = []
    parent: List[int] = []
    supported_surface_cell_types = {5, 7, 9}  # VTK_TRIANGLE, VTK_POLYGON, VTK_QUAD
    for cell_id in range(int(mesh.n_cells)):
        cell_type = int(np.asarray(mesh.celltypes).reshape(-1)[cell_id])
        if cell_type not in supported_surface_cell_types:
            raise ADBSatSurfaceMappingError(
                f"Unsupported VTU cell type {cell_type} at cell {cell_id}; "
                "the canonical file must contain only linear triangles, quads or polygons"
            )
        cell_triangles = list(_fan_triangulation(_cell_point_ids(mesh, cell_id)))
        triangles.extend(cell_triangles)
        parent.extend([cell_id] * len(cell_triangles))
    triangle_array = np.asarray(triangles, dtype=np.int64)
    parent_array = np.asarray(parent, dtype=np.int64)
    tri_area, tri_center, tri_normal = _triangle_geometry(points, triangle_array)
    face_area, _face_center_fallback, face_normal_fallback = _reference_geometry(
        parent_array,
        tri_area,
        tri_center,
        tri_normal,
        int(mesh.n_cells),
    )
    piclas_area = np.asarray(
        mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"],
        dtype=float,
    ) * scale**2
    if piclas_area.shape != face_area.shape or not np.allclose(piclas_area, face_area, rtol=1.0e-10, atol=1.0e-14):
        raise ADBSatSurfaceMappingError(
            "Deterministic triangulation does not preserve the PICLAS VTU cell areas"
        )
    face_center = np.asarray(mesh.cell_centers().points, dtype=float) * scale
    face_normal = face_normal_fallback
    try:
        with_normals = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        candidate = np.asarray(with_normals.cell_data["Normals"], dtype=float)
        if candidate.shape == face_normal.shape:
            norms = np.linalg.norm(candidate, axis=1)
            if np.all(norms > 0.0):
                face_normal = candidate / norms[:, None]
    except Exception:
        pass
    return CanonicalSurfaceMapping(
        source_vtu=str(source),
        mesh_fingerprint=_mesh_fingerprint(points, triangle_array, parent_array),
        points=points,
        triangles=triangle_array,
        triangle_to_reference_cell=parent_array,
        triangle_area=tri_area,
        triangle_center=tri_center,
        triangle_normal=tri_normal,
        reference_face_area=face_area,
        reference_face_center=face_center,
        reference_face_normal=face_normal,
        length_scale_to_m=scale,
    )


def write_adbsat_obj(path: str | Path, mapping: CanonicalSurfaceMapping, material_id: int = 1) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Canonical ADBSat mesh generated from PICLAS VTU"]
    lines.extend(f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in mapping.points)
    lines.append(f"usemtl {int(material_id)}")
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mapping.triangles)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_adbsat_mat(path: str | Path, mapping: CanonicalSurfaceMapping, material_id: int = 1) -> None:
    try:
        from scipy.io import savemat
    except Exception as exc:  # pragma: no cover
        raise ADBSatSurfaceMappingError("scipy is required for ADBSat MAT generation") from exc

    xyz = mapping.points[mapping.triangles]
    length_ref = float(np.ptp(mapping.points[:, 0]))
    if length_ref <= 0.0:
        length_ref = float(np.max(np.ptp(mapping.points, axis=0)))
    meshdata = {
        "XData": xyz[:, :, 0].T,
        "YData": xyz[:, :, 1].T,
        "ZData": xyz[:, :, 2].T,
        "MatID": np.full(mapping.n_triangles, int(material_id), dtype=np.int64),
        "Areas": mapping.triangle_area,
        "SurfN": mapping.triangle_normal.T,
        "BariC": mapping.triangle_center.T,
        "Lref": max(length_ref, 1.0e-12),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        out,
        {
            "meshdata": meshdata,
            "mesh_fingerprint": np.asarray([mapping.mesh_fingerprint]),
            "source_vtu": np.asarray([mapping.source_vtu]),
        },
    )


def write_surface_mapping(path: str | Path, mapping: CanonicalSurfaceMapping) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source_vtu=np.asarray([mapping.source_vtu]),
        mesh_fingerprint=np.asarray([mapping.mesh_fingerprint]),
        length_scale_to_m=np.asarray([mapping.length_scale_to_m]),
        points=mapping.points,
        triangles=mapping.triangles,
        triangle_to_reference_cell=mapping.triangle_to_reference_cell,
        triangle_area=mapping.triangle_area,
        triangle_center=mapping.triangle_center,
        triangle_normal=mapping.triangle_normal,
        reference_face_area=mapping.reference_face_area,
        reference_face_center=mapping.reference_face_center,
        reference_face_normal=mapping.reference_face_normal,
    )


def load_surface_mapping(path: str | Path) -> CanonicalSurfaceMapping:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        return CanonicalSurfaceMapping(
            source_vtu=str(np.asarray(data["source_vtu"]).reshape(-1)[0]),
            mesh_fingerprint=str(np.asarray(data["mesh_fingerprint"]).reshape(-1)[0]),
            points=np.asarray(data["points"], dtype=float),
            triangles=np.asarray(data["triangles"], dtype=np.int64),
            triangle_to_reference_cell=np.asarray(data["triangle_to_reference_cell"], dtype=np.int64),
            triangle_area=np.asarray(data["triangle_area"], dtype=float),
            triangle_center=np.asarray(data["triangle_center"], dtype=float),
            triangle_normal=np.asarray(data["triangle_normal"], dtype=float),
            reference_face_area=np.asarray(data["reference_face_area"], dtype=float),
            reference_face_center=np.asarray(data["reference_face_center"], dtype=float),
            reference_face_normal=np.asarray(data["reference_face_normal"], dtype=float),
            length_scale_to_m=float(np.asarray(data["length_scale_to_m"]).reshape(-1)[0]),
        )


def aggregate_panel_traction_to_reference(
    panel_traction: np.ndarray,
    mapping: CanonicalSurfaceMapping,
) -> np.ndarray:
    values = np.asarray(panel_traction, dtype=float)
    single = values.ndim == 2
    if single:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1:] != (mapping.n_triangles, 3):
        raise ADBSatSurfaceMappingError(
            f"panel_traction must have shape (n_samples, {mapping.n_triangles}, 3) or ({mapping.n_triangles}, 3)"
        )
    force = np.zeros((values.shape[0], mapping.n_reference_faces, 3), dtype=float)
    weighted = values * mapping.triangle_area[None, :, None]
    for sample_idx in range(values.shape[0]):
        np.add.at(force[sample_idx], mapping.triangle_to_reference_cell, weighted[sample_idx])
    result = force / mapping.reference_face_area[None, :, None]
    return result[0] if single else result


def build_and_write_adbsat_surface(
    vtu_path: str | Path,
    obj_path: str | Path,
    mat_path: str | Path,
    mapping_path: str | Path,
    *,
    length_scale_to_m: float = 1.0,
    material_id: int = 1,
) -> Dict[str, Any]:
    mapping = build_mapping_from_vtu(vtu_path, length_scale_to_m=length_scale_to_m)
    write_adbsat_obj(obj_path, mapping, material_id=material_id)
    write_adbsat_mat(mat_path, mapping, material_id=material_id)
    write_surface_mapping(mapping_path, mapping)
    summary = {
        "source_vtu": mapping.source_vtu,
        "obj_path": str(Path(obj_path).resolve()),
        "mat_path": str(Path(mat_path).resolve()),
        "mapping_path": str(Path(mapping_path).resolve()),
        "mesh_fingerprint": mapping.mesh_fingerprint,
        "n_points": int(mapping.points.shape[0]),
        "n_reference_faces": mapping.n_reference_faces,
        "n_adbsat_triangles": mapping.n_triangles,
        "surface_area_m2": float(np.sum(mapping.reference_face_area)),
        "length_scale_to_m": mapping.length_scale_to_m,
    }
    summary_path = Path(mapping_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
