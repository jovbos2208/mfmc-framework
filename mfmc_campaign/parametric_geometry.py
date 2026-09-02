from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


GENERATOR_NAME = "mfmc-cylinder-hex"
GENERATOR_VERSION = "1.0.0"


class ParametricGeometryError(ValueError):
    pass


@dataclass(frozen=True)
class CylinderHexSpec:
    nose_length_fraction: float = 0.20
    tail_length_fraction: float = 0.20
    width_height_ratio: float = 1.0
    chamfer_fraction: float = 0.15
    total_length_m: float = 1.0
    target_volume_m3: float = 0.12
    max_width_m: float = 0.60
    max_height_m: float = 0.60
    nose_end_scale: float = 0.20
    tail_end_scale: float = 0.45
    axial_segments_per_taper: int = 1

    def scaled(self, linear_factor: float) -> "CylinderHexSpec":
        factor = float(linear_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ParametricGeometryError("uniform linear scale factor must be finite and positive")
        return replace(
            self,
            total_length_m=self.total_length_m * factor,
            target_volume_m3=self.target_volume_m3 * factor**3,
            max_width_m=self.max_width_m * factor,
            max_height_m=self.max_height_m * factor,
        )

    def validate(self) -> None:
        values = np.asarray(
            [
                self.nose_length_fraction,
                self.tail_length_fraction,
                self.width_height_ratio,
                self.chamfer_fraction,
                self.total_length_m,
                self.target_volume_m3,
                self.max_width_m,
                self.max_height_m,
                self.nose_end_scale,
                self.tail_end_scale,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ParametricGeometryError("All geometry parameters must be finite")
        if not 0.05 <= self.nose_length_fraction <= 0.40:
            raise ParametricGeometryError("nose_length_fraction must be in [0.05, 0.40]")
        if not 0.05 <= self.tail_length_fraction <= 0.40:
            raise ParametricGeometryError("tail_length_fraction must be in [0.05, 0.40]")
        if self.nose_length_fraction + self.tail_length_fraction > 0.75:
            raise ParametricGeometryError("nose and tail fractions must sum to at most 0.75")
        if not 0.50 <= self.width_height_ratio <= 2.0:
            raise ParametricGeometryError("width_height_ratio must be in [0.50, 2.0]")
        if not 0.02 <= self.chamfer_fraction <= 0.35:
            raise ParametricGeometryError("chamfer_fraction must be in [0.02, 0.35]")
        if self.total_length_m <= 0.0 or self.target_volume_m3 <= 0.0:
            raise ParametricGeometryError("length and target volume must be positive")
        if self.max_width_m <= 0.0 or self.max_height_m <= 0.0:
            raise ParametricGeometryError("maximum envelope dimensions must be positive")
        if not 0.05 <= self.nose_end_scale < 1.0 or not 0.05 <= self.tail_end_scale < 1.0:
            raise ParametricGeometryError("end scales must be in [0.05, 1.0)")
        if int(self.axial_segments_per_taper) < 1:
            raise ParametricGeometryError("axial_segments_per_taper must be at least one")


@dataclass(frozen=True)
class SurfaceMesh:
    points: np.ndarray
    triangles: np.ndarray
    triangle_area: np.ndarray
    triangle_normal: np.ndarray
    triangle_center: np.ndarray
    mesh_fingerprint: str


def _cross_section(width: float, height: float, chamfer_fraction: float) -> np.ndarray:
    half_w = 0.5 * width
    half_h = 0.5 * height
    cut_w = chamfer_fraction * width
    cut_h = chamfer_fraction * height
    return np.asarray(
        [
            [-half_w + cut_w, -half_h],
            [half_w - cut_w, -half_h],
            [half_w, -half_h + cut_h],
            [half_w, half_h - cut_h],
            [half_w - cut_w, half_h],
            [-half_w + cut_w, half_h],
            [-half_w, half_h - cut_h],
            [-half_w, -half_h + cut_h],
        ],
        dtype=float,
    )


def _scale_square_integral(spec: CylinderHexSpec) -> float:
    nose = spec.nose_length_fraction * spec.total_length_m
    tail = spec.tail_length_fraction * spec.total_length_m
    center = spec.total_length_m - nose - tail
    nose_factor = (spec.nose_end_scale**2 + spec.nose_end_scale + 1.0) / 3.0
    tail_factor = (1.0 + spec.tail_end_scale + spec.tail_end_scale**2) / 3.0
    return nose * nose_factor + center + tail * tail_factor


def _derived_dimensions(spec: CylinderHexSpec) -> Dict[str, float]:
    section_factor = 1.0 - 2.0 * spec.chamfer_fraction**2
    maximum_section_area = spec.target_volume_m3 / _scale_square_integral(spec)
    height = np.sqrt(maximum_section_area / (spec.width_height_ratio * section_factor))
    width = spec.width_height_ratio * height
    return {
        "width_m": float(width),
        "height_m": float(height),
        "maximum_cross_section_area_m2": float(maximum_section_area),
        "center_length_m": float(
            spec.total_length_m * (1.0 - spec.nose_length_fraction - spec.tail_length_fraction)
        ),
    }


def _stations(spec: CylinderHexSpec) -> List[Tuple[float, float]]:
    count = int(spec.axial_segments_per_taper)
    nose_end = spec.nose_length_fraction * spec.total_length_m
    tail_start = spec.total_length_m * (1.0 - spec.tail_length_fraction)
    stations: List[Tuple[float, float]] = []
    for index in range(count + 1):
        fraction = index / count
        stations.append(
            (nose_end * fraction, spec.nose_end_scale + fraction * (1.0 - spec.nose_end_scale))
        )
    if tail_start > nose_end:
        stations.append((tail_start, 1.0))
    for index in range(1, count + 1):
        fraction = index / count
        stations.append(
            (
                tail_start + fraction * (spec.total_length_m - tail_start),
                1.0 + fraction * (spec.tail_end_scale - 1.0),
            )
        )
    return stations


def _triangle_geometry(points: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = points[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    if np.any(norm <= 1.0e-14) or np.any(~np.isfinite(norm)):
        raise ParametricGeometryError("Generated surface contains degenerate triangles")
    return 0.5 * norm, cross / norm[:, None], np.mean(xyz, axis=1)


def _fingerprint(points: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (points.astype("<f8", copy=False), triangles.astype("<i8", copy=False)):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def surface_mesh_from_points(points: np.ndarray, triangles: np.ndarray) -> SurfaceMesh:
    """Build a surface representation from an existing connectivity graph."""
    point_array = np.asarray(points, dtype=float)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ParametricGeometryError("surface points must have shape (n, 3)")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ParametricGeometryError("surface triangles must have shape (m, 3)")
    if len(point_array) < 4 or len(triangle_array) < 4:
        raise ParametricGeometryError("surface mesh is too small to enclose a volume")
    if np.min(triangle_array) < 0 or np.max(triangle_array) >= len(point_array):
        raise ParametricGeometryError("surface triangle connectivity is out of bounds")
    oriented = _orient_outward(point_array, triangle_array)
    area, normal, center = _triangle_geometry(point_array, oriented)
    return SurfaceMesh(
        points=point_array,
        triangles=oriented,
        triangle_area=area,
        triangle_normal=normal,
        triangle_center=center,
        mesh_fingerprint=_fingerprint(point_array, oriented),
    )


def projected_reference_area(mesh: SurfaceMesh, axis: int = 0) -> float:
    """Return the two-sided projected area normal to one Cartesian axis."""
    if axis not in (0, 1, 2):
        raise ParametricGeometryError("projected-area axis must be 0, 1, or 2")
    return float(0.5 * np.sum(mesh.triangle_area * np.abs(mesh.triangle_normal[:, axis])))


def _orient_outward(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    oriented = np.array(triangles, copy=True)
    center = np.mean(points, axis=0)
    xyz = points[oriented]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    face_centers = np.mean(xyz, axis=1)
    inward = np.einsum("ij,ij->i", normals, face_centers - center) < 0.0
    oriented[inward, 1], oriented[inward, 2] = (
        oriented[inward, 2].copy(),
        oriented[inward, 1].copy(),
    )
    return oriented


def generate_cylinder_hex(spec: CylinderHexSpec) -> Tuple[SurfaceMesh, Dict[str, float]]:
    spec.validate()
    derived = _derived_dimensions(spec)
    if derived["width_m"] > spec.max_width_m + 1.0e-12:
        raise ParametricGeometryError(
            f"Fixed-volume width {derived['width_m']:.6g} m exceeds envelope {spec.max_width_m:.6g} m"
        )
    if derived["height_m"] > spec.max_height_m + 1.0e-12:
        raise ParametricGeometryError(
            f"Fixed-volume height {derived['height_m']:.6g} m exceeds envelope {spec.max_height_m:.6g} m"
        )
    section = _cross_section(derived["width_m"], derived["height_m"], spec.chamfer_fraction)
    stations = _stations(spec)
    points: List[List[float]] = []
    for x, scale in stations:
        points.extend([[x, scale * y, scale * z] for y, z in section])
    ring_size = len(section)
    triangles: List[List[int]] = []
    for ring in range(len(stations) - 1):
        first = ring * ring_size
        second = (ring + 1) * ring_size
        for index in range(ring_size):
            next_index = (index + 1) % ring_size
            triangles.append([first + index, second + index, second + next_index])
            triangles.append([first + index, second + next_index, first + next_index])
    nose_center = len(points)
    points.append([0.0, 0.0, 0.0])
    tail_center = len(points)
    points.append([spec.total_length_m, 0.0, 0.0])
    last_ring = (len(stations) - 1) * ring_size
    for index in range(ring_size):
        next_index = (index + 1) % ring_size
        triangles.append([nose_center, index, next_index])
        triangles.append([tail_center, last_ring + next_index, last_ring + index])
    point_array = np.asarray(points, dtype=float)
    triangle_array = _orient_outward(point_array, np.asarray(triangles, dtype=np.int64))
    area, normal, center = _triangle_geometry(point_array, triangle_array)
    mesh = SurfaceMesh(
        points=point_array,
        triangles=triangle_array,
        triangle_area=area,
        triangle_normal=normal,
        triangle_center=center,
        mesh_fingerprint=_fingerprint(point_array, triangle_array),
    )
    return mesh, derived


def validate_surface(mesh: SurfaceMesh, spec: CylinderHexSpec) -> Dict[str, Any]:
    edges = np.sort(
        np.concatenate(
            [mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    _unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    watertight = bool(np.all(edge_counts == 2))
    signed_volume = float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                mesh.points[mesh.triangles[:, 0]],
                np.cross(
                    mesh.points[mesh.triangles[:, 1]],
                    mesh.points[mesh.triangles[:, 2]],
                ),
            )
        )
        / 6.0
    )
    volume_error = abs(signed_volume - spec.target_volume_m3)
    volume_tolerance = max(1.0e-12, 1.0e-10 * spec.target_volume_m3)
    plane_offsets = np.einsum(
        "fvi,fi->fv",
        mesh.points[None, :, :] - mesh.points[mesh.triangles[:, 0]][:, None, :],
        mesh.triangle_normal,
    )
    convex = bool(np.max(plane_offsets) <= 1.0e-10 * max(1.0, spec.total_length_m))
    bounds_min = np.min(mesh.points, axis=0)
    bounds_max = np.max(mesh.points, axis=0)
    envelope = bounds_max - bounds_min
    outward = signed_volume > 0.0
    checks = {
        "finite": bool(np.all(np.isfinite(mesh.points)) and np.all(np.isfinite(mesh.triangle_area))),
        "nondegenerate_triangles": bool(np.all(mesh.triangle_area > 1.0e-14)),
        "watertight_two_manifold": watertight,
        "outward_orientation": outward,
        "convex_no_self_intersection_certificate": convex,
        "fixed_volume_within_tolerance": bool(volume_error <= volume_tolerance),
        "length_envelope": bool(envelope[0] <= spec.total_length_m + 1.0e-12),
        "width_envelope": bool(envelope[1] <= spec.max_width_m + 1.0e-12),
        "height_envelope": bool(envelope[2] <= spec.max_height_m + 1.0e-12),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "signed_volume_m3": signed_volume,
        "target_volume_m3": spec.target_volume_m3,
        "absolute_volume_error_m3": volume_error,
        "surface_area_m2": float(np.sum(mesh.triangle_area)),
        "bounds_min_m": bounds_min.tolist(),
        "bounds_max_m": bounds_max.tolist(),
        "envelope_m": envelope.tolist(),
        "n_vertices": int(len(mesh.points)),
        "n_triangles": int(len(mesh.triangles)),
        "n_unique_edges": int(len(_unique_edges)),
        "euler_characteristic": int(len(mesh.points) - len(_unique_edges) + len(mesh.triangles)),
    }


def _design_fingerprint(spec: CylinderHexSpec) -> str:
    payload = {
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "spec": asdict(spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_obj(path: Path, mesh: SurfaceMesh) -> None:
    lines = [f"# {GENERATOR_NAME} {GENERATOR_VERSION}", "o cylinder_hex", "usemtl 1"]
    lines.extend(f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in mesh.points)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.triangles)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_stl(path: Path, mesh: SurfaceMesh) -> None:
    lines = ["solid cylinder_hex"]
    for normal, triangle in zip(mesh.triangle_normal, mesh.triangles):
        lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
        lines.append("    outer loop")
        for point in mesh.points[triangle]:
            lines.append(f"      vertex {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}")
        lines.extend(["    endloop", "  endfacet"])
    lines.append("endsolid cylinder_hex")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_adbsat_mat(path: Path, mesh: SurfaceMesh, reference_area_m2: float) -> None:
    try:
        from scipy.io import savemat
    except Exception as exc:  # pragma: no cover
        raise ParametricGeometryError("scipy is required to write the ADBSat MAT asset") from exc
    xyz = mesh.points[mesh.triangles]
    savemat(
        path,
        {
            "meshdata": {
                "XData": xyz[:, :, 0].T,
                "YData": xyz[:, :, 1].T,
                "ZData": xyz[:, :, 2].T,
                "MatID": np.ones(len(mesh.triangles), dtype=np.int64),
                "Areas": mesh.triangle_area,
                "SurfN": mesh.triangle_normal.T,
                "BariC": mesh.triangle_center.T,
                "Lref": float(np.ptp(mesh.points[:, 0])),
                "Aref": float(reference_area_m2),
            },
            "mesh_fingerprint": np.asarray([mesh.mesh_fingerprint]),
            "generator": np.asarray([f"{GENERATOR_NAME}-{GENERATOR_VERSION}"]),
        },
    )
    # MATLAB v5 headers normally contain a wall-clock timestamp. Replace only
    # that free-text field so identical designs produce byte-identical assets.
    header = f"MATLAB 5.0 MAT-file, {GENERATOR_NAME} {GENERATOR_VERSION}".encode("ascii")
    with path.open("r+b") as target:
        target.write(header.ljust(116, b" "))


def write_gmsh_exterior_geo(
    path: Path,
    mesh: SurfaceMesh,
    spec: CylinderHexSpec,
    *,
    body_mesh_size_m: float = 0.06,
    farfield_mesh_size_m: float = 0.30,
) -> None:
    """Write a built-in-kernel Gmsh gas domain with the body as a cavity."""
    if body_mesh_size_m <= 0.0 or farfield_mesh_size_m <= 0.0:
        raise ParametricGeometryError("Gmsh mesh sizes must be positive")
    lines = [
        f"// {GENERATOR_NAME} {GENERATOR_VERSION}",
        "SetFactory(\"Built-in\");",
        f"lcBody = {body_mesh_size_m:.17g};",
        f"lcFar = {farfield_mesh_size_m:.17g};",
    ]
    for tag, (x, y, z) in enumerate(mesh.points, start=1):
        lines.append(f"Point({tag}) = {{{x:.17g}, {y:.17g}, {z:.17g}, lcBody}};")

    edge_tags: Dict[Tuple[int, int], int] = {}
    oriented_edges: List[List[int]] = []
    next_line = 1
    for triangle in mesh.triangles:
        face_edges: List[int] = []
        for start_zero, end_zero in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            start = start_zero + 1
            end = end_zero + 1
            key = (min(start, end), max(start, end))
            if key not in edge_tags:
                edge_tags[key] = next_line
                lines.append(f"Line({next_line}) = {{{key[0]}, {key[1]}}};")
                next_line += 1
            tag = edge_tags[key]
            face_edges.append(tag if (start, end) == key else -tag)
        oriented_edges.append(face_edges)

    body_surfaces: List[int] = []
    next_loop = 1
    next_surface = 1
    for edges in oriented_edges:
        lines.append(f"Curve Loop({next_loop}) = {{{', '.join(str(value) for value in edges)}}};")
        lines.append(f"Plane Surface({next_surface}) = {{{next_loop}}};")
        body_surfaces.append(next_surface)
        next_loop += 1
        next_surface += 1
    body_loop = 1
    lines.append(f"Surface Loop({body_loop}) = {{{', '.join(str(value) for value in body_surfaces)}}};")

    length = spec.total_length_m
    x_min, x_max = -0.5 * length, 2.0 * length
    y_min, y_max = -length, length
    z_min, z_max = -length, length
    outer_points = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]
    first_outer_point = len(mesh.points) + 1
    for offset, (x, y, z) in enumerate(outer_points):
        lines.append(
            f"Point({first_outer_point + offset}) = {{{x:.17g}, {y:.17g}, {z:.17g}, lcFar}};"
        )
    p = [first_outer_point + index for index in range(8)]
    outer_edge_pairs = [
        (p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[0]),
        (p[4], p[5]), (p[5], p[6]), (p[6], p[7]), (p[7], p[4]),
        (p[0], p[4]), (p[1], p[5]), (p[2], p[6]), (p[3], p[7]),
    ]
    outer_lines: List[int] = []
    for start, end in outer_edge_pairs:
        outer_lines.append(next_line)
        lines.append(f"Line({next_line}) = {{{start}, {end}}};")
        next_line += 1
    l = outer_lines
    outer_loops = [
        [l[0], l[1], l[2], l[3]],
        [l[4], l[5], l[6], l[7]],
        [l[0], l[9], -l[4], -l[8]],
        [l[1], l[10], -l[5], -l[9]],
        [l[2], l[11], -l[6], -l[10]],
        [l[3], l[8], -l[7], -l[11]],
    ]
    outer_surfaces: List[int] = []
    for edges in outer_loops:
        lines.append(f"Curve Loop({next_loop}) = {{{', '.join(str(value) for value in edges)}}};")
        lines.append(f"Plane Surface({next_surface}) = {{{next_loop}}};")
        outer_surfaces.append(next_surface)
        next_loop += 1
        next_surface += 1
    outer_loop = 2
    lines.append(f"Surface Loop({outer_loop}) = {{{', '.join(str(value) for value in outer_surfaces)}}};")
    lines.append(f"Volume(1) = {{{outer_loop}, {body_loop}}};")
    lines.append("Physical Volume(\"GAS\", 1) = {1};")
    lines.append(f"Physical Surface(\"IN\", 2) = {{{outer_surfaces[5]}}};")
    lines.append(
        f"Physical Surface(\"OUT\", 3) = "
        f"{{{', '.join(str(value) for value in outer_surfaces[:5])}}};"
    )
    lines.append(
        f"Physical Surface(\"CYLINDER_HEX\", 4) = "
        f"{{{', '.join(str(value) for value in body_surfaces)}}};"
    )
    lines.extend(
        [
            "Mesh.Algorithm = 6;",
            "Mesh.Algorithm3D = 1;",
            "Mesh.RecombineAll = 0;",
            "Mesh.SubdivisionAlgorithm = 0;",
            "Mesh.Optimize = 1;",
            "Mesh.MshFileVersion = 2.2;",
            "Mesh.Binary = 0;",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def build_geometry_assets(
    output_dir: str | Path,
    spec: CylinderHexSpec,
    *,
    design_id: str | None = None,
    uniform_scale_factor: float = 1.0,
    body_mesh_size_m: float = 0.06,
    farfield_mesh_size_m: float = 0.30,
) -> Dict[str, Any]:
    mesh, derived = generate_cylinder_hex(spec)
    validation = validate_surface(mesh, spec)
    if not validation["valid"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise ParametricGeometryError(f"Generated geometry failed validation: {failed}")
    design_fingerprint = _design_fingerprint(spec)
    geometry_id = design_id or f"cylinder_hex_{design_fingerprint[:12]}"
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    obj_path = target / f"{geometry_id}.obj"
    stl_path = target / f"{geometry_id}.stl"
    mat_path = target / f"{geometry_id}.mat"
    surface_path = target / f"{geometry_id}.surface.npz"
    gmsh_geo_path = target / f"{geometry_id}.exterior.geo"
    manifest_path = target / f"{geometry_id}.manifest.json"
    write_obj(obj_path, mesh)
    write_stl(stl_path, mesh)
    write_adbsat_mat(mat_path, mesh, derived["maximum_cross_section_area_m2"])
    write_gmsh_exterior_geo(
        gmsh_geo_path,
        mesh,
        spec,
        body_mesh_size_m=body_mesh_size_m,
        farfield_mesh_size_m=farfield_mesh_size_m,
    )
    np.savez_compressed(
        surface_path,
        points=mesh.points,
        triangles=mesh.triangles,
        triangle_area=mesh.triangle_area,
        triangle_normal=mesh.triangle_normal,
        triangle_center=mesh.triangle_center,
        mesh_fingerprint=np.asarray([mesh.mesh_fingerprint]),
        design_fingerprint=np.asarray([design_fingerprint]),
    )
    numeric_metadata = {
        "nose_length_fraction": spec.nose_length_fraction,
        "tail_length_fraction": spec.tail_length_fraction,
        "width_height_ratio": spec.width_height_ratio,
        "chamfer_fraction": spec.chamfer_fraction,
        "target_volume_m3": spec.target_volume_m3,
        "width_m": derived["width_m"],
        "height_m": derived["height_m"],
        "reference_area_m2": derived["maximum_cross_section_area_m2"],
    }
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "geometry_id": geometry_id,
        "design_fingerprint": design_fingerprint,
        "mesh_fingerprint": mesh.mesh_fingerprint,
        "specification": asdict(spec),
        "derived_geometry": derived,
        "reference_area": {
            "value_m2": derived["maximum_cross_section_area_m2"],
            "convention": "maximum cross-section projected normal to positive body x-axis",
            "optimization_quantity": "C_D_times_reference_area_m2",
            "attitude_note": "Use wind-projected area for attitude-dependent solver normalization; compare designs using dimensional drag.",
        },
        "coordinate_system": {
            "units": "m",
            "body_axis": "+x nose-to-tail",
            "width_axis": "y",
            "height_axis": "z",
            "origin": "nose cap center",
        },
        "validation": validation,
        "assets": {
            "adbsat_obj": obj_path.name,
            "adbsat_mat": mat_path.name,
            "meshing_surface_stl": stl_path.name,
            "canonical_surface_npz": surface_path.name,
            "gmsh_exterior_geo": gmsh_geo_path.name,
            "piclas_volume_mesh": None,
            "piclas_mesh_status": "requires independent HOPR/Gmsh volume-mesh stage and ingestion validation",
        },
        "campaign_geometry_descriptor": {
            "id": geometry_id,
            "name": geometry_id,
            "characteristic_length": spec.total_length_m,
            "geometry_class": "parametric_cylinder_hex",
            "tags": ["parametric", "fixed_volume", "convex", "generated_from_scratch"],
            "metadata": numeric_metadata,
        },
        "provenance": {
            "source_geometry": "analytical_parameterization",
            "external_mesh_reused": False,
            "uniform_linear_scale_factor": float(uniform_scale_factor),
            "scaling_laws": {
                "length": "factor",
                "area": "factor^2",
                "volume": "factor^3",
            },
        },
        "gmsh_mesh_controls": {
            "body_mesh_size_m": float(body_mesh_size_m),
            "farfield_mesh_size_m": float(farfield_mesh_size_m),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_json"] = str(manifest_path)
    return manifest


def validate_gmsh_msh2(path: str | Path, *, expected_gas_volume_m3: float) -> Dict[str, Any]:
    source = Path(path).resolve()
    lines = source.read_text(encoding="ascii").splitlines()
    try:
        node_start = lines.index("$Nodes")
        element_start = lines.index("$Elements")
    except ValueError as exc:
        raise ParametricGeometryError("Expected an ASCII Gmsh MSH 2.2 file") from exc
    node_count = int(lines[node_start + 1])
    nodes: Dict[int, np.ndarray] = {}
    for line in lines[node_start + 2 : node_start + 2 + node_count]:
        fields = line.split()
        nodes[int(fields[0])] = np.asarray([float(value) for value in fields[1:4]], dtype=float)
    element_count = int(lines[element_start + 1])
    tetra_volumes: List[float] = []
    triangle_physical_counts = {2: 0, 3: 0, 4: 0}
    parsed_elements = 0
    for line in lines[element_start + 2 : element_start + 2 + element_count]:
        fields = [int(value) for value in line.split()]
        element_type = fields[1]
        tag_count = fields[2]
        tags = fields[3 : 3 + tag_count]
        connectivity = fields[3 + tag_count :]
        parsed_elements += 1
        if element_type == 2 and tags and tags[0] in triangle_physical_counts:
            triangle_physical_counts[tags[0]] += 1
        elif element_type == 4:
            xyz = np.asarray([nodes[node] for node in connectivity[:4]], dtype=float)
            volume = abs(float(np.dot(xyz[1] - xyz[0], np.cross(xyz[2] - xyz[0], xyz[3] - xyz[0])))) / 6.0
            tetra_volumes.append(volume)
    if parsed_elements != element_count or len(nodes) != node_count:
        raise ParametricGeometryError("Gmsh node or element count does not match its section header")
    if not tetra_volumes:
        raise ParametricGeometryError("Gmsh mesh contains no tetrahedral gas elements")
    volumes = np.asarray(tetra_volumes, dtype=float)
    total_volume = float(np.sum(volumes))
    tolerance = max(1.0e-9, 1.0e-8 * expected_gas_volume_m3)
    checks = {
        "finite_positive_tetrahedra": bool(np.all(np.isfinite(volumes)) and np.all(volumes > 1.0e-16)),
        "gas_volume_within_tolerance": bool(abs(total_volume - expected_gas_volume_m3) <= tolerance),
        "inlet_boundary_present": triangle_physical_counts[2] > 0,
        "outlet_boundary_present": triangle_physical_counts[3] > 0,
        "spacecraft_boundary_present": triangle_physical_counts[4] > 0,
    }
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "mesh_fingerprint": digest,
        "n_nodes": node_count,
        "n_elements": element_count,
        "n_tetrahedra": len(volumes),
        "minimum_tetrahedron_volume_m3": float(np.min(volumes)),
        "maximum_tetrahedron_volume_m3": float(np.max(volumes)),
        "gas_volume_m3": total_volume,
        "expected_gas_volume_m3": expected_gas_volume_m3,
        "absolute_gas_volume_error_m3": abs(total_volume - expected_gas_volume_m3),
        "boundary_triangle_counts": {
            "IN": triangle_physical_counts[2],
            "OUT": triangle_physical_counts[3],
            "CYLINDER_HEX": triangle_physical_counts[4],
        },
    }


def build_gmsh_exterior_mesh(
    manifest_json: str | Path,
    *,
    gmsh_executable: str = "gmsh",
) -> Dict[str, Any]:
    manifest_path = Path(manifest_json).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    executable = shutil.which(gmsh_executable)
    if executable is None:
        raise ParametricGeometryError(f"Gmsh executable not found: {gmsh_executable}")
    geo_path = manifest_path.parent / manifest["assets"]["gmsh_exterior_geo"]
    mesh_path = manifest_path.parent / f"{manifest['geometry_id']}.exterior.msh"
    command = [executable, "-3", str(geo_path), "-o", str(mesh_path), "-format", "msh2", "-v", "2"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise ParametricGeometryError(
            f"Gmsh failed with exit code {result.returncode}: {(result.stderr or result.stdout).strip()}"
        )
    spec = CylinderHexSpec(**manifest["specification"])
    outer_volume = 2.5 * spec.total_length_m * 2.0 * spec.total_length_m * 2.0 * spec.total_length_m
    validation = validate_gmsh_msh2(
        mesh_path,
        expected_gas_volume_m3=outer_volume - spec.target_volume_m3,
    )
    if not validation["valid"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise ParametricGeometryError(f"Generated Gmsh mesh failed validation: {failed}")
    version = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True
    ).stdout.strip()
    manifest["assets"]["gmsh_exterior_msh"] = mesh_path.name
    manifest["assets"]["piclas_mesh_status"] = (
        "validated Gmsh exterior tetra mesh; HOPR/PICLas HDF5 conversion and ingestion remain required"
    )
    manifest["gmsh_exterior_mesh"] = {
        "gmsh_version": version,
        "command": [
            Path(value).name if value in {executable, str(geo_path), str(mesh_path)} else value
            for value in command
        ],
        "validation": validation,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "gmsh_exterior_mesh_validated",
        "manifest_json": str(manifest_path),
        "mesh_path": str(mesh_path),
        **validation,
    }


def write_pyhope_config(
    path: str | Path,
    *,
    project_name: str,
    gmsh_mesh: str,
    split_tetrahedra_to_hexahedra: bool = True,
) -> Path:
    """Write the PyHOPE configuration used to convert the exterior Gmsh mesh."""
    target = Path(path).resolve()
    lines = [
        f"ProjectName = {project_name}",
        "OutputFormat = HDF5",
        "Mode = external",
        f"Filename = {gmsh_mesh}",
        "NGeo = 1",
        "MeshSorting = SFC",
        f"doSplitToHex = {'T' if split_tetrahedra_to_hexahedra else 'F'}",
        "",
        "BoundaryName = IN",
        "BoundaryType = (/3,0,0,0/)",
        "BoundaryName = OUT",
        "BoundaryType = (/3,0,0,0/)",
        "BoundaryName = CYLINDER_HEX",
        "BoundaryType = (/4,0,0,0/)",
        "",
        "CheckElemJacobians = T",
        "CheckConnectivity = T",
        "CheckWatertightness = T",
        "CheckSurfaceNormals = T",
        "CheckInternalBoundaries = T",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def validate_piclas_hdf5_mesh(path: str | Path) -> Dict[str, Any]:
    """Perform cheap structural checks on a PyHOPE/HOPR PICLas mesh."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the meshing environment
        raise ParametricGeometryError("h5py is required to validate the PICLas HDF5 mesh") from exc

    source = Path(path).resolve()
    required_datasets = {"BCNames", "BCType", "ElemInfo", "NodeCoords", "SideInfo"}
    with h5py.File(source, "r") as mesh_file:
        dataset_names = set(mesh_file.keys())
        missing = sorted(required_datasets - dataset_names)
        if missing:
            raise ParametricGeometryError(f"PICLas HDF5 mesh is missing datasets: {missing}")
        boundary_names = []
        for value in mesh_file["BCNames"][...].reshape(-1):
            decoded = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            boundary_names.append(decoded.strip().upper())
        node_coordinates = np.asarray(mesh_file["NodeCoords"][...], dtype=float)
        elem_info = np.asarray(mesh_file["ElemInfo"][...], dtype=int)
        side_info = np.asarray(mesh_file["SideInfo"][...], dtype=int)
        attributes = {
            key: int(mesh_file.attrs[key])
            for key in ("Ngeo", "nBCs", "nElems", "nNodes", "nSides", "nUniqueNodes", "nUniqueSides")
            if key in mesh_file.attrs
        }
        pyhope_version = str(mesh_file.attrs.get("PyHOPEVersion", "unknown"))

    expected_boundaries = {"IN", "OUT", "CYLINDER_HEX"}
    object_bc_ids = [index + 1 for index, name in enumerate(boundary_names) if name == "CYLINDER_HEX"]
    face_nodes = {
        1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [0, 1, 5, 4],
        4: [1, 2, 6, 5], 5: [2, 3, 7, 6], 6: [3, 0, 4, 7],
    }

    def resolve_side(side_id: int) -> np.ndarray | None:
        current = abs(int(side_id))
        seen: set[int] = set()
        for _ in range(32):
            if current <= 0 or current > len(side_info) or current in seen:
                return None
            seen.add(current)
            row = side_info[current - 1]
            if int(row[2]) > 0 and int(row[3]) // 10 in face_nodes:
                return row
            current = abs(int(row[1]))
        return None

    twice_projected_area = 0.0
    for boundary_row in side_info[np.isin(side_info[:, 4], object_bc_ids)]:
        side_row = resolve_side(int(boundary_row[1]))
        if side_row is None:
            continue
        elem_index = int(side_row[2]) - 1
        face_id = int(side_row[3]) // 10
        if elem_index < 0 or elem_index >= len(elem_info):
            continue
        start, stop = int(elem_info[elem_index, 4]), int(elem_info[elem_index, 5])
        element_nodes = node_coordinates[start:stop]
        if len(element_nodes) < 8:
            continue
        points = element_nodes[face_nodes[face_id]]
        area_vector = np.zeros(3, dtype=float)
        for point_index in range(1, len(points) - 1):
            area_vector += 0.5 * np.cross(
                points[point_index] - points[0], points[point_index + 1] - points[0]
            )
        twice_projected_area += abs(float(area_vector[0]))
    projected_reference_area = 0.5 * twice_projected_area
    checks = {
        "required_datasets_present": not missing,
        "finite_node_coordinates": bool(node_coordinates.size and np.all(np.isfinite(node_coordinates))),
        "positive_element_count": attributes.get("nElems", 0) > 0,
        "expected_boundaries_present": expected_boundaries.issubset(set(boundary_names)),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "mesh_fingerprint": hashlib.sha256(source.read_bytes()).hexdigest(),
        "boundary_names": boundary_names,
        "pyhope_version": pyhope_version,
        "x_projected_reference_area_m2": float(projected_reference_area),
        **attributes,
    }


def parse_pyhope_scaled_jacobian_histogram(output: str) -> Dict[str, int]:
    """Extract PyHOPE's element-quality histogram from ANSI-formatted output."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", str(output))
    labels = ("<0.0", "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", ">0.9-1.0")
    histogram: Dict[str, int] = {}
    for line in clean.splitlines():
        fields = [field.strip() for field in line.split("│") if field.strip()]
        if not fields or fields[0] not in labels:
            continue
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", fields[-1])
        if numbers:
            histogram[fields[0]] = int(round(float(numbers[-1])))
    return histogram


def build_piclas_hdf5_mesh(
    manifest_json: str | Path,
    *,
    pyhope_executable: str = "pyhope",
    split_tetrahedra_to_hexahedra: bool = True,
) -> Dict[str, Any]:
    """Convert a validated Gmsh mesh to PICLas HDF5 with PyHOPE."""
    manifest_path = Path(manifest_json).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gmsh_name = manifest.get("assets", {}).get("gmsh_exterior_msh")
    if not gmsh_name:
        raise ParametricGeometryError("Manifest has no validated gmsh_exterior_msh asset")
    gmsh_path = manifest_path.parent / gmsh_name
    if not gmsh_path.is_file():
        raise ParametricGeometryError(f"Gmsh input mesh not found: {gmsh_path}")
    executable = shutil.which(pyhope_executable)
    if executable is None:
        raise ParametricGeometryError(f"PyHOPE executable not found: {pyhope_executable}")

    geometry_id = str(manifest["geometry_id"])
    config_path = write_pyhope_config(
        manifest_path.parent / f"{geometry_id}.pyhope.ini",
        project_name=geometry_id,
        gmsh_mesh=gmsh_path.name,
        split_tetrahedra_to_hexahedra=split_tetrahedra_to_hexahedra,
    )
    command = [executable, config_path.name]
    result = subprocess.run(
        command,
        cwd=manifest_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise ParametricGeometryError(f"PyHOPE failed with exit code {result.returncode}: {diagnostic}")

    mesh_path = manifest_path.parent / f"{geometry_id}_mesh.h5"
    if not mesh_path.is_file():
        raise ParametricGeometryError(f"PyHOPE did not create the expected mesh: {mesh_path}")
    validation = validate_piclas_hdf5_mesh(mesh_path)
    if not validation["valid"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise ParametricGeometryError(f"PyHOPE mesh failed structural validation: {failed}")

    jacobian_histogram = parse_pyhope_scaled_jacobian_histogram(result.stdout)
    validation["scaled_jacobian_histogram"] = jacobian_histogram
    validation["negative_scaled_jacobian_elements"] = int(jacobian_histogram.get("<0.0", 0))
    validation["low_quality_scaled_jacobian_elements_0_to_0p1"] = int(
        jacobian_histogram.get("0.0-0.1", 0)
    )
    version = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True
    ).stdout.strip()
    manifest["assets"]["pyhope_config"] = config_path.name
    manifest["assets"]["piclas_volume_mesh"] = mesh_path.name
    manifest["assets"]["piclas_mesh_status"] = (
        "PyHOPE conversion and structural validation passed; PICLas smoke-run validation remains required"
    )
    manifest["pyhope_conversion"] = {
        "pyhope_version": version or validation["pyhope_version"],
        "command": [Path(executable).name, config_path.name],
        "split_tetrahedra_to_hexahedra": bool(split_tetrahedra_to_hexahedra),
        "validation": validation,
    }
    manifest["campaign_geometry_descriptor"]["metadata"]["hf_mesh"] = mesh_path.name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "piclas_hdf5_mesh_validated",
        "manifest_json": str(manifest_path),
        "mesh_path": str(mesh_path),
        **validation,
    }
