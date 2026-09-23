from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .parametric_geometry import (
    ParametricGeometryError,
    SurfaceMesh,
    surface_mesh_from_points,
    write_pyhope_config,
)


GENERATOR_NAME = "mfmc-paper1-benchmark"
GENERATOR_VERSION = "1.0.0"
BODY_BOUNDARY = "BENCHMARK_BODY"


@dataclass(frozen=True)
class BenchmarkGeometry:
    geometry_id: str
    kind: str
    mesh: SurfaceMesh
    face_region: np.ndarray
    region_names: tuple[str, ...]
    diameter_m: float
    reference_area_m2: float
    solid_volume_m3: float
    opening_normal: tuple[float, float, float] | None = None


def _signed_volume(mesh: SurfaceMesh) -> float:
    xyz = mesh.points[mesh.triangles]
    return float(np.sum(np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2]))) / 6.0)


def _edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.sort(
        np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
        axis=1,
    )
    return np.unique(edges, axis=0, return_counts=True)


def _orient_by_expected(points: np.ndarray, triangles: np.ndarray, expected: np.ndarray) -> np.ndarray:
    result = np.asarray(triangles, dtype=np.int64).copy()
    xyz = points[result]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    flip = np.einsum("ij,ij->i", normals, expected) < 0.0
    result[flip, 1], result[flip, 2] = result[flip, 2].copy(), result[flip, 1].copy()
    return result


def make_sphere(*, diameter_m: float = 0.1, n_theta: int = 12, n_phi: int = 24) -> BenchmarkGeometry:
    if n_theta < 4 or n_phi < 8 or n_phi % 2:
        raise ParametricGeometryError("sphere resolution requires n_theta>=4 and even n_phi>=8")
    radius = 0.5 * float(diameter_m)
    points: list[list[float]] = [[radius, 0.0, 0.0]]
    for i in range(1, n_theta):
        theta = math.pi * i / n_theta
        for j in range(n_phi):
            phi = 2.0 * math.pi * j / n_phi
            points.append(
                [radius * math.cos(theta), radius * math.sin(theta) * math.cos(phi), radius * math.sin(theta) * math.sin(phi)]
            )
    points.append([-radius, 0.0, 0.0])
    north, south = 0, len(points) - 1
    triangles: list[list[int]] = []
    for j in range(n_phi):
        triangles.append([north, 1 + j, 1 + (j + 1) % n_phi])
    for i in range(n_theta - 2):
        a = 1 + i * n_phi
        b = a + n_phi
        for j in range(n_phi):
            k = (j + 1) % n_phi
            triangles.extend(([a + j, b + j, b + k], [a + j, b + k, a + k]))
    last = 1 + (n_theta - 2) * n_phi
    for j in range(n_phi):
        triangles.append([last + j, south, last + (j + 1) % n_phi])
    mesh = surface_mesh_from_points(np.asarray(points), np.asarray(triangles))
    return BenchmarkGeometry(
        geometry_id="sphere_d0p1",
        kind="sphere",
        mesh=mesh,
        face_region=np.ones(len(mesh.triangles), dtype=np.int64),
        region_names=("outer",),
        diameter_m=diameter_m,
        reference_area_m2=math.pi * diameter_m**2 / 4.0,
        solid_volume_m3=4.0 * math.pi * radius**3 / 3.0,
    )


def make_hemisphere_shell(
    *, outer_radius_m: float = 0.05, inner_radius_m: float = 0.048, n_theta: int = 12, n_phi: int = 24
) -> BenchmarkGeometry:
    """Closed finite-thickness shell; aperture is at x=0 with outward normal -x."""
    if not 0.0 < inner_radius_m < outer_radius_m:
        raise ParametricGeometryError("shell radii must satisfy 0 < inner < outer")
    if n_theta < 3 or n_phi < 8 or n_phi % 2:
        raise ParametricGeometryError("shell resolution requires n_theta>=3 and even n_phi>=8")
    points: list[list[float]] = []

    def add_hemi(radius: float) -> tuple[int, list[list[int]]]:
        start = len(points)
        points.append([radius, 0.0, 0.0])
        for i in range(1, n_theta + 1):
            theta = 0.5 * math.pi * i / n_theta
            for j in range(n_phi):
                phi = 2.0 * math.pi * j / n_phi
                points.append(
                    [radius * math.cos(theta), radius * math.sin(theta) * math.cos(phi), radius * math.sin(theta) * math.sin(phi)]
                )
        faces: list[list[int]] = []
        for j in range(n_phi):
            faces.append([start, start + 1 + j, start + 1 + (j + 1) % n_phi])
        for i in range(n_theta - 1):
            a = start + 1 + i * n_phi
            b = a + n_phi
            for j in range(n_phi):
                k = (j + 1) % n_phi
                faces.extend(([a + j, b + j, b + k], [a + j, b + k, a + k]))
        return start + 1 + (n_theta - 1) * n_phi, faces

    outer_rim, outer_faces = add_hemi(outer_radius_m)
    inner_rim, inner_faces = add_hemi(inner_radius_m)
    point_array = np.asarray(points, dtype=float)
    outer = np.asarray(outer_faces, dtype=np.int64)
    inner = np.asarray(inner_faces, dtype=np.int64)
    outer_centers = np.mean(point_array[outer], axis=1)
    inner_centers = np.mean(point_array[inner], axis=1)
    outer = _orient_by_expected(point_array, outer, outer_centers)
    inner = _orient_by_expected(point_array, inner, -inner_centers)
    rim: list[list[int]] = []
    for j in range(n_phi):
        k = (j + 1) % n_phi
        rim.extend(
            ([outer_rim + j, outer_rim + k, inner_rim + k], [outer_rim + j, inner_rim + k, inner_rim + j])
        )
    rim_array = _orient_by_expected(
        point_array, np.asarray(rim, dtype=np.int64), np.tile([-1.0, 0.0, 0.0], (len(rim), 1))
    )
    triangles = np.vstack((outer, inner, rim_array))
    # Do not use centroid-based orientation: it reverses the concave inner wall.
    from .parametric_geometry import _fingerprint, _triangle_geometry

    area, normal, center = _triangle_geometry(point_array, triangles)
    mesh = SurfaceMesh(point_array, triangles, area, normal, center, _fingerprint(point_array, triangles))
    regions = np.concatenate(
        (np.full(len(outer), 1), np.full(len(inner), 2), np.full(len(rim_array), 3))
    ).astype(np.int64)
    return BenchmarkGeometry(
        geometry_id="hemisphere_shell_ro0p05_ri0p048",
        kind="hemisphere_shell",
        mesh=mesh,
        face_region=regions,
        region_names=("outer", "inner", "rim"),
        diameter_m=2.0 * outer_radius_m,
        reference_area_m2=math.pi * outer_radius_m**2,
        solid_volume_m3=2.0 * math.pi * (outer_radius_m**3 - inner_radius_m**3) / 3.0,
        opening_normal=(-1.0, 0.0, 0.0),
    )


def validate_geometry(geometry: BenchmarkGeometry) -> dict[str, Any]:
    mesh = geometry.mesh
    unique_edges, counts = _edges(mesh.triangles)
    sorted_faces = np.sort(mesh.triangles, axis=1)
    duplicate_faces = len(sorted_faces) - len(np.unique(sorted_faces, axis=0))
    signed_volume = _signed_volume(mesh)
    projected_x = 0.5 * float(np.sum(mesh.triangle_area * np.abs(mesh.triangle_normal[:, 0])))
    bbox_min = np.min(mesh.points, axis=0)
    bbox_max = np.max(mesh.points, axis=0)
    normal_lengths = np.linalg.norm(mesh.triangle_normal, axis=1)
    checks = {
        "finite": bool(np.all(np.isfinite(mesh.points)) and np.all(np.isfinite(mesh.triangle_area))),
        "nondegenerate_triangles": bool(np.all(mesh.triangle_area > 1.0e-14)),
        "no_duplicate_faces": duplicate_faces == 0,
        "watertight_two_manifold": bool(np.all(counts == 2)),
        "outward_closed_surface_orientation": signed_volume > 0.0,
        "unit_normals": bool(np.allclose(normal_lengths, 1.0, atol=1.0e-12)),
        "positive_solid_volume": geometry.solid_volume_m3 > 0.0,
        "projected_area_matches_reference": bool(np.isclose(projected_x, geometry.reference_area_m2, rtol=0.03)),
    }
    if geometry.kind == "hemisphere_shell":
        inner = geometry.face_region == 2
        rim = geometry.face_region == 3
        checks["inner_normals_face_cavity"] = bool(
            np.all(np.einsum("ij,ij->i", mesh.triangle_normal[inner], mesh.triangle_center[inner]) < 0.0)
        )
        checks["rim_normals_face_aperture"] = bool(np.all(mesh.triangle_normal[rim, 0] < -0.999999))
        checks["cavity_aperture_unobstructed"] = bool(np.all(mesh.triangle_center[:, 0] >= -1.0e-12))
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "n_vertices": int(len(mesh.points)),
        "n_triangles": int(len(mesh.triangles)),
        "n_unique_edges": int(len(unique_edges)),
        "euler_characteristic": int(len(mesh.points) - len(unique_edges) + len(mesh.triangles)),
        "duplicate_face_count": int(duplicate_faces),
        "surface_area_m2": float(np.sum(mesh.triangle_area)),
        "signed_solid_volume_m3": signed_volume,
        "analytic_solid_volume_m3": geometry.solid_volume_m3,
        "projected_x_area_m2": projected_x,
        "bounds_min_m": bbox_min.tolist(),
        "bounds_max_m": bbox_max.tolist(),
        "region_triangle_counts": {
            name: int(np.count_nonzero(geometry.face_region == index))
            for index, name in enumerate(geometry.region_names, start=1)
        },
    }


def _write_obj(path: Path, geometry: BenchmarkGeometry) -> None:
    lines = [f"# {GENERATOR_NAME} {GENERATOR_VERSION}", f"o {geometry.geometry_id}"]
    lines.extend(f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in geometry.mesh.points)
    previous = None
    for region, triangle in zip(geometry.face_region, geometry.mesh.triangles):
        if int(region) != previous:
            lines.append(f"usemtl {int(region)}")
            previous = int(region)
        lines.append("f " + " ".join(str(int(index) + 1) for index in triangle))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_stl(path: Path, geometry: BenchmarkGeometry) -> None:
    lines = [f"solid {geometry.geometry_id}"]
    for normal, triangle in zip(geometry.mesh.triangle_normal, geometry.mesh.triangles):
        lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
        lines.append("    outer loop")
        lines.extend(
            f"      vertex {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
            for point in geometry.mesh.points[triangle]
        )
        lines.extend(("    endloop", "  endfacet"))
    lines.append(f"endsolid {geometry.geometry_id}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_mat(path: Path, geometry: BenchmarkGeometry) -> None:
    from scipy.io import savemat

    xyz = geometry.mesh.points[geometry.mesh.triangles]
    savemat(
        path,
        {
            "meshdata": {
                "XData": xyz[:, :, 0].T,
                "YData": xyz[:, :, 1].T,
                "ZData": xyz[:, :, 2].T,
                "MatID": geometry.face_region,
                "Areas": geometry.mesh.triangle_area,
                "SurfN": geometry.mesh.triangle_normal.T,
                "BariC": geometry.mesh.triangle_center.T,
                "Lref": geometry.diameter_m,
                "Aref": geometry.reference_area_m2,
            },
            "region_names": np.asarray(geometry.region_names, dtype=object),
            "mesh_fingerprint": np.asarray([geometry.mesh.mesh_fingerprint]),
        },
    )


def _write_geo(
    path: Path,
    geometry: BenchmarkGeometry,
    *,
    body_mesh_size_m: float,
    farfield_mesh_size_m: float,
    domain_scale: float,
) -> float:
    mesh = geometry.mesh
    d = geometry.diameter_m
    x_min, x_max = -domain_scale * d, domain_scale * d
    transverse = domain_scale * d
    lines = [
        f"// {GENERATOR_NAME} {GENERATOR_VERSION}",
        'SetFactory("Built-in");',
        f"lcBody = {body_mesh_size_m:.17g};",
        f"lcFar = {farfield_mesh_size_m:.17g};",
    ]
    for tag, point in enumerate(mesh.points, start=1):
        lines.append(f"Point({tag}) = {{{point[0]:.17g}, {point[1]:.17g}, {point[2]:.17g}, lcBody}};")
    edge_tags: dict[tuple[int, int], int] = {}
    face_edges: list[list[int]] = []
    next_line = 1
    for triangle in mesh.triangles:
        directed: list[int] = []
        for a0, b0 in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            a, b = int(a0) + 1, int(b0) + 1
            key = (min(a, b), max(a, b))
            if key not in edge_tags:
                edge_tags[key] = next_line
                lines.append(f"Line({next_line}) = {{{key[0]}, {key[1]}}};")
                next_line += 1
            directed.append(edge_tags[key] if (a, b) == key else -edge_tags[key])
        face_edges.append(directed)
    body_surfaces: list[int] = []
    next_loop = 1
    next_surface = 1
    for edges in face_edges:
        lines.append(f"Curve Loop({next_loop}) = {{{', '.join(map(str, edges))}}};")
        lines.append(f"Plane Surface({next_surface}) = {{{next_loop}}};")
        body_surfaces.append(next_surface)
        next_loop += 1
        next_surface += 1
    lines.append(f"Surface Loop(1) = {{{', '.join(map(str, body_surfaces))}}};")
    first = len(mesh.points) + 1
    outer_points = [
        (x_min, -transverse, -transverse), (x_max, -transverse, -transverse),
        (x_max, transverse, -transverse), (x_min, transverse, -transverse),
        (x_min, -transverse, transverse), (x_max, -transverse, transverse),
        (x_max, transverse, transverse), (x_min, transverse, transverse),
    ]
    for offset, point in enumerate(outer_points):
        lines.append(f"Point({first + offset}) = {{{point[0]:.17g}, {point[1]:.17g}, {point[2]:.17g}, lcFar}};")
    p = [first + i for i in range(8)]
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    outer_lines: list[int] = []
    for a, b in pairs:
        outer_lines.append(next_line)
        lines.append(f"Line({next_line}) = {{{p[a]}, {p[b]}}};")
        next_line += 1
    l = outer_lines
    loops = ([l[0], l[1], l[2], l[3]], [l[4], l[5], l[6], l[7]], [l[0], l[9], -l[4], -l[8]], [l[1], l[10], -l[5], -l[9]], [l[2], l[11], -l[6], -l[10]], [l[3], l[8], -l[7], -l[11]])
    outer_surfaces: list[int] = []
    for edges in loops:
        lines.append(f"Curve Loop({next_loop}) = {{{', '.join(map(str, edges))}}};")
        lines.append(f"Plane Surface({next_surface}) = {{{next_loop}}};")
        outer_surfaces.append(next_surface)
        next_loop += 1
        next_surface += 1
    lines.append(f"Surface Loop(2) = {{{', '.join(map(str, outer_surfaces))}}};")
    lines.extend(
        (
            "Volume(1) = {2, 1};",
            'Physical Volume("GAS", 1) = {1};',
            f'Physical Surface("IN", 2) = {{{outer_surfaces[5]}}};',
            f'Physical Surface("OUT", 3) = {{{", ".join(map(str, outer_surfaces[:5]))}}};',
            f'Physical Surface("{BODY_BOUNDARY}", 4) = {{{", ".join(map(str, body_surfaces))}}};',
            "Mesh.Algorithm = 6;", "Mesh.Algorithm3D = 1;", "Mesh.RecombineAll = 0;", "Mesh.SubdivisionAlgorithm = 0;", "Mesh.Optimize = 1;",
            "Mesh.MshFileVersion = 2.2;", "Mesh.Binary = 0;",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return (x_max - x_min) * (2.0 * transverse) ** 2 - _signed_volume(geometry.mesh)


def write_assets(
    output_dir: str | Path,
    geometry: BenchmarkGeometry,
    *,
    resolution: str,
    body_mesh_size_m: float,
    farfield_mesh_size_m: float,
    domain_scale: float = 2.5,
) -> Path:
    validation = validate_geometry(geometry)
    if not validation["valid"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise ParametricGeometryError(f"benchmark geometry failed validation: {failed}")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{geometry.geometry_id}_{resolution}"
    obj, stl, mat = (target / f"{stem}{suffix}" for suffix in (".obj", ".stl", ".mat"))
    surface, geo, ini = target / f"{stem}.surface.npz", target / f"{stem}.exterior.geo", target / f"{stem}.pyhope.ini"
    _write_obj(obj, geometry)
    _write_stl(stl, geometry)
    _write_mat(mat, geometry)
    expected_gas_volume = _write_geo(
        geo, geometry, body_mesh_size_m=body_mesh_size_m, farfield_mesh_size_m=farfield_mesh_size_m, domain_scale=domain_scale
    )
    write_pyhope_config(
        ini, project_name=stem, gmsh_mesh=f"{stem}.exterior.msh", split_tetrahedra_to_hexahedra=True
    )
    # The generic PICLas adapter supports an explicit object boundary name.
    ini.write_text(ini.read_text(encoding="ascii").replace("CYLINDER_HEX", BODY_BOUNDARY), encoding="ascii")
    np.savez_compressed(
        surface,
        points=geometry.mesh.points,
        triangles=geometry.mesh.triangles,
        triangle_area=geometry.mesh.triangle_area,
        triangle_normal=geometry.mesh.triangle_normal,
        triangle_center=geometry.mesh.triangle_center,
        face_region=geometry.face_region,
        region_names=np.asarray(geometry.region_names),
        mesh_fingerprint=np.asarray([geometry.mesh.mesh_fingerprint]),
    )
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "geometry_id": geometry.geometry_id,
        "geometry_kind": geometry.kind,
        "resolution": resolution,
        "units": "m",
        "diameter_m": geometry.diameter_m,
        "characteristic_length_m": geometry.diameter_m,
        "reference_area_m2": geometry.reference_area_m2,
        "reference_area_convention": "pi*D^2/4, projected normal to body x-axis",
        "coordinate_system": {"nominal_molecular_velocity": "+x", "transverse_axes": ["y", "z"], "origin": "sphere/shell center"},
        "opening_normal": geometry.opening_normal,
        "nominal_orientation": "cup-forward" if geometry.opening_normal else "orientation-invariant",
        "surface_regions": {str(i): name for i, name in enumerate(geometry.region_names, 1)},
        "surface_validation": validation,
        "mesh_controls": {
            "body_mesh_size_m": body_mesh_size_m,
            "farfield_mesh_size_m": farfield_mesh_size_m,
            "domain_half_extent_in_diameters": domain_scale,
            "expected_gas_volume_m3": expected_gas_volume,
        },
        "assets": {
            "adbsat_obj": obj.name, "adbsat_mat": mat.name, "surface_stl": stl.name,
            "canonical_surface_npz": surface.name, "gmsh_exterior_geo": geo.name,
            "gmsh_exterior_msh": None, "pyhope_config": ini.name, "piclas_volume_mesh": None,
        },
        "piclas_boundary_mapping": {"IN": "open/inflow", "OUT": "open/outflow", BODY_BOUNDARY: "reflective spacecraft wall"},
        "status": "surface_statically_validated; volume generation pending",
    }
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = target / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
