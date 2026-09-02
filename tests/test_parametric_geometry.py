from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mfmc_campaign.parametric_geometry import (
    CylinderHexSpec,
    ParametricGeometryError,
    build_geometry_assets,
    build_gmsh_exterior_mesh,
    generate_cylinder_hex,
    parse_pyhope_scaled_jacobian_histogram,
    validate_piclas_hdf5_mesh,
    validate_surface,
)


@pytest.mark.parametrize(
    "spec",
    [
        CylinderHexSpec(),
        CylinderHexSpec(nose_length_fraction=0.05, tail_length_fraction=0.40, width_height_ratio=0.65, chamfer_fraction=0.02),
        CylinderHexSpec(nose_length_fraction=0.40, tail_length_fraction=0.05, width_height_ratio=1.55, chamfer_fraction=0.35),
    ],
)
def test_generated_geometry_is_watertight_convex_and_fixed_volume(spec: CylinderHexSpec) -> None:
    mesh, _derived = generate_cylinder_hex(spec)
    validation = validate_surface(mesh, spec)

    assert validation["valid"]
    assert validation["euler_characteristic"] == 2
    assert validation["absolute_volume_error_m3"] < 1.0e-12
    assert validation["checks"]["convex_no_self_intersection_certificate"]


def test_generation_is_byte_deterministic_and_does_not_require_source_mesh(tmp_path: Path) -> None:
    first = build_geometry_assets(tmp_path / "first", CylinderHexSpec(), design_id="baseline")
    second = build_geometry_assets(tmp_path / "second", CylinderHexSpec(), design_id="baseline")

    assert first["design_fingerprint"] == second["design_fingerprint"]
    assert first["mesh_fingerprint"] == second["mesh_fingerprint"]
    for suffix in ("obj", "stl", "mat", "surface.npz", "exterior.geo"):
        left = (tmp_path / "first" / f"baseline.{suffix}").read_bytes()
        right = (tmp_path / "second" / f"baseline.{suffix}").read_bytes()
        assert left == right
    manifest = json.loads((tmp_path / "first" / "baseline.manifest.json").read_text())
    assert manifest["provenance"]["external_mesh_reused"] is False
    assert manifest["assets"]["piclas_volume_mesh"] is None
    assert manifest["gmsh_mesh_controls"] == {
        "body_mesh_size_m": 0.06,
        "farfield_mesh_size_m": 0.30,
    }


def test_geometry_asset_mesh_controls_are_written_to_geo_and_manifest(tmp_path: Path) -> None:
    manifest = build_geometry_assets(
        tmp_path,
        CylinderHexSpec().scaled(0.1),
        design_id="refined",
        uniform_scale_factor=0.1,
        body_mesh_size_m=0.015,
        farfield_mesh_size_m=0.075,
    )

    geo = (tmp_path / "refined.exterior.geo").read_text(encoding="ascii")
    assert "lcBody = 0.014999999999999999;" in geo
    assert "lcFar = 0.074999999999999997;" in geo
    assert manifest["gmsh_mesh_controls"]["body_mesh_size_m"] == 0.015
    assert manifest["gmsh_mesh_controls"]["farfield_mesh_size_m"] == 0.075


def test_fixed_volume_solution_rejects_envelope_violation() -> None:
    with pytest.raises(ParametricGeometryError, match="exceeds envelope"):
        generate_cylinder_hex(CylinderHexSpec(target_volume_m3=0.5, max_width_m=0.3, max_height_m=0.3))


def test_design_parameters_change_fingerprint_but_not_volume() -> None:
    baseline, _ = generate_cylinder_hex(CylinderHexSpec())
    changed, _ = generate_cylinder_hex(CylinderHexSpec(chamfer_fraction=0.25))

    assert baseline.mesh_fingerprint != changed.mesh_fingerprint
    assert np.isclose(validate_surface(baseline, CylinderHexSpec())["signed_volume_m3"], 0.12)
    assert np.isclose(
        validate_surface(changed, CylinderHexSpec(chamfer_fraction=0.25))["signed_volume_m3"],
        0.12,
    )


def test_uniform_scaling_obeys_length_area_and_volume_laws() -> None:
    reference = CylinderHexSpec()
    scaled = reference.scaled(0.1)
    _mesh_reference, dimensions_reference = generate_cylinder_hex(reference)
    mesh_scaled, dimensions_scaled = generate_cylinder_hex(scaled)
    validation_scaled = validate_surface(mesh_scaled, scaled)

    assert scaled.total_length_m == pytest.approx(reference.total_length_m * 0.1)
    assert scaled.target_volume_m3 == pytest.approx(reference.target_volume_m3 * 0.001)
    assert dimensions_scaled["width_m"] == pytest.approx(dimensions_reference["width_m"] * 0.1)
    assert dimensions_scaled["height_m"] == pytest.approx(dimensions_reference["height_m"] * 0.1)
    assert dimensions_scaled["maximum_cross_section_area_m2"] == pytest.approx(
        dimensions_reference["maximum_cross_section_area_m2"] * 0.01
    )
    assert validation_scaled["signed_volume_m3"] == pytest.approx(0.00012)


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh executable unavailable")
def test_gmsh_exterior_mesh_has_gas_volume_and_named_boundaries(tmp_path: Path) -> None:
    manifest = build_geometry_assets(tmp_path, CylinderHexSpec(), design_id="gmsh_test")
    summary = build_gmsh_exterior_mesh(manifest["manifest_json"])
    repeated = build_gmsh_exterior_mesh(manifest["manifest_json"])

    assert summary["valid"]
    assert summary["n_tetrahedra"] > 0
    assert summary["boundary_triangle_counts"]["IN"] > 0
    assert summary["boundary_triangle_counts"]["OUT"] > 0
    assert summary["boundary_triangle_counts"]["CYLINDER_HEX"] > 0
    assert summary["absolute_gas_volume_error_m3"] < 1.0e-8
    assert summary["mesh_fingerprint"] == repeated["mesh_fingerprint"]


def test_piclas_hdf5_structural_validation(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    mesh_path = tmp_path / "geometry_mesh.h5"
    with h5py.File(mesh_path, "w") as mesh_file:
        mesh_file.attrs.update({"Ngeo": 1, "nBCs": 3, "nElems": 1, "nNodes": 8, "nSides": 6})
        mesh_file.attrs["PyHOPEVersion"] = "1.0.0"
        mesh_file.create_dataset("BCNames", data=np.asarray([b"in", b"out", b"cylinder_hex"], dtype="S255"))
        mesh_file.create_dataset("BCType", data=np.zeros((3, 4), dtype=np.int64))
        mesh_file.create_dataset("ElemInfo", data=np.zeros((1, 6), dtype=np.int64))
        mesh_file.create_dataset("NodeCoords", data=np.zeros((8, 3), dtype=float))
        mesh_file.create_dataset("SideInfo", data=np.zeros((6, 5), dtype=np.int64))

    report = validate_piclas_hdf5_mesh(mesh_path)

    assert report["valid"]
    assert report["boundary_names"] == ["IN", "OUT", "CYLINDER_HEX"]
    assert report["nElems"] == 1


def test_piclas_hdf5_reference_area_uses_element_owned_local_sides(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    mesh_path = tmp_path / "unit_cube_mesh.h5"
    node_coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    # GlobalSideID values deliberately do not point to SideInfo rows. HOPR
    # stores ownership through ElemInfo's side range, not through this ID.
    side_info = np.zeros((6, 5), dtype=np.int64)
    side_info[:, 1] = np.arange(101, 107)
    side_info[:, 4] = 3
    with h5py.File(mesh_path, "w") as mesh_file:
        mesh_file.attrs.update({"Ngeo": 1, "nBCs": 3, "nElems": 1, "nNodes": 8, "nSides": 6})
        mesh_file.create_dataset("BCNames", data=np.asarray([b"IN", b"OUT", b"CYLINDER_HEX"], dtype="S255"))
        mesh_file.create_dataset("BCType", data=np.zeros((3, 4), dtype=np.int64))
        mesh_file.create_dataset("ElemInfo", data=np.asarray([[118, 1, 0, 6, 0, 8]], dtype=np.int64))
        mesh_file.create_dataset("NodeCoords", data=node_coordinates)
        mesh_file.create_dataset("SideInfo", data=side_info)

    report = validate_piclas_hdf5_mesh(mesh_path)

    assert report["x_projected_reference_area_m2"] == pytest.approx(1.0)
    assert report["object_boundary_side_count"] == 6
    assert report["resolved_object_boundary_side_count"] == 6


def test_pyhope_scaled_jacobian_histogram_parser_handles_ansi_output() -> None:
    output = (
        "\x1b[1m│<0.0      │\x1b[0m 0.00\n"
        "\x1b[1m│ 0.0-0.1  │\x1b[0m ▇▇ 724.00\n"
        "\x1b[1m│>0.9-1.0  │\x1b[0m 2.00\n"
    )

    assert parse_pyhope_scaled_jacobian_histogram(output) == {
        "<0.0": 0,
        "0.0-0.1": 724,
        ">0.9-1.0": 2,
    }
