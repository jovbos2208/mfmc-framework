from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mfmc_campaign.geometry_control_nodes import (
    CONTROL_NODE_VARIABLES,
    build_control_node_geometry_assets,
    deform_cylinder_hex_control_nodes,
    generate_control_node_refinement_manifest,
)
from mfmc_campaign.parametric_geometry import CylinderHexSpec


def test_zero_control_nodes_preserve_volume_and_symmetry() -> None:
    spec = CylinderHexSpec()
    mesh, metadata = deform_cylinder_hex_control_nodes(spec, np.zeros(len(CONTROL_NODE_VARIABLES)))
    assert metadata["validation"]["valid"]
    assert np.isclose(metadata["validation"]["signed_volume_m3"], spec.target_volume_m3)
    reflected_y = {tuple(np.round([x, -y, z], 12)) for x, y, z in mesh.points}
    reflected_z = {tuple(np.round([x, y, -z], 12)) for x, y, z in mesh.points}
    points = {tuple(np.round(point, 12)) for point in mesh.points}
    assert reflected_y == points
    assert reflected_z == points


def test_control_node_asset_manifest_records_actual_nodes(tmp_path: Path) -> None:
    parameters = {name: 0.0 for name in CONTROL_NODE_VARIABLES}
    parameters["shoulder_width_log_scale"] = 0.05
    result = build_control_node_geometry_assets(
        tmp_path, CylinderHexSpec(), parameters, design_id="control_test"
    )
    manifest = json.loads(Path(result["manifest_json"]).read_text())
    assert manifest["parameterization"] == "symmetric_surface_control_nodes"
    assert manifest["validation"]["valid"]
    assert manifest["control_node_deformation"]["control_nodes_before"] != manifest[
        "control_node_deformation"
    ]["control_nodes_after"]
    for asset in ("adbsat_obj", "adbsat_mat", "canonical_surface_npz", "gmsh_exterior_geo"):
        assert (tmp_path / manifest["assets"][asset]).is_file()


def test_control_node_poll_is_deterministic_and_bounded(tmp_path: Path) -> None:
    first = generate_control_node_refinement_manifest(
        tmp_path / "first" / "candidates.json", count=6, iteration=2
    )
    second = generate_control_node_refinement_manifest(
        tmp_path / "second" / "candidates.json", count=6, iteration=2
    )
    assert [row["parameters"] for row in first["designs"]] == [
        row["parameters"] for row in second["designs"]
    ]
    assert all(row["validation"]["valid"] for row in first["designs"])
    assert all(
        0.0 <= value <= 1.0
        for row in first["designs"]
        for value in row["normalized_parameters"]
    )


def test_first_control_node_poll_includes_undeformed_baseline(tmp_path: Path) -> None:
    result = generate_control_node_refinement_manifest(
        tmp_path / "candidates.json", count=6, iteration=1, include_center=True
    )
    baseline = result["candidates"][0]
    assert baseline["selection_basis"] == "control_node_baseline"
    assert all(value == 0.0 for value in baseline["parameters"].values())
    assert baseline["validation"]["valid"]
