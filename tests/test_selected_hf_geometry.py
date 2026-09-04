from __future__ import annotations

import json
from pathlib import Path

from mfmc_campaign.geometry_design import build_cylinder_hex_design
from mfmc_campaign.selected_hf_geometry import (
    _workflow_config,
    build_sequential_hf_suite,
    build_selected_hf_suite,
    select_common_hf_samples,
)


def test_common_hf_sample_selection_is_deterministic_and_space_filling() -> None:
    samples = [
        {"x": float(index), "y": float((index * 7) % 11), "density_scale": 10.0}
        for index in range(20)
    ]
    first = select_common_hf_samples(samples, 5)
    second = select_common_hf_samples(samples, 5)
    assert first == second
    assert len(first) == len(set(first)) == 5


def test_hf_workflow_uses_lf_regime_common_samples_and_reference_area() -> None:
    base = {
        "adapter": {"kwargs": {"payload_defaults": {"density_scale": 10.0}}},
        "request": {"geometry": {}, "regime": {}, "metadata": {}},
    }
    lf = {
        "regime": {"id": "CUBE_300KM", "descriptors": {"altitude_km": 300.0}},
        "metadata": {"environment_model": "pymsis_hwm14", "density_scale": 10.0},
        "samples": [
            {"density_state_scale": 0.9, "density_scale": 10.0},
            {"density_state_scale": 1.1, "density_scale": 10.0},
        ],
    }
    config = _workflow_config(
        base,
        geometry_id="cylinder_hex_wp5_008",
        mesh_reference="geometry/cylinder_hex_wp5/L1/cylinder_hex_wp5_008/cylinder_hex_wp5_008_mesh.h5",
        reference_area_m2=0.0017,
        lf_config=lf,
        sample_indices=[1],
    )
    request = config["request"]
    assert request["regime"]["descriptors"]["altitude_km"] == 300.0
    assert request["geometry"]["metadata"]["reference_area_m2"] == 0.0017
    assert request["samples"][0]["density_state_scale"] == 1.1
    assert request["samples"][0]["density_scale"] == 10.0
    assert request["metadata"]["flow_zero_direction"] == [1.0, 0.0, 0.0]


def test_suite_builds_only_selected_nonvalidation_geometries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "piclas").mkdir()
    design = build_cylinder_hex_design(
        tmp_path / "outputs" / "design", n_designs=8, n_validation=2, maximin_trials=4
    )
    eligible = [row for row in design["designs"] if row["eligible_for_model_fitting"]][:2]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected": [
                    {"geometry_id": row["geometry_id"], "selection_order": index, "selection_basis": "test"}
                    for index, row in enumerate(eligible)
                ],
                "untouched_validation_geometry_ids": design["validation_geometry_ids"],
            }
        )
    )
    lf_config_path = tmp_path / "lf.json"
    lf_config_path.write_text(
        json.dumps(
            {
                "regime": {"id": "CUBE_300KM", "descriptors": {"altitude_km": 300.0}},
                "metadata": {"environment_model": "pymsis_hwm14", "density_scale": 10.0},
                "samples": [{"x": float(index), "density_scale": 10.0} for index in range(10)],
            }
        )
    )
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"adapter": {"kwargs": {}}, "request": {}}))

    def fake_gmsh(manifest_path, **_kwargs):
        manifest = json.loads(Path(manifest_path).read_text())
        mesh = Path(manifest_path).parent / f"{manifest['geometry_id']}.exterior.msh"
        mesh.write_text("mock")
        current = json.loads(Path(manifest_path).read_text())
        current["assets"]["gmsh_exterior_msh"] = mesh.name
        Path(manifest_path).write_text(json.dumps(current))
        return {"n_tetrahedra": 100}

    def fake_h5(manifest_path, **_kwargs):
        manifest = json.loads(Path(manifest_path).read_text())
        mesh = Path(manifest_path).parent / f"{manifest['geometry_id']}_mesh.h5"
        mesh.write_bytes(b"mock-h5")
        return {"mesh_path": str(mesh), "nElems": 400, "mesh_fingerprint": "abc"}

    monkeypatch.setattr("mfmc_campaign.selected_hf_geometry.build_gmsh_exterior_mesh", fake_gmsh)
    monkeypatch.setattr("mfmc_campaign.selected_hf_geometry.build_piclas_hdf5_mesh", fake_h5)
    suite = build_selected_hf_suite(
        selection_path,
        design["manifest_json"],
        lf_config_path,
        output_root=tmp_path / "piclas" / "geometry" / "wp5" / "L1",
        config_output_dir=tmp_path / "configs",
        base_config_json=base_path,
        hf_samples_per_geometry=3,
    )
    assert suite["total_hf_runs"] == 6
    assert [row["geometry_id"] for row in suite["geometries"]] == [row["geometry_id"] for row in eligible]
    assert not set(design["validation_geometry_ids"]).intersection(
        row["geometry_id"] for row in suite["geometries"]
    )
    assert all((tmp_path / row["workflow_config"]).is_file() for row in suite["geometries"])


def test_sequential_suite_reuses_mesh_and_inserts_acquired_samples(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs" / "initial"
    config_dir.mkdir(parents=True)
    geometry_ids = ["cylinder_hex_wp5_000", "cylinder_hex_wp5_008"]
    initial_rows = []
    for geometry_id in geometry_ids:
        config_path = config_dir / f"{geometry_id}.json"
        config_path.write_text(json.dumps({
            "request": {
                "geometry": {"id": geometry_id, "tags": ["initial"]},
                "metadata": {"case_name": "initial"},
                "sample_ids": ["hf-crn-0001"],
                "samples": [{"x": 1.0}],
            }
        }))
        initial_rows.append({
            "geometry_id": geometry_id,
            "workflow_config": str(config_path.relative_to(tmp_path)),
            "mesh_reference": f"geometry/{geometry_id}_mesh.h5",
        })
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({
        "mesh_level": "L1", "untouched_validation_geometry_ids": ["validation"],
        "geometries": initial_rows,
    }))
    selected_ids = ["wp1-crn-0003", "wp1-crn-0007"]
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({"geometries": [
        {"geometry_id": geometry_id, "selected_canonical_sample_ids": selected_ids}
        for geometry_id in geometry_ids
    ]}))
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"uncertainty_samples": {
        "wp1-crn-0003": {"x": 3.0}, "wp1-crn-0007": {"x": 7.0},
    }}))
    suite = build_sequential_hf_suite(
        initial, acquisition, bundle,
        config_output_dir=tmp_path / "configs" / "next",
        suite_output_json=tmp_path / "next_suite.json",
    )
    assert suite["total_hf_runs"] == 4
    assert suite["common_canonical_sample_ids"] == selected_ids
    generated = json.loads((tmp_path / suite["geometries"][0]["workflow_config"]).read_text())
    assert generated["request"]["sample_ids"] == ["hf-crn-0003", "hf-crn-0007"]
    assert [row["x"] for row in generated["request"]["samples"]] == [3.0, 7.0]
