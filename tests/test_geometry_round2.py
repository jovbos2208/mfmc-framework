from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mfmc_campaign.geometry_design import VARIABLES
from mfmc_campaign.geometry_round2 import (
    _stage_generated_geometry,
    build_round2_piclas_suite,
    merge_round2_results,
    select_round2_geometries,
)
from mfmc_campaign.sparse_pce import SparsePCEModel
from scripts import run_cylinder_hex_round2_suite


def _linear_model(path: Path, *, coefficient: float, mean: float) -> None:
    model = SparsePCEModel(
        input_names=["geometry__nose_length_fraction"],
        active_input_indices=np.asarray([0]),
        input_mean=np.asarray([0.0]),
        input_scale=np.asarray([1.0]),
        multi_indices=np.asarray([[1]]),
        basis_mean=np.asarray([0.0]),
        basis_scale=np.asarray([1.0]),
        standardized_coefficients=np.asarray([coefficient]),
        output_mean=mean,
        output_scale=1.0,
        alpha=0.0,
        degree=1,
        q_norm=1.0,
        max_interaction=1,
    )
    model.write_json(str(path))


def test_round2_runner_keeps_operational_logs_out_of_json_stdout(
    tmp_path: Path, capsys,
) -> None:
    config = tmp_path / "workflow.json"
    config.write_text("{}")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({
        "geometries": [{
            "geometry_id": "geometry-0",
            "tpmc_workflow_config": config.name,
        }],
    }))

    def noisy_submit(_config, *, state_path):
        print("Job 123 submitted")
        return {"status": "submitted", "state_path": str(state_path)}

    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        with patch.object(
            sys,
            "argv",
            [
                "run_cylinder_hex_round2_suite.py",
                "submit",
                "--suite",
                str(suite),
                "--run-root",
                str(tmp_path / "runs"),
                "--fidelity",
                "tpmc",
                "--execute",
            ],
        ), patch.object(
            run_cylinder_hex_round2_suite,
            "submit_workflow",
            side_effect=noisy_submit,
        ):
            assert run_cylinder_hex_round2_suite.main() == 0
    finally:
        os.chdir(previous)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["workflows"][0]["status"] == "submitted"
    assert "Job 123 submitted" in captured.err


def test_round2_selection_excludes_existing_and_validation_geometries(tmp_path: Path) -> None:
    ids = [f"cylinder_hex_wp5_{index:03d}" for index in range(7)]
    designs = []
    for index, geometry_id in enumerate(ids):
        row = {
            "geometry_id": geometry_id,
            "eligible_for_model_fitting": index != 6,
            "role": "validation" if index == 6 else "lf_training",
        }
        for variable_index, name in enumerate(VARIABLES):
            row[f"normalized_{name}"] = 0.05 + 0.12 * index + 0.01 * variable_index
        designs.append(row)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"designs": designs, "validation_geometry_ids": [ids[6]]}))
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["geometry_id", "mean_drag", "std_drag", "q95_drag"])
        writer.writeheader()
        for index, geometry_id in enumerate(ids[:6]):
            writer.writerow({"geometry_id": geometry_id, "mean_drag": 1 + index, "std_drag": 0.1, "q95_drag": 2 + index})
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "selected_geometry_ids": ids[:2],
        "uncertainty_samples": {
            "wp1-crn-0000": {"x": 0.0},
            "wp1-crn-0001": {"x": 1.0},
        },
    }))
    tpmc = tmp_path / "tpmc.json"
    delta = tmp_path / "delta.json"
    _linear_model(tpmc, coefficient=0.001, mean=0.003)
    _linear_model(delta, coefficient=0.0, mean=1.0e-6)
    surrogate = tmp_path / "surrogate.json"
    surrogate.write_text(json.dumps({
        "input_names": ["geometry__nose_length_fraction"],
        "models": {"tpmc": str(tpmc), "dsmc_minus_tpmc": str(delta)},
        "model_selection": {"selected_surrogate": "lf_pce"},
    }))
    result = select_round2_geometries(
        design, metrics, bundle, surrogate, tmp_path / "selection.json", count=3, round_number=3
    )
    selected = {row["geometry_id"] for row in result["selected"]}
    assert len(selected) == 3
    assert not selected.intersection(ids[:2])
    assert ids[6] not in selected
    assert result["round"] == 3
    assert result["surrogate_used_for_acquisition"] == "lf_pce"


def test_round2_merge_adds_new_geometry_and_both_fidelities(tmp_path: Path) -> None:
    geometry_id = "cylinder_hex_wp5_008"
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "selected_geometry_ids": [], "geometries": {}, "evaluations": [], "counts": {},
        "uncertainty_samples": {"wp1-crn-0003": {"x": 1.0}},
    }))
    suite = tmp_path / "suite.json"
    geometry = {
        "geometry_id": geometry_id, "design": {"reference_area_m2": 0.002},
        "reference_area_m2": 0.002, "manifest_json": "manifest.json", "mesh_path": "mesh.h5",
        "mesh_reference": "geometry/mesh.h5", "n_tetrahedra": 1, "n_hexahedra": 4,
        "hdf5_fingerprint": "abc", "selection_order": 0, "selection_basis": "test",
    }
    suite.write_text(json.dumps({"round": 3, "geometries": [geometry]}))
    for model_id, sample_id in (("PICLas_DSMC", "hf-crn-0003"), ("PICLas_TPMC", "wp1-crn-0003")):
        result_dir = tmp_path / "runs" / geometry_id / model_id
        result_dir.mkdir(parents=True)
        (result_dir / "piclas_results.json").write_text(json.dumps({
            "status": "collected", "sample_ids": [sample_id], "costs_cpu_hours": [1.0],
            "values_by_qoi": {"C_D": [2.0]},
        }))
    result = merge_round2_results(bundle, suite, tmp_path / "runs", tmp_path / "merged.json")
    assert result["new_rows"] == 2
    merged = json.loads((tmp_path / "merged.json").read_text())
    assert merged["selected_geometry_ids"] == [geometry_id]
    assert merged["counts"][f"{geometry_id}/PICLas_DSMC"] == 1
    assert merged["counts"][f"{geometry_id}/PICLas_TPMC"] == 1
    assert "round3_suite" in merged["geometries"][geometry_id]
    assert merged["study_id"].endswith("round3")


def test_round2_builder_supports_clean_tpmc_only_suite(tmp_path: Path) -> None:
    geometry_id = "cylinder_hex_wp5_004"
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        selection = tmp_path / "selection.json"
        selection.write_text(json.dumps({
            "selected": [{
                "geometry_id": geometry_id,
                "selection_order": 0,
                "selection_basis": "mfmc_test",
            }],
            "untouched_validation_geometry_ids": ["cylinder_hex_wp5_031"],
        }))
        design_root = tmp_path / "design"
        source_dir = design_root / "designs" / geometry_id
        source_dir.mkdir(parents=True)
        source_manifest = source_dir / f"{geometry_id}.manifest.json"
        source_manifest.write_text(json.dumps({
            "specification": {},
            "provenance": {"uniform_linear_scale_factor": 0.1},
        }))
        design = design_root / "geometry_design_manifest.json"
        design.write_text(json.dumps({"designs": [{
            "geometry_id": geometry_id,
            "manifest_path": f"designs/{geometry_id}/{geometry_id}.manifest.json",
            "reference_area_m2": 0.002,
        }]}))
        lf = tmp_path / "lf.json"
        lf.write_text(json.dumps({
            "regime": {},
            "metadata": {},
            "samples": [{"state": value} for value in (0.0, 0.5, 1.0)],
        }))
        base = tmp_path / "base.json"
        base.write_text(json.dumps({
            "adapter": {"model_id": "PICLas_DSMC", "fidelity": "hf", "kwargs": {}},
            "request": {},
        }))
        geometry_dir = tmp_path / "piclas" / "geometry" / geometry_id
        geometry_dir.mkdir(parents=True)
        generated_manifest = geometry_dir / f"{geometry_id}.manifest.json"
        generated_manifest.write_text(json.dumps({"assets": {}}))
        mesh = geometry_dir / f"{geometry_id}_mesh.h5"
        mesh.write_bytes(b"test")
        with patch(
            "mfmc_campaign.geometry_round2.build_geometry_assets",
            return_value={"manifest_json": str(generated_manifest)},
        ), patch(
            "mfmc_campaign.geometry_round2.validate_piclas_hdf5_mesh",
            return_value={"valid": True, "nElems": 4, "mesh_fingerprint": "abc", "x_projected_reference_area_m2": 0.002},
        ):
            result = build_round2_piclas_suite(
                selection,
                design,
                lf,
                output_root=tmp_path / "piclas" / "geometry",
                config_output_dir=tmp_path / "configs",
                suite_output_json=tmp_path / "suite.json",
                base_config_json=base,
                n_dsmc=0,
                n_tpmc=3,
                round_number=4,
                mpi_procs=36,
                simulator_module="PICLas_prandtl",
            )
        row = result["geometries"][0]
        assert "tpmc_workflow_config" in row
        assert "dsmc_workflow_config" not in row
        assert result["total_dsmc_runs"] == 0
        assert result["total_tpmc_runs"] == 3
        assert result["mpi_procs"] == 36
        assert result["simulator_module"] == "PICLas_prandtl"
        tpmc_config = json.loads(
            (tmp_path / "configs" / f"{geometry_id}_tpmc.json").read_text()
        )
        assert tpmc_config["adapter"]["kwargs"]["mpi_procs"] == 36
        assert tpmc_config["adapter"]["kwargs"]["simulator_module"] == "PICLas_prandtl"
    finally:
        os.chdir(previous)


def test_round2_stages_generated_control_node_surface_without_regeneration(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    assets = {}
    for key, name in (
        ("adbsat_obj", "surface.obj"),
        ("adbsat_mat", "surface.mat"),
        ("meshing_surface_stl", "surface.stl"),
        ("canonical_surface_npz", "surface.npz"),
        ("gmsh_exterior_geo", "surface.geo"),
    ):
        (source_dir / name).write_bytes(f"unique-{key}".encode())
        assets[key] = name
    source_manifest = source_dir / "control.manifest.json"
    source_manifest.write_text(json.dumps({
        "parameterization": "symmetric_surface_control_nodes",
        "assets": assets,
        "mesh_fingerprint": "node-exact-fingerprint",
    }))
    staged = _stage_generated_geometry(source_manifest, tmp_path / "staged")
    assert staged["mesh_fingerprint"] == "node-exact-fingerprint"
    assert staged["assets"]["piclas_volume_mesh"] is None
    for key, name in assets.items():
        assert (tmp_path / "staged" / name).read_bytes() == f"unique-{key}".encode()
