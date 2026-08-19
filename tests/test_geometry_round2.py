from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mfmc_campaign.geometry_design import VARIABLES
from mfmc_campaign.geometry_round2 import merge_round2_results, select_round2_geometries
from mfmc_campaign.sparse_pce import SparsePCEModel


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
