from __future__ import annotations

import json
from pathlib import Path

from mfmc_campaign.geometry_design import VARIABLES
from mfmc_campaign.geometry_mf_surrogate import (
    fit_geometry_multifidelity_surrogate,
    merge_sequential_hf_results,
)


def test_geometry_mf_surrogate_pairs_models_and_builds_nested_acquisition(tmp_path: Path) -> None:
    geometry_ids = [f"cylinder_hex_wp5_{index:03d}" for index in range(4)]
    sample_ids = [f"wp1-crn-{index:04d}" for index in range(8)]
    geometries = {}
    uncertainty = {}
    evaluations = []
    for index, geometry_id in enumerate(geometry_ids):
        design = {f"normalized_{name}": 0.15 + 0.2 * index for name in VARIABLES}
        geometries[geometry_id] = {"design": design}
    for index, sample_id in enumerate(sample_ids):
        uncertainty[sample_id] = {"density_state_scale": 0.7 + 0.08 * index, "aoa_deg": -4.0 + index}
    for geometry_index, geometry_id in enumerate(geometry_ids):
        for sample_index, sample_id in enumerate(sample_ids):
            tpmc = 0.003 + 2.0e-4 * geometry_index + 3.0e-5 * sample_index
            evaluations.append({
                "geometry_id": geometry_id, "canonical_sample_id": sample_id,
                "model_id": "PICLas_TPMC", "drag_area_m2": tpmc,
            })
            if sample_index < 3:
                evaluations.append({
                    "geometry_id": geometry_id, "canonical_sample_id": sample_id,
                    "model_id": "PICLas_DSMC", "drag_area_m2": tpmc + 1.0e-5 * (geometry_index + 1),
                })
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "schema_version": 1,
        "qoi": "drag_area_m2",
        "reference_area_convention": "canonical_manifest_area",
        "selected_geometry_ids": geometry_ids,
        "geometries": geometries,
        "uncertainty_samples": uncertainty,
        "evaluations": evaluations,
    }))
    result = fit_geometry_multifidelity_surrogate(
        bundle, tmp_path / "fit", degree=1, max_interaction=1,
        target_hf_per_geometry=5, acquisition_geometry_count=2,
    )
    assert result["n_hf"] == 12
    assert result["n_lf"] == 32
    assert Path(result["geometry_held_out_metrics_csv"]).is_file()
    plan = json.loads(Path(result["next_hf_acquisition_json"]).read_text())
    assert len(plan["geometries"]) == 2
    assert all(row["additional_hf_count"] == 2 for row in plan["geometries"])
    assert all(
        sample_id not in {"wp1-crn-0000", "wp1-crn-0001", "wp1-crn-0002"}
        for row in plan["geometries"] for sample_id in row["selected_canonical_sample_ids"]
    )


def test_merge_sequential_hf_results_adds_drag_area_rows(tmp_path: Path) -> None:
    geometry_id = "cylinder_hex_wp5_000"
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "study_id": "pilot",
        "geometries": {geometry_id: {"design": {"reference_area_m2": 0.002}}},
        "uncertainty_samples": {"wp1-crn-0007": {"x": 1.0}},
        "evaluations": [],
        "counts": {},
    }))
    result_dir = tmp_path / "runs" / geometry_id
    result_dir.mkdir(parents=True)
    (result_dir / "piclas_results.json").write_text(json.dumps({
        "status": "collected",
        "sample_ids": ["hf-crn-0007"],
        "costs_cpu_hours": [11.0],
        "values_by_qoi": {"C_D": [2.5]},
    }))
    result = merge_sequential_hf_results(bundle, tmp_path / "runs", tmp_path / "merged.json")
    assert result["new_hf_rows"] == 1
    merged = json.loads((tmp_path / "merged.json").read_text())
    assert merged["evaluations"][0]["canonical_sample_id"] == "wp1-crn-0007"
    assert merged["evaluations"][0]["drag_area_m2"] == 0.005
