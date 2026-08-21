from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mfmc_campaign.geometry_design import VARIABLES
from mfmc_campaign.geometry_metric_gp import GeometryMetricGPModel, fit_geometry_metric_gp


def test_geometry_metric_gp_writes_reloadable_models_and_honest_loo(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    geometry_ids = [f"cylinder_hex_wp5_{index:03d}" for index in range(8)]
    samples = [f"wp1-crn-{index:04d}" for index in range(24)]
    geometries = {}
    evaluations = []
    for geometry_index, geometry_id in enumerate(geometry_ids):
        point = rng.random(4)
        geometries[geometry_id] = {
            "design": {f"normalized_{name}": float(point[index]) for index, name in enumerate(VARIABLES)}
        }
        center = 0.003 + 2.0e-4 * point[0] - 1.0e-4 * point[1] + 8.0e-5 * point[2]
        spread = 2.0e-5 + 1.0e-5 * point[3]
        for sample_index, sample_id in enumerate(samples):
            value = center + spread * np.sin(0.7 * sample_index)
            evaluations.append({
                "geometry_id": geometry_id,
                "canonical_sample_id": sample_id,
                "model_id": "PICLas_TPMC",
                "drag_area_m2": float(value),
            })
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "reference_area_convention": "canonical_manifest_area",
        "selected_geometry_ids": geometry_ids,
        "geometries": geometries,
        "uncertainty_samples": {sample_id: {} for sample_id in samples},
        "evaluations": evaluations,
    }))
    result = fit_geometry_metric_gp(
        bundle,
        tmp_path / "gp",
        bootstrap_count=40,
        optimizer_restarts=0,
        seed=12,
    )
    assert result["n_geometries"] == 8
    assert len(result["validation"]["rows"]) == 3
    assert len(Path(result["loo_predictions_csv"]).read_text().splitlines()) == 1 + 8 * 3
    model = GeometryMetricGPModel.read_json(result["models"]["mean_drag"])
    prediction, uncertainty = model.predict(np.full(4, 0.5))
    assert prediction.shape == uncertainty.shape == (1,)
    assert np.isfinite(prediction[0]) and uncertainty[0] >= 0.0
