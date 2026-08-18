from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mfmc_campaign.geometry_design import (
    VARIABLES,
    build_cylinder_hex_design,
    maximin_latin_hypercube,
    select_initial_hf_designs,
)


def test_maximin_lhs_is_deterministic_and_stratified() -> None:
    first = maximin_latin_hypercube(9, 4, seed=17, trials=12)
    second = maximin_latin_hypercube(9, 4, seed=17, trials=12)
    assert np.array_equal(first, second)
    for column in range(first.shape[1]):
        assert sorted(np.floor(first[:, column] * 9).astype(int)) == list(range(9))


def test_design_builds_assets_and_immutable_validation_split(tmp_path: Path) -> None:
    summary = build_cylinder_hex_design(
        tmp_path / "first", n_designs=8, n_validation=2, seed=91, maximin_trials=8
    )
    repeated = build_cylinder_hex_design(
        tmp_path / "second", n_designs=8, n_validation=2, seed=91, maximin_trials=8
    )
    assert summary["validation_geometry_ids"] == repeated["validation_geometry_ids"]
    assert summary["n_lf_training"] == 6
    assert summary["density_scale_required_for_kn_similarity"] == 10.0
    assert summary["designs"][0]["role"] == "baseline_lf_training"
    for row in summary["designs"]:
        geometry_id = row["geometry_id"]
        assert (tmp_path / "first" / "adbsat_runtime_assets" / "inou" / "obj_files" / f"{geometry_id}.obj").is_file()
        assert (tmp_path / "first" / "adbsat_runtime_assets" / "inou" / "models" / f"{geometry_id}.mat").is_file()
        manifest = json.loads((tmp_path / "first" / row["manifest_path"]).read_text())
        assert manifest["validation"]["valid"]


def test_initial_hf_selection_excludes_validation_and_keeps_baseline(tmp_path: Path) -> None:
    summary = build_cylinder_hex_design(
        tmp_path / "design", n_designs=8, n_validation=2, seed=44, maximin_trials=8
    )
    metrics_path = tmp_path / "lf_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["geometry_id", "mean_drag", "std_drag", "q95_drag", "reference_area_convention"],
        )
        writer.writeheader()
        for index, row in enumerate(summary["designs"]):
            if row["eligible_for_model_fitting"]:
                writer.writerow(
                    {
                        "geometry_id": row["geometry_id"],
                        "mean_drag": 1.0 + 0.03 * index,
                        "std_drag": 0.2 - 0.01 * index,
                        "q95_drag": 1.4 + 0.01 * index,
                        "reference_area_convention": "canonical_manifest_area",
                    }
                )
    selection = select_initial_hf_designs(
        summary["design_csv"], metrics_path, tmp_path / "selection.json", count=4
    )
    selected = [row["geometry_id"] for row in selection["selected"]]
    assert selected[0] == "cylinder_hex_wp5_000"
    assert len(selected) == len(set(selected)) == 4
    assert not set(selected).intersection(summary["validation_geometry_ids"])
    assert selection["selected"][0]["selection_basis"] == "baseline"
