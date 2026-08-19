from __future__ import annotations

import json
from pathlib import Path

from mfmc_campaign.geometry_learning_curve import build_geometry_learning_curve


def _manifest(path: Path, n: int, lf_rmse: float, mf_rmse: float) -> None:
    path.write_text(json.dumps({
        "n_geometries": n,
        "n_hf": 5 * n,
        "n_lf": 90 * n,
        "geometry_held_out_summary": [
            {"method": "lf_pce", "mean_geometry_rmse": lf_rmse, "median_geometry_rmse": 0.9 * lf_rmse},
            {"method": "mf_pce", "mean_geometry_rmse": mf_rmse, "median_geometry_rmse": 0.9 * mf_rmse},
        ],
    }))


def test_learning_curve_selects_lf_when_mf_gain_is_below_threshold(tmp_path: Path) -> None:
    first = tmp_path / "six.json"
    second = tmp_path / "nine.json"
    _manifest(first, 6, 1.3e-4, 1.299e-4)
    _manifest(second, 12, 9.5e-5, 9.49e-5)
    result = build_geometry_learning_curve([second, first], tmp_path / "curve")
    assert [row["n_geometries"] for row in result["rows"]] == [6, 12]
    assert all(row["selected_surrogate"] == "lf_pce" for row in result["rows"])
    assert result["status"] == "target_met"
    assert Path(result["learning_curve_csv"]).is_file()
