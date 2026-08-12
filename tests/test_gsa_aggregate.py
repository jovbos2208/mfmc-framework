from __future__ import annotations

import csv
import json
from pathlib import Path

from mfmc_campaign.gsa_aggregate import aggregate_surrogate_gsa


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_surrogate_gsa_marks_hf_poor_case_exploratory(tmp_path: Path) -> None:
    pce = tmp_path / "cube_300km" / "surrogate_pce"
    pce.mkdir(parents=True)
    metrics = pce / "metrics.csv"
    sources = pce / "sources.csv"
    intervals = pce / "intervals.csv"
    _csv(metrics, [{"fold": "aggregate", "method": "mf_residual", "lf_model_id": "TPMC", "rmse": 0.1, "r2": 0.95}])
    _csv(sources, [{"name": "density", "first_order": 0.8, "total_effect": 0.9}])
    _csv(intervals, [{"name": "density", "first_q025": 0.6, "first_q975": 0.9, "total_q025": 0.7, "total_q975": 0.98, "bootstrap_repetitions_successful": 20}])
    (pce / "surrogate_pce_manifest.json").write_text(json.dumps({
        "best_lf_model_id": "TPMC",
        "metrics_csv": str(metrics),
        "load_summary": {"original_model_counts": {"PICLas_DSMC": 10, "TPMC": 100}},
    }), encoding="utf-8")
    (pce / "gsa_summary.json").write_text(json.dumps({
        "status": "sobol_complete_with_refit_bootstrap",
        "source_sobol_csv": str(sources),
        "refit_bootstrap_intervals_csv": str(intervals),
        "quality_flags": ["insufficient_hf_samples_lt_30"],
    }), encoding="utf-8")

    summary = aggregate_surrogate_gsa(str(tmp_path))
    rows = list(csv.DictReader(open(summary["case_summary_csv"], encoding="utf-8")))

    assert summary["case_count"] == 1
    assert summary["primary_case_count"] == 0
    assert rows[0]["eligible_primary_claims"] == "False"
    assert rows[0]["quality_tier"] == "insufficient_hf_exploratory"
