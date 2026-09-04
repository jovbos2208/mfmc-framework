from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mfmc_campaign.surrogate_pce_analysis import fit_multifidelity_pce_analysis


def _write_synthetic_multifidelity_dataset(path: Path) -> None:
    rng = np.random.default_rng(20260317)
    input_names = [f"input__x{index}" for index in range(6)]
    fieldnames = [
        "cell_id",
        "qoi",
        "model_id",
        "fidelity",
        "repetition",
        "sample_fingerprint",
        "value",
        *input_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for repetition in range(6):
            x = rng.normal(size=(100, 6))
            lf = (
                1.0
                + 1.1 * x[:, 0]
                - 0.9 * x[:, 1]
                + 0.7 * x[:, 2]
                + 0.5 * x[:, 3]
                - 0.4 * x[:, 4]
                + 0.3 * x[:, 5]
                + 0.5 * x[:, 0] * x[:, 1]
                - 0.35 * x[:, 2] * x[:, 3]
                + 0.25 * x[:, 4] ** 2
                - 0.2 * x[:, 5] ** 2
            )
            hf = lf + 0.15 * x[:, 0] * x[:, 2]
            for index in range(100):
                common = {
                    "cell_id": f"cell-{repetition}",
                    "qoi": "C_D",
                    "repetition": repetition,
                    "sample_fingerprint": f"sample-{repetition}-{index}",
                    **{name: x[index, column] for column, name in enumerate(input_names)},
                }
                writer.writerow({**common, "model_id": "LF", "fidelity": "lf", "value": lf[index]})
                if index < 5:
                    writer.writerow({**common, "model_id": "HF", "fidelity": "hf", "value": hf[index]})


def test_multifidelity_pce_analysis_improves_sparse_hf_fit(tmp_path: Path) -> None:
    dataset = tmp_path / "surrogate_dataset.csv"
    _write_synthetic_multifidelity_dataset(dataset)

    summary = fit_multifidelity_pce_analysis(
        str(dataset),
        str(tmp_path / "analysis"),
        qoi="C_D",
        degree=2,
        q_norm=1.0,
        max_interaction=2,
        cv_folds=3,
        max_rows_per_model=80,
    )

    assert summary["best_lf_model_id"] == "LF"
    with Path(summary["metrics_csv"]).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    aggregate = {
        row["method"]: row
        for row in rows
        if row["fold"] == "aggregate" and row["lf_model_id"] == "LF"
    }
    assert float(aggregate["mf_residual"]["r2"]) > 0.98
    assert float(aggregate["mf_residual"]["rmse"]) < float(aggregate["hf_only"]["rmse"])
    assert Path(summary["model_files"]["delta_LF"]).exists()
