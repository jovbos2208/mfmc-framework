from __future__ import annotations

import numpy as np

from mfmc_campaign.surrogate_gsa import estimate_surrogate_sobol, refit_bootstrap_source_sobol
from mfmc_campaign.surrogate_pce_analysis import ModelDataset


def test_jansen_sobol_recovers_additive_variance_fractions() -> None:
    rng = np.random.default_rng(20260317)
    x_a = rng.normal(size=(120_000, 2))
    x_b = rng.normal(size=(120_000, 2))
    predictor = lambda values: values[:, 0] + 2.0 * values[:, 1]

    rows, intervals, _variance = estimate_surrogate_sobol(
        predictor,
        x_a,
        x_b,
        {"x0": [0], "x1": [1]},
        bootstrap=20,
        seed=42,
    )
    by_name = {row["name"]: row for row in rows}

    assert abs(by_name["x0"]["first_order"] - 0.2) < 0.015
    assert abs(by_name["x0"]["total_effect"] - 0.2) < 0.015
    assert abs(by_name["x1"]["first_order"] - 0.8) < 0.015
    assert abs(by_name["x1"]["total_effect"] - 0.8) < 0.015
    assert len(intervals) == 2


def test_refit_bootstrap_resamples_repetition_blocks_and_refits_models() -> None:
    rng = np.random.default_rng(91)
    repetitions = np.repeat(np.arange(8), 4)
    x = rng.uniform(-1.0, 1.0, size=(len(repetitions), 2))
    lf_y = 1.0 + x[:, 0] + 0.4 * x[:, 1]
    hf_y = lf_y + 0.2 * x[:, 0] * x[:, 1] + rng.normal(0.0, 0.005, len(lf_y))
    keys = np.asarray([f"key-{index}" for index in range(len(repetitions))])
    cells = np.asarray([f"cell-{rep}" for rep in repetitions])
    common = {
        "input_names": ["input__x0", "input__x1"],
        "x": x,
        "repetitions": repetitions,
        "cell_ids": cells,
        "sample_fingerprints": keys,
    }
    hf = ModelDataset(model_id="hf", fidelity="hf", y=hf_y, **common)
    lf = ModelDataset(model_id="lf", fidelity="lf", y=lf_y, **common)
    x_a = rng.uniform(-1.0, 1.0, size=(1_000, 2))
    x_b = rng.uniform(-1.0, 1.0, size=(1_000, 2))

    samples, intervals, diagnostics = refit_bootstrap_source_sobol(
        hf,
        lf,
        x_a,
        x_b,
        {"source.x0": [0], "source.x1": [1]},
        repetitions=6,
        seed=17,
    )

    assert diagnostics["successful"] == 6
    assert diagnostics["failed"] == 0
    assert len(samples) == 12
    assert len(intervals) == 2
    assert all(row["bootstrap_kind"] == "grouped_repetition_surrogate_refit" for row in intervals)
