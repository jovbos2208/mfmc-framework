from __future__ import annotations

import numpy as np

from mfmc_campaign.sparse_pce import (
    fit_sparse_pce,
    grouped_cv_splits,
    hyperbolic_multi_indices,
    regression_metrics,
)


def test_hyperbolic_basis_obeys_degree_and_interaction_limits() -> None:
    indices = hyperbolic_multi_indices(5, 3, q_norm=0.75, max_interaction=2)

    assert len(indices) > 5
    assert np.all(np.count_nonzero(indices, axis=1) <= 2)
    assert np.all(np.sum(indices.astype(float) ** 0.75, axis=1) <= 3.0**0.75 + 1.0e-12)
    assert not np.any(np.all(indices == 0, axis=1))


def test_grouped_splits_never_mix_repetitions() -> None:
    groups = np.repeat(np.arange(5), 4)
    for train, test in grouped_cv_splits(groups, folds=5):
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_sparse_pce_recovers_known_polynomial() -> None:
    rng = np.random.default_rng(20260317)
    x_train = rng.normal(size=(700, 4))
    x_test = rng.normal(size=(300, 4))

    def function(x: np.ndarray) -> np.ndarray:
        return 1.25 + 2.0 * x[:, 0] - 1.5 * x[:, 1] ** 2 + 0.8 * x[:, 0] * x[:, 2]

    y_train = function(x_train)
    y_test = function(x_test)
    model = fit_sparse_pce(
        x_train,
        y_train,
        ["x0", "x1", "x2", "unused"],
        groups=np.repeat(np.arange(7), 100),
        degree=2,
        q_norm=1.0,
        max_interaction=2,
    )
    metrics = regression_metrics(y_test, model.predict(x_test))

    assert metrics["r2"] > 0.999
    assert metrics["rmse"] < 0.03
    assert np.count_nonzero(np.abs(model.standardized_coefficients) > 1.0e-8) <= 6
