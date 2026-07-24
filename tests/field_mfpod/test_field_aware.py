import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfmc_campaign.field_mfpod.allocation import (
    AllocationOptions,
    optimize_allocation,
    optimize_field_allocation,
)
from mfmc_campaign.field_mfpod.covariance_operator import (
    FullFieldMFMCStatistics,
    estimate_full_field_mfmc,
    explicit_full_field_covariance,
    solve_full_field_pod,
)
from mfmc_campaign.field_mfpod.field_statistics import compute_field_pilot_statistics


def paired_fields(seed=8, samples=80, dimension=6):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(samples, dimension))
    t = 0.9 * h + 0.2 * rng.normal(size=h.shape)
    s = 0.65 * h + 0.45 * rng.normal(size=h.shape)
    return {"DSMC": h, "TPMC": t, "SENTMAN": s}


def test_hilbert_covariances_match_explicit_rank_one_calculation():
    fields = paired_fields(samples=12, dimension=4)
    reference = np.mean(fields["DSMC"], axis=0)
    result = compute_field_pilot_statistics(
        fields, reference_field=reference, covariance_ridge=0.0
    )
    models = result.models
    centered = {name: fields[name] - reference for name in models}
    mean_expected = np.empty((3, 3))
    second_expected = np.empty((3, 3))
    rank_one = {
        name: np.asarray([np.outer(row, row).reshape(-1) for row in centered[name]])
        for name in models
    }
    for i, left in enumerate(models):
        for j, right in enumerate(models):
            mean_expected[i, j] = np.sum(
                (centered[left] - centered[left].mean(axis=0))
                * (centered[right] - centered[right].mean(axis=0))
            ) / (centered[left].shape[0] - 1)
            second_expected[i, j] = np.sum(
                (rank_one[left] - rank_one[left].mean(axis=0))
                * (rank_one[right] - rank_one[right].mean(axis=0))
            ) / (rank_one[left].shape[0] - 1)
    np.testing.assert_allclose(result.mean_covariance_raw, mean_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.second_moment_covariance_raw, second_expected, rtol=1e-12, atol=1e-12)


def test_rank_one_hilbert_schmidt_identity():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(2, 9))
    assert np.isclose(np.sum(np.outer(x, x) * np.outer(y, y)), (x @ y) ** 2)


def test_field_allocation_is_integer_nested_and_has_separate_weights():
    fields = paired_fields(samples=120)
    result = optimize_field_allocation(
        fields,
        {"DSMC": 1.0, "TPMC": 0.15, "SENTMAN": 0.03},
        AllocationOptions(
            budget=10.0,
            minimum_target=2,
            maximum_counts={"DSMC": 9, "TPMC": 30, "SENTMAN": 50},
            max_ratios={"TPMC": 10.0},
            mode="enumeration",
            mean_weight=0.25,
            second_moment_weight=0.75,
        ),
    )
    assert result.total_cost <= 10.0 + 1e-12
    assert all(isinstance(value, int) for value in result.counts.values())
    for control in ("TPMC", "SENTMAN"):
        assert result.counts[control] == 0 or result.counts[control] >= result.counts["DSMC"]
    assert set(result.control_weights) == {"mean", "second_moment"}
    assert result.diagnostics["tpmc_basis_used"] is False


def test_optional_informative_control_is_activated_by_scalable_strategies():
    rng = np.random.default_rng(27)
    high = rng.normal(size=(120, 5))
    control = high + 0.02 * rng.normal(size=high.shape)
    fields = {"DSMC": high, "TPMC": control}
    costs = {"DSMC": 1.0, "TPMC": 0.08}

    for mode in ("greedy", "continuous_round"):
        result = optimize_field_allocation(
            fields,
            costs,
            AllocationOptions(
                budget=20.0,
                minimum_target=2,
                minimum_counts={},
                maximum_counts={"DSMC": 20, "TPMC": 200},
                min_ratios={},
                max_ratios={"TPMC": 10.0},
                mode=mode,
                mean_weight=0.25,
                second_moment_weight=0.75,
            ),
        )
        assert result.counts["TPMC"] > result.counts["DSMC"]
        assert result.total_cost <= 20.0 + 1.0e-12


def test_full_field_mean_and_covariance_action_match_explicit_estimator():
    fields = paired_fields(samples=18, dimension=5)
    counts = {"DSMC": 6, "TPMC": 12, "SENTMAN": 15}
    reference = np.mean(fields["DSMC"][:4], axis=0)
    beta_mu = {"TPMC": -0.45, "SENTMAN": 0.2}
    beta_m = {"TPMC": -0.7, "SENTMAN": 0.15}
    result = estimate_full_field_mfmc(
        fields,
        counts,
        reference_field=reference,
        mean_weights=beta_mu,
        second_moment_weights=beta_m,
    )
    x = {name: values - reference for name, values in fields.items()}
    expected_mean = fields["DSMC"][:6].mean(axis=0)
    expected_centered_mean = x["DSMC"][:6].mean(axis=0)
    expected_second = x["DSMC"][:6].T @ x["DSMC"][:6] / 6
    for name in ("TPMC", "SENTMAN"):
        n_i = counts[name]
        expected_mean += beta_mu[name] * (
            fields[name][:n_i].mean(axis=0) - fields[name][:6].mean(axis=0)
        )
        expected_centered_mean += beta_mu[name] * (
            x[name][:n_i].mean(axis=0) - x[name][:6].mean(axis=0)
        )
        expected_second += beta_m[name] * (
            x[name][:n_i].T @ x[name][:n_i] / n_i - x[name][:6].T @ x[name][:6] / 6
        )
    expected_covariance = expected_second - np.outer(expected_centered_mean, expected_centered_mean)
    np.testing.assert_allclose(result.mean_field, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        explicit_full_field_covariance(result), expected_covariance, rtol=1e-12, atol=1e-12
    )


def test_matrix_free_pod_matches_explicit_eigendecomposition():
    fields = paired_fields(samples=24, dimension=10)
    statistics = estimate_full_field_mfmc(
        fields,
        {"DSMC": 10, "TPMC": 18},
        reference_field=np.mean(fields["DSMC"][:5], axis=0),
        mean_weights={"TPMC": -0.5},
        second_moment_weights={"TPMC": -0.6},
    )
    result = solve_full_field_pod(statistics, n_modes=4, random_seed=31)
    expected_values, expected_modes = np.linalg.eigh(explicit_full_field_covariance(statistics))
    order = np.argsort(expected_values)[::-1][:4]
    np.testing.assert_allclose(result.eigenvalues, expected_values[order], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        np.abs(result.modes.T @ expected_modes[:, order]), np.eye(4), atol=1e-6
    )
    assert result.diagnostics["maximum_eigenpair_residual"] < 1e-7


def test_drag_optimal_and_complete_field_optimal_allocations_can_differ():
    rng = np.random.default_rng(7)
    h = rng.normal(size=(500, 8))
    t = rng.normal(size=h.shape)
    t[:, 0] = h[:, 0] + 0.05 * rng.normal(size=h.shape[0])
    s = h + 0.25 * rng.normal(size=h.shape)
    s[:, 0] = rng.normal(size=h.shape[0])
    options = AllocationOptions(
        budget=10.0,
        minimum_target=2,
        maximum_counts={"DSMC": 9, "TPMC": 30, "SENTMAN": 30},
        mode="enumeration",
    )
    costs = {"DSMC": 1.0, "TPMC": 0.1, "SENTMAN": 0.1}
    field_result = optimize_field_allocation(
        {"DSMC": h, "TPMC": t, "SENTMAN": s}, costs, options
    )
    drag_result = optimize_allocation(
        {"DSMC": h[:, 0], "TPMC": t[:, 0], "SENTMAN": s[:, 0]}, costs, options
    )
    assert field_result.counts != drag_result.counts
    assert field_result.counts["SENTMAN"] > field_result.counts["TPMC"]
    assert drag_result.counts["TPMC"] > drag_result.counts["SENTMAN"]


def test_negative_ritz_diagnostics_and_small_negative_clipping_are_preserved():
    diagonal = np.diag([2.0, -1.0e-12, -0.5])
    operator = LinearOperator(
        diagonal.shape,
        matvec=lambda vector: diagonal @ vector,
        rmatvec=lambda vector: diagonal @ vector,
        dtype=float,
    )
    statistics = FullFieldMFMCStatistics(
        mean_field=np.zeros(3),
        centered_mean=np.zeros(3),
        covariance=operator,
        reference_field=np.zeros(3),
        counts={"DSMC": 2},
        mean_weights={},
        second_moment_weights={},
        diagnostics={},
    )
    result = solve_full_field_pod(
        statistics,
        n_modes=3,
        negative_eigenvalue_tolerance=1.0e-10,
        clip_small_negative_eigenvalues=True,
    )
    assert result.diagnostics["negative_eigenvalue_count"] == 2
    assert result.diagnostics["large_negative_eigenvalue_count"] == 1
    assert result.diagnostics["small_negative_clipping_applied"]
    assert result.eigenvalues[1] == 0.0
    assert result.eigenvalues[2] == -0.5
