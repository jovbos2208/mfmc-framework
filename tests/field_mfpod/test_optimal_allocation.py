import numpy as np
import pytest
import yaml

from mfmc_campaign.field_mfpod.allocation import (
    AllocationOptions,
    build_moment_features,
    compare_allocation_strategies,
    optimize_allocation,
)
from mfmc_campaign.field_mfpod.models import MFPODError
from mfmc_campaign.field_mfpod.config import load_config
from mfmc_campaign.field_mfpod.workflow import (
    field_comparison_allocations,
    optimal_allocation,
)


def scalar_pilot(seed=4, n=500, sentman_noise=0.35):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=n)
    t = h + 0.18 * rng.normal(size=n)
    s = 0.7 * h + sentman_noise * rng.normal(size=n)
    return {"DSMC": h, "TPMC": t, "SENTMAN": s}


def options(mode="enumeration", **kwargs):
    values = dict(
        budget=12.0,
        minimum_target=2,
        minimum_counts={"TPMC": 2},
        min_ratios={"TPMC": 1.0},
        max_ratios={"TPMC": 10.0},
        maximum_counts={"DSMC": 10, "TPMC": 30, "SENTMAN": 30},
        mode=mode,
        random_seed=19,
    )
    values.update(kwargs)
    return AllocationOptions(**values)


def test_moment_features_contain_mean_and_symmetric_products():
    b = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    features = build_moment_features(b)
    np.testing.assert_allclose(features, [[1, 2, 1, 2, 4], [3, 4, 9, 12, 16]])


def test_enumeration_is_reference_minimum_and_budget_constraints_are_exact():
    pilot = scalar_pilot()
    costs = {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05}
    result = optimize_allocation(pilot, costs, options())
    assert result.total_cost <= 12.0 + 1e-12
    assert result.counts["TPMC"] >= result.counts["DSMC"]
    assert result.counts["TPMC"] <= 10 * result.counts["DSMC"]
    assert result.objective == min(row["objective"] for row in result.candidate_table)


def test_continuous_round_and_greedy_return_feasible_integer_allocations():
    pilot = scalar_pilot()
    costs = {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05}
    for mode in ("continuous_round", "greedy"):
        result = optimize_allocation(pilot, costs, options(mode))
        assert all(isinstance(value, int) for value in result.counts.values())
        assert result.total_cost <= 12.0 + 1e-12
        assert result.counts["TPMC"] <= 10 * result.counts["DSMC"]


def test_nan_rows_are_dropped_and_negative_correlation_is_valid():
    pilot = scalar_pilot(n=100)
    pilot["TPMC"] = -pilot["TPMC"]
    pilot["SENTMAN"][3] = np.nan
    result = optimize_allocation(pilot, {"DSMC": 1, "TPMC": .2, "SENTMAN": .1}, options())
    assert result.diagnostics["dropped_nonfinite_rows"] == 1
    assert np.isfinite(result.objective)


def test_all_nonfinite_pilot_fails_clearly():
    with pytest.raises(MFPODError, match="Fewer than two finite"):
        optimize_allocation(
            {"DSMC": [np.nan, np.nan], "TPMC": [1.0, 2.0]},
            {"DSMC": 1.0, "TPMC": 0.1},
            AllocationOptions(budget=5.0),
        )


def test_nearly_singular_and_redundant_controls_are_psd_regularized():
    rng = np.random.default_rng(9)
    h = rng.normal(size=200)
    t = h + 1e-12 * rng.normal(size=200)
    pilot = {"DSMC": h, "TPMC": t, "SENTMAN": t.copy()}
    result = optimize_allocation(pilot, {"DSMC": 1, "TPMC": .2, "SENTMAN": .01}, options())
    assert np.isfinite(result.objective)
    assert result.diagnostics["maximum_condition_number"] >= 1.0
    assert result.diagnostics["regularization"] > 0.0


def test_useless_control_is_not_required_and_hf_only_fallback_is_available():
    rng = np.random.default_rng(10)
    pilot = {"DSMC": rng.normal(size=300), "SENTMAN": rng.normal(size=300)}
    result = optimize_allocation(
        pilot,
        {"DSMC": 1.0, "SENTMAN": 0.5},
        AllocationOptions(budget=8.0, maximum_counts={"DSMC": 8, "SENTMAN": 16}, mode="enumeration"),
    )
    assert result.counts["SENTMAN"] == 0
    assert result.counts["DSMC"] == 8


def test_extremely_cheap_informative_sentman_gets_extra_samples():
    pilot = scalar_pilot(sentman_noise=0.12)
    result = optimize_allocation(
        pilot,
        {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.001},
        options(maximum_counts={"DSMC": 8, "TPMC": 30, "SENTMAN": 100}),
    )
    assert result.counts["SENTMAN"] > result.counts["DSMC"]


def test_tpmc_cap_can_be_active():
    pilot = scalar_pilot()
    result = optimize_allocation(
        {"DSMC": pilot["DSMC"], "TPMC": pilot["TPMC"]},
        {"DSMC": 1.0, "TPMC": 0.001},
        AllocationOptions(
            budget=4.0,
            minimum_target=2,
            min_ratios={"TPMC": 1.0},
            max_ratios={"TPMC": 10.0},
            maximum_counts={"DSMC": 3, "TPMC": 100},
            mode="enumeration",
        ),
    )
    assert result.counts["TPMC"] == 10 * result.counts["DSMC"]


def test_sentman_helps_when_incrementally_informative_but_not_when_redundant_and_costly():
    pilot = scalar_pilot(sentman_noise=0.08)
    informative = optimize_allocation(
        pilot,
        {"DSMC": 1.0, "TPMC": 0.25, "SENTMAN": 0.01},
        options(),
    )
    redundant = {**pilot, "SENTMAN": pilot["TPMC"].copy()}
    costly = optimize_allocation(
        redundant,
        {"DSMC": 1.0, "TPMC": 0.25, "SENTMAN": 1.0},
        options(),
    )
    assert informative.counts["SENTMAN"] > 0
    assert costly.counts["SENTMAN"] == 0


def test_bootstrap_robust_is_reproducible_and_reports_uncertainty():
    pilot = scalar_pilot(n=80)
    costs = {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05}
    opts = options("bootstrap_robust", bootstrap_repeats=30, robust_quantile=.9)
    first = optimize_allocation(pilot, costs, opts)
    second = optimize_allocation(pilot, costs, opts)
    assert first.counts == second.counts
    assert first.objective == second.objective
    assert first.bootstrap_summary["random_seed"] == 19


def test_strategy_comparison_includes_hf_only_and_reference_enumeration():
    pilot = scalar_pilot(n=100)
    rows = compare_allocation_strategies(
        pilot,
        {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05},
        options("continuous_round"),
    )
    methods = {row["method"] for row in rows}
    assert {
        "HF-only",
        "two-fidelity-TPMC",
        "fixed-minimum-ratios",
        "enumeration",
        "greedy",
        "continuous_round",
    } <= methods


def test_yaml_workflow_writes_allocation_diagnostics_and_candidate_tables(tmp_path):
    pilot = scalar_pilot(n=80)
    archive = tmp_path / "pilot.npz"
    np.savez(archive, **pilot)
    config = {
        "case_name": "allocation-cli-test",
        "output_root": str(tmp_path / "out"),
        "costs": {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05},
        "allocation_optimization": {
            "pilot_response_archive": str(archive),
            "responses_are_features": True,
            "budget": 8.0,
            "mode": "enumeration",
            "minimum_target": 2,
            "min_ratios": {"TPMC": 1.0},
            "max_ratios": {"TPMC": 10.0},
            "maximum_counts": {"DSMC": 7, "TPMC": 20, "SENTMAN": 20},
            "bootstrap_repeats": 0,
        },
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(path)
    result = optimal_allocation(cfg)
    out = cfg.output_dir / "allocation"
    assert result["feasible"]
    assert (out / "optimal_allocation.json").exists()
    assert (out / "optimal_allocation_candidates.csv").exists()
    assert (out / "allocation_strategy_comparison.csv").exists()


def test_external_pilot_plans_union_for_equal_cost_comparators(tmp_path):
    pilot = scalar_pilot(n=80)
    fields = {
        name: np.column_stack((values, values * values))
        for name, values in pilot.items()
    }
    archive = tmp_path / "pilot_fields.npz"
    np.savez(
        archive,
        **fields,
        **{f"CD_{name}": values for name, values in pilot.items()},
    )
    config = {
        "case_name": "comparison-union-test",
        "output_root": str(tmp_path / "out"),
        "high_fidelity": "DSMC",
        "control_variates": ["TPMC", "SENTMAN"],
        "costs": {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05},
        "field_allocation": {
            "pilot_field_archive": str(archive),
            "mode": "continuous_round",
            "bootstrap_repeats": 0,
            "mean_weight": 0.25,
            "second_moment_weight": 0.75,
        },
        "allocation_constraints": {
            "budget": 8.0,
            "minimum_target": 2,
            "maximum_counts": {"DSMC": 8, "TPMC": 30, "SENTMAN": 60},
            "max_ratios": {"TPMC": 10.0},
        },
        "validation": {
            "compare_scalar_drag_allocation": True,
            "fixed_ratios": {"TPMC": 5.0, "SENTMAN": 10.0},
        },
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    allocations = field_comparison_allocations(
        load_config(path),
        maximum_counts=config["allocation_constraints"]["maximum_counts"],
    )
    assert {
        "DSMC-only",
        "two-fidelity-TPMC",
        "fixed-ratios",
        "scalar-drag-allocation",
    } <= set(allocations)
    required = {
        name: max(result.counts.get(name, 0) for result in allocations.values())
        for name in ("DSMC", "TPMC", "SENTMAN")
    }
    assert required["DSMC"] == 8
    assert required["TPMC"] >= 8
    assert required["SENTMAN"] >= 8
