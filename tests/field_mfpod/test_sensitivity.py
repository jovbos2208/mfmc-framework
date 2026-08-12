import csv

import numpy as np
import yaml

from mfmc_campaign.field_mfpod.config import load_config
from mfmc_campaign.field_mfpod.sensitivity import (
    _nested_prefix_indices,
    _permuted_nested_fields,
    run_field_sensitivity,
)


def test_permutation_preserves_target_control_pairing():
    production = {
        "DSMC": np.arange(12, dtype=float).reshape(4, 3),
        "TPMC": np.arange(24, dtype=float).reshape(8, 3),
    }
    ids = {
        "DSMC": np.asarray(["s0", "s1", "s2", "s3"]),
        "TPMC": np.asarray(["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]),
    }
    result = _permuted_nested_fields(
        production,
        ids,
        {"DSMC": 2, "TPMC": 6},
        random_seed=91,
    )
    target_lookup = {tuple(row): sample_id for row, sample_id in zip(production["DSMC"], ids["DSMC"])}
    control_lookup = {tuple(row): sample_id for row, sample_id in zip(production["TPMC"], ids["TPMC"])}
    target_prefix = [target_lookup[tuple(row)] for row in result["DSMC"][:2]]
    control_prefix = [control_lookup[tuple(row)] for row in result["TPMC"][:2]]
    assert control_prefix == target_prefix


def test_permutations_are_reproducible():
    production = {
        "DSMC": np.arange(12, dtype=float).reshape(4, 3),
        "TPMC": np.arange(24, dtype=float).reshape(8, 3),
    }
    ids = {
        "DSMC": np.asarray(["s0", "s1", "s2", "s3"]),
        "TPMC": np.asarray([f"s{i}" for i in range(8)]),
    }
    first = _permuted_nested_fields(
        production, ids, {"DSMC": 2, "TPMC": 6}, random_seed=91
    )
    second = _permuted_nested_fields(
        production, ids, {"DSMC": 2, "TPMC": 6}, random_seed=91
    )
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_reference_prefixes_are_reproducible_and_nested():
    first = _nested_prefix_indices(20, [4, 8, 12], random_seed=73)
    second = _nested_prefix_indices(20, [4, 8, 12], random_seed=73)
    assert all(np.array_equal(first[count], second[count]) for count in first)
    assert np.array_equal(first[4], first[8][:4])
    assert np.array_equal(first[8], first[12][:8])


def test_sensitivity_writes_repeated_m0_and_reference_outputs(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    original_suptitle = Figure.suptitle

    def capture_title(self, title, *args, **kwargs):
        titles.append(str(title))
        return original_suptitle(self, title, *args, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", capture_title)
    rng = np.random.default_rng(14)
    dimension = 6
    pilot_dsmc = rng.normal(size=(12, dimension))
    pilot_tpmc = pilot_dsmc + 0.1 * rng.normal(size=pilot_dsmc.shape)
    pilot_sentman = 0.5 * pilot_dsmc + 0.7 * rng.normal(size=pilot_dsmc.shape)
    stream = rng.normal(size=(18, dimension))
    production = {
        "DSMC": stream[:6],
        "TPMC": stream[:12] + 0.1 * rng.normal(size=(12, dimension)),
        "SENTMAN": 0.5 * stream + 0.7 * rng.normal(size=stream.shape),
    }
    ids = {
        "DSMC": np.asarray([f"s{i}" for i in range(6)]),
        "TPMC": np.asarray([f"s{i}" for i in range(12)]),
        "SENTMAN": np.asarray([f"s{i}" for i in range(18)]),
    }
    reference = rng.normal(size=(8, dimension))
    results = tmp_path / "case"
    snapshots = results / "snapshots"
    pilot_dir = results / "pilot"
    snapshots.mkdir(parents=True)
    pilot_dir.mkdir(parents=True)
    np.savez_compressed(
        snapshots / "prepared_field_snapshots.npz",
        pilot_DSMC=pilot_dsmc,
        pilot_TPMC=pilot_tpmc,
        pilot_SENTMAN=pilot_sentman,
        pilot_CD_DSMC=np.sum(pilot_dsmc, axis=1),
        pilot_CD_TPMC=np.sum(pilot_tpmc, axis=1),
        pilot_CD_SENTMAN=np.sum(pilot_sentman, axis=1),
        production_DSMC=production["DSMC"],
        production_TPMC=production["TPMC"],
        production_SENTMAN=production["SENTMAN"],
        production_CD_DSMC=np.sum(production["DSMC"], axis=1),
        production_CD_TPMC=np.sum(production["TPMC"], axis=1),
        production_CD_SENTMAN=np.sum(production["SENTMAN"], axis=1),
        production_ids_DSMC=ids["DSMC"],
        production_ids_TPMC=ids["TPMC"],
        production_ids_SENTMAN=ids["SENTMAN"],
        reference_DSMC=reference,
    )
    np.savez_compressed(
        pilot_dir / "field_pilot_statistics.npz",
        reference_field=np.mean(pilot_dsmc, axis=0),
    )
    config = {
        "case_name": "sensitivity-test",
        "high_fidelity": "DSMC",
        "control_variates": ["TPMC", "SENTMAN"],
        "costs": {"DSMC": 1.0, "TPMC": 0.2, "SENTMAN": 0.05},
        "field_allocation": {
            "enabled": True,
            "mode": "continuous_round",
            "bootstrap_repeats": 0,
            "mean_weight": 0.25,
            "second_moment_weight": 0.75,
            "random_seed": 19,
        },
        "allocation_constraints": {
            "budget": 6.0,
            "minimum_counts": {},
            "min_ratios": {},
            "max_ratios": {},
        },
        "pod": {"number_of_modes": 3},
        "validation": {
            "covariance_probe_count": 5,
            "covariance_probe_seed": 22,
            "fixed_ratios": {"TPMC": 2.0, "SENTMAN": 3.0},
        },
    }
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_field_sensitivity(
        load_config(config_path),
        results_dir=results,
        minimum_targets=[2, 4],
        reference_sizes=[4, 8],
        repetitions=2,
        random_seed=73,
    )
    assert result["rows"] == 40
    assert result["summary_rows"] == 20
    assert result["reference_rows"] == 4
    with (results / "sensitivity" / "m0_repetitions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["minimum_target"] for row in rows} == {"2", "4"}
    assert {row["reference_sample_count"] for row in rows} == {"4", "8"}
    assert {row["method"] for row in rows} == {
        "field-aware-m0-2",
        "field-aware-m0-4",
        "DSMC-only",
        "fixed-ratios",
        "two-fidelity-TPMC",
        "scalar-drag-allocation",
    }
    with (results / "sensitivity" / "m0_summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    required = {
        "mean_field_relative_error_minimum",
        "mean_field_relative_error_maximum",
        "mean_field_relative_error_win_rate_vs_dsmc_only",
        "mean_field_relative_error_win_rate_vs_fixed_ratios",
        "minimum_ritz_eigenvalue_median",
        "negative_eigenvalue_count_median",
    }
    assert required <= set(summaries[0])
    assert any("sensitivity-test" in title for title in titles)
    assert (results / "sensitivity" / "pod_subspace_diagnostics.json").is_file()
    assert (results / "sensitivity" / "case_findings.md").is_file()
