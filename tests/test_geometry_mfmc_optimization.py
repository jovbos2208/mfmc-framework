from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mfmc_campaign.geometry_mfmc_optimization import (
    allocate_two_fidelity_moments,
    analyze_geometry_mfmc_bundle,
    compare_geometry_estimates,
    confidence_aware_pareto_ids,
    decide_optimization_stop,
    estimate_geometry_mfmc,
    merge_geometry_mfmc_tpmc_results,
    select_geometry_mfmc_optimization_batch,
)
from mfmc_campaign.geometry_local_validation import (
    build_dsmc_validation_report,
    generate_local_refinement_manifest,
    select_dsmc_finalists,
)
from mfmc_campaign.geometry_optimization_workflow import initialize_workflow


def _bundle(geometry_count: int = 3, sample_count: int = 80) -> dict:
    rng = np.random.default_rng(91)
    evaluations = []
    geometry_ids = [f"cylinder_hex_wp5_{index:03d}" for index in range(geometry_count)]
    for geometry_index, geometry_id in enumerate(geometry_ids):
        for sample_index in range(sample_count):
            sample_id = f"wp1-crn-{sample_index:04d}"
            state = rng.normal()
            sentman = 0.0035 + geometry_index * 8.0e-5 + 1.5e-4 * state
            tpmc = sentman - geometry_index * 3.0e-5 + 2.0e-5 * rng.normal()
            for model_id, value, cost in (
                ("PICLas_TPMC", tpmc, 1.0),
                ("Sentman", sentman, 0.01),
            ):
                evaluations.append({
                    "geometry_id": geometry_id,
                    "canonical_sample_id": sample_id,
                    "model_id": model_id,
                    "drag_area_m2": float(value),
                    "cost_cpu_hours": cost,
                })
    return {
        "selected_geometry_ids": geometry_ids,
        "uncertainty_samples": {
            f"wp1-crn-{index:04d}": {"state": index / max(sample_count - 1, 1)}
            for index in range(sample_count)
        },
        "geometries": {geometry_id: {"design": {}} for geometry_id in geometry_ids},
        "evaluations": evaluations,
    }


def test_two_fidelity_allocation_respects_budget_and_nesting() -> None:
    values = np.linspace(0.8, 1.2, 5)
    result = allocate_two_fidelity_moments(
        values + 0.01,
        values,
        target_cost=1.0,
        control_cost=0.01,
        total_budget=20.0,
        pilot_count=5,
        maximum_target_count=80,
        maximum_control_count=80,
    )
    assert result["allocated_cost_cpu_hours"] <= 20.0
    assert result["production_control_count"] >= result["production_target_count"]
    assert result["total_target_count"] >= 7


def test_geometry_mfmc_estimates_moments_with_independent_pilot() -> None:
    result = estimate_geometry_mfmc(
        _bundle(geometry_count=1),
        "cylinder_hex_wp5_000",
        bootstrap_repeats=80,
    )
    assert result["allocation"]["allocated_cost_cpu_hours"] <= 20.0
    assert set(result["pilot_sample_ids"]).isdisjoint(result["production_target_sample_ids"])
    assert result["std_drag"] > 0.0
    assert result["mean_standard_error"] > 0.0
    assert len(result["mean_ci95"]) == len(result["std_ci95"]) == 2
    assert result["total_cost_cpu_hours"] <= 20.0 * result["reference_tpmc_cost_cpu_hours"]
    assert result["budget_contract_satisfied"]
    assert len(result["bootstrap_distributions"]["mean_drag"]) == 80


def test_measured_cost_variation_is_trimmed_to_hard_budget() -> None:
    bundle = _bundle(geometry_count=1)
    for row in bundle["evaluations"]:
        if row["model_id"] == "PICLas_TPMC" and row["canonical_sample_id"].endswith(("0018", "0019")):
            row["cost_cpu_hours"] = 8.0
    result = estimate_geometry_mfmc(bundle, "cylinder_hex_wp5_000", bootstrap_repeats=40)
    assert result["total_cost_cpu_hours"] <= result["hard_budget_cpu_hours"]
    assert result["allocation"]["cost_accounting"].startswith("sum_of_selected_measured")


def test_control_is_rejected_for_weak_correlation_and_no_bootstrap_gain() -> None:
    bundle = _bundle(geometry_count=1)
    weak = estimate_geometry_mfmc(
        bundle, "cylinder_hex_wp5_000", bootstrap_repeats=40,
        minimum_abs_control_correlation=0.999999,
    )
    assert weak["mean_estimator"] == "tpmc_only"
    assert "mean_control_inactive" in weak["quality_flags"]
    with patch(
        "mfmc_campaign.geometry_mfmc_optimization._bootstrap_moments",
        return_value=(
            np.linspace(-1.0, 1.0, 40), np.linspace(0.0, 1.0, 40),
            np.linspace(-0.01, 0.01, 40), np.linspace(0.10, 0.11, 40),
        ),
    ):
        rejected = estimate_geometry_mfmc(
            bundle, "cylinder_hex_wp5_000", bootstrap_repeats=40,
            minimum_abs_control_correlation=0.0,
        )
    assert rejected["mean_estimator"] == "tpmc_only"
    assert "mean_control_rejected_no_bootstrap_gain" in rejected["quality_flags"]


def test_geometry_mfmc_bundle_analysis_writes_ranked_seed_metrics(tmp_path: Path) -> None:
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(_bundle()))
    result = analyze_geometry_mfmc_bundle(
        source,
        tmp_path / "analysis",
        bootstrap_repeats=40,
    )
    assert result["n_geometries"] == 3
    assert result["best_robust_geometry_id"] in result["pareto_geometry_ids"]
    assert Path(result["metrics_csv"]).is_file()
    assert Path(result["details_json"]).is_file()
    assert len(result["common_random_number_sample_ids"]) == 80
    repeated = analyze_geometry_mfmc_bundle(
        source, tmp_path / "analysis_repeated", bootstrap_repeats=40,
    )
    first_details = json.loads(Path(result["details_json"]).read_text())
    repeated_details = json.loads(Path(repeated["details_json"]).read_text())
    assert first_details == repeated_details
    assert Path(result["budget_csv"]).is_file()
    assert Path(result["pareto_csv"]).is_file()
    assert Path(result["fallback_statistics_csv"]).is_file()


def test_surrogate_free_batch_excludes_evaluated_and_validation(tmp_path: Path) -> None:
    geometry_ids = [f"cylinder_hex_wp5_{index:03d}" for index in range(7)]
    designs = []
    for index, geometry_id in enumerate(geometry_ids):
        row = {
            "geometry_id": geometry_id,
            "eligible_for_model_fitting": index != 6,
        }
        for variable_index, name in enumerate((
            "nose_length_fraction", "tail_length_fraction", "width_height_ratio", "chamfer_fraction"
        )):
            row[f"normalized_{name}"] = 0.1 * index + 0.01 * variable_index
        designs.append(row)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "designs": designs,
        "baseline_geometry_id": geometry_ids[0],
        "validation_geometry_ids": [geometry_ids[6]],
    }))
    sentman = tmp_path / "sentman.csv"
    sentman.write_text(
        "geometry_id,mean_drag,std_drag\n"
        + "".join(f"{geometry_id},{0.003 + 1e-5 * index},{0.0002 - 1e-5 * index}\n" for index, geometry_id in enumerate(geometry_ids))
    )
    bundle = tmp_path / "paired.json"
    bundle.write_text(json.dumps({"selected_geometry_ids": geometry_ids[:2]}))
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "geometry_id,robust_score_vs_baseline\n"
        f"{geometry_ids[0]},1.0\n{geometry_ids[1]},0.95\n"
    )
    result = select_geometry_mfmc_optimization_batch(
        design, sentman, bundle, metrics, tmp_path / "selection.json", count=3
    )
    selected = {row["geometry_id"] for row in result["selected"]}
    assert len(selected) == 3
    assert not selected.intersection(geometry_ids[:2])
    assert geometry_ids[6] not in selected
    assert result["method"].endswith("without_geometry_surrogate")


def test_merge_tpmc_optimization_results_adds_sentman_controls(tmp_path: Path) -> None:
    geometry_id = "cylinder_hex_wp5_004"
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "selected_geometry_ids": [],
        "geometries": {},
        "evaluations": [],
        "uncertainty_samples": {"wp1-crn-0001": {"state": 1.0}},
    }))
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"geometries": [{
        "geometry_id": geometry_id,
        "design": {"reference_area_m2": 0.002},
        "reference_area_m2": 0.002,
        "manifest_json": "manifest.json",
        "mesh_path": "mesh.h5",
        "mesh_reference": "geometry/mesh.h5",
        "n_tetrahedra": 1,
        "n_hexahedra": 4,
        "hdf5_fingerprint": "abc",
        "selection_order": 0,
        "selection_basis": "test",
        "tpmc_workflow_config": "config.json",
    }]}))
    result_dir = tmp_path / "runs" / geometry_id / "PICLas_TPMC"
    result_dir.mkdir(parents=True)
    (result_dir / "piclas_results.json").write_text(json.dumps({
        "status": "collected",
        "sample_ids": ["wp1-crn-0001"],
        "costs_cpu_hours": [1.0],
        "values_by_qoi": {"C_D": [2.0]},
    }))
    sentman = tmp_path / "sentman.csv"
    sentman.write_text(
        "geometry_id,sample_id,C_D,drag_area_m2,cost_cpu_hours\n"
        f"{geometry_id},wp1-crn-0001,1.5,0.003,0.01\n"
    )
    result = merge_geometry_mfmc_tpmc_results(
        bundle, suite, tmp_path / "runs", sentman, tmp_path / "merged.json"
    )
    assert result["counts"][f"{geometry_id}/PICLas_TPMC"] == 1
    assert result["counts"][f"{geometry_id}/Sentman"] == 1


def _estimate(geometry_id: str, mean: float, std: float, spread: float, *, paired: bool = True) -> dict:
    offsets = np.linspace(-spread, spread, 101)
    return {
        "geometry_id": geometry_id,
        "mean_drag": mean,
        "std_drag": std,
        "bootstrap_distributions": {
            "mean_drag": (mean + offsets).tolist(),
            "std_drag": (std + 0.5 * offsets).tolist(),
        },
        "bootstrap_pairing": {
            "seed": 11 if paired else 12,
            "pilot_sample_ids": ["p0", "p1"],
            "production_target_sample_ids": ["s0", "s1", "s2"],
        },
    }


def test_paired_geometry_comparison_and_confidence_dominance() -> None:
    incumbent = _estimate("incumbent", 1.0, 0.2, 0.01)
    candidate = _estimate("candidate", 0.9, 0.15, 0.01)
    comparison = compare_geometry_estimates(candidate, incumbent)
    assert comparison["comparison_method"] == "paired_common_random_numbers"
    assert comparison["confidence_dominates"]
    pareto, _comparisons = confidence_aware_pareto_ids([incumbent, candidate])
    assert pareto == {"candidate"}


def test_unpaired_uncertain_candidate_is_not_confidently_dominant() -> None:
    incumbent = _estimate("incumbent", 1.0, 0.2, 0.1)
    candidate = _estimate("candidate", 0.99, 0.19, 0.1, paired=False)
    comparison = compare_geometry_estimates(candidate, incumbent, seed=9)
    assert comparison["comparison_method"] == "independent_bootstrap"
    assert not comparison["confidence_dominates"]
    pareto, _comparisons = confidence_aware_pareto_ids([incumbent, candidate], seed=9)
    assert pareto == {"incumbent", "candidate"}


def test_stop_logic_requires_multiple_stable_iterations() -> None:
    one = [{"confidence_pareto_geometry_ids": ["a"], "best_improvement_probability": 0.1}]
    assert decide_optimization_stop(one)["decision"] == "continue_optimization"
    two = one * 2
    assert decide_optimization_stop(two)["decision"] == "pareto_set_stable"
    assert decide_optimization_stop([], design_space_exhausted=True)["decision"] == "design_space_exhausted"


def _design_and_details(tmp_path: Path) -> tuple[Path, Path]:
    names = ("nose_length_fraction", "tail_length_fraction", "width_height_ratio", "chamfer_fraction")
    physical = ([0.20, 0.20, 1.0, 0.15], [0.18, 0.22, 0.9, 0.12], [0.24, 0.18, 1.1, 0.20])
    bounds = {name: pair for name, pair in zip(names, ([0.08, 0.32], [0.08, 0.32], [0.65, 1.55], [0.05, 0.30]))}
    designs = []
    for index, values in enumerate(physical):
        row = {"geometry_id": f"g{index}", "eligible_for_model_fitting": True}
        for name, value in zip(names, values):
            low, high = bounds[name]
            row[name] = value
            row[f"normalized_{name}"] = (value - low) / (high - low)
        designs.append(row)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "baseline_geometry_id": "g0", "validation_geometry_ids": ["reserved0"],
        "bounds": bounds, "designs": designs,
    }))
    details = tmp_path / "details.json"
    details.write_text(json.dumps({"geometries": [
        {"geometry_id": "g0", "mean_drag": 1.0, "std_drag": 0.2},
        {"geometry_id": "g1", "mean_drag": 0.9, "std_drag": 0.18},
        {"geometry_id": "g2", "mean_drag": 0.95, "std_drag": 0.12},
    ]}))
    return design, details


def test_local_candidates_respect_bounds_and_reserved_validation(tmp_path: Path) -> None:
    design, details = _design_and_details(tmp_path)
    result = generate_local_refinement_manifest(design, details, tmp_path / "local.json", count=3)
    assert len(result["candidates"]) == 3
    assert result["validation_geometry_ids_excluded"] == ["reserved0"]
    for candidate in result["candidates"]:
        assert all(0.0 <= value <= 1.0 for value in candidate["normalized_parameters"])
        assert candidate["validation"]["valid"]
        assert candidate["geometry_id"] != "reserved0"


def test_dsmc_finalists_and_paired_report_do_not_update_optimization(tmp_path: Path) -> None:
    design, details = _design_and_details(tmp_path)
    finalists = select_dsmc_finalists(design, details, tmp_path / "finalists.json", maximum_finalists=4)
    evaluations = []
    for finalist in finalists["finalists"]:
        for sample_index, delta in enumerate((-0.02, 0.0, 0.02)):
            for model, shift in (("PICLas_TPMC", 0.0), ("PICLas_DSMC", 0.01)):
                evaluations.append({
                    "geometry_id": finalist["geometry_id"], "model_id": model,
                    "canonical_sample_id": f"crn-{sample_index}", "drag_area_m2": 1.0 + delta + shift,
                })
    bundle = tmp_path / "validation_bundle.json"
    bundle.write_text(json.dumps({"evaluations": evaluations}))
    report = build_dsmc_validation_report(bundle, finalists["output_json"], tmp_path / "report", bootstrap_repeats=40)
    assert report["optimization_updated_from_dsmc"] is False
    assert Path(report["report_json"]).is_file() and Path(report["report_csv"]).is_file()


def test_workflow_rejects_non_64_mpi_configuration(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"budget_hf_equivalent": 20.0, "mpi_processes": 32}))
    try:
        initialize_workflow(config, tmp_path / "state.json")
    except ValueError as exc:
        assert "64 MPI" in str(exc)
    else:
        raise AssertionError("non-64 MPI workflow configuration was accepted")


def test_workflow_initialize_is_idempotent_without_erasing_state(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"budget_hf_equivalent": 20.0, "mpi_processes": 64}))
    state = tmp_path / "state.json"
    first = initialize_workflow(config, state)
    payload = json.loads(state.read_text())
    payload["sentinel"] = "preserved"
    state.write_text(json.dumps(payload))
    repeated = initialize_workflow(config, state)
    assert first["study_id"] == repeated["study_id"]
    assert repeated["idempotent_reuse"] is True
    assert json.loads(state.read_text())["sentinel"] == "preserved"
