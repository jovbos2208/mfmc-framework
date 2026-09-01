from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .geometry_design import VARIABLES


TARGET_MODEL = "PICLas_TPMC"
CONTROL_MODEL = "Sentman"
QOI = "drag_area_m2"


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _ci(values: np.ndarray, confidence_level: float = 0.95) -> list[float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = 0.5 * (1.0 - confidence_level)
    return [float(value) for value in np.quantile(values, [alpha, 1.0 - alpha])]


def _lookup(payload: Mapping[str, Any], model_id: str, geometry_id: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in payload["evaluations"]:
        if str(row["model_id"]) != model_id or str(row["geometry_id"]) != geometry_id:
            continue
        sample_id = str(row["canonical_sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate {model_id} evaluation for {(geometry_id, sample_id)}")
        result[sample_id] = row
    return result


def _positive_median_cost(rows: Mapping[str, Mapping[str, Any]], model_id: str) -> float:
    values = np.asarray([float(row.get("cost_cpu_hours", np.nan)) for row in rows.values()])
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        raise ValueError(f"No positive measured cost is available for {model_id}")
    return float(np.median(values))


def _beta(target: np.ndarray, control: np.ndarray) -> float:
    variance = float(np.var(control, ddof=1))
    if len(target) < 2 or not np.isfinite(variance) or variance <= 1.0e-30:
        return 0.0
    covariance = float(np.cov(target, control, ddof=1)[0, 1])
    return covariance / variance if np.isfinite(covariance) else 0.0


def _gated_beta(target: np.ndarray, control: np.ndarray, minimum_abs_correlation: float) -> float:
    return _beta(target, control) if abs(_correlation(target, control)) >= minimum_abs_correlation else 0.0


def _correlation(target: np.ndarray, control: np.ndarray) -> float:
    if len(target) < 2 or np.std(target, ddof=1) <= 1.0e-30 or np.std(control, ddof=1) <= 1.0e-30:
        return 0.0
    value = float(np.corrcoef(target, control)[0, 1])
    return value if np.isfinite(value) else 0.0


def _space_filling_sample_order(
    uncertainty_samples: Mapping[str, Mapping[str, Any]], sample_ids: Sequence[str]
) -> list[str]:
    ids = sorted(str(value) for value in sample_ids)
    excluded = {"database_index", "random_seed", "seed", "operations.seed", "density_scale"}
    numeric_names = sorted(
        name
        for name in set().union(*(uncertainty_samples[sample_id].keys() for sample_id in ids))
        if name not in excluded
        and all(isinstance(uncertainty_samples[sample_id].get(name), (int, float)) for sample_id in ids)
    )
    if not numeric_names:
        return ids
    values = np.asarray([
        [float(uncertainty_samples[sample_id][name]) for name in numeric_names] for sample_id in ids
    ])
    span = np.ptp(values, axis=0)
    span[span == 0.0] = 1.0
    points = (values - np.min(values, axis=0)) / span
    selected = [int(np.argmin(np.linalg.norm(points - 0.5, axis=1)))]
    while len(selected) < len(ids):
        remaining = [index for index in range(len(ids)) if index not in selected]
        selected.append(max(
            remaining,
            key=lambda index: (
                float(np.min(np.linalg.norm(points[index] - points[selected], axis=1))),
                ids[index],
            ),
        ))
    return [ids[index] for index in selected]


def allocate_two_fidelity_moments(
    pilot_target: np.ndarray,
    pilot_control: np.ndarray,
    *,
    target_cost: float,
    control_cost: float,
    total_budget: float,
    pilot_count: int,
    maximum_target_count: int,
    maximum_control_count: int,
    mean_weight: float = 0.5,
    second_moment_weight: float = 0.5,
) -> Dict[str, Any]:
    """Enumerate a nested integer allocation for first- and second-moment MFMC."""
    if target_cost <= 0.0 or control_cost <= 0.0 or total_budget <= 0.0:
        raise ValueError("MFMC costs and budget must be positive")
    if pilot_count < 2 or len(pilot_target) != pilot_count or len(pilot_control) != pilot_count:
        raise ValueError("Pilot arrays must have the declared count and at least two paired samples")
    if mean_weight < 0.0 or second_moment_weight < 0.0 or mean_weight + second_moment_weight <= 0.0:
        raise ValueError("At least one moment-allocation weight must be positive")
    pilot_cost = pilot_count * (target_cost + control_cost)
    remaining = total_budget - pilot_cost
    if remaining < 2.0 * (target_cost + control_cost):
        raise ValueError("Budget leaves fewer than two independent production pairs after the pilot")
    correlations = np.asarray([
        _correlation(pilot_target, pilot_control),
        _correlation(pilot_target**2, pilot_control**2),
    ])
    weights = np.asarray([mean_weight, second_moment_weight], dtype=float)
    weights /= np.sum(weights)
    best: tuple[float, int, int, float] | None = None
    candidate_count = 0
    max_target_production = maximum_target_count - pilot_count
    max_control_production = maximum_control_count - pilot_count
    for n_target in range(2, max_target_production + 1):
        for n_control in range(n_target, max_control_production + 1):
            production_cost = n_target * target_cost + n_control * control_cost
            if production_cost > remaining + 1.0e-12:
                break
            rho2 = np.clip(correlations * correlations, 0.0, 1.0)
            relative_variances = (1.0 - rho2) / n_target + rho2 / n_control
            objective = float(weights @ relative_variances)
            candidate_count += 1
            candidate = (objective, -n_target, -n_control, production_cost)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No feasible nested MFMC allocation exists for the available sample pools")
    objective, negative_target, negative_control, production_cost = best
    n_target = -negative_target
    n_control = -negative_control
    return {
        "pilot_count": int(pilot_count),
        "production_target_count": int(n_target),
        "production_control_count": int(n_control),
        "total_target_count": int(pilot_count + n_target),
        "total_control_count": int(pilot_count + n_control),
        "target_cost_cpu_hours": float(target_cost),
        "control_cost_cpu_hours": float(control_cost),
        "budget_cpu_hours": float(total_budget),
        "allocated_cost_cpu_hours": float(pilot_cost + production_cost),
        "unused_budget_cpu_hours": float(total_budget - pilot_cost - production_cost),
        "pilot_mean_correlation": float(correlations[0]),
        "pilot_second_moment_correlation": float(correlations[1]),
        "normalized_moment_variance_objective": float(objective),
        "enumerated_candidate_count": int(candidate_count),
    }


def _mfmc_moments(
    target: np.ndarray,
    control_paired: np.ndarray,
    control_full: np.ndarray,
    beta_mean: float,
    beta_second: float,
) -> tuple[float, float, float, float]:
    mean = float(np.mean(target) + beta_mean * (np.mean(control_full) - np.mean(control_paired)))
    second = float(
        np.mean(target**2)
        + beta_second * (np.mean(control_full**2) - np.mean(control_paired**2))
    )
    raw_variance = second - mean * mean
    return mean, second, raw_variance, float(np.sqrt(max(raw_variance, 0.0)))


def _bootstrap_moments(
    pilot_target: np.ndarray,
    pilot_control: np.ndarray,
    target: np.ndarray,
    control_paired: np.ndarray,
    control_extra: np.ndarray,
    *,
    repeats: int,
    seed: int,
    minimum_abs_control_correlation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if repeats < 2:
        raise ValueError("bootstrap_repeats must be at least two")
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    standard_deviations = np.empty(repeats, dtype=float)
    target_only_means = np.empty(repeats, dtype=float)
    target_only_standard_deviations = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        pilot_indices = rng.integers(0, len(pilot_target), len(pilot_target))
        paired_indices = rng.integers(0, len(target), len(target))
        pilot_y = pilot_target[pilot_indices]
        pilot_x = pilot_control[pilot_indices]
        beta_mean = _gated_beta(pilot_y, pilot_x, minimum_abs_control_correlation)
        beta_second = _gated_beta(pilot_y**2, pilot_x**2, minimum_abs_control_correlation)
        y = target[paired_indices]
        x_paired = control_paired[paired_indices]
        if len(control_extra):
            extra_indices = rng.integers(0, len(control_extra), len(control_extra))
            x_full = np.concatenate([x_paired, control_extra[extra_indices]])
        else:
            x_full = x_paired
        means[repeat], _second, _variance, standard_deviations[repeat] = _mfmc_moments(
            y, x_paired, x_full, beta_mean, beta_second
        )
        target_only_means[repeat] = float(np.mean(y))
        target_only_standard_deviations[repeat] = float(np.std(y, ddof=1))
    return means, standard_deviations, target_only_means, target_only_standard_deviations


def estimate_geometry_mfmc(
    bundle: Mapping[str, Any],
    geometry_id: str,
    *,
    budget_hf_equivalent: float = 20.0,
    pilot_count: int = 5,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260822,
    mean_weight: float = 0.5,
    second_moment_weight: float = 0.5,
    target_cost_override: float | None = None,
    control_cost_override: float | None = None,
    common_sample_order: Sequence[str] | None = None,
    minimum_abs_control_correlation: float = 0.5,
) -> Dict[str, Any]:
    """Estimate TPMC drag mean/std with Sentman as a nested control variate."""
    target_rows = _lookup(bundle, TARGET_MODEL, geometry_id)
    control_rows = _lookup(bundle, CONTROL_MODEL, geometry_id)
    if not target_rows or not control_rows:
        raise ValueError(f"{geometry_id} requires both {TARGET_MODEL} and {CONTROL_MODEL} evaluations")
    available_common = set(target_rows) & set(control_rows)
    if common_sample_order is None:
        common_ids = sorted(available_common)
    else:
        common_ids = [str(value) for value in common_sample_order if str(value) in available_common]
        if len(common_ids) != len(common_sample_order):
            raise ValueError(f"{geometry_id} is missing samples from the prescribed common order")
    if len(common_ids) <= pilot_count + 1:
        raise ValueError(f"{geometry_id} has too few paired TPMC/Sentman samples for pilot and production")
    if len(control_rows) <= pilot_count + 1:
        raise ValueError(f"{geometry_id} has too few Sentman samples")
    target_cost = float(target_cost_override) if target_cost_override is not None else _positive_median_cost(target_rows, TARGET_MODEL)
    control_cost = float(control_cost_override) if control_cost_override is not None else _positive_median_cost(control_rows, CONTROL_MODEL)
    available_target = np.asarray([float(target_rows[sample_id][QOI]) for sample_id in common_ids])
    available_control = np.asarray([float(control_rows[sample_id][QOI]) for sample_id in common_ids])
    total_budget = float(budget_hf_equivalent) * target_cost
    pilot_ids = common_ids[:pilot_count]
    pilot_target = np.asarray([float(target_rows[sample_id][QOI]) for sample_id in pilot_ids])
    pilot_control = np.asarray([float(control_rows[sample_id][QOI]) for sample_id in pilot_ids])
    allocation = allocate_two_fidelity_moments(
        pilot_target,
        pilot_control,
        target_cost=target_cost,
        control_cost=control_cost,
        total_budget=total_budget,
        pilot_count=pilot_count,
        maximum_target_count=len(common_ids),
        maximum_control_count=len(control_rows),
        mean_weight=mean_weight,
        second_moment_weight=second_moment_weight,
    )
    n_target = int(allocation["production_target_count"])
    n_control = int(allocation["production_control_count"])
    paired_ids = common_ids[pilot_count:pilot_count + n_target]
    excluded = set(pilot_ids) | set(paired_ids)
    extra_ids = [sample_id for sample_id in sorted(control_rows) if sample_id not in excluded]
    extra_ids = extra_ids[:n_control - n_target]
    planned_cost = float(allocation["allocated_cost_cpu_hours"])

    def measured_cost() -> float:
        return float(
            sum(float(target_rows[sample_id]["cost_cpu_hours"]) for sample_id in pilot_ids + paired_ids)
            + sum(float(control_rows[sample_id]["cost_cpu_hours"]) for sample_id in pilot_ids + paired_ids + extra_ids)
        )

    while measured_cost() > total_budget + 1.0e-12:
        if extra_ids:
            extra_ids.pop()
            n_control -= 1
        elif len(paired_ids) > 2:
            paired_ids.pop()
            n_target -= 1
            n_control -= 1
        else:
            raise ValueError(f"{geometry_id} has no feasible allocation under measured run costs")
    actual_cost = measured_cost()
    allocation.update({
        "production_target_count": n_target,
        "production_control_count": n_control,
        "total_target_count": pilot_count + n_target,
        "total_control_count": pilot_count + n_control,
        "planned_median_cost_cpu_hours": planned_cost,
        "allocated_cost_cpu_hours": actual_cost,
        "unused_budget_cpu_hours": total_budget - actual_cost,
        "cost_accounting": "sum_of_selected_measured_run_costs_including_pilot",
    })
    full_control_ids = paired_ids + extra_ids
    if len(full_control_ids) != n_control:
        raise ValueError("Control sample pool cannot satisfy the selected nested allocation")
    target = np.asarray([float(target_rows[sample_id][QOI]) for sample_id in paired_ids])
    control_paired = np.asarray([float(control_rows[sample_id][QOI]) for sample_id in paired_ids])
    control_full = np.asarray([float(control_rows[sample_id][QOI]) for sample_id in full_control_ids])
    control_extra = np.asarray([float(control_rows[sample_id][QOI]) for sample_id in extra_ids])
    if not 0.0 <= minimum_abs_control_correlation < 1.0:
        raise ValueError("minimum_abs_control_correlation must be in [0, 1)")
    beta_mean = _gated_beta(pilot_target, pilot_control, minimum_abs_control_correlation)
    beta_second = _gated_beta(
        pilot_target**2, pilot_control**2, minimum_abs_control_correlation
    )
    mfmc_mean, second, raw_variance, mfmc_standard_deviation = _mfmc_moments(
        target, control_paired, control_full, beta_mean, beta_second
    )
    bootstrap_mean, bootstrap_std, bootstrap_target_mean, bootstrap_target_std = _bootstrap_moments(
        pilot_target,
        pilot_control,
        target,
        control_paired,
        control_extra,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
        minimum_abs_control_correlation=minimum_abs_control_correlation,
    )
    mfmc_mean_se = float(np.std(bootstrap_mean, ddof=1))
    mfmc_std_se = float(np.std(bootstrap_std, ddof=1))
    target_mean_se = float(np.std(bootstrap_target_mean, ddof=1))
    target_std_se = float(np.std(bootstrap_target_std, ddof=1))
    mean_control_accepted = beta_mean != 0.0 and mfmc_mean_se < target_mean_se
    std_control_accepted = beta_second != 0.0 and raw_variance >= 0.0 and mfmc_std_se < target_std_se
    target_only_mean = float(np.mean(target))
    target_only_std = float(np.std(target, ddof=1))
    mean = mfmc_mean if mean_control_accepted else target_only_mean
    standard_deviation = mfmc_standard_deviation if std_control_accepted else target_only_std
    selected_bootstrap_mean = bootstrap_mean if mean_control_accepted else bootstrap_target_mean
    selected_bootstrap_std = bootstrap_std if std_control_accepted else bootstrap_target_std
    flags: list[str] = []
    if abs(float(allocation["pilot_mean_correlation"])) < minimum_abs_control_correlation:
        flags.append("weak_mean_control_correlation")
    if abs(float(allocation["pilot_second_moment_correlation"])) < minimum_abs_control_correlation:
        flags.append("weak_second_moment_control_correlation")
    if raw_variance < 0.0:
        flags.append("negative_raw_variance_clipped")
    if beta_mean == 0.0:
        flags.append("mean_control_inactive")
    elif not mean_control_accepted:
        flags.append("mean_control_rejected_no_bootstrap_gain")
    if beta_second == 0.0:
        flags.append("second_moment_control_inactive")
    elif not std_control_accepted:
        flags.append("std_control_rejected_no_bootstrap_gain")
    if int(allocation["total_control_count"]) == len(control_rows):
        flags.append("control_pool_exhausted")
    allocated_cost = actual_cost
    if allocated_cost > total_budget + 1.0e-12:
        raise AssertionError(
            f"{geometry_id} allocation exceeds its hard budget: {allocated_cost} > {total_budget}"
        )
    fallback_reasons = {
        "mean": [flag for flag in flags if flag.startswith("mean_control_") or flag == "weak_mean_control_correlation"],
        "std": [flag for flag in flags if flag.startswith(("second_moment_control_", "std_control_")) or flag == "weak_second_moment_control_correlation"],
    }
    estimator_class = (
        "sentman_both_moments"
        if mean_control_accepted and std_control_accepted
        else "sentman_mean_only"
        if mean_control_accepted
        else "sentman_second_moment_only"
        if std_control_accepted
        else "tpmc_only"
    )
    tpmc_only_count = min(len(common_ids), int(total_budget // target_cost))
    return {
        "geometry_id": geometry_id,
        "target_model": TARGET_MODEL,
        "control_model": CONTROL_MODEL,
        "qoi": QOI,
        "budget_hf_equivalent": float(budget_hf_equivalent),
        "reference_tpmc_cost_cpu_hours": target_cost,
        "hard_budget_cpu_hours": total_budget,
        "total_cost_cpu_hours": allocated_cost,
        "budget_contract_satisfied": True,
        "allocation": allocation,
        "pilot_sample_ids": pilot_ids,
        "production_target_sample_ids": paired_ids,
        "production_control_only_sample_ids": extra_ids,
        "beta_mean": float(beta_mean),
        "beta_second_moment": float(beta_second),
        "minimum_abs_control_correlation": float(minimum_abs_control_correlation),
        "available_pair_mean_correlation_diagnostic": _correlation(
            available_target, available_control
        ),
        "available_pair_second_moment_correlation_diagnostic": _correlation(
            available_target**2, available_control**2
        ),
        "mean_estimator": "mfmc" if mean_control_accepted else "tpmc_only",
        "std_estimator": "mfmc_moments" if std_control_accepted else "tpmc_only",
        "estimator_class": estimator_class,
        "fallback_reasons": fallback_reasons,
        "mean_drag": mean,
        "second_moment_drag": second,
        "variance_drag": float(standard_deviation**2),
        "std_drag": standard_deviation,
        "raw_mfmc_mean_drag": mfmc_mean,
        "raw_mfmc_std_drag": mfmc_standard_deviation,
        "tpmc_only_mean_drag": target_only_mean,
        "tpmc_only_std_drag": target_only_std,
        "mean_standard_error": float(np.std(selected_bootstrap_mean, ddof=1)),
        "std_standard_error": float(np.std(selected_bootstrap_std, ddof=1)),
        "mean_ci95": _ci(selected_bootstrap_mean),
        "std_ci95": _ci(selected_bootstrap_std),
        "bootstrap_distributions": {
            "mean_drag": [float(value) for value in selected_bootstrap_mean],
            "std_drag": [float(value) for value in selected_bootstrap_std],
        },
        "bootstrap_pairing": {
            "seed": int(bootstrap_seed),
            "pilot_sample_ids": pilot_ids,
            "production_target_sample_ids": paired_ids,
        },
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_seed": int(bootstrap_seed),
        "equal_budget_tpmc_only_count": int(tpmc_only_count),
        "quality_flags": flags,
    }


def compare_geometry_estimates(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    confidence_level: float = 0.95,
    improvement_probability_threshold: float = 0.95,
    seed: int = 20260822,
) -> Dict[str, Any]:
    """Compare two estimates, preserving CRN pairing whenever bootstrap draws align."""
    if not 0.5 < improvement_probability_threshold < 1.0:
        raise ValueError("improvement_probability_threshold must be in (0.5, 1)")
    candidate_bootstrap = candidate.get("bootstrap_distributions", {})
    incumbent_bootstrap = incumbent.get("bootstrap_distributions", {})
    candidate_pairing = candidate.get("bootstrap_pairing", {})
    incumbent_pairing = incumbent.get("bootstrap_pairing", {})
    common_ids = sorted(
        set(map(str, candidate_pairing.get("production_target_sample_ids", [])))
        & set(map(str, incumbent_pairing.get("production_target_sample_ids", [])))
    )
    paired = bool(common_ids) and candidate_pairing == incumbent_pairing
    comparisons: Dict[str, Any] = {}
    method = "independent_bootstrap"
    for objective in ("mean_drag", "std_drag"):
        left = np.asarray(candidate_bootstrap.get(objective, []), dtype=float)
        right = np.asarray(incumbent_bootstrap.get(objective, []), dtype=float)
        if not len(left) or not len(right):
            raise ValueError(f"Missing bootstrap distribution for {objective}")
        if paired and len(left) == len(right):
            differences = left - right
            method = "paired_common_random_numbers"
        else:
            draw_count = max(len(left), len(right))
            rng = np.random.default_rng(_stable_seed(seed, str(candidate["geometry_id"]), str(incumbent["geometry_id"]), objective))
            differences = (
                left[rng.integers(0, len(left), draw_count)]
                - right[rng.integers(0, len(right), draw_count)]
            )
        interval = _ci(differences, confidence_level)
        probability = float(np.mean(differences < 0.0))
        comparisons[objective] = {
            "difference_candidate_minus_incumbent": float(candidate[objective] - incumbent[objective]),
            "confidence_interval": interval,
            "probability_candidate_improves": probability,
            "statistically_improved": bool(
                interval[1] < 0.0 and probability >= improvement_probability_threshold
            ),
        }
    confidence_dominates = bool(
        all(comparisons[name]["confidence_interval"][1] <= 0.0 for name in comparisons)
        and any(comparisons[name]["statistically_improved"] for name in comparisons)
    )
    return {
        "candidate_geometry_id": str(candidate["geometry_id"]),
        "incumbent_geometry_id": str(incumbent["geometry_id"]),
        "comparison_method": method,
        "common_crn_sample_ids": common_ids if paired else [],
        "confidence_level": float(confidence_level),
        "improvement_probability_threshold": float(improvement_probability_threshold),
        "objectives": comparisons,
        "confidence_dominates": confidence_dominates,
    }


def confidence_aware_pareto_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence_level: float = 0.95,
    improvement_probability_threshold: float = 0.95,
    seed: int = 20260822,
) -> tuple[set[str], list[Dict[str, Any]]]:
    dominated: set[str] = set()
    comparisons: list[Dict[str, Any]] = []
    for candidate in rows:
        for challenger in rows:
            if candidate is challenger:
                continue
            comparison = compare_geometry_estimates(
                challenger,
                candidate,
                confidence_level=confidence_level,
                improvement_probability_threshold=improvement_probability_threshold,
                seed=seed,
            )
            comparisons.append(comparison)
            if comparison["confidence_dominates"]:
                dominated.add(str(candidate["geometry_id"]))
                break
    return {str(row["geometry_id"]) for row in rows} - dominated, comparisons


def decide_optimization_stop(
    history: Sequence[Mapping[str, Any]],
    *,
    budget_exhausted: bool = False,
    design_space_exhausted: bool = False,
    stable_iterations_required: int = 2,
    minimum_improvement_probability: float = 0.95,
    objective_uncertainty_target: float | None = None,
) -> Dict[str, Any]:
    """Return one deterministic, machine-readable optimization stop decision."""
    if stable_iterations_required < 2:
        raise ValueError("stable_iterations_required must be at least two")
    recent = list(history[-stable_iterations_required:])
    stable_pareto = len(recent) == stable_iterations_required and len({
        tuple(sorted(map(str, row.get("confidence_pareto_geometry_ids", [])))) for row in recent
    }) == 1
    no_improvement = len(recent) == stable_iterations_required and all(
        float(row.get("best_improvement_probability", 0.0)) < minimum_improvement_probability
        for row in recent
    )
    uncertainty_met = bool(recent) and objective_uncertainty_target is not None and all(
        max(float(row.get("best_mean_standard_error", np.inf)), float(row.get("best_std_standard_error", np.inf)))
        <= objective_uncertainty_target
        for row in recent
    )
    if budget_exhausted:
        decision = "budget_exhausted"
    elif design_space_exhausted:
        decision = "design_space_exhausted"
    elif uncertainty_met:
        decision = "objective_uncertainty_target_met"
    elif stable_pareto:
        decision = "pareto_set_stable"
    elif no_improvement:
        decision = "no_significant_improvement"
    else:
        decision = "continue_optimization"
    return {
        "decision": decision,
        "stop": decision != "continue_optimization",
        "stable_iterations_required": int(stable_iterations_required),
        "iterations_considered": len(recent),
        "criteria": {
            "budget_exhausted": bool(budget_exhausted),
            "design_space_exhausted": bool(design_space_exhausted),
            "no_significant_improvement": no_improvement,
            "pareto_set_stable": stable_pareto,
            "objective_uncertainty_target_met": uncertainty_met,
        },
    }


def _pareto_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    values = np.asarray([[float(row["mean_drag"]), float(row["std_drag"])] for row in rows])
    result: set[str] = set()
    for index, row in enumerate(rows):
        dominated = np.any(
            np.all(values <= values[index], axis=1)
            & np.any(values < values[index], axis=1)
        )
        if not dominated:
            result.add(str(row["geometry_id"]))
    return result


def analyze_geometry_mfmc_bundle(
    bundle_json: str | Path,
    output_dir: str | Path,
    *,
    budget_hf_equivalent: float = 20.0,
    pilot_count: int = 5,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260822,
    mean_objective_weight: float = 0.5,
    std_objective_weight: float = 0.5,
    target_cost_override: float | None = None,
    control_cost_override: float | None = None,
    minimum_abs_control_correlation: float = 0.5,
    confidence_level: float = 0.95,
    improvement_probability_threshold: float = 0.95,
) -> Dict[str, Any]:
    source = Path(bundle_json).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    geometry_ids = [str(value) for value in payload["selected_geometry_ids"]]
    common_target_ids: set[str] | None = None
    for geometry_id in geometry_ids:
        target_ids = set(_lookup(payload, TARGET_MODEL, geometry_id))
        control_ids = set(_lookup(payload, CONTROL_MODEL, geometry_id))
        available = target_ids & control_ids
        common_target_ids = available if common_target_ids is None else common_target_ids & available
    if not common_target_ids:
        raise ValueError("Selected geometries have no common TPMC/Sentman sample IDs")
    common_sample_order = _space_filling_sample_order(
        payload["uncertainty_samples"], sorted(common_target_ids)
    )
    results = [
        estimate_geometry_mfmc(
            payload,
            geometry_id,
            budget_hf_equivalent=budget_hf_equivalent,
            pilot_count=pilot_count,
            bootstrap_repeats=bootstrap_repeats,
            bootstrap_seed=bootstrap_seed,
            mean_weight=mean_objective_weight,
            second_moment_weight=std_objective_weight,
            target_cost_override=target_cost_override,
            control_cost_override=control_cost_override,
            common_sample_order=common_sample_order,
            minimum_abs_control_correlation=minimum_abs_control_correlation,
        )
        for geometry_id in geometry_ids
    ]
    baseline = next((row for row in results if row["geometry_id"].endswith("_000")), results[0])
    pareto = _pareto_ids(results)
    confidence_pareto, comparisons = confidence_aware_pareto_ids(
        results,
        confidence_level=confidence_level,
        improvement_probability_threshold=improvement_probability_threshold,
        seed=bootstrap_seed,
    )
    metric_rows: list[Dict[str, Any]] = []
    weight_sum = mean_objective_weight + std_objective_weight
    if weight_sum <= 0.0:
        raise ValueError("Optimization objective weights must have a positive sum")
    for result in results:
        robust_score = (
            mean_objective_weight * float(result["mean_drag"]) / float(baseline["mean_drag"])
            + std_objective_weight * float(result["std_drag"]) / max(float(baseline["std_drag"]), 1.0e-30)
        ) / weight_sum
        allocation = result["allocation"]
        metric_rows.append({
            "geometry_id": result["geometry_id"],
            "mean_drag": result["mean_drag"],
            "mean_standard_error": result["mean_standard_error"],
            "std_drag": result["std_drag"],
            "std_standard_error": result["std_standard_error"],
            "robust_score_vs_baseline": robust_score,
            "pareto_mean_std": result["geometry_id"] in pareto,
            "confidence_pareto_mean_std": result["geometry_id"] in confidence_pareto,
            "n_tpmc_total": allocation["total_target_count"],
            "n_sentman_total": allocation["total_control_count"],
            "pilot_mean_correlation": allocation["pilot_mean_correlation"],
            "pilot_second_moment_correlation": allocation["pilot_second_moment_correlation"],
            "available_pair_mean_correlation_diagnostic": result[
                "available_pair_mean_correlation_diagnostic"
            ],
            "available_pair_second_moment_correlation_diagnostic": result[
                "available_pair_second_moment_correlation_diagnostic"
            ],
            "mean_estimator": result["mean_estimator"],
            "std_estimator": result["std_estimator"],
            "estimator_class": result["estimator_class"],
            "beta_mean": result["beta_mean"],
            "beta_second_moment": result["beta_second_moment"],
            "reference_tpmc_cost_cpu_hours": result["reference_tpmc_cost_cpu_hours"],
            "control_cost_cpu_hours": allocation["control_cost_cpu_hours"],
            "hard_budget_cpu_hours": result["hard_budget_cpu_hours"],
            "allocated_cost_cpu_hours": allocation["allocated_cost_cpu_hours"],
            "budget_contract_satisfied": result["budget_contract_satisfied"],
            "mean_fallback_reasons": ";".join(result["fallback_reasons"]["mean"]),
            "std_fallback_reasons": ";".join(result["fallback_reasons"]["std"]),
            "quality_flags": ";".join(result["quality_flags"]),
        })
    metric_rows.sort(key=lambda row: (float(row["robust_score_vs_baseline"]), str(row["geometry_id"])))
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    metrics_path = target / "geometry_mfmc_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    details_path = target / "geometry_mfmc_details.json"
    details_path.write_text(json.dumps({"schema_version": 2, "geometries": results}, indent=2, sort_keys=True) + "\n")
    comparisons_path = target / "geometry_mfmc_comparisons.json"
    comparisons_path.write_text(json.dumps({
        "schema_version": 1,
        "confidence_level": confidence_level,
        "improvement_probability_threshold": improvement_probability_threshold,
        "comparisons": comparisons,
    }, indent=2, sort_keys=True) + "\n")
    pareto_path = target / "geometry_pareto.csv"
    pareto_rows = [
        row for row in metric_rows
        if row["pareto_mean_std"] or row["confidence_pareto_mean_std"]
    ]
    with pareto_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pareto_rows[0]))
        writer.writeheader()
        writer.writerows(pareto_rows)
    budget_path = target / "geometry_budget.csv"
    budget_fields = [
        "geometry_id", "n_tpmc_total", "n_sentman_total",
        "reference_tpmc_cost_cpu_hours", "control_cost_cpu_hours",
        "hard_budget_cpu_hours", "allocated_cost_cpu_hours", "budget_contract_satisfied",
    ]
    with budget_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=budget_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in budget_fields} for row in metric_rows)
    estimator_classes = (
        "sentman_both_moments", "sentman_mean_only", "sentman_second_moment_only", "tpmc_only"
    )
    fallback_rows = [
        {"estimator_class": name, "geometry_count": sum(row["estimator_class"] == name for row in metric_rows)}
        for name in estimator_classes
    ]
    fallback_path = target / "geometry_estimator_fallback.csv"
    with fallback_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fallback_rows[0]))
        writer.writeheader()
        writer.writerows(fallback_rows)
    any_control_accepted = any(
        result["mean_estimator"] == "mfmc" or result["std_estimator"] == "mfmc_moments"
        for result in results
    )
    summary = {
        "schema_version": 2,
        "status": (
            "optimization_seed_metrics_complete_with_mfmc"
            if any_control_accepted
            else "optimization_seed_metrics_complete_tpmc_only_fallback"
        ),
        "bundle_json": str(source),
        "target_model": TARGET_MODEL,
        "control_model": CONTROL_MODEL,
        "budget_hf_equivalent_per_geometry": float(budget_hf_equivalent),
        "pilot_count_per_geometry": int(pilot_count),
        "minimum_abs_control_correlation": float(minimum_abs_control_correlation),
        "bootstrap_repeats": int(bootstrap_repeats),
        "common_random_number_sample_ids": common_sample_order,
        "n_geometries": len(results),
        "any_sentman_control_accepted": any_control_accepted,
        "baseline_geometry_id": baseline["geometry_id"],
        "best_robust_geometry_id": metric_rows[0]["geometry_id"],
        "pareto_geometry_ids": sorted(pareto),
        "confidence_pareto_geometry_ids": sorted(confidence_pareto),
        "objective": {
            "mean_weight": float(mean_objective_weight),
            "std_weight": float(std_objective_weight),
            "normalization": "ratio_to_baseline_mfmc_estimate",
        },
        "metrics_csv": str(metrics_path),
        "details_json": str(details_path),
        "comparisons_json": str(comparisons_path),
        "pareto_csv": str(pareto_path),
        "budget_csv": str(budget_path),
        "fallback_statistics_csv": str(fallback_path),
    }
    manifest_path = target / "geometry_mfmc_optimization_seed.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {**summary, "manifest_json": str(manifest_path)}


def _read_csv(path: str | Path) -> list[Dict[str, str]]:
    with Path(path).resolve().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _unit_scale(values: np.ndarray, *, smaller_is_better: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    span = float(np.ptp(array))
    scaled = np.zeros_like(array) if span <= 1.0e-15 else (array - np.min(array)) / span
    return 1.0 - scaled if smaller_is_better else scaled


def select_geometry_mfmc_optimization_batch(
    design_manifest_json: str | Path,
    sentman_metrics_csv: str | Path,
    paired_bundle_json: str | Path,
    current_mfmc_metrics_csv: str | Path,
    output_json: str | Path,
    *,
    count: int = 4,
    iteration: int = 1,
    mean_objective_weight: float = 0.5,
    std_objective_weight: float = 0.5,
) -> Dict[str, Any]:
    """Select a derivative-free batch without a geometry response surrogate."""
    design = json.loads(Path(design_manifest_json).resolve().read_text(encoding="utf-8"))
    bundle = json.loads(Path(paired_bundle_json).resolve().read_text(encoding="utf-8"))
    sentman = {row["geometry_id"]: row for row in _read_csv(sentman_metrics_csv)}
    current_metrics = _read_csv(current_mfmc_metrics_csv)
    current_ids = set(str(value) for value in bundle["selected_geometry_ids"])
    validation_ids = set(str(value) for value in design["validation_geometry_ids"])
    design_by_id = {str(row["geometry_id"]): row for row in design["designs"]}
    candidates = [
        row for row in design["designs"]
        if bool(row["eligible_for_model_fitting"])
        and str(row["geometry_id"]) not in current_ids
        and str(row["geometry_id"]) not in validation_ids
    ]
    if count < 1:
        raise ValueError("Optimization batch size must be positive")
    if not candidates:
        raise ValueError("Optimization design space is exhausted")
    count = min(count, len(candidates))
    missing = [row["geometry_id"] for row in candidates if row["geometry_id"] not in sentman]
    if missing:
        raise ValueError(f"Sentman metrics are missing for optimization candidates: {missing[:10]}")
    if not current_metrics:
        raise ValueError("At least one completed geometry MFMC estimate is required")
    supported = [
        row for row in current_metrics
        if str(row.get("confidence_pareto_mean_std", "")).strip().lower() in {"1", "true", "yes"}
    ]
    incumbent_pool = supported or current_metrics
    best = min(incumbent_pool, key=lambda row: (float(row["robust_score_vs_baseline"]), row["geometry_id"]))
    best_point = np.asarray([
        float(design_by_id[best["geometry_id"]][f"normalized_{name}"]) for name in VARIABLES
    ])
    current_points = np.asarray([
        [float(design_by_id[geometry_id][f"normalized_{name}"]) for name in VARIABLES]
        for geometry_id in sorted(current_ids)
    ])
    candidate_points = np.asarray([
        [float(row[f"normalized_{name}"]) for name in VARIABLES] for row in candidates
    ])
    baseline_sentman = sentman.get(str(design["baseline_geometry_id"]))
    if baseline_sentman is None:
        raise ValueError("Sentman metrics do not contain the baseline geometry")
    weight_sum = mean_objective_weight + std_objective_weight
    if weight_sum <= 0.0:
        raise ValueError("Optimization weights must have a positive sum")
    sentman_scores = np.asarray([
        (
            mean_objective_weight * float(sentman[row["geometry_id"]]["mean_drag"]) / float(baseline_sentman["mean_drag"])
            + std_objective_weight * float(sentman[row["geometry_id"]]["std_drag"]) / max(float(baseline_sentman["std_drag"]), 1.0e-30)
        ) / weight_sum
        for row in candidates
    ])
    distance_to_best = np.linalg.norm(candidate_points - best_point[None, :], axis=1)
    distance_to_evaluated = np.min(
        np.linalg.norm(candidate_points[:, None, :] - current_points[None, :, :], axis=2), axis=1
    )
    sentman_values = np.asarray([
        [float(sentman[row["geometry_id"]]["mean_drag"]), float(sentman[row["geometry_id"]]["std_drag"])]
        for row in candidates
    ])
    sentman_pareto: set[int] = set()
    for index in range(len(candidates)):
        dominated = np.any(
            np.all(sentman_values <= sentman_values[index], axis=1)
            & np.any(sentman_values < sentman_values[index], axis=1)
        )
        if not dominated:
            sentman_pareto.add(index)
    selected: list[int] = []
    bases: list[str] = []

    def add(index: int, basis: str) -> None:
        if index not in selected and len(selected) < count:
            selected.append(index)
            bases.append(basis)

    add(int(np.argmin(sentman_scores)), "sentman_robust_objective")
    if len(selected) < count:
        local_order = np.argsort(distance_to_best)
        add(next(int(index) for index in local_order if int(index) not in selected), "local_pattern_search")
    if len(selected) < count:
        coverage_order = np.argsort(-distance_to_evaluated)
        add(next(int(index) for index in coverage_order if int(index) not in selected), "geometry_space_filling")
    while len(selected) < count:
        score = (
            0.50 * _unit_scale(sentman_scores, smaller_is_better=True)
            + 0.30 * _unit_scale(distance_to_best, smaller_is_better=True)
            + 0.20 * _unit_scale(distance_to_evaluated)
        )
        score[[int(index) for index in selected]] = -np.inf
        add(int(np.argmax(score)), "combined_derivative_free_search")
    diagnostics: list[Dict[str, Any]] = []
    combined_scores = (
        0.50 * _unit_scale(sentman_scores, smaller_is_better=True)
        + 0.30 * _unit_scale(distance_to_best, smaller_is_better=True)
        + 0.20 * _unit_scale(distance_to_evaluated)
    )
    for index, row in enumerate(candidates):
        metrics = sentman[row["geometry_id"]]
        diagnostics.append({
            "geometry_id": row["geometry_id"],
            "sentman_mean_drag": float(metrics["mean_drag"]),
            "sentman_std_drag": float(metrics["std_drag"]),
            "sentman_robust_score_vs_baseline": float(sentman_scores[index]),
            "sentman_pareto_mean_std": index in sentman_pareto,
            "distance_to_current_best": float(distance_to_best[index]),
            "distance_to_evaluated_set": float(distance_to_evaluated[index]),
            "selection_score": float(combined_scores[index]),
        })
    by_id = {row["geometry_id"]: row for row in diagnostics}
    rows = [
        {
            "selection_order": order,
            "selection_basis": basis,
            "geometry_id": candidates[index]["geometry_id"],
            **by_id[candidates[index]["geometry_id"]],
        }
        for order, (index, basis) in enumerate(zip(selected, bases))
    ]
    result = {
        "schema_version": 1,
        "study_id": f"vleo_cylinder_hex_wp6_mfmc_optimization_iteration_{iteration}",
        "iteration": int(iteration),
        "method": "derivative_free_discrete_batch_without_geometry_surrogate",
        "count": len(rows),
        "selected": rows,
        "current_best_geometry_id": best["geometry_id"],
        "incumbent_selection": "confidence_pareto_then_robust_score" if supported else "robust_score_fallback",
        "previous_training_geometry_ids": sorted(current_ids),
        "untouched_validation_geometry_ids": sorted(validation_ids),
        "candidate_diagnostics": diagnostics,
    }
    target = Path(output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = target.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {**result, "output_json": str(target), "output_csv": str(csv_path)}


def merge_geometry_mfmc_tpmc_results(
    bundle_json: str | Path,
    suite_json: str | Path,
    run_root: str | Path,
    sentman_results_csv: str | Path,
    output_json: str | Path,
) -> Dict[str, Any]:
    """Merge a TPMC-only optimization batch and its existing Sentman controls."""
    bundle = json.loads(Path(bundle_json).resolve().read_text(encoding="utf-8"))
    suite = json.loads(Path(suite_json).resolve().read_text(encoding="utf-8"))
    root = Path(run_root).resolve()
    sentman_rows = _read_csv(sentman_results_csv)
    sentman_by_geometry: Dict[str, list[Dict[str, str]]] = {}
    for row in sentman_rows:
        sentman_by_geometry.setdefault(str(row["geometry_id"]), []).append(row)
    evaluations = list(bundle["evaluations"])
    positions = {
        (str(row["geometry_id"]), str(row["model_id"]), str(row["canonical_sample_id"])): index
        for index, row in enumerate(evaluations)
    }
    added = 0
    replaced = 0

    def upsert(row: Dict[str, Any]) -> None:
        nonlocal added, replaced
        key = (str(row["geometry_id"]), str(row["model_id"]), str(row["canonical_sample_id"]))
        if key in positions:
            evaluations[positions[key]] = row
            replaced += 1
        else:
            positions[key] = len(evaluations)
            evaluations.append(row)
            added += 1

    for geometry in suite["geometries"]:
        geometry_id = str(geometry["geometry_id"])
        bundle["geometries"][geometry_id] = {
            "design": geometry["design"],
            "mfmc_optimization_suite": {
                key: geometry[key]
                for key in (
                    "geometry_id", "manifest_json", "mesh_path", "mesh_reference",
                    "reference_area_m2", "n_tetrahedra", "n_hexahedra", "hdf5_fingerprint",
                    "selection_order", "selection_basis",
                )
            },
        }
        area = float(geometry["reference_area_m2"])
        result_path = root / geometry_id / TARGET_MODEL / "piclas_results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing collected TPMC optimization result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "collected":
            raise ValueError(f"TPMC optimization result is not collected: {result_path}")
        for sample_id, cd, cost in zip(
            result["sample_ids"], result["values_by_qoi"]["C_D"], result["costs_cpu_hours"]
        ):
            canonical_id = str(sample_id).replace("hf-crn-", "wp1-crn-")
            if canonical_id not in bundle["uncertainty_samples"]:
                raise ValueError(f"Unknown TPMC optimization sample: {sample_id}")
            value = float(cd) * area
            upsert({
                "C_D": float(cd),
                "C_D2": float(cd) ** 2,
                "canonical_sample_id": canonical_id,
                "cost_cpu_hours": float(cost),
                "drag_area_m2": value,
                "drag_area_m4": value * value,
                "fidelity": "hf",
                "geometry_id": geometry_id,
                "model_id": TARGET_MODEL,
                "reference_area_m2": area,
                "sample_id": str(sample_id),
            })
        controls = sentman_by_geometry.get(geometry_id, [])
        if not controls:
            raise ValueError(f"Sentman result CSV has no rows for {geometry_id}")
        for control in controls:
            canonical_id = str(control["sample_id"])
            upsert({
                "C_D": float(control["C_D"]),
                "C_D2": float(control["C_D"]) ** 2,
                "canonical_sample_id": canonical_id,
                "cost_cpu_hours": float(control["cost_cpu_hours"]),
                "drag_area_m2": float(control["drag_area_m2"]),
                "drag_area_m4": float(control["drag_area_m2"]) ** 2,
                "fidelity": "lf",
                "geometry_id": geometry_id,
                "model_id": CONTROL_MODEL,
                "reference_area_m2": area,
                "sample_id": canonical_id,
            })
    evaluations.sort(key=lambda row: (str(row["geometry_id"]), str(row["model_id"]), str(row["canonical_sample_id"])))
    bundle["evaluations"] = evaluations
    bundle["selected_geometry_ids"] = sorted(bundle["geometries"])
    counts: Dict[str, int] = {}
    for row in evaluations:
        key = f"{row['geometry_id']}/{row['model_id']}"
        counts[key] = counts.get(key, 0) + 1
    bundle["counts"] = dict(sorted(counts.items()))
    bundle["study_id"] = "vleo_cylinder_hex_wp6_tpmc_sentman_mfmc_optimization"
    target = Path(output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "output_json": str(target),
        "new_rows": added,
        "replaced_rows": replaced,
        "selected_geometry_ids": bundle["selected_geometry_ids"],
        "counts": bundle["counts"],
    }
