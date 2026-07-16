"""Cost-aware allocation for nested multiple-control estimators.

The field-aware API uses complete-field Hilbert covariances for the mean and
second-moment blocks.  The older reduced-feature API remains available for
reproducibility, but is not the production default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from math import ceil, floor
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .metrics import evaluate_subspace
from .models import MFPODError
from .operator import compute_mfpod
from .field_statistics import clean_paired_fields, compute_field_pilot_statistics


@dataclass(frozen=True)
class AllocationOptions:
    """Constraints and numerical choices for :func:`optimize_allocation`.

    A zero minimum count makes a control optional.  An active control is always
    nested with the target, so its count must be at least ``n_target``.
    ``min_ratios`` can make a control mandatory (TPMC normally has ratio 1),
    while ``max_ratios={"TPMC": 10}`` implements the production cap.
    """

    budget: float
    target: str = "DSMC"
    minimum_target: int = 2
    minimum_counts: Mapping[str, int] = field(default_factory=dict)
    maximum_counts: Mapping[str, int] = field(default_factory=dict)
    min_ratios: Mapping[str, float] = field(default_factory=dict)
    max_ratios: Mapping[str, float] = field(default_factory=lambda: {"TPMC": 10.0})
    mode: str = "continuous_round"
    bootstrap_repeats: int = 0
    robust_quantile: float = 0.90
    random_seed: int = 2202
    covariance_ridge: float = 1.0e-10
    psd_floor: float = 0.0
    max_enumeration_candidates: int = 250_000
    feature_weights: Sequence[float] | None = None
    mean_weight: float = 0.25
    second_moment_weight: float = 0.75


@dataclass
class AllocationResult:
    counts: dict[str, int]
    total_cost: float
    objective: float
    mode: str
    feasible: bool
    diagnostics: dict
    candidate_table: list[dict]
    bootstrap_summary: dict | None = None
    control_weights: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def build_moment_features(
    coefficients: np.ndarray,
    *,
    mean_weight: float = 1.0,
    second_moment_weight: float = 1.0,
    include_diagonal: bool = True,
) -> np.ndarray:
    """Return per-sample reduced mean and symmetric second-moment features."""

    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise MFPODError("Reduced coefficients must have shape (n>=2, r>=1)")
    if not np.all(np.isfinite(values)):
        raise MFPODError("Reduced coefficients contain NaN or infinite values")
    i, j = np.triu_indices(values.shape[1], k=0 if include_diagonal else 1)
    blocks = []
    if mean_weight > 0.0:
        blocks.append(np.sqrt(float(mean_weight)) * values)
    if second_moment_weight > 0.0 and i.size:
        blocks.append(np.sqrt(float(second_moment_weight)) * values[:, i] * values[:, j])
    if not blocks:
        raise MFPODError("At least one positive feature weight is required")
    return np.column_stack(blocks)


def _clean_responses(
    responses: Mapping[str, np.ndarray], target: str
) -> tuple[list[str], np.ndarray, dict]:
    names = [str(target).upper()] + sorted(
        str(name).upper() for name in responses if str(name).upper() != str(target).upper()
    )
    lookup = {str(name).upper(): np.asarray(value, dtype=float) for name, value in responses.items()}
    if names[0] not in lookup:
        raise MFPODError(f"Pilot responses do not contain target {names[0]}")
    arrays = [lookup[name] for name in names]
    arrays = [a[:, None] if a.ndim == 1 else a for a in arrays]
    if any(a.ndim != 2 for a in arrays):
        raise MFPODError("Every pilot response must be one- or two-dimensional")
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise MFPODError("Pilot responses must have identical paired shapes")
    finite = np.logical_and.reduce([np.all(np.isfinite(a), axis=1) for a in arrays])
    dropped = int(shape[0] - np.sum(finite))
    if np.sum(finite) < 2:
        raise MFPODError("Fewer than two finite paired pilot samples remain")
    stacked = np.stack([a[finite] for a in arrays], axis=1)  # sample, model, feature
    return names, stacked, {"paired_rows": int(np.sum(finite)), "dropped_nonfinite_rows": dropped}


def _psd_covariances(data: np.ndarray, ridge: float, floor_value: float) -> tuple[np.ndarray, dict]:
    _, n_models, n_features = data.shape
    covariances = np.empty((n_features, n_models, n_models), dtype=float)
    clipped = 0
    condition_numbers = []
    for feature in range(n_features):
        cov = np.cov(data[:, :, feature], rowvar=False, ddof=1)
        cov = np.atleast_2d(cov).astype(float)
        cov = np.nan_to_num(0.5 * (cov + cov.T), nan=0.0, posinf=0.0, neginf=0.0)
        eigvals, eigvecs = np.linalg.eigh(cov)
        scale = max(float(np.max(np.abs(eigvals))), 1.0)
        eig_floor = max(float(floor_value), 0.0) * scale
        clipped += int(np.sum(eigvals < eig_floor))
        eigvals = np.maximum(eigvals, eig_floor)
        cov = (eigvecs * eigvals) @ eigvecs.T
        cov += max(float(ridge), 0.0) * scale * np.eye(n_models)
        covariances[feature] = 0.5 * (cov + cov.T)
        condition_numbers.append(float(np.linalg.cond(covariances[feature])))
    return covariances, {
        "psd_eigenvalues_clipped": clipped,
        "maximum_condition_number": float(max(condition_numbers, default=1.0)),
        "regularization": float(ridge),
    }


def _bounds(names: list[str], costs: Mapping[str, float], options: AllocationOptions) -> tuple[np.ndarray, np.ndarray]:
    budget = float(options.budget)
    lower = []
    upper = []
    for index, name in enumerate(names):
        cost = float(costs[name])
        if not np.isfinite(cost) or cost <= 0.0:
            raise MFPODError(f"Cost for {name} must be finite and positive")
        minimum = options.minimum_target if index == 0 else int(options.minimum_counts.get(name, 0))
        maximum = int(options.maximum_counts.get(name, floor(budget / cost)))
        lower.append(max(0, int(minimum)))
        upper.append(maximum)
    return np.asarray(lower, dtype=int), np.asarray(upper, dtype=int)


def _feasible(counts: Sequence[float], names: list[str], costs: Mapping[str, float], options: AllocationOptions, *, integer: bool) -> bool:
    values = np.asarray(counts, dtype=float)
    if integer and not np.all(np.equal(values, np.floor(values))):
        return False
    lower, upper = _bounds(names, costs, options)
    if np.any(values < lower - 1.0e-10) or np.any(values > upper + 1.0e-10):
        return False
    n_h = values[0]
    if n_h < 1.0 or float(np.dot(values, [costs[n] for n in names])) > options.budget + 1.0e-9:
        return False
    for index, name in enumerate(names[1:], start=1):
        count = values[index]
        min_ratio = float(options.min_ratios.get(name, 0.0))
        max_ratio = float(options.max_ratios.get(name, np.inf))
        if count > 1.0e-12 and count < n_h - 1.0e-10:
            return False
        if count + 1.0e-10 < min_ratio * n_h or count > max_ratio * n_h + 1.0e-10:
            return False
    return True


def _objective_from_covariance(
    counts: Sequence[float], covariance: np.ndarray, feature_weights: np.ndarray
) -> tuple[float, np.ndarray]:
    values = np.asarray(counts, dtype=float)
    n_h = values[0]
    active = np.flatnonzero(values[1:] > 0.0) + 1
    total = 0.0
    beta_norms = []
    eps = np.finfo(float).eps
    for feature, cov in enumerate(covariance):
        base = max(float(cov[0, 0]), eps)
        if active.size == 0:
            variance = base / n_h
            beta = np.empty(0)
        else:
            n_l = values[active]
            c = cov[0, active] * (1.0 / n_l - 1.0 / n_h)
            mins = np.minimum.outer(n_l, n_l)
            q = cov[np.ix_(active, active)] * (1.0 / n_h - 1.0 / mins)
            q = 0.5 * (q + q.T)
            beta = -np.linalg.pinv(q, rcond=1.0e-12) @ c if np.any(np.abs(q) > eps) else np.zeros(active.size)
            variance = base / n_h + 2.0 * float(beta @ c) + float(beta @ q @ beta)
        total += float(feature_weights[feature]) * max(float(variance), 0.0) / base
        beta_norms.append(float(np.linalg.norm(beta)))
    return float(total / np.sum(feature_weights)), np.asarray(beta_norms)


def _block_control_weights(
    counts: Sequence[float], covariance: np.ndarray, names: Sequence[str]
) -> dict[str, float]:
    """Return optimal nested control weights for one Hilbert block."""

    values = np.asarray(counts, dtype=float)
    n_h = values[0]
    active = np.flatnonzero(values[1:] > 0.0) + 1
    result = {str(name): 0.0 for name in names[1:]}
    if active.size == 0:
        return result
    n_l = values[active]
    c = covariance[0, active] * (1.0 / n_l - 1.0 / n_h)
    q = covariance[np.ix_(active, active)] * (
        1.0 / n_h - 1.0 / np.minimum.outer(n_l, n_l)
    )
    q = 0.5 * (q + q.T)
    beta = -np.linalg.pinv(q, rcond=1.0e-12) @ c if np.any(np.abs(q) > np.finfo(float).eps) else np.zeros(active.size)
    for index, value in zip(active, beta):
        result[str(names[index])] = float(value)
    return result


def _candidate_row(counts, names, costs, objective, strategy, robust=None):
    row = {f"n_{name}": int(value) for name, value in zip(names, counts)}
    row.update({
        "total_cost": float(sum(int(value) * float(costs[name]) for name, value in zip(names, counts))),
        "objective": float(objective),
        "strategy": strategy,
    })
    if robust is not None:
        row["robust_objective"] = float(robust)
    return row


def _enumerate_counts(names, costs, options):
    lower, upper = _bounds(names, costs, options)
    estimated = int(np.prod(upper - lower + 1, dtype=object))
    if estimated > options.max_enumeration_candidates:
        raise MFPODError(
            f"Direct enumeration bound ({estimated:,}) exceeds max_enumeration_candidates="
            f"{options.max_enumeration_candidates:,}"
        )
    for values in product(*(range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper))):
        if _feasible(values, names, costs, options, integer=True):
            yield np.asarray(values, dtype=int)


def _minimum_feasible(names, costs, options):
    lower, _ = _bounds(names, costs, options)
    counts = lower.copy()
    n_h = counts[0]
    for index, name in enumerate(names[1:], start=1):
        if counts[index] > 0 or options.min_ratios.get(name, 0.0) > 0.0:
            counts[index] = max(counts[index], int(ceil(max(1.0, options.min_ratios.get(name, 0.0)) * n_h)))
    return counts


def _greedy_counts(names, costs, options, covariance, weights):
    counts = _minimum_feasible(names, costs, options)
    if not _feasible(counts, names, costs, options, integer=True):
        # DSMC-only is the deterministic fallback when optional controls are unaffordable.
        counts = np.zeros(len(names), dtype=int)
        counts[0] = options.minimum_target
    if not _feasible(counts, names, costs, options, integer=True):
        raise MFPODError("Budget cannot satisfy the minimum allocation constraints")
    rows = []
    current, _ = _objective_from_covariance(counts, covariance, weights)
    rows.append(_candidate_row(counts, names, costs, current, "greedy"))
    for _ in range(100_000):
        choices = []
        # Activating an optional control requires a paired block, not one sample.
        for index, name in enumerate(names):
            trial = counts.copy()
            trial[index] += 1 if index == 0 or trial[index] > 0 else counts[0]
            if index == 0:
                for j, control in enumerate(names[1:], start=1):
                    if trial[j] > 0 and trial[j] < trial[0]:
                        trial[j] = trial[0]
            if not _feasible(trial, names, costs, options, integer=True):
                continue
            value, _ = _objective_from_covariance(trial, covariance, weights)
            incremental_cost = sum((trial[j] - counts[j]) * costs[n] for j, n in enumerate(names))
            gain = (current - value) / incremental_cost
            choices.append((gain, -value, -index, trial, value))
        if not choices:
            break
        best = max(choices, key=lambda item: (item[0], item[1], item[2]))
        if best[0] <= 1.0e-15:
            break
        counts, current = best[3], best[4]
        rows.append(_candidate_row(counts, names, costs, current, "greedy"))
    return counts, rows


def _continuous_round_counts(names, costs, options, covariance, weights):
    lower, upper = _bounds(names, costs, options)
    start = _minimum_feasible(names, costs, options).astype(float)
    if not _feasible(start, names, costs, options, integer=False):
        start = lower.astype(float)
    constraints = [{"type": "ineq", "fun": lambda x: options.budget - np.dot(x, [costs[n] for n in names])}]
    for index, name in enumerate(names[1:], start=1):
        min_ratio = float(options.min_ratios.get(name, 0.0))
        max_ratio = float(options.max_ratios.get(name, np.inf))
        if min_ratio > 0.0:
            constraints.append({"type": "ineq", "fun": lambda x, i=index, r=min_ratio: x[i] - r * x[0]})
        if np.isfinite(max_ratio):
            constraints.append({"type": "ineq", "fun": lambda x, i=index, r=max_ratio: r * x[0] - x[i]})
        # Continuous relaxation treats every configured control as active.
        constraints.append({"type": "ineq", "fun": lambda x, i=index: x[i] - x[0]})

    def objective(x):
        return _objective_from_covariance(x, covariance, weights)[0]

    result = minimize(objective, start, method="SLSQP", bounds=list(zip(lower, upper)), constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1.0e-12})
    center = np.clip(result.x if result.success else start, lower, upper)
    candidates = []
    neighborhoods = []
    for value, lo, hi in zip(center, lower, upper):
        base = {int(floor(value)), int(ceil(value)), int(round(value))}
        neighborhoods.append(sorted(v for v in base if lo <= v <= hi))
    for candidate in product(*neighborhoods):
        if _feasible(candidate, names, costs, options, integer=True):
            candidates.append(np.asarray(candidate, dtype=int))
    if not candidates:
        candidate = np.floor(center).astype(int)
        while not _feasible(candidate, names, costs, options, integer=True) and np.any(candidate > lower):
            index = int(np.argmax(candidate - lower)); candidate[index] -= 1
        if _feasible(candidate, names, costs, options, integer=True):
            candidates.append(candidate)
    if not candidates:
        raise MFPODError("Continuous relaxation could not be rounded to a feasible integer allocation")
    rows = []
    for candidate in candidates:
        value, _ = _objective_from_covariance(candidate, covariance, weights)
        rows.append(_candidate_row(candidate, names, costs, value, "continuous_round"))
    selected = min(candidates, key=lambda x: (_objective_from_covariance(x, covariance, weights)[0],
                                               sum(x[i] * costs[n] for i, n in enumerate(names)), tuple(-x)))
    return selected, rows, {"success": bool(result.success), "message": str(result.message),
                            "continuous_counts": {n: float(v) for n, v in zip(names, center)},
                            "continuous_objective": float(objective(center))}


def optimize_allocation(
    pilot_responses: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
    options: AllocationOptions,
) -> AllocationResult:
    """Select a reproducible feasible allocation from paired pilot responses.

    Modes are ``enumeration``, ``continuous_round``, ``greedy``, and
    ``bootstrap_robust``.  The robust mode uses enumeration when it is small
    enough and continuous rounding plus greedy candidates otherwise.
    """

    target = options.target.upper()
    names, data, cleaning = _clean_responses(pilot_responses, target)
    normalized_costs = {str(k).upper(): float(v) for k, v in costs.items()}
    missing = [name for name in names if name not in normalized_costs]
    if missing:
        raise MFPODError(f"Missing costs for {missing}")
    if options.budget <= 0.0 or not np.isfinite(options.budget):
        raise MFPODError("Budget must be finite and positive")
    weights = np.ones(data.shape[2], dtype=float) if options.feature_weights is None else np.asarray(options.feature_weights, dtype=float)
    if weights.shape != (data.shape[2],) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise MFPODError("feature_weights must be finite, nonnegative, and match the response features")
    covariance, covariance_diag = _psd_covariances(data, options.covariance_ridge, options.psd_floor)
    mode = options.mode.lower()
    if mode not in {"enumeration", "continuous_round", "greedy", "bootstrap_robust"}:
        raise MFPODError(f"Unknown allocation mode {options.mode!r}")
    rows = []
    continuous_diag = None
    if mode == "enumeration":
        candidates = list(_enumerate_counts(names, normalized_costs, options))
        for counts in candidates:
            value, _ = _objective_from_covariance(counts, covariance, weights)
            rows.append(_candidate_row(counts, names, normalized_costs, value, "enumeration"))
    elif mode == "greedy":
        selected_counts, rows = _greedy_counts(names, normalized_costs, options, covariance, weights)
        candidates = [selected_counts]
    else:
        selected_counts, continuous_rows, continuous_diag = _continuous_round_counts(
            names, normalized_costs, options, covariance, weights
        )
        greedy_counts, greedy_rows = _greedy_counts(names, normalized_costs, options, covariance, weights)
        rows = continuous_rows + greedy_rows
        unique = {}
        for row in rows:
            key = tuple(int(row[f"n_{name}"]) for name in names)
            unique[key] = np.asarray(key, dtype=int)
        candidates = list(unique.values()) + [selected_counts, greedy_counts]
        if mode == "bootstrap_robust":
            try:
                candidates.extend(_enumerate_counts(names, normalized_costs, options))
            except MFPODError:
                pass
    if not candidates:
        raise MFPODError("No feasible allocation candidates")

    bootstrap_summary = None
    score_covariances = [covariance]
    repeats = int(options.bootstrap_repeats)
    if mode == "bootstrap_robust":
        repeats = max(repeats, 100)
    if repeats > 0:
        rng = np.random.default_rng(options.random_seed)
        score_covariances = []
        for _ in range(repeats):
            indices = rng.integers(0, data.shape[0], size=data.shape[0])
            draw, _ = _psd_covariances(data[indices], options.covariance_ridge, options.psd_floor)
            score_covariances.append(draw)

    evaluated = []
    seen = set()
    for counts in candidates:
        key = tuple(int(v) for v in counts)
        if key in seen or not _feasible(counts, names, normalized_costs, options, integer=True):
            continue
        seen.add(key)
        base_value, beta_norms = _objective_from_covariance(counts, covariance, weights)
        distribution = np.asarray([_objective_from_covariance(counts, cov, weights)[0] for cov in score_covariances])
        robust = float(np.quantile(distribution, options.robust_quantile)) if repeats > 0 else base_value
        row = _candidate_row(counts, names, normalized_costs, base_value, mode, robust if repeats > 0 else None)
        row["mean_beta_norm"] = float(np.mean(beta_norms))
        evaluated.append((robust if mode == "bootstrap_robust" else base_value, base_value, key, row, distribution))
    if not evaluated:
        raise MFPODError("No feasible allocation remained after validation")
    # Deterministic rule: objective, then cost, then more target samples, then lexical counts.
    chosen = min(evaluated, key=lambda item: (
        item[0], item[3]["total_cost"], -item[2][0], tuple(-value for value in item[2][1:])
    ))
    selected_row = chosen[3]
    if repeats > 0:
        bootstrap_summary = {
            "repeats": repeats,
            "quantile": float(options.robust_quantile),
            "mean": float(np.mean(chosen[4])),
            "standard_deviation": float(np.std(chosen[4], ddof=1)) if repeats > 1 else 0.0,
            "lower_05": float(np.quantile(chosen[4], 0.05)),
            "upper_95": float(np.quantile(chosen[4], 0.95)),
            "random_seed": int(options.random_seed),
        }
    diagnostics = {
        **cleaning,
        **covariance_diag,
        "models": names,
        "objective_definition": "weighted normalized trace variance of reduced mean/second-moment features",
        "interpretation": "pilot/model-optimal; POD sensitivity is represented by second-moment error",
        "projection_and_truncation_error_in_objective": False,
        "tie_breaking_rule": "objective, lower cost, more DSMC, then lexicographically more controls",
        "constraints": {
            "budget": float(options.budget),
            "minimum_counts": dict(options.minimum_counts),
            "maximum_counts": dict(options.maximum_counts),
            "min_ratios": dict(options.min_ratios),
            "max_ratios": dict(options.max_ratios),
        },
    }
    if continuous_diag is not None:
        diagnostics["continuous_relaxation"] = continuous_diag
    return AllocationResult(
        counts={name: int(selected_row[f"n_{name}"]) for name in names},
        total_cost=float(selected_row["total_cost"]),
        objective=float(chosen[0]),
        mode=mode,
        feasible=True,
        diagnostics=diagnostics,
        candidate_table=[item[3] for item in sorted(evaluated, key=lambda item: (item[0], item[2]))],
        bootstrap_summary=bootstrap_summary,
    )


def optimize_field_allocation(
    pilot_fields: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
    options: AllocationOptions,
    *,
    reference_field: np.ndarray | None = None,
) -> AllocationResult:
    """Allocate samples from complete-field Hilbert covariance blocks.

    The first block targets integrated mean-field MSE and the second targets
    Frobenius MSE of the centered second moment.  Separate optimal control
    weights are calculated for the two blocks for every count candidate.
    """

    if options.budget <= 0.0 or not np.isfinite(options.budget):
        raise MFPODError("Budget must be finite and positive")
    block_weights = np.asarray(
        [options.mean_weight, options.second_moment_weight], dtype=float
    )
    if (
        not np.all(np.isfinite(block_weights))
        or np.any(block_weights < 0.0)
        or not np.isclose(np.sum(block_weights), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise MFPODError("mean_weight and second_moment_weight must be nonnegative and sum to one")

    target = options.target.upper()
    names, cleaned_fields, cleaning = clean_paired_fields(pilot_fields, target)
    statistics = compute_field_pilot_statistics(
        cleaned_fields,
        target=target,
        reference_field=reference_field,
        covariance_ridge=options.covariance_ridge,
        psd_floor=options.psd_floor,
    )
    covariance = statistics.covariance_blocks
    normalized_costs = {str(k).upper(): float(v) for k, v in costs.items()}
    missing = [name for name in names if name not in normalized_costs]
    if missing:
        raise MFPODError(f"Missing costs for {missing}")

    mode = options.mode.lower()
    if mode not in {"enumeration", "continuous_round", "greedy", "bootstrap_robust"}:
        raise MFPODError(f"Unknown allocation mode {options.mode!r}")
    rows: list[dict] = []
    continuous_diag = None
    if mode == "enumeration":
        candidates = list(_enumerate_counts(list(names), normalized_costs, options))
    elif mode == "greedy":
        selected, rows = _greedy_counts(list(names), normalized_costs, options, covariance, block_weights)
        candidates = [selected]
    else:
        rounded, rounded_rows, continuous_diag = _continuous_round_counts(
            list(names), normalized_costs, options, covariance, block_weights
        )
        greedy, greedy_rows = _greedy_counts(
            list(names), normalized_costs, options, covariance, block_weights
        )
        rows = rounded_rows + greedy_rows
        candidates = [rounded, greedy]
        candidates.extend(
            np.asarray(tuple(int(row[f"n_{name}"]) for name in names), dtype=int)
            for row in rows
        )
        if mode == "bootstrap_robust":
            try:
                candidates.extend(_enumerate_counts(list(names), normalized_costs, options))
            except MFPODError:
                pass
    if not candidates:
        raise MFPODError("No feasible field-allocation candidates")

    repeats = int(options.bootstrap_repeats)
    if mode == "bootstrap_robust":
        repeats = max(repeats, 100)
    score_covariances = [covariance]
    if repeats > 0:
        rng = np.random.default_rng(options.random_seed)
        score_covariances = []
        sample_count = cleaned_fields[names[0]].shape[0]
        for _ in range(repeats):
            indices = rng.integers(0, sample_count, size=sample_count)
            draw = compute_field_pilot_statistics(
                {name: cleaned_fields[name][indices] for name in names},
                target=target,
                reference_field=statistics.reference_field,
                covariance_ridge=options.covariance_ridge,
                psd_floor=options.psd_floor,
            )
            score_covariances.append(draw.covariance_blocks)

    evaluated = []
    seen = set()
    for counts in candidates:
        key = tuple(int(value) for value in counts)
        if key in seen or not _feasible(counts, list(names), normalized_costs, options, integer=True):
            continue
        seen.add(key)
        base_value, _ = _objective_from_covariance(counts, covariance, block_weights)
        distribution = np.asarray(
            [_objective_from_covariance(counts, draw, block_weights)[0] for draw in score_covariances]
        )
        robust_value = (
            float(np.quantile(distribution, options.robust_quantile))
            if repeats > 0
            else base_value
        )
        selection_value = robust_value if mode == "bootstrap_robust" else base_value
        row = _candidate_row(
            counts,
            list(names),
            normalized_costs,
            base_value,
            mode,
            robust_value if repeats > 0 else None,
        )
        row["unused_budget"] = float(options.budget - row["total_cost"])
        row["active_controls"] = ",".join(
            name for name, count in zip(names[1:], counts[1:]) if count > 0
        )
        mean_beta = _block_control_weights(counts, covariance[0], names)
        second_beta = _block_control_weights(counts, covariance[1], names)
        for name in names[1:]:
            row[f"beta_mu_{name}"] = mean_beta[name]
            row[f"beta_M_{name}"] = second_beta[name]
        evaluated.append((selection_value, base_value, key, row, distribution, mean_beta, second_beta))
    if not evaluated:
        raise MFPODError("No feasible field allocation remained after validation")

    chosen = min(
        evaluated,
        key=lambda item: (
            item[0],
            item[3]["total_cost"],
            -item[2][0],
            tuple(-value for value in item[2][1:]),
        ),
    )
    bootstrap_summary = None
    if repeats > 0:
        winner_counts: dict[tuple[int, ...], int] = {}
        for draw_index in range(repeats):
            winner = min(
                evaluated,
                key=lambda item: (
                    item[4][draw_index],
                    item[3]["total_cost"],
                    -item[2][0],
                    tuple(-value for value in item[2][1:]),
                ),
            )[2]
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        bootstrap_summary = {
            "repeats": repeats,
            "quantile": float(options.robust_quantile),
            "selected_objective_mean": float(np.mean(chosen[4])),
            "selected_objective_standard_deviation": float(np.std(chosen[4], ddof=1)) if repeats > 1 else 0.0,
            "selected_objective_lower_05": float(np.quantile(chosen[4], 0.05)),
            "selected_objective_upper_95": float(np.quantile(chosen[4], 0.95)),
            "selection_frequencies": [
                {
                    "counts": {name: int(value) for name, value in zip(names, counts)},
                    "frequency": int(frequency),
                }
                for counts, frequency in sorted(winner_counts.items())
            ],
            "random_seed": int(options.random_seed),
        }

    diagnostics = {
        **cleaning,
        "models": list(names),
        "reference_field": statistics.reference_field.tolist(),
        "pilot_statistics": statistics.diagnostics,
        "mean_covariance_raw": statistics.mean_covariance_raw.tolist(),
        "mean_covariance": statistics.mean_covariance.tolist(),
        "second_moment_covariance_raw": statistics.second_moment_covariance_raw.tolist(),
        "second_moment_covariance": statistics.second_moment_covariance.tolist(),
        "objective_definition": "weighted normalized Hilbert mean variance and Hilbert-Schmidt second-moment variance",
        "objective_weights": {
            "mean": float(options.mean_weight),
            "second_moment": float(options.second_moment_weight),
        },
        "interpretation": "pilot/model-optimal full-field allocation",
        "tpmc_basis_used": False,
        "tie_breaking_rule": "objective, lower cost, more DSMC, then lexicographically more controls",
        "constraints": {
            "budget": float(options.budget),
            "minimum_target": int(options.minimum_target),
            "minimum_counts": dict(options.minimum_counts),
            "maximum_counts": dict(options.maximum_counts),
            "min_ratios": dict(options.min_ratios),
            "max_ratios": dict(options.max_ratios),
        },
    }
    if continuous_diag is not None:
        diagnostics["continuous_relaxation"] = continuous_diag
    selected_row = chosen[3]
    return AllocationResult(
        counts={name: int(selected_row[f"n_{name}"]) for name in names},
        total_cost=float(selected_row["total_cost"]),
        objective=float(chosen[0]),
        mode=mode,
        feasible=True,
        diagnostics=diagnostics,
        candidate_table=[item[3] for item in sorted(evaluated, key=lambda item: (item[0], item[2]))],
        bootstrap_summary=bootstrap_summary,
        control_weights={"mean": chosen[5], "second_moment": chosen[6]},
    )


def compare_field_allocation_strategies(
    pilot_fields: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
    options: AllocationOptions,
) -> list[dict]:
    """Compare field-aware algorithms and fidelity subsets under one budget."""

    normalized = {str(name).upper(): values for name, values in pilot_fields.items()}
    normalized_costs = {str(name).upper(): float(value) for name, value in costs.items()}
    target = options.target.upper()
    rows = []
    for mode in dict.fromkeys((options.mode, "greedy", "continuous_round", "enumeration")):
        try:
            result = optimize_field_allocation(
                normalized,
                normalized_costs,
                AllocationOptions(**{**asdict(options), "mode": mode, "bootstrap_repeats": 0}),
            )
            rows.append({"method": mode, **result.counts, "cost": result.total_cost, "objective": result.objective, "feasible": True})
        except MFPODError as exc:
            rows.append({"method": mode, "feasible": False, "reason": str(exc)})
    subsets = [("DSMC-only", [target])]
    if "TPMC" in normalized:
        subsets.append(("two-fidelity-TPMC", [target, "TPMC"]))
    if "SENTMAN" in normalized:
        subsets.append(("two-fidelity-SENTMAN", [target, "SENTMAN"]))
    for label, subset in subsets:
        subset_options = AllocationOptions(
            **{
                **asdict(options),
                "mode": "enumeration",
                "bootstrap_repeats": 0,
                "minimum_counts": {k: v for k, v in options.minimum_counts.items() if str(k).upper() in subset},
                "maximum_counts": {k: v for k, v in options.maximum_counts.items() if str(k).upper() in subset},
                "min_ratios": {k: v for k, v in options.min_ratios.items() if str(k).upper() in subset},
                "max_ratios": {k: v for k, v in options.max_ratios.items() if str(k).upper() in subset},
            }
        )
        try:
            result = optimize_field_allocation(
                {name: normalized[name] for name in subset},
                {name: normalized_costs[name] for name in subset},
                subset_options,
            )
            rows.append({"method": label, **result.counts, "cost": result.total_cost, "objective": result.objective, "feasible": True})
        except MFPODError as exc:
            rows.append({"method": label, "feasible": False, "reason": str(exc)})
    return rows


def compare_allocation_strategies(pilot_responses, costs, options: AllocationOptions) -> list[dict]:
    """Compare requested optimization, greedy, enumeration (when small), and baselines."""

    rows = []
    modes = [options.mode, "greedy", "continuous_round", "enumeration"]
    for mode in dict.fromkeys(modes):
        try:
            result = optimize_allocation(pilot_responses, costs, AllocationOptions(**{**asdict(options), "mode": mode}))
            rows.append({"method": mode, **result.counts, "cost": result.total_cost, "objective": result.objective, "feasible": True})
        except MFPODError as exc:
            rows.append({"method": mode, "feasible": False, "reason": str(exc)})
    # HF-only and two-fidelity are evaluated through the same model and constraints.
    target = options.target.upper()
    hf_count = max(options.minimum_target, int(floor(options.budget / float(costs[target]))))
    hf_problem = {target: pilot_responses[target]}
    try:
        result = optimize_allocation(hf_problem, {target: costs[target]}, AllocationOptions(
            budget=options.budget, target=target, minimum_target=options.minimum_target,
            maximum_counts={target: hf_count}, mode="enumeration",
            feature_weights=options.feature_weights, covariance_ridge=options.covariance_ridge,
        ))
        rows.append({"method": "HF-only", **result.counts, "cost": result.total_cost, "objective": result.objective, "feasible": True})
    except MFPODError as exc:
        rows.append({"method": "HF-only", "feasible": False, "reason": str(exc)})

    normalized_responses = {str(name).upper(): values for name, values in pilot_responses.items()}
    normalized_costs = {str(name).upper(): float(value) for name, value in costs.items()}
    if "TPMC" in normalized_responses:
        two_options = AllocationOptions(
            **{
                **asdict(options),
                "mode": "continuous_round",
                "minimum_counts": {k: v for k, v in options.minimum_counts.items() if str(k).upper() == "TPMC"},
                "maximum_counts": {k: v for k, v in options.maximum_counts.items() if str(k).upper() in {target, "TPMC"}},
                "min_ratios": {k: v for k, v in options.min_ratios.items() if str(k).upper() == "TPMC"},
                "max_ratios": {k: v for k, v in options.max_ratios.items() if str(k).upper() == "TPMC"},
                "bootstrap_repeats": 0,
            }
        )
        try:
            result = optimize_allocation(
                {target: normalized_responses[target], "TPMC": normalized_responses["TPMC"]},
                {target: normalized_costs[target], "TPMC": normalized_costs["TPMC"]},
                two_options,
            )
            rows.append({"method": "two-fidelity-TPMC", **result.counts, "cost": result.total_cost, "objective": result.objective, "feasible": True})
        except MFPODError as exc:
            rows.append({"method": "two-fidelity-TPMC", "feasible": False, "reason": str(exc)})

    # A transparent fixed-ratio baseline uses each configured minimum ratio and
    # spends the remaining budget by increasing the whole nested block.
    names, data, _ = _clean_responses(normalized_responses, target)
    covariance, _ = _psd_covariances(data, options.covariance_ridge, options.psd_floor)
    weights = np.ones(data.shape[2], dtype=float) if options.feature_weights is None else np.asarray(options.feature_weights, dtype=float)
    fixed = np.zeros(len(names), dtype=int)
    for n_h in range(options.minimum_target, int(options.budget / normalized_costs[target]) + 1):
        trial = np.zeros(len(names), dtype=int); trial[0] = n_h
        for index, name in enumerate(names[1:], start=1):
            mandatory = int(options.minimum_counts.get(name, 0)) > 0 or float(options.min_ratios.get(name, 0.0)) > 0.0
            if mandatory:
                trial[index] = max(int(options.minimum_counts.get(name, 0)), int(ceil(max(1.0, options.min_ratios.get(name, 0.0)) * n_h)))
        if _feasible(trial, names, normalized_costs, options, integer=True):
            fixed = trial
    if fixed[0] > 0:
        value, _ = _objective_from_covariance(fixed, covariance, weights)
        rows.append({"method": "fixed-minimum-ratios", **{name: int(v) for name, v in zip(names, fixed)},
                     "cost": float(sum(fixed[i] * normalized_costs[name] for i, name in enumerate(names))),
                     "objective": value, "feasible": True})
    else:
        rows.append({"method": "fixed-minimum-ratios", "feasible": False, "reason": "No feasible fixed-ratio block"})
    return rows


# Backwards-compatible two-fidelity allocation used by the published demonstrator.
def allocate_counts(budget: float, hf_cost: float, lf_cost: float, *, mode: str = "fixed_budget_fraction", hf_budget_fraction: float = 0.5, m_H: int | None = None, m_L: int | None = None) -> dict:
    if mode == "explicit_counts":
        if m_H is None or m_L is None: raise MFPODError("explicit_counts requires m_H and m_L")
        cost = hf_cost * m_H + lf_cost * m_L
        if not (1 <= m_H < m_L) or cost > budget + 1e-12: raise MFPODError("Explicit MFPOD counts are invalid or exceed budget")
        return {"m_H": int(m_H), "m_L": int(m_L), "cost": float(cost)}
    fraction = 0.5 if mode == "equal_model_budget" else hf_budget_fraction
    if mode not in {"fixed_budget_fraction", "equal_model_budget"}: raise MFPODError(f"Unsupported allocation mode {mode!r}")
    mh = max(1, int(np.floor(fraction * budget / hf_cost)))
    ml = int(np.floor((budget - hf_cost * mh) / lf_cost))
    while mh >= ml and mh > 1:
        mh -= 1; ml = int(np.floor((budget - hf_cost * mh) / lf_cost))
    if mh >= ml: raise MFPODError("Budget cannot support nested m_H < m_L")
    return {"m_H": mh, "m_L": ml, "cost": float(hf_cost * mh + lf_cost * ml), "hf_budget_fraction_realized": float(hf_cost * mh / budget)}


def select_empirical_allocation(hf_pilot: np.ndarray, lf_pilot: np.ndarray, *, budget: float, hf_cost: float, lf_cost: float, candidate_fractions: Iterable[float], alpha: float, target_r: int, validation_fraction: float = 0.4, repeats: int = 20, random_seed: int = 1101) -> dict:
    h, l = np.asarray(hf_pilot), np.asarray(lf_pilot)
    if h.shape != l.shape: raise MFPODError("Allocation pilot requires paired HF/LF snapshots")
    rng = np.random.default_rng(random_seed); rows=[]
    for fraction in candidate_fractions:
        allocation = allocate_counts(budget, hf_cost, lf_cost, hf_budget_fraction=float(fraction))
        for repeat in range(repeats):
            order = rng.permutation(h.shape[0]); n_val = max(1, int(round(validation_fraction * h.shape[0])))
            val, train = order[:n_val], order[n_val:]
            mh = min(allocation["m_H"], max(1, len(train) - 1)); ml = min(allocation["m_L"], len(train))
            if ml <= mh: continue
            hp, lp, le = h[train[:mh]], l[train[:mh]], l[train[mh:ml]]
            result = compute_mfpod(hp, lp, le, alpha, n_modes=min(target_r, 2 * mh + ml - mh))
            metric = evaluate_subspace(result.modes[:, :target_r], h[val])["projection_error"]
            rows.append({"fraction": float(fraction), "repeat": repeat, "m_H": mh, "m_L": ml, "heldout_hf_projection_error": metric})
    summaries=[]
    for fraction in candidate_fractions:
        vals = [r["heldout_hf_projection_error"] for r in rows if r["fraction"] == float(fraction)]
        if vals: summaries.append({"fraction": float(fraction), "median_metric": float(np.median(vals)), "mean_metric": float(np.mean(vals))})
    if not summaries: raise MFPODError("No feasible allocation candidates")
    selected = min(summaries, key=lambda x: (x["median_metric"], -x["fraction"]))
    return {"description": "pilot-selected empirical allocation", "metric": "heldout_hf_projection_error", "candidate_results": rows, "candidate_summaries": summaries, "selected": selected, "tie_breaking_rule": "lowest median metric, then largest HF fraction", "random_seed": random_seed}
