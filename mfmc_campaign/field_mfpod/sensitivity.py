"""Postprocess existing field-MFMC pools for allocation sensitivity studies."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .allocation import (
    AllocationOptions,
    AllocationResult,
    optimize_allocation,
    optimize_field_allocation,
)
from .config import MFPODConfig
from .covariance_operator import (
    covariance_probe_error,
    estimate_full_field_mfmc,
    explicit_full_field_covariance,
    solve_full_field_pod,
)
from .field_validation import (
    leading_eigenvalue_error,
    pod_validation_metrics,
    relative_field_error,
)
from .models import MFPODError, jsonable


_ERROR_METRICS = (
    "mean_field_relative_error",
    "covariance_probe_relative_error",
    "leading_eigenvalue_mean_relative_error",
    "maximum_principal_angle_rad",
    "projector_distance_fro",
    "heldout_projection_error",
)

_AGGREGATE_METRICS = (
    *_ERROR_METRICS,
    "reference_projection_error",
    "minimum_ritz_eigenvalue",
    "negative_eigenvalue_count",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no rows"}])


def _positive_ints(values: Iterable[int], label: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result or any(value <= 0 for value in result):
        raise MFPODError(f"{label} must contain positive integers")
    return result


def _nested_prefix_indices(
    pool_size: int, sizes: Sequence[int], *, random_seed: int
) -> dict[int, np.ndarray]:
    """Return reproducible nested prefixes from one random permutation."""

    counts = _positive_ints(sizes, "prefix sizes")
    if max(counts) > int(pool_size):
        raise MFPODError(f"prefix size exceeds available pool ({pool_size})")
    order = np.random.default_rng(int(random_seed)).permutation(int(pool_size))
    return {count: order[:count].copy() for count in counts}


def _load_existing_pools(results_dir: Path) -> dict[str, Any]:
    prepared_path = results_dir / "snapshots" / "prepared_field_snapshots.npz"
    pilot_path = results_dir / "pilot" / "field_pilot_statistics.npz"
    if not prepared_path.is_file() or not pilot_path.is_file():
        raise MFPODError(
            "Sensitivity analysis requires snapshots/prepared_field_snapshots.npz "
            "and pilot/field_pilot_statistics.npz"
        )
    with np.load(prepared_path, allow_pickle=False) as data:
        models = tuple(
            name.removeprefix("production_")
            for name in data.files
            if name.startswith("production_")
            and not name.startswith("production_ids_")
            and not name.startswith("production_CD_")
            and name != "production_paired_sample_ids"
        )
        models = ("DSMC", *sorted(name for name in models if name != "DSMC"))
        production = {
            name: np.asarray(data[f"production_{name}"], dtype=float) for name in models
        }
        production_ids = {
            name: np.asarray(data[f"production_ids_{name}"]).astype(str) for name in models
        }
        pilot = {name: np.asarray(data[f"pilot_{name}"], dtype=float) for name in models}
        pilot_drag = {
            name: np.asarray(data[f"pilot_CD_{name}"], dtype=float).reshape(-1)
            for name in models
            if f"pilot_CD_{name}" in data.files
        }
        reference = np.asarray(data["reference_DSMC"], dtype=float)
    with np.load(pilot_path, allow_pickle=False) as data:
        reference_field = np.asarray(data["reference_field"], dtype=float)

    dimension = production["DSMC"].shape[1]
    if any(values.ndim != 2 or values.shape[1] != dimension for values in production.values()):
        raise MFPODError("Production pools do not share one full-field state dimension")
    if any(len(production_ids[name]) != len(production[name]) for name in models):
        raise MFPODError("Production sample ids do not match production field rows")
    if reference.ndim != 2 or reference.shape[1] != dimension:
        raise MFPODError("Reference DSMC fields do not match the production state dimension")
    return {
        "models": models,
        "production": production,
        "production_ids": production_ids,
        "pilot": pilot,
        "pilot_drag": pilot_drag,
        "reference": reference,
        "reference_field": reference_field,
        "prepared_path": prepared_path,
        "pilot_path": pilot_path,
    }


def _permuted_nested_fields(
    production: Mapping[str, np.ndarray],
    sample_ids: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    *,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Permute pools while retaining the target/control paired prefix."""

    requested = {
        str(name).upper(): int(value) for name, value in counts.items() if int(value) > 0
    }
    n_target = requested.get("DSMC", 0)
    if n_target <= 0:
        raise MFPODError("A sensitivity allocation must contain DSMC")
    rng = np.random.default_rng(int(random_seed))
    target_order = rng.permutation(len(production["DSMC"]))
    target_ids = np.asarray(sample_ids["DSMC"]).astype(str)[target_order]
    result = {"DSMC": np.asarray(production["DSMC"])[target_order]}
    paired_ids = target_ids[:n_target].tolist()

    for name in sorted(requested):
        if name == "DSMC":
            continue
        ids = np.asarray(sample_ids[name]).astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(ids)}
        missing = [sample_id for sample_id in paired_ids if sample_id not in lookup]
        if missing:
            raise MFPODError(
                f"{name} production pool is missing paired DSMC ids {missing[:5]}"
            )
        paired_rows = [lookup[sample_id] for sample_id in paired_ids]
        paired_set = set(paired_ids)
        remaining = np.asarray(
            [index for index, sample_id in enumerate(ids) if sample_id not in paired_set],
            dtype=int,
        )
        remaining = remaining[rng.permutation(remaining.size)]
        order = np.concatenate((np.asarray(paired_rows, dtype=int), remaining))
        if requested[name] > order.size:
            raise MFPODError(
                f"Requested {requested[name]} {name} fields but only {order.size} exist"
            )
        result[name] = np.asarray(production[name])[order]
    return result


def _allocation_options(
    cfg: MFPODConfig,
    *,
    minimum_target: int,
    maximum_counts: Mapping[str, int],
) -> AllocationOptions:
    settings = cfg.raw.get("field_allocation", {}) or {}
    constraints = cfg.raw.get("allocation_constraints", {}) or {}
    costs = {str(name).upper(): float(value) for name, value in cfg.raw.get("costs", {}).items()}
    budget = constraints.get("budget")
    if budget is None and constraints.get("budget_hf_equivalent") is not None:
        budget = float(constraints["budget_hf_equivalent"]) * costs["DSMC"]
    if budget is None:
        raise MFPODError("Sensitivity analysis requires an allocation budget")
    return AllocationOptions(
        budget=float(budget),
        target="DSMC",
        minimum_target=int(minimum_target),
        minimum_counts={
            str(key).upper(): int(value)
            for key, value in (constraints.get("minimum_counts", {}) or {}).items()
        },
        maximum_counts={str(key).upper(): int(value) for key, value in maximum_counts.items()},
        min_ratios={
            str(key).upper(): float(value)
            for key, value in (constraints.get("min_ratios", {}) or {}).items()
        },
        max_ratios={
            str(key).upper(): float(value)
            for key, value in (constraints.get("max_ratios", {}) or {}).items()
        },
        mode=str(settings.get("mode", "bootstrap_robust")),
        bootstrap_repeats=int(settings.get("bootstrap_repeats", 200)),
        robust_quantile=float(settings.get("robust_quantile", 0.90)),
        random_seed=int(settings.get("random_seed", cfg.random_seed)),
        covariance_ridge=float(settings.get("covariance_ridge", 1.0e-10)),
        psd_floor=float(settings.get("psd_floor", 0.0)),
        max_enumeration_candidates=int(settings.get("max_enumeration_candidates", 250_000)),
        mean_weight=float(settings.get("mean_weight", 0.25)),
        second_moment_weight=float(settings.get("second_moment_weight", 0.75)),
    )


def _fixed_ratio_allocation(
    cfg: MFPODConfig,
    pilot: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
    maximum_counts: Mapping[str, int],
) -> AllocationResult | None:
    constraints = cfg.raw.get("allocation_constraints", {}) or {}
    budget = float(
        constraints.get(
            "budget",
            float(constraints.get("budget_hf_equivalent", 0.0)) * costs["DSMC"],
        )
    )
    ratios = (cfg.raw.get("validation", {}) or {}).get("fixed_ratios", {}) or {}
    minimum = int(constraints.get("minimum_target", 2))
    fixed = None
    for n_target in range(minimum, maximum_counts["DSMC"] + 1):
        trial = {"DSMC": n_target}
        for name in pilot:
            if name != "DSMC":
                trial[name] = max(
                    n_target, int(np.ceil(float(ratios.get(name, 1.0)) * n_target))
                )
        if (
            sum(trial[name] * costs[name] for name in trial) <= budget + 1.0e-12
            and all(trial[name] <= maximum_counts[name] for name in trial)
        ):
            fixed = trial
    if fixed is None:
        return None
    options = _allocation_options(
        cfg, minimum_target=fixed["DSMC"], maximum_counts=fixed
    )
    options = AllocationOptions(
        **{
            **asdict(options),
            "budget": sum(fixed[name] * costs[name] for name in fixed),
            "minimum_counts": {
                name: count for name, count in fixed.items() if name != "DSMC"
            },
            "min_ratios": {},
            "max_ratios": {},
            "mode": "enumeration",
            "bootstrap_repeats": 0,
        }
    )
    return optimize_field_allocation(
        {name: pilot[name] for name in fixed},
        {name: costs[name] for name in fixed},
        options,
    )


def _supplementary_allocations(
    cfg: MFPODConfig,
    pilot: Mapping[str, np.ndarray],
    pilot_drag: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
    maximum_counts: Mapping[str, int],
) -> dict[str, AllocationResult]:
    """Create predeclared two-fidelity/scalar-drag baselines when possible."""

    result: dict[str, AllocationResult] = {}
    constraints = cfg.raw.get("allocation_constraints", {}) or {}
    base_minimum = int(constraints.get("minimum_target", 2))
    if "TPMC" in pilot:
        names = ("DSMC", "TPMC")
        options = _allocation_options(
            cfg,
            minimum_target=base_minimum,
            maximum_counts={name: maximum_counts[name] for name in names},
        )
        options = AllocationOptions(
            **{
                **asdict(options),
                "minimum_counts": {
                    key: value
                    for key, value in options.minimum_counts.items()
                    if key == "TPMC"
                },
                "min_ratios": {
                    key: value
                    for key, value in options.min_ratios.items()
                    if key == "TPMC"
                },
                "max_ratios": {
                    key: value
                    for key, value in options.max_ratios.items()
                    if key == "TPMC"
                },
                "mode": "continuous_round",
                "bootstrap_repeats": 0,
            }
        )
        result["two-fidelity-TPMC"] = optimize_field_allocation(
            {name: pilot[name] for name in names},
            {name: costs[name] for name in names},
            options,
        )
    compare_drag = bool(
        (cfg.raw.get("validation", {}) or {}).get(
            "compare_scalar_drag_allocation", True
        )
    )
    if compare_drag and set(pilot_drag) == set(pilot):
        options = _allocation_options(
            cfg,
            minimum_target=base_minimum,
            maximum_counts=maximum_counts,
        )
        scalar_options = AllocationOptions(
            **{**asdict(options), "mode": "enumeration", "bootstrap_repeats": 0}
        )
        try:
            selected = optimize_allocation(pilot_drag, costs, scalar_options)
        except MFPODError as exc:
            if "exceeds max_enumeration_candidates" not in str(exc):
                raise
            selected = optimize_allocation(
                pilot_drag,
                costs,
                AllocationOptions(
                    **{
                        **asdict(scalar_options),
                        "mode": "continuous_round",
                    }
                ),
            )
        counts = {name: int(value) for name, value in selected.counts.items()}
        locked = AllocationOptions(
            **{
                **asdict(options),
                "budget": selected.total_cost,
                "minimum_target": counts["DSMC"],
                "minimum_counts": {
                    name: count
                    for name, count in counts.items()
                    if name != "DSMC" and count > 0
                },
                "maximum_counts": counts,
                "min_ratios": {},
                "max_ratios": {},
                "mode": "enumeration",
                "bootstrap_repeats": 0,
            }
        )
        result["scalar-drag-allocation"] = optimize_field_allocation(
            {name: pilot[name] for name in counts},
            {name: costs[name] for name in counts},
            locked,
        )
    return result


def _evaluate(
    fields: Mapping[str, np.ndarray],
    allocation: AllocationResult,
    *,
    reference_field: np.ndarray,
    reference_fields: np.ndarray,
    reference_statistics: Any,
    reference_pod: Any,
    full_reference_fields: np.ndarray,
    pod_settings: Mapping[str, Any],
    validation_settings: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    weights = allocation.control_weights or {}
    active_counts = {
        name: count for name, count in allocation.counts.items() if int(count) > 0
    }
    active = {name: fields[name] for name in active_counts}
    estimate = estimate_full_field_mfmc(
        active,
        active_counts,
        reference_field=reference_field,
        mean_weights=weights.get("mean", {}),
        second_moment_weights=weights.get("second_moment", {}),
    )
    pod = solve_full_field_pod(
        estimate,
        n_modes=int(pod_settings.get("number_of_modes", 5)),
        tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
        max_iterations=int(pod_settings.get("max_iterations", 5000)),
        negative_eigenvalue_tolerance=float(
            pod_settings.get("negative_eigenvalue_tolerance", 1.0e-10)
        ),
        clip_small_negative_eigenvalues=bool(
            pod_settings.get("clip_small_negative_eigenvalues", False)
        ),
        random_seed=int(random_seed),
    )
    subspace = pod_validation_metrics(
        pod.modes,
        reference_pod.modes,
        full_reference_fields,
        estimated_mean=estimate.mean_field,
        reference_mean=reference_statistics.mean_field,
    )
    eigenvalue = leading_eigenvalue_error(pod.eigenvalues, reference_pod.eigenvalues)
    return {
        "mean_field_relative_error": relative_field_error(
            estimate.mean_field, reference_statistics.mean_field
        ),
        "covariance_probe_relative_error": covariance_probe_error(
            estimate.covariance,
            reference_statistics.covariance,
            probe_count=int(validation_settings.get("covariance_probe_count", 100)),
            random_seed=int(validation_settings.get("covariance_probe_seed", 4401)),
        ),
        "leading_eigenvalue_mean_relative_error": eigenvalue["mean_relative_error"],
        "maximum_principal_angle_rad": subspace["maximum_principal_angle_rad"],
        "projector_distance_fro": subspace["projector_distance_fro"],
        "heldout_projection_error": subspace["projection_error"],
        "reference_projection_error": subspace["reference_projection_error"],
        "minimum_ritz_eigenvalue": pod.diagnostics["minimum_computed_ritz_eigenvalue"],
        "negative_eigenvalue_count": pod.diagnostics["negative_eigenvalue_count"],
        "reference_sample_count": int(reference_fields.shape[0]),
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["minimum_target"]),
            int(row["reference_sample_count"]),
            str(row["method"]),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for (minimum_target, reference_count, method), values in sorted(groups.items()):
        first = values[0]
        summary: dict[str, Any] = {
            "minimum_target": minimum_target,
            "reference_sample_count": reference_count,
            "method": method,
            "repetitions": len(values),
            "n_DSMC": first.get("n_DSMC", 0),
            "n_TPMC": first.get("n_TPMC", 0),
            "n_SENTMAN": first.get("n_SENTMAN", 0),
            "configured_cost": first["configured_cost"],
            "allocation_objective": first.get("allocation_objective"),
        }
        baseline_by_name = {
            name: {
                int(row["repetition"]): row
                for row in groups.get((minimum_target, reference_count, name), [])
            }
            for name in ("DSMC-only", "fixed-ratios")
        }
        for metric in _AGGREGATE_METRICS:
            data = np.asarray([float(row[metric]) for row in values], dtype=float)
            summary[f"{metric}_median"] = float(np.median(data))
            summary[f"{metric}_q25"] = float(np.quantile(data, 0.25))
            summary[f"{metric}_q75"] = float(np.quantile(data, 0.75))
            summary[f"{metric}_minimum"] = float(np.min(data))
            summary[f"{metric}_maximum"] = float(np.max(data))
            for baseline_name, baseline_rows in baseline_by_name.items():
                label = baseline_name.lower().replace("-", "_")
                paired = [
                    float(row[metric])
                    < float(baseline_rows[int(row["repetition"])][metric])
                    for row in values
                    if int(row["repetition"]) in baseline_rows
                    and method != baseline_name
                ]
                summary[f"{metric}_win_rate_vs_{label}"] = (
                    float(np.mean(paired)) if paired else None
                )
        summaries.append(summary)
    return summaries


def _reference_convergence(
    reference_fields: np.ndarray,
    reference_field: np.ndarray,
    reference_sizes: Sequence[int],
    *,
    repetitions: int,
    random_seed: int,
    pod_settings: Mapping[str, Any],
    validation_settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_statistics = estimate_full_field_mfmc(
        {"DSMC": reference_fields},
        {"DSMC": len(reference_fields)},
        reference_field=reference_field,
    )
    full_pod = solve_full_field_pod(
        full_statistics,
        n_modes=int(pod_settings.get("number_of_modes", 5)),
        tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
        max_iterations=int(pod_settings.get("max_iterations", 5000)),
        random_seed=int(random_seed),
    )
    rows = []
    for repetition in range(repetitions):
        prefixes = _nested_prefix_indices(
            len(reference_fields),
            reference_sizes,
            random_seed=int(random_seed) + 100_003 + repetition,
        )
        for count, indices in prefixes.items():
            subset = reference_fields[indices]
            statistics = estimate_full_field_mfmc(
                {"DSMC": subset},
                {"DSMC": count},
                reference_field=reference_field,
            )
            pod = solve_full_field_pod(
                statistics,
                n_modes=int(pod_settings.get("number_of_modes", 5)),
                tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
                max_iterations=int(pod_settings.get("max_iterations", 5000)),
                random_seed=int(random_seed) + repetition,
            )
            subspace = pod_validation_metrics(
                pod.modes,
                full_pod.modes,
                reference_fields,
                estimated_mean=statistics.mean_field,
                reference_mean=full_statistics.mean_field,
            )
            eigenvalue = leading_eigenvalue_error(pod.eigenvalues, full_pod.eigenvalues)
            rows.append(
                {
                    "repetition": repetition,
                    "reference_sample_count": count,
                    "mean_field_relative_error": relative_field_error(
                        statistics.mean_field, full_statistics.mean_field
                    ),
                    "covariance_probe_relative_error": covariance_probe_error(
                        statistics.covariance,
                        full_statistics.covariance,
                        probe_count=int(
                            validation_settings.get("covariance_probe_count", 100)
                        ),
                        random_seed=int(
                            validation_settings.get("covariance_probe_seed", 4401)
                        ),
                    ),
                    "leading_eigenvalue_mean_relative_error": eigenvalue[
                        "mean_relative_error"
                    ],
                    "maximum_principal_angle_rad": subspace[
                        "maximum_principal_angle_rad"
                    ],
                    "projector_distance_fro": subspace["projector_distance_fro"],
                    "heldout_projection_error": subspace["projection_error"],
                    "full_reference_projection_error": subspace[
                        "reference_projection_error"
                    ],
                }
            )
    summaries = []
    for count in reference_sizes:
        values = [row for row in rows if row["reference_sample_count"] == count]
        summary: dict[str, Any] = {
            "reference_sample_count": count,
            "repetitions": len(values),
        }
        for metric in (
            *_ERROR_METRICS,
            "full_reference_projection_error",
        ):
            data = np.asarray([float(row[metric]) for row in values], dtype=float)
            summary[f"{metric}_median"] = float(np.median(data))
            summary[f"{metric}_q25"] = float(np.quantile(data, 0.25))
            summary[f"{metric}_q75"] = float(np.quantile(data, 0.75))
        summaries.append(summary)
    return rows, summaries


def _write_figures(
    output_dir: Path,
    case_name: str,
    allocation_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    reference_summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib.pyplot as plt

    paths = []
    minimum_targets = np.asarray(
        [int(row["minimum_target"]) for row in allocation_rows], dtype=int
    )
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    for name, label in (
        ("n_DSMC", "DSMC"),
        ("n_TPMC", "TPMC"),
        ("n_SENTMAN", "Sentman"),
    ):
        axis.plot(
            minimum_targets,
            [int(row[name]) for row in allocation_rows],
            marker="o",
            label=label,
        )
    axis.set_xlabel(r"minimum target count $m_0$")
    axis.set_ylabel("selected production count")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        path = output_dir / f"m0_allocation_counts{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(str(path))
    plt.close(fig)

    reference_count = max(int(row["reference_sample_count"]) for row in summaries)
    selected = [
        row
        for row in summaries
        if int(row["reference_sample_count"]) == reference_count
        and str(row["method"]).startswith("field-aware-m0-")
    ]
    selected.sort(key=lambda row: int(row["minimum_target"]))
    baseline = next(
        row
        for row in summaries
        if int(row["reference_sample_count"]) == reference_count
        and row["method"] == "DSMC-only"
    )
    fixed = next(
        (
            row
            for row in summaries
            if int(row["reference_sample_count"]) == reference_count
            and row["method"] == "fixed-ratios"
        ),
        None,
    )
    labels = {
        "mean_field_relative_error": "mean-field relative error",
        "covariance_probe_relative_error": "covariance probe error",
        "leading_eigenvalue_mean_relative_error": "mean relative eigenvalue error",
        "maximum_principal_angle_rad": "maximum principal angle [rad]",
        "projector_distance_fro": "projector distance",
        "heldout_projection_error": "DSMC projection error",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=True)
    x = np.asarray([int(row["minimum_target"]) for row in selected])
    for axis, metric in zip(axes.flat, _ERROR_METRICS):
        median = np.asarray([float(row[f"{metric}_median"]) for row in selected])
        lower = np.asarray([float(row[f"{metric}_q25"]) for row in selected])
        upper = np.asarray([float(row[f"{metric}_q75"]) for row in selected])
        axis.plot(x, median, marker="o", label="field-aware median")
        axis.fill_between(x, lower, upper, alpha=0.2, label="field-aware IQR")
        axis.axhline(
            float(baseline[f"{metric}_median"]),
            color="black",
            linestyle="--",
            label="DSMC-only",
        )
        if fixed is not None:
            axis.axhline(
                float(fixed[f"{metric}_median"]),
                color="tab:orange",
                linestyle=":",
                label="fixed ratios",
            )
        axis.set_title(labels[metric])
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel(r"minimum target count $m_0$")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
    )
    fig.suptitle(
        f"{case_name} sensitivity against the {reference_count}-DSMC reference"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    for suffix in (".png", ".pdf"):
        path = output_dir / f"m0_validation_metrics_reference_{reference_count}{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(str(path))
    plt.close(fig)

    reference_metrics = (
        "mean_field_relative_error",
        "covariance_probe_relative_error",
        "leading_eigenvalue_mean_relative_error",
        "projector_distance_fro",
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True)
    x = np.asarray(
        [int(row["reference_sample_count"]) for row in reference_summaries]
    )
    for axis, metric in zip(axes.flat, reference_metrics):
        median = np.asarray(
            [float(row[f"{metric}_median"]) for row in reference_summaries]
        )
        lower = np.asarray(
            [float(row[f"{metric}_q25"]) for row in reference_summaries]
        )
        upper = np.asarray(
            [float(row[f"{metric}_q75"]) for row in reference_summaries]
        )
        axis.plot(x, median, marker="o")
        axis.fill_between(x, lower, upper, alpha=0.2)
        axis.set_title(labels[metric])
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("DSMC reference count")
    fig.suptitle(
        f"{case_name}: independent DSMC-reference convergence against all 50 fields"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in (".png", ".pdf"):
        path = output_dir / f"reference_convergence{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _operator_diagnostics(
    *,
    fields: Mapping[str, np.ndarray],
    allocation: AllocationResult,
    reference_field: np.ndarray,
    reference_statistics: Any,
    reference_pod: Any,
    pod_settings: Mapping[str, Any],
    random_seed: int,
    explicit_dimension_limit: int = 5000,
) -> dict[str, Any]:
    """Diagnose one signed MFMC covariance operator without solver work."""

    counts = {
        name: int(count)
        for name, count in allocation.counts.items()
        if int(count) > 0
    }
    active = {name: np.asarray(fields[name]) for name in counts}
    weights = allocation.control_weights or {}
    estimate = estimate_full_field_mfmc(
        active,
        counts,
        reference_field=reference_field,
        mean_weights=weights.get("mean", {}),
        second_moment_weights=weights.get("second_moment", {}),
    )
    pod = solve_full_field_pod(
        estimate,
        n_modes=int(pod_settings.get("number_of_modes", 5)),
        tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
        max_iterations=int(pod_settings.get("max_iterations", 5000)),
        negative_eigenvalue_tolerance=float(
            pod_settings.get("negative_eigenvalue_tolerance", 1.0e-10)
        ),
        clip_small_negative_eigenvalues=bool(
            pod_settings.get("clip_small_negative_eigenvalues", False)
        ),
        random_seed=random_seed,
    )
    mode_count = min(5, pod.modes.shape[1], reference_pod.modes.shape[1])
    overlap = np.clip(
        np.abs(
            np.sum(
                pod.modes[:, :mode_count] * reference_pod.modes[:, :mode_count],
                axis=0,
            )
        ),
        0.0,
        1.0,
    )
    singular_values = np.linalg.svd(
        reference_pod.modes[:, :mode_count].T @ pod.modes[:, :mode_count],
        compute_uv=False,
    )
    diagnostics: dict[str, Any] = {
        "counts": counts,
        "configured_cost": allocation.total_cost,
        "allocation_objective": allocation.objective,
        "mean_weights": estimate.mean_weights,
        "second_moment_weights": estimate.second_moment_weights,
        "signed_operator_terms": estimate.diagnostics["operator_terms"],
        "individual_direction_angles_rad": np.arccos(overlap),
        "principal_angles_rad": np.arccos(
            np.clip(np.sort(singular_values)[::-1], 0.0, 1.0)
        ),
        "estimated_eigenvalues": pod.eigenvalues,
        "minimum_ritz_eigenvalue": pod.diagnostics[
            "minimum_computed_ritz_eigenvalue"
        ],
        "negative_eigenvalue_count": pod.diagnostics["negative_eigenvalue_count"],
        "state_dimension": int(estimate.covariance.shape[0]),
        "explicit_matrix_formed": False,
    }
    dimension = estimate.covariance.shape[0]
    if dimension > explicit_dimension_limit:
        diagnostics["matrix_free_reason"] = (
            f"state dimension {dimension} exceeds explicit limit "
            f"{explicit_dimension_limit}"
        )
        diagnostics["covariance_probe_relative_error"] = covariance_probe_error(
            estimate.covariance,
            reference_statistics.covariance,
            probe_count=100,
            random_seed=random_seed,
        )
        return diagnostics

    covariance = explicit_full_field_covariance(estimate)
    reference_covariance = explicit_full_field_covariance(reference_statistics)
    error = covariance - reference_covariance
    reference_fro = max(np.linalg.norm(reference_covariance, ord="fro"), np.finfo(float).eps)
    reference_spectral = max(np.linalg.norm(reference_covariance, ord=2), np.finfo(float).eps)
    basis = reference_pod.modes[:, :mode_count]
    block_error = basis.T @ error @ basis
    coupling = error @ basis - basis @ block_error
    eigenvalues = np.asarray(reference_pod.eigenvalues, dtype=float)
    gap = (
        float(eigenvalues[4] - eigenvalues[5])
        if eigenvalues.size >= 6
        else None
    )

    centered = {
        name: active[name][: counts[name]] - np.asarray(reference_field)
        for name in counts
    }
    n_h = counts["DSMC"]
    contributions = [
        centered["DSMC"][:n_h].T @ centered["DSMC"][:n_h] / n_h
    ]
    for name, beta in estimate.second_moment_weights.items():
        if beta == 0.0:
            continue
        contributions.append(
            beta * centered[name].T @ centered[name] / counts[name]
        )
        contributions.append(
            -beta * centered[name][:n_h].T @ centered[name][:n_h] / n_h
        )
    contributions.append(-np.outer(estimate.centered_mean, estimate.centered_mean))
    component_norm_sum = float(
        sum(np.linalg.norm(value, ord="fro") for value in contributions)
    )
    diagnostics.update(
        {
            "explicit_matrix_formed": True,
            "relative_frobenius_covariance_error": float(
                np.linalg.norm(error, ord="fro") / reference_fro
            ),
            "relative_spectral_covariance_error": float(
                np.linalg.norm(error, ord=2) / reference_spectral
            ),
            "reference_eigenbasis_top5_block_error_fro": float(
                np.linalg.norm(block_error, ord="fro") / reference_fro
            ),
            "top5_off_diagonal_error_fro": float(
                np.linalg.norm(block_error - np.diag(np.diag(block_error)), ord="fro")
                / reference_fro
            ),
            "top5_to_complement_coupling_fro": float(
                np.linalg.norm(coupling, ord="fro") / reference_fro
            ),
            "lambda5_minus_lambda6": gap,
            "coupling_to_lambda5_lambda6_gap": (
                float(np.linalg.norm(coupling, ord=2) / gap)
                if gap is not None and gap > 0.0
                else None
            ),
            "signed_operator_cancellation_ratio_fro": (
                float(np.linalg.norm(covariance, ord="fro") / component_norm_sum)
                if component_norm_sum > 0.0
                else None
            ),
        }
    )
    return diagnostics


def _pod_diagnostics(
    pools: Mapping[str, Any],
    allocations: Mapping[int, AllocationResult],
    dsmc_only: AllocationResult,
    fixed: AllocationResult | None,
    supplementary: Mapping[str, AllocationResult],
    *,
    pod_settings: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    reference_statistics = estimate_full_field_mfmc(
        {"DSMC": pools["reference"]},
        {"DSMC": len(pools["reference"])},
        reference_field=pools["reference_field"],
    )
    reference_pod = solve_full_field_pod(
        reference_statistics,
        n_modes=min(6, pools["reference"].shape[1]),
        tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
        max_iterations=int(pod_settings.get("max_iterations", 5000)),
        random_seed=random_seed,
    )
    eigenvalues = np.asarray(reference_pod.eigenvalues, dtype=float)
    payload: dict[str, Any] = {
        "reference": {
            "description": "independent 50-DSMC numerical reference",
            "leading_eigenvalues": eigenvalues,
            "eigenvalue_gaps": eigenvalues[:-1] - eigenvalues[1:],
            "lambda5_minus_lambda6": (
                float(eigenvalues[4] - eigenvalues[5])
                if eigenvalues.size >= 6
                else None
            ),
        },
        "methods": {},
        "interpretation_cautions": [
            "Moment accuracy and POD-subspace accuracy need not rank methods identically.",
            "A small lambda5-lambda6 gap amplifies top-5/complement coupling into mode rotation.",
            "Global scalar control weights cannot correct direction-dependent or local TPMC bias.",
        ],
    }
    methods: list[tuple[str, AllocationResult]] = [
        (f"field-aware-m0-{m0}", result)
        for m0, result in sorted(allocations.items())
    ]
    methods.append(("DSMC-only", dsmc_only))
    if fixed is not None:
        methods.append(("fixed-ratios", fixed))
    methods.extend(supplementary.items())
    for index, (name, allocation) in enumerate(methods):
        fields = _permuted_nested_fields(
            pools["production"],
            pools["production_ids"],
            allocation.counts,
            random_seed=random_seed,
        )
        payload["methods"][name] = _operator_diagnostics(
            fields=fields,
            allocation=allocation,
            reference_field=pools["reference_field"],
            reference_statistics=reference_statistics,
            reference_pod=reference_pod,
            pod_settings=pod_settings,
            random_seed=random_seed + index,
        )
    return payload


def _write_case_findings(
    path: Path,
    *,
    case_name: str,
    summaries: Sequence[Mapping[str, Any]],
    reference_count: int,
) -> None:
    field_rows = [
        row
        for row in summaries
        if int(row["reference_sample_count"]) == reference_count
        and str(row["method"]).startswith("field-aware-m0-")
    ]
    metrics = {
        "mean-field": "mean_field_relative_error",
        "covariance-probe": "covariance_probe_relative_error",
        "leading-eigenvalue": "leading_eigenvalue_mean_relative_error",
        "POD projector": "projector_distance_fro",
        "held-out projection": "heldout_projection_error",
    }
    lines = [
        f"# {case_name}: sensitivity findings",
        "",
        "All values below are generated from `m0_summary.csv`. The 50-field DSMC set is a numerical reference, not physical truth or proof of convergence to the infinite DSMC population.",
        "",
        "## Metric-specific optima",
        "",
    ]
    for label, metric in metrics.items():
        best = min(field_rows, key=lambda row: float(row[f"{metric}_median"]))
        lines.append(
            f"- {label}: m0={best['minimum_target']}, median "
            f"{float(best[f'{metric}_median']):.6g} "
            f"(IQR {float(best[f'{metric}_q25']):.6g}--"
            f"{float(best[f'{metric}_q75']):.6g}; range "
            f"{float(best[f'{metric}_minimum']):.6g}--"
            f"{float(best[f'{metric}_maximum']):.6g})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The field-moment, eigenvalue, and five-dimensional POD-subspace metrics are distinct objectives. Better mean or eigenvalue estimates therefore do not imply a better top-five subspace, particularly when the fifth/sixth reference eigengap is small or covariance error couples the retained subspace to its complement.",
            "",
            "The selected allocation is bootstrap-robust under the configured field objective and scenario costs. It is not claimed to be globally optimal. Configured costs are kept separate from measured CPU-hours.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_field_sensitivity(
    cfg: MFPODConfig,
    *,
    results_dir: str | Path | None = None,
    minimum_targets: Sequence[int] = (2, 4, 6, 8, 10, 12, 16, 20),
    reference_sizes: Sequence[int] = (10, 20, 30, 40, 50),
    repetitions: int = 30,
    random_seed: int = 20260727,
) -> dict[str, Any]:
    """Run allocation, production-order, and DSMC-reference sensitivity offline."""

    root = Path(results_dir).resolve() if results_dir is not None else cfg.output_dir
    minimum_targets = _positive_ints(minimum_targets, "minimum_targets")
    reference_sizes = _positive_ints(reference_sizes, "reference_sizes")
    if repetitions <= 0:
        raise MFPODError("repetitions must be positive")
    pools = _load_existing_pools(root)
    models = pools["models"]
    maximum_counts = {
        name: int(pools["production"][name].shape[0]) for name in models
    }
    if max(minimum_targets) > maximum_counts["DSMC"]:
        raise MFPODError(
            f"minimum_target exceeds the available DSMC pool ({maximum_counts['DSMC']})"
        )
    if max(reference_sizes) > len(pools["reference"]):
        raise MFPODError(
            f"reference size exceeds the available DSMC reference pool ({len(pools['reference'])})"
        )
    costs = {
        name: float((cfg.raw.get("costs", {}) or {})[name]) for name in models
    }
    allocations: dict[int, AllocationResult] = {}
    for minimum_target in minimum_targets:
        allocations[minimum_target] = optimize_field_allocation(
            pools["pilot"],
            costs,
            _allocation_options(
                cfg,
                minimum_target=minimum_target,
                maximum_counts=maximum_counts,
            ),
        )
    dsmc_count = min(
        maximum_counts["DSMC"],
        int(
            np.floor(
                _allocation_options(
                    cfg,
                    minimum_target=minimum_targets[0],
                    maximum_counts=maximum_counts,
                ).budget
                / costs["DSMC"]
            )
        ),
    )
    dsmc_only = optimize_field_allocation(
        {"DSMC": pools["pilot"]["DSMC"]},
        {"DSMC": costs["DSMC"]},
        AllocationOptions(
            budget=dsmc_count * costs["DSMC"],
            minimum_target=dsmc_count,
            maximum_counts={"DSMC": dsmc_count},
            minimum_counts={},
            min_ratios={},
            max_ratios={},
            mode="enumeration",
            bootstrap_repeats=0,
        ),
    )
    fixed = _fixed_ratio_allocation(
        cfg, pools["pilot"], costs, maximum_counts
    )
    supplementary = _supplementary_allocations(
        cfg,
        pools["pilot"],
        pools["pilot_drag"],
        costs,
        maximum_counts,
    )
    pod_settings = cfg.raw.get("pod", {}) or {}
    validation_settings = cfg.raw.get("validation", {}) or {}
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        production_seed = int(random_seed) + repetition
        reference_prefixes = _nested_prefix_indices(
            len(pools["reference"]),
            reference_sizes,
            random_seed=int(random_seed) + 50_021 + repetition,
        )
        reference_by_size = {}
        for reference_count, indices in reference_prefixes.items():
            reference_fields = pools["reference"][indices]
            reference_statistics = estimate_full_field_mfmc(
                {"DSMC": reference_fields},
                {"DSMC": reference_count},
                reference_field=pools["reference_field"],
            )
            reference_pod = solve_full_field_pod(
                reference_statistics,
                n_modes=int(pod_settings.get("number_of_modes", 5)),
                tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
                max_iterations=int(pod_settings.get("max_iterations", 5000)),
                random_seed=int(random_seed) + repetition,
            )
            reference_by_size[reference_count] = (
                reference_fields,
                reference_statistics,
                reference_pod,
            )
        for minimum_target, selected in allocations.items():
            methods = {
                f"field-aware-m0-{minimum_target}": selected,
                "DSMC-only": dsmc_only,
            }
            if fixed is not None:
                methods["fixed-ratios"] = fixed
            methods.update(supplementary)
            for method, allocation in methods.items():
                fields = _permuted_nested_fields(
                    pools["production"],
                    pools["production_ids"],
                    allocation.counts,
                    random_seed=production_seed,
                )
                for reference_count in reference_sizes:
                    (
                        reference_fields,
                        reference_statistics,
                        reference_pod,
                    ) = reference_by_size[reference_count]
                    metrics = _evaluate(
                        fields,
                        allocation,
                        reference_field=pools["reference_field"],
                        reference_fields=reference_fields,
                        reference_statistics=reference_statistics,
                        reference_pod=reference_pod,
                        full_reference_fields=pools["reference"],
                        pod_settings=pod_settings,
                        validation_settings=validation_settings,
                        random_seed=int(random_seed) + repetition,
                    )
                    rows.append(
                        {
                            "minimum_target": minimum_target,
                            "repetition": repetition,
                            "production_seed": production_seed,
                            "method": method,
                            "n_DSMC": allocation.counts.get("DSMC", 0),
                            "n_TPMC": allocation.counts.get("TPMC", 0),
                            "n_SENTMAN": allocation.counts.get("SENTMAN", 0),
                            "configured_cost": allocation.total_cost,
                            "allocation_objective": allocation.objective,
                            **{
                                f"mean_weight_{name}": value
                                for name, value in (allocation.control_weights or {})
                                .get("mean", {})
                                .items()
                            },
                            **{
                                f"second_moment_weight_{name}": value
                                for name, value in (allocation.control_weights or {})
                                .get("second_moment", {})
                                .items()
                            },
                            **metrics,
                        }
                    )
    summaries = _summaries(rows)
    reference_rows, reference_summaries = _reference_convergence(
        pools["reference"],
        pools["reference_field"],
        reference_sizes,
        repetitions=repetitions,
        random_seed=random_seed,
        pod_settings=pod_settings,
        validation_settings=validation_settings,
    )
    out = root / "sensitivity"
    allocation_rows = []
    for minimum_target, result in allocations.items():
        frequency = None
        if result.bootstrap_summary:
            selected_counts = result.counts
            frequency = next(
                (
                    row["frequency"]
                    for row in result.bootstrap_summary.get("selection_frequencies", [])
                    if row["counts"] == selected_counts
                ),
                None,
            )
        allocation_rows.append(
            {
                "minimum_target": minimum_target,
                "n_DSMC": result.counts.get("DSMC", 0),
                "n_TPMC": result.counts.get("TPMC", 0),
                "n_SENTMAN": result.counts.get("SENTMAN", 0),
                "configured_cost": result.total_cost,
                "objective": result.objective,
                "bootstrap_selected_frequency": frequency,
                "bootstrap_repeats": (
                    result.bootstrap_summary.get("repeats")
                    if result.bootstrap_summary
                    else 0
                ),
                **{
                    f"mean_weight_{name}": value
                    for name, value in (result.control_weights or {})
                    .get("mean", {})
                    .items()
                },
                **{
                    f"second_moment_weight_{name}": value
                    for name, value in (result.control_weights or {})
                    .get("second_moment", {})
                    .items()
                },
            }
        )
    _write_csv(out / "m0_allocations.csv", allocation_rows)
    _write_csv(out / "m0_repetitions.csv", rows)
    _write_csv(out / "m0_summary.csv", summaries)
    _write_csv(out / "reference_convergence_repetitions.csv", reference_rows)
    _write_csv(out / "reference_convergence_summary.csv", reference_summaries)
    _write_json(
        out / "m0_allocation_details.json",
        {
            str(minimum_target): result.as_dict()
            for minimum_target, result in allocations.items()
        },
    )
    pod_diagnostics = _pod_diagnostics(
        pools,
        allocations,
        dsmc_only,
        fixed,
        supplementary,
        pod_settings=pod_settings,
        random_seed=random_seed,
    )
    _write_json(out / "pod_subspace_diagnostics.json", pod_diagnostics)
    _write_case_findings(
        out / "case_findings.md",
        case_name=cfg.case_name,
        summaries=summaries,
        reference_count=max(reference_sizes),
    )
    figures = _write_figures(
        out, cfg.case_name, allocation_rows, summaries, reference_summaries
    )
    metadata = {
        "case": cfg.case_name,
        "results_dir": str(root),
        "prepared_fields": str(pools["prepared_path"]),
        "pilot_statistics": str(pools["pilot_path"]),
        "minimum_targets": list(minimum_targets),
        "reference_sizes": list(reference_sizes),
        "repetitions": int(repetitions),
        "random_seed": int(random_seed),
        "production_permutation": (
            "shared DSMC prefix; each active control begins with the same paired "
            "DSMC ids and independently permutes its remaining pool"
        ),
        "reference_protocol": (
            "nested random subsets of the independent DSMC reference pool; "
            "all 50 reference fields are used for projection evaluation"
        ),
        "costs": costs,
        "maximum_available_counts": maximum_counts,
        "allocation_mode": str(
            (cfg.raw.get("field_allocation", {}) or {}).get(
                "mode", "bootstrap_robust"
            )
        ),
        "comparators": [
            "DSMC-only",
            *(["fixed-ratios"] if fixed else []),
            *supplementary,
        ],
        "files": {
            "allocations": str(out / "m0_allocations.csv"),
            "repetitions": str(out / "m0_repetitions.csv"),
            "summary": str(out / "m0_summary.csv"),
            "reference_repetitions": str(
                out / "reference_convergence_repetitions.csv"
            ),
            "reference_summary": str(out / "reference_convergence_summary.csv"),
            "allocation_details": str(out / "m0_allocation_details.json"),
            "pod_diagnostics": str(out / "pod_subspace_diagnostics.json"),
            "case_findings": str(out / "case_findings.md"),
        },
        "figures": figures,
    }
    _write_json(out / "sensitivity_metadata.json", metadata)
    return {
        "output": str(out),
        "allocations": allocation_rows,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "reference_rows": len(reference_rows),
        "files": metadata["files"],
        "figures": figures,
    }
