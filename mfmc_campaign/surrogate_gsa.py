from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .campaign import _find_geometry
from .sampling import InputModel, SamplingContext
from .sparse_pce import SparsePCEModel, fit_sparse_pce
from .surrogate_pce_analysis import ModelDataset, load_surrogate_model_datasets, paired_indices


def jansen_indices(
    f_a: np.ndarray,
    f_b: np.ndarray,
    f_ab: np.ndarray,
    variance: float | None = None,
) -> Tuple[float, float]:
    """Return centered Saltelli first-order and Jansen total-effect estimates."""
    a = np.asarray(f_a, dtype=float)
    b = np.asarray(f_b, dtype=float)
    ab = np.asarray(f_ab, dtype=float)
    if not (a.shape == b.shape == ab.shape):
        raise ValueError("Jansen evaluation arrays must have identical shapes")
    var = float(variance) if variance is not None else float(np.var(np.concatenate([a, b]), ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        raise ValueError("Surrogate output variance must be positive")
    first_order = float(np.mean((b - np.mean(b)) * (ab - a))) / var
    total_effect = 0.5 * float(np.mean((a - ab) ** 2)) / var
    return first_order, total_effect


def _first_matching_row(path: str, qoi: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if str(row.get("qoi", "")) == qoi:
                return row
    raise RuntimeError(f"No rows for QoI {qoi} in {path}")


def _load_config(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("config", payload))


def _sample_matrix(
    cfg: Dict[str, Any],
    input_names: Sequence[str],
    reference_row: Dict[str, str],
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, str]]:
    regime_id = str(reference_row["regime_id"])
    active_sources = [source for source in str(reference_row["active_sources"]).split("+") if source]
    input_model = InputModel(cfg.get("variables", []), cfg.get("sampling", {}), cfg.get("regime_label_map", {}))
    samples = input_model.sample(count, SamplingContext(regime_id, active_sources), rng)
    geometry = _find_geometry(cfg, str(reference_row["geometry_id"]))
    geometry_metadata = geometry.get("metadata", {}) if isinstance(geometry.get("metadata"), dict) else {}
    characteristic_length = geometry.get("characteristic_length")
    variable_sources = {str(var["name"]): str(var["source_block"]) for var in cfg.get("variables", [])}
    matrix = np.empty((count, len(input_names)), dtype=float)
    source_by_input: Dict[str, str] = {}
    for column, input_name in enumerate(input_names):
        if input_name == "geometry_characteristic_length":
            value = characteristic_length
            matrix[:, column] = float(value)
            source_by_input[input_name] = "geometry.fixed"
        elif input_name.startswith("geometry__"):
            name = input_name[len("geometry__") :]
            matrix[:, column] = float(geometry_metadata[name])
            source_by_input[input_name] = "geometry.fixed"
        elif input_name.startswith("input__"):
            name = input_name[len("input__") :]
            matrix[:, column] = [float(sample[name]) for sample in samples]
            source_by_input[input_name] = variable_sources.get(name, "unmapped")
        else:
            raise KeyError(f"Unsupported surrogate input column: {input_name}")
    return matrix, source_by_input


def _bootstrap_interval(
    f_a: np.ndarray,
    f_b: np.ndarray,
    f_ab: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    first_values = np.empty(repetitions, dtype=float)
    total_values = np.empty(repetitions, dtype=float)
    count = len(f_a)
    for repetition in range(repetitions):
        indices = rng.integers(0, count, size=count)
        variance = float(np.var(np.concatenate([f_a[indices], f_b[indices]]), ddof=1))
        first_values[repetition], total_values[repetition] = jansen_indices(
            f_a[indices], f_b[indices], f_ab[indices], variance
        )
    return {
        "first_median": float(np.median(first_values)),
        "first_q025": float(np.quantile(first_values, 0.025)),
        "first_q975": float(np.quantile(first_values, 0.975)),
        "total_median": float(np.median(total_values)),
        "total_q025": float(np.quantile(total_values, 0.025)),
        "total_q975": float(np.quantile(total_values, 0.975)),
    }


def estimate_surrogate_sobol(
    predictor: Callable[[np.ndarray], np.ndarray],
    x_a: np.ndarray,
    x_b: np.ndarray,
    groups: Dict[str, Sequence[int]],
    *,
    bootstrap: int = 200,
    seed: int = 20260317,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    f_a = np.asarray(predictor(x_a), dtype=float)
    f_b = np.asarray(predictor(x_b), dtype=float)
    variance = float(np.var(np.concatenate([f_a, f_b]), ddof=1))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("Surrogate output has zero or non-finite variance")
    rng = np.random.default_rng(seed)
    point_rows: List[Dict[str, Any]] = []
    interval_rows: List[Dict[str, Any]] = []
    for group_name, columns in groups.items():
        hybrid = np.array(x_a, copy=True)
        hybrid[:, list(columns)] = x_b[:, list(columns)]
        f_ab = np.asarray(predictor(hybrid), dtype=float)
        first, total = jansen_indices(f_a, f_b, f_ab, variance)
        point_rows.append(
            {
                "name": group_name,
                "first_order": first,
                "total_effect": total,
                "output_variance": variance,
                "mc_samples": len(f_a),
            }
        )
        if bootstrap > 0:
            interval_rows.append(
                {
                    "name": group_name,
                    "bootstrap_kind": "conditional_mc_given_fitted_surrogate",
                    "bootstrap_repetitions": bootstrap,
                    **_bootstrap_interval(f_a, f_b, f_ab, repetitions=bootstrap, rng=rng),
                }
            )
    return point_rows, interval_rows, variance


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_dataset(dataset: ModelDataset, groups_drawn: np.ndarray) -> ModelDataset:
    indices: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for draw, group in enumerate(groups_drawn):
        selected = np.flatnonzero(dataset.repetitions == group)
        if len(selected) == 0:
            raise ValueError(f"No rows found for repetition {group} in {dataset.model_id}")
        indices.append(selected)
        labels.append(np.full(len(selected), draw, dtype=int))
    chosen = np.concatenate(indices)
    result = dataset.subset(chosen)
    result.repetitions = np.concatenate(labels)
    return result


def refit_bootstrap_source_sobol(
    hf: ModelDataset,
    lf: ModelDataset,
    x_a: np.ndarray,
    x_b: np.ndarray,
    source_groups: Dict[str, Sequence[int]],
    *,
    degree: int = 3,
    q_norm: float = 0.75,
    max_interaction: int | None = 2,
    cv_folds: int = 5,
    repetitions: int = 100,
    seed: int = 20260320,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Refit LF and discrepancy PCEs after repetition-level block resampling."""
    if hf.input_names != lf.input_names:
        raise ValueError("HF and LF datasets use different input columns")
    hf_paired, lf_paired = paired_indices(hf, lf)
    if len(hf_paired) != len(hf.y):
        raise ValueError(f"LF model {lf.model_id} is missing {len(hf.y) - len(hf_paired)} HF pairs")
    lf_pair_by_hf = {int(h): int(l) for h, l in zip(hf_paired, lf_paired)}
    unique_groups = np.unique(hf.repetitions)
    if len(unique_groups) < 4:
        raise ValueError("At least four HF repetition blocks are required for refit bootstrap")
    rng = np.random.default_rng(seed)
    sample_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for bootstrap_id in range(int(repetitions)):
        groups_drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        try:
            hf_boot = _bootstrap_dataset(hf, groups_drawn)
            lf_boot = _bootstrap_dataset(lf, groups_drawn)
            lf_model = fit_sparse_pce(
                lf_boot.x,
                lf_boot.y,
                lf_boot.input_names,
                groups=lf_boot.repetitions,
                degree=degree,
                q_norm=q_norm,
                max_interaction=max_interaction,
                cv_folds=min(cv_folds, len(unique_groups)),
            )
            residual_targets: List[np.ndarray] = []
            residual_x: List[np.ndarray] = []
            residual_groups: List[np.ndarray] = []
            for draw, group in enumerate(groups_drawn):
                original_hf = np.flatnonzero(hf.repetitions == group)
                original_lf = np.asarray([lf_pair_by_hf[int(index)] for index in original_hf], dtype=int)
                residual_x.append(hf.x[original_hf])
                residual_targets.append(hf.y[original_hf] - lf.y[original_lf])
                residual_groups.append(np.full(len(original_hf), draw, dtype=int))
            residual_model = fit_sparse_pce(
                np.concatenate(residual_x),
                np.concatenate(residual_targets),
                hf.input_names,
                groups=np.concatenate(residual_groups),
                degree=degree,
                q_norm=q_norm,
                max_interaction=max_interaction,
                cv_folds=min(cv_folds, len(unique_groups)),
            )
            mc_indices = rng.integers(0, len(x_a), size=len(x_a))
            a = x_a[mc_indices]
            b = x_b[mc_indices]
            predictor = lambda values: lf_model.predict(values) + residual_model.predict(values)
            f_a = predictor(a)
            f_b = predictor(b)
            variance = float(np.var(np.concatenate([f_a, f_b]), ddof=1))
            for source, columns in source_groups.items():
                hybrid = np.array(a, copy=True)
                hybrid[:, list(columns)] = b[:, list(columns)]
                first, total = jansen_indices(f_a, f_b, predictor(hybrid), variance)
                sample_rows.append(
                    {
                        "bootstrap_id": bootstrap_id,
                        "name": source,
                        "first_order": first,
                        "total_effect": total,
                        "output_variance": variance,
                        "hf_blocks": len(unique_groups),
                        "hf_rows": len(hf_boot.y),
                        "lf_rows": len(lf_boot.y),
                    }
                )
        except (ValueError, RuntimeError) as exc:
            failures.append({"bootstrap_id": bootstrap_id, "error": str(exc)})

    interval_rows: List[Dict[str, Any]] = []
    for source in source_groups:
        selected = [row for row in sample_rows if row["name"] == source]
        if not selected:
            continue
        first = np.asarray([row["first_order"] for row in selected], dtype=float)
        total = np.asarray([row["total_effect"] for row in selected], dtype=float)
        interval_rows.append(
            {
                "name": source,
                "bootstrap_kind": "grouped_repetition_surrogate_refit",
                "bootstrap_repetitions_requested": int(repetitions),
                "bootstrap_repetitions_successful": len(selected),
                "first_median": float(np.median(first)),
                "first_q025": float(np.quantile(first, 0.025)),
                "first_q975": float(np.quantile(first, 0.975)),
                "total_median": float(np.median(total)),
                "total_q025": float(np.quantile(total, 0.025)),
                "total_q975": float(np.quantile(total, 0.975)),
            }
        )
    diagnostics = {
        "bootstrap_kind": "grouped_repetition_surrogate_refit",
        "requested": int(repetitions),
        "successful": len({int(row["bootstrap_id"]) for row in sample_rows}),
        "failed": len(failures),
        "failures": failures,
        "conditions_on_selected_lf_model": lf.model_id,
    }
    return sample_rows, interval_rows, diagnostics


def run_surrogate_gsa(
    case_dir: str,
    *,
    pce_dir: str | None = None,
    output_dir: str | None = None,
    qoi: str = "C_D",
    mc_samples: int = 20_000,
    bootstrap: int = 200,
    refit_bootstrap: int = 100,
    refit_mc_samples: int = 5_000,
    refit_max_rows_per_model: int = 0,
    seed: int = 20260317,
) -> Dict[str, Any]:
    case_path = Path(case_dir).resolve()
    pce_path = Path(pce_dir).resolve() if pce_dir else case_path / "surrogate_pce"
    target = Path(output_dir).resolve() if output_dir else pce_path
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pce_path / "surrogate_pce_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("qoi") != qoi:
        raise ValueError(f"PCE manifest QoI {manifest.get('qoi')} does not match requested {qoi}")
    best_lf = str(manifest["best_lf_model_id"])
    lf_model = SparsePCEModel.read_json(manifest["model_files"][f"lf_{best_lf}"])
    residual_model = SparsePCEModel.read_json(manifest["model_files"][f"delta_{best_lf}"])
    if lf_model.input_names != residual_model.input_names:
        raise ValueError("LF and residual models use different input schemas")

    dataset_csv = str(manifest["surrogate_dataset_csv"])
    cfg = _load_config(str(case_path / "config_snapshot.json"))
    reference_row = _first_matching_row(dataset_csv, qoi)
    rng = np.random.default_rng(seed)
    x_a, source_by_input = _sample_matrix(cfg, lf_model.input_names, reference_row, mc_samples, rng)
    x_b, _ = _sample_matrix(cfg, lf_model.input_names, reference_row, mc_samples, rng)
    predictor = lambda values: lf_model.predict(values) + residual_model.predict(values)

    variable_groups = {
        input_name: [column]
        for column, input_name in enumerate(lf_model.input_names)
        if source_by_input.get(input_name) not in {"geometry.fixed", "unmapped"}
    }
    source_groups: Dict[str, List[int]] = {}
    for column, input_name in enumerate(lf_model.input_names):
        source = source_by_input.get(input_name, "unmapped")
        if source in {"geometry.fixed", "unmapped"}:
            continue
        source_groups.setdefault(source, []).append(column)
    variable_rows, variable_intervals, variance = estimate_surrogate_sobol(
        predictor, x_a, x_b, variable_groups, bootstrap=bootstrap, seed=seed + 1
    )
    source_rows, source_intervals, _ = estimate_surrogate_sobol(
        predictor, x_a, x_b, source_groups, bootstrap=bootstrap, seed=seed + 2
    )
    for row in variable_rows:
        row["source_block"] = source_by_input[row["name"]]
    for row in variable_intervals:
        row["level"] = "variable"
        row["source_block"] = source_by_input[row["name"]]
    for row in source_intervals:
        row["level"] = "source"

    refit_sample_rows: List[Dict[str, Any]] = []
    refit_interval_rows: List[Dict[str, Any]] = []
    refit_diagnostics: Dict[str, Any] | None = None
    if refit_bootstrap > 0:
        effective_refit_max_rows = (
            int(refit_max_rows_per_model)
            if refit_max_rows_per_model > 0
            else int(manifest.get("max_rows_per_model", 0))
        )
        datasets, _ = load_surrogate_model_datasets(
            dataset_csv,
            qoi=qoi,
            max_rows_per_model=effective_refit_max_rows,
        )
        hf_ids = [model_id for model_id, data in datasets.items() if data.fidelity.lower() in {"hf", "high"}]
        if len(hf_ids) != 1:
            raise ValueError(f"Expected exactly one HF model, found {hf_ids}")
        if best_lf not in datasets:
            raise ValueError(f"Selected LF model {best_lf} is absent from the surrogate dataset")
        refit_count = min(int(refit_mc_samples), len(x_a))
        refit_sample_rows, refit_interval_rows, refit_diagnostics = refit_bootstrap_source_sobol(
            datasets[hf_ids[0]],
            datasets[best_lf],
            x_a[:refit_count],
            x_b[:refit_count],
            source_groups,
            degree=int(manifest["degree"]),
            q_norm=float(manifest["q_norm"]),
            max_interaction=manifest.get("max_interaction"),
            cv_folds=int(manifest["cv_folds"]),
            repetitions=refit_bootstrap,
            seed=seed + 3,
        )

    variable_path = target / "gsa_sobol_variable.csv"
    source_path = target / "gsa_sobol_source.csv"
    interval_path = target / "gsa_bootstrap_intervals.csv"
    refit_samples_path = target / "gsa_refit_bootstrap_samples.csv"
    refit_intervals_path = target / "gsa_refit_bootstrap_intervals.csv"
    refit_diagnostics_path = target / "gsa_refit_bootstrap_diagnostics.json"
    for path, rows in (
        (variable_path, variable_rows),
        (source_path, source_rows),
        (interval_path, variable_intervals + source_intervals),
        (refit_samples_path, refit_sample_rows),
        (refit_intervals_path, refit_interval_rows),
    ):
        _write_csv(path, rows)
    if refit_diagnostics is not None:
        refit_diagnostics_path.write_text(
            json.dumps(refit_diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )

    quality_flags = list(manifest.get("quality_flags", []))
    if refit_bootstrap > 0:
        quality_flags.append("refit_bootstrap_conditions_on_selected_lf_model")
        if refit_diagnostics and refit_diagnostics["successful"] < 0.8 * refit_bootstrap:
            quality_flags.append("refit_bootstrap_success_rate_below_80_percent")
    else:
        quality_flags.append("bootstrap_intervals_condition_on_fitted_surrogate")
    summary = {
        "status": "sobol_complete_with_refit_bootstrap" if refit_bootstrap > 0 else "sobol_complete_conditional_bootstrap",
        "case_dir": str(case_path),
        "pce_dir": str(pce_path),
        "qoi": qoi,
        "best_lf_model_id": best_lf,
        "mc_samples": mc_samples,
        "bootstrap": bootstrap,
        "refit_bootstrap": refit_bootstrap,
        "refit_mc_samples": min(refit_mc_samples, mc_samples),
        "refit_max_rows_per_model": effective_refit_max_rows if refit_bootstrap > 0 else None,
        "seed": seed,
        "output_variance": variance,
        "sobol_estimators": "centered_saltelli_first_order+jansen_total_effect",
        "quality_flags": quality_flags,
        "variable_sobol_csv": str(variable_path),
        "source_sobol_csv": str(source_path),
        "bootstrap_intervals_csv": str(interval_path),
        "refit_bootstrap_samples_csv": str(refit_samples_path) if refit_sample_rows else None,
        "refit_bootstrap_intervals_csv": str(refit_intervals_path) if refit_interval_rows else None,
        "refit_bootstrap_diagnostics_json": str(refit_diagnostics_path) if refit_diagnostics else None,
    }
    summary_path = target / "gsa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
