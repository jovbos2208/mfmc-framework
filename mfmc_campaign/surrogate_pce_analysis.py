from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .sparse_pce import SparsePCEModel, fit_sparse_pce, grouped_cv_splits, regression_metrics


@dataclass
class ModelDataset:
    model_id: str
    fidelity: str
    input_names: List[str]
    x: np.ndarray
    y: np.ndarray
    repetitions: np.ndarray
    cell_ids: np.ndarray
    sample_fingerprints: np.ndarray

    def subset(self, indices: np.ndarray) -> "ModelDataset":
        return ModelDataset(
            model_id=self.model_id,
            fidelity=self.fidelity,
            input_names=self.input_names,
            x=self.x[indices],
            y=self.y[indices],
            repetitions=self.repetitions[indices],
            cell_ids=self.cell_ids[indices],
            sample_fingerprints=self.sample_fingerprints[indices],
        )


def load_surrogate_model_datasets(
    surrogate_dataset_csv: str,
    *,
    qoi: str,
    max_rows_per_model: int = 0,
) -> Tuple[Dict[str, ModelDataset], Dict[str, Any]]:
    """Load one QoI from a strict surrogate dataset using only numeric inputs."""
    rows_by_model: Dict[str, Dict[str, Any]] = {}
    dropped_nonfinite = 0
    with open(surrogate_dataset_csv, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        header = list(reader.fieldnames or [])
        input_names = [
            name
            for name in header
            if name.startswith("input__")
            or name.startswith("geometry__")
            or name == "geometry_characteristic_length"
        ]
        if not input_names:
            raise ValueError(f"No numeric surrogate input columns found in {surrogate_dataset_csv}")
        for row in reader:
            if str(row.get("qoi", "")) != qoi:
                continue
            try:
                values = np.asarray([float(row[name]) for name in input_names], dtype=float)
                target = float(row.get("value", "nan"))
                repetition = int(float(row.get("repetition", "nan")))
            except (TypeError, ValueError):
                dropped_nonfinite += 1
                continue
            if not np.all(np.isfinite(values)) or not np.isfinite(target):
                dropped_nonfinite += 1
                continue
            model_id = str(row.get("model_id", ""))
            if not model_id:
                continue
            payload = rows_by_model.setdefault(
                model_id,
                {
                    "fidelity": str(row.get("fidelity", "")),
                    "x": [],
                    "y": [],
                    "repetitions": [],
                    "cell_ids": [],
                    "sample_fingerprints": [],
                },
            )
            payload["x"].append(values)
            payload["y"].append(target)
            payload["repetitions"].append(repetition)
            payload["cell_ids"].append(str(row.get("cell_id", "")))
            payload["sample_fingerprints"].append(str(row.get("sample_fingerprint", "")))

    datasets: Dict[str, ModelDataset] = {}
    original_counts: Dict[str, int] = {}
    for model_id, payload in sorted(rows_by_model.items()):
        count = len(payload["y"])
        original_counts[model_id] = count
        dataset = ModelDataset(
            model_id=model_id,
            fidelity=str(payload["fidelity"]),
            input_names=list(input_names),
            x=np.asarray(payload["x"], dtype=float),
            y=np.asarray(payload["y"], dtype=float),
            repetitions=np.asarray(payload["repetitions"], dtype=int),
            cell_ids=np.asarray(payload["cell_ids"], dtype=str),
            sample_fingerprints=np.asarray(payload["sample_fingerprints"], dtype=str),
        )
        datasets[model_id] = dataset
    if not datasets:
        raise RuntimeError(f"No finite rows found for QoI {qoi} in {surrogate_dataset_csv}")
    if max_rows_per_model > 0:
        hf_ids = [
            model_id
            for model_id, dataset in datasets.items()
            if dataset.fidelity.lower() in {"hf", "high"}
        ]
        if len(hf_ids) == 1:
            hf_full = datasets[hf_ids[0]]
            hf_selection = np.unique(
                np.linspace(0, len(hf_full.y) - 1, min(max_rows_per_model, len(hf_full.y)), dtype=int)
            )
            hf_selected = hf_full.subset(hf_selection)
            datasets[hf_ids[0]] = hf_selected
            for model_id, dataset in list(datasets.items()):
                if model_id == hf_ids[0]:
                    continue
                base = set(
                    np.unique(
                        np.linspace(
                            0,
                            len(dataset.y) - 1,
                            min(max_rows_per_model, len(dataset.y)),
                            dtype=int,
                        )
                    ).tolist()
                )
                _hf_pair, lf_pair = paired_indices(hf_selected, dataset)
                base.update(int(index) for index in lf_pair)
                datasets[model_id] = dataset.subset(np.asarray(sorted(base), dtype=int))
        else:
            for model_id, dataset in list(datasets.items()):
                indices = np.unique(
                    np.linspace(
                        0,
                        len(dataset.y) - 1,
                        min(max_rows_per_model, len(dataset.y)),
                        dtype=int,
                    )
                )
                datasets[model_id] = dataset.subset(indices)
    return datasets, {
        "qoi": qoi,
        "input_names": input_names,
        "original_model_counts": original_counts,
        "loaded_model_counts": {model_id: len(dataset.y) for model_id, dataset in datasets.items()},
        "dropped_nonfinite_rows": dropped_nonfinite,
    }


def paired_indices(hf: ModelDataset, lf: ModelDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Pair rows only within the same archived cell and physical sample hash."""
    lf_lookup: Dict[Tuple[str, str], int] = {}
    for index, key in enumerate(zip(lf.cell_ids, lf.sample_fingerprints)):
        if key in lf_lookup:
            raise ValueError(f"Duplicate LF provenance key for model {lf.model_id}: {key}")
        lf_lookup[key] = index
    hf_indices: List[int] = []
    lf_indices: List[int] = []
    for hf_index, key in enumerate(zip(hf.cell_ids, hf.sample_fingerprints)):
        lf_index = lf_lookup.get(key)
        if lf_index is None:
            continue
        if not np.allclose(hf.x[hf_index], lf.x[lf_index], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Paired HF/LF inputs disagree for provenance key: {key}")
        hf_indices.append(hf_index)
        lf_indices.append(lf_index)
    return np.asarray(hf_indices, dtype=int), np.asarray(lf_indices, dtype=int)


def _fit(
    dataset: ModelDataset,
    *,
    degree: int,
    q_norm: float,
    max_interaction: int | None,
    cv_folds: int,
) -> SparsePCEModel:
    return fit_sparse_pce(
        dataset.x,
        dataset.y,
        dataset.input_names,
        groups=dataset.repetitions,
        degree=degree,
        q_norm=q_norm,
        max_interaction=max_interaction,
        cv_folds=cv_folds,
    )


def _metric_row(
    *,
    fold: int | str,
    method: str,
    lf_model_id: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, Any]:
    return {
        "fold": fold,
        "method": method,
        "lf_model_id": lf_model_id,
        **regression_metrics(truth, prediction),
    }


def cross_validate_multifidelity_pce(
    hf: ModelDataset,
    lf: ModelDataset,
    *,
    degree: int = 3,
    q_norm: float = 0.75,
    max_interaction: int | None = 2,
    cv_folds: int = 5,
) -> List[Dict[str, Any]]:
    if hf.input_names != lf.input_names:
        raise ValueError("HF and LF datasets use different input columns")
    hf_paired, lf_paired = paired_indices(hf, lf)
    if len(hf_paired) != len(hf.y):
        raise ValueError(f"LF model {lf.model_id} is missing {len(hf.y) - len(hf_paired)} HF pairs")
    lf_pair_by_hf = {int(h): int(l) for h, l in zip(hf_paired, lf_paired)}
    rows: List[Dict[str, Any]] = []
    aggregate_values: Dict[str, Dict[str, List[np.ndarray]]] = {
        method: {"truth": [], "prediction": []}
        for method in ("lf_only", "hf_only", "mf_residual")
    }
    for fold, (hf_train, hf_test) in enumerate(grouped_cv_splits(hf.repetitions, folds=cv_folds)):
        train_groups = set(int(value) for value in hf.repetitions[hf_train])
        lf_train = np.asarray(
            [index for index, group in enumerate(lf.repetitions) if int(group) in train_groups],
            dtype=int,
        )
        hf_train_data = hf.subset(hf_train)
        lf_train_data = lf.subset(lf_train)
        hf_model = _fit(
            hf_train_data,
            degree=degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
            cv_folds=min(cv_folds, len(train_groups)),
        )
        lf_model = _fit(
            lf_train_data,
            degree=degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
            cv_folds=min(cv_folds, len(train_groups)),
        )
        paired_lf_train = np.asarray([lf_pair_by_hf[int(index)] for index in hf_train], dtype=int)
        residual_data = ModelDataset(
            model_id=f"delta_{lf.model_id}",
            fidelity="residual",
            input_names=hf.input_names,
            x=hf.x[hf_train],
            y=hf.y[hf_train] - lf.y[paired_lf_train],
            repetitions=hf.repetitions[hf_train],
            cell_ids=hf.cell_ids[hf_train],
            sample_fingerprints=hf.sample_fingerprints[hf_train],
        )
        residual_model = _fit(
            residual_data,
            degree=degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
            cv_folds=min(cv_folds, len(train_groups)),
        )
        truth = hf.y[hf_test]
        lf_prediction = lf_model.predict(hf.x[hf_test])
        hf_prediction = hf_model.predict(hf.x[hf_test])
        mf_prediction = lf_prediction + residual_model.predict(hf.x[hf_test])
        predictions = {
            "lf_only": lf_prediction,
            "hf_only": hf_prediction,
            "mf_residual": mf_prediction,
        }
        for method, prediction in predictions.items():
            aggregate_values[method]["truth"].append(truth)
            aggregate_values[method]["prediction"].append(prediction)
        rows.append(
            _metric_row(
                fold=fold,
                method="lf_only",
                lf_model_id=lf.model_id,
                truth=truth,
                prediction=lf_prediction,
            )
        )
        rows.append(
            _metric_row(
                fold=fold,
                method="hf_only",
                lf_model_id=lf.model_id,
                truth=truth,
                prediction=hf_prediction,
            )
        )
        rows.append(
            _metric_row(
                fold=fold,
                method="mf_residual",
                lf_model_id=lf.model_id,
                truth=truth,
                prediction=mf_prediction,
            )
        )
    for method in ("lf_only", "hf_only", "mf_residual"):
        aggregate = {
            "fold": "aggregate",
            "method": method,
            "lf_model_id": lf.model_id,
            **regression_metrics(
                np.concatenate(aggregate_values[method]["truth"]),
                np.concatenate(aggregate_values[method]["prediction"]),
            ),
        }
        rows.append(aggregate)
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _coefficient_rows(label: str, model: SparsePCEModel) -> List[Dict[str, Any]]:
    rows = []
    active_names = [model.input_names[index] for index in model.active_input_indices]
    effective = model.effective_coefficients()
    for powers, standardized, coefficient in zip(
        model.multi_indices, model.standardized_coefficients, effective
    ):
        factors = [
            f"{name}^{int(power)}" if int(power) != 1 else name
            for name, power in zip(active_names, powers)
            if int(power) > 0
        ]
        rows.append(
            {
                "model_label": label,
                "term": "*".join(factors),
                "total_degree": int(np.sum(powers)),
                "interaction_order": int(np.count_nonzero(powers)),
                "standardized_coefficient": float(standardized),
                "effective_coefficient": float(coefficient),
                "selected": int(abs(float(standardized)) > 1.0e-12),
            }
        )
    return rows


def fit_multifidelity_pce_analysis(
    surrogate_dataset_csv: str,
    output_dir: str,
    *,
    qoi: str = "C_D",
    degree: int = 3,
    q_norm: float = 0.75,
    max_interaction: int | None = 2,
    cv_folds: int = 5,
    max_rows_per_model: int = 0,
) -> Dict[str, Any]:
    datasets, load_summary = load_surrogate_model_datasets(
        surrogate_dataset_csv,
        qoi=qoi,
        max_rows_per_model=max_rows_per_model,
    )
    hf_ids = [model_id for model_id, data in datasets.items() if data.fidelity.lower() in {"hf", "high"}]
    if len(hf_ids) != 1:
        raise ValueError(f"Expected exactly one HF model, found: {hf_ids}")
    hf = datasets[hf_ids[0]]
    lf_datasets = [data for model_id, data in datasets.items() if model_id != hf.model_id]
    if not lf_datasets:
        raise ValueError("No LF datasets found")

    target = Path(output_dir).resolve()
    model_dir = target / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[Dict[str, Any]] = []
    for lf in lf_datasets:
        metric_rows.extend(
            cross_validate_multifidelity_pce(
                hf,
                lf,
                degree=degree,
                q_norm=q_norm,
                max_interaction=max_interaction,
                cv_folds=cv_folds,
            )
        )
    aggregate_mf = [
        row for row in metric_rows if row["fold"] == "aggregate" and row["method"] == "mf_residual"
    ]
    best_lf_id = min(aggregate_mf, key=lambda row: float(row["rmse"]))["lf_model_id"]
    quality_flags: List[str] = []
    if len(hf.y) < 30:
        quality_flags.append("insufficient_hf_samples_lt_30")
    best_metrics = next(row for row in aggregate_mf if row["lf_model_id"] == best_lf_id)
    if float(best_metrics["r2"]) < 0.3:
        quality_flags.append("weak_mf_out_of_fold_r2")
    elif float(best_metrics["r2"]) < 0.7:
        quality_flags.append("moderate_mf_out_of_fold_r2")

    full_models: Dict[str, SparsePCEModel] = {}
    hf_model = _fit(
        hf,
        degree=degree,
        q_norm=q_norm,
        max_interaction=max_interaction,
        cv_folds=cv_folds,
    )
    full_models["hf_only"] = hf_model
    for lf in lf_datasets:
        lf_model = _fit(
            lf,
            degree=degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
            cv_folds=cv_folds,
        )
        hf_pair, lf_pair = paired_indices(hf, lf)
        residual = ModelDataset(
            model_id=f"delta_{lf.model_id}",
            fidelity="residual",
            input_names=hf.input_names,
            x=hf.x[hf_pair],
            y=hf.y[hf_pair] - lf.y[lf_pair],
            repetitions=hf.repetitions[hf_pair],
            cell_ids=hf.cell_ids[hf_pair],
            sample_fingerprints=hf.sample_fingerprints[hf_pair],
        )
        residual_model = _fit(
            residual,
            degree=degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
            cv_folds=cv_folds,
        )
        full_models[f"lf_{lf.model_id}"] = lf_model
        full_models[f"delta_{lf.model_id}"] = residual_model

    coefficient_rows: List[Dict[str, Any]] = []
    model_files: Dict[str, str] = {}
    for label, model in full_models.items():
        model_path = model_dir / f"{_safe_name(label)}.json"
        model.write_json(str(model_path))
        model_files[label] = str(model_path)
        coefficient_rows.extend(_coefficient_rows(label, model))

    metrics_path = target / "gsa_surrogate_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    coefficients_path = target / "gsa_coefficients.csv"
    with coefficients_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(coefficient_rows[0]))
        writer.writeheader()
        writer.writerows(coefficient_rows)

    summary = {
        "status": "pce_complete_gsa_pending" if not quality_flags else "pce_complete_with_quality_flags",
        "surrogate_dataset_csv": os.path.abspath(surrogate_dataset_csv),
        "qoi": qoi,
        "degree": degree,
        "q_norm": q_norm,
        "max_interaction": max_interaction,
        "cv_folds": cv_folds,
        "max_rows_per_model": max_rows_per_model,
        "best_lf_model_id": best_lf_id,
        "quality_flags": quality_flags,
        "load_summary": load_summary,
        "model_files": model_files,
        "metrics_csv": str(metrics_path),
        "coefficients_csv": str(coefficients_path),
    }
    manifest_path = target / "surrogate_pce_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["manifest_json"] = str(manifest_path)
    return summary
