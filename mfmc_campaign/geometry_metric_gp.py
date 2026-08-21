from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning

from .geometry_design import VARIABLES
from .geometry_mf_surrogate import _evaluation_lookup, _robust_metrics
from .sparse_pce import regression_metrics


METRICS = ("mean_drag", "std_drag", "q95_drag")
DEFAULT_RELATIVE_RMSE_TARGETS = {
    "mean_drag": 0.03,
    "std_drag": 0.15,
    "q95_drag": 0.03,
}


def _matern52(left: np.ndarray, right: np.ndarray, constant: float, length_scale: np.ndarray) -> np.ndarray:
    delta = (left[:, None, :] - right[None, :, :]) / length_scale[None, None, :]
    radius = np.sqrt(np.sum(delta * delta, axis=2))
    scaled = np.sqrt(5.0) * radius
    return constant * (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)


@dataclass(frozen=True)
class GeometryMetricGPModel:
    metric: str
    input_names: tuple[str, ...]
    x_train: np.ndarray
    y_train_standardized: np.ndarray
    y_mean: float
    y_scale: float
    noise_variance_standardized: np.ndarray
    constant: float
    length_scale: np.ndarray
    jitter: float = 1.0e-10

    def predict(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.atleast_2d(np.asarray(values, dtype=float))
        covariance = _matern52(self.x_train, self.x_train, self.constant, self.length_scale)
        covariance[np.diag_indices_from(covariance)] += self.noise_variance_standardized + self.jitter
        cross = _matern52(self.x_train, x, self.constant, self.length_scale)
        weights = np.linalg.solve(covariance, self.y_train_standardized)
        mean = cross.T @ weights
        solved = np.linalg.solve(covariance, cross)
        variance = np.maximum(self.constant - np.sum(cross * solved, axis=0), 0.0)
        return self.y_mean + self.y_scale * mean, self.y_scale * np.sqrt(variance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "model_type": "heteroskedastic_matern52_gaussian_process",
            "metric": self.metric,
            "input_names": list(self.input_names),
            "x_train": self.x_train.tolist(),
            "y_train_standardized": self.y_train_standardized.tolist(),
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "noise_variance_standardized": self.noise_variance_standardized.tolist(),
            "constant": self.constant,
            "length_scale": self.length_scale.tolist(),
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryMetricGPModel":
        return cls(
            metric=str(payload["metric"]),
            input_names=tuple(str(value) for value in payload["input_names"]),
            x_train=np.asarray(payload["x_train"], dtype=float),
            y_train_standardized=np.asarray(payload["y_train_standardized"], dtype=float),
            y_mean=float(payload["y_mean"]),
            y_scale=float(payload["y_scale"]),
            noise_variance_standardized=np.asarray(payload["noise_variance_standardized"], dtype=float),
            constant=float(payload["constant"]),
            length_scale=np.asarray(payload["length_scale"], dtype=float),
            jitter=float(payload.get("jitter", 1.0e-10)),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "GeometryMetricGPModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fit_gp(
    metric: str,
    x: np.ndarray,
    y: np.ndarray,
    noise_variance: np.ndarray,
    *,
    optimizer_restarts: int,
    seed: int,
) -> GeometryMetricGPModel:
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y, ddof=1))
    if not np.isfinite(y_scale) or y_scale < 1.0e-15:
        y_scale = 1.0
    standardized = (y - y_mean) / y_scale
    standardized_noise = np.maximum(noise_variance / (y_scale * y_scale), 1.0e-10)
    kernel = ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.ones(x.shape[1]),
        length_scale_bounds=(1.0e-2, 1.0e2),
        nu=2.5,
    )
    with warnings.catch_warnings():
        # Boundary length scales are retained in the JSON artifact and mean that
        # a geometry direction is effectively flat; repeating this warning for
        # every leave-one-out fit obscures the actual validation report.
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = GaussianProcessRegressor(
            kernel=kernel,
            alpha=standardized_noise + 1.0e-10,
            normalize_y=False,
            n_restarts_optimizer=max(0, int(optimizer_restarts)),
            random_state=int(seed),
        ).fit(x, standardized)
    return GeometryMetricGPModel(
        metric=metric,
        input_names=tuple(VARIABLES),
        x_train=x,
        y_train_standardized=standardized,
        y_mean=y_mean,
        y_scale=y_scale,
        noise_variance_standardized=standardized_noise,
        constant=float(fitted.kernel_.k1.constant_value),
        length_scale=np.asarray(fitted.kernel_.k2.length_scale, dtype=float),
    )


def _bootstrap_metric_variances(values: np.ndarray, count: int, rng: np.random.Generator) -> Dict[str, float]:
    if count < 2:
        raise ValueError("bootstrap_count must be at least two")
    indices = rng.integers(0, len(values), size=(count, len(values)))
    replicas = values[indices]
    estimates = {
        "mean_drag": np.mean(replicas, axis=1),
        "std_drag": np.std(replicas, axis=1, ddof=1),
        "q95_drag": np.quantile(replicas, 0.95, axis=1),
    }
    return {name: float(max(np.var(samples, ddof=1), 1.0e-24)) for name, samples in estimates.items()}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_geometry_metric_gp(
    bundle_json: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_count: int = 500,
    optimizer_restarts: int = 4,
    seed: int = 20260821,
    relative_rmse_targets: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Fit three geometry-only GPs to empirical TPMC mean, standard deviation and q95.

    Bootstrap variances provide geometry- and metric-specific observation noise.
    Leave-one-geometry-out predictions are always refitted without the held-out point.
    """
    source = Path(bundle_json).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("reference_area_convention") != "canonical_manifest_area":
        raise ValueError("Geometry metric GP requires canonical_manifest_area")
    geometry_ids = [str(value) for value in payload["selected_geometry_ids"]]
    if len(geometry_ids) < 4:
        raise ValueError("At least four geometries are required for geometry-held-out GP validation")
    lookup = _evaluation_lookup(payload, "PICLas_TPMC")
    x = np.asarray([
        [float(payload["geometries"][geometry_id]["design"][f"normalized_{name}"]) for name in VARIABLES]
        for geometry_id in geometry_ids
    ])
    rng = np.random.default_rng(seed)
    observed_rows: list[Dict[str, Any]] = []
    metric_values: Dict[str, list[float]] = {name: [] for name in METRICS}
    metric_noise: Dict[str, list[float]] = {name: [] for name in METRICS}
    for geometry_id in geometry_ids:
        keys = sorted(key for key in lookup if key[0] == geometry_id)
        if len(keys) < 5:
            raise ValueError(f"Geometry {geometry_id} has fewer than five TPMC evaluations")
        values = np.asarray([float(lookup[key]["drag_area_m2"]) for key in keys], dtype=float)
        metrics = _robust_metrics(values)
        noise = _bootstrap_metric_variances(values, bootstrap_count, rng)
        row: Dict[str, Any] = {"geometry_id": geometry_id, "n_tpmc": len(values)}
        row.update({f"normalized_{name}": float(x[len(observed_rows), index]) for index, name in enumerate(VARIABLES)})
        for name in METRICS:
            metric_values[name].append(float(metrics[name]))
            metric_noise[name].append(float(noise[name]))
            row[name] = float(metrics[name])
            row[f"{name}_bootstrap_se"] = float(np.sqrt(noise[name]))
        observed_rows.append(row)

    loo_rows: list[Dict[str, Any]] = []
    for held_out, geometry_id in enumerate(geometry_ids):
        train = np.arange(len(geometry_ids)) != held_out
        for metric_index, metric in enumerate(METRICS):
            y = np.asarray(metric_values[metric], dtype=float)
            noise = np.asarray(metric_noise[metric], dtype=float)
            model = _fit_gp(
                metric, x[train], y[train], noise[train],
                optimizer_restarts=optimizer_restarts,
                seed=seed + held_out * len(METRICS) + metric_index,
            )
            prediction, posterior_std = model.predict(x[held_out])
            observation_std = float(np.sqrt(noise[held_out]))
            predictive_std = float(np.hypot(posterior_std[0], observation_std))
            loo_rows.append({
                "held_out_geometry_id": geometry_id,
                "metric": metric,
                "observed": float(y[held_out]),
                "predicted": float(prediction[0]),
                "posterior_std": float(posterior_std[0]),
                "observation_bootstrap_std": observation_std,
                "combined_predictive_std": predictive_std,
                "error": float(prediction[0] - y[held_out]),
            })

    targets = {**DEFAULT_RELATIVE_RMSE_TARGETS, **dict(relative_rmse_targets or {})}
    validation_rows: list[Dict[str, Any]] = []
    for metric in METRICS:
        rows = [row for row in loo_rows if row["metric"] == metric]
        observed = np.asarray([float(row["observed"]) for row in rows])
        predicted = np.asarray([float(row["predicted"]) for row in rows])
        scores = regression_metrics(observed, predicted)
        relative_rmse = float(scores["rmse"] / max(float(np.mean(np.abs(observed))), 1.0e-15))
        combined_std = np.asarray([float(row["combined_predictive_std"]) for row in rows])
        absolute_error = np.abs(predicted - observed)
        validation_rows.append({
            "metric": metric,
            **scores,
            "relative_rmse": relative_rmse,
            "relative_rmse_target": float(targets[metric]),
            "target_met": relative_rmse <= float(targets[metric]),
            "coverage_95": float(np.mean(absolute_error <= 1.96 * combined_std)),
            "mean_absolute_standardized_error": float(
                np.mean(absolute_error / np.maximum(combined_std, 1.0e-15))
            ),
        })

    target = Path(output_dir).resolve()
    model_dir = target / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_paths: Dict[str, str] = {}
    for metric_index, metric in enumerate(METRICS):
        model = _fit_gp(
            metric,
            x,
            np.asarray(metric_values[metric]),
            np.asarray(metric_noise[metric]),
            optimizer_restarts=optimizer_restarts,
            seed=seed + 1000 + metric_index,
        )
        path = model_dir / f"{metric}_gp.json"
        model.write_json(path)
        model_paths[metric] = str(path)

    observations_path = target / "geometry_metric_observations.csv"
    loo_path = target / "geometry_metric_gp_loo.csv"
    validation_path = target / "geometry_metric_gp_validation.csv"
    _write_csv(observations_path, observed_rows)
    _write_csv(loo_path, loo_rows)
    _write_csv(validation_path, validation_rows)
    ready = len(geometry_ids) >= 12 and all(bool(row["target_met"]) for row in validation_rows)
    summary = {
        "schema_version": 1,
        "status": "geometry_metric_gp_ready_for_optimization_candidate" if ready else "more_geometry_acquisition_required",
        "bundle_json": str(source),
        "source_model": "PICLas_TPMC",
        "fidelity_interpretation": "TPMC robust metrics validated against paired DSMC; no learned DSMC discrepancy applied",
        "qoi": "drag_area_m2",
        "n_geometries": len(geometry_ids),
        "geometry_ids": geometry_ids,
        "input_names": list(VARIABLES),
        "outputs": list(METRICS),
        "noise_model": {
            "type": "nonparametric_bootstrap_variance",
            "bootstrap_count": int(bootstrap_count),
            "seed": int(seed),
        },
        "validation": {
            "method": "leave_one_geometry_out_refit",
            "minimum_geometry_count": 12,
            "rows": validation_rows,
            "all_targets_met": all(bool(row["target_met"]) for row in validation_rows),
        },
        "models": model_paths,
        "observations_csv": str(observations_path),
        "loo_predictions_csv": str(loo_path),
        "validation_csv": str(validation_path),
    }
    manifest_path = target / "geometry_metric_gp_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "manifest_json": str(manifest_path)}
