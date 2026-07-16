"""Full-field multiple-control MFMC estimators and matrix-free POD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .models import MFPODError, PODResult, jsonable


@dataclass
class FullFieldMFMCStatistics:
    """Estimated DSMC statistics plus their matrix-free covariance action."""

    mean_field: np.ndarray
    centered_mean: np.ndarray
    covariance: LinearOperator
    reference_field: np.ndarray
    counts: dict[str, int]
    mean_weights: dict[str, float]
    second_moment_weights: dict[str, float]
    diagnostics: dict

    def metadata(self) -> dict:
        return jsonable(
            {
                "mean_field": self.mean_field,
                "centered_mean": self.centered_mean,
                "reference_field": self.reference_field,
                "counts": self.counts,
                "mean_weights": self.mean_weights,
                "second_moment_weights": self.second_moment_weights,
                "diagnostics": self.diagnostics,
            }
        )


def _validated_production_fields(
    fields: Mapping[str, np.ndarray], counts: Mapping[str, int], target: str
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, int]]:
    target = str(target).upper()
    arrays = {str(name).upper(): np.asarray(value, dtype=np.float64) for name, value in fields.items()}
    requested = {str(name).upper(): int(value) for name, value in counts.items() if int(value) > 0}
    if target not in arrays or target not in requested:
        raise MFPODError(f"Production fields and counts must contain target {target}")
    models = (target, *sorted(name for name in requested if name != target))
    state_dimension = arrays[target].shape[1] if arrays[target].ndim == 2 else -1
    n_h = requested[target]
    for name in models:
        array = arrays.get(name)
        if array is None or array.ndim != 2 or array.shape[1] != state_dimension:
            raise MFPODError("All active production fields must have shape (sample, common_state)")
        if requested[name] > array.shape[0]:
            raise MFPODError(f"Requested {requested[name]} {name} fields but only {array.shape[0]} are available")
        if not np.all(np.isfinite(array[: requested[name]])):
            raise MFPODError(f"Active {name} production fields contain NaN or infinite values")
        if name != target and requested[name] < n_h:
            raise MFPODError(f"Active control {name} must have at least the {n_h} paired target rows")
    if n_h < 1 or state_dimension < 1:
        raise MFPODError("At least one target sample and one state degree of freedom are required")
    return models, arrays, requested


def estimate_full_field_mfmc(
    fields: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
    *,
    reference_field: np.ndarray,
    mean_weights: Mapping[str, float] | None = None,
    second_moment_weights: Mapping[str, float] | None = None,
    target: str = "DSMC",
) -> FullFieldMFMCStatistics:
    """Estimate the DSMC mean and covariance from nested full-field samples.

    Rows ``[:n_H]`` of every active control are assumed paired with the target.
    Remaining control rows are the additional low-fidelity evaluations.
    """

    models, arrays, requested = _validated_production_fields(fields, counts, target)
    target = models[0]
    n_h = requested[target]
    dimension = arrays[target].shape[1]
    reference = np.asarray(reference_field, dtype=np.float64).reshape(-1)
    if reference.shape != (dimension,) or not np.all(np.isfinite(reference)):
        raise MFPODError("reference_field must be finite and match the full-field dimension")
    beta_mu = {str(name).upper(): float(value) for name, value in (mean_weights or {}).items()}
    beta_m = {str(name).upper(): float(value) for name, value in (second_moment_weights or {}).items()}
    if not all(np.isfinite(value) for value in (*beta_mu.values(), *beta_m.values())):
        raise MFPODError("Control weights must be finite")

    centered = {name: arrays[name][: requested[name]] - reference for name in models}
    centered_mean = np.mean(centered[target][:n_h], axis=0)
    for name in models[1:]:
        n_i = requested[name]
        centered_mean += beta_mu.get(name, 0.0) * (
            np.mean(centered[name][:n_i], axis=0) - np.mean(centered[name][:n_h], axis=0)
        )
    mean_field = reference + centered_mean

    # Each tuple contributes coefficient * X.T @ X without forming X.T @ X.
    blocks: list[tuple[float, np.ndarray, str]] = [
        (1.0 / n_h, centered[target][:n_h], f"{target}:paired")
    ]
    for name in models[1:]:
        beta = beta_m.get(name, 0.0)
        if beta == 0.0:
            continue
        n_i = requested[name]
        blocks.append((beta / n_i, centered[name][:n_i], f"{name}:all"))
        blocks.append((-beta / n_h, centered[name][:n_h], f"{name}:paired"))

    def action(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64).reshape(dimension)
        result = np.zeros(dimension, dtype=np.float64)
        for coefficient, snapshots, _ in blocks:
            result += coefficient * (snapshots.T @ (snapshots @ value))
        result -= centered_mean * float(centered_mean @ value)
        return result

    operator = LinearOperator(
        (dimension, dimension), matvec=action, rmatvec=action, dtype=np.float64
    )
    return FullFieldMFMCStatistics(
        mean_field=mean_field,
        centered_mean=centered_mean,
        covariance=operator,
        reference_field=reference,
        counts={name: requested[name] for name in models},
        mean_weights={name: beta_mu.get(name, 0.0) for name in models[1:]},
        second_moment_weights={name: beta_m.get(name, 0.0) for name in models[1:]},
        diagnostics={
            "target": target,
            "models": list(models),
            "state_dimension": dimension,
            "operator_terms": [
                {"label": label, "coefficient": coefficient, "sample_count": int(snapshots.shape[0])}
                for coefficient, snapshots, label in blocks
            ],
            "matrix_formed": False,
            "separate_mean_and_second_moment_weights": True,
        },
    )


def explicit_full_field_covariance(statistics: FullFieldMFMCStatistics) -> np.ndarray:
    """Materialize the operator for small-problem verification only."""

    dimension = statistics.covariance.shape[0]
    identity = np.eye(dimension)
    matrix = np.column_stack([statistics.covariance @ identity[:, j] for j in range(dimension)])
    return 0.5 * (matrix + matrix.T)


def solve_full_field_pod(
    statistics: FullFieldMFMCStatistics,
    *,
    n_modes: int,
    tolerance: float = 1.0e-8,
    max_iterations: int = 5000,
    negative_eigenvalue_tolerance: float = 1.0e-10,
    clip_small_negative_eigenvalues: bool = False,
    random_seed: int = 2202,
) -> PODResult:
    """Compute leading full-field POD modes and preserve raw PSD diagnostics."""

    dimension = statistics.covariance.shape[0]
    if n_modes < 1 or not np.isfinite(tolerance) or tolerance <= 0.0 or max_iterations < 1:
        raise MFPODError("POD mode count, tolerance, and iteration limit must be positive")
    if negative_eigenvalue_tolerance < 0.0 or not np.isfinite(negative_eigenvalue_tolerance):
        raise MFPODError("negative_eigenvalue_tolerance must be finite and nonnegative")
    requested = min(int(n_modes), dimension)
    if dimension == 1 or requested == dimension:
        matrix = explicit_full_field_covariance(statistics)
        raw_values, modes = np.linalg.eigh(matrix)
        order = np.argsort(raw_values)[::-1][:requested]
        raw_values, modes = raw_values[order], modes[:, order]
        backend = "explicit_eigh"
    else:
        rng = np.random.default_rng(random_seed)
        v0 = rng.normal(size=dimension)
        raw_values, modes = eigsh(
            statistics.covariance,
            k=requested,
            which="LA",
            tol=float(tolerance),
            maxiter=int(max_iterations),
            v0=v0,
        )
        order = np.argsort(raw_values)[::-1]
        raw_values, modes = raw_values[order], modes[:, order]
        backend = "eigsh_linear_operator"
    residuals = np.asarray(
        [
            np.linalg.norm(statistics.covariance @ modes[:, j] - raw_values[j] * modes[:, j])
            for j in range(raw_values.size)
        ]
    )
    small_negative = (raw_values < 0.0) & (raw_values >= -negative_eigenvalue_tolerance)
    large_negative = raw_values < -negative_eigenvalue_tolerance
    eigenvalues = raw_values.copy()
    if clip_small_negative_eigenvalues:
        eigenvalues[small_negative] = 0.0
    diagnostics = {
        **statistics.diagnostics,
        "backend": backend,
        "raw_ritz_values": raw_values.tolist(),
        "minimum_computed_ritz_eigenvalue": float(np.min(raw_values)),
        "negative_eigenvalue_count": int(np.sum(raw_values < 0.0)),
        "large_negative_eigenvalue_count": int(np.sum(large_negative)),
        "maximum_negative_eigenvalue_magnitude": float(np.max(np.abs(raw_values[raw_values < 0.0]))) if np.any(raw_values < 0.0) else 0.0,
        "negative_eigenvalue_tolerance": float(negative_eigenvalue_tolerance),
        "small_negative_clipping_applied": bool(clip_small_negative_eigenvalues and np.any(small_negative)),
        "clipped_indices": np.flatnonzero(small_negative).tolist() if clip_small_negative_eigenvalues else [],
        "eigenpair_residuals": residuals.tolist(),
        "maximum_eigenpair_residual": float(np.max(residuals)) if residuals.size else 0.0,
        "eigensolver_tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "random_seed": int(random_seed),
        "converged": True,
    }
    return PODResult(modes=modes, eigenvalues=eigenvalues, backend=backend, diagnostics=diagnostics)


def covariance_probe_error(
    estimate: LinearOperator,
    reference: LinearOperator,
    *,
    probe_count: int = 100,
    random_seed: int = 4401,
) -> float:
    """Estimate relative covariance-operator error with fixed Gaussian probes."""

    if estimate.shape != reference.shape or probe_count < 1:
        raise MFPODError("Probe operators must have matching shape and probe_count must be positive")
    rng = np.random.default_rng(random_seed)
    numerator = 0.0
    denominator = 0.0
    for _ in range(probe_count):
        probe = rng.normal(size=estimate.shape[1])
        reference_action = reference @ probe
        difference = estimate @ probe - reference_action
        numerator += float(difference @ difference)
        denominator += float(reference_action @ reference_action)
    if denominator <= np.finfo(float).eps:
        raise MFPODError("Reference covariance has negligible probe energy")
    return float(np.sqrt(numerator / denominator))
