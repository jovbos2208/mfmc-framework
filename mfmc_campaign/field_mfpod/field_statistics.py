"""Pilot statistics for allocation of complete surface-load fields.

The functions in this module operate directly in the discrete surface Hilbert
space.  They never construct a TPMC basis or a ``(state, state)`` second-moment
matrix.  The Hilbert--Schmidt block is evaluated from snapshot Gram matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .models import MFPODError, jsonable


@dataclass(frozen=True)
class FieldPilotStatistics:
    """Two model-covariance blocks used by field-aware allocation."""

    models: tuple[str, ...]
    reference_field: np.ndarray
    mean_covariance_raw: np.ndarray
    mean_covariance: np.ndarray
    second_moment_covariance_raw: np.ndarray
    second_moment_covariance: np.ndarray
    diagnostics: dict

    @property
    def covariance_blocks(self) -> np.ndarray:
        return np.stack((self.mean_covariance, self.second_moment_covariance))

    def as_dict(self) -> dict:
        return jsonable(
            {
                "models": self.models,
                "reference_field": self.reference_field,
                "mean_covariance_raw": self.mean_covariance_raw,
                "mean_covariance": self.mean_covariance,
                "second_moment_covariance_raw": self.second_moment_covariance_raw,
                "second_moment_covariance": self.second_moment_covariance,
                "diagnostics": self.diagnostics,
            }
        )


def clean_paired_fields(
    fields: Mapping[str, np.ndarray], target: str = "DSMC"
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict]:
    """Validate paired full fields and remove rows nonfinite in any model."""

    target = str(target).upper()
    lookup = {str(name).upper(): np.asarray(values, dtype=np.float64) for name, values in fields.items()}
    if target not in lookup:
        raise MFPODError(f"Paired fields do not contain target {target}")
    models = (target, *sorted(name for name in lookup if name != target))
    arrays = [lookup[name] for name in models]
    if any(array.ndim != 2 for array in arrays):
        raise MFPODError("Every paired field array must have shape (sample, state)")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise MFPODError("Paired field arrays must have identical sample and state dimensions")
    if arrays[0].shape[0] < 2 or arrays[0].shape[1] < 1:
        raise MFPODError("At least two paired samples and one field degree of freedom are required")
    finite = np.logical_and.reduce([np.all(np.isfinite(array), axis=1) for array in arrays])
    kept = int(np.sum(finite))
    if kept < 2:
        raise MFPODError("Fewer than two finite paired pilot fields remain")
    return (
        models,
        {name: lookup[name][finite] for name in models},
        {
            "paired_rows": kept,
            "dropped_nonfinite_rows": int(finite.size - kept),
            "state_dimension": int(arrays[0].shape[1]),
        },
    )


def regularize_model_covariance(
    covariance: np.ndarray, *, ridge: float = 1.0e-10, psd_floor: float = 0.0
) -> tuple[np.ndarray, dict]:
    """Symmetrize, PSD-project, and ridge a small model covariance matrix."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MFPODError("Model covariance must be square")
    if not np.all(np.isfinite(matrix)):
        raise MFPODError("Model covariance contains NaN or infinite values")
    if not np.isfinite(ridge) or ridge < 0.0 or not np.isfinite(psd_floor) or psd_floor < 0.0:
        raise MFPODError("Covariance ridge and PSD floor must be finite and nonnegative")
    symmetric = 0.5 * (matrix + matrix.T)
    raw_values, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(raw_values))), np.finfo(float).eps)
    absolute_floor = float(psd_floor) * scale
    projected_values = np.maximum(raw_values, absolute_floor)
    corrected = (vectors * projected_values) @ vectors.T
    ridge_value = float(ridge) * scale
    corrected += ridge_value * np.eye(matrix.shape[0])
    corrected = 0.5 * (corrected + corrected.T)
    final_values = np.linalg.eigvalsh(corrected)
    return corrected, {
        "raw_eigenvalues": raw_values.tolist(),
        "corrected_eigenvalues": final_values.tolist(),
        "psd_eigenvalues_clipped": int(np.sum(raw_values < absolute_floor)),
        "psd_floor_absolute": absolute_floor,
        "ridge_absolute": ridge_value,
        "condition_number": float(np.linalg.cond(corrected)),
    }


def mean_field_model_covariance(centered_fields: Mapping[str, np.ndarray], models: tuple[str, ...]) -> np.ndarray:
    """Return ``Gamma^(mu)`` using complete-field inner products."""

    demeaned = {name: centered_fields[name] - np.mean(centered_fields[name], axis=0) for name in models}
    sample_count = centered_fields[models[0]].shape[0]
    result = np.empty((len(models), len(models)), dtype=np.float64)
    for i, left in enumerate(models):
        for j, right in enumerate(models[i:], start=i):
            value = float(np.einsum("nd,nd->", demeaned[left], demeaned[right]) / (sample_count - 1))
            result[i, j] = result[j, i] = value
    return result


def second_moment_model_covariance(
    centered_fields: Mapping[str, np.ndarray], models: tuple[str, ...]
) -> np.ndarray:
    """Return ``Gamma^(M)`` from squared cross-Gram matrices.

    For model pair ``(l, m)``, the diagonal of the squared Gram matrix contains
    ``<Y_l^r, Y_m^r>`` while its full sum gives
    ``p^2 <mean(Y_l), mean(Y_m)>``.
    """

    sample_count = centered_fields[models[0]].shape[0]
    result = np.empty((len(models), len(models)), dtype=np.float64)
    for i, left in enumerate(models):
        for j, right in enumerate(models[i:], start=i):
            gram = centered_fields[left] @ centered_fields[right].T
            squared = np.square(gram)
            paired_sum = float(np.trace(squared))
            mean_inner = float(np.sum(squared) / (sample_count * sample_count))
            value = (paired_sum - sample_count * mean_inner) / (sample_count - 1)
            result[i, j] = result[j, i] = value
    return result


def compute_field_pilot_statistics(
    fields: Mapping[str, np.ndarray],
    *,
    target: str = "DSMC",
    reference_field: np.ndarray | None = None,
    covariance_ridge: float = 1.0e-10,
    psd_floor: float = 0.0,
) -> FieldPilotStatistics:
    """Estimate full-field mean and second-moment allocation statistics."""

    models, paired, cleaning = clean_paired_fields(fields, target)
    state_dimension = paired[models[0]].shape[1]
    if reference_field is None:
        reference = np.mean(paired[models[0]], axis=0)
        reference_source = "pilot_target_mean"
    else:
        reference = np.asarray(reference_field, dtype=np.float64).reshape(-1)
        if reference.shape != (state_dimension,) or not np.all(np.isfinite(reference)):
            raise MFPODError("reference_field must be a finite vector matching the field dimension")
        reference_source = "provided"
    centered = {name: paired[name] - reference for name in models}
    mean_raw = mean_field_model_covariance(centered, models)
    second_raw = second_moment_model_covariance(centered, models)
    mean_covariance, mean_diagnostics = regularize_model_covariance(
        mean_raw, ridge=covariance_ridge, psd_floor=psd_floor
    )
    second_covariance, second_diagnostics = regularize_model_covariance(
        second_raw, ridge=covariance_ridge, psd_floor=psd_floor
    )
    return FieldPilotStatistics(
        models=models,
        reference_field=reference,
        mean_covariance_raw=mean_raw,
        mean_covariance=mean_covariance,
        second_moment_covariance_raw=second_raw,
        second_moment_covariance=second_covariance,
        diagnostics={
            **cleaning,
            "reference_source": reference_source,
            "mean_block": mean_diagnostics,
            "second_moment_block": second_diagnostics,
            "second_moment_computation": "squared_snapshot_cross_gram",
        },
    )
