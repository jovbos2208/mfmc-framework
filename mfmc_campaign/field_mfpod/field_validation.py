"""Validation metrics for full-field DSMC-target MFMC and POD."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .metrics import evaluate_subspace
from .models import MFPODError


def relative_field_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if estimate.shape != reference.shape:
        raise MFPODError("Field estimate and reference must have matching shape")
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(estimate - reference) / denominator) if denominator > 0.0 else float(np.linalg.norm(estimate))


def leading_eigenvalue_error(estimate: np.ndarray, reference: np.ndarray) -> dict:
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    count = min(estimate.size, reference.size)
    if count == 0:
        raise MFPODError("At least one eigenvalue is required")
    scale = np.maximum(np.abs(reference[:count]), np.finfo(float).eps)
    relative = np.abs(estimate[:count] - reference[:count]) / scale
    return {
        "relative_errors": relative.tolist(),
        "mean_relative_error": float(np.mean(relative)),
        "maximum_relative_error": float(np.max(relative)),
    }


def pod_validation_metrics(
    estimated_modes: np.ndarray,
    reference_modes: np.ndarray,
    heldout_dsmc_fields: np.ndarray,
    *,
    estimated_mean: np.ndarray,
    reference_mean: np.ndarray,
) -> dict:
    heldout = np.asarray(heldout_dsmc_fields, dtype=float)
    estimated_centered = heldout - np.asarray(estimated_mean, dtype=float)
    reference_centered = heldout - np.asarray(reference_mean, dtype=float)
    metrics = evaluate_subspace(
        np.asarray(estimated_modes, dtype=float),
        estimated_centered,
        np.asarray(reference_modes, dtype=float),
    )
    reference_metrics = evaluate_subspace(np.asarray(reference_modes, dtype=float), reference_centered)
    metrics["reference_projection_error"] = reference_metrics["projection_error"]
    return metrics


def panelwise_mean_standard_deviation_error(
    estimated_mean: np.ndarray,
    estimated_standard_deviation: np.ndarray,
    reference_fields: np.ndarray,
) -> dict:
    fields = np.asarray(reference_fields, dtype=float)
    if fields.ndim != 2 or fields.shape[1] % 3:
        raise MFPODError("Panelwise metrics require flattened three-component fields")
    reference_mean = np.mean(fields, axis=0)
    reference_std = np.std(fields, axis=0, ddof=1)
    mean_error = np.asarray(estimated_mean, dtype=float) - reference_mean
    std_error = np.asarray(estimated_standard_deviation, dtype=float) - reference_std
    return {
        "mean_relative_error": relative_field_error(estimated_mean, reference_mean),
        "standard_deviation_relative_error": relative_field_error(estimated_standard_deviation, reference_std),
        "panel_mean_error_norms": np.linalg.norm(mean_error.reshape(-1, 3), axis=1).tolist(),
        "panel_standard_deviation_error_norms": np.linalg.norm(std_error.reshape(-1, 3), axis=1).tolist(),
    }


def normal_tangential_error(
    estimated: np.ndarray, reference: np.ndarray, face_normals: np.ndarray
) -> dict:
    estimate = np.asarray(estimated, dtype=float).reshape(-1, 3)
    truth = np.asarray(reference, dtype=float).reshape(-1, 3)
    normals = np.asarray(face_normals, dtype=float)
    if estimate.shape != truth.shape or normals.shape != estimate.shape:
        raise MFPODError("Normal/tangential metrics require one normal per three-component face")
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm <= 0.0):
        raise MFPODError("Face normals must be nonzero")
    unit = normals / norm[:, None]
    estimate_normal = np.sum(estimate * unit, axis=1)
    truth_normal = np.sum(truth * unit, axis=1)
    estimate_tangent = estimate - estimate_normal[:, None] * unit
    truth_tangent = truth - truth_normal[:, None] * unit
    return {
        "normal_relative_error": relative_field_error(estimate_normal, truth_normal),
        "tangential_relative_error": relative_field_error(estimate_tangent, truth_tangent),
    }


def functional_errors(
    estimated_mean: np.ndarray,
    reference_mean: np.ndarray,
    functionals: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    result = {}
    for name, functional in functionals.items():
        linear = np.asarray(functional, dtype=float)
        estimate = np.asarray(estimated_mean) @ linear
        reference = np.asarray(reference_mean) @ linear
        denominator = np.maximum(np.abs(reference), np.finfo(float).eps)
        result[str(name)] = {
            "estimate": np.asarray(estimate).tolist(),
            "reference": np.asarray(reference).tolist(),
            "relative_error": np.asarray(np.abs(estimate - reference) / denominator).tolist(),
        }
    return result
