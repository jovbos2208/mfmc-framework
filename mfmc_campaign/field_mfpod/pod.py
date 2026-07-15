from __future__ import annotations

import numpy as np

from .models import MFPODError, PODResult


def compute_pod(snapshots: np.ndarray, n_modes: int | None = None, label: str = "POD") -> PODResult:
    x = np.asarray(snapshots, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise MFPODError("POD snapshots must have shape (n_samples, n_state)")
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    k = min(n_modes or vt.shape[0], vt.shape[0])
    modes = vt[:k].T
    values = singular[:k] ** 2 / x.shape[0]
    return PODResult(modes=modes, eigenvalues=values, backend="svd", diagnostics={"method": label, "rank": int(np.linalg.matrix_rank(x)), "orthogonality_error_fro": float(np.linalg.norm(modes.T @ modes - np.eye(k)))})


def compute_hf_pod(snapshots: np.ndarray, n_modes: int | None = None) -> PODResult:
    return compute_pod(snapshots, n_modes, "HF-only POD")


def compute_lf_pod(snapshots: np.ndarray, n_modes: int | None = None) -> PODResult:
    return compute_pod(snapshots, n_modes, "LF-only POD")


def select_dimensions(eigenvalues: np.ndarray, fixed_dimensions: list[int], thresholds: list[float]) -> dict:
    vals = np.clip(np.asarray(eigenvalues, dtype=float), 0, None)
    cumulative = np.cumsum(vals) / np.sum(vals) if np.sum(vals) > 0 else np.zeros(vals.size)
    return {
        "fixed": [int(r) for r in fixed_dimensions if 1 <= r <= vals.size],
        "thresholds": [{"threshold": float(k), "selected_dimension": int(np.searchsorted(cumulative, k) + 1) if cumulative.size and cumulative[-1] >= k else None} for k in thresholds],
    }
