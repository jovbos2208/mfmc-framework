from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles


def principal_angles(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return subspace_angles(np.asarray(reference), np.asarray(estimate))


def evaluate_subspace(basis: np.ndarray, test_snapshots: np.ndarray, reference_basis: np.ndarray | None = None) -> dict:
    v, z = np.asarray(basis, dtype=float), np.asarray(test_snapshots, dtype=float)
    projected = (z @ v) @ v.T
    denominator = float(np.linalg.norm(z, ord="fro") ** 2)
    captured = float(np.linalg.norm(z @ v, ord="fro") ** 2 / denominator) if denominator else 1.0
    error = float(np.linalg.norm(z - projected, ord="fro") ** 2 / denominator) if denominator else 0.0
    result = {"captured_energy": captured, "projection_error": error, "energy_identity_error": abs(error - (1 - captured)), "finite": bool(np.isfinite(captured) and np.isfinite(error))}
    if reference_basis is not None:
        r = min(v.shape[1], reference_basis.shape[1])
        angles = principal_angles(reference_basis[:, :r], v[:, :r])
        result.update({"principal_angles_rad": angles.tolist(), "maximum_principal_angle_rad": float(np.max(angles)), "rms_principal_angle_rad": float(np.sqrt(np.mean(angles**2))), "sum_squared_sines": float(np.sum(np.sin(angles) ** 2)), "projector_distance_fro": float(np.linalg.norm(v[:, :r] @ v[:, :r].T - reference_basis[:, :r] @ reference_basis[:, :r].T, ord="fro"))})
    return result


def qoi_projection_diagnostics(functional: np.ndarray, basis: np.ndarray, snapshots: np.ndarray) -> dict:
    ell, v, z = np.asarray(functional), np.asarray(basis), np.asarray(snapshots)
    direct = z @ ell
    reconstructed = ((z @ v) @ v.T) @ ell
    variance = float(np.var(direct, ddof=1)) if direct.size > 1 else 0.0
    return {"rmse": float(np.sqrt(np.mean((direct - reconstructed) ** 2))), "variance_captured": float(np.var(reconstructed, ddof=1) / variance) if variance > 0 else 1.0}
