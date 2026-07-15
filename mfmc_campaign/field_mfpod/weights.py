from __future__ import annotations

from typing import Optional

import numpy as np

from .models import MFPODError


def estimate_global_mfpod_weight(
    hf_paired: np.ndarray,
    lf_paired: np.ndarray,
    *,
    variance_tolerance: float = 1.0e-14,
    bootstrap_repeats: int = 500,
    random_seed: int = 1101,
    alpha_bounds: Optional[tuple[float, float]] = None,
) -> dict:
    """Estimate the v1 global energy control-variate weight from paired data."""
    h = np.asarray(hf_paired, dtype=float)
    l = np.asarray(lf_paired, dtype=float)
    if h.shape != l.shape or h.ndim != 2 or h.shape[0] < 2:
        raise MFPODError("Global weight requires at least two paired, same-size snapshot rows")
    x = np.einsum("ij,ij->i", h, h)
    y = np.einsum("ij,ij->i", l, l)
    var_y = float(np.var(y, ddof=1))
    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    raw = cov_xy / var_y if np.isfinite(var_y) and var_y > variance_tolerance else 0.0
    bounded = raw
    if alpha_bounds is not None:
        bounded = float(np.clip(raw, alpha_bounds[0], alpha_bounds[1]))
    rng = np.random.default_rng(random_seed)
    boots = []
    for _ in range(max(0, int(bootstrap_repeats))):
        idx = rng.integers(0, x.size, x.size)
        vy = float(np.var(y[idx], ddof=1))
        if vy > variance_tolerance:
            boots.append(float(np.cov(x[idx], y[idx], ddof=1)[0, 1] / vy))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    return {
        "paired_pilot_size": int(x.size), "sample_covariance": cov_xy,
        "hf_energy_variance": float(np.var(x, ddof=1)), "lf_energy_variance": var_y,
        "energy_correlation": corr, "alpha_raw": float(raw), "alpha": float(bounded),
        "alpha_bounds": alpha_bounds,
        "bootstrap_interval_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [float(raw), float(raw)],
        "bootstrap_values": boots,
        "small_lf_variance_warning": bool(var_y <= variance_tolerance),
        "unstable_alpha_warning": bool(boots and (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) > 2 * max(abs(raw), 1e-12)),
    }


def residual_energy_weight(hf_paired: np.ndarray, lf_paired: np.ndarray, basis: np.ndarray, variance_tolerance: float = 1e-14) -> dict:
    def residual_energy(x: np.ndarray) -> np.ndarray:
        if basis.size == 0:
            return np.einsum("ij,ij->i", x, x)
        residual = x - (x @ basis) @ basis.T
        return np.einsum("ij,ij->i", residual, residual)
    x, y = residual_energy(hf_paired), residual_energy(lf_paired)
    vy = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    cov = float(np.cov(x, y, ddof=1)[0, 1]) if y.size > 1 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    return {"alpha": cov / vy if vy > variance_tolerance else 0.0, "hf_residual_variance": float(np.var(x, ddof=1)) if x.size > 1 else 0.0, "lf_residual_variance": vy, "residual_energy_correlation": corr}
