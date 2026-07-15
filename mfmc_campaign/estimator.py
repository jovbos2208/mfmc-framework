from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


EPS = 1e-14


def _nan_if_invalid(x: float) -> float:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return float("nan")
    return float(x)


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    arr = np.asarray(x, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_var(x: np.ndarray, ddof: int = 1) -> float:
    arr = np.asarray(x, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size <= ddof:
        return float("nan")
    return float(np.var(finite, ddof=ddof))


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(x.size, y.size)
    x_arr = np.asarray(x[:n], dtype=float)
    y_arr = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _safe_cov(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> float:
    x_arr, y_arr = _finite_pairs(x, y)
    if x_arr.size <= ddof or y_arr.size <= ddof:
        return float("nan")
    return float(np.cov(x_arr, y_arr, ddof=ddof)[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_arr, y_arr = _finite_pairs(x, y)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    if np.std(x_arr) < EPS or np.std(y_arr) < EPS:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_arr, y_arr = _finite_pairs(x, y)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x_arr))
    ry = np.argsort(np.argsort(y_arr))
    return pearson_corr(rx.astype(float), ry.astype(float))


def linear_r2(y_true: np.ndarray, y_pred_source: np.ndarray) -> float:
    if y_true.size < 2 or y_pred_source.size < 2:
        return float("nan")
    n = min(y_true.size, y_pred_source.size)
    y = y_true[:n].astype(float)
    x = y_pred_source[:n].astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    if int(np.sum(mask)) < 2:
        return float("nan")
    y = y[mask]
    x = x[mask]
    if np.nanstd(y) < EPS or np.nanstd(x) < EPS:
        return float("nan")

    X = np.column_stack([np.ones_like(x), x])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return float("nan")
    y_fit = X @ coef
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def _cv_beta(cov_hl: float, var_l: float) -> float:
    if not np.isfinite(cov_hl) or not np.isfinite(var_l) or abs(var_l) < EPS:
        return float("nan")
    return float(cov_hl / var_l)


def _residual_variance(var_h: float, cov_hl: float, var_l: float) -> float:
    if not np.isfinite(var_h):
        return float("nan")
    if not np.isfinite(cov_hl) or not np.isfinite(var_l) or abs(var_l) < EPS:
        return float("nan")
    val = var_h - (cov_hl * cov_hl) / var_l
    return float(max(val, 0.0))


def _matched_cost_hf_error(reference_mean: float, hf_samples: np.ndarray, lf_cost: float, hf_cost: float) -> float:
    if not np.isfinite(reference_mean) or hf_samples.size < 2:
        return float("nan")
    if hf_cost <= 0.0 or lf_cost <= 0.0:
        return float("nan")
    hf_mean = _safe_mean(hf_samples)
    if not np.isfinite(hf_mean):
        return float("nan")
    return float(abs(hf_mean - reference_mean))


def compute_mfmc_diagnostics(
    qoi: str,
    pilot_hf: np.ndarray,
    pilot_lf: np.ndarray,
    prod_hf: np.ndarray,
    prod_lf_full: np.ndarray,
    prod_lf_paired: np.ndarray,
    hf_costs: np.ndarray,
    lf_costs_full: np.ndarray,
    reference: float,
) -> Dict[str, Any]:
    hf_mean = _safe_mean(prod_hf)
    lf_mean = _safe_mean(prod_lf_full)

    hf_var = _safe_var(prod_hf)
    lf_var = _safe_var(prod_lf_full)

    cov_hl = _safe_cov(prod_hf, prod_lf_paired, ddof=1)

    pair_n = min(prod_hf.size, prod_lf_paired.size)
    pearson = pearson_corr(prod_hf[:pair_n], prod_lf_paired[:pair_n])
    spearman = spearman_corr(prod_hf[:pair_n], prod_lf_paired[:pair_n])
    r2_lin = linear_r2(prod_hf[:pair_n], prod_lf_paired[:pair_n])

    _, pilot_lf_paired = _finite_pairs(pilot_hf, pilot_lf)
    pilot_cov = _safe_cov(pilot_hf, pilot_lf, ddof=1)
    pilot_var_l = _safe_var(pilot_lf_paired)
    beta = _cv_beta(pilot_cov, pilot_var_l)

    residual_var = _residual_variance(hf_var, cov_hl, lf_var)

    hf_cost = _safe_mean(hf_costs)
    lf_cost = _safe_mean(lf_costs_full)
    cost_ratio = float(hf_cost / lf_cost) if np.isfinite(hf_cost) and np.isfinite(lf_cost) and lf_cost > 0 else float("nan")

    # CV estimator using production LF full and LF paired means.
    lf_full_mean = _safe_mean(prod_lf_full)
    lf_paired_mean = _safe_mean(prod_lf_paired)
    if np.isfinite(beta):
        mfmc_estimate = float(hf_mean - beta * (lf_paired_mean - lf_full_mean))
    else:
        mfmc_estimate = float(hf_mean)

    hf_only_estimate = float(hf_mean)

    if np.isfinite(reference):
        realized_mfmc_error = float(abs(mfmc_estimate - reference))
        realized_hf_error = float(abs(hf_only_estimate - reference))
    else:
        realized_mfmc_error = float("nan")
        realized_hf_error = float("nan")

    predicted_variance_reduction = (
        float(1.0 - residual_var / hf_var)
        if np.isfinite(residual_var) and np.isfinite(hf_var) and hf_var > EPS
        else float("nan")
    )
    residual_ratio = (
        float(residual_var / hf_var)
        if np.isfinite(residual_var) and np.isfinite(hf_var) and hf_var > EPS
        else float("nan")
    )

    if np.isfinite(realized_mfmc_error) and np.isfinite(realized_hf_error) and realized_mfmc_error > EPS:
        cost_normalized_gain = float((realized_hf_error / realized_mfmc_error) * (hf_cost / (hf_cost + lf_cost)))
    else:
        cost_normalized_gain = float("nan")

    unstable_weight = int(not np.isfinite(beta) or abs(beta) > 1e4)
    underperform_hf = int(
        np.isfinite(realized_mfmc_error)
        and np.isfinite(realized_hf_error)
        and realized_mfmc_error > realized_hf_error
    )

    return {
        "qoi": qoi,
        "rho_hat": _nan_if_invalid(pearson),
        "beta_hat": _nan_if_invalid(beta),
        "r2_lin_hat": _nan_if_invalid(r2_lin),
        "residual_var_hat": _nan_if_invalid(residual_var),
        "residual_ratio_hat": _nan_if_invalid(residual_ratio),
        "hf_mean": _nan_if_invalid(hf_mean),
        "lf_mean": _nan_if_invalid(lf_mean),
        "hf_variance": _nan_if_invalid(hf_var),
        "lf_variance": _nan_if_invalid(lf_var),
        "covariance_hf_lf": _nan_if_invalid(cov_hl),
        "pearson_correlation": _nan_if_invalid(pearson),
        "spearman_correlation": _nan_if_invalid(spearman),
        "control_variate_beta": _nan_if_invalid(beta),
        "residual_variance": _nan_if_invalid(residual_var),
        "cost_ratio": _nan_if_invalid(cost_ratio),
        "predicted_variance_reduction": _nan_if_invalid(predicted_variance_reduction),
        "mfmc_estimate": _nan_if_invalid(mfmc_estimate),
        "hf_only_estimate": _nan_if_invalid(hf_only_estimate),
        "reference_estimate": _nan_if_invalid(reference),
        "realized_mfmc_error": _nan_if_invalid(realized_mfmc_error),
        "realized_hf_error": _nan_if_invalid(realized_hf_error),
        "cost_normalized_gain": _nan_if_invalid(cost_normalized_gain),
        "unstable_weight": unstable_weight,
        "underperform_hf": underperform_hf,
    }


def _finite_matrix_rows(h: np.ndarray, lfs: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not lfs:
        return np.asarray([], dtype=float), np.empty((0, 0), dtype=float)
    n = min([h.size] + [lf.size for lf in lfs])
    h_arr = np.asarray(h[:n], dtype=float)
    lf_mat = np.column_stack([np.asarray(lf[:n], dtype=float) for lf in lfs])
    mask = np.isfinite(h_arr) & np.all(np.isfinite(lf_mat), axis=1)
    return h_arr[mask], lf_mat[mask, :]


def _safe_cov_matrix(mat: np.ndarray, ddof: int = 1) -> np.ndarray:
    if mat.ndim != 2 or mat.shape[0] <= ddof or mat.shape[1] == 0:
        return np.empty((0, 0), dtype=float)
    return np.asarray(np.cov(mat, rowvar=False, ddof=ddof), dtype=float)


def compute_multi_lf_mfmc_diagnostics(
    qoi: str,
    pilot_hf: np.ndarray,
    pilot_lfs: Dict[str, np.ndarray],
    prod_hf: np.ndarray,
    prod_lf_full: Dict[str, np.ndarray],
    prod_lf_paired: Dict[str, np.ndarray],
    hf_costs: np.ndarray,
    lf_costs_full: Dict[str, np.ndarray],
    reference: float,
) -> Dict[str, Any]:
    lf_ids = [lf_id for lf_id in pilot_lfs if lf_id in prod_lf_full and lf_id in prod_lf_paired]
    if not lf_ids:
        return compute_mfmc_diagnostics(
            qoi=qoi,
            pilot_hf=pilot_hf,
            pilot_lf=np.asarray([], dtype=float),
            prod_hf=prod_hf,
            prod_lf_full=np.asarray([], dtype=float),
            prod_lf_paired=np.asarray([], dtype=float),
            hf_costs=hf_costs,
            lf_costs_full=np.asarray([], dtype=float),
            reference=reference,
        )

    pilot_h, pilot_l = _finite_matrix_rows(pilot_hf, [pilot_lfs[lf_id] for lf_id in lf_ids])
    prod_h_pair, prod_l_pair = _finite_matrix_rows(prod_hf, [prod_lf_paired[lf_id] for lf_id in lf_ids])

    hf_mean = _safe_mean(prod_hf)
    hf_var = _safe_var(prod_hf)
    lf_means = np.asarray([_safe_mean(prod_lf_full[lf_id]) for lf_id in lf_ids], dtype=float)
    lf_pair_means = np.asarray([_safe_mean(prod_lf_paired[lf_id]) for lf_id in lf_ids], dtype=float)
    lf_vars = np.asarray([_safe_var(prod_lf_full[lf_id]) for lf_id in lf_ids], dtype=float)

    beta = np.full(len(lf_ids), np.nan, dtype=float)
    residual_var = float("nan")
    if pilot_h.size >= 2 and pilot_l.shape[0] >= 2:
        sigma_ll = _safe_cov_matrix(pilot_l, ddof=1)
        cov_lh = np.asarray([_safe_cov(pilot_h, pilot_l[:, i], ddof=1) for i in range(pilot_l.shape[1])], dtype=float)
        try:
            beta = np.linalg.pinv(sigma_ll) @ cov_lh
        except Exception:
            beta = np.full(len(lf_ids), np.nan, dtype=float)

    if prod_h_pair.size >= 2 and prod_l_pair.shape[0] >= 2:
        sigma_prod = _safe_cov_matrix(prod_l_pair, ddof=1)
        cov_prod = np.asarray([_safe_cov(prod_h_pair, prod_l_pair[:, i], ddof=1) for i in range(prod_l_pair.shape[1])], dtype=float)
        try:
            residual_var = float(max(hf_var - cov_prod @ np.linalg.pinv(sigma_prod) @ cov_prod, 0.0))
        except Exception:
            residual_var = float("nan")

    if np.all(np.isfinite(beta)) and np.all(np.isfinite(lf_pair_means)) and np.all(np.isfinite(lf_means)):
        mfmc_estimate = float(hf_mean - beta @ (lf_pair_means - lf_means))
    else:
        mfmc_estimate = float(hf_mean)

    hf_only_estimate = float(hf_mean)
    if np.isfinite(reference):
        realized_mfmc_error = float(abs(mfmc_estimate - reference))
        realized_hf_error = float(abs(hf_only_estimate - reference))
    else:
        realized_mfmc_error = float("nan")
        realized_hf_error = float("nan")

    pears = np.asarray([pearson_corr(prod_h_pair, prod_l_pair[:, i]) for i in range(prod_l_pair.shape[1])], dtype=float)
    spears = np.asarray([spearman_corr(prod_h_pair, prod_l_pair[:, i]) for i in range(prod_l_pair.shape[1])], dtype=float)
    r2s = np.asarray([linear_r2(prod_h_pair, prod_l_pair[:, i]) for i in range(prod_l_pair.shape[1])], dtype=float)
    covs = np.asarray([_safe_cov(prod_h_pair, prod_l_pair[:, i], ddof=1) for i in range(prod_l_pair.shape[1])], dtype=float)

    finite_abs_pears = np.abs(pears[np.isfinite(pears)])
    pearson = float(np.max(finite_abs_pears)) if finite_abs_pears.size else float("nan")
    spearman = _safe_mean(spears)
    r2_lin = _safe_mean(r2s)
    lf_mean = _safe_mean(lf_means)
    lf_var = _safe_mean(lf_vars)
    cov_hl = _safe_mean(covs)

    hf_cost = _safe_mean(hf_costs)
    lf_cost_values = np.asarray([_safe_mean(lf_costs_full.get(lf_id, np.asarray([], dtype=float))) for lf_id in lf_ids], dtype=float)
    lf_cost = float(np.nansum(lf_cost_values)) if np.any(np.isfinite(lf_cost_values)) else float("nan")
    cost_ratio = float(hf_cost / lf_cost) if np.isfinite(hf_cost) and np.isfinite(lf_cost) and lf_cost > 0 else float("nan")

    predicted_variance_reduction = (
        float(1.0 - residual_var / hf_var)
        if np.isfinite(residual_var) and np.isfinite(hf_var) and hf_var > EPS
        else float("nan")
    )
    residual_ratio = (
        float(residual_var / hf_var)
        if np.isfinite(residual_var) and np.isfinite(hf_var) and hf_var > EPS
        else float("nan")
    )

    if np.isfinite(realized_mfmc_error) and np.isfinite(realized_hf_error) and realized_mfmc_error > EPS:
        cost_normalized_gain = float((realized_hf_error / realized_mfmc_error) * (hf_cost / (hf_cost + lf_cost)))
    else:
        cost_normalized_gain = float("nan")

    beta_norm = float(np.linalg.norm(beta[np.isfinite(beta)])) if np.any(np.isfinite(beta)) else float("nan")
    unstable_weight = int(not np.all(np.isfinite(beta)) or beta_norm > 1e4)
    underperform_hf = int(
        np.isfinite(realized_mfmc_error)
        and np.isfinite(realized_hf_error)
        and realized_mfmc_error > realized_hf_error
    )

    return {
        "qoi": qoi,
        "rho_hat": _nan_if_invalid(pearson),
        "beta_hat": _nan_if_invalid(beta_norm),
        "r2_lin_hat": _nan_if_invalid(r2_lin),
        "residual_var_hat": _nan_if_invalid(residual_var),
        "residual_ratio_hat": _nan_if_invalid(residual_ratio),
        "hf_mean": _nan_if_invalid(hf_mean),
        "lf_mean": _nan_if_invalid(lf_mean),
        "hf_variance": _nan_if_invalid(hf_var),
        "lf_variance": _nan_if_invalid(lf_var),
        "covariance_hf_lf": _nan_if_invalid(cov_hl),
        "pearson_correlation": _nan_if_invalid(pearson),
        "spearman_correlation": _nan_if_invalid(spearman),
        "control_variate_beta": _nan_if_invalid(beta_norm),
        "residual_variance": _nan_if_invalid(residual_var),
        "cost_ratio": _nan_if_invalid(cost_ratio),
        "predicted_variance_reduction": _nan_if_invalid(predicted_variance_reduction),
        "mfmc_estimate": _nan_if_invalid(mfmc_estimate),
        "hf_only_estimate": _nan_if_invalid(hf_only_estimate),
        "reference_estimate": _nan_if_invalid(reference),
        "realized_mfmc_error": _nan_if_invalid(realized_mfmc_error),
        "realized_hf_error": _nan_if_invalid(realized_hf_error),
        "cost_normalized_gain": _nan_if_invalid(cost_normalized_gain),
        "unstable_weight": unstable_weight,
        "underperform_hf": underperform_hf,
    }


def compute_paper_mfmc_diagnostics(
    qoi: str,
    pilot_hf: np.ndarray,
    pilot_lfs: Dict[str, np.ndarray],
    prod_hf: np.ndarray,
    prod_lf_full: Dict[str, np.ndarray],
    lf_sample_counts: Dict[str, int],
    hf_costs: np.ndarray,
    lf_costs_full: Dict[str, np.ndarray],
    reference: float,
    lf_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Peherstorfer/Willcox/Gunzburger MFMC estimator with nested per-LF counts."""
    lf_ids = [
        lf_id
        for lf_id in (lf_order or list(pilot_lfs))
        if lf_id in pilot_lfs and lf_id in prod_lf_full and int(lf_sample_counts.get(lf_id, 0)) > 0
    ]
    if not lf_ids:
        return compute_mfmc_diagnostics(
            qoi=qoi,
            pilot_hf=pilot_hf,
            pilot_lf=np.asarray([], dtype=float),
            prod_hf=prod_hf,
            prod_lf_full=np.asarray([], dtype=float),
            prod_lf_paired=np.asarray([], dtype=float),
            hf_costs=hf_costs,
            lf_costs_full=np.asarray([], dtype=float),
            reference=reference,
        )

    hf_mean = _safe_mean(prod_hf)
    hf_var = _safe_var(prod_hf)
    hf_cost = _safe_mean(hf_costs)
    hf_count = int(prod_hf.size)

    betas: List[float] = []
    pears: List[float] = []
    spears: List[float] = []
    r2s: List[float] = []
    covs: List[float] = []
    lf_means: List[float] = []
    lf_vars: List[float] = []
    correction = 0.0
    valid_correction = False
    predicted_var = float(hf_var / hf_count) if np.isfinite(hf_var) and hf_count > 0 else float("nan")

    prev_count = hf_count
    for lf_id in lf_ids:
        full = np.asarray(prod_lf_full.get(lf_id, []), dtype=float)
        count = min(int(lf_sample_counts.get(lf_id, full.size)), int(full.size))
        count = max(0, count)
        prev = min(prev_count, count)
        full_slice = full[:count]
        prev_slice = full[:prev]

        pilot_lf = np.asarray(pilot_lfs.get(lf_id, []), dtype=float)
        _, pilot_lf_paired = _finite_pairs(pilot_hf, pilot_lf)
        pilot_cov = _safe_cov(pilot_hf, pilot_lf, ddof=1)
        pilot_var_l = _safe_var(pilot_lf_paired)
        beta = _cv_beta(pilot_cov, pilot_var_l)
        betas.append(beta)

        pair_n = min(prod_hf.size, full_slice.size)
        pair_lf = full_slice[:pair_n]
        cov_hl = _safe_cov(prod_hf[:pair_n], pair_lf, ddof=1)
        lf_var = _safe_var(full_slice)
        covs.append(cov_hl)
        lf_vars.append(lf_var)
        lf_means.append(_safe_mean(full_slice))
        pears.append(pearson_corr(prod_hf[:pair_n], pair_lf))
        spears.append(spearman_corr(prod_hf[:pair_n], pair_lf))
        r2s.append(linear_r2(prod_hf[:pair_n], pair_lf))

        lf_full_mean = _safe_mean(full_slice)
        lf_prev_mean = _safe_mean(prev_slice)
        if np.isfinite(beta) and np.isfinite(lf_full_mean) and np.isfinite(lf_prev_mean):
            correction += float(beta) * (lf_full_mean - lf_prev_mean)
            valid_correction = True

        if (
            np.isfinite(predicted_var)
            and np.isfinite(pilot_cov)
            and np.isfinite(pilot_var_l)
            and pilot_var_l > EPS
            and count > prev
            and prev > 0
        ):
            predicted_var -= float((1.0 / prev - 1.0 / count) * (pilot_cov * pilot_cov / pilot_var_l))

        prev_count = count

    mfmc_estimate = float(hf_mean + correction) if valid_correction else float(hf_mean)
    hf_only_estimate = float(hf_mean)

    if np.isfinite(reference):
        realized_mfmc_error = float(abs(mfmc_estimate - reference))
        realized_hf_error = float(abs(hf_only_estimate - reference))
    else:
        realized_mfmc_error = float("nan")
        realized_hf_error = float("nan")

    pears_arr = np.asarray(pears, dtype=float)
    finite_pears = pears_arr[np.isfinite(pears_arr)]
    pearson = float(finite_pears[np.argmax(np.abs(finite_pears))]) if finite_pears.size else float("nan")
    beta_arr = np.asarray(betas, dtype=float)
    finite_beta = beta_arr[np.isfinite(beta_arr)]
    beta_norm = float(np.linalg.norm(finite_beta)) if finite_beta.size else float("nan")

    lf_cost_values = np.asarray([_safe_mean(lf_costs_full.get(lf_id, np.asarray([], dtype=float))) for lf_id in lf_ids], dtype=float)
    lf_cost = float(np.nansum(lf_cost_values)) if np.any(np.isfinite(lf_cost_values)) else float("nan")
    cost_ratio = float(hf_cost / lf_cost) if np.isfinite(hf_cost) and np.isfinite(lf_cost) and lf_cost > 0 else float("nan")
    residual_ratio = (
        float(max(predicted_var, 0.0) / (hf_var / hf_count))
        if np.isfinite(predicted_var) and np.isfinite(hf_var) and hf_count > 0 and hf_var > EPS
        else float("nan")
    )
    predicted_variance_reduction = (
        float(1.0 - residual_ratio) if np.isfinite(residual_ratio) else float("nan")
    )

    if np.isfinite(realized_mfmc_error) and np.isfinite(realized_hf_error) and realized_mfmc_error > EPS:
        cost_normalized_gain = float((realized_hf_error / realized_mfmc_error) * (hf_cost / (hf_cost + lf_cost)))
    else:
        cost_normalized_gain = float("nan")

    unstable_weight = int(not np.all(np.isfinite(beta_arr)) or (np.isfinite(beta_norm) and beta_norm > 1e4))
    underperform_hf = int(
        np.isfinite(realized_mfmc_error)
        and np.isfinite(realized_hf_error)
        and realized_mfmc_error > realized_hf_error
    )

    return {
        "qoi": qoi,
        "rho_hat": _nan_if_invalid(pearson),
        "beta_hat": _nan_if_invalid(beta_norm),
        "r2_lin_hat": _nan_if_invalid(_safe_mean(np.asarray(r2s, dtype=float))),
        "residual_var_hat": _nan_if_invalid(max(predicted_var, 0.0) if np.isfinite(predicted_var) else float("nan")),
        "residual_ratio_hat": _nan_if_invalid(residual_ratio),
        "hf_mean": _nan_if_invalid(hf_mean),
        "lf_mean": _nan_if_invalid(_safe_mean(np.asarray(lf_means, dtype=float))),
        "hf_variance": _nan_if_invalid(hf_var),
        "lf_variance": _nan_if_invalid(_safe_mean(np.asarray(lf_vars, dtype=float))),
        "covariance_hf_lf": _nan_if_invalid(_safe_mean(np.asarray(covs, dtype=float))),
        "pearson_correlation": _nan_if_invalid(pearson),
        "spearman_correlation": _nan_if_invalid(_safe_mean(np.asarray(spears, dtype=float))),
        "control_variate_beta": _nan_if_invalid(beta_norm),
        "residual_variance": _nan_if_invalid(max(predicted_var, 0.0) if np.isfinite(predicted_var) else float("nan")),
        "cost_ratio": _nan_if_invalid(cost_ratio),
        "predicted_variance_reduction": _nan_if_invalid(predicted_variance_reduction),
        "mfmc_estimate": _nan_if_invalid(mfmc_estimate),
        "hf_only_estimate": _nan_if_invalid(hf_only_estimate),
        "reference_estimate": _nan_if_invalid(reference),
        "realized_mfmc_error": _nan_if_invalid(realized_mfmc_error),
        "realized_hf_error": _nan_if_invalid(realized_hf_error),
        "cost_normalized_gain": _nan_if_invalid(cost_normalized_gain),
        "unstable_weight": unstable_weight,
        "underperform_hf": underperform_hf,
        "paper_mfmc_sample_counts": ";".join([f"HF={hf_count}"] + [f"{lf_id}={int(lf_sample_counts.get(lf_id, 0))}" for lf_id in lf_ids]),
        "paper_mfmc_betas": ";".join(f"{lf_id}={beta:.12g}" for lf_id, beta in zip(lf_ids, betas)),
    }


def derive_quantities(direct_means: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if "C_D" in direct_means and "C_D2" in direct_means:
        var_cd = float(direct_means["C_D2"] - direct_means["C_D"] ** 2)
        out["Var_C_D"] = {
            "value": var_cd,
            "derived": True,
            "expression": "E[C_D2]-E[C_D]^2",
        }
    if "C_L" in direct_means and "C_L2" in direct_means:
        var_cl = float(direct_means["C_L2"] - direct_means["C_L"] ** 2)
        out["Var_C_L"] = {
            "value": var_cl,
            "derived": True,
            "expression": "E[C_L2]-E[C_L]^2",
        }
    if "C_Y" in direct_means and "C_Y2" in direct_means:
        var_cy = float(direct_means["C_Y2"] - direct_means["C_Y"] ** 2)
        out["Var_C_Y"] = {
            "value": var_cy,
            "derived": True,
            "expression": "E[C_Y2]-E[C_Y]^2",
        }
    if "C_Mz" in direct_means and "C_Mz2" in direct_means:
        var_cmz = float(direct_means["C_Mz2"] - direct_means["C_Mz"] ** 2)
        out["Var_C_Mz"] = {
            "value": var_cmz,
            "derived": True,
            "expression": "E[C_Mz2]-E[C_Mz]^2",
        }
    return out


def pilot_robustness_metrics(
    pilot_hf: np.ndarray,
    pilot_lf: np.ndarray,
    pilot_sizes: List[int],
    repetitions: int,
    rng: np.random.Generator,
    hf_cost: float = float("nan"),
    lf_cost: float = float("nan"),
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    max_n = min(pilot_hf.size, pilot_lf.size)
    if max_n < 4:
        return records

    for n in pilot_sizes:
        if n < 4 or n > max_n:
            continue

        betas: List[float] = []
        cors: List[float] = []
        alloc_ratios: List[float] = []
        gains: List[float] = []
        underperform = 0
        negative = 0
        unstable = 0
        full_mean = _safe_mean(pilot_hf)

        for _ in range(repetitions):
            idx = rng.choice(max_n, size=n, replace=False)
            x = pilot_hf[idx]
            y = pilot_lf[idx]

            cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
            var_y = _safe_var(y)
            beta = _cv_beta(cov_xy, var_y)
            corr = pearson_corr(x, y)

            betas.append(beta)
            cors.append(corr)
            if np.isfinite(beta) and beta < 0:
                negative += 1
            if not np.isfinite(beta) or abs(beta) > 1e4:
                unstable += 1

            # Simple recommended LF/HF allocation ratio proxy from pilot correlation/cost.
            if np.isfinite(corr) and np.isfinite(hf_cost) and np.isfinite(lf_cost) and hf_cost > 0 and lf_cost > 0:
                corr2 = max(0.0, min(0.999999, corr * corr))
                alloc = ((corr2 / max(EPS, (1.0 - corr2))) * (hf_cost / lf_cost)) ** 0.5
            else:
                alloc = float("nan")
            alloc_ratios.append(float(alloc))

            # Pseudo underperformance estimate using split-sample approximation.
            if max_n > n + 1 and np.isfinite(beta):
                mask = np.ones(max_n, dtype=bool)
                mask[idx] = False
                x_prod = pilot_hf[mask]
                y_prod = pilot_lf[mask]
                n_prod = min(x_prod.size, y_prod.size)
                if n_prod >= 2:
                    hf_est = _safe_mean(x_prod[:n_prod])
                    mfmc_est = float(hf_est - beta * (_safe_mean(y_prod[:n_prod]) - _safe_mean(pilot_lf)))
                    hf_err = abs(hf_est - full_mean)
                    mfmc_err = abs(mfmc_est - full_mean)
                    if np.isfinite(hf_err) and np.isfinite(mfmc_err):
                        if mfmc_err > hf_err:
                            underperform += 1
                        if mfmc_err > EPS:
                            gains.append(float(hf_err / mfmc_err))

        records.append(
            {
                "pilot_size": int(n),
                "correlation_std": _safe_var(np.asarray(cors), ddof=1) ** 0.5 if len(cors) > 1 else float("nan"),
                "correlation_mean": _safe_mean(np.asarray(cors)),
                "beta_std": _safe_var(np.asarray(betas), ddof=1) ** 0.5 if len(betas) > 1 else float("nan"),
                "beta_mean": _safe_mean(np.asarray(betas)),
                "allocation_ratio_std": _safe_var(np.asarray(alloc_ratios), ddof=1) ** 0.5 if len(alloc_ratios) > 1 else float("nan"),
                "allocation_ratio_mean": _safe_mean(np.asarray(alloc_ratios)),
                "negative_weight_frequency": float(negative / max(1, repetitions)),
                "unstable_weight_frequency": float(unstable / max(1, repetitions)),
                "underperform_frequency": float(underperform / max(1, repetitions)),
                "gain_mean": _safe_mean(np.asarray(gains)),
                "gain_p10": float(np.nanpercentile(np.asarray(gains), 10.0)) if gains else float("nan"),
                "gain_p50": float(np.nanpercentile(np.asarray(gains), 50.0)) if gains else float("nan"),
                "gain_p90": float(np.nanpercentile(np.asarray(gains), 90.0)) if gains else float("nan"),
            }
        )

    return records


def beta_stability_metrics(
    pilot_hf: np.ndarray,
    pilot_lf: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    max_n = min(pilot_hf.size, pilot_lf.size)
    if max_n < 4 or repetitions <= 0:
        return {"beta_sign_flip_prob": float("nan"), "beta_cv": float("nan")}

    paired_hf, paired_lf = _finite_pairs(pilot_hf[:max_n], pilot_lf[:max_n])
    full_cov = _safe_cov(paired_hf, paired_lf, ddof=1)
    full_var_l = _safe_var(paired_lf)
    full_beta = _cv_beta(full_cov, full_var_l)
    if not np.isfinite(full_beta) or abs(full_beta) < EPS:
        base_sign = 0
    else:
        base_sign = 1 if full_beta > 0 else -1

    betas: List[float] = []
    flips = 0
    valid_sign_trials = 0
    for _ in range(int(repetitions)):
        idx = rng.choice(max_n, size=max_n, replace=True)
        x = pilot_hf[idx]
        y = pilot_lf[idx]
        cov_xy = _safe_cov(x, y, ddof=1)
        var_y = _safe_var(y)
        beta = _cv_beta(cov_xy, var_y)
        betas.append(beta)

        if base_sign == 0 or not np.isfinite(beta) or abs(beta) < EPS:
            continue
        valid_sign_trials += 1
        sign = 1 if beta > 0 else -1
        if sign != base_sign:
            flips += 1

    finite_betas = np.asarray([b for b in betas if np.isfinite(b)], dtype=float)
    if finite_betas.size >= 2 and abs(np.mean(finite_betas)) > EPS:
        beta_cv = float(np.std(finite_betas, ddof=1) / abs(np.mean(finite_betas)))
    else:
        beta_cv = float("nan")

    if valid_sign_trials > 0:
        sign_flip_prob = float(flips / valid_sign_trials)
    else:
        sign_flip_prob = float("nan")

    return {"beta_sign_flip_prob": sign_flip_prob, "beta_cv": beta_cv}


def statistical_flags(metrics: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if np.isfinite(metrics.get("hf_variance", np.nan)) and metrics["hf_variance"] < 1e-16:
        flags.append("near_zero_hf_variance")
    if np.isfinite(metrics.get("lf_variance", np.nan)) and metrics["lf_variance"] < 1e-16:
        flags.append("near_zero_lf_variance")
    if not np.isfinite(metrics.get("pearson_correlation", np.nan)):
        flags.append("ill_defined_correlation")
    if not np.isfinite(metrics.get("control_variate_beta", np.nan)):
        flags.append("unstable_control_variate")
    if np.isfinite(metrics.get("residual_variance", np.nan)) and metrics["residual_variance"] < 0:
        flags.append("negative_residual_variance")
    return flags
