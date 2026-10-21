#!/usr/bin/env python3
"""Audit the paired PICLas/ADBSat CLL pilot without modifying raw results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    values = np.empty(draws)
    for i in range(draws):
        indices = rng.integers(0, n, n)
        values[i] = np.corrcoef(x[indices], y[indices])[0, 1]
    quantiles = np.quantile(values[np.isfinite(values)], [0.025, 0.1, 0.5, 0.9, 0.975])
    return dict(zip(("q025", "q10", "q50", "q90", "q975"), map(float, quantiles)))


def bootstrap_mean_difference(x: np.ndarray, y: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    difference = x - y
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    values = difference[indices].mean(axis=1)
    quantiles = np.quantile(values, [0.025, 0.5, 0.975])
    return dict(zip(("q025", "q50", "q975"), map(float, quantiles)))


def analyze_qoi(rows: list[dict[str, str]], qoi: str, seed: int, draws: int) -> dict[str, object]:
    selected = [row for row in rows if row["qoi"] == qoi]
    by_model: dict[str, dict[str, dict[str, str]]] = {}
    for row in selected:
        by_model.setdefault(row["model_id"], {})[row["sample_id"]] = row
    hf_name = next(name for name in by_model if name.startswith("PICLas"))
    lf_name = next(name for name in by_model if name.startswith("ADBSat"))
    sample_ids = sorted(set(by_model[hf_name]) & set(by_model[lf_name]))
    if len(sample_ids) != len(by_model[hf_name]) or len(sample_ids) != len(by_model[lf_name]):
        raise ValueError(f"{qoi}: PICLas and ADBSat samples are not exactly paired")
    hf = np.array([float(by_model[hf_name][sid]["value"]) for sid in sample_ids])
    lf = np.array([float(by_model[lf_name][sid]["value"]) for sid in sample_ids])
    hf_cost = np.array([float(by_model[hf_name][sid]["cost"]) for sid in sample_ids])
    lf_cost = np.array([float(by_model[lf_name][sid]["cost"]) for sid in sample_ids])
    rho = float(np.corrcoef(hf, lf)[0, 1])
    beta = float(np.cov(hf, lf, ddof=1)[0, 1] / np.var(lf, ddof=1))
    difference = hf - lf
    regression_intercept = float(hf.mean() - beta * lf.mean())
    regression_residual = hf - (regression_intercept + beta * lf)
    cost_ratio = float(hf_cost.mean() / lf_cost.mean())
    allocation_ratio = math.sqrt(max(0.0, rho * rho / (1.0 - rho * rho) * cost_ratio))
    # Equal-cost MFMC allocation against an all-HF estimator with n baseline samples.
    budget = len(hf) * float(hf_cost.mean())
    n_hf = budget / (float(hf_cost.mean()) + allocation_ratio * float(lf_cost.mean()))
    n_lf = allocation_ratio * n_hf
    relative_variance = (1.0 - rho * rho) * len(hf) / n_hf + rho * rho * len(hf) / n_lf
    return {
        "n_pairs": len(sample_ids),
        "hf_model": hf_name,
        "lf_model": lf_name,
        "hf_mean": float(hf.mean()),
        "hf_sd": float(hf.std(ddof=1)),
        "hf_min": float(hf.min()),
        "hf_max": float(hf.max()),
        "lf_mean": float(lf.mean()),
        "lf_sd": float(lf.std(ddof=1)),
        "lf_min": float(lf.min()),
        "lf_max": float(lf.max()),
        "pearson_rho": rho,
        "rho_bootstrap": bootstrap_correlation(hf, lf, seed, draws),
        "paired_difference_hf_minus_lf_mean": float(difference.mean()),
        "paired_difference_hf_minus_lf_sd": float(difference.std(ddof=1)),
        "paired_difference_mean_bootstrap": bootstrap_mean_difference(hf, lf, seed + 1000, draws),
        "paired_mae": float(np.abs(difference).mean()),
        "paired_rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_mean_bias_percent": float(100.0 * difference.mean() / hf.mean()),
        "control_variate_beta": beta,
        "regression_intercept": regression_intercept,
        "regression_residual_sd": float(regression_residual.std(ddof=2)),
        "residual_variance_fraction": 1.0 - rho * rho,
        "hf_mean_cost_cpu_h": float(hf_cost.mean()),
        "lf_mean_cost_cpu_h": float(lf_cost.mean()),
        "hf_to_lf_cost_ratio": cost_ratio,
        "optimal_lf_per_hf": allocation_ratio,
        "equal_cost_n_hf_continuous": n_hf,
        "equal_cost_n_lf_continuous": n_lf,
        "equal_cost_relative_variance": relative_variance,
        "equal_cost_relative_standard_error": math.sqrt(relative_variance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()

    raw = args.results_dir / "model_evaluations.csv"
    rows = read_csv(raw)
    if not rows:
        raise ValueError("model_evaluations.csv is empty")
    keys = [(r["sample_id"], r["qoi"], r["model_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample/qoi/model rows found")

    report = {
        "analysis_seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "raw_file": raw.name,
        "raw_sha256": sha256(raw),
        "n_rows": len(rows),
        "qoi": {
            qoi: analyze_qoi(rows, qoi, args.seed + i, args.bootstrap_draws)
            for i, qoi in enumerate(sorted({r["qoi"] for r in rows}))
        },
        "interpretation_limits": [
            "The baseline-only pilot cannot estimate paired differences between geometries.",
            "No repeated PICLas seeds at fixed CLL inputs are available to separate particle noise from GSI response variation.",
            "The config snapshot records +x flow, but a generated parameter.ini or equivalent run artifact is required to prove solver-side application.",
            "MFMC efficiency estimates assume the pilot correlation transfers to the geometries used in the search.",
        ],
    }
    json_path = args.results_dir / "phase3_pilot_audit.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    cd = report["qoi"]["C_D"]
    ci = cd["rho_bootstrap"]
    difference_ci = cd["paired_difference_mean_bootstrap"]
    scale_ratio = max(cd["hf_mean"], cd["lf_mean"]) / max(
        min(cd["hf_mean"], cd["lf_mean"]), 1.0e-300
    )
    markdown = f"""# Phase-3 audit of the CLL baseline pilot

Raw input: `{raw.name}`  
SHA-256: `{report['raw_sha256']}`  
Pairs: {cd['n_pairs']}  
Bootstrap: {args.bootstrap_draws} paired resamples, seed {args.seed}

## Primary diagnostic (`C_D`)

| Quantity | Result |
|---|---:|
| PICLas mean ± SD | {cd['hf_mean']:.6g} ± {cd['hf_sd']:.6g} |
| ADBSat mean ± SD | {cd['lf_mean']:.6g} ± {cd['lf_sd']:.6g} |
| Pearson correlation | {cd['pearson_rho']:.4f} |
| Paired-bootstrap 95% interval for correlation | [{ci['q025']:.4f}, {ci['q975']:.4f}] |
| Mean paired difference, PICLas − ADBSat | {cd['paired_difference_hf_minus_lf_mean']:.6g} |
| Paired-difference SD | {cd['paired_difference_hf_minus_lf_sd']:.6g} |
| Bootstrap 95% interval for mean difference | [{difference_ci['q025']:.6g}, {difference_ci['q975']:.6g}] |
| Paired MAE / RMSE | {cd['paired_mae']:.6g} / {cd['paired_rmse']:.6g} |
| Relative mean bias | {cd['relative_mean_bias_percent']:.3f}% |
| HF-on-LF regression intercept / slope | {cd['regression_intercept']:.6g} / {cd['control_variate_beta']:.6g} |
| Regression residual SD | {cd['regression_residual_sd']:.6g} |
| Mean PICLas cost | {cd['hf_mean_cost_cpu_h']:.6g} CPU-h |
| Mean ADBSat cost | {cd['lf_mean_cost_cpu_h']:.6g} CPU-h |
| PICLas/ADBSat cost ratio | {cd['hf_to_lf_cost_ratio']:.1f} |
| Continuous optimum LF evaluations per HF evaluation | {cd['optimal_lf_per_hf']:.1f} |
| Equal-cost MFMC variance / plain-MC variance | {cd['equal_cost_relative_variance']:.4f} |
| Equal-cost MFMC SE / plain-MC SE | {cd['equal_cost_relative_standard_error']:.4f} |

The corrected baseline geometry exhibits exceptionally high paired correlation at the sampled CLL points. ADBSat is also very cheap and its mean response closely matches PICLas (mean ratio {scale_ratio:.3g}). This is strong evidence for retaining it as a provisional control variate on this geometry, but it is not yet evidence that the correlation transfers to perturbed geometries. The continuous MFMC allocation is an asymptotic cost-model result rather than a production recommendation; it must be bounded by the finite two-dimensional input domain, implementation overhead, and geometry-transfer tests.

## Gate G3 decision

**Not passed.** This pilot supports retaining ADBSat-CLL as a provisional control variate, but it does not determine the optimization or holdout sample sizes. Before G3 can close:

1. archive solver-side evidence that the PICLas jobs used axial `+x` flow and the intended mesh/reference area;
2. run repeated PICLas seeds at fixed CLL points to estimate numerical/particle noise;
3. run the same paired GSI points on the baseline and at least one perturbed geometry to estimate the variance of geometry differences and correlation transfer;
4. use those estimates to set `N_search`, holdout size, DSMC scope, and the MFMC activation threshold;
5. repair or replace `pilot_robustness.csv` uncertainty fields: its near-zero correlation SD is not a sampling-uncertainty estimate, and its gain columns are `NaN`.
"""
    (args.results_dir / "phase3_pilot_audit.md").write_text(markdown, encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
