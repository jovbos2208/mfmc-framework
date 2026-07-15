from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _group_mean(rows: List[Dict[str, Any]], key: str, value: str) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        k = str(row.get(key, ""))
        v = _to_float(row.get(value))
        if np.isfinite(v):
            acc[k].append(v)
    return {k: float(np.mean(vs)) for k, vs in acc.items() if vs}


def _save_bar(data: Dict[str, float], title: str, ylabel: str, path: str) -> None:
    if not data:
        return
    labels = list(data.keys())
    vals = [data[k] for k in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _save_scatter(rows: List[Dict[str, Any]], x_key: str, y_key: str, title: str, path: str) -> None:
    xs, ys = [], []
    for row in rows:
        x = _to_float(row.get(x_key))
        y = _to_float(row.get(y_key))
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, s=12, alpha=0.7)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _save_boxplot(rows: List[Dict[str, Any]], group_key: str, value_key: str, title: str, path: str) -> None:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        g = str(row.get(group_key, ""))
        v = _to_float(row.get(value_key))
        if g and np.isfinite(v):
            groups[g].append(v)
    if not groups:
        return

    labels = sorted(groups)
    data = [groups[k] for k in labels]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, vert=True)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(value_key)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _save_qoi_dependence(rows: List[Dict[str, Any]], output_dir: str) -> None:
    by_qoi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qoi[str(row.get("qoi", "unknown"))].append(row)

    for qoi, qrows in by_qoi.items():
        budgets, gains = [], []
        for r in qrows:
            b = _to_float(r.get("budget"))
            g = _to_float(r.get("cost_normalized_gain"))
            if np.isfinite(b) and np.isfinite(g):
                budgets.append(b)
                gains.append(g)
        if not budgets:
            continue

        plt.figure(figsize=(8, 5))
        plt.scatter(budgets, gains, s=10)
        plt.xlabel("budget")
        plt.ylabel("cost_normalized_gain")
        plt.title(f"QoI Dependence: {qoi}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"qoi_dependence_{qoi}.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=220)
        plt.close()


def _safe_file_token(value: str) -> str:
    token = str(value).strip().replace("/", "_")
    return token.replace(" ", "_")


def _save_source_regime_heatmaps(
    rows: List[Dict[str, Any]],
    output_dir: str,
    metric_key: str,
    prefix: str,
    title_metric: str,
) -> None:
    direct_rows = [r for r in rows if str(r.get("quantity_kind", "")) == "direct"]
    if not direct_rows:
        return

    qois = sorted({str(r.get("qoi", "")) for r in direct_rows if str(r.get("qoi", ""))})
    geoms = sorted({str(r.get("geometry_id", "")) for r in direct_rows if str(r.get("geometry_id", ""))})

    for qoi in qois:
        for geom in geoms:
            subset = [r for r in direct_rows if str(r.get("qoi", "")) == qoi and str(r.get("geometry_id", "")) == geom]
            if not subset:
                continue
            sources = sorted({str(r.get("active_sources", "")) for r in subset if str(r.get("active_sources", ""))})
            regimes = sorted({str(r.get("regime_id", "")) for r in subset if str(r.get("regime_id", ""))})
            if not sources or not regimes:
                continue

            matrix = np.full((len(regimes), len(sources)), np.nan, dtype=float)
            for i, reg in enumerate(regimes):
                for j, src in enumerate(sources):
                    vals = []
                    for row in subset:
                        if str(row.get("regime_id", "")) != reg or str(row.get("active_sources", "")) != src:
                            continue
                        val = _to_float(row.get(metric_key))
                        if np.isfinite(val):
                            vals.append(val)
                    if vals:
                        matrix[i, j] = float(np.mean(vals))

            if not np.isfinite(matrix).any():
                continue

            plt.figure(figsize=(max(8, 1.2 * len(sources)), max(4.5, 0.9 * len(regimes))))
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                continue
            if metric_key in {"rho_hat", "pearson_correlation"}:
                vmin = min(vmin, -1.0)
                vmax = max(vmax, 1.0)
            cmap = "coolwarm" if "corr" in metric_key or metric_key == "rho_hat" else "viridis"
            plt.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xticks(range(len(sources)), sources, rotation=45, ha="right")
            plt.yticks(range(len(regimes)), regimes)
            plt.colorbar(label=title_metric)
            plt.title(f"{title_metric}: source x regime ({qoi}, {geom})")
            plt.tight_layout()
            filename = f"{prefix}_{_safe_file_token(qoi)}_{_safe_file_token(geom)}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=220)
            plt.close()


def _save_budget_gain_quantile_curves(rows: List[Dict[str, Any]], output_dir: str) -> None:
    direct_rows = [r for r in rows if str(r.get("quantity_kind", "")) == "direct"]
    if not direct_rows:
        return
    by_qoi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in direct_rows:
        by_qoi[str(row.get("qoi", "unknown"))].append(row)

    for qoi, qrows in by_qoi.items():
        by_budget: Dict[float, List[float]] = defaultdict(list)
        by_budget_fail: Dict[float, List[float]] = defaultdict(list)
        for row in qrows:
            b = _to_float(row.get("budget"))
            g = _to_float(row.get("cost_normalized_gain"))
            f = _to_float(row.get("underperform_hf"))
            if np.isfinite(b) and np.isfinite(g):
                by_budget[b].append(g)
            if np.isfinite(b) and np.isfinite(f):
                by_budget_fail[b].append(f)
        if not by_budget:
            continue

        budgets = sorted(by_budget)
        p05 = [float(np.nanpercentile(np.asarray(by_budget[b], dtype=float), 5.0)) for b in budgets]
        p50 = [float(np.nanpercentile(np.asarray(by_budget[b], dtype=float), 50.0)) for b in budgets]
        p95 = [float(np.nanpercentile(np.asarray(by_budget[b], dtype=float), 95.0)) for b in budgets]
        fail_rate = [float(np.mean(np.asarray(by_budget_fail.get(b, [float("nan")]), dtype=float))) for b in budgets]

        plt.figure(figsize=(9, 5))
        x = np.asarray(budgets, dtype=float)
        plt.plot(x, p50, marker="o", label="gain p50")
        plt.fill_between(x, p05, p95, alpha=0.25, label="gain p05-p95")
        plt.xlabel("budget")
        plt.ylabel("HF-error / MFMC-error")
        plt.title(f"Budget-Gain Quantiles ({qoi})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"budget_gain_quantiles_{_safe_file_token(qoi)}.png"), dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(x, fail_rate, marker="o", color="tab:red")
        plt.xlabel("budget")
        plt.ylabel("MFMC fail rate")
        plt.title(f"Budget Failure Rate ({qoi})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"budget_fail_rate_{_safe_file_token(qoi)}.png"), dpi=220)
        plt.close()


def _parse_pair_sources(active_sources: str) -> Optional[Tuple[str, str]]:
    parts = [p for p in str(active_sources).split("+") if p]
    if len(parts) != 2:
        return None
    a, b = sorted(parts)
    return a, b


def _save_interaction_heatmaps(rows: List[Dict[str, Any]], output_dir: str) -> None:
    pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    singles_corr: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        qoi = str(row.get("qoi", ""))
        if qoi != "C_D":
            continue

        act = str(row.get("active_sources", ""))
        pair = _parse_pair_sources(act)
        if pair is not None:
            pairs[pair].append(row)
        elif act and "+" not in act:
            corr = _to_float(row.get("pearson_correlation"))
            if np.isfinite(corr):
                singles_corr[act].append(corr)

    if not pairs:
        return

    sources = sorted({s for pair in pairs for s in pair})
    idx = {s: i for i, s in enumerate(sources)}
    gain_mat = np.full((len(sources), len(sources)), np.nan)
    deg_mat = np.full((len(sources), len(sources)), np.nan)

    singles_mean = {k: float(np.mean(v)) for k, v in singles_corr.items() if v}

    for (a, b), rset in pairs.items():
        gains = [_to_float(r.get("cost_normalized_gain")) for r in rset]
        gains = [v for v in gains if np.isfinite(v)]
        corrs = [_to_float(r.get("pearson_correlation")) for r in rset]
        corrs = [v for v in corrs if np.isfinite(v)]
        if not gains and not corrs:
            continue

        i, j = idx[a], idx[b]
        if gains:
            gv = float(np.mean(gains))
            gain_mat[i, j] = gain_mat[j, i] = gv
        if corrs:
            pair_corr = float(np.mean(corrs))
            ref = np.nan
            if a in singles_mean and b in singles_mean:
                ref = 0.5 * (singles_mean[a] + singles_mean[b])
            if np.isfinite(ref):
                dv = pair_corr - ref
                deg_mat[i, j] = deg_mat[j, i] = dv

    # Gain heatmap
    if np.isfinite(gain_mat).any():
        plt.figure(figsize=(8, 7))
        plt.imshow(gain_mat, cmap="viridis", vmin=np.nanmin(gain_mat), vmax=np.nanmax(gain_mat))
        plt.xticks(range(len(sources)), sources, rotation=45, ha="right")
        plt.yticks(range(len(sources)), sources)
        plt.colorbar(label="mean gain")
        plt.title("Interaction Effects: Pairwise MFMC Gain")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "interaction_effects_gain_heatmap.png"), dpi=220)
        plt.close()

    # Correlation degradation heatmap
    if np.isfinite(deg_mat).any():
        plt.figure(figsize=(8, 7))
        plt.imshow(deg_mat, cmap="coolwarm")
        plt.xticks(range(len(sources)), sources, rotation=45, ha="right")
        plt.yticks(range(len(sources)), sources)
        plt.colorbar(label="pair corr - isolated corr")
        plt.title("Interaction Effects: Correlation Degradation")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "interaction_effects_corr_degradation_heatmap.png"), dpi=220)
        plt.close()


def _save_pilot_robustness(rows: List[Dict[str, Any]], output_dir: str) -> None:
    if not rows:
        return

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('lf_model_id')}::{row.get('qoi')}"
        by_model[key].append(row)

    for key, rset in by_model.items():
        ordered = sorted(rset, key=lambda rr: _to_float(rr.get("pilot_size")))
        xs = [_to_float(r.get("pilot_size")) for r in ordered]
        xs = [x for x in xs if np.isfinite(x)]
        if not xs:
            continue

        beta_mean = [_to_float(r.get("beta_mean")) for r in ordered]
        beta_std = [_to_float(r.get("beta_std")) for r in ordered]
        underperf = [_to_float(r.get("underperform_frequency")) for r in ordered]
        gain_p10 = [_to_float(r.get("gain_p10")) for r in ordered]
        gain_p50 = [_to_float(r.get("gain_p50")) for r in ordered]
        gain_p90 = [_to_float(r.get("gain_p90")) for r in ordered]

        # Weight vs pilot size
        plt.figure(figsize=(9, 5))
        plt.plot(xs, beta_mean, marker="o", label="beta_mean")
        plt.plot(xs, beta_std, marker="s", label="beta_std")
        plt.xlabel("pilot_size")
        plt.ylabel("control-variate weight")
        plt.title(f"Pilot Robustness: Weight vs Pilot Size ({key})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pilot_robustness_weight_{key.replace('::', '_')}.png"), dpi=220)
        plt.close()

        # Gain distribution vs pilot size
        plt.figure(figsize=(9, 5))
        plt.plot(xs, gain_p10, marker="^", label="gain_p10")
        plt.plot(xs, gain_p50, marker="o", label="gain_p50")
        plt.plot(xs, gain_p90, marker="v", label="gain_p90")
        plt.xlabel("pilot_size")
        plt.ylabel("HF-error / MFMC-error")
        plt.title(f"Pilot Robustness: Gain Distribution ({key})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pilot_robustness_gain_distribution_{key.replace('::', '_')}.png"), dpi=220)
        plt.close()

        # Underperformance probability
        plt.figure(figsize=(9, 5))
        plt.plot(xs, underperf, marker="o", color="tab:red")
        plt.xlabel("pilot_size")
        plt.ylabel("underperformance probability")
        plt.title(f"Pilot Robustness: Underperformance Probability ({key})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pilot_robustness_underperformance_{key.replace('::', '_')}.png"), dpi=220)
        plt.close()


def generate_plots(output_dir: str) -> List[str]:
    results_csv = os.path.join(output_dir, "results_long.csv")
    robustness_csv = os.path.join(output_dir, "pilot_robustness.csv")

    rows = _read_rows(results_csv)
    robust_rows = _read_rows(robustness_csv)
    if not rows:
        return []

    saved: List[str] = []

    # Source ranking
    source_corr = _group_mean(rows, "active_sources", "pearson_correlation")
    p = os.path.join(output_dir, "source_ranking_correlation.png")
    _save_bar(source_corr, "Source Ranking: Correlation", "mean Pearson", p)

    source_gain = _group_mean(rows, "active_sources", "cost_normalized_gain")
    p2 = os.path.join(output_dir, "source_ranking_gain.png")
    _save_bar(source_gain, "Source Ranking: Gain", "mean gain", p2)

    source_resid = _group_mean(rows, "active_sources", "residual_variance")
    p3 = os.path.join(output_dir, "source_ranking_residual_variance.png")
    _save_bar(source_resid, "Source Ranking: Residual Variance", "mean residual variance", p3)

    # Regime dependence
    p4 = os.path.join(output_dir, "regime_dependence_altitude_vs_gain.png")
    _save_scatter(rows, "regime_altitude_km", "cost_normalized_gain", "Regime Dependence: Gain vs Altitude", p4)

    p5 = os.path.join(output_dir, "regime_dependence_knudsen_vs_correlation.png")
    _save_scatter(rows, "regime_knudsen_number", "pearson_correlation", "Regime Dependence: Correlation vs Knudsen", p5)

    p6 = os.path.join(output_dir, "regime_dependence_solar_vs_gain.png")
    _save_boxplot(rows, "regime_solar_activity_state", "cost_normalized_gain", "Regime Dependence: Gain vs Solar Activity", p6)

    # Interaction effects
    _save_interaction_heatmaps(rows, output_dir)

    # QoI dependence
    _save_qoi_dependence(rows, output_dir)

    # Matrix heatmaps and budget quantile curves
    _save_source_regime_heatmaps(
        rows,
        output_dir,
        metric_key="rho_hat",
        prefix="correlation_heatmap",
        title_metric="rho_hat",
    )
    _save_source_regime_heatmaps(
        rows,
        output_dir,
        metric_key="residual_ratio_hat",
        prefix="residual_ratio_heatmap",
        title_metric="residual_ratio_hat",
    )
    _save_budget_gain_quantile_curves(rows, output_dir)

    # Geometry dependence
    p7 = os.path.join(output_dir, "geometry_dependence_gain_by_class.png")
    _save_boxplot(rows, "geometry_class", "cost_normalized_gain", "Geometry Dependence: Gain by Geometry Class", p7)

    # Pilot robustness
    _save_pilot_robustness(robust_rows, output_dir)

    for fn in os.listdir(output_dir):
        if fn.endswith(".png"):
            saved.append(os.path.join(output_dir, fn))

    return sorted(set(saved))
