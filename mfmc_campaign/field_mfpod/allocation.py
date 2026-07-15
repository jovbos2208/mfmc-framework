from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import evaluate_subspace
from .models import MFPODError
from .operator import compute_mfpod


def allocate_counts(budget: float, hf_cost: float, lf_cost: float, *, mode: str = "fixed_budget_fraction", hf_budget_fraction: float = 0.5, m_H: int | None = None, m_L: int | None = None) -> dict:
    if mode == "explicit_counts":
        if m_H is None or m_L is None: raise MFPODError("explicit_counts requires m_H and m_L")
        cost = hf_cost * m_H + lf_cost * m_L
        if not (1 <= m_H < m_L) or cost > budget + 1e-12: raise MFPODError("Explicit MFPOD counts are invalid or exceed budget")
        return {"m_H": int(m_H), "m_L": int(m_L), "cost": float(cost)}
    fraction = 0.5 if mode == "equal_model_budget" else hf_budget_fraction
    if mode not in {"fixed_budget_fraction", "equal_model_budget"}: raise MFPODError(f"Unsupported allocation mode {mode!r}")
    mh = max(1, int(np.floor(fraction * budget / hf_cost)))
    ml = int(np.floor((budget - hf_cost * mh) / lf_cost))
    while mh >= ml and mh > 1:
        mh -= 1; ml = int(np.floor((budget - hf_cost * mh) / lf_cost))
    if mh >= ml: raise MFPODError("Budget cannot support nested m_H < m_L")
    return {"m_H": mh, "m_L": ml, "cost": float(hf_cost * mh + lf_cost * ml), "hf_budget_fraction_realized": float(hf_cost * mh / budget)}


def select_empirical_allocation(hf_pilot: np.ndarray, lf_pilot: np.ndarray, *, budget: float, hf_cost: float, lf_cost: float, candidate_fractions: Iterable[float], alpha: float, target_r: int, validation_fraction: float = 0.4, repeats: int = 20, random_seed: int = 1101) -> dict:
    h, l = np.asarray(hf_pilot), np.asarray(lf_pilot)
    if h.shape != l.shape: raise MFPODError("Allocation pilot requires paired HF/LF snapshots")
    rng = np.random.default_rng(random_seed); rows=[]
    for fraction in candidate_fractions:
        allocation = allocate_counts(budget, hf_cost, lf_cost, hf_budget_fraction=float(fraction))
        for repeat in range(repeats):
            order = rng.permutation(h.shape[0]); n_val = max(1, int(round(validation_fraction * h.shape[0])))
            val, train = order[:n_val], order[n_val:]
            mh = min(allocation["m_H"], max(1, len(train) - 1)); ml = min(allocation["m_L"], len(train))
            if ml <= mh: continue
            hp, lp, le = h[train[:mh]], l[train[:mh]], l[train[mh:ml]]
            result = compute_mfpod(hp, lp, le, alpha, n_modes=min(target_r, 2 * mh + ml - mh))
            metric = evaluate_subspace(result.modes[:, :target_r], h[val])["projection_error"]
            rows.append({"fraction": float(fraction), "repeat": repeat, "m_H": mh, "m_L": ml, "heldout_hf_projection_error": metric})
    summaries=[]
    for fraction in candidate_fractions:
        vals = [r["heldout_hf_projection_error"] for r in rows if r["fraction"] == float(fraction)]
        if vals: summaries.append({"fraction": float(fraction), "median_metric": float(np.median(vals)), "mean_metric": float(np.mean(vals))})
    if not summaries: raise MFPODError("No feasible allocation candidates")
    selected = min(summaries, key=lambda x: (x["median_metric"], -x["fraction"]))
    return {"description": "pilot-selected empirical allocation", "metric": "heldout_hf_projection_error", "candidate_results": rows, "candidate_summaries": summaries, "selected": selected, "tie_breaking_rule": "lowest median metric, then largest HF fraction", "random_seed": random_seed}
