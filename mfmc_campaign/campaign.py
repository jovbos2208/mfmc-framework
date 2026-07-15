from __future__ import annotations

import hashlib
import csv
import json
import os
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .adapters import build_adapter_registry, make_request
from .config import load_and_validate, normalize_config, validate_or_raise
from .estimator import (
    beta_stability_metrics,
    compute_mfmc_diagnostics,
    compute_multi_lf_mfmc_diagnostics,
    compute_paper_mfmc_diagnostics,
    derive_quantities,
    pilot_robustness_metrics,
    statistical_flags,
)
from .experiments import generate_experiment_cells
from .output import EvaluationCache, ResultStore, export_predictive_dataset, write_summary_tables
from .plotting import generate_plots
from .qoi_registry import build_qoi_registry
from .reproducibility import get_run_fingerprint
from .sampling import InputModel, SamplingContext
from .types import EvaluationResult


def _hash_samples(
    model_id: str,
    phase: str,
    qoi: str,
    sample_ids: List[str],
    samples: List[Dict[str, Any]],
    seed: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "model_id": model_id,
        "phase": phase,
        "qoi": qoi,
        "sample_ids": sample_ids,
        "samples": samples,
        "seed": seed,
        "metadata": metadata or {},
    }
    txt = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % (2**31 - 1)


def _find_geometry(config: Dict[str, Any], geometry_id: str) -> Dict[str, Any]:
    for g in config.get("geometries", []):
        gid = str(g.get("id", g.get("name", "geometry")))
        if gid == geometry_id:
            return g
    raise KeyError(f"Unknown geometry id '{geometry_id}'")


def _find_regime(config: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
    for r in config.get("regimes", []):
        rid = str(r.get("id", r.get("label", "regime")))
        if rid == regime_id:
            return r
    raise KeyError(f"Unknown regime id '{regime_id}'")


def _to_eval_result(payload: Dict[str, Any]):
    return EvaluationResult(
        values_by_qoi={k: list(v) for k, v in payload["values_by_qoi"].items()},
        costs=list(payload["costs"]),
        sample_ids=list(payload["sample_ids"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _cache_key_for_request(request, qoi: str, phase: str) -> str:
    return _hash_samples(
        model_id=request.model_id,
        phase=phase,
        qoi=qoi,
        sample_ids=request.sample_ids,
        samples=request.samples,
        seed=request.seed,
        metadata=dict(getattr(request, "metadata", {}) or {}),
    )


def _cache_payload_from_result(result) -> Dict[str, Any]:
    return {
        "values_by_qoi": result.values_by_qoi,
        "costs": result.costs,
        "sample_ids": result.sample_ids,
        "metadata": result.metadata,
    }


def _evaluate_with_cache(cache: EvaluationCache, adapter, request, qoi: str, phase: str):
    key = _cache_key_for_request(request, qoi, phase)

    cached = cache.get(key)
    if cached is not None:
        print(
            f"[eval] cache hit phase={phase} model={request.model_id} "
            f"qoi={qoi} n={len(request.samples)}",
            flush=True,
        )
        return _to_eval_result(cached)

    print(
        f"[eval] running phase={phase} model={request.model_id} "
        f"qoi={qoi} n={len(request.samples)}",
        flush=True,
    )
    result = adapter.evaluate(request)
    print(
        f"[eval] completed phase={phase} model={request.model_id} "
        f"qoi={qoi} n={len(result.sample_ids)}",
        flush=True,
    )
    cache.set(key, _cache_payload_from_result(result))
    return result


def _evaluate_many_with_cache(cache: EvaluationCache, jobs: List[Tuple[str, Any, Any, str, str]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    pending: List[Tuple[str, Any, Any, str, str, str, Any]] = []

    for name, adapter, request, qoi, phase in jobs:
        key = _cache_key_for_request(request, qoi, phase)
        cached = cache.get(key)
        if cached is not None:
            print(
                f"[eval] cache hit phase={phase} model={request.model_id} "
                f"qoi={qoi} n={len(request.samples)}",
                flush=True,
            )
            results[name] = _to_eval_result(cached)
            continue

        if hasattr(adapter, "submit") and hasattr(adapter, "collect"):
            print(
                f"[eval] submitting phase={phase} model={request.model_id} "
                f"qoi={qoi} n={len(request.samples)}",
                flush=True,
            )
            handle = adapter.submit(request)
            pending.append((name, adapter, request, qoi, phase, key, handle))
        else:
            results[name] = _evaluate_with_cache(cache, adapter, request, qoi, phase)

    for name, adapter, request, qoi, phase, key, handle in pending:
        print(
            f"[eval] collecting phase={phase} model={request.model_id} "
            f"qoi={qoi} n={len(request.samples)}",
            flush=True,
        )
        result = adapter.collect(handle)
        print(
            f"[eval] completed phase={phase} model={request.model_id} "
            f"qoi={qoi} n={len(result.sample_ids)}",
            flush=True,
        )
        cache.set(key, _cache_payload_from_result(result))
        results[name] = result

    return results


def _sources_match(row_value: str, active_sources: List[str]) -> bool:
    expected = "+".join(active_sources)
    expected_sorted = "+".join(sorted(active_sources))
    return row_value in {expected, expected_sorted}


def _model_set_matches(row_value: str, expected_value: str) -> bool:
    row_parts = [part for part in str(row_value).split("+") if part]
    expected_parts = [part for part in str(expected_value).split("+") if part]
    if not row_parts or not expected_parts:
        return str(row_value) == str(expected_value)
    return set(row_parts) == set(expected_parts)


def _fmt_num(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.6g}"


def _fmt_list(values: List[Any], max_items: int = 4) -> str:
    text = [str(value) for value in values]
    if len(text) <= max_items:
        return ", ".join(text)
    return ", ".join(text[:max_items]) + f", ... ({len(text)} total)"


def _campaign_overview(
    cfg: Dict[str, Any],
    cells: List[Any],
    output_dir: str,
    *,
    should_resume: bool,
    pilots_only: bool,
    max_prod: int,
    hf_fraction: float,
    min_hf: int,
    trajectory_sampling_enabled: bool,
) -> None:
    geometries = sorted({str(cell.geometry_id) for cell in cells})
    regimes = sorted({str(cell.regime_id) for cell in cells})
    qois = sorted({str(cell.qoi) for cell in cells})
    budgets = []
    for cell in cells:
        if cell.budget not in budgets:
            budgets.append(cell.budget)
    lf_models = sorted({str(cell.lf_model_id) for cell in cells})
    pilot_sizes = sorted({int(cell.pilot_size) for cell in cells})
    repetitions = sorted({int(cell.repetition) for cell in cells})
    source_sets = {
        "+".join(sorted(cell.active_source_blocks)) if cell.active_source_blocks else "none"
        for cell in cells
    }
    print("", flush=True)
    print("=== MFMC campaign overview ===", flush=True)
    print(
        f"study={cfg.get('study', {}).get('id')} mode={cfg.get('study', {}).get('mode')} "
        f"backend={cfg.get('execution', {}).get('backend')} resume={should_resume} pilots_only={pilots_only}",
        flush=True,
    )
    print(f"output={output_dir}", flush=True)
    print(
        f"cells={len(cells)} geometries={_fmt_list(geometries)} regimes={_fmt_list(regimes)} "
        f"qois={_fmt_list(qois)}",
        flush=True,
    )
    print(
        f"budgets={_fmt_list([_fmt_num(budget) for budget in budgets], max_items=8)} "
        f"pilot_sizes={_fmt_list(pilot_sizes)} repetitions={_fmt_list(repetitions)}",
        flush=True,
    )
    print(
        f"hf={cfg.get('models', {}).get('hf', {}).get('id')} lf={_fmt_list(lf_models)} "
        f"max_prod={max_prod} min_hf={min_hf} hf_fraction={_fmt_num(hf_fraction)}",
        flush=True,
    )
    print("correlation/allocation: reported per budget after pilot robustness is available", flush=True)
    print(
        f"source_sets={len(source_sets)} trajectory_sampling={trajectory_sampling_enabled}",
        flush=True,
    )
    print("==============================", flush=True)
    print("", flush=True)


def _pilot_robustness_csv_path(cfg: Dict[str, Any]) -> Optional[str]:
    pilot_cfg = cfg.get("pilot", {})
    if not isinstance(pilot_cfg, dict):
        return None
    explicit = pilot_cfg.get("robustness_csv", pilot_cfg.get("source_robustness_csv"))
    if explicit:
        return _resolve_config_relative_path(cfg, str(explicit))
    pilot_dir = pilot_cfg.get("dir", pilot_cfg.get("pilot_dir"))
    if pilot_dir:
        return _resolve_config_relative_path(cfg, os.path.join(str(pilot_dir), "pilot_robustness.csv"))
    return None


def _pilot_model_evaluations_csv_path(cfg: Dict[str, Any]) -> Optional[str]:
    pilot_cfg = cfg.get("pilot", {})
    if not isinstance(pilot_cfg, dict):
        return None
    explicit = pilot_cfg.get("source_model_evaluations_csv", pilot_cfg.get("model_evaluations_csv"))
    if explicit:
        return _resolve_config_relative_path(cfg, str(explicit))
    pilot_dir = pilot_cfg.get("dir", pilot_cfg.get("pilot_dir"))
    if pilot_dir:
        return _resolve_config_relative_path(cfg, os.path.join(str(pilot_dir), "model_evaluations.csv"))
    return None


def _resolve_config_relative_path(cfg: Dict[str, Any], path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    base_dir = cfg.get("_config_dir")
    if base_dir:
        return os.path.abspath(os.path.join(str(base_dir), path))
    return path


def _load_pilot_robustness_rows(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        print("[pilot] No external pilot robustness CSV configured.", flush=True)
        return []
    if not os.path.exists(path):
        print(f"[pilot] External pilot robustness CSV not found: {path}", flush=True)
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(f"[pilot] Loaded {len(rows)} external pilot robustness rows from: {path}", flush=True)
    print(
        "[pilot] Robustness rows can override correlation/beta/allocation, "
        "but they do not replace pilot model evaluations.",
        flush=True,
    )
    return rows


def _row_float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _row_cost(row: Dict[str, Any]) -> float:
    value = row.get("cost", None)
    if value not in {None, ""}:
        try:
            return float(value)
        except Exception:
            pass
    try:
        return float(list(row.values())[-1])
    except Exception:
        return float("nan")


def _configured_hf_cost(cfg: Dict[str, Any], cell) -> float:
    metadata = cfg.get("budget", {}).get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["measured_mean_hf_cost", "hf_cost", "mean_hf_cost"]:
            value = _row_float(metadata, key)
            if np.isfinite(value) and value > 0:
                return value
    budget = cell.budget
    if np.isfinite(float(budget)) and float(budget) > 0:
        return float(budget)
    return 1.0


def _configured_lf_cost(cfg: Dict[str, Any], lf_model_id: str) -> float:
    metadata = cfg.get("budget", {}).get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["measured_mean_lf_costs", "lf_costs", "mean_lf_costs"]:
            values = metadata.get(key)
            if isinstance(values, dict):
                value = _row_float(values, lf_model_id)
                if np.isfinite(value) and value > 0:
                    return value
        for key in ["measured_mean_lf_cost", "lf_cost", "mean_lf_cost"]:
            value = _row_float(metadata, key)
            if np.isfinite(value) and value > 0:
                return value
    return 1.0


def _safe_mean_array(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _pilot_robustness_row(
    rows: List[Dict[str, Any]],
    *,
    cell,
    lf_model_id: str,
    qoi: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    qoi = qoi or cell.qoi
    for row in rows:
        if str(row.get("qoi", "")).strip() != qoi:
            continue
        row_hf_model_id = str(row.get("hf_model_id", "")).strip()
        if row_hf_model_id and row_hf_model_id != cell.hf_model_id:
            continue
        row_lf_model_id = str(row.get("lf_model_id", "")).strip()
        if row_lf_model_id != lf_model_id and not _model_set_matches(row_lf_model_id, str(cell.lf_model_id)):
            continue
        row_geometry_id = str(row.get("geometry_id", "")).strip()
        if row_geometry_id and row_geometry_id != cell.geometry_id:
            continue
        row_regime_id = str(row.get("regime_id", "")).strip()
        if row_regime_id and row_regime_id != cell.regime_id:
            continue
        row_sources = str(row.get("active_sources", "")).strip()
        if row_sources and not _sources_match(row_sources, list(cell.active_source_blocks)):
            continue
        try:
            if int(float(row.get("pilot_size", "nan"))) != int(cell.pilot_size):
                continue
            if row.get("repetition") not in {"", None} and int(float(row.get("repetition", "nan"))) != int(cell.repetition):
                continue
        except Exception:
            continue
        return row
    return None


def _external_pilot_robustness_by_lf(
    rows: List[Dict[str, Any]],
    *,
    cell,
    lf_model_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    return {
        lf_model_id: row
        for lf_model_id in lf_model_ids
        if (row := _pilot_robustness_row(rows, cell=cell, lf_model_id=lf_model_id)) is not None
    }


def _external_pilot_result(
    *,
    path: Optional[str],
    cell,
    phase: str,
    model_id: str,
    qoi: str,
    qois: Optional[List[str]] = None,
    source_study_id: Optional[str] = None,
    source_mode: Optional[str] = None,
    lf_model_id_filter: Optional[str] = None,
    match_lf_model_id: bool = True,
) -> Optional[EvaluationResult]:
    if not path or not os.path.exists(path):
        return None

    qoi_list = [str(item) for item in (qois or [qoi])]
    if str(qoi) not in qoi_list:
        qoi_list.insert(0, str(qoi))
    qoi_set = set(qoi_list)
    values_by_rep: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: {item: {} for item in qoi_list})
    costs_by_rep: Dict[str, Dict[int, float]] = defaultdict(dict)
    sample_ids_by_rep: Dict[str, Dict[int, str]] = defaultdict(dict)
    active_sources = list(cell.active_source_blocks)
    reject_counts: Dict[str, int] = defaultdict(int)
    rows_seen = 0
    rows_after_identity = 0

    def field(row: Dict[str, Any], key: str) -> str:
        return str(row.get(key, "")).strip()

    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_seen += 1
            if field(row, "phase") != phase:
                reject_counts["phase"] += 1
                continue
            if field(row, "model_id") != model_id:
                reject_counts["model_id"] += 1
                continue
            row_qoi = field(row, "qoi")
            if row_qoi not in qoi_set:
                reject_counts["qoi"] += 1
                continue
            if field(row, "geometry_id") != cell.geometry_id:
                reject_counts["geometry_id"] += 1
                continue
            if field(row, "regime_id") != cell.regime_id:
                reject_counts["regime_id"] += 1
                continue
            if field(row, "hf_model_id") != cell.hf_model_id:
                reject_counts["hf_model_id"] += 1
                continue
            if match_lf_model_id:
                row_lf_model_id = field(row, "lf_model_id")
                expected_lf_model_id = str(lf_model_id_filter or cell.lf_model_id)
                cell_lf_model_id = str(cell.lf_model_id)
                if row_lf_model_id != expected_lf_model_id and not _model_set_matches(row_lf_model_id, cell_lf_model_id):
                    reject_counts["lf_model_id"] += 1
                    continue
            if not _sources_match(field(row, "active_sources"), active_sources):
                reject_counts["active_sources"] += 1
                continue
            if source_study_id is not None and field(row, "study_id") != source_study_id:
                reject_counts["study_id"] += 1
                continue
            if source_mode is not None and field(row, "mode") != source_mode:
                reject_counts["mode"] += 1
                continue
            rows_after_identity += 1
            try:
                if int(float(row.get("pilot_size", "nan"))) != int(cell.pilot_size):
                    reject_counts["pilot_size"] += 1
                    continue
                source_repetition = str(int(float(row.get("repetition", "nan"))))
                sample_index = int(float(row.get("sample_index", "nan")))
                value = float(row.get("value", "nan"))
                cost = _row_cost(row)
            except Exception:
                reject_counts["numeric_parse"] += 1
                continue
            if sample_index < 0 or sample_index >= int(cell.pilot_size):
                reject_counts["sample_index"] += 1
                continue
            if np.isfinite(value):
                values_by_rep[source_repetition].setdefault(row_qoi, {})[sample_index] = value
            if np.isfinite(cost):
                costs_by_rep[source_repetition][sample_index] = cost
            sample_ids_by_rep[source_repetition][sample_index] = str(row.get("sample_id", f"pilot_{sample_index}"))

    expected_indices = range(int(cell.pilot_size))

    def complete_repetition(rep: str) -> bool:
        primary_values = values_by_rep.get(rep, {}).get(str(qoi), {})
        costs = costs_by_rep.get(rep, {})
        return all(idx in primary_values for idx in expected_indices) and all(idx in costs for idx in expected_indices)

    requested_rep = str(int(cell.repetition))
    complete_reps = [rep for rep in values_by_rep if complete_repetition(rep)]
    selected_rep = requested_rep if complete_repetition(requested_rep) else (sorted(complete_reps, key=lambda rep: int(rep))[0] if complete_reps else "")
    if selected_rep and selected_rep != requested_rep:
        print(
            f"[pilot] external model evaluations using source repetition={selected_rep} "
            f"for campaign repetition={cell.repetition} phase={phase} model={model_id} qoi={qoi}",
            flush=True,
        )

    selected_values_by_qoi = values_by_rep.get(selected_rep, {})
    selected_costs = costs_by_rep.get(selected_rep, {})
    selected_sample_ids = sample_ids_by_rep.get(selected_rep, {})
    primary_values = selected_values_by_qoi.get(str(qoi), {})
    missing_values = [idx for idx in expected_indices if idx not in primary_values]
    missing_costs = [idx for idx in expected_indices if idx not in selected_costs]
    if not selected_rep or missing_values or missing_costs:
        top_rejects = ", ".join(
            f"{key}={value}" for key, value in sorted(reject_counts.items(), key=lambda item: item[1], reverse=True)[:6]
        )
        rep_counts = ", ".join(
            f"{rep}:{len(values_by_rep[rep].get(str(qoi), {}))}/{len(costs_by_rep.get(rep, {}))}"
            for rep in sorted(values_by_rep, key=lambda item: int(item))[:8]
        )
        print(
            f"[pilot] external model evaluations no complete match phase={phase} model={model_id} "
            f"qoi={qoi} lf_filter={lf_model_id_filter or ''} rep={cell.repetition} "
            f"rows_seen={rows_seen} rows_after_identity={rows_after_identity} "
            f"values={len(primary_values)}/{int(cell.pilot_size)} costs={len(selected_costs)}/{int(cell.pilot_size)} "
            f"missing_values={missing_values[:5]} missing_costs={missing_costs[:5]} "
            f"source_reps={rep_counts or 'none'} rejects={top_rejects or 'none'}",
            flush=True,
        )
        return None

    ordered_values_by_qoi = {
        item: [qoi_values[idx] for idx in range(int(cell.pilot_size))]
        for item, qoi_values in selected_values_by_qoi.items()
        if all(idx in qoi_values for idx in range(int(cell.pilot_size)))
    }
    ordered_costs = [selected_costs[idx] for idx in range(int(cell.pilot_size))]
    ordered_sample_ids = [selected_sample_ids.get(idx, f"pilot_{idx}") for idx in range(int(cell.pilot_size))]
    return EvaluationResult(
        values_by_qoi=ordered_values_by_qoi,
        costs=ordered_costs,
        sample_ids=ordered_sample_ids,
        metadata={"source_model_evaluations_csv": path, "source_repetition": selected_rep},
    )


def _external_pilot_cost_array(
    *,
    path: Optional[str],
    cell,
    phase: str,
    model_id: str,
    qoi: str,
    source_study_id: Optional[str] = None,
    source_mode: Optional[str] = None,
    lf_model_id_filter: Optional[str] = None,
    match_lf_model_id: bool = True,
) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None

    costs_by_rep: Dict[str, Dict[int, float]] = defaultdict(dict)
    reject_counts: Dict[str, int] = defaultdict(int)
    rows_seen = 0
    rows_after_identity = 0
    active_sources = list(cell.active_source_blocks)

    def field(row: Dict[str, Any], key: str) -> str:
        return str(row.get(key, "")).strip()

    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_seen += 1
            if field(row, "phase") != phase:
                reject_counts["phase"] += 1
                continue
            if field(row, "model_id") != model_id:
                reject_counts["model_id"] += 1
                continue
            if field(row, "qoi") != qoi:
                reject_counts["qoi"] += 1
                continue
            if field(row, "geometry_id") != cell.geometry_id:
                reject_counts["geometry_id"] += 1
                continue
            if field(row, "regime_id") != cell.regime_id:
                reject_counts["regime_id"] += 1
                continue
            if field(row, "hf_model_id") != cell.hf_model_id:
                reject_counts["hf_model_id"] += 1
                continue
            if match_lf_model_id:
                row_lf_model_id = field(row, "lf_model_id")
                expected_lf_model_id = str(lf_model_id_filter or cell.lf_model_id)
                cell_lf_model_id = str(cell.lf_model_id)
                if row_lf_model_id != expected_lf_model_id and not _model_set_matches(row_lf_model_id, cell_lf_model_id):
                    reject_counts["lf_model_id"] += 1
                    continue
            if not _sources_match(field(row, "active_sources"), active_sources):
                reject_counts["active_sources"] += 1
                continue
            if source_study_id is not None and field(row, "study_id") != source_study_id:
                reject_counts["study_id"] += 1
                continue
            if source_mode is not None and field(row, "mode") != source_mode:
                reject_counts["mode"] += 1
                continue
            rows_after_identity += 1
            try:
                if int(float(row.get("pilot_size", "nan"))) != int(cell.pilot_size):
                    reject_counts["pilot_size"] += 1
                    continue
                source_repetition = str(int(float(row.get("repetition", "nan"))))
                sample_index = int(float(row.get("sample_index", "nan")))
                cost = _row_cost(row)
            except Exception:
                reject_counts["numeric_parse"] += 1
                continue
            if sample_index < 0 or sample_index >= int(cell.pilot_size) or not np.isfinite(cost):
                reject_counts["sample_index_or_cost"] += 1
                continue
            costs_by_rep[source_repetition][sample_index] = cost

    expected_indices = range(int(cell.pilot_size))

    def complete_repetition(rep: str) -> bool:
        costs = costs_by_rep.get(rep, {})
        return all(idx in costs for idx in expected_indices)

    requested_rep = str(int(cell.repetition))
    complete_reps = [rep for rep in costs_by_rep if complete_repetition(rep)]
    selected_rep = requested_rep if complete_repetition(requested_rep) else (sorted(complete_reps, key=lambda rep: int(rep))[0] if complete_reps else "")
    if not selected_rep:
        top_rejects = ", ".join(
            f"{key}={value}" for key, value in sorted(reject_counts.items(), key=lambda item: item[1], reverse=True)[:6]
        )
        rep_counts = ", ".join(
            f"{rep}:{len(costs_by_rep.get(rep, {}))}"
            for rep in sorted(costs_by_rep, key=lambda item: int(item))[:8]
        )
        print(
            f"[pilot] external model evaluation costs no complete match phase={phase} model={model_id} "
            f"qoi={qoi} lf_filter={lf_model_id_filter or ''} rep={cell.repetition} "
            f"rows_seen={rows_seen} rows_after_identity={rows_after_identity} "
            f"source_reps={rep_counts or 'none'} rejects={top_rejects or 'none'}",
            flush=True,
        )
        return None
    if selected_rep != requested_rep:
        print(
            f"[pilot] external model evaluation costs using source repetition={selected_rep} "
            f"for campaign repetition={cell.repetition} phase={phase} model={model_id} qoi={qoi}",
            flush=True,
        )
    selected_costs = costs_by_rep[selected_rep]
    return np.asarray([selected_costs[idx] for idx in expected_indices], dtype=float)


def _lf_model_ids_for_cell(cell) -> List[str]:
    return [part for part in str(cell.lf_model_id).split("+") if part]


def _paper_mfmc_enabled(cfg: Dict[str, Any]) -> bool:
    strategy = str(cfg.get("models", {}).get("lf_strategy", "separate")).lower()
    return strategy in {"paper_mfmc", "peherstorfer", "optimal_model_management", "nested_mfmc"}


def _direct_qois_for_config(cfg: Dict[str, Any]) -> List[str]:
    return [str(qoi) for qoi in cfg.get("qois", {}).get("direct", []) if isinstance(qoi, str)]


def _batch_cd_cd2_enabled(cfg: Dict[str, Any]) -> bool:
    direct_qois = set(_direct_qois_for_config(cfg))
    return bool(cfg.get("qois", {}).get("batch_cd_cd2", True)) and {"C_D", "C_D2"}.issubset(direct_qois)


def _skip_batched_qoi_cell(cfg: Dict[str, Any], qoi: str) -> bool:
    return _batch_cd_cd2_enabled(cfg) and str(qoi) == "C_D2"


def _evaluation_qois_for_cell(cfg: Dict[str, Any], qoi: str) -> List[str]:
    if _batch_cd_cd2_enabled(cfg) and str(qoi) == "C_D":
        return ["C_D", "C_D2"]
    return [str(qoi)]


def _single_allocation_all_qois_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("qois", {}).get("single_allocation_all_qois", False))


def _production_sizes(
    cell_budget: float,
    hf_cost: float,
    lf_cost: float,
    hf_fraction: float,
    fallback_hf: int,
    cap: int,
    min_hf: int = 4,
    allocation_ratio: float = float("nan"),
) -> Tuple[int, int]:
    min_hf = max(1, int(min_hf))
    if not np.isfinite(hf_cost) or hf_cost <= 0:
        return fallback_hf, fallback_hf
    if not np.isfinite(lf_cost) or lf_cost <= 0:
        return fallback_hf, fallback_hf

    if np.isfinite(allocation_ratio) and allocation_ratio > 0:
        ratio = max(1.0, float(allocation_ratio))
        n_hf = max(min_hf, int(cell_budget / (hf_cost + ratio * lf_cost)))
        n_hf = max(min_hf, min(n_hf, cap))
        remaining = max(0.0, cell_budget - n_hf * hf_cost)
        affordable_lf = n_hf + int(remaining / lf_cost)
        target_lf = int(np.ceil(n_hf * ratio))
        n_lf = max(n_hf, min(target_lf, affordable_lf, cap))
        return n_hf, n_lf

    hf_budget = max(cell_budget * hf_fraction, hf_cost * min_hf)
    n_hf = max(min_hf, int(hf_budget / hf_cost))

    remaining = max(0.0, cell_budget - n_hf * hf_cost)
    n_lf_extra = int(remaining / lf_cost)
    n_lf = max(n_hf, n_hf + n_lf_extra)

    n_hf = max(min_hf, min(n_hf, cap))
    n_lf = max(n_hf, min(n_lf, cap))
    return n_hf, n_lf


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = min(np.asarray(x).size, np.asarray(y).size)
    if n < 2:
        return float("nan")
    x_arr = np.asarray(x[:n], dtype=float)
    y_arr = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.nanstd(x_arr) < 1e-14 or np.nanstd(y_arr) < 1e-14:
        return float("nan")
    return float(abs(np.corrcoef(x_arr, y_arr)[0, 1]))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = min(np.asarray(x).size, np.asarray(y).size)
    if n < 2:
        return float("nan")
    x_arr = np.asarray(x[:n], dtype=float)
    y_arr = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.nanstd(x_arr) < 1e-14 or np.nanstd(y_arr) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _safe_beta(hf: np.ndarray, lf: np.ndarray) -> float:
    n = min(np.asarray(hf).size, np.asarray(lf).size)
    if n < 2:
        return float("nan")
    h = np.asarray(hf[:n], dtype=float)
    l = np.asarray(lf[:n], dtype=float)
    mask = np.isfinite(h) & np.isfinite(l)
    if int(np.sum(mask)) < 2:
        return float("nan")
    h = h[mask]
    l = l[mask]
    var_l = float(np.var(l, ddof=1)) if l.size > 1 else float("nan")
    if not np.isfinite(var_l) or abs(var_l) < 1e-14:
        return float("nan")
    return float(np.cov(h, l, ddof=1)[0, 1] / var_l)


def _paper_mfmc_allocation(
    *,
    pilot_hf: np.ndarray,
    pilot_lfs: Dict[str, np.ndarray],
    pilot_hf_cost: float,
    pilot_lf_costs: Dict[str, float],
    cell_budget: float,
    min_hf: int,
    cap: int,
) -> Tuple[List[str], Dict[str, int], Dict[str, float], Dict[str, float]]:
    lf_stats: List[Tuple[str, float, float]] = []
    for lf_model_id, values in pilot_lfs.items():
        corr = _safe_abs_corr(pilot_hf, np.asarray(values, dtype=float))
        cost = float(pilot_lf_costs.get(lf_model_id, float("nan")))
        if np.isfinite(corr) and np.isfinite(cost) and cost > 0.0:
            lf_stats.append((lf_model_id, corr, cost))

    if not lf_stats:
        return [], {"__hf__": max(1, int(min_hf))}, {}, {}

    lf_stats.sort(key=lambda item: item[1], reverse=True)
    ordered_lf_ids = [item[0] for item in lf_stats]
    rho2 = [min(0.999999, max(0.0, item[1] * item[1])) for item in lf_stats]
    costs = {item[0]: item[2] for item in lf_stats}
    corrs = {item[0]: item[1] for item in lf_stats}

    denominator = max(1e-12, 1.0 - rho2[0])
    ratios: Dict[str, float] = {}
    previous_ratio = 1.0
    for idx, lf_model_id in enumerate(ordered_lf_ids):
        next_rho2 = rho2[idx + 1] if idx + 1 < len(rho2) else 0.0
        numerator = max(0.0, rho2[idx] - next_rho2)
        raw_ratio = (
            (float(pilot_hf_cost) * numerator) / (float(costs[lf_model_id]) * denominator)
            if np.isfinite(pilot_hf_cost) and pilot_hf_cost > 0.0
            else float("nan")
        )
        ratio = float(np.sqrt(raw_ratio)) if np.isfinite(raw_ratio) and raw_ratio > 0.0 else previous_ratio
        ratio = max(previous_ratio, 1.0, ratio)
        ratios[lf_model_id] = ratio
        previous_ratio = ratio

    weighted_unit_cost = float(pilot_hf_cost) if np.isfinite(pilot_hf_cost) and pilot_hf_cost > 0.0 else 0.0
    weighted_unit_cost += float(sum(costs[lf_model_id] * ratios[lf_model_id] for lf_model_id in ordered_lf_ids))
    if weighted_unit_cost > 0.0 and np.isfinite(weighted_unit_cost):
        n_hf = int(float(cell_budget) / weighted_unit_cost)
    else:
        n_hf = int(min_hf)
    n_hf = max(1, int(min_hf), n_hf)
    n_hf = min(int(cap), n_hf)

    sample_counts: Dict[str, int] = {"__hf__": n_hf}
    spent_budget = (
        float(n_hf) * float(pilot_hf_cost)
        if np.isfinite(pilot_hf_cost) and pilot_hf_cost > 0.0
        else 0.0
    )
    previous_count = n_hf
    for lf_model_id in ordered_lf_ids:
        count = int(np.ceil(float(n_hf) * ratios[lf_model_id]))
        count = max(previous_count, count)
        lf_cost = float(costs[lf_model_id])
        remaining_budget = float(cell_budget) - spent_budget
        affordable_count = int(remaining_budget / lf_cost) if lf_cost > 0.0 else 0
        count = min(int(cap), count, affordable_count)
        if count >= previous_count:
            sample_counts[lf_model_id] = count
            spent_budget += float(count) * lf_cost
            previous_count = count
        else:
            sample_counts[lf_model_id] = 0

    return ordered_lf_ids, sample_counts, ratios, corrs


def _paper_mfmc_pilot_robustness_metrics(
    *,
    pilot_hf: np.ndarray,
    pilot_lfs: Dict[str, np.ndarray],
    pilot_sizes: List[int],
    repetitions: int,
    rng: np.random.Generator,
    hf_cost: float,
    lf_costs: Dict[str, float],
) -> List[Dict[str, Any]]:
    lf_ids = [lf_model_id for lf_model_id, values in pilot_lfs.items() if np.asarray(values).size]
    if not lf_ids:
        return []

    arrays = {lf_model_id: np.asarray(pilot_lfs[lf_model_id], dtype=float) for lf_model_id in lf_ids}
    max_n = min([np.asarray(pilot_hf).size] + [arr.size for arr in arrays.values()])
    if max_n < 4:
        return []

    hf = np.asarray(pilot_hf[:max_n], dtype=float)
    records: List[Dict[str, Any]] = []
    for n in pilot_sizes:
        if n < 4 or n > max_n:
            continue

        by_lf: Dict[str, Dict[str, List[float]]] = {
            lf_model_id: {"corr": [], "beta": [], "ratio": [], "negative": [], "unstable": []}
            for lf_model_id in lf_ids
        }

        for _ in range(repetitions):
            idx = rng.choice(max_n, size=n, replace=False)
            x = hf[idx]
            sub_lfs = {lf_model_id: arrays[lf_model_id][idx] for lf_model_id in lf_ids}
            _, _, ratios, _ = _paper_mfmc_allocation(
                pilot_hf=x,
                pilot_lfs=sub_lfs,
                pilot_hf_cost=hf_cost,
                pilot_lf_costs=lf_costs,
                cell_budget=1.0e12,
                min_hf=1,
                cap=1_000_000_000,
            )

            for lf_model_id in lf_ids:
                y = sub_lfs[lf_model_id]
                corr = _safe_corr(x, y)
                beta = _safe_beta(x, y)
                ratio = float(ratios.get(lf_model_id, float("nan")))
                by_lf[lf_model_id]["corr"].append(corr)
                by_lf[lf_model_id]["beta"].append(beta)
                by_lf[lf_model_id]["ratio"].append(ratio)
                by_lf[lf_model_id]["negative"].append(1.0 if np.isfinite(beta) and beta < 0.0 else 0.0)
                by_lf[lf_model_id]["unstable"].append(1.0 if not np.isfinite(beta) or abs(beta) > 1e4 else 0.0)

        for lf_model_id in lf_ids:
            corr_arr = np.asarray(by_lf[lf_model_id]["corr"], dtype=float)
            beta_arr = np.asarray(by_lf[lf_model_id]["beta"], dtype=float)
            ratio_arr = np.asarray(by_lf[lf_model_id]["ratio"], dtype=float)
            negative_arr = np.asarray(by_lf[lf_model_id]["negative"], dtype=float)
            unstable_arr = np.asarray(by_lf[lf_model_id]["unstable"], dtype=float)
            records.append(
                {
                    "lf_model_id": lf_model_id,
                    "pilot_size": int(n),
                    "correlation_mean": float(np.nanmean(corr_arr)) if np.any(np.isfinite(corr_arr)) else float("nan"),
                    "correlation_std": float(np.nanstd(corr_arr, ddof=1)) if corr_arr.size > 1 else float("nan"),
                    "beta_mean": float(np.nanmean(beta_arr)) if np.any(np.isfinite(beta_arr)) else float("nan"),
                    "beta_std": float(np.nanstd(beta_arr, ddof=1)) if beta_arr.size > 1 else float("nan"),
                    "allocation_ratio_mean": float(np.nanmean(ratio_arr)) if np.any(np.isfinite(ratio_arr)) else float("nan"),
                    "allocation_ratio_std": float(np.nanstd(ratio_arr, ddof=1)) if ratio_arr.size > 1 else float("nan"),
                    "negative_weight_frequency": float(np.nanmean(negative_arr)) if negative_arr.size else float("nan"),
                    "unstable_weight_frequency": float(np.nanmean(unstable_arr)) if unstable_arr.size else float("nan"),
                    "underperform_frequency": float("nan"),
                    "gain_mean": float("nan"),
                    "gain_p10": float("nan"),
                    "gain_p50": float("nan"),
                    "gain_p90": float("nan"),
                }
            )

    return records


def _pilot_robustness_correlation(robust_rows: List[Dict[str, Any]], pilot_size: int) -> float:
    for row in robust_rows:
        try:
            row_size = int(float(row.get("pilot_size", "nan")))
            corr = float(row.get("correlation_mean", "nan"))
        except Exception:
            continue
        if row_size == int(pilot_size) and np.isfinite(corr):
            return corr
    return float("nan")


def _use_pilot_correlation(metrics: Dict[str, Any], robust_rows: List[Dict[str, Any]], pilot_size: int) -> bool:
    corr = _pilot_robustness_correlation(robust_rows, pilot_size)
    if not np.isfinite(corr):
        return False

    metrics["rho_hat"] = float(corr)
    metrics["pearson_correlation"] = float(corr)
    metrics["r2_lin_hat"] = float(max(0.0, min(1.0, corr * corr)))
    return True


def _external_allocation_ratio(rows_by_lf: Dict[str, Dict[str, Any]]) -> float:
    ratios = [
        _row_float(row, "allocation_ratio_mean")
        for row in rows_by_lf.values()
        if np.isfinite(_row_float(row, "allocation_ratio_mean"))
    ]
    return float(max(ratios)) if ratios else float("nan")


def _robustness_allocation_ratio(rows: List[Dict[str, Any]], pilot_size: int) -> float:
    ratios = []
    for row in rows:
        try:
            row_size = int(float(row.get("pilot_size", "nan")))
        except Exception:
            continue
        if row_size != int(pilot_size):
            continue
        ratio = _row_float(row, "allocation_ratio_mean")
        if np.isfinite(ratio) and ratio > 0:
            ratios.append(float(ratio))
    return float(max(ratios)) if ratios else float("nan")


def _multi_lf_pilot_robustness_metrics(
    *,
    pilot_hf: np.ndarray,
    pilot_lfs: Dict[str, np.ndarray],
    pilot_sizes: List[int],
    repetitions: int,
    rng: np.random.Generator,
    hf_cost: float,
    lf_cost: float,
) -> List[Dict[str, Any]]:
    lf_ids = [lf_model_id for lf_model_id, values in pilot_lfs.items() if np.asarray(values).size]
    if not lf_ids:
        return []

    arrays = [np.asarray(pilot_lfs[lf_model_id], dtype=float) for lf_model_id in lf_ids]
    max_n = min([np.asarray(pilot_hf).size] + [arr.size for arr in arrays])
    if max_n < 4:
        return []

    hf = np.asarray(pilot_hf[:max_n], dtype=float)
    lf_matrix = np.column_stack([arr[:max_n] for arr in arrays])
    finite_mask = np.isfinite(hf) & np.all(np.isfinite(lf_matrix), axis=1)
    hf = hf[finite_mask]
    lf_matrix = lf_matrix[finite_mask]
    max_n = int(hf.size)
    if max_n < 4:
        return []

    records: List[Dict[str, Any]] = []
    for n in pilot_sizes:
        if n < 4 or n > max_n:
            continue

        betas: List[float] = []
        cors: List[float] = []
        alloc_ratios: List[float] = []
        negative = 0
        unstable = 0
        underperform = 0

        for _ in range(repetitions):
            idx = rng.choice(max_n, size=n, replace=False)
            x = hf[idx]
            y = lf_matrix[idx, :]

            try:
                sigma_ll = np.asarray(np.cov(y, rowvar=False, ddof=1), dtype=float)
                cov_lh = np.asarray([np.cov(x, y[:, j], ddof=1)[0, 1] for j in range(y.shape[1])], dtype=float)
                beta_vec = np.linalg.pinv(np.atleast_2d(sigma_ll)) @ cov_lh
            except Exception:
                beta_vec = np.full(y.shape[1], np.nan, dtype=float)

            finite_beta = beta_vec[np.isfinite(beta_vec)]
            beta_norm = float(np.linalg.norm(finite_beta)) if finite_beta.size else float("nan")
            betas.append(beta_norm)
            if np.any(finite_beta < 0):
                negative += 1
            if not np.isfinite(beta_norm) or beta_norm > 1e4:
                unstable += 1

            pair_corrs = []
            for j in range(y.shape[1]):
                try:
                    corr = float(np.corrcoef(x, y[:, j])[0, 1])
                except Exception:
                    corr = float("nan")
                if np.isfinite(corr):
                    pair_corrs.append(corr)
            if pair_corrs:
                corr = float(pair_corrs[int(np.argmax(np.abs(pair_corrs)))])
            else:
                corr = float("nan")
            cors.append(corr)

            if np.isfinite(corr) and np.isfinite(hf_cost) and np.isfinite(lf_cost) and hf_cost > 0 and lf_cost > 0:
                corr2 = max(0.0, min(0.999999, corr * corr))
                alloc = ((corr2 / max(1e-12, (1.0 - corr2))) * (hf_cost / lf_cost)) ** 0.5
            else:
                alloc = float("nan")
            alloc_ratios.append(float(alloc))

            if max_n > n + 1 and finite_beta.size == y.shape[1]:
                mask = np.ones(max_n, dtype=bool)
                mask[idx] = False
                x_prod = hf[mask]
                y_prod = lf_matrix[mask, :]
                hf_est = float(np.nanmean(x_prod)) if x_prod.size else float("nan")
                full_mean = float(np.nanmean(hf))
                lf_delta = np.nanmean(y_prod, axis=0) - np.nanmean(lf_matrix, axis=0)
                mfmc_est = float(hf_est - beta_vec @ lf_delta)
                if np.isfinite(hf_est) and np.isfinite(mfmc_est) and abs(mfmc_est - full_mean) > abs(hf_est - full_mean):
                    underperform += 1

        gains = np.asarray([], dtype=float)
        records.append(
            {
                "pilot_size": int(n),
                "correlation_std": float(np.nanstd(np.asarray(cors), ddof=1)) if len(cors) > 1 else float("nan"),
                "correlation_mean": float(np.nanmean(np.asarray(cors))) if np.any(np.isfinite(cors)) else float("nan"),
                "beta_std": float(np.nanstd(np.asarray(betas), ddof=1)) if len(betas) > 1 else float("nan"),
                "beta_mean": float(np.nanmean(np.asarray(betas))) if np.any(np.isfinite(betas)) else float("nan"),
                "allocation_ratio_std": float(np.nanstd(np.asarray(alloc_ratios), ddof=1)) if len(alloc_ratios) > 1 else float("nan"),
                "allocation_ratio_mean": float(np.nanmean(np.asarray(alloc_ratios))) if np.any(np.isfinite(alloc_ratios)) else float("nan"),
                "negative_weight_frequency": float(negative / max(1, repetitions)),
                "unstable_weight_frequency": float(unstable / max(1, repetitions)),
                "underperform_frequency": float(underperform / max(1, repetitions)),
                "gain_mean": float(np.nanmean(gains)) if gains.size else float("nan"),
                "gain_p10": float(np.nanpercentile(gains, 10.0)) if gains.size else float("nan"),
                "gain_p50": float(np.nanpercentile(gains, 50.0)) if gains.size else float("nan"),
                "gain_p90": float(np.nanpercentile(gains, 90.0)) if gains.size else float("nan"),
            }
        )

    return records


def _apply_external_pilot_robustness(
    metrics: Dict[str, Any],
    rows_by_lf: Dict[str, Dict[str, Any]],
    *,
    lf_model_ids: List[str],
    prod_lf_full_by_id: Dict[str, np.ndarray],
    prod_lf_paired_by_id: Dict[str, np.ndarray],
    reference: float,
) -> bool:
    if not rows_by_lf:
        return False

    corrs = np.asarray([_row_float(row, "correlation_mean") for row in rows_by_lf.values()], dtype=float)
    finite_corrs = corrs[np.isfinite(corrs)]
    if finite_corrs.size:
        corr = float(finite_corrs[np.argmax(np.abs(finite_corrs))])
        metrics["rho_hat"] = corr
        metrics["pearson_correlation"] = corr
        metrics["r2_lin_hat"] = float(max(0.0, min(1.0, corr * corr)))

    betas = np.asarray(
        [
            _row_float(rows_by_lf[lf_model_id], "beta_mean")
            for lf_model_id in lf_model_ids
            if lf_model_id in rows_by_lf
        ],
        dtype=float,
    )
    beta_lf_ids = [lf_model_id for lf_model_id in lf_model_ids if lf_model_id in rows_by_lf]
    finite_beta = np.isfinite(betas)
    if not np.any(finite_beta):
        return bool(finite_corrs.size)

    beta_norm = float(np.linalg.norm(betas[finite_beta]))
    metrics["beta_hat"] = beta_norm if betas.size > 1 else float(betas[finite_beta][0])
    metrics["control_variate_beta"] = metrics["beta_hat"]
    metrics["unstable_weight"] = int(beta_norm > 1e4)

    hf_mean = _row_float(metrics, "hf_mean")
    if np.isfinite(hf_mean):
        correction = 0.0
        valid_correction = False
        for lf_model_id, beta in zip(beta_lf_ids, betas):
            if not np.isfinite(beta):
                continue
            lf_full_mean = float(_safe_mean_array(prod_lf_full_by_id.get(lf_model_id, np.asarray([], dtype=float))))
            lf_paired_mean = float(_safe_mean_array(prod_lf_paired_by_id.get(lf_model_id, np.asarray([], dtype=float))))
            if np.isfinite(lf_full_mean) and np.isfinite(lf_paired_mean):
                correction += float(beta) * (lf_paired_mean - lf_full_mean)
                valid_correction = True
        if valid_correction:
            mfmc_estimate = float(hf_mean - correction)
            metrics["mfmc_estimate"] = mfmc_estimate
            if np.isfinite(reference):
                metrics["realized_mfmc_error"] = float(abs(mfmc_estimate - reference))
                hf_only = _row_float(metrics, "hf_only_estimate")
                if np.isfinite(hf_only):
                    metrics["realized_hf_error"] = float(abs(hf_only - reference))

    return True


def _validate_trajectory_samples(cfg: Dict[str, Any], samples: List[Dict[str, Any]], phase: str, cell_id: str) -> None:
    trajectory_cfg = cfg.get("sampling", {}).get("trajectory", {})
    if not isinstance(trajectory_cfg, dict) or not bool(trajectory_cfg.get("enabled", False)):
        return

    missing_trajectory = [idx for idx, sample in enumerate(samples) if "trajectory_index" not in sample]
    if missing_trajectory:
        raise ValueError(
            f"Trajectory sampling is enabled, but {phase} samples for cell '{cell_id}' "
            f"are missing trajectory_index at indices {missing_trajectory[:10]}"
            f"{'...' if len(missing_trajectory) > 10 else ''}. "
            "Check that the running YAML contains sampling.trajectory.enabled/path and that the updated code is on the cluster."
        )

    env_model = str(cfg.get("execution", {}).get("environment", {}).get("model", "csv"))
    atmosphere_mode = str(trajectory_cfg.get("atmosphere", "from_csv")).lower()
    if env_model == "csv" and atmosphere_mode in {"from_csv", "csv", "precomputed"}:
        missing_atmosphere = [idx for idx, sample in enumerate(samples) if "atmosphere_row" not in sample]
        if missing_atmosphere:
            raise ValueError(
                f"Trajectory sampling is enabled with CSV atmosphere, but {phase} samples for cell '{cell_id}' "
                f"are missing atmosphere_row at indices {missing_atmosphere[:10]}"
                f"{'...' if len(missing_atmosphere) > 10 else ''}. "
                "The trajectory CSV must contain density_kg_m3, temperature_K, x_o_fraction, "
                "x_n2_fraction, x_o2_fraction, and x_he_fraction, or use execution.environment.model=pymsis_hwm14."
            )


def _build_result_row(
    cell,
    geometry: Dict[str, Any],
    regime: Dict[str, Any],
    metrics: Dict[str, Any],
    quantity_kind: str,
    qoi_expression: str,
    flags: List[str],
) -> Dict[str, Any]:
    regime_desc = regime.get("descriptors", {})
    return {
        "study_id": cell.study_id,
        "cell_id": cell.cell_id(),
        "mode": cell.mode,
        "geometry_id": cell.geometry_id,
        "geometry_name": geometry.get("name", cell.geometry_id),
        "geometry_class": geometry.get("geometry_class"),
        "geometry_characteristic_length": geometry.get("characteristic_length"),
        "regime_id": cell.regime_id,
        "regime_label": regime.get("label", cell.regime_id),
        "active_sources": list(cell.active_source_blocks),
        "qoi": cell.qoi,
        "quantity_kind": quantity_kind,
        "qoi_expression": qoi_expression,
        "hf_model_id": cell.hf_model_id,
        "lf_model_id": cell.lf_model_id,
        "pilot_size": cell.pilot_size,
        "budget": cell.budget,
        "repetition": cell.repetition,
        "seed": cell.seed,
        "flags": flags,
        "regime_altitude_km": regime_desc.get("altitude_km"),
        "regime_knudsen_number": regime_desc.get("knudsen_number", regime_desc.get("knudsen_proxy")),
        "regime_speed_ratio": regime_desc.get("speed_ratio"),
        "regime_freestream_temperature": regime_desc.get("freestream_temperature"),
        "regime_composition_descriptor": regime_desc.get("composition_descriptor"),
        "regime_solar_activity_state": regime_desc.get("solar_activity_state"),
        "regime_geomagnetic_activity_state": regime_desc.get("geomagnetic_activity_state"),
        "regime_wind_state": regime_desc.get("wind_state"),
        "regime_surface_state": regime_desc.get("surface_state"),
        **metrics,
    }


def _append_model_evaluation_rows(store: ResultStore, cell, request, result, phase: str) -> None:
    values_by_qoi = result.values_by_qoi if isinstance(result.values_by_qoi, dict) else {}
    qois = [str(q) for q in request.qois]
    max_len = max(
        [len(result.sample_ids), len(result.costs)] + [len(values_by_qoi.get(q, [])) for q in qois],
        default=0,
    )
    rows: List[Dict[str, Any]] = []
    for qoi in qois:
        vals = list(values_by_qoi.get(qoi, []))
        for idx in range(max_len):
            rows.append(
                {
                    "study_id": cell.study_id,
                    "cell_id": cell.cell_id(),
                    "phase": phase,
                    "mode": cell.mode,
                    "geometry_id": cell.geometry_id,
                    "regime_id": cell.regime_id,
                    "active_sources": list(cell.active_source_blocks),
                    "qoi": qoi,
                    "model_id": request.model_id,
                    "fidelity": request.fidelity,
                    "hf_model_id": cell.hf_model_id,
                    "lf_model_id": cell.lf_model_id,
                    "pilot_size": cell.pilot_size,
                    "budget": cell.budget,
                    "repetition": cell.repetition,
                    "seed": request.seed,
                    "sample_id": result.sample_ids[idx] if idx < len(result.sample_ids) else "",
                    "sample_index": idx,
                    "value": vals[idx] if idx < len(vals) else float("nan"),
                    "cost": result.costs[idx] if idx < len(result.costs) else float("nan"),
                }
            )
    if hasattr(store, "append_model_evaluations"):
        store.append_model_evaluations(rows)
    else:
        for row in rows:
            store.append_model_evaluation(row)


def run_campaign(
    config: Dict[str, Any],
    resume: bool = False,
    pilots_only: bool = False,
) -> Dict[str, Any]:
    cfg = validate_or_raise(normalize_config(config))
    qoi_registry = build_qoi_registry(cfg)

    output_dir = str(cfg.get("outputs", {}).get("dir", "campaign_outputs/default"))
    store = ResultStore(output_dir)
    should_resume = bool(resume or cfg.get("execution", {}).get("resume", False))
    if not should_resume:
        store.reset_outputs(keep_cache=True)

    cache = EvaluationCache(store.cache_json)
    registry = build_adapter_registry(cfg)
    cells = generate_experiment_cells(cfg)

    if cfg.get("outputs", {}).get("write_config_snapshot", True):
        store.write_config_snapshot(cfg, get_run_fingerprint())

    done = store.load_completed_cell_ids() if should_resume else set()
    if done:
        print(f"[resume] completed cells already recorded={len(done)}", flush=True)

    trajectory_sampling_enabled = bool(cfg.get("sampling", {}).get("trajectory", {}).get("enabled", False))
    input_model = InputModel(
        cfg.get("variables", []),
        cfg.get("sampling", {}),
        regime_label_map=cfg.get("regime_label_map", {}),
    )
    if trajectory_sampling_enabled and not getattr(input_model, "trajectory_records", []):
        raise ValueError(
            "sampling.trajectory.enabled is true, but InputModel loaded zero trajectory records. "
            "Check sampling.trajectory.path and make sure the updated mfmc_campaign/sampling.py is deployed."
        )

    by_base_cell: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    external_model_eval_robustness_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    max_prod = int(cfg.get("sampling", {}).get("max_production_samples", 2000))
    hf_fraction = float(cfg.get("budget", {}).get("hf_fraction", 0.25))
    min_hf = int(cfg.get("budget", {}).get("min_hf", cfg.get("sampling", {}).get("min_hf", 4)))
    backend = str(cfg.get("execution", {}).get("backend", "mock"))
    reuse_pilot_across_budgets = bool(cfg.get("pilot", {}).get("reuse_across_budgets", False))
    _campaign_overview(
        cfg,
        cells,
        output_dir,
        should_resume=should_resume,
        pilots_only=pilots_only,
        max_prod=max_prod,
        hf_fraction=hf_fraction,
        min_hf=min_hf,
        trajectory_sampling_enabled=trajectory_sampling_enabled,
    )
    budget_order: List[Any] = []
    for generated_cell in cells:
        if generated_cell.budget not in budget_order:
            budget_order.append(generated_cell.budget)
    budget_totals: Dict[Any, int] = defaultdict(int)
    budget_seen: Dict[Any, int] = defaultdict(int)
    for generated_cell in cells:
        budget_totals[generated_cell.budget] += 1
    external_pilot_csv = _pilot_model_evaluations_csv_path(cfg)
    external_pilot_robustness_csv = _pilot_robustness_csv_path(cfg)
    external_pilot_robustness_rows = _load_pilot_robustness_rows(external_pilot_robustness_csv)
    if trajectory_sampling_enabled and not bool(cfg.get("pilot", {}).get("allow_external_with_trajectory", False)):
        if external_pilot_csv:
            print(
                "[pilot] Ignoring external pilot model evaluations because trajectory sampling is enabled "
                "and pilot.allow_external_with_trajectory is false.",
                flush=True,
            )
        external_pilot_csv = None
    elif external_pilot_csv:
        if os.path.exists(str(external_pilot_csv)):
            print(f"[pilot] External pilot model evaluations CSV found: {external_pilot_csv}", flush=True)
        else:
            print(f"[pilot] External pilot model evaluations CSV not found: {external_pilot_csv}", flush=True)
    else:
        print("[pilot] No external pilot model evaluations CSV configured.", flush=True)
    external_pilot_study_id = cfg.get("pilot", {}).get("source_study_id")
    external_pilot_mode = cfg.get("pilot", {}).get("source_mode")

    n_executed = 0
    n_skipped = 0
    single_allocation_all_qois = _single_allocation_all_qois_enabled(cfg)
    direct_qois = _direct_qois_for_config(cfg)
    if single_allocation_all_qois and not direct_qois:
        raise ValueError("qois.single_allocation_all_qois=true requires at least one direct QoI.")
    anchor_qoi = "C_D" if ("C_D" in direct_qois and _batch_cd_cd2_enabled(cfg)) else (direct_qois[0] if direct_qois else "")
    if single_allocation_all_qois:
        print(
            f"[qoi] single_allocation_all_qois enabled anchor_qoi={anchor_qoi} all_qois={'+'.join(direct_qois)}",
            flush=True,
        )

    for cell_index, cell in enumerate(cells, start=1):
        cid = cell.cell_id()
        budget_seen[cell.budget] += 1
        budget_index = budget_order.index(cell.budget) + 1 if cell.budget in budget_order else 0
        budget_total = len(budget_order)
        budget_item_index = budget_seen[cell.budget]
        budget_item_total = budget_totals[cell.budget]
        if cid in done:
            n_skipped += 1
            print(
                f"[skip] budget {budget_index}/{budget_total} item {budget_item_index}/{budget_item_total} "
                f"qoi={cell.qoi} budget={_fmt_num(cell.budget)} already complete",
                flush=True,
            )
            continue
        if _skip_batched_qoi_cell(cfg, cell.qoi):
            n_skipped += 1
            print(
                f"[skip] budget {budget_index}/{budget_total} item {budget_item_index}/{budget_item_total} "
                f"qoi={cell.qoi} B={_fmt_num(cell.budget)} covered by C_D batch",
                flush=True,
            )
            continue
        if single_allocation_all_qois and str(cell.qoi) != anchor_qoi:
            n_skipped += 1
            print(
                f"[skip] budget {budget_index}/{budget_total} item {budget_item_index}/{budget_item_total} "
                f"qoi={cell.qoi} B={_fmt_num(cell.budget)} covered by single_allocation_all_qois",
                flush=True,
            )
            continue

        eval_qois = direct_qois if single_allocation_all_qois else _evaluation_qois_for_cell(cfg, cell.qoi)
        eval_qoi_key = "+".join(eval_qois)

        geometry = _find_geometry(cfg, cell.geometry_id)
        regime = _find_regime(cfg, cell.regime_id)
        lf_model_ids = _lf_model_ids_for_cell(cell)
        multi_lf_cell = len(lf_model_ids) > 1
        paper_mfmc_cell = multi_lf_cell and _paper_mfmc_enabled(cfg)
        print(
            f"[cell {cell_index}/{len(cells)}] budget {budget_index}/{budget_total} "
            f"item {budget_item_index}/{budget_item_total} qoi={cell.qoi} "
            f"B={_fmt_num(cell.budget)} geometry={cell.geometry_id} regime={cell.regime_id} "
            f"lf={'+'.join(lf_model_ids)} pilot={cell.pilot_size} rep={cell.repetition}"
            f"{' strategy=paper_mfmc' if paper_mfmc_cell else ''}",
            flush=True,
        )

        pilot_seed = (
            _stable_seed(
                cfg.get("seeds", {}).get("global", 12345),
                cell.study_id,
                cell.mode,
                cell.geometry_id,
                cell.regime_id,
                "+".join(sorted(cell.active_source_blocks)),
                cell.qoi,
                cell.hf_model_id,
                cell.repetition,
                cell.pilot_size,
                "shared_pilot",
            )
            if reuse_pilot_across_budgets
            else cell.seed
        )
        rng = np.random.default_rng(pilot_seed)
        context = SamplingContext(regime_id=cell.regime_id, active_source_blocks=cell.active_source_blocks)

        pilot_samples = input_model.sample(cell.pilot_size, context, rng)
        _validate_trajectory_samples(cfg, pilot_samples, "pilot", cid)
        pilot_ids = [f"pilot_{i}" for i in range(cell.pilot_size)]
        print(f"[pilot] samples ready n={len(pilot_samples)} seed={pilot_seed}", flush=True)

        hf_adapter = registry.get(cell.hf_model_id)

        meta = {
            "aos_deg": cfg.get("execution", {}).get("aos_deg", 0),
            "aoa_deg": cfg.get("execution", {}).get("aoa_deg", 0),
            "geometry_id": geometry.get("id", geometry.get("name", cell.geometry_id)),
            "geometry_name": geometry.get("name", geometry.get("id", cell.geometry_id)),
            "geometry_class": geometry.get("geometry_class"),
        }
        if isinstance(geometry.get("metadata"), dict):
            meta.update(geometry.get("metadata", {}))
        env_cfg = cfg.get("execution", {}).get("environment", {})
        if isinstance(env_cfg, dict):
            meta.update(env_cfg)
            if "model" in env_cfg and "environment_model" not in meta:
                meta["environment_model"] = env_cfg.get("model")
        for key in [
            "flow_zero_direction",
            "flow_zero_direction_xyz",
            "zero_flow_direction",
            "zero_flow_direction_xyz",
            "adbsat_aos_offset_deg",
            "adbsat_aos_offset",
        ]:
            if key in cfg.get("execution", {}):
                meta[key] = cfg.get("execution", {}).get(key)

        pilot_hf_req = make_request(
            study_id=cell.study_id,
            cell_id=cid,
            model_id=cell.hf_model_id,
            fidelity="hf",
            qois=eval_qois,
            geometry=geometry,
            regime=regime,
            active_source_blocks=cell.active_source_blocks,
            sample_ids=pilot_ids,
            samples=pilot_samples,
            seed=pilot_seed,
            metadata=meta,
        )
        pilot_hf_res = _external_pilot_result(
            path=str(external_pilot_csv) if external_pilot_csv else None,
            cell=cell,
            phase="pilot_hf",
            model_id=cell.hf_model_id,
            qoi=cell.qoi,
            qois=eval_qois,
            source_study_id=str(external_pilot_study_id) if external_pilot_study_id is not None else None,
            source_mode=str(external_pilot_mode) if external_pilot_mode is not None else None,
            match_lf_model_id=False,
        )

        pilot_lf_reqs = {}
        pilot_lf_results = {}
        for lf_idx, lf_model_id in enumerate(lf_model_ids):
            req = make_request(
                study_id=cell.study_id,
                cell_id=cid,
                model_id=lf_model_id,
                fidelity="lf",
                qois=eval_qois,
                geometry=geometry,
                regime=regime,
                active_source_blocks=cell.active_source_blocks,
                sample_ids=pilot_ids,
                samples=pilot_samples,
                seed=pilot_seed + 17 + lf_idx,
                metadata=meta,
            )
            pilot_lf_reqs[lf_model_id] = req
            pilot_lf_results[lf_model_id] = _external_pilot_result(
                path=str(external_pilot_csv) if external_pilot_csv else None,
                cell=cell,
                phase="pilot_lf",
                model_id=lf_model_id,
                qoi=cell.qoi,
                qois=eval_qois,
                source_study_id=str(external_pilot_study_id) if external_pilot_study_id is not None else None,
                source_mode=str(external_pilot_mode) if external_pilot_mode is not None else None,
                lf_model_id_filter=lf_model_id,
            )

        all_lf_external = all(res is not None for res in pilot_lf_results.values())
        any_lf_external = any(res is not None for res in pilot_lf_results.values())
        hf_external = pilot_hf_res is not None
        partial_external_pilot_used = bool(hf_external or any_lf_external)
        external_pilot_used = bool(hf_external and all_lf_external)
        if external_pilot_csv:
            if external_pilot_used:
                print("[pilot] external model evaluations matched", flush=True)
            elif partial_external_pilot_used:
                missing_models = []
                if not hf_external:
                    missing_models.append(cell.hf_model_id)
                missing_models.extend(lf_model_id for lf_model_id in lf_model_ids if pilot_lf_results[lf_model_id] is None)
                matched_models = []
                if hf_external:
                    matched_models.append(cell.hf_model_id)
                matched_models.extend(lf_model_id for lf_model_id in lf_model_ids if pilot_lf_results[lf_model_id] is not None)
                print(
                    f"[pilot] external model evaluations partially matched; "
                    f"matched={'+'.join(matched_models) or 'none'} "
                    f"missing={'+'.join(missing_models) or 'none'}; evaluating missing only",
                    flush=True,
                )
            else:
                print(
                    "[pilot] external model evaluations did not match; evaluating pilot samples",
                    flush=True,
                )

        external_robustness_by_lf = _external_pilot_robustness_by_lf(
            external_pilot_robustness_rows,
            cell=cell,
            lf_model_ids=lf_model_ids,
        )
        if external_pilot_robustness_csv:
            if external_robustness_by_lf:
                matched_lfs = "+".join(sorted(external_robustness_by_lf))
                print(f"[pilot] external robustness matched lf={matched_lfs}", flush=True)
            else:
                print(
                    "[pilot] external robustness did not match; using current pilot robustness",
                    flush=True,
                )

        if external_pilot_used and external_robustness_by_lf:
            print(
                "[pilot] external model evaluations matched; recomputing robustness from model evaluations "
                "instead of using external robustness overrides",
                flush=True,
            )
            external_robustness_by_lf = {}

        all_lf_robustness_external = all(lf_model_id in external_robustness_by_lf for lf_model_id in lf_model_ids)
        skip_pilot_model_evaluations = (
            bool(cfg.get("pilot", {}).get("use_robustness_without_model_evaluations", True))
            and all_lf_robustness_external
            and not external_pilot_used
            and not partial_external_pilot_used
        )
        if paper_mfmc_cell and skip_pilot_model_evaluations:
            skip_pilot_model_evaluations = False
            print(
                "[pilot] paper_mfmc requires coupled pilot values for per-LF allocation; "
                "external robustness alone is not sufficient.",
                flush=True,
            )
        if external_robustness_by_lf and not all_lf_robustness_external:
            missing_lfs = sorted(set(lf_model_ids) - set(external_robustness_by_lf))
            print(
                f"[pilot] external robustness incomplete; missing lf={'+'.join(missing_lfs)}. "
                "Pilot model evaluations are still required.",
                flush=True,
            )
        if skip_pilot_model_evaluations:
            print(
                "[pilot] skipping pilot HF/LF evaluations; using external robustness correlation/beta/allocation",
                flush=True,
            )

        external_pilot_hf_cost_array = _external_pilot_cost_array(
            path=str(external_pilot_csv) if external_pilot_csv else None,
            cell=cell,
            phase="pilot_hf",
            model_id=cell.hf_model_id,
            qoi=cell.qoi,
            source_study_id=str(external_pilot_study_id) if external_pilot_study_id is not None else None,
            source_mode=str(external_pilot_mode) if external_pilot_mode is not None else None,
            match_lf_model_id=False,
        )
        external_pilot_lf_cost_arrays = {
            lf_model_id: _external_pilot_cost_array(
                path=str(external_pilot_csv) if external_pilot_csv else None,
                cell=cell,
                phase="pilot_lf",
                model_id=lf_model_id,
                qoi=cell.qoi,
                source_study_id=str(external_pilot_study_id) if external_pilot_study_id is not None else None,
                source_mode=str(external_pilot_mode) if external_pilot_mode is not None else None,
                lf_model_id_filter=lf_model_id,
            )
            for lf_model_id in lf_model_ids
        }

        if not skip_pilot_model_evaluations:
            pilot_jobs: List[Tuple[str, Any, Any, str, str]] = []
            if pilot_hf_res is None:
                pilot_jobs.append(("__hf__", hf_adapter, pilot_hf_req, eval_qoi_key, "pilot_hf"))
            for lf_model_id in lf_model_ids:
                if pilot_lf_results[lf_model_id] is None:
                    pilot_jobs.append(
                        (
                            lf_model_id,
                            registry.get(lf_model_id),
                            pilot_lf_reqs[lf_model_id],
                            eval_qoi_key,
                            "pilot_lf",
                        )
                    )
            if pilot_jobs:
                pilot_eval_results = _evaluate_many_with_cache(cache, pilot_jobs)
                if "__hf__" in pilot_eval_results:
                    pilot_hf_res = pilot_eval_results["__hf__"]
                for lf_model_id in lf_model_ids:
                    if lf_model_id in pilot_eval_results:
                        pilot_lf_results[lf_model_id] = pilot_eval_results[lf_model_id]
        if cfg.get("outputs", {}).get("write_model_evaluations", True) and not skip_pilot_model_evaluations:
            _append_model_evaluation_rows(store, cell, pilot_hf_req, pilot_hf_res, "pilot_hf")
            for lf_model_id in lf_model_ids:
                _append_model_evaluation_rows(store, cell, pilot_lf_reqs[lf_model_id], pilot_lf_results[lf_model_id], "pilot_lf")

        if skip_pilot_model_evaluations:
            pilot_hf = np.asarray([], dtype=float)
            pilot_lfs = {lf_model_id: np.asarray([], dtype=float) for lf_model_id in lf_model_ids}
            missing_external_costs = []
            if external_pilot_hf_cost_array is None:
                missing_external_costs.append(cell.hf_model_id)
            missing_external_costs.extend(
                lf_model_id
                for lf_model_id in lf_model_ids
                if external_pilot_lf_cost_arrays.get(lf_model_id) is None
            )
            if external_pilot_csv and missing_external_costs:
                raise ValueError(
                    "External pilot robustness matched, but measured costs could not be read from "
                    f"model_evaluations.csv for model(s): {'+'.join(missing_external_costs)}. "
                    "Refusing to use configured/default LF costs because that would corrupt allocation. "
                    f"Check phase/model_id/qoi/pilot_size/repetition matching in {external_pilot_csv}."
                )
            pilot_hf_cost_array = (
                external_pilot_hf_cost_array
                if external_pilot_hf_cost_array is not None
                else np.asarray([_configured_hf_cost(cfg, cell)], dtype=float)
            )
            pilot_lf_cost_arrays = {
                lf_model_id: (
                    external_pilot_lf_cost_arrays[lf_model_id]
                    if external_pilot_lf_cost_arrays.get(lf_model_id) is not None
                    else np.asarray([_configured_lf_cost(cfg, lf_model_id)], dtype=float)
                )
                for lf_model_id in lf_model_ids
            }
        else:
            pilot_hf = np.asarray(pilot_hf_res.values_by_qoi[cell.qoi], dtype=float)
            pilot_lfs = {
                lf_model_id: np.asarray(pilot_lf_results[lf_model_id].values_by_qoi[cell.qoi], dtype=float)
                for lf_model_id in lf_model_ids
            }
            pilot_hf_cost_array = np.asarray(pilot_hf_res.costs, dtype=float)
            pilot_lf_cost_arrays = {
                lf_model_id: np.asarray(pilot_lf_results[lf_model_id].costs, dtype=float)
                for lf_model_id in lf_model_ids
            }
        pilot_lf = pilot_lfs[lf_model_ids[0]]
        pilot_hf_cost = float(np.nanmean(pilot_hf_cost_array))
        pilot_lf_costs = {
            lf_model_id: float(np.nanmean(pilot_lf_cost_arrays[lf_model_id]))
            for lf_model_id in lf_model_ids
        }
        pilot_lf_cost = float(np.nansum(list(pilot_lf_costs.values()))) if multi_lf_cell else pilot_lf_costs[lf_model_ids[0]]
        if skip_pilot_model_evaluations:
            cost_parts = ", ".join(f"{lf_model_id}={pilot_lf_costs[lf_model_id]:.6g}" for lf_model_id in lf_model_ids)
            print(
                f"[pilot] external cost source=model_evaluations hf_cost={pilot_hf_cost:.6g} "
                f"lf_costs={{ {cost_parts} }} lf_cost_total={pilot_lf_cost:.6g}",
                flush=True,
            )

        robust_reps = int(cfg.get("pilot", {}).get("robustness_repetitions", 20))
        robust_sizes = (
            [int(v) for v in cfg.get("pilot", {}).get("sizes", [cell.pilot_size])]
            if pilots_only
            else [cell.pilot_size]
        )
        reuse_external_model_eval_stats = bool(
            cfg.get("pilot", {}).get("reuse_external_model_evaluation_stats", True)
        )
        external_stats_cache_key = (
            cell.study_id,
            cell.mode,
            cell.geometry_id,
            cell.regime_id,
            "+".join(sorted(cell.active_source_blocks)),
            cell.qoi,
            cell.hf_model_id,
            cell.lf_model_id,
            cell.pilot_size,
            tuple(robust_sizes),
            robust_reps,
        )
        if skip_pilot_model_evaluations:
            robust_rows = list(external_robustness_by_lf.values())
            print(
                f"[pilot] external robustness accepted hf_cost={pilot_hf_cost:.6g} "
                f"lf_cost={pilot_lf_cost:.6g}",
                flush=True,
            )
        else:
            if (
                external_pilot_used
                and reuse_external_model_eval_stats
                and external_stats_cache_key in external_model_eval_robustness_cache
            ):
                robust_rows = external_model_eval_robustness_cache[external_stats_cache_key]
                print(
                    f"[pilot] reused robustness from external model_evaluations sizes={robust_sizes} "
                    f"repetitions={robust_reps} hf_cost_mean={pilot_hf_cost:.6g} "
                    f"lf_cost_mean={pilot_lf_cost:.6g}",
                    flush=True,
                )
            else:
                robustness_rng = (
                    np.random.default_rng(
                        _stable_seed(
                            cfg.get("seeds", {}).get("global", 12345),
                            "external_model_evaluation_stats",
                            *external_stats_cache_key,
                        )
                    )
                    if external_pilot_used and reuse_external_model_eval_stats
                    else rng
                )
                if paper_mfmc_cell:
                    robust_rows = _paper_mfmc_pilot_robustness_metrics(
                        pilot_hf=pilot_hf,
                        pilot_lfs=pilot_lfs,
                        pilot_sizes=robust_sizes,
                        repetitions=robust_reps,
                        rng=robustness_rng,
                        hf_cost=pilot_hf_cost,
                        lf_costs=pilot_lf_costs,
                    )
                elif multi_lf_cell:
                    robust_rows = _multi_lf_pilot_robustness_metrics(
                        pilot_hf=pilot_hf,
                        pilot_lfs=pilot_lfs,
                        pilot_sizes=robust_sizes,
                        repetitions=robust_reps,
                        rng=robustness_rng,
                        hf_cost=pilot_hf_cost,
                        lf_cost=pilot_lf_cost,
                    )
                else:
                    robust_rows = pilot_robustness_metrics(
                        pilot_hf=pilot_hf,
                        pilot_lf=pilot_lf,
                        pilot_sizes=robust_sizes,
                        repetitions=robust_reps,
                        rng=robustness_rng,
                        hf_cost=pilot_hf_cost,
                        lf_cost=pilot_lf_cost,
                    )
                if external_pilot_used and reuse_external_model_eval_stats:
                    external_model_eval_robustness_cache[external_stats_cache_key] = robust_rows
                    print(
                        f"[pilot] computed robustness from external model_evaluations sizes={robust_sizes} "
                        f"repetitions={robust_reps} hf_cost_mean={pilot_hf_cost:.6g} "
                        f"lf_cost_mean={pilot_lf_cost:.6g}",
                        flush=True,
                    )
                else:
                    print(
                        f"[pilot] local robustness computed sizes={robust_sizes} "
                        f"repetitions={robust_reps} hf_cost_mean={pilot_hf_cost:.6g} "
                        f"lf_cost_mean={pilot_lf_cost:.6g}",
                        flush=True,
                    )
            for rr in robust_rows:
                store.append_robustness(
                    {
                        "study_id": cell.study_id,
                        "cell_id": cid,
                        "mode": cell.mode,
                        "geometry_id": cell.geometry_id,
                        "geometry_class": geometry.get("geometry_class"),
                        "regime_id": cell.regime_id,
                        "active_sources": list(cell.active_source_blocks),
                        "qoi": cell.qoi,
                        "hf_model_id": cell.hf_model_id,
                        "lf_model_id": cell.lf_model_id,
                        "repetition": cell.repetition,
                        **rr,
                    }
                )

        beta_stability = beta_stability_metrics(
            pilot_hf=pilot_hf,
            pilot_lf=pilot_lf,
            repetitions=robust_reps,
            rng=rng,
        )

        if pilots_only:
            for result_qoi in eval_qois:
                result_cell = cell if result_qoi == cell.qoi else replace(cell, qoi=result_qoi)
                qoi_pilot_hf = np.asarray(pilot_hf_res.values_by_qoi.get(result_qoi, []), dtype=float)
                qoi_pilot_lfs = {
                    lf_model_id: np.asarray(pilot_lf_results[lf_model_id].values_by_qoi.get(result_qoi, []), dtype=float)
                    for lf_model_id in lf_model_ids
                }
                if result_qoi == cell.qoi:
                    qoi_robust_rows = robust_rows
                elif qoi_pilot_hf.size and qoi_pilot_lfs[lf_model_ids[0]].size:
                    if paper_mfmc_cell:
                        qoi_robust_rows = _paper_mfmc_pilot_robustness_metrics(
                            pilot_hf=qoi_pilot_hf,
                            pilot_lfs=qoi_pilot_lfs,
                            pilot_sizes=robust_sizes,
                            repetitions=robust_reps,
                            rng=rng,
                            hf_cost=pilot_hf_cost,
                            lf_costs=pilot_lf_costs,
                        )
                    elif multi_lf_cell:
                        qoi_robust_rows = _multi_lf_pilot_robustness_metrics(
                            pilot_hf=qoi_pilot_hf,
                            pilot_lfs=qoi_pilot_lfs,
                            pilot_sizes=robust_sizes,
                            repetitions=robust_reps,
                            rng=rng,
                            hf_cost=pilot_hf_cost,
                            lf_cost=pilot_lf_cost,
                        )
                    else:
                        qoi_robust_rows = pilot_robustness_metrics(
                            pilot_hf=qoi_pilot_hf,
                            pilot_lf=qoi_pilot_lfs[lf_model_ids[0]],
                            pilot_sizes=robust_sizes,
                            repetitions=robust_reps,
                            rng=rng,
                            hf_cost=pilot_hf_cost,
                            lf_cost=pilot_lf_cost,
                        )
                    for rr in qoi_robust_rows:
                        store.append_robustness(
                            {
                                "study_id": cell.study_id,
                                "cell_id": result_cell.cell_id(),
                                "mode": cell.mode,
                                "geometry_id": cell.geometry_id,
                                "geometry_class": geometry.get("geometry_class"),
                                "regime_id": cell.regime_id,
                                "active_sources": list(cell.active_source_blocks),
                                "qoi": result_qoi,
                                "hf_model_id": cell.hf_model_id,
                                "lf_model_id": cell.lf_model_id,
                                "repetition": cell.repetition,
                                **rr,
                            }
                        )
                else:
                    qoi_robust_rows = []

                metrics = compute_mfmc_diagnostics(
                    qoi=result_qoi,
                    pilot_hf=qoi_pilot_hf,
                    pilot_lf=qoi_pilot_lfs[lf_model_ids[0]],
                    prod_hf=qoi_pilot_hf,
                    prod_lf_full=qoi_pilot_lfs[lf_model_ids[0]],
                    prod_lf_paired=qoi_pilot_lfs[lf_model_ids[0]],
                    hf_costs=pilot_hf_cost_array,
                    lf_costs_full=pilot_lf_cost_arrays[lf_model_ids[0]],
                    reference=float("nan"),
                )
                metrics.update(
                    beta_stability_metrics(
                        pilot_hf=qoi_pilot_hf,
                        pilot_lf=qoi_pilot_lfs[lf_model_ids[0]],
                        repetitions=robust_reps,
                        rng=rng,
                    )
                )
                qoi_external_robustness = external_robustness_by_lf if result_qoi == cell.qoi else {}
                pilot_corr_used = _apply_external_pilot_robustness(
                    metrics,
                    qoi_external_robustness,
                    lf_model_ids=lf_model_ids,
                    prod_lf_full_by_id=qoi_pilot_lfs,
                    prod_lf_paired_by_id=qoi_pilot_lfs,
                    reference=float("nan"),
                )
                if not pilot_corr_used:
                    pilot_corr_used = _use_pilot_correlation(metrics, qoi_robust_rows, cell.pilot_size)
                flags = statistical_flags(metrics) + ["pilot_only"]
                if result_qoi != cell.qoi:
                    flags.append(f"batched_with_{cell.qoi}")
                if pilot_corr_used:
                    flags.append("pilot_correlation_used")
                if qoi_external_robustness:
                    flags.append("external_pilot_robustness")
                if partial_external_pilot_used and result_qoi == cell.qoi:
                    flags.append("external_pilot_model_evaluations")
                if multi_lf_cell:
                    flags.append("multi_lf")
                if paper_mfmc_cell:
                    flags.append("paper_mfmc")
                row = _build_result_row(
                    result_cell,
                    geometry,
                    regime,
                    metrics=metrics,
                    quantity_kind=qoi_registry.quantity_kind(result_qoi),
                    qoi_expression=qoi_registry.expression(result_qoi),
                    flags=flags,
                )
                store.append_result(row)
            n_executed += 1
            print(
                f"[done] pilot-only qoi={cell.qoi} B={_fmt_num(cell.budget)} "
                f"executed={n_executed} skipped={n_skipped}",
                flush=True,
            )
            continue

        allocation_ratio_qoi = cell.qoi
        allocation_robust_rows = robust_rows
        if single_allocation_all_qois and not skip_pilot_model_evaluations:
            per_qoi_robust_rows: Dict[str, List[Dict[str, Any]]] = {cell.qoi: robust_rows}
            for alloc_qoi in eval_qois:
                if alloc_qoi == cell.qoi:
                    continue
                alloc_pilot_hf = np.asarray(pilot_hf_res.values_by_qoi.get(alloc_qoi, []), dtype=float)
                alloc_pilot_lfs = {
                    lf_model_id: np.asarray(pilot_lf_results[lf_model_id].values_by_qoi.get(alloc_qoi, []), dtype=float)
                    for lf_model_id in lf_model_ids
                }
                if not alloc_pilot_hf.size or not alloc_pilot_lfs[lf_model_ids[0]].size:
                    continue
                if multi_lf_cell:
                    alloc_rows = _multi_lf_pilot_robustness_metrics(
                        pilot_hf=alloc_pilot_hf,
                        pilot_lfs=alloc_pilot_lfs,
                        pilot_sizes=robust_sizes,
                        repetitions=robust_reps,
                        rng=rng,
                        hf_cost=pilot_hf_cost,
                        lf_cost=pilot_lf_cost,
                    )
                else:
                    alloc_rows = pilot_robustness_metrics(
                        pilot_hf=alloc_pilot_hf,
                        pilot_lf=alloc_pilot_lfs[lf_model_ids[0]],
                        pilot_sizes=robust_sizes,
                        repetitions=robust_reps,
                        rng=rng,
                        hf_cost=pilot_hf_cost,
                        lf_cost=pilot_lf_cost,
                    )
                if alloc_rows:
                    per_qoi_robust_rows[alloc_qoi] = alloc_rows

            weakest_abs_corr = float("inf")
            for alloc_qoi, alloc_rows in per_qoi_robust_rows.items():
                corr = _pilot_robustness_correlation(alloc_rows, cell.pilot_size)
                if np.isfinite(corr) and abs(corr) < weakest_abs_corr:
                    weakest_abs_corr = abs(corr)
                    allocation_ratio_qoi = alloc_qoi
                    allocation_robust_rows = alloc_rows
            print(
                f"[allocation] single_allocation_all_qois anchor={allocation_ratio_qoi} "
                f"abs_rho={_fmt_num(weakest_abs_corr)}",
                flush=True,
            )

        external_allocation_ratio = _external_allocation_ratio(external_robustness_by_lf)
        allocation_ratio = (
            external_allocation_ratio
            if np.isfinite(external_allocation_ratio) and external_allocation_ratio > 0
            else _robustness_allocation_ratio(allocation_robust_rows, cell.pilot_size)
        )
        paper_lf_order: List[str] = []
        paper_sample_counts: Dict[str, int] = {}
        paper_ratios: Dict[str, float] = {}
        paper_corrs: Dict[str, float] = {}
        if paper_mfmc_cell:
            paper_lf_order, paper_sample_counts, paper_ratios, paper_corrs = _paper_mfmc_allocation(
                pilot_hf=pilot_hf,
                pilot_lfs=pilot_lfs,
                pilot_hf_cost=pilot_hf_cost,
                pilot_lf_costs=pilot_lf_costs,
                cell_budget=cell.budget,
                min_hf=min_hf,
                cap=max_prod,
            )
            if not paper_lf_order:
                paper_lf_order = list(lf_model_ids)
            n_hf = int(paper_sample_counts.get("__hf__", max(min_hf, 1)))
            n_lf = max([n_hf] + [int(paper_sample_counts.get(lf_model_id, n_hf)) for lf_model_id in paper_lf_order])
            count_parts = " ".join(
                [f"HF={n_hf}"]
                + [
                    f"{lf_model_id}={int(paper_sample_counts.get(lf_model_id, n_hf))}"
                    for lf_model_id in paper_lf_order
                ]
            )
            ratio_parts = " ".join(
                f"{lf_model_id}={paper_ratios.get(lf_model_id, float('nan')):.6g}"
                for lf_model_id in paper_lf_order
            )
            corr_parts = " ".join(
                f"{lf_model_id}={paper_corrs.get(lf_model_id, float('nan')):.6g}"
                for lf_model_id in paper_lf_order
            )
            print(
                f"[allocation] budget {budget_index}/{budget_total} item {budget_item_index}/{budget_item_total} "
                f"B={_fmt_num(cell.budget)} qoi={allocation_ratio_qoi} paper_mfmc "
                f"counts={{ {count_parts} }} ratios={{ {ratio_parts} }} rho={{ {corr_parts} }}",
                flush=True,
            )
        else:
            n_hf, n_lf = _production_sizes(
                cell_budget=cell.budget,
                hf_cost=pilot_hf_cost,
                lf_cost=pilot_lf_cost,
                hf_fraction=hf_fraction,
                fallback_hf=max(min_hf, cell.pilot_size),
                cap=max_prod,
                min_hf=min_hf,
                allocation_ratio=allocation_ratio,
            )
            pilot_corr = _pilot_robustness_correlation(allocation_robust_rows, cell.pilot_size)
            print(
                f"[allocation] budget {budget_index}/{budget_total} item {budget_item_index}/{budget_item_total} "
                f"B={_fmt_num(cell.budget)} qoi={allocation_ratio_qoi} rho={_fmt_num(pilot_corr)} "
                f"ratio={_fmt_num(allocation_ratio)} n_hf={n_hf} n_lf={n_lf}",
                flush=True,
            )

        prod_samples_full = input_model.sample(n_lf, context, rng)
        _validate_trajectory_samples(cfg, prod_samples_full, "production", cid)
        prod_ids_full = [f"prod_{i}" for i in range(n_lf)]

        prod_samples_hf = prod_samples_full[:n_hf]
        prod_ids_hf = prod_ids_full[:n_hf]
        print(
            f"[production] samples ready hf={len(prod_samples_hf)} lf_full={len(prod_samples_full)}",
            flush=True,
        )

        prod_hf_req = make_request(
            study_id=cell.study_id,
            cell_id=cid,
            model_id=cell.hf_model_id,
            fidelity="hf",
            qois=eval_qois,
            geometry=geometry,
            regime=regime,
            active_source_blocks=cell.active_source_blocks,
            sample_ids=prod_ids_hf,
            samples=prod_samples_hf,
            seed=cell.seed + 101,
            metadata=meta,
        )
        prod_lf_full_reqs = {}
        prod_lf_pair_reqs = {}
        empty_lf_results = {
            lf_model_id: EvaluationResult(
                values_by_qoi={qoi_name: [] for qoi_name in eval_qois},
                costs=[],
                sample_ids=[],
                metadata=dict(meta),
            )
            for lf_model_id in lf_model_ids
        }
        for lf_idx, lf_model_id in enumerate(lf_model_ids):
            lf_full_n = int(paper_sample_counts.get(lf_model_id, n_lf)) if paper_mfmc_cell else n_lf
            if lf_full_n > 0:
                prod_lf_full_reqs[lf_model_id] = make_request(
                    study_id=cell.study_id,
                    cell_id=cid,
                    model_id=lf_model_id,
                    fidelity="lf",
                    qois=eval_qois,
                    geometry=geometry,
                    regime=regime,
                    active_source_blocks=cell.active_source_blocks,
                    sample_ids=prod_ids_full[:lf_full_n],
                    samples=prod_samples_full[:lf_full_n],
                    seed=cell.seed + 203 + lf_idx,
                    metadata=meta,
                )
            if not paper_mfmc_cell:
                prod_lf_pair_reqs[lf_model_id] = make_request(
                    study_id=cell.study_id,
                    cell_id=cid,
                    model_id=lf_model_id,
                    fidelity="lf",
                    qois=eval_qois,
                    geometry=geometry,
                    regime=regime,
                    active_source_blocks=cell.active_source_blocks,
                    sample_ids=prod_ids_hf,
                    samples=prod_samples_hf,
                    seed=cell.seed + 307 + lf_idx,
                    metadata=meta,
                )

        production_jobs: List[Tuple[str, Any, Any, str, str]] = [
            ("__prod_hf__", hf_adapter, prod_hf_req, eval_qoi_key, "prod_hf")
        ]
        prod_lf_full_results = {}
        prod_lf_pair_results = {}
        for lf_model_id in lf_model_ids:
            lf_adapter = registry.get(lf_model_id)
            if lf_model_id in prod_lf_full_reqs:
                production_jobs.append(
                    (
                        f"{lf_model_id}__full",
                        lf_adapter,
                        prod_lf_full_reqs[lf_model_id],
                        eval_qoi_key,
                        "prod_lf_full",
                    )
                )
            if not paper_mfmc_cell:
                production_jobs.append(
                    (
                        f"{lf_model_id}__pair",
                        lf_adapter,
                        prod_lf_pair_reqs[lf_model_id],
                        eval_qoi_key,
                        "prod_lf_pair",
                    )
                )
        production_results = _evaluate_many_with_cache(cache, production_jobs)
        prod_hf_res = production_results["__prod_hf__"]
        for lf_model_id in lf_model_ids:
            prod_lf_full_results[lf_model_id] = production_results.get(
                f"{lf_model_id}__full",
                empty_lf_results[lf_model_id],
            )
            if not paper_mfmc_cell:
                prod_lf_pair_results[lf_model_id] = production_results[f"{lf_model_id}__pair"]
        if cfg.get("outputs", {}).get("write_model_evaluations", True):
            _append_model_evaluation_rows(store, cell, prod_hf_req, prod_hf_res, "prod_hf")
            for lf_model_id in lf_model_ids:
                if lf_model_id in prod_lf_full_reqs:
                    _append_model_evaluation_rows(
                        store,
                        cell,
                        prod_lf_full_reqs[lf_model_id],
                        prod_lf_full_results[lf_model_id],
                        "prod_lf_full",
                    )
                if not paper_mfmc_cell:
                    _append_model_evaluation_rows(
                        store,
                        cell,
                        prod_lf_pair_reqs[lf_model_id],
                        prod_lf_pair_results[lf_model_id],
                        "prod_lf_pair",
                    )

        hf_costs = np.asarray(prod_hf_res.costs, dtype=float)
        lf_costs_full_by_id = {
            lf_model_id: np.asarray(prod_lf_full_results[lf_model_id].costs, dtype=float)
            for lf_model_id in lf_model_ids
        }

        ref_res = None
        if backend == "mock":
            ref_req = make_request(
                study_id=cell.study_id,
                cell_id=cid,
                model_id=cell.hf_model_id,
                fidelity="hf",
                qois=eval_qois,
                geometry=geometry,
                regime=regime,
                active_source_blocks=cell.active_source_blocks,
                sample_ids=prod_ids_full,
                samples=prod_samples_full,
                seed=cell.seed + 401,
                metadata=meta,
            )
            ref_res = _evaluate_with_cache(cache, hf_adapter, ref_req, eval_qoi_key, "reference_hf")
            if cfg.get("outputs", {}).get("write_model_evaluations", True):
                _append_model_evaluation_rows(store, cell, ref_req, ref_res, "reference_hf")

        base_key = (
            cell.study_id,
            cell.mode,
            cell.geometry_id,
            cell.regime_id,
            "+".join(sorted(cell.active_source_blocks)),
            cell.hf_model_id,
            cell.lf_model_id,
            cell.repetition,
            cell.pilot_size,
            cell.budget,
        )

        rows_written: List[Dict[str, Any]] = []
        for result_qoi in eval_qois:
            result_cell = cell if result_qoi == cell.qoi else replace(cell, qoi=result_qoi)
            qoi_external_robustness = external_robustness_by_lf if result_qoi == cell.qoi else {}
            qoi_robust_rows = robust_rows if result_qoi == cell.qoi else []

            if skip_pilot_model_evaluations:
                qoi_pilot_hf = np.asarray([], dtype=float)
                qoi_pilot_lfs = {lf_model_id: np.asarray([], dtype=float) for lf_model_id in lf_model_ids}
            else:
                qoi_pilot_hf = np.asarray(pilot_hf_res.values_by_qoi.get(result_qoi, []), dtype=float)
                qoi_pilot_lfs = {
                    lf_model_id: np.asarray(pilot_lf_results[lf_model_id].values_by_qoi.get(result_qoi, []), dtype=float)
                    for lf_model_id in lf_model_ids
                }
                if result_qoi != cell.qoi and qoi_pilot_hf.size and qoi_pilot_lfs[lf_model_ids[0]].size:
                    qoi_external_stats_cache_key = (
                        cell.study_id,
                        cell.mode,
                        cell.geometry_id,
                        cell.regime_id,
                        "+".join(sorted(cell.active_source_blocks)),
                        result_qoi,
                        cell.hf_model_id,
                        cell.lf_model_id,
                        cell.pilot_size,
                        tuple(robust_sizes),
                        robust_reps,
                    )
                    if (
                        external_pilot_used
                        and reuse_external_model_eval_stats
                        and qoi_external_stats_cache_key in external_model_eval_robustness_cache
                    ):
                        qoi_robust_rows = external_model_eval_robustness_cache[qoi_external_stats_cache_key]
                    else:
                        qoi_robustness_rng = (
                            np.random.default_rng(
                                _stable_seed(
                                    cfg.get("seeds", {}).get("global", 12345),
                                    "external_model_evaluation_stats",
                                    *qoi_external_stats_cache_key,
                                )
                            )
                            if external_pilot_used and reuse_external_model_eval_stats
                            else rng
                        )
                        if paper_mfmc_cell:
                            qoi_robust_rows = _paper_mfmc_pilot_robustness_metrics(
                                pilot_hf=qoi_pilot_hf,
                                pilot_lfs=qoi_pilot_lfs,
                                pilot_sizes=robust_sizes,
                                repetitions=robust_reps,
                                rng=qoi_robustness_rng,
                                hf_cost=pilot_hf_cost,
                                lf_costs=pilot_lf_costs,
                            )
                        elif multi_lf_cell:
                            qoi_robust_rows = _multi_lf_pilot_robustness_metrics(
                                pilot_hf=qoi_pilot_hf,
                                pilot_lfs=qoi_pilot_lfs,
                                pilot_sizes=robust_sizes,
                                repetitions=robust_reps,
                                rng=qoi_robustness_rng,
                                hf_cost=pilot_hf_cost,
                                lf_cost=pilot_lf_cost,
                            )
                        else:
                            qoi_robust_rows = pilot_robustness_metrics(
                                pilot_hf=qoi_pilot_hf,
                                pilot_lf=qoi_pilot_lfs[lf_model_ids[0]],
                                pilot_sizes=robust_sizes,
                                repetitions=robust_reps,
                                rng=qoi_robustness_rng,
                                hf_cost=pilot_hf_cost,
                                lf_cost=pilot_lf_cost,
                            )
                        if external_pilot_used and reuse_external_model_eval_stats:
                            external_model_eval_robustness_cache[qoi_external_stats_cache_key] = qoi_robust_rows
                    for rr in qoi_robust_rows:
                        store.append_robustness(
                            {
                                "study_id": cell.study_id,
                                "cell_id": result_cell.cell_id(),
                                "mode": cell.mode,
                                "geometry_id": cell.geometry_id,
                                "geometry_class": geometry.get("geometry_class"),
                                "regime_id": cell.regime_id,
                                "active_sources": list(cell.active_source_blocks),
                                "qoi": result_qoi,
                                "hf_model_id": cell.hf_model_id,
                                "lf_model_id": cell.lf_model_id,
                                "repetition": cell.repetition,
                                **rr,
                            }
                        )

            qoi_prod_hf = np.asarray(prod_hf_res.values_by_qoi.get(result_qoi, []), dtype=float)
            qoi_prod_lf_full_by_id = {
                lf_model_id: np.asarray(prod_lf_full_results[lf_model_id].values_by_qoi.get(result_qoi, []), dtype=float)
                for lf_model_id in lf_model_ids
            }
            qoi_prod_lf_paired_by_id = {
                lf_model_id: (
                    np.asarray(prod_lf_pair_results[lf_model_id].values_by_qoi.get(result_qoi, []), dtype=float)
                    if not paper_mfmc_cell
                    else np.asarray(prod_lf_full_results[lf_model_id].values_by_qoi.get(result_qoi, []), dtype=float)[
                        : int(paper_sample_counts.get(lf_model_id, n_hf))
                    ]
                )
                for lf_model_id in lf_model_ids
            }
            qoi_reference = (
                float(np.nanmean(np.asarray(ref_res.values_by_qoi.get(result_qoi, []), dtype=float)))
                if ref_res is not None
                else float("nan")
            )

            if paper_mfmc_cell:
                metrics = compute_paper_mfmc_diagnostics(
                    qoi=result_qoi,
                    pilot_hf=qoi_pilot_hf,
                    pilot_lfs=qoi_pilot_lfs,
                    prod_hf=qoi_prod_hf,
                    prod_lf_full=qoi_prod_lf_full_by_id,
                    lf_sample_counts={
                        lf_model_id: int(paper_sample_counts.get(lf_model_id, n_hf))
                        for lf_model_id in lf_model_ids
                    },
                    hf_costs=hf_costs,
                    lf_costs_full=lf_costs_full_by_id,
                    reference=qoi_reference,
                    lf_order=paper_lf_order,
                )
            elif multi_lf_cell:
                metrics = compute_multi_lf_mfmc_diagnostics(
                    qoi=result_qoi,
                    pilot_hf=qoi_pilot_hf,
                    pilot_lfs=qoi_pilot_lfs,
                    prod_hf=qoi_prod_hf,
                    prod_lf_full=qoi_prod_lf_full_by_id,
                    prod_lf_paired=qoi_prod_lf_paired_by_id,
                    hf_costs=hf_costs,
                    lf_costs_full=lf_costs_full_by_id,
                    reference=qoi_reference,
                )
            else:
                metrics = compute_mfmc_diagnostics(
                    qoi=result_qoi,
                    pilot_hf=qoi_pilot_hf,
                    pilot_lf=qoi_pilot_lfs[lf_model_ids[0]],
                    prod_hf=qoi_prod_hf,
                    prod_lf_full=qoi_prod_lf_full_by_id[lf_model_ids[0]],
                    prod_lf_paired=qoi_prod_lf_paired_by_id[lf_model_ids[0]],
                    hf_costs=hf_costs,
                    lf_costs_full=lf_costs_full_by_id[lf_model_ids[0]],
                    reference=qoi_reference,
                )

            metrics.update(
                beta_stability_metrics(
                    pilot_hf=qoi_pilot_hf,
                    pilot_lf=qoi_pilot_lfs[lf_model_ids[0]],
                    repetitions=robust_reps,
                    rng=rng,
                )
            )
            pilot_corr_used = False
            if not paper_mfmc_cell:
                pilot_corr_used = _apply_external_pilot_robustness(
                    metrics,
                    qoi_external_robustness,
                    lf_model_ids=lf_model_ids,
                    prod_lf_full_by_id=qoi_prod_lf_full_by_id,
                    prod_lf_paired_by_id=qoi_prod_lf_paired_by_id,
                    reference=qoi_reference,
                )
            if not pilot_corr_used:
                pilot_corr_used = _use_pilot_correlation(metrics, qoi_robust_rows, cell.pilot_size)

            flags = statistical_flags(metrics)
            if result_qoi != cell.qoi:
                flags.append(f"batched_with_{cell.qoi}")
            if pilot_corr_used:
                flags.append("pilot_correlation_used")
            if qoi_external_robustness:
                flags.append("external_pilot_robustness")
            if partial_external_pilot_used and result_qoi == cell.qoi:
                flags.append("external_pilot_model_evaluations")
            if multi_lf_cell:
                flags.append("multi_lf")
            if paper_mfmc_cell:
                flags.append("paper_mfmc")
            row = _build_result_row(
                result_cell,
                geometry,
                regime,
                metrics=metrics,
                quantity_kind=qoi_registry.quantity_kind(result_qoi),
                qoi_expression=qoi_registry.expression(result_qoi),
                flags=flags,
            )
            store.append_result(row)
            rows_written.append(row)
            by_base_cell[base_key][result_qoi] = row

            direct: Dict[str, float] = {}
            for key in ("C_D", "C_D2", "C_L", "C_L2", "C_Y", "C_Y2", "C_Mz", "C_Mz2"):
                if key in by_base_cell[base_key]:
                    direct[key] = float(by_base_cell[base_key][key]["mfmc_estimate"])
            derived = derive_quantities(direct)
            for derived_name, derived_info in derived.items():
                anchor_key = derived_name.split("_", 1)[-1]
                anchor = by_base_cell[base_key].get(anchor_key)
                if anchor is None:
                    anchor = next(iter(by_base_cell[base_key].values()), None)
                if anchor is None:
                    continue
                drow = dict(anchor)
                drow["qoi"] = derived_name
                drow["cell_id"] = f"{cid}|derived={derived_name}"
                drow["quantity_kind"] = "derived"
                drow["qoi_expression"] = str(derived_info.get("expression", ""))
                drow["mfmc_estimate"] = float(derived_info.get("value", float("nan")))
                drow["hf_only_estimate"] = float("nan")
                drow["reference_estimate"] = float("nan")
                drow["realized_mfmc_error"] = float("nan")
                drow["realized_hf_error"] = float("nan")
                drow["flags"] = ["derived_quantity"]
                store.append_result(drow)

        n_executed += 1
        primary_row = rows_written[0] if rows_written else {}
        print(
            f"[done] budget {budget_index}/{budget_total} item {budget_item_index}/{budget_item_total} "
            f"qoi={eval_qoi_key} B={_fmt_num(cell.budget)} mfmc={_fmt_num(primary_row.get('mfmc_estimate'))} "
            f"executed={n_executed} skipped={n_skipped}",
            flush=True,
        )

    cache.flush()
    print(f"[campaign] evaluations complete executed={n_executed} skipped={n_skipped}; writing summaries", flush=True)

    if cfg.get("outputs", {}).get("write_parquet", False):
        store.write_optional_parquet(store.results_csv)

    table_paths = write_summary_tables(output_dir, store.results_csv, store.robustness_csv)

    plot_files: List[str] = []
    if cfg.get("outputs", {}).get("plots", True):
        plot_files = generate_plots(output_dir)

    predictive_csv = os.path.join(output_dir, "predictive_dataset.csv")
    if cfg.get("study", {}).get("mode") == "predictive_dataset_export":
        export_predictive_dataset(store.results_csv, predictive_csv, store.robustness_csv)

    summary = {
        "study_id": cfg.get("study", {}).get("id"),
        "output_dir": output_dir,
        "results_csv": store.results_csv,
        "model_evaluations_csv": store.model_evaluations_csv if os.path.exists(store.model_evaluations_csv) else None,
        "robustness_csv": store.robustness_csv,
        "config_snapshot": store.config_json,
        "summary_json": store.summary_json,
        "summary_tables": table_paths,
        "plot_files": plot_files,
        "predictive_dataset": predictive_csv if os.path.exists(predictive_csv) else None,
        "n_cells_total": len(cells),
        "n_cells_executed": n_executed,
        "n_cells_skipped": n_skipped,
        "pilots_only": pilots_only,
    }
    store.write_summary(summary)
    return summary


def postprocess_outputs(output_dir: str) -> Dict[str, Any]:
    results_csv = os.path.join(output_dir, "results_long.csv")
    robustness_csv = os.path.join(output_dir, "pilot_robustness.csv")

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results file not found: {results_csv}")

    table_paths = write_summary_tables(output_dir, results_csv, robustness_csv)
    plot_files = generate_plots(output_dir)
    return {
        "output_dir": output_dir,
        "summary_tables": table_paths,
        "plot_files": plot_files,
    }


def run_campaign_from_path(config_path: str, resume: bool = False, pilots_only: bool = False) -> Dict[str, Any]:
    cfg = load_and_validate(config_path)
    return run_campaign(cfg, resume=resume, pilots_only=pilots_only)
