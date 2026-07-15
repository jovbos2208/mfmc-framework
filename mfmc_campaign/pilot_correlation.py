from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .adapters import build_adapter_registry, make_request
from .config import load_and_validate, validate_or_raise
from .estimator import (
    beta_stability_metrics,
    compute_mfmc_diagnostics,
    pilot_robustness_metrics,
    statistical_flags,
)
from .experiments import _mode_specific_active_sources
from .output import ResultStore
from .qoi_registry import build_qoi_registry
from .reproducibility import derive_seed, get_run_fingerprint
from .sampling import InputModel, SamplingContext
from .types import EvaluationRequest, EvaluationResult


def _fingerprint_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _fingerprint_payload(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_payload(v) for v in value]
    if isinstance(value, np.ndarray):
        return _fingerprint_payload(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def _hash_payload(value: Any) -> str:
    payload = json.dumps(_fingerprint_payload(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_fingerprints(samples: Sequence[Dict[str, Any]]) -> List[str]:
    return [_hash_payload(sample) for sample in samples]


def _request_fingerprints(request: EvaluationRequest) -> List[str]:
    geometry_payload = {
        "geometry_id": request.geometry.geometry_id,
        "name": request.geometry.name,
        "characteristic_length": request.geometry.characteristic_length,
        "geometry_class": request.geometry.geometry_class,
        "tags": request.geometry.tags,
        "metadata": request.geometry.metadata,
    }
    regime_payload = {
        "regime_id": request.regime.regime_id,
        "label": request.regime.label,
        "descriptors": request.regime.descriptors,
        "metadata": request.regime.metadata,
    }
    common = {
        "study_id": request.study_id,
        "model_id": request.model_id,
        "fidelity": request.fidelity,
        "qois": list(request.qois),
        "geometry": geometry_payload,
        "regime": regime_payload,
        "active_source_blocks": sorted(request.active_source_blocks),
        "seed": int(request.seed),
        "metadata": request.metadata,
    }
    return [
        _hash_payload(
            {
                **common,
                "sample_id": request.sample_ids[idx] if idx < len(request.sample_ids) else "",
                "sample": sample,
            }
        )
        for idx, sample in enumerate(request.samples)
    ]


def _direct_qois(config: Dict[str, Any]) -> List[str]:
    return [str(q) for q in config.get("qois", {}).get("direct", []) if isinstance(q, str)]


def _find_geometry(config: Dict[str, Any], geometry_id: str) -> Dict[str, Any]:
    for geometry in config.get("geometries", []):
        gid = str(geometry.get("id", geometry.get("name", "geometry")))
        if gid == geometry_id:
            return geometry
    raise KeyError(f"Unknown geometry id '{geometry_id}'")


def _find_regime(config: Dict[str, Any], regime_id: str) -> Dict[str, Any]:
    for regime in config.get("regimes", []):
        rid = str(regime.get("id", regime.get("label", "regime")))
        if rid == regime_id:
            return regime
    raise KeyError(f"Unknown regime id '{regime_id}'")


def _build_metadata(config: Dict[str, Any], geometry: Dict[str, Any], geometry_id: str) -> Dict[str, Any]:
    meta = {
        "aos_deg": config.get("execution", {}).get("aos_deg", 0.0),
        "aoa_deg": config.get("execution", {}).get("aoa_deg", 0.0),
        "geometry_id": geometry.get("id", geometry.get("name", geometry_id)),
        "geometry_name": geometry.get("name", geometry.get("id", geometry_id)),
        "geometry_class": geometry.get("geometry_class"),
    }
    if isinstance(geometry.get("metadata"), dict):
        meta.update(geometry.get("metadata", {}))
    env_cfg = config.get("execution", {}).get("environment", {})
    if isinstance(env_cfg, dict):
        meta.update(env_cfg)
        if "model" in env_cfg and "environment_model" not in meta:
            meta["environment_model"] = env_cfg.get("model")
    return meta


def _build_batch_cell_id(
    *,
    study_id: str,
    mode: str,
    geometry_id: str,
    regime_id: str,
    active_sources: Sequence[str],
    hf_model_id: str,
    lf_model_id: str,
    repetition: int,
    pilot_size: int,
) -> str:
    sources = "+".join(sorted(active_sources)) if active_sources else "none"
    return (
        f"{study_id}|pilot_correlation|{mode}|{geometry_id}|{regime_id}|"
        f"{hf_model_id}|{lf_model_id}|src={sources}|rep={repetition}|pilot={pilot_size}"
    )


def _append_model_evaluation_rows(
    *,
    store: ResultStore,
    study_id: str,
    cell_id: str,
    mode: str,
    geometry_id: str,
    regime_id: str,
    active_sources: Sequence[str],
    qois: Sequence[str],
    model_id: str,
    fidelity: str,
    hf_model_id: str,
    lf_model_id: str,
    pilot_size: int,
    repetition: int,
    seed: int,
    sample_ids: Sequence[str],
    values_by_qoi: Dict[str, Sequence[float]],
    costs: Sequence[float],
    phase: str,
    sample_indices: Sequence[int] | None = None,
    sample_fingerprints: Sequence[str] | None = None,
    request_fingerprints: Sequence[str] | None = None,
) -> None:
    max_len = max([len(sample_ids), len(costs)] + [len(values_by_qoi.get(q, [])) for q in qois], default=0)
    output_indices = list(sample_indices) if sample_indices is not None else list(range(max_len))
    for qoi in qois:
        values = list(values_by_qoi.get(qoi, []))
        for idx in range(max_len):
            sample_index = output_indices[idx] if idx < len(output_indices) else idx
            store.append_model_evaluation(
                {
                    "study_id": study_id,
                    "cell_id": cell_id,
                    "phase": phase,
                    "mode": mode,
                    "geometry_id": geometry_id,
                    "regime_id": regime_id,
                    "active_sources": list(active_sources),
                    "qoi": qoi,
                    "model_id": model_id,
                    "fidelity": fidelity,
                    "hf_model_id": hf_model_id,
                    "lf_model_id": lf_model_id,
                    "pilot_size": pilot_size,
                    "budget": pilot_size,
                    "repetition": repetition,
                    "seed": seed,
                    "sample_id": sample_ids[idx] if idx < len(sample_ids) else "",
                    "sample_index": sample_index,
                    "sample_fingerprint": sample_fingerprints[idx]
                    if sample_fingerprints is not None and idx < len(sample_fingerprints)
                    else "",
                    "request_fingerprint": request_fingerprints[idx]
                    if request_fingerprints is not None and idx < len(request_fingerprints)
                    else "",
                    "value": values[idx] if idx < len(values) else float("nan"),
                    "cost": costs[idx] if idx < len(costs) else float("nan"),
                }
            )


def _existing_model_evaluation_result(
    *,
    path: str,
    cell_id: str | None,
    phase: str,
    model_id: str,
    qois: Sequence[str],
    seed: int,
    sample_ids: Sequence[str],
    study_id: str | None = None,
    mode: str | None = None,
    geometry_id: str | None = None,
    regime_id: str | None = None,
    active_sources: Sequence[str] | None = None,
    pilot_size: int | None = None,
    repetition: int | None = None,
    expected_sample_fingerprints: Sequence[str] | None = None,
    expected_request_fingerprints: Sequence[str] | None = None,
) -> Tuple[EvaluationResult, List[int]]:
    values_by_qoi = {str(qoi): [float("nan")] * len(sample_ids) for qoi in qois}
    costs = [float("nan")] * len(sample_ids)
    active_sources_key = "+".join(active_sources or [])
    active_sources_sorted_key = "+".join(sorted(active_sources or []))

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if cell_id is not None and row.get("cell_id") != cell_id:
                    continue
                if row.get("phase") != phase or row.get("model_id") != model_id:
                    continue
                if study_id is not None and row.get("study_id") != study_id:
                    continue
                if mode is not None and row.get("mode") != mode:
                    continue
                if geometry_id is not None and row.get("geometry_id") != geometry_id:
                    continue
                if regime_id is not None and row.get("regime_id") != regime_id:
                    continue
                if active_sources is not None:
                    row_sources = row.get("active_sources", "")
                    if row_sources not in {active_sources_key, active_sources_sorted_key}:
                        continue
                try:
                    if int(float(row.get("seed", "nan"))) != int(seed):
                        continue
                    sample_index = int(float(row.get("sample_index", "nan")))
                    if pilot_size is not None and int(float(row.get("pilot_size", "nan"))) != int(pilot_size):
                        continue
                    if repetition is not None and int(float(row.get("repetition", "nan"))) != int(repetition):
                        continue
                except Exception:
                    continue
                if sample_index < 0 or sample_index >= len(sample_ids):
                    continue
                if row.get("sample_id") != sample_ids[sample_index]:
                    continue
                if expected_sample_fingerprints is not None:
                    if row.get("sample_fingerprint") != expected_sample_fingerprints[sample_index]:
                        continue
                if expected_request_fingerprints is not None:
                    if row.get("request_fingerprint") != expected_request_fingerprints[sample_index]:
                        continue

                qoi = str(row.get("qoi", ""))
                if qoi not in values_by_qoi:
                    continue
                try:
                    value = float(row.get("value", "nan"))
                    cost = float(row.get("cost", "nan"))
                except Exception:
                    continue
                if np.isfinite(value):
                    values_by_qoi[qoi][sample_index] = value
                if np.isfinite(cost):
                    costs[sample_index] = cost

    missing = [
        idx
        for idx in range(len(sample_ids))
        if not np.isfinite(costs[idx])
        or any(not np.isfinite(values_by_qoi[str(qoi)][idx]) for qoi in qois)
    ]
    return EvaluationResult(values_by_qoi=values_by_qoi, costs=costs, sample_ids=list(sample_ids)), missing


def _merge_evaluation_subset(
    base: EvaluationResult,
    subset: EvaluationResult,
    missing_indices: Sequence[int],
    qois: Sequence[str],
) -> EvaluationResult:
    pos_by_sample = {sid: pos for pos, sid in enumerate(subset.sample_ids)}
    for full_pos in missing_indices:
        sample_id = base.sample_ids[full_pos]
        subset_pos = pos_by_sample.get(sample_id)
        if subset_pos is None:
            continue
        if subset_pos < len(subset.costs):
            base.costs[full_pos] = float(subset.costs[subset_pos])
        for qoi in qois:
            vals = list(subset.values_by_qoi.get(str(qoi), []))
            if subset_pos < len(vals):
                base.values_by_qoi[str(qoi)][full_pos] = float(vals[subset_pos])
    return base


def _evaluate_with_model_evaluation_reuse(
    *,
    store: ResultStore,
    adapter: Any,
    request: EvaluationRequest,
    reuse_existing: bool,
    study_id: str,
    mode: str,
    geometry_id: str,
    regime_id: str,
    active_sources: Sequence[str],
    hf_model_id: str,
    lf_model_id: str,
    pilot_size: int,
    repetition: int,
    phase: str,
    write_model_evaluations: bool,
    reuse_across_cell_ids: bool = False,
    match_request_fingerprint: bool = True,
) -> EvaluationResult:
    sample_fingerprints = _sample_fingerprints(request.samples)
    request_fingerprints = _request_fingerprints(request)
    if reuse_existing:
        result, missing_indices = _existing_model_evaluation_result(
            path=store.model_evaluations_csv,
            cell_id=None if reuse_across_cell_ids else request.cell_id,
            phase=phase,
            model_id=request.model_id,
            qois=request.qois,
            seed=request.seed,
            sample_ids=request.sample_ids,
            study_id=study_id if reuse_across_cell_ids else None,
            mode=mode if reuse_across_cell_ids else None,
            geometry_id=geometry_id if reuse_across_cell_ids else None,
            regime_id=regime_id if reuse_across_cell_ids else None,
            active_sources=active_sources if reuse_across_cell_ids else None,
            pilot_size=pilot_size if reuse_across_cell_ids else None,
            repetition=repetition if reuse_across_cell_ids else None,
            expected_sample_fingerprints=sample_fingerprints,
            expected_request_fingerprints=request_fingerprints if match_request_fingerprint else None,
        )
    else:
        result = EvaluationResult(
            values_by_qoi={str(qoi): [float("nan")] * len(request.sample_ids) for qoi in request.qois},
            costs=[float("nan")] * len(request.sample_ids),
            sample_ids=list(request.sample_ids),
        )
        missing_indices = list(range(len(request.sample_ids)))

    if missing_indices:
        subset_request = replace(
            request,
            sample_ids=[request.sample_ids[idx] for idx in missing_indices],
            samples=[request.samples[idx] for idx in missing_indices],
        )
        subset_result = adapter.evaluate(subset_request)
        result = _merge_evaluation_subset(result, subset_result, missing_indices, request.qois)
        if write_model_evaluations:
            _append_model_evaluation_rows(
                store=store,
                study_id=study_id,
                cell_id=request.cell_id,
                mode=mode,
                geometry_id=geometry_id,
                regime_id=regime_id,
                active_sources=active_sources,
                qois=request.qois,
                model_id=request.model_id,
                fidelity=request.fidelity,
                hf_model_id=hf_model_id,
                lf_model_id=lf_model_id,
                pilot_size=pilot_size,
                repetition=repetition,
                seed=request.seed,
                sample_ids=subset_result.sample_ids,
                values_by_qoi=subset_result.values_by_qoi,
                costs=subset_result.costs,
                phase=phase,
                sample_indices=missing_indices,
                sample_fingerprints=[sample_fingerprints[idx] for idx in missing_indices],
                request_fingerprints=[request_fingerprints[idx] for idx in missing_indices],
            )

    return result


def _model_evaluations_complete(
    *,
    store: ResultStore,
    cell_id: str,
    hf_request: EvaluationRequest,
    lf_request: EvaluationRequest,
    study_id: str,
    mode: str,
    geometry_id: str,
    regime_id: str,
    active_sources: Sequence[str],
    pilot_size: int,
    repetition: int,
    match_request_fingerprint: bool = True,
) -> bool:
    hf_sample_fingerprints = _sample_fingerprints(hf_request.samples)
    hf_request_fingerprints = _request_fingerprints(hf_request)
    lf_sample_fingerprints = _sample_fingerprints(lf_request.samples)
    lf_request_fingerprints = _request_fingerprints(lf_request)
    _, hf_missing = _existing_model_evaluation_result(
        path=store.model_evaluations_csv,
        cell_id=None,
        phase="pilot_hf",
        model_id=hf_request.model_id,
        qois=hf_request.qois,
        seed=hf_request.seed,
        sample_ids=hf_request.sample_ids,
        study_id=study_id,
        mode=mode,
        geometry_id=geometry_id,
        regime_id=regime_id,
        active_sources=active_sources,
        pilot_size=pilot_size,
        repetition=repetition,
        expected_sample_fingerprints=hf_sample_fingerprints,
        expected_request_fingerprints=hf_request_fingerprints if match_request_fingerprint else None,
    )
    _, lf_missing = _existing_model_evaluation_result(
        path=store.model_evaluations_csv,
        cell_id=cell_id,
        phase="pilot_lf",
        model_id=lf_request.model_id,
        qois=lf_request.qois,
        seed=lf_request.seed,
        sample_ids=lf_request.sample_ids,
        expected_sample_fingerprints=lf_sample_fingerprints,
        expected_request_fingerprints=lf_request_fingerprints if match_request_fingerprint else None,
    )
    return not hf_missing and not lf_missing


def _build_result_row(
    *,
    study_id: str,
    cell_id: str,
    mode: str,
    geometry: Dict[str, Any],
    geometry_id: str,
    regime: Dict[str, Any],
    regime_id: str,
    active_sources: Sequence[str],
    qoi: str,
    quantity_kind: str,
    qoi_expression: str,
    hf_model_id: str,
    lf_model_id: str,
    pilot_size: int,
    repetition: int,
    seed: int,
    metrics: Dict[str, Any],
    flags: Sequence[str],
) -> Dict[str, Any]:
    regime_desc = regime.get("descriptors", {})
    return {
        "study_id": study_id,
        "cell_id": cell_id,
        "mode": mode,
        "geometry_id": geometry_id,
        "geometry_name": geometry.get("name", geometry_id),
        "geometry_class": geometry.get("geometry_class"),
        "geometry_characteristic_length": geometry.get("characteristic_length"),
        "regime_id": regime_id,
        "regime_label": regime.get("label", regime_id),
        "active_sources": list(active_sources),
        "qoi": qoi,
        "quantity_kind": quantity_kind,
        "qoi_expression": qoi_expression,
        "hf_model_id": hf_model_id,
        "lf_model_id": lf_model_id,
        "pilot_size": pilot_size,
        "budget": float(pilot_size),
        "repetition": repetition,
        "seed": seed,
        "flags": list(flags),
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


def _lf_model_ids(config: Dict[str, Any]) -> List[str]:
    return [str(cfg.get("id", "lf")) for cfg in config.get("models", {}).get("lf", [])]


def run_pilot_correlation(config: Dict[str, Any], resume: bool = False) -> Dict[str, Any]:
    cfg = validate_or_raise(config)
    qoi_registry = build_qoi_registry(cfg)
    qois = _direct_qois(cfg)
    if not qois:
        raise ValueError("No direct QoIs configured for pilot-correlation run")

    output_dir = str(cfg.get("outputs", {}).get("dir", "campaign_outputs/pilot_correlation/default"))
    store = ResultStore(output_dir)
    should_resume = bool(resume or cfg.get("execution", {}).get("resume", False))
    if not should_resume:
        store.reset_outputs(keep_cache=True)
    if cfg.get("outputs", {}).get("write_config_snapshot", True):
        store.write_config_snapshot(cfg, get_run_fingerprint())

    done = store.load_completed_cell_ids() if should_resume else set()
    registry = build_adapter_registry(cfg)
    input_model = InputModel(
        cfg.get("variables", []),
        cfg.get("sampling", {}),
        regime_label_map=cfg.get("regime_label_map", {}),
    )

    study_id = str(cfg.get("study", {}).get("id", "pilot_correlation"))
    mode = str(cfg.get("study", {}).get("mode", "source_isolation"))
    global_seed = int(cfg.get("seeds", {}).get("global", 12345))
    pilot_size = int(cfg.get("pilot", {}).get("size", 32))
    robust_sizes = [int(v) for v in cfg.get("pilot", {}).get("sizes", [pilot_size])]
    robust_reps = int(cfg.get("pilot", {}).get("robustness_repetitions", 20))
    hf_model_id = str(cfg.get("models", {}).get("hf", {}).get("id", "hf"))
    match_request_fingerprint = bool(cfg.get("pilot", {}).get("resume_match_request_fingerprint", True))
    if should_resume and not match_request_fingerprint:
        print(
            "[pilot-correlation] resume will match existing model_evaluations by sample_fingerprint "
            "without requiring request_fingerprint equality.",
            flush=True,
        )

    n_batches = 0
    n_skipped = 0

    for geometry in cfg.get("geometries", []):
        geometry_id = str(geometry.get("id", geometry.get("name", "geometry")))
        geometry = _find_geometry(cfg, geometry_id)
        meta = _build_metadata(cfg, geometry, geometry_id)

        for regime in cfg.get("regimes", []):
            regime_id = str(regime.get("id", regime.get("label", "regime")))
            regime = _find_regime(cfg, regime_id)

            for active_sources in _mode_specific_active_sources(cfg):
                active_sources = list(active_sources)
                active_sources_key = "+".join(sorted(active_sources))

                for repetition in range(int(cfg.get("repetitions", 1))):
                    batch_seed = derive_seed(
                        global_seed,
                        study_id,
                        "pilot_correlation",
                        mode,
                        geometry_id,
                        regime_id,
                        active_sources_key,
                        repetition,
                        pilot_size,
                    )
                    rng = np.random.default_rng(batch_seed)
                    context = SamplingContext(regime_id=regime_id, active_source_blocks=active_sources)
                    samples = input_model.sample(pilot_size, context, rng)
                    sample_ids = [f"pilot_{i}" for i in range(pilot_size)]

                    hf_seed = derive_seed(batch_seed, hf_model_id, "hf_eval")
                    hf_request = make_request(
                        study_id=study_id,
                        cell_id=f"{study_id}|hf_shared|{geometry_id}|{regime_id}|{active_sources_key or 'none'}|rep={repetition}",
                        model_id=hf_model_id,
                        fidelity="hf",
                        qois=qois,
                        geometry=geometry,
                        regime=regime,
                        active_source_blocks=active_sources,
                        sample_ids=sample_ids,
                        samples=samples,
                        seed=hf_seed,
                        metadata=meta,
                    )
                    hf_result_shared = None
                    if not should_resume:
                        hf_result_shared = registry.get(hf_model_id).evaluate(hf_request)

                    for lf_model_id in _lf_model_ids(cfg):
                        cell_id = _build_batch_cell_id(
                            study_id=study_id,
                            mode=mode,
                            geometry_id=geometry_id,
                            regime_id=regime_id,
                            active_sources=active_sources,
                            hf_model_id=hf_model_id,
                            lf_model_id=lf_model_id,
                            repetition=repetition,
                            pilot_size=pilot_size,
                        )

                        lf_seed = derive_seed(batch_seed, lf_model_id, "lf_eval")
                        lf_request = make_request(
                            study_id=study_id,
                            cell_id=cell_id,
                            model_id=lf_model_id,
                            fidelity="lf",
                            qois=qois,
                            geometry=geometry,
                            regime=regime,
                            active_source_blocks=active_sources,
                            sample_ids=sample_ids,
                            samples=samples,
                            seed=lf_seed,
                            metadata=meta,
                        )
                        cell_was_done = cell_id in done
                        if cell_was_done:
                            if _model_evaluations_complete(
                                store=store,
                                cell_id=cell_id,
                                hf_request=hf_request,
                                lf_request=lf_request,
                                study_id=study_id,
                                mode=mode,
                                geometry_id=geometry_id,
                                regime_id=regime_id,
                                active_sources=active_sources,
                                pilot_size=pilot_size,
                                repetition=repetition,
                                match_request_fingerprint=match_request_fingerprint,
                            ):
                                n_skipped += 1
                                continue
                            store.remove_cell_outputs(cell_id)
                            done.discard(cell_id)
                            cell_was_done = False

                        if should_resume:
                            hf_result = _evaluate_with_model_evaluation_reuse(
                                store=store,
                                adapter=registry.get(hf_model_id),
                                request=replace(hf_request, cell_id=cell_id),
                                reuse_existing=True,
                                study_id=study_id,
                                mode=mode,
                                geometry_id=geometry_id,
                                regime_id=regime_id,
                                active_sources=active_sources,
                                hf_model_id=hf_model_id,
                                lf_model_id=lf_model_id,
                                pilot_size=pilot_size,
                                repetition=repetition,
                                phase="pilot_hf",
                                write_model_evaluations=bool(cfg.get("outputs", {}).get("write_model_evaluations", True)),
                                reuse_across_cell_ids=True,
                                match_request_fingerprint=match_request_fingerprint,
                            )
                        else:
                            hf_result = hf_result_shared
                            if hf_result is None:
                                raise RuntimeError("Internal error: missing shared HF pilot result")
                            if cfg.get("outputs", {}).get("write_model_evaluations", True):
                                _append_model_evaluation_rows(
                                    store=store,
                                    study_id=study_id,
                                    cell_id=cell_id,
                                    mode=mode,
                                    geometry_id=geometry_id,
                                    regime_id=regime_id,
                                    active_sources=active_sources,
                                    qois=qois,
                                    model_id=hf_model_id,
                                    fidelity="hf",
                                    hf_model_id=hf_model_id,
                                    lf_model_id=lf_model_id,
                                    pilot_size=pilot_size,
                                    repetition=repetition,
                                    seed=hf_seed,
                                    sample_ids=hf_result.sample_ids,
                                    values_by_qoi=hf_result.values_by_qoi,
                                    costs=hf_result.costs,
                                    phase="pilot_hf",
                                    sample_fingerprints=_sample_fingerprints(hf_request.samples),
                                    request_fingerprints=_request_fingerprints(hf_request),
                                )

                        lf_result = _evaluate_with_model_evaluation_reuse(
                            store=store,
                            adapter=registry.get(lf_model_id),
                            request=lf_request,
                            reuse_existing=should_resume,
                            study_id=study_id,
                            mode=mode,
                            geometry_id=geometry_id,
                            regime_id=regime_id,
                            active_sources=active_sources,
                            hf_model_id=hf_model_id,
                            lf_model_id=lf_model_id,
                            pilot_size=pilot_size,
                            repetition=repetition,
                            phase="pilot_lf",
                            write_model_evaluations=bool(cfg.get("outputs", {}).get("write_model_evaluations", True)),
                            match_request_fingerprint=match_request_fingerprint,
                        )

                        hf_costs = np.asarray(hf_result.costs, dtype=float)
                        lf_costs = np.asarray(lf_result.costs, dtype=float)
                        hf_cost_mean = float(np.nanmean(hf_costs)) if hf_costs.size else float("nan")
                        lf_cost_mean = float(np.nanmean(lf_costs)) if lf_costs.size else float("nan")

                        for qoi in qois:
                            pilot_hf = np.asarray(hf_result.values_by_qoi.get(qoi, []), dtype=float)
                            pilot_lf = np.asarray(lf_result.values_by_qoi.get(qoi, []), dtype=float)

                            robust_rng = np.random.default_rng(derive_seed(batch_seed, lf_model_id, qoi, "robustness"))
                            robust_rows = pilot_robustness_metrics(
                                pilot_hf=pilot_hf,
                                pilot_lf=pilot_lf,
                                pilot_sizes=robust_sizes,
                                repetitions=robust_reps,
                                rng=robust_rng,
                                hf_cost=hf_cost_mean,
                                lf_cost=lf_cost_mean,
                            )
                            for robust in robust_rows:
                                store.append_robustness(
                                    {
                                        "study_id": study_id,
                                        "cell_id": cell_id,
                                        "mode": mode,
                                        "geometry_id": geometry_id,
                                        "geometry_class": geometry.get("geometry_class"),
                                        "regime_id": regime_id,
                                        "active_sources": list(active_sources),
                                        "qoi": qoi,
                                        "hf_model_id": hf_model_id,
                                        "lf_model_id": lf_model_id,
                                        "repetition": repetition,
                                        **robust,
                                    }
                                )

                            beta_rng = np.random.default_rng(derive_seed(batch_seed, lf_model_id, qoi, "beta_stability"))
                            metrics = compute_mfmc_diagnostics(
                                qoi=qoi,
                                pilot_hf=pilot_hf,
                                pilot_lf=pilot_lf,
                                prod_hf=pilot_hf,
                                prod_lf_full=pilot_lf,
                                prod_lf_paired=pilot_lf,
                                hf_costs=hf_costs,
                                lf_costs_full=lf_costs,
                                reference=float("nan"),
                            )
                            metrics.update(
                                beta_stability_metrics(
                                    pilot_hf=pilot_hf,
                                    pilot_lf=pilot_lf,
                                    repetitions=robust_reps,
                                    rng=beta_rng,
                                )
                            )
                            flags = statistical_flags(metrics) + ["pilot_only", "shared_multi_qoi_batch"]
                            store.append_result(
                                _build_result_row(
                                    study_id=study_id,
                                    cell_id=cell_id,
                                    mode=mode,
                                    geometry=geometry,
                                    geometry_id=geometry_id,
                                    regime=regime,
                                    regime_id=regime_id,
                                    active_sources=active_sources,
                                    qoi=qoi,
                                    quantity_kind=qoi_registry.quantity_kind(qoi),
                                    qoi_expression=qoi_registry.expression(qoi),
                                    hf_model_id=hf_model_id,
                                    lf_model_id=lf_model_id,
                                    pilot_size=pilot_size,
                                    repetition=repetition,
                                    seed=batch_seed,
                                    metrics=metrics,
                                    flags=flags,
                                )
                            )

                        n_batches += 1

    summary = {
        "output_dir": output_dir,
        "study_id": study_id,
        "mode": mode,
        "pilot_size": pilot_size,
        "qois": qois,
        "hf_model_id": hf_model_id,
        "lf_model_ids": _lf_model_ids(cfg),
        "executed_batches": n_batches,
        "skipped_batches": n_skipped,
    }
    store.write_summary(summary)
    return summary


def run_pilot_correlation_from_path(config_path: str, resume: bool = False) -> Dict[str, Any]:
    cfg = load_and_validate(config_path)
    return run_pilot_correlation(cfg, resume=resume)
