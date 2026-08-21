from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .geometry_design import VARIABLES
from .sparse_pce import SparsePCEModel, fit_sparse_pce, regression_metrics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _numeric_uncertainty_columns(samples: Mapping[str, Mapping[str, Any]]) -> list[str]:
    names = sorted({name for row in samples.values() for name in row})
    selected: list[str] = []
    for name in names:
        try:
            values = np.asarray([float(row[name]) for row in samples.values()], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)):
            selected.append(name)
    return selected


def _geometry_inputs(payload: Mapping[str, Any], geometry_id: str) -> list[float]:
    design = payload["geometries"][geometry_id]["design"]
    return [float(design[f"normalized_{name}"]) for name in VARIABLES]


def _build_inputs(
    payload: Mapping[str, Any], geometry_id: str, sample_id: str, uncertainty_names: Sequence[str]
) -> list[float]:
    sample = payload["uncertainty_samples"][sample_id]
    return _geometry_inputs(payload, geometry_id) + [float(sample[name]) for name in uncertainty_names]


def _evaluation_lookup(payload: Mapping[str, Any], model_id: str) -> Dict[tuple[str, str], Mapping[str, Any]]:
    lookup: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload["evaluations"]:
        if row["model_id"] != model_id:
            continue
        key = (str(row["geometry_id"]), str(row["canonical_sample_id"]))
        if key in lookup:
            raise ValueError(f"Duplicate {model_id} evaluation for {key}")
        lookup[key] = row
    return lookup


def _common_sample_ids(
    payload: Mapping[str, Any], model_id: str, geometry_ids: Sequence[str]
) -> list[str]:
    lookup = _evaluation_lookup(payload, model_id)
    common: set[str] | None = None
    for geometry_id in geometry_ids:
        available = {sample_id for candidate_geometry, sample_id in lookup if candidate_geometry == geometry_id}
        common = available if common is None else common & available
    return sorted(common or set())


def _balanced_training_ids(
    payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
    geometry_ids: Sequence[str],
    *,
    model_id: str,
    count: int | None,
) -> list[str] | None:
    if count is None:
        return None
    if count < 1:
        raise ValueError(f"Balanced {model_id} count must be positive")
    reference_geometry_ids = [str(value) for value in reference_payload["selected_geometry_ids"]]
    common = _common_sample_ids(reference_payload, model_id, reference_geometry_ids)
    if len(common) < count:
        raise ValueError(
            f"Balance reference has only {len(common)} common {model_id} samples; {count} requested"
        )
    selected = common[:count]
    available = _evaluation_lookup(payload, model_id)
    missing = [
        (geometry_id, sample_id)
        for geometry_id in geometry_ids
        for sample_id in selected
        if (geometry_id, sample_id) not in available
    ]
    if missing:
        raise ValueError(
            f"Bundle is missing {len(missing)} balanced {model_id} evaluations; first missing pair: {missing[0]}"
        )
    return selected


def _fit_model(
    x: np.ndarray,
    y: np.ndarray,
    input_names: Sequence[str],
    groups: Sequence[str],
    *,
    degree: int,
    q_norm: float,
    max_interaction: int,
) -> SparsePCEModel:
    return fit_sparse_pce(
        x,
        y,
        input_names,
        groups=groups,
        degree=degree,
        q_norm=q_norm,
        max_interaction=max_interaction,
        cv_folds=min(5, len(set(groups))),
    )


def _robust_metrics(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean_drag": float(np.mean(values)),
        "std_drag": float(np.std(values, ddof=1)),
        "q95_drag": float(np.quantile(values, 0.95)),
    }


def _scaled_rows(x: np.ndarray) -> np.ndarray:
    scale = np.ptp(x, axis=0)
    active = scale > 1.0e-14
    if not np.any(active):
        return np.zeros((len(x), 1), dtype=float)
    return (x[:, active] - np.min(x[:, active], axis=0)) / scale[active]


def _maximin_candidates(
    candidates: np.ndarray,
    candidate_ids: Sequence[str],
    anchors: np.ndarray,
    count: int,
) -> list[tuple[str, float]]:
    if count <= 0:
        return []
    combined = _scaled_rows(np.vstack([candidates, anchors]))
    candidate_scaled = combined[: len(candidates)]
    anchor_scaled = combined[len(candidates) :]
    selected: list[int] = []
    result: list[tuple[str, float]] = []
    for _ in range(min(count, len(candidates))):
        reference = anchor_scaled if not selected else np.vstack([anchor_scaled, candidate_scaled[selected]])
        distances = np.min(np.linalg.norm(candidate_scaled[:, None, :] - reference[None, :, :], axis=2), axis=1)
        distances[selected] = -np.inf
        index = max(range(len(candidates)), key=lambda item: (float(distances[item]), candidate_ids[item]))
        selected.append(index)
        result.append((str(candidate_ids[index]), float(distances[index])))
    return result


def fit_geometry_multifidelity_surrogate(
    bundle_json: str | Path,
    output_dir: str | Path,
    *,
    degree: int = 2,
    q_norm: float = 0.75,
    max_interaction: int = 2,
    target_hf_per_geometry: int = 10,
    acquisition_geometry_count: int = 3,
    minimum_mf_relative_improvement: float = 0.01,
    target_geometry_rmse: float = 1.0e-4,
    training_lf_per_geometry: int | None = None,
    training_hf_per_geometry: int | None = None,
    balance_reference_bundle_json: str | Path | None = None,
) -> Dict[str, Any]:
    """Fit a DSMC-target WP5 surrogate and propose nested additional HF states.

    Geometry-held-out folds are used for the honest generalization diagnostic.
    Optional balanced training uses identical common-random-number states for every
    geometry and every learning-curve stage. Robust metrics still use every TPMC
    evaluation available for each geometry.
    """
    if not 0.0 <= minimum_mf_relative_improvement < 1.0:
        raise ValueError("minimum_mf_relative_improvement must be in [0, 1)")
    if target_geometry_rmse <= 0.0:
        raise ValueError("target_geometry_rmse must be positive")
    source = Path(bundle_json).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("reference_area_convention") != "canonical_manifest_area":
        raise ValueError("WP5 fitting requires canonical_manifest_area")
    qoi = str(payload.get("qoi", "drag_area_m2"))
    if qoi != "drag_area_m2":
        raise ValueError(f"Expected drag_area_m2 bundle, received {qoi}")

    geometry_ids = [str(value) for value in payload["selected_geometry_ids"]]
    uncertainty_names = _numeric_uncertainty_columns(payload["uncertainty_samples"])
    input_names = [f"geometry__{name}" for name in VARIABLES] + [f"input__{name}" for name in uncertainty_names]
    hf_all = _evaluation_lookup(payload, "PICLas_DSMC")
    lf_all = _evaluation_lookup(payload, "PICLas_TPMC")
    if not hf_all or not lf_all:
        raise ValueError("Bundle must contain PICLas_DSMC and PICLas_TPMC evaluations")

    reference_source = Path(balance_reference_bundle_json).resolve() if balance_reference_bundle_json else source
    reference_payload = (
        json.loads(reference_source.read_text(encoding="utf-8"))
        if reference_source != source
        else payload
    )
    selected_lf_ids = _balanced_training_ids(
        payload, reference_payload, geometry_ids,
        model_id="PICLas_TPMC", count=training_lf_per_geometry,
    )
    selected_hf_ids = _balanced_training_ids(
        payload, reference_payload, geometry_ids,
        model_id="PICLas_DSMC", count=training_hf_per_geometry,
    )
    lf = {
        key: row for key, row in lf_all.items()
        if key[0] in geometry_ids and (selected_lf_ids is None or key[1] in selected_lf_ids)
    }
    hf = {
        key: row for key, row in hf_all.items()
        if key[0] in geometry_ids and (selected_hf_ids is None or key[1] in selected_hf_ids)
    }

    lf_keys = sorted(lf)
    lf_x = np.asarray([_build_inputs(payload, *key, uncertainty_names) for key in lf_keys], dtype=float)
    lf_y = np.asarray([float(lf[key][qoi]) for key in lf_keys], dtype=float)
    lf_groups = np.asarray([key[0] for key in lf_keys], dtype=str)
    hf_keys = sorted(hf)
    missing_pairs = [key for key in hf_keys if key not in lf_all]
    if missing_pairs:
        raise ValueError(f"TPMC is missing {len(missing_pairs)} DSMC pairs")
    hf_x = np.asarray([_build_inputs(payload, *key, uncertainty_names) for key in hf_keys], dtype=float)
    hf_y = np.asarray([float(hf[key][qoi]) for key in hf_keys], dtype=float)
    paired_lf_y = np.asarray([float(lf_all[key][qoi]) for key in hf_keys], dtype=float)
    hf_groups = np.asarray([key[0] for key in hf_keys], dtype=str)
    residual_y = hf_y - paired_lf_y

    cv_rows: list[Dict[str, Any]] = []
    for held_out in geometry_ids:
        hf_train = hf_groups != held_out
        hf_test = ~hf_train
        lf_train = lf_groups != held_out
        if np.count_nonzero(hf_test) == 0:
            continue
        lf_model = _fit_model(
            lf_x[lf_train], lf_y[lf_train], input_names, lf_groups[lf_train],
            degree=degree, q_norm=q_norm, max_interaction=max_interaction,
        )
        delta_model = _fit_model(
            hf_x[hf_train], residual_y[hf_train], input_names, hf_groups[hf_train],
            degree=degree, q_norm=q_norm, max_interaction=max_interaction,
        )
        predictions = {
            "observed_tpmc": paired_lf_y[hf_test],
            "lf_pce": lf_model.predict(hf_x[hf_test]),
            "mf_pce": lf_model.predict(hf_x[hf_test]) + delta_model.predict(hf_x[hf_test]),
            "observed_tpmc_plus_delta": paired_lf_y[hf_test] + delta_model.predict(hf_x[hf_test]),
        }
        for method, prediction in predictions.items():
            cv_rows.append({"held_out_geometry_id": held_out, "method": method, **regression_metrics(hf_y[hf_test], prediction)})

    aggregate_cv = []
    for method in sorted({str(row["method"]) for row in cv_rows}):
        rows = [row for row in cv_rows if row["method"] == method]
        aggregate_cv.append({
            "method": method,
            "mean_geometry_rmse": float(np.mean([float(row["rmse"]) for row in rows])),
            "median_geometry_rmse": float(np.median([float(row["rmse"]) for row in rows])),
        })
    aggregate_by_method = {row["method"]: row for row in aggregate_cv}
    lf_cv_rmse = float(aggregate_by_method["lf_pce"]["mean_geometry_rmse"])
    mf_cv_rmse = float(aggregate_by_method["mf_pce"]["mean_geometry_rmse"])
    relative_improvement = (lf_cv_rmse - mf_cv_rmse) / lf_cv_rmse
    selected_surrogate = (
        "mf_pce" if relative_improvement >= float(minimum_mf_relative_improvement) else "lf_pce"
    )
    correction_applied = selected_surrogate == "mf_pce"

    lf_model = _fit_model(
        lf_x, lf_y, input_names, lf_groups,
        degree=degree, q_norm=q_norm, max_interaction=max_interaction,
    )
    delta_model = _fit_model(
        hf_x, residual_y, input_names, hf_groups,
        degree=degree, q_norm=q_norm, max_interaction=max_interaction,
    )
    target = Path(output_dir).resolve()
    model_dir = target / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    lf_model.write_json(str(model_dir / "tpmc_pce.json"))
    delta_model.write_json(str(model_dir / "dsmc_minus_tpmc_pce.json"))

    metric_rows: list[Dict[str, Any]] = []
    full_inputs_by_geometry: Dict[str, np.ndarray] = {}
    all_lf_keys = sorted(key for key in lf_all if key[0] in geometry_ids)
    all_hf_keys = sorted(key for key in hf_all if key[0] in geometry_ids)
    for geometry_id in geometry_ids:
        keys = [key for key in all_lf_keys if key[0] == geometry_id]
        x = np.asarray([_build_inputs(payload, *key, uncertainty_names) for key in keys], dtype=float)
        selected_values = np.asarray([float(lf_all[key][qoi]) for key in keys])
        if correction_applied:
            selected_values = selected_values + delta_model.predict(x)
        full_inputs_by_geometry[geometry_id] = x
        metric_rows.append({
            "geometry_id": geometry_id,
            "n_tpmc": len(keys),
            "n_dsmc": sum(key[0] == geometry_id for key in all_hf_keys),
            "selected_surrogate": selected_surrogate,
            "dsmc_discrepancy_applied": correction_applied,
            **_robust_metrics(selected_values),
        })
    metric_rows.sort(key=lambda row: (float(row["q95_drag"]), float(row["mean_drag"]), row["geometry_id"]))
    baseline = next((row for row in metric_rows if row["geometry_id"].endswith("_000")), metric_rows[0])
    for row in metric_rows:
        row["delta_mean_vs_baseline"] = float(row["mean_drag"]) - float(baseline["mean_drag"])
        row["delta_q95_vs_baseline"] = float(row["q95_drag"]) - float(baseline["q95_drag"])

    acquisition_geometries = [baseline["geometry_id"]]
    for row in metric_rows:
        if row["geometry_id"] not in acquisition_geometries:
            acquisition_geometries.append(row["geometry_id"])
        if len(acquisition_geometries) >= acquisition_geometry_count:
            break
    acquisitions: list[Dict[str, Any]] = []
    for geometry_id in acquisition_geometries:
        existing = sorted(key[1] for key in all_hf_keys if key[0] == geometry_id)
        candidates = sorted(key[1] for key in all_lf_keys if key[0] == geometry_id and key[1] not in existing)
        candidate_x = np.asarray([_build_inputs(payload, geometry_id, sample_id, uncertainty_names) for sample_id in candidates])
        anchor_x = np.asarray([_build_inputs(payload, geometry_id, sample_id, uncertainty_names) for sample_id in existing])
        additional = max(0, target_hf_per_geometry - len(existing))
        selected = _maximin_candidates(candidate_x, candidates, anchor_x, additional)
        acquisitions.append({
            "geometry_id": geometry_id,
            "existing_hf_count": len(existing),
            "target_hf_count": target_hf_per_geometry,
            "additional_hf_count": len(selected),
            "selection_method": "uncertainty_space_maximin_nested_in_existing_tpmc",
            "selected_canonical_sample_ids": [sample_id for sample_id, _score in selected],
            "selection_scores": [score for _sample_id, score in selected],
        })

    cv_path = target / "geometry_held_out_metrics.csv"
    metrics_path = target / "corrected_robust_metrics.csv"
    selected_metrics_path = target / "selected_robust_metrics.csv"
    acquisition_path = target / "next_hf_acquisition.json"
    _write_csv(cv_path, cv_rows)
    _write_csv(metrics_path, metric_rows)
    _write_csv(selected_metrics_path, metric_rows)
    acquisition_path.write_text(json.dumps({"schema_version": 1, "geometries": acquisitions}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hf_counts = {geometry_id: int(np.count_nonzero(hf_groups == geometry_id)) for geometry_id in geometry_ids}
    quality_flags: list[str] = []
    if len(geometry_ids) < 12:
        quality_flags.append("hf_geometry_count_lt_12")
    minimum_hf = min(hf_counts.values())
    if minimum_hf < 5:
        quality_flags.append("minimum_hf_pairs_per_training_geometry_lt_5")
    if len(set(hf_counts.values())) > 1:
        quality_flags.append("unbalanced_hf_pairs_across_geometries")
    selected_cv_rmse = mf_cv_rmse if correction_applied else lf_cv_rmse
    geometry_count_ready = len(geometry_ids) >= 12
    accuracy_ready = selected_cv_rmse <= float(target_geometry_rmse)
    status = (
        "surrogate_ready_for_optimization_candidate"
        if geometry_count_ready and accuracy_ready
        else "surrogate_complete_more_geometry_acquisition_required"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "bundle_json": str(source),
        "qoi": qoi,
        "input_names": input_names,
        "n_geometries": len(geometry_ids),
        "n_hf": len(hf_y),
        "n_lf": len(lf_y),
        "available_n_hf": len(all_hf_keys),
        "available_n_lf": len(all_lf_keys),
        "hf_counts_by_geometry": hf_counts,
        "training_sample_balance": {
            "enabled": selected_lf_ids is not None or selected_hf_ids is not None,
            "reference_bundle_json": str(reference_source),
            "lf_per_geometry": training_lf_per_geometry,
            "hf_per_geometry": training_hf_per_geometry,
            "lf_canonical_sample_ids": selected_lf_ids,
            "hf_canonical_sample_ids": selected_hf_ids,
        },
        "quality_flags": quality_flags,
        "geometry_held_out_summary": aggregate_cv,
        "model_selection": {
            "selected_surrogate": selected_surrogate,
            "correction_applied": correction_applied,
            "lf_pce_mean_geometry_rmse": lf_cv_rmse,
            "mf_pce_mean_geometry_rmse": mf_cv_rmse,
            "relative_mf_improvement": relative_improvement,
            "minimum_relative_improvement_required": float(minimum_mf_relative_improvement),
            "target_geometry_rmse": float(target_geometry_rmse),
            "selected_geometry_rmse": selected_cv_rmse,
            "geometry_count_ready": geometry_count_ready,
            "accuracy_ready": accuracy_ready,
        },
        "models": {
            "tpmc": str(model_dir / "tpmc_pce.json"),
            "dsmc_minus_tpmc": str(model_dir / "dsmc_minus_tpmc_pce.json"),
        },
        "geometry_held_out_metrics_csv": str(cv_path),
        "corrected_robust_metrics_csv": str(metrics_path),
        "selected_robust_metrics_csv": str(selected_metrics_path),
        "next_hf_acquisition_json": str(acquisition_path),
    }
    manifest_path = target / "geometry_mf_surrogate_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "manifest_json": str(manifest_path)}


def merge_sequential_hf_results(
    bundle_json: str | Path,
    run_root: str | Path,
    output_json: str | Path,
) -> Dict[str, Any]:
    """Add collected sequential DSMC rows to a paired bundle without duplicates."""
    source = Path(bundle_json).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    root = Path(run_root).resolve()
    evaluations = list(payload["evaluations"])
    positions = {
        (str(row["geometry_id"]), str(row["model_id"]), str(row["canonical_sample_id"])): index
        for index, row in enumerate(evaluations)
    }
    merged = 0
    replaced = 0
    result_files = sorted(root.glob("*/piclas_results.json"))
    if not result_files:
        raise FileNotFoundError(f"No collected piclas_results.json files found below {root}")
    for result_path in result_files:
        geometry_id = result_path.parent.name
        if geometry_id not in payload["geometries"]:
            raise ValueError(f"Sequential result has unknown geometry: {geometry_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "collected":
            raise ValueError(f"Sequential result is not collected: {result_path}")
        area = float(payload["geometries"][geometry_id]["design"]["reference_area_m2"])
        sample_ids = list(result["sample_ids"])
        cd_values = list(result["values_by_qoi"]["C_D"])
        costs = list(result["costs_cpu_hours"])
        if not (len(sample_ids) == len(cd_values) == len(costs)):
            raise ValueError(f"Inconsistent result lengths in {result_path}")
        for sample_id, cd, cost in zip(sample_ids, cd_values, costs):
            canonical_id = str(sample_id).replace("hf-crn-", "wp1-crn-")
            if canonical_id not in payload["uncertainty_samples"]:
                raise ValueError(f"Unknown uncertainty sample in {result_path}: {sample_id}")
            value = float(cd)
            row = {
                "C_D": value,
                "C_D2": value * value,
                "canonical_sample_id": canonical_id,
                "cost_cpu_hours": float(cost),
                "drag_area_m2": value * area,
                "drag_area_m4": (value * area) ** 2,
                "fidelity": "hf",
                "geometry_id": geometry_id,
                "model_id": "PICLas_DSMC",
                "reference_area_m2": area,
                "sample_id": str(sample_id),
            }
            key = (geometry_id, "PICLas_DSMC", canonical_id)
            if key in positions:
                evaluations[positions[key]] = row
                replaced += 1
            else:
                positions[key] = len(evaluations)
                evaluations.append(row)
                merged += 1
    evaluations.sort(key=lambda row: (str(row["geometry_id"]), str(row["model_id"]), str(row["canonical_sample_id"])))
    payload["evaluations"] = evaluations
    counts: Dict[str, int] = {}
    for row in evaluations:
        key = f"{row['geometry_id']}/{row['model_id']}"
        counts[key] = counts.get(key, 0) + 1
    payload["counts"] = dict(sorted(counts.items()))
    payload["study_id"] = "vleo_cylinder_hex_wp5_paired_after_sequential_hf_round1"
    target = Path(output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "output_json": str(target),
        "new_hf_rows": merged,
        "replaced_hf_rows": replaced,
        "total_evaluations": len(evaluations),
        "counts": payload["counts"],
    }
