from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .adapters import make_request
from .campaign import _cache_key_for_request, _find_geometry, _find_regime, _stable_seed
from .experiments import generate_experiment_cells
from .fingerprints import request_fingerprints, sample_fingerprints
from .output import MODEL_EVALUATION_COLUMNS, SAMPLE_INPUT_BASE_COLUMNS
from .sampling import InputModel, SamplingContext
from .surrogate_dataset import export_surrogate_dataset


PRODUCTION_PHASES = {"prod_hf", "prod_lf_full"}
EXACT_AUDIT_STATUS = "cache_hash_match"


def _load_snapshot(case_dir: Path) -> Dict[str, Any]:
    payload = json.loads((case_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    return dict(payload.get("config", payload))


def _load_cache_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in evaluation cache: {path}")
    return set(payload)


def _read_audit(path: Path) -> Dict[str, List[Dict[str, str]]]:
    rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            case = str(row.get("case", "")).strip()
            if case:
                rows[case].append(row)
    return rows


def _integer(row: Dict[str, str], name: str) -> int:
    return int(float(row.get(name, "0") or 0))


def _selected_indices(count: int, cap: int) -> set[int]:
    if count <= 0:
        return set()
    if cap <= 0 or count <= cap:
        return set(range(count))
    return set(np.unique(np.linspace(0, count - 1, cap, dtype=int)).tolist())


def _external_pilot_was_used(case_dir: Path) -> bool:
    with (case_dir / "results_long.csv").open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if "external_pilot_model_evaluations" in str(row.get("flags", "")):
                return True
    return False


def _advance_preproduction_rng(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    pilot_size: int,
    external_pilot_used: bool,
) -> None:
    repetitions = int(cfg.get("pilot", {}).get("robustness_repetitions", 20))
    if not external_pilot_used:
        for _ in range(repetitions):
            rng.choice(pilot_size, size=pilot_size, replace=False)
    for _ in range(repetitions):
        rng.choice(pilot_size, size=pilot_size, replace=True)
    if bool(cfg.get("qois", {}).get("single_allocation_all_qois", False)):
        for _ in range(repetitions):
            rng.choice(pilot_size, size=pilot_size, replace=False)


def _metadata(cfg: Dict[str, Any], geometry: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "aos_deg": cfg.get("execution", {}).get("aos_deg", 0),
        "aoa_deg": cfg.get("execution", {}).get("aoa_deg", 0),
        "geometry_id": geometry.get("id", geometry.get("name")),
        "geometry_name": geometry.get("name", geometry.get("id")),
        "geometry_class": geometry.get("geometry_class"),
    }
    if isinstance(geometry.get("metadata"), dict):
        metadata.update(geometry["metadata"])
    environment = cfg.get("execution", {}).get("environment", {})
    if isinstance(environment, dict):
        metadata.update(environment)
        if "model" in environment and "environment_model" not in metadata:
            metadata["environment_model"] = environment["model"]
    for key in (
        "flow_zero_direction",
        "flow_zero_direction_xyz",
        "zero_flow_direction",
        "zero_flow_direction_xyz",
        "adbsat_aos_offset_deg",
        "adbsat_aos_offset",
    ):
        if key in cfg.get("execution", {}):
            metadata[key] = cfg["execution"][key]
    return metadata


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        return number if np.isfinite(number) else None
    return None


def _direct_qois(cfg: Dict[str, Any]) -> List[str]:
    direct = [str(qoi) for qoi in cfg.get("qois", {}).get("direct", [])]
    return ["C_D", "C_D2"] if {"C_D", "C_D2"}.issubset(direct) else ["C_D"]


def _request_seed(cell: Any, model_id: str, lf_ids: Sequence[str]) -> Tuple[str, str, int]:
    if model_id == cell.hf_model_id:
        return "prod_hf", "hf", int(cell.seed) + 101
    if model_id not in lf_ids:
        raise KeyError(f"Unknown LF model in reconstruction audit: {model_id}")
    return "prod_lf_full", "lf", int(cell.seed) + 203 + list(lf_ids).index(model_id)


def _reconstruct_case_requests(
    case_dir: Path,
    audit_rows: Sequence[Dict[str, str]],
    cap_per_request: int,
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = _load_snapshot(case_dir)
    if bool(cfg.get("sampling", {}).get("trajectory", {}).get("enabled", False)):
        raise ValueError(f"Trajectory case requires a separate environment audit: {case_dir.name}")
    non_exact = [row for row in audit_rows if row.get("status") != EXACT_AUDIT_STATUS]
    if non_exact:
        raise ValueError(f"Case {case_dir.name} contains non-exact reconstruction audit rows")

    cells = {cell.cell_id(): cell for cell in generate_experiment_cells(cfg)}
    by_cell: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in audit_rows:
        by_cell[str(row["cell_id"])].append(row)

    input_model = InputModel(cfg.get("variables", []), cfg.get("sampling", {}), cfg.get("regime_label_map", {}))
    cache_keys = _load_cache_keys(case_dir / "evaluation_cache.json")
    lf_ids = [str(model.get("id")) for model in cfg.get("models", {}).get("lf", [])]
    qois = _direct_qois(cfg)
    qoi_key = "+".join(qois)
    reuse_pilot = bool(cfg.get("pilot", {}).get("reuse_across_budgets", False))
    external_pilot = _external_pilot_was_used(case_dir)
    records: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    backfill_audit: List[Dict[str, Any]] = []

    for cell_id, requests in sorted(by_cell.items()):
        cell = cells.get(cell_id)
        if cell is None:
            raise KeyError(f"Archived cell is not reproducible from config snapshot: {cell_id}")
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
            if reuse_pilot
            else int(cell.seed)
        )
        rng = np.random.default_rng(pilot_seed)
        context = SamplingContext(cell.regime_id, cell.active_source_blocks)
        input_model.sample(int(cell.pilot_size), context, rng)
        _advance_preproduction_rng(cfg, rng, int(cell.pilot_size), external_pilot)
        maximum_count = max(_integer(row, "production_samples") for row in requests)
        samples = input_model.sample(maximum_count, context, rng)
        sample_ids = [f"prod_{index}" for index in range(maximum_count)]
        geometry = _find_geometry(cfg, cell.geometry_id)
        regime = _find_regime(cfg, cell.regime_id)
        metadata = _metadata(cfg, geometry)
        hf_audit_row = next(
            (row for row in requests if str(row["model_id"]) == str(cell.hf_model_id)),
            None,
        )
        if hf_audit_row is None:
            raise ValueError(f"No HF production request in reconstruction audit for cell: {cell_id}")
        hf_count = _integer(hf_audit_row, "production_samples")
        paired_hf_indices = _selected_indices(hf_count, cap_per_request)

        for audit_row in requests:
            model_id = str(audit_row["model_id"])
            count = _integer(audit_row, "production_samples")
            phase, fidelity, seed = _request_seed(cell, model_id, lf_ids)
            request = make_request(
                study_id=cell.study_id,
                cell_id=cell_id,
                model_id=model_id,
                fidelity=fidelity,
                qois=qois,
                geometry=geometry,
                regime=regime,
                active_source_blocks=cell.active_source_blocks,
                sample_ids=sample_ids[:count],
                samples=samples[:count],
                seed=seed,
                metadata=metadata,
            )
            cache_key = _cache_key_for_request(request, qoi_key, phase)
            archived_key = str(audit_row.get("request_cache_key", ""))
            if cache_key != archived_key or cache_key not in cache_keys:
                raise ValueError(
                    f"Reconstructed request hash mismatch for case={case_dir.name} cell={cell_id} model={model_id}"
                )

            selected = _selected_indices(count, cap_per_request)
            if model_id != cell.hf_model_id:
                selected.update(index for index in paired_hf_indices if index < count)
            indices = sorted(selected)
            selected_samples = [samples[index] for index in indices]
            selected_ids = [sample_ids[index] for index in indices]
            selected_request = make_request(
                study_id=cell.study_id,
                cell_id=cell_id,
                model_id=model_id,
                fidelity=fidelity,
                qois=qois,
                geometry=geometry,
                regime=regime,
                active_source_blocks=cell.active_source_blocks,
                sample_ids=selected_ids,
                samples=selected_samples,
                seed=seed,
                metadata=metadata,
            )
            sample_hashes = sample_fingerprints(selected_samples)
            request_hashes = request_fingerprints(selected_request)
            for position, sample_index in enumerate(indices):
                records[(cell_id, model_id, sample_index)] = {
                    "sample": selected_samples[position],
                    "sample_id": selected_ids[position],
                    "sample_fingerprint": sample_hashes[position],
                    "request_fingerprint": request_hashes[position],
                    "geometry_characteristic_length": selected_request.geometry.characteristic_length,
                    "geometry_metadata": selected_request.geometry.metadata,
                }
            backfill_audit.append(
                {
                    "case": case_dir.name,
                    "cell_id": cell_id,
                    "model_id": model_id,
                    "phase": phase,
                    "production_samples": count,
                    "selected_samples": len(indices),
                    "paired_hf_samples_included": len(paired_hf_indices.intersection(selected)),
                    "cache_key": cache_key,
                    "status": "cache_hash_reverified",
                }
            )
        del samples

    return records, backfill_audit


def _write_case_outputs(
    case_dir: Path,
    target_dir: Path,
    records: Dict[Tuple[str, str, int], Dict[str, Any]],
    backfill_audit: List[Dict[str, Any]],
) -> Dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    evaluations_target = target_dir / "model_evaluations.csv"
    inputs_target = target_dir / "sample_inputs.csv"
    evaluations_tmp = target_dir / ".model_evaluations.csv.tmp"
    inputs_tmp = target_dir / ".sample_inputs.csv.tmp"

    numeric_input_names = sorted(
        {
            str(name)
            for record in records.values()
            for name, value in record["sample"].items()
            if _numeric(value) is not None
        }
    )
    numeric_geometry_names = sorted(
        {
            str(name)
            for record in records.values()
            for name, value in record["geometry_metadata"].items()
            if _numeric(value) is not None
        }
    )
    input_columns = SAMPLE_INPUT_BASE_COLUMNS
    input_columns = input_columns + [f"input__{name}" for name in numeric_input_names]
    input_columns = input_columns + [f"geometry__{name}" for name in numeric_geometry_names]
    output_rows = 0
    finite_rows = 0
    phase_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()

    try:
        with (case_dir / "model_evaluations.csv").open("r", encoding="utf-8", newline="") as source, \
            evaluations_tmp.open("w", encoding="utf-8", newline="") as evaluation_file, \
            inputs_tmp.open("w", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(source)
            missing = [name for name in MODEL_EVALUATION_COLUMNS if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Legacy model evaluations are missing columns: {', '.join(missing)}")
            evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=MODEL_EVALUATION_COLUMNS)
            input_writer = csv.DictWriter(input_file, fieldnames=input_columns)
            evaluation_writer.writeheader()
            input_writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                if row.get("phase") not in PRODUCTION_PHASES:
                    continue
                try:
                    sample_index = int(float(row.get("sample_index", "nan")))
                except ValueError:
                    continue
                key = (str(row.get("cell_id", "")), str(row.get("model_id", "")), sample_index)
                record = records.get(key)
                if record is None:
                    continue
                if row.get("sample_id") != record["sample_id"]:
                    raise ValueError(f"Sample ID mismatch at {case_dir / 'model_evaluations.csv'}:{row_number}")
                evaluation = {name: row.get(name, "") for name in MODEL_EVALUATION_COLUMNS}
                evaluation["sample_fingerprint"] = record["sample_fingerprint"]
                evaluation["request_fingerprint"] = record["request_fingerprint"]
                evaluation_writer.writerow(evaluation)

                input_row = {name: evaluation.get(name, "") for name in SAMPLE_INPUT_BASE_COLUMNS}
                input_row["geometry_characteristic_length"] = record["geometry_characteristic_length"]
                for name in numeric_input_names:
                    value = _numeric(record["sample"].get(name))
                    input_row[f"input__{name}"] = "" if value is None else value
                for name in numeric_geometry_names:
                    value = _numeric(record["geometry_metadata"].get(name))
                    input_row[f"geometry__{name}"] = "" if value is None else value
                input_writer.writerow(input_row)
                output_rows += 1
                phase_counts[str(row.get("phase", ""))] += 1
                model_counts[str(row.get("model_id", ""))] += 1
                try:
                    finite_rows += int(math.isfinite(float(row.get("value", "nan"))))
                except ValueError:
                    pass

        if output_rows == 0:
            raise RuntimeError(f"No matching production evaluations selected for {case_dir.name}")
        os.replace(evaluations_tmp, evaluations_target)
        os.replace(inputs_tmp, inputs_target)
    finally:
        for path in (evaluations_tmp, inputs_tmp):
            if path.exists():
                path.unlink()

    audit_path = target_dir / "backfill_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(backfill_audit[0]))
        writer.writeheader()
        writer.writerows(backfill_audit)
    shutil.copy2(case_dir / "config_snapshot.json", target_dir / "config_snapshot.json")
    surrogate_summary = export_surrogate_dataset(
        str(inputs_target),
        str(evaluations_target),
        str(target_dir / "surrogate_dataset.csv"),
    )
    return {
        "case": case_dir.name,
        "request_count": len(backfill_audit),
        "selected_sample_records": len(records),
        "model_evaluation_rows": output_rows,
        "finite_model_evaluation_rows": finite_rows,
        "phase_counts": dict(sorted(phase_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "surrogate_rows": surrogate_summary["written_rows"],
        "output_dir": str(target_dir),
    }


def backfill_legacy_surrogate_data(
    campaign_root: str,
    output_root: str,
    *,
    reconstruction_audit_csv: str | None = None,
    cases: Sequence[str] | None = None,
    cap_per_request: int = 2500,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Reconstruct a hash-verified production subset from legacy campaigns."""
    root = Path(campaign_root).resolve()
    destination = Path(output_root).resolve()
    audit_path = (
        Path(reconstruction_audit_csv).resolve()
        if reconstruction_audit_csv
        else root / "uncertainty_sensitivity_analysis" / "reconstruction_audit.csv"
    )
    if not audit_path.exists():
        raise FileNotFoundError(f"Reconstruction audit not found: {audit_path}")
    audit_by_case = _read_audit(audit_path)
    requested_cases = set(cases or audit_by_case)
    summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for case_name in sorted(requested_cases):
        case_dir = root / case_name
        rows = audit_by_case.get(case_name, [])
        if not rows:
            skipped.append({"case": case_name, "reason": "missing_reconstruction_audit"})
            continue
        statuses = {str(row.get("status", "")) for row in rows}
        if statuses != {EXACT_AUDIT_STATUS}:
            skipped.append({"case": case_name, "reason": "not_all_requests_exactly_hash_audited"})
            continue
        target_dir = destination / case_name
        manifest_path = target_dir / "backfill_manifest.json"
        if manifest_path.exists() and not overwrite:
            summaries.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Backfill target is not empty: {target_dir}; use --overwrite")
        target_dir.mkdir(parents=True, exist_ok=True)
        records, request_audit = _reconstruct_case_requests(case_dir, rows, cap_per_request)
        summary = _write_case_outputs(case_dir, target_dir, records, request_audit)
        summary["cap_per_request"] = cap_per_request
        summary["source_case_dir"] = str(case_dir)
        manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(summary)

    aggregate = {
        "campaign_root": str(root),
        "output_root": str(destination),
        "reconstruction_audit_csv": str(audit_path),
        "cap_per_request": cap_per_request,
        "completed_cases": [summary["case"] for summary in summaries],
        "skipped_cases": skipped,
        "model_evaluation_rows": sum(int(summary["model_evaluation_rows"]) for summary in summaries),
        "surrogate_rows": sum(int(summary["surrogate_rows"]) for summary in summaries),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "backfill_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    return aggregate
