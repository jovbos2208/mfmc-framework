#!/usr/bin/env python3
"""Fill selected campaign cases to a target number of physical PICLas HF runs.

The script is dry-run by default.  With ``--execute`` it samples new,
deterministic validation inputs from each case's config snapshot, evaluates
``C_D`` and ``C_D2`` through the configured PICLas HF adapter, and appends each
completed batch to the case's ``model_evaluations.csv`` and ``sample_inputs.csv``.

Historical pilot rows are physically deduplicated by their exact C_D output:
their request/sample fingerprints may encode repetition context even when the
underlying PICLas result was reused.  Newly generated samples are additionally
deduplicated by the framework's physical sample fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Always prefer the framework checkout adjacent to this script.  A server may
# also expose an older mfmc_campaign package through PYTHONPATH/site-packages.
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from mfmc_campaign.adapters import build_adapter_registry, make_request
from mfmc_campaign.campaign import _append_model_evaluation_rows
from mfmc_campaign.experiments import generate_experiment_cells
from mfmc_campaign.fingerprints import sample_fingerprints
from mfmc_campaign.output import ResultStore
from mfmc_campaign.sampling import InputModel, SamplingContext


DEFAULT_CASES = (
    "cube_100km",
    "cube_200km",
    "cube_300km",
    "cube_400km",
    "goce_20131021_d3",
)
COUNTED_PHASES = {"pilot_hf", "prod_hf", "validation_hf"}


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12], 16) % (2**31 - 1)


def load_snapshot(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "config_snapshot.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config snapshot: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise TypeError(f"Config snapshot does not contain a mapping: {path}")
    return config


def find_case_dir(campaign_root: Path, case: str) -> Path:
    direct = campaign_root / case
    if (direct / "config_snapshot.json").is_file():
        return direct.resolve()
    matches = sorted(
        path.parent.resolve()
        for path in campaign_root.rglob("config_snapshot.json")
        if path.parent.name == case
    )
    if not matches:
        raise FileNotFoundError(f"Could not find case '{case}' below {campaign_root}")
    if len(matches) > 1:
        rendered = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"Case '{case}' is ambiguous below {campaign_root}:\n  {rendered}")
    return matches[0]


def inventory_hf(csv_path: Path) -> tuple[set[str], set[str], dict[str, int]]:
    """Return exact physical C_D outputs, known sample hashes, and phase rows."""
    values: set[str] = set()
    sample_hashes: set[str] = set()
    phase_rows: dict[str, int] = {}
    if not csv_path.is_file():
        return values, sample_hashes, phase_rows
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"phase", "qoi", "fidelity", "value", "sample_fingerprint"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} lacks required columns: {sorted(missing)}")
        for row in reader:
            phase = str(row.get("phase", ""))
            if phase not in COUNTED_PHASES or row.get("qoi") != "C_D" or row.get("fidelity") != "hf":
                continue
            value = str(row.get("value", "")).strip()
            try:
                numeric = float(value)
            except ValueError:
                continue
            if not math.isfinite(numeric):
                continue
            # Use the serialized solver output. Reused historical pilot results
            # are bit-identical while their context-dependent hashes are not.
            values.add(value)
            fingerprint = str(row.get("sample_fingerprint", "")).strip()
            if fingerprint:
                sample_hashes.add(fingerprint)
            phase_rows[phase] = phase_rows.get(phase, 0) + 1
    return values, sample_hashes, phase_rows


def select_anchor(config: dict[str, Any]):
    candidates = [
        cell for cell in generate_experiment_cells(config)
        if cell.qoi == "C_D" and cell.repetition == 0
    ]
    if not candidates:
        raise ValueError("Config produces no repetition-0 C_D experiment cell")
    return max(candidates, key=lambda cell: float(cell.budget))


def find_entry(entries: Iterable[dict[str, Any]], entry_id: str, label: str) -> dict[str, Any]:
    for entry in entries:
        if str(entry.get("id", entry.get("name", ""))) == entry_id:
            return entry
    raise KeyError(f"Unknown {label} id '{entry_id}'")


def request_metadata(config: dict[str, Any], geometry: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    execution = config.get("execution", {})
    meta: dict[str, Any] = {
        "aos_deg": execution.get("aos_deg", 0),
        "aoa_deg": execution.get("aoa_deg", 0),
        "geometry_id": geometry.get("id", geometry.get("name")),
        "geometry_name": geometry.get("name", geometry.get("id")),
        "geometry_class": geometry.get("geometry_class"),
    }
    if isinstance(geometry.get("metadata"), dict):
        meta.update(geometry["metadata"])
    environment = execution.get("environment", {})
    if isinstance(environment, dict):
        meta.update(environment)
        if "model" in environment and "environment_model" not in meta:
            meta["environment_model"] = environment["model"]
    descriptors = regime.get("descriptors", {})
    if isinstance(descriptors, dict):
        for key in ("aos_deg", "aoa_deg"):
            if key in descriptors:
                meta[key] = descriptors[key]
    for key in (
        "flow_zero_direction", "flow_zero_direction_xyz", "zero_flow_direction",
        "zero_flow_direction_xyz", "adbsat_aos_offset_deg", "adbsat_aos_offset",
    ):
        if key in execution:
            meta[key] = execution[key]
    meta["validation_purpose"] = "independent_hf_reference_fill"
    return meta


def validate_piclas_config(config: dict[str, Any]) -> None:
    execution = config.get("execution", {})
    hf = config.get("models", {}).get("hf", {})
    kwargs = hf.get("kwargs", {})
    if execution.get("backend") != "legacy_slurm":
        raise ValueError("HF fill requires execution.backend=legacy_slurm")
    if hf.get("kind", "legacy_piclas") != "legacy_piclas":
        raise ValueError("HF fill requires models.hf.kind=legacy_piclas")
    if kwargs.get("simulator_module", "PICLas") != "PICLas":
        raise ValueError("HF fill is restricted to simulator_module=PICLas")


def generate_candidates(
    config: dict[str, Any], anchor, known_hashes: set[str], count: int, case: str, target: int
) -> tuple[list[dict[str, Any]], list[str]]:
    input_model = InputModel(
        config.get("variables", []), config.get("sampling", {}),
        regime_label_map=config.get("regime_label_map", {}),
    )
    context = SamplingContext(
        regime_id=anchor.regime_id,
        active_source_blocks=anchor.active_source_blocks,
    )
    seed = stable_seed("hf_validation_fill", case, target)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    selected_hashes: list[str] = []
    seen = set(known_hashes)
    attempts = 0
    while len(selected) < count:
        draw_count = max(64, 2 * (count - len(selected)))
        batch = input_model.sample(draw_count, context, rng)
        for sample, fingerprint in zip(batch, sample_fingerprints(batch)):
            attempts += 1
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            selected.append(sample)
            selected_hashes.append(fingerprint)
            if len(selected) == count:
                break
        if attempts > max(10000, 100 * count) and len(selected) < count:
            raise RuntimeError(
                f"Could generate only {len(selected)}/{count} unique inputs; "
                "check discrete/trajectory sampling support"
            )
    return selected, selected_hashes


@contextmanager
def case_lock(case_dir: Path):
    lock_path = case_dir / ".validation_hf_fill.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another HF-fill process holds {lock_path}") from exc
        yield


def run_case(case_dir: Path, case: str, target: int, batch_size: int, execute: bool) -> None:
    config = load_snapshot(case_dir)
    validate_piclas_config(config)
    output_csv = case_dir / "model_evaluations.csv"
    physical_values, known_hashes, phase_rows = inventory_hf(output_csv)
    current = len(physical_values)
    missing = max(0, target - current)
    print(
        f"[{case}] directory={case_dir}\n"
        f"[{case}] physical_hf={current} target={target} missing={missing} "
        f"stored_rows_by_phase={phase_rows}",
        flush=True,
    )
    if missing == 0 or not execute:
        return

    anchor = select_anchor(config)
    geometry = find_entry(config.get("geometries", []), anchor.geometry_id, "geometry")
    regime = find_entry(config.get("regimes", []), anchor.regime_id, "regime")
    metadata = request_metadata(config, geometry, regime)
    hf_id = str(config.get("models", {}).get("hf", {}).get("id", anchor.hf_model_id))
    samples, hashes = generate_candidates(config, anchor, known_hashes, missing, case, target)
    adapter = build_adapter_registry(config).get(hf_id)
    store = ResultStore(str(case_dir))
    request_seed = stable_seed("hf_validation_piclas", case, target)

    with case_lock(case_dir):
        for start in range(0, missing, batch_size):
            stop = min(start + batch_size, missing)
            batch_samples = samples[start:stop]
            batch_hashes = hashes[start:stop]
            sample_ids = [f"validation_hf_{fingerprint[:16]}" for fingerprint in batch_hashes]
            request = make_request(
                study_id=anchor.study_id,
                cell_id=anchor.cell_id(),
                model_id=hf_id,
                fidelity="hf",
                qois=["C_D", "C_D2"],
                geometry=geometry,
                regime=regime,
                active_source_blocks=anchor.active_source_blocks,
                sample_ids=sample_ids,
                samples=batch_samples,
                seed=request_seed + start,
                metadata=metadata,
            )
            print(f"[{case}] evaluating batch {start + 1}-{stop}/{missing}", flush=True)
            result = adapter.evaluate(request)
            cd = np.asarray(result.values_by_qoi.get("C_D", []), dtype=float)
            cd2 = np.asarray(result.values_by_qoi.get("C_D2", []), dtype=float)
            if len(cd) != len(batch_samples) or len(cd2) != len(batch_samples):
                raise RuntimeError(
                    f"[{case}] incomplete PICLas result: C_D={len(cd)}, C_D2={len(cd2)}, "
                    f"expected={len(batch_samples)}"
                )
            if not np.all(np.isfinite(cd)) or not np.all(np.isfinite(cd2)):
                raise RuntimeError(f"[{case}] non-finite PICLas result; batch was not appended")
            _append_model_evaluation_rows(store, anchor, request, result, "validation_hf")
            print(f"[{case}] appended {len(batch_samples)} physical HF runs", flush=True)

    final_values, _, _ = inventory_hf(output_csv)
    if len(final_values) < target:
        raise RuntimeError(f"[{case}] finished with only {len(final_values)}/{target} unique C_D outputs")
    print(f"[{case}] complete: physical_hf={len(final_values)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=Path("campaign_outputs"))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--target", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--execute", action="store_true", help="Submit PICLas jobs; otherwise print the plan only")
    args = parser.parse_args()
    if args.target < 1:
        parser.error("--target must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    root = args.campaign_root.resolve()
    for case in args.cases:
        run_case(find_case_dir(root, case), case, args.target, args.batch_size, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
