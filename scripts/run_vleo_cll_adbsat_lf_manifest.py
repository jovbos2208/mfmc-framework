#!/usr/bin/env python3
"""Evaluate one geometry on a locked ADBSat-CLL sample manifest."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any


_SIMULATE = None
_ADBSAT_PATH = ""
_GEOMETRY_ID = ""
_QUIET_ADBSAT = False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init_worker(adbsat_path: str, geometry_id: str, quiet_adbsat: bool) -> None:
    global _SIMULATE, _ADBSAT_PATH, _GEOMETRY_ID, _QUIET_ADBSAT
    sys.path.insert(0, adbsat_path)
    import simulate

    _SIMULATE = simulate
    _ADBSAT_PATH = adbsat_path
    _GEOMETRY_ID = geometry_id
    _QUIET_ADBSAT = quiet_adbsat


def _evaluate_sample(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, sample = task
    payload = {
        "environment_model": "csv",
        "geometry_id": _GEOMETRY_ID,
        "lf_model": _GEOMETRY_ID,
        "flow_zero_direction": [1.0, 0.0, 0.0],
        "aos_deg": float(sample.get("aos_deg", 0.0)),
        "aoa_deg": float(sample.get("aoa_deg", 0.0)),
        "adbsat_aos_offset_deg": 0.0,
        "cll_alpha_n": sample["cll_alpha_n"],
        "cll_sigma_t": sample["cll_sigma_t"],
        "wall_temperature_k": 300.0,
        "relative_speed_mps": 7718.371,
        "atmosphere_row": [
            2.65669645265334e-11, 0.0, 0.0, 1.0e15, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 961.0,
        ],
        "write_input_audit": False,
        "write_panel_surface_field": False,
        "write_mat": False,
        "surface_sample_id": sample["sample_id"],
    }

    def call():
        return _SIMULATE.run_simulation(
            "CLL",
            930000 + index,
            313.0,
            0.0,
            _ADBSAT_PATH,
            env_payload_path=payload,
        )

    if _QUIET_ADBSAT:
        with contextlib.redirect_stdout(io.StringIO()):
            coeff, runtime_ms = call()
    else:
        coeff, runtime_ms = call()
    cd = float(coeff["C_D"] if isinstance(coeff, dict) else coeff[0])
    return {
        "sample_id": sample["sample_id"],
        "sample_index": index,
        "role": sample["role"],
        "cll_alpha_n": sample["cll_alpha_n"],
        "cll_sigma_t": sample["cll_sigma_t"],
        "aoa_deg": float(sample.get("aoa_deg", 0.0)),
        "aos_deg": float(sample.get("aos_deg", 0.0)),
        "C_D": cd,
        "C_D2": cd * cd,
        "runtime_ms": float(runtime_ms),
    }


def _write_rows(path: Path, rows_by_id: dict[str, dict[str, Any]]) -> None:
    rows = sorted(rows_by_id.values(), key=lambda row: int(row["sample_index"]))
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-adbsat", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")

    root = args.framework_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = manifest_payload["samples"]
    csv_path = output_dir / "adbsat_lf_5000.csv"

    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows_by_id = {row["sample_id"]: row for row in csv.DictReader(handle)}

    pending = [
        (index, sample)
        for index, sample in enumerate(samples)
        if sample["sample_id"] not in rows_by_id
    ]
    workers = min(args.workers, len(pending)) if pending else 0
    adbsat_path = str((root / "ADBSat-PyVersion").resolve())

    if pending:
        # Import once before creating the pool. Missing runtime dependencies then
        # fail immediately instead of making multiprocessing replace workers in
        # an endless initializer-failure loop.
        sys.path.insert(0, adbsat_path)
        import simulate

        geometry_mat = Path(adbsat_path) / "inou" / "models" / f"{args.geometry_id}.mat"
        if not geometry_mat.is_file():
            raise FileNotFoundError(f"ADBSat geometry MAT not found: {geometry_mat}")
        simulate._load_geometry_mat(str(geometry_mat))

        chunksize = max(1, len(pending) // max(1, workers * 8))
        # Linux fork preserves the preloaded, read-only MAT arrays using
        # copy-on-write, avoiding one shared-filesystem read per worker.
        context = multiprocessing.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(adbsat_path, args.geometry_id, args.quiet_adbsat),
        ) as pool:
            iterator = pool.imap_unordered(_evaluate_sample, pending, chunksize=chunksize)
            for completed, row in enumerate(iterator, start=1):
                rows_by_id[str(row["sample_id"])] = row
                if completed % args.checkpoint_every == 0:
                    _write_rows(csv_path, rows_by_id)
                    print(
                        f"completed={len(rows_by_id)}/{len(samples)} workers={workers}",
                        flush=True,
                    )

    _write_rows(csv_path, rows_by_id)
    report = {
        "schema_version": 2,
        "geometry_id": args.geometry_id,
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "n": len(rows_by_id),
        "workers_requested": args.workers,
        "workers_used_this_invocation": workers,
        "output": str(csv_path),
        "reference_area_convention": "A_total/2",
        "geometry_cache": "preloaded_parent_copy_on_write",
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
