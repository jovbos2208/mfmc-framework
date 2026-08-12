#!/usr/bin/env python3
"""Merge deterministic single-m0 sensitivity batches into one case result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mfmc_campaign.field_mfpod.sensitivity import (
    _write_case_findings,
    _write_csv,
    _write_figures,
    _write_json,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build(batch_roots: list[Path], output: Path, case_name: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    allocations, repetitions, summaries = [], [], []
    allocation_details: dict = {}
    pod_methods: dict = {}
    pod_reference = None
    cautions = None
    reference_hashes = set()
    metadata_rows = []
    for root in batch_roots:
        sensitivity = root / "sensitivity"
        allocations.extend(read_csv(sensitivity / "m0_allocations.csv"))
        repetitions.extend(read_csv(sensitivity / "m0_repetitions.csv"))
        summaries.extend(read_csv(sensitivity / "m0_summary.csv"))
        allocation_details.update(
            json.loads((sensitivity / "m0_allocation_details.json").read_text())
        )
        pod = json.loads((sensitivity / "pod_subspace_diagnostics.json").read_text())
        pod_reference = pod_reference or pod["reference"]
        cautions = cautions or pod["interpretation_cautions"]
        for name, value in pod["methods"].items():
            if name.startswith("field-aware-m0-") or name not in pod_methods:
                pod_methods[name] = value
        reference_hashes.add(
            (sensitivity / "reference_convergence_repetitions.csv").read_bytes()
        )
        metadata_rows.append(json.loads((sensitivity / "sensitivity_metadata.json").read_text()))
    if len(reference_hashes) != 1:
        raise RuntimeError("Reference convergence differs between deterministic batches")
    allocations.sort(key=lambda row: int(row["minimum_target"]))
    repetitions.sort(
        key=lambda row: (
            int(row["minimum_target"]),
            int(row["repetition"]),
            int(row["reference_sample_count"]),
            row["method"],
        )
    )
    summaries.sort(
        key=lambda row: (
            int(row["minimum_target"]),
            int(row["reference_sample_count"]),
            row["method"],
        )
    )
    reference_repetitions = read_csv(
        batch_roots[0] / "sensitivity" / "reference_convergence_repetitions.csv"
    )
    reference_summaries = read_csv(
        batch_roots[0] / "sensitivity" / "reference_convergence_summary.csv"
    )
    _write_csv(output / "m0_allocations.csv", allocations)
    _write_csv(output / "m0_repetitions.csv", repetitions)
    _write_csv(output / "m0_summary.csv", summaries)
    _write_csv(
        output / "reference_convergence_repetitions.csv", reference_repetitions
    )
    _write_csv(output / "reference_convergence_summary.csv", reference_summaries)
    _write_json(output / "m0_allocation_details.json", allocation_details)
    _write_json(
        output / "pod_subspace_diagnostics.json",
        {
            "reference": pod_reference,
            "methods": pod_methods,
            "interpretation_cautions": cautions,
        },
    )
    _write_case_findings(
        output / "case_findings.md",
        case_name=case_name,
        summaries=summaries,
        reference_count=50,
    )
    figures = _write_figures(
        output, case_name, allocations, summaries, reference_summaries
    )
    first = metadata_rows[0]
    metadata = {
        **first,
        "results_dir": str(output.parent),
        "minimum_targets": [int(row["minimum_target"]) for row in allocations],
        "execution_granularity": "eight deterministic single-m0 batches",
        "files": {
            "allocations": str(output / "m0_allocations.csv"),
            "repetitions": str(output / "m0_repetitions.csv"),
            "summary": str(output / "m0_summary.csv"),
            "reference_repetitions": str(
                output / "reference_convergence_repetitions.csv"
            ),
            "reference_summary": str(
                output / "reference_convergence_summary.csv"
            ),
            "allocation_details": str(output / "m0_allocation_details.json"),
            "pod_diagnostics": str(output / "pod_subspace_diagnostics.json"),
            "case_findings": str(output / "case_findings.md"),
        },
        "figures": figures,
    }
    _write_json(output / "sensitivity_metadata.json", metadata)
    return {
        "allocations": len(allocations),
        "repetitions": len(repetitions),
        "summaries": len(summaries),
        "reference_repetitions": len(reference_repetitions),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                [path.resolve() for path in args.batch_root],
                args.output.resolve(),
                args.case_name,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
