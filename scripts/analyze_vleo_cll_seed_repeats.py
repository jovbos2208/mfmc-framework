#!/usr/bin/env python3
"""Summarize fixed-CLL PICLas seed repetitions without treating them as GSI samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    sample_ids = list(payload["sample_ids"])
    values = np.asarray(payload["values_by_qoi"]["C_D"], dtype=float)
    if len(sample_ids) != len(values):
        raise ValueError("sample_ids and C_D values have different lengths")
    groups: dict[str, list[float]] = {}
    for sample_id, value in zip(sample_ids, values):
        anchor = str(sample_id).split("-seed-", 1)[0]
        groups.setdefault(anchor, []).append(float(value))
    expected = {"low", "mid", "high"}
    if set(groups) != expected or any(len(group) != 5 for group in groups.values()):
        raise ValueError("Expected exactly five repetitions for low, mid, and high anchors")
    summary = {
        anchor: {
            "n": len(group),
            "mean_c_d": float(np.mean(group)),
            "sd_c_d": float(np.std(group, ddof=1)),
            "cv_percent": float(100.0 * np.std(group, ddof=1) / np.mean(group)),
            "range_c_d": float(np.ptp(group)),
        }
        for anchor, group in sorted(groups.items())
    }
    summary["pooled"] = {
        "within_anchor_variance_c_d": float(
            sum((len(group) - 1) * np.var(group, ddof=1) for group in groups.values())
            / sum(len(group) - 1 for group in groups.values())
        )
    }
    output = args.output or args.results.with_name("piclas_seed_repeat_audit.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
