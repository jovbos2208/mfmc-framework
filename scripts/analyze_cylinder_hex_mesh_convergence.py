#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _stats(values: list[float]) -> dict:
    if len(values) < 2:
        raise ValueError("Each mesh level needs at least two independent C_D values")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    se = std / math.sqrt(len(values))
    # t_0.975 for df=4; the generated study uses five seeds per level.
    t95 = 2.7764451051977987 if len(values) == 5 else 1.96
    return {
        "n": len(values),
        "mean_cd": mean,
        "sample_std_cd": std,
        "standard_error_cd": se,
        "relative_sample_std": std / abs(mean),
        "ci95_cd": [mean - t95 * se, mean + t95 * se],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze L0/L1/L2 Cylinder-Hex C_D convergence.")
    parser.add_argument("--suite", required=True, help="mesh_convergence_suite.json")
    parser.add_argument("--l0-results", required=True)
    parser.add_argument("--l1-results", required=True)
    parser.add_argument("--l2-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    mesh_levels = {row["level"]: row for row in suite["levels"]}
    result_paths = {"L0": args.l0_results, "L1": args.l1_results, "L2": args.l2_results}
    levels = []
    for level in ("L0", "L1", "L2"):
        payload = json.loads(Path(result_paths[level]).read_text(encoding="utf-8"))
        row = _stats([float(value) for value in payload["values_by_qoi"]["C_D"]])
        row["level"] = level
        row["results_path"] = str(Path(result_paths[level]).resolve())
        if level in mesh_levels:
            row.update(
                {
                    "n_hexahedra": mesh_levels[level]["n_hexahedra"],
                    "characteristic_cell_size_m": mesh_levels[level]["characteristic_cell_size_m"],
                }
            )
        levels.append(row)

    comparisons = []
    for coarse, fine in zip(levels[:-1], levels[1:]):
        difference = fine["mean_cd"] - coarse["mean_cd"]
        combined_se = math.hypot(coarse["standard_error_cd"], fine["standard_error_cd"])
        comparisons.append(
            {
                "coarse": coarse["level"],
                "fine": fine["level"],
                "delta_cd": difference,
                "relative_delta": difference / coarse["mean_cd"],
                "combined_standard_error": combined_se,
                "difference_in_combined_se": abs(difference) / combined_se if combined_se else None,
            }
        )

    output = {"schema_version": 1, "levels": levels, "comparisons": comparisons}
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
