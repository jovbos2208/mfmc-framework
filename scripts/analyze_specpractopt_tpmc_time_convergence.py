#!/usr/bin/env python3
"""Analyze paired-seed PICLas TPMC convergence with simulation end time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfmc_matplotlib")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


QOIS = ("C_D", "C_L")
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622}


def _t95(n: int) -> float:
    return T95.get(n - 1, 1.96)


def summary_stats(values: list[float]) -> dict[str, float | int | list[float]]:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("At least two finite values are required")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    sem = std / math.sqrt(len(values))
    half_width = _t95(len(values)) * sem
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "standard_error": sem,
        "relative_sample_std": std / abs(mean) if mean else float("nan"),
        "ci95": [mean - half_width, mean + half_width],
    }


def paired_stats(values: list[float], reference: list[float]) -> dict[str, float | int | list[float]]:
    if len(values) != len(reference):
        raise ValueError("Paired levels must contain the same number of seeds")
    differences = [value - target for value, target in zip(values, reference)]
    report = summary_stats(differences)
    report["mean_relative_to_reference"] = report["mean"] / abs(statistics.fmean(reference))
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary_rows: list[dict], output: Path) -> None:
    mpl.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"C_D": "#0072B2", "C_L": "#D55E00"}
    labels = {"C_D": r"$C_D$", "C_L": r"$C_L$"}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex="col", layout="constrained")
    for column, qoi in enumerate(QOIS):
        rows = [row for row in summary_rows if row["qoi"] == qoi]
        x = [row["t_end_s"] for row in rows]
        y = [row["mean"] for row in rows]
        yerr = [[row["mean"] - row["ci95_low"] for row in rows], [row["ci95_high"] - row["mean"] for row in rows]]
        axes[0, column].errorbar(x, y, yerr=yerr, color=colors[qoi], marker="o", linewidth=1.8, capsize=3)
        axes[0, column].set_ylabel(f"Mean {labels[qoi]} (95% CI)")
        axes[0, column].set_title(f"TPMC end-time convergence: {labels[qoi]}")
        delta = [100.0 * row["paired_mean_relative"] for row in rows]
        delta_low = [100.0 * row["paired_ci95_low_relative"] for row in rows]
        delta_high = [100.0 * row["paired_ci95_high_relative"] for row in rows]
        axes[1, column].errorbar(
            x,
            delta,
            yerr=[[value - low for value, low in zip(delta, delta_low)], [high - value for value, high in zip(delta, delta_high)]],
            color=colors[qoi],
            marker="o",
            linewidth=1.8,
            capsize=3,
        )
        axes[1, column].axhline(0.0, color="#444444", linewidth=0.9)
        axes[1, column].set_ylabel("Paired difference to longest run (%)")
        axes[1, column].set_xlabel("Simulation end time (s)")
        for ax in axes[:, column]:
            ax.set_xscale("log")
            ax.grid(True, which="major", color="#D0D0D0", linewidth=0.7)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def analyze(suite_path: Path, run_root: Path, output_dir: Path) -> dict:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    levels = sorted(suite["levels"], key=lambda row: float(row["t_end_s"]))
    values: dict[tuple[str, str], list[float]] = {}
    raw_rows: list[dict] = []
    for level in levels:
        results_path = run_root / level["id"] / "piclas_results.json"
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        sample_ids = list(payload["sample_ids"])
        expected_ids = [f"seed-{seed}" for seed in level["seeds"]]
        if sample_ids != expected_ids:
            raise ValueError(f"Seed order mismatch in {results_path}: {sample_ids} != {expected_ids}")
        for qoi in QOIS:
            qoi_values = [float(value) for value in payload["values_by_qoi"][qoi]]
            values[(level["id"], qoi)] = qoi_values
            for sample_id, value in zip(sample_ids, qoi_values):
                raw_rows.append({"level": level["id"], "t_end_s": level["t_end_s"], "qoi": qoi, "sample_id": sample_id, "value": value})

    reference = levels[-1]
    summary_rows: list[dict] = []
    for level in levels:
        for qoi in QOIS:
            stats = summary_stats(values[(level["id"], qoi)])
            paired = paired_stats(values[(level["id"], qoi)], values[(reference["id"], qoi)])
            reference_mean = statistics.fmean(values[(reference["id"], qoi)])
            summary_rows.append(
                {
                    "level": level["id"],
                    "t_end_s": level["t_end_s"],
                    "time_steps": level["time_steps"],
                    "sampling_iterations": level["sampling_iterations"],
                    "qoi": qoi,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "sample_std": stats["sample_std"],
                    "standard_error": stats["standard_error"],
                    "relative_sample_std": stats["relative_sample_std"],
                    "ci95_low": stats["ci95"][0],
                    "ci95_high": stats["ci95"][1],
                    "paired_mean_difference": paired["mean"],
                    "paired_mean_relative": paired["mean_relative_to_reference"],
                    "paired_ci95_low": paired["ci95"][0],
                    "paired_ci95_high": paired["ci95"][1],
                    "paired_ci95_low_relative": paired["ci95"][0] / abs(reference_mean),
                    "paired_ci95_high_relative": paired["ci95"][1] / abs(reference_mean),
                    "reference_level": reference["id"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "raw_values.csv", raw_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    report = {"schema_version": 1, "suite": str(suite_path.resolve()), "reference_level": reference["id"], "summary": summary_rows}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot(summary_rows, output_dir / "tpmc_time_convergence")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/specpractopt_tpmc_time_convergence/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/specpractopt_tpmc_time_convergence/analysis"))
    args = parser.parse_args()
    report = analyze(args.suite.resolve(), args.run_root.resolve(), args.output_dir.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
