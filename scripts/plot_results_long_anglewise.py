#!/usr/bin/env python3
"""Plot C_D, C_L and their derived variances from results_long.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


QOIS = ("C_D", "C_L", "Var_C_D", "Var_C_L")


def read_data(path: Path) -> dict[str, list[tuple[float, float]]]:
    values = {qoi: [] for qoi in QOIS}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"regime_aos_deg", "qoi", "mfmc_estimate"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            qoi = row["qoi"]
            if qoi not in values or row["mfmc_estimate"] in {"", "nan"}:
                continue
            values[qoi].append((float(row["regime_aos_deg"]), float(row["mfmc_estimate"])))
    for qoi in QOIS:
        values[qoi].sort()
        if not values[qoi]:
            raise ValueError(f"no finite {qoi} estimates found")
    return values


def render(data: dict[str, list[tuple[float, float]]], output: Path, *, zoom: bool) -> None:
    mpl.rcParams.update({"font.size": 9, "axes.labelsize": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True, layout="constrained")
    specs = (
        (axes[0, 0], "C_D", r"$C_D$", "#0072B2"),
        (axes[0, 1], "C_L", r"$C_L$", "#E69F00"),
        (axes[1, 0], "Var_C_D", r"$\mathrm{Var}(C_D)$", "#0072B2"),
        (axes[1, 1], "Var_C_L", r"$\mathrm{Var}(C_L)$", "#E69F00"),
    )
    for ax, qoi, ylabel, color in specs:
        points = [(x, y) for x, y in data[qoi] if not zoom or x > -90.0]
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        ax.plot(x, y, color=color, marker="o", ms=3.2, lw=1.4)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9D9D9", lw=0.6)
        negative = [(xx, yy) for xx, yy in points if qoi.startswith("Var_") and yy < 0.0]
        if negative:
            ax.scatter(*zip(*negative), color="#D55E00", marker="x", s=32, zorder=3)
        if not zoom and qoi == "C_L":
            ax.set_yscale("symlog", linthresh=0.01)
        if not zoom and qoi.startswith("Var_") and all(yy > 0.0 for yy in y):
            ax.set_yscale("log")
    for ax in axes[1, :]:
        ax.set_xlabel("Angle of sideslip (deg)")
    title = "Angle-wise MFMC estimates"
    if zoom:
        title += " (−90° anomaly omitted for scale)"
    fig.suptitle(title, fontsize=11)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.input.with_name("anglewise_cd_cl_variances")
    data = read_data(args.input)
    for suffix, zoom in (("_full.png", False), ("_zoom.png", True)):
        output = Path(str(prefix) + suffix)
        render(data, output, zoom=zoom)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
