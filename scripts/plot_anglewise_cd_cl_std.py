#!/usr/bin/env python3
"""Plot angle-wise C_D and C_L estimates with ±1 standard-deviation bands."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def read_data(path: Path) -> list[dict[str, float]]:
    fields = {"angle_deg", "C_D", "C_L", "Var_C_D", "Var_C_L"}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = [{field: float(row[field]) for field in fields} for row in reader]
    return sorted(rows, key=lambda row: row["angle_deg"])


def plot(input_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    rows = read_data(input_path)
    angle = [row["angle_deg"] for row in rows]

    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    series = (
        ("C_D", "Var_C_D", r"$C_D$", "#0072B2", "o"),
        ("C_L", "Var_C_L", r"$C_L$", "#E69F00", "s"),
    )
    for mean_field, var_field, label, color, marker in series:
        mean = [row[mean_field] for row in rows]
        std = [math.sqrt(row[var_field]) if row[var_field] >= 0.0 else math.nan for row in rows]
        lower = [value - sigma for value, sigma in zip(mean, std)]
        upper = [value + sigma for value, sigma in zip(mean, std)]
        ax.fill_between(angle, lower, upper, color=color, alpha=0.20, linewidth=0.0)
        ax.plot(angle, mean, color=color, marker=marker, ms=3.4, lw=1.5, label=label + r" ($\pm1$ SD)")

        invalid_x = [row["angle_deg"] for row in rows if row[var_field] < 0.0]
        invalid_y = [row[mean_field] for row in rows if row[var_field] < 0.0]
        if invalid_x:
            ax.scatter(
                invalid_x,
                invalid_y,
                color="#D55E00",
                marker="x",
                s=42,
                linewidths=1.5,
                zorder=4,
                label=f"{label}: negative variance (no SD)",
            )

    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax.set(
        xlabel="Angle of sideslip (deg)",
        ylabel="Aerodynamic coefficient",
        xlim=(min(angle), max(angle)),
        title=r"Angle-wise MFMC estimates with $\pm1$ standard deviation",
    )
    ax.grid(True, color="#D9D9D9", lw=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper center")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.input.with_name(args.input.stem + "_cd_cl_std")
    for path in plot(args.input, prefix):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
