#!/usr/bin/env python3
"""Plot angle-wise MFMC moments and flag physically invalid variances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {"drag": "#0072B2", "lift": "#E69F00", "negative": "#D55E00"}


def read_rows(path: Path) -> list[dict[str, float]]:
    required = {"angle_deg", "C_D", "C_D2", "C_L", "C_L2", "Var_C_D", "Var_C_L"}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = [{name: float(value) for name, value in row.items() if name in required} for row in reader]
    return sorted(rows, key=lambda row: row["angle_deg"])


def plot(input_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    rows = read_rows(input_path)
    angle = [row["angle_deg"] for row in rows]

    mpl.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), sharex=True, layout="constrained")
    specs = [
        (0, 0, "C_D", r"$C_D$", COLORS["drag"]),
        (0, 1, "C_L", r"$C_L$", COLORS["lift"]),
        (1, 0, "C_D2", r"$E[C_D^2]$", COLORS["drag"]),
        (1, 1, "C_L2", r"$E[C_L^2]$", COLORS["lift"]),
    ]
    for row_idx, col_idx, field, ylabel, color in specs:
        ax = axes[row_idx, col_idx]
        ax.plot(angle, [row[field] for row in rows], color=color, marker="o", ms=3.2, lw=1.4)
        ax.set_ylabel(ylabel)

    for col_idx, (field, ylabel, color) in enumerate((
        ("Var_C_D", r"$\mathrm{Var}(C_D)$", COLORS["drag"]),
        ("Var_C_L", r"$\mathrm{Var}(C_L)$", COLORS["lift"]),
    )):
        ax = axes[2, col_idx]
        values = [row[field] for row in rows]
        ax.plot(angle, values, color=color, marker="o", ms=3.2, lw=1.4, label="estimate")
        negative_x = [x for x, y in zip(angle, values) if y < 0.0]
        negative_y = [y for y in values if y < 0.0]
        if negative_x:
            ax.scatter(
                negative_x, negative_y, marker="x", s=30, linewidths=1.4,
                color=COLORS["negative"], zorder=3, label=f"negative ({len(negative_x)})",
            )
        ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
        ax.set(xlabel="Angle of sideslip (deg)", ylabel=ylabel)
        ax.legend(frameon=False, loc="best")

    for label, ax in zip("abcdef", axes.flat):
        ax.text(0.015, 0.96, f"({label})", transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(True, color="#D9D9D9", lw=0.6)
        ax.set_xlim(min(angle), max(angle))

    fig.suptitle("Angle-wise MFMC estimates (Maxwell)", fontsize=11)
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
    prefix = args.output_prefix or args.input.with_name(args.input.stem + "_moments")
    png, pdf = plot(args.input, prefix)
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
