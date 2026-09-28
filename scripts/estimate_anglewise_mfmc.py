#!/usr/bin/env python3
"""Estimate angle-wise MFMC moments from a model_evaluations.csv file.

The control-variate weight is estimated independently for every angle and QoI
from the first N paired production evaluations.  Production estimates use

    mean(HF) - beta * (mean(LF_paired) - mean(LF_full)).

Only the paired samples are retained in memory; the (potentially very large)
LF-full sample is reduced to online sums while the CSV is streamed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


QOIS = ("C_D", "C_D2", "C_L", "C_L2")
ANGLE_RE = re.compile(r"(?:^|_)AOS_([PM])(\d+(?:\.\d+)?)(?:_|$)")


@dataclass
class Sample:
    index: int
    value: float


@dataclass
class RunningMean:
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.count += 1
            self.total += value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan


def parse_angle(regime_id: str) -> float:
    """Return AoS in degrees from e.g. ``..._AOS_M045_AOA_...``."""
    match = ANGLE_RE.search(regime_id)
    if not match:
        raise ValueError(f"cannot extract AOS angle from regime_id={regime_id!r}")
    sign = -1.0 if match.group(1) == "M" else 1.0
    return sign * float(match.group(2))


def _sample_key(row: dict[str, str]) -> str:
    # The fingerprint represents the physical random input and is therefore
    # safer than relying on row order.  Older files may only have sample_id.
    return row.get("sample_fingerprint") or row["sample_id"]


def _store_sample(
    target: dict[tuple[float, str], dict[str, Sample]],
    key: tuple[float, str],
    row: dict[str, str],
) -> None:
    sample_key = _sample_key(row)
    sample = Sample(index=int(row["sample_index"]), value=float(row["value"]))
    previous = target[key].get(sample_key)
    if previous is not None and previous != sample:
        raise ValueError(f"conflicting duplicate for {key}, sample {sample_key}")
    target[key][sample_key] = sample


def _paired_values(
    hf: dict[str, Sample], lf: dict[str, Sample], pair_count: int
) -> tuple[list[float], list[float]]:
    common = sorted(
        hf.keys() & lf.keys(),
        key=lambda sample_key: (hf[sample_key].index, sample_key),
    )
    selected = common[:pair_count]
    return [hf[s].value for s in selected], [lf[s].value for s in selected]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def weight_and_correlation(hf: list[float], lf: list[float]) -> tuple[float, float]:
    if len(hf) != len(lf) or len(hf) < 2:
        raise ValueError("at least two paired HF/LF samples are required")
    mh, ml = _mean(hf), _mean(lf)
    cov = sum((h - mh) * (l - ml) for h, l in zip(hf, lf)) / (len(hf) - 1)
    var_h = sum((h - mh) ** 2 for h in hf) / (len(hf) - 1)
    var_l = sum((l - ml) ** 2 for l in lf) / (len(lf) - 1)
    if var_l <= 0.0:
        raise ValueError("paired LF sample has zero variance")
    beta = cov / var_l
    rho = cov / math.sqrt(var_h * var_l) if var_h > 0.0 else math.nan
    return beta, rho


def estimate_file(
    input_path: Path,
    *,
    pair_count: int = 32,
    repetition: int | None = 0,
    hf_model: str | None = "PICLas_TPMC",
    lf_model: str | None = None,
) -> list[dict[str, float | int | str]]:
    prod_hf: dict[tuple[float, str], dict[str, Sample]] = defaultdict(dict)
    prod_lf_pair: dict[tuple[float, str], dict[str, Sample]] = defaultdict(dict)
    prod_lf_full: dict[tuple[float, str], RunningMean] = defaultdict(RunningMean)

    required = {"phase", "fidelity", "model_id", "qoi", "regime_id", "value", "sample_index", "sample_id"}
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["qoi"] not in QOIS:
                continue
            if repetition is not None and int(row.get("repetition", 0)) != repetition:
                continue
            if row["fidelity"] == "hf" and hf_model and row["model_id"] != hf_model:
                continue
            if row["fidelity"] == "lf" and lf_model and row["model_id"] != lf_model:
                continue
            key = (parse_angle(row["regime_id"]), row["qoi"])
            phase = row["phase"]
            if phase == "prod_hf":
                _store_sample(prod_hf, key, row)
            elif phase == "prod_lf_pair":
                _store_sample(prod_lf_pair, key, row)
            elif phase == "prod_lf_full":
                prod_lf_full[key].add(float(row["value"]))

    keys = sorted(prod_hf.keys() | prod_lf_pair.keys())
    if not keys:
        raise ValueError("no prod_hf/prod_lf_pair rows found after filtering")

    long_rows: list[dict[str, float | int | str]] = []
    for key in keys:
        angle, qoi = key
        weight_hf, weight_lf = _paired_values(prod_hf[key], prod_lf_pair[key], pair_count)
        if len(weight_hf) < pair_count:
            raise ValueError(
                f"angle {angle:g}, {qoi}: found {len(weight_hf)} paired production samples, "
                f"need {pair_count}"
            )
        beta, rho = weight_and_correlation(weight_hf, weight_lf)

        estimate_hf, estimate_lf = _paired_values(prod_hf[key], prod_lf_pair[key], 10**18)
        lf_full = prod_lf_full[key]
        source = "production"
        if len(estimate_hf) != len(estimate_lf):
            raise ValueError(f"angle {angle:g}, {qoi}: production HF/LF pair count mismatch")
        if not estimate_hf or not lf_full.count:
            raise ValueError(f"angle {angle:g}, {qoi}: incomplete production sample")

        hf_mean = _mean(estimate_hf)
        lf_pair_mean = _mean(estimate_lf)
        estimate = hf_mean - beta * (lf_pair_mean - lf_full.mean)
        long_rows.append(
            {
                "angle_deg": angle,
                "qoi": qoi,
                "estimate": estimate,
                "beta": beta,
                "rho": rho,
                "n_weight_pairs": len(weight_hf),
                "n_estimate_pairs": len(estimate_hf),
                "n_lf_full": lf_full.count,
                "hf_mean": hf_mean,
                "lf_pair_mean": lf_pair_mean,
                "lf_full_mean": lf_full.mean,
                "estimate_source": source,
            }
        )
    return long_rows


def write_wide(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    by_angle: dict[float, dict[str, dict[str, float | int | str]]] = defaultdict(dict)
    for row in rows:
        by_angle[float(row["angle_deg"])][str(row["qoi"])] = row
    missing = {angle: set(QOIS) - set(values) for angle, values in by_angle.items()}
    missing = {angle: qois for angle, qois in missing.items() if qois}
    if missing:
        raise ValueError(f"missing QoIs by angle: {missing}")

    fields = ["angle_deg"]
    for qoi in QOIS:
        fields.extend((qoi, f"beta_{qoi}", f"rho_{qoi}"))
    fields.extend(("Var_C_D", "Var_C_L", "n_weight_pairs", "n_estimate_pairs", "n_lf_full", "estimate_source"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for angle in sorted(by_angle):
            values = by_angle[angle]
            out: dict[str, float | int | str] = {"angle_deg": angle}
            for qoi in QOIS:
                out[qoi] = values[qoi]["estimate"]
                out[f"beta_{qoi}"] = values[qoi]["beta"]
                out[f"rho_{qoi}"] = values[qoi]["rho"]
            out["Var_C_D"] = float(out["C_D2"]) - float(out["C_D"]) ** 2
            out["Var_C_L"] = float(out["C_L2"]) - float(out["C_L"]) ** 2
            representative = values["C_D"]
            for name in ("n_weight_pairs", "n_estimate_pairs", "n_lf_full", "estimate_source"):
                out[name] = representative[name]
            writer.writerow(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="model_evaluations.csv")
    parser.add_argument("-o", "--output", type=Path, default=Path("anglewise_mfmc_estimates.csv"))
    parser.add_argument("--pair-count", type=int, default=32)
    parser.add_argument("--repetition", type=int, default=0, help="use -1 to include all repetitions")
    parser.add_argument("--hf-model", default="PICLas_TPMC")
    parser.add_argument("--lf-model", default=None, help="optional exact ADBSat model_id, e.g. Maxwell")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.pair_count < 2:
        raise SystemExit("--pair-count must be at least 2")
    repetition = None if args.repetition < 0 else args.repetition
    try:
        rows = estimate_file(
            args.input,
            pair_count=args.pair_count,
            repetition=repetition,
            hf_model=args.hf_model or None,
            lf_model=args.lf_model or None,
        )
        write_wide(rows, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    angles = len({float(row["angle_deg"]) for row in rows})
    print(f"wrote {angles} angles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
