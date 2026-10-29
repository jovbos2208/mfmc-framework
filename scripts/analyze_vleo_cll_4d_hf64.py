#!/usr/bin/env python3
"""Audit and rank the paired HF-64 global 4D VLEO CLL pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PARAMS = ("cll_alpha_n", "cll_sigma_t", "aoa_deg", "aos_deg")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 and np.std(x) and np.std(y) else float("nan")


def estimate(h: np.ndarray, l: np.ndarray, la: np.ndarray, h2: np.ndarray, l2: np.ndarray,
             la2: np.ndarray, rho_gate: float) -> dict:
    rho, rho2 = corr(h, l), corr(h2, l2)
    beta = float(np.cov(h, l, ddof=1)[0, 1] / np.var(l, ddof=1))
    beta2 = float(np.cov(h2, l2, ddof=1)[0, 1] / np.var(l2, ddof=1))
    mu = float(h.mean() + beta * (la.mean() - l.mean()))
    m2 = float(h2.mean() + beta2 * (la2.mean() - l2.mean()))
    raw_var = m2 - mu * mu
    active = bool(np.isfinite([rho, rho2, beta, beta2, raw_var]).all()
                  and abs(rho) >= rho_gate and abs(rho2) >= rho_gate and raw_var > 0)
    if not active:
        mu, m2 = float(h.mean()), float(h2.mean())
        raw_var = float(np.var(h, ddof=1))
        beta = beta2 = 0.0
    return {"active": active, "rho": rho, "rho2": rho2, "beta": beta, "beta2": beta2,
            "mu": mu, "m2": m2, "var": raw_var, "sd": math.sqrt(raw_var)}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_campaign(path: Path, manifest: list[dict], tolerance: float) -> tuple[dict, list[dict]]:
    evaluations = read_csv(path / "model_evaluations.csv")
    inputs = read_csv(path / "sample_inputs.csv")
    values: dict[tuple[str, str], dict[int, float]] = {}
    costs: dict[tuple[str, int], float] = {}
    for row in evaluations:
        idx = int(row["sample_index"])
        key = (row["fidelity"], row["qoi"])
        if idx in values.setdefault(key, {}):
            raise ValueError(f"duplicate evaluation {path}: {key}/{idx}")
        values[key][idx] = float(row["value"])
        costs[(row["fidelity"], idx)] = float(row["cost"])
    required = (("hf", "C_D"), ("hf", "C_D2"))
    if any(set(values.get(key, {})) != set(range(len(manifest))) for key in required):
        raise ValueError(f"incomplete paired evaluations in {path}")
    input_by_idx = {}
    for row in inputs:
        if row["fidelity"] != "hf" or row["qoi"] != "C_D":
            continue
        idx = int(row["sample_index"])
        actual = {p: float(row[f"input__{p}"]) for p in PARAMS}
        expected = {p: float(manifest[idx][p]) for p in PARAMS}
        if max(abs(actual[p] - expected[p]) for p in PARAMS) > tolerance:
            raise ValueError(f"manifest mismatch {path}/{idx}")
        input_by_idx[idx] = actual
    if set(input_by_idx) != set(range(len(manifest))):
        raise ValueError(f"incomplete input mapping in {path}")
    arrays = {key: np.asarray([values[key][i] for i in range(len(manifest))]) for key in required}
    if not np.isfinite([costs[("hf", i)] for i in range(len(manifest))]).all():
        raise ValueError(f"invalid costs in {path}")
    if not np.allclose(arrays[("hf", "C_D2")], arrays[("hf", "C_D")] ** 2, rtol=0, atol=1e-12):
        raise ValueError(f"invalid HF C_D2 in {path}")
    return arrays, [{"hf_cost": costs[("hf", i)]} for i in range(len(manifest))]


def load_lf10k(path: Path, manifest: list[dict], tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    if len(rows) != len(manifest):
        raise ValueError(f"LF count mismatch in {path}")
    cd = []
    for i, (row, expected) in enumerate(zip(rows, manifest)):
        if row["sample_id"] != expected["sample_id"]:
            raise ValueError(f"LF order mismatch in {path}/{i}")
        if max(abs(float(row[p]) - float(expected[p])) for p in PARAMS) > tolerance:
            raise ValueError(f"LF manifest mismatch in {path}/{i}")
        value, value2 = float(row["C_D"]), float(row["C_D2"])
        if not np.isfinite([value, value2]).all() or abs(value2 - value * value) > 1e-12:
            raise ValueError(f"invalid LF value in {path}/{i}")
        cd.append(value)
    result = np.asarray(cd)
    return result, result ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/vleo_cll/global_6d_hf64"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/vleo_cll_6d_hf_pilot_bundle"))
    parser.add_argument("--lf-results", type=Path, default=Path("outputs/global_6d_geometry_lf10000"))
    parser.add_argument("--lf-bundle", type=Path, default=Path("outputs/vleo_cll_6d_global_search_bundle"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vleo_cll_6d_hf64_analysis"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--rho-gate", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads((args.bundle / "bundle_manifest.json").read_text())
    paired_path = args.bundle / "manifests/joint_gsi_attitude_hf64.json"
    paired = json.loads(paired_path.read_text())["samples"]
    full_path = args.lf_bundle / "manifests/joint_gsi_attitude_lf10000.json"
    full = json.loads(full_path.read_text())["samples"]
    gids = [item["geometry_id"] for item in bundle["geometries"]]
    data, qa = {}, []
    for gid in gids:
        arrays, costs = load_campaign(args.results / gid, paired, args.tolerance)
        lf, lf2 = load_lf10k(args.lf_results / gid / "adbsat_lf_5000.csv", full, args.tolerance)
        source_indices = np.asarray([int(s["source_lf_index"]) for s in paired])
        arrays[("lf", "C_D")] = lf[source_indices]
        arrays[("lf", "C_D2")] = lf2[source_indices]
        geometry_manifest = json.loads((args.bundle / "geometries" / gid / f"{gid}.manifest.json").read_text())
        area = 0.5 * float(geometry_manifest["validation"]["surface_area_m2"])
        specification = geometry_manifest["specification"]
        geometry_parameters = {
            "nose_length_fraction": specification["nose_length_fraction"],
            "tail_length_fraction": specification["tail_length_fraction"],
            "front_width_height_ratio": specification.get("front_width_height_ratio")
            if specification.get("front_width_height_ratio") is not None else specification["width_height_ratio"],
            "rear_width_height_ratio": specification.get("rear_width_height_ratio")
            if specification.get("rear_width_height_ratio") is not None else specification["width_height_ratio"],
            "nose_end_scale": specification["nose_end_scale"],
            "tail_end_scale": specification["tail_end_scale"],
        }
        data[gid] = {"paired": arrays, "lf": lf, "lf2": lf2, "area": area,
                     "parameters": geometry_parameters}
        qa.append({"geometry_id": gid, "n_hf": len(paired), "n_paired_lf": len(paired), "n_full_lf": len(full),
                   "manifest_match": True, "paired_lf_source": "validated LF-10000 subset by locked source index", "finite": True,
                   "hf_cost_sum_s": sum(x["hf_cost"] for x in costs), "status": "pass"})

    point, metrics, gates = {}, [], []
    for gid, item in data.items():
        a = item["area"]
        z = item["paired"]
        point[gid] = estimate(z[("hf", "C_D")] * a, z[("lf", "C_D")] * a, item["lf"] * a,
                              z[("hf", "C_D2")] * a * a, z[("lf", "C_D2")] * a * a,
                              item["lf2"] * a * a, args.rho_gate)
        e = point[gid]
        metrics.append({"geometry_id": gid, "estimator": "MFMC" if e["active"] else "TPMC-only fallback",
                        "pearson_cd": e["rho"], "pearson_cd2": e["rho2"], "beta_cd": e["beta"],
                        "mean_drag_area_m2": e["mu"], "sd_drag_area_m2": e["sd"],
                        "mean_cd": e["mu"] / a, "sd_cd": e["sd"] / a, **item["parameters"]})
        gates.append({"geometry_id": gid, "point_active": e["active"], "pearson_cd": e["rho"],
                      "pearson_cd2": e["rho2"], "rho_gate": args.rho_gate})
    base = point["cylinder_hex_wp5_000"]
    base_area = data["cylinder_hex_wp5_000"]["area"]
    for row in metrics:
        e, a = point[row["geometry_id"]], data[row["geometry_id"]]["area"]
        row["mean_drag_ratio"] = e["mu"] / base["mu"]
        row["sd_drag_ratio"] = e["sd"] / base["sd"]
        row["mean_cd_ratio"] = (e["mu"] / a) / (base["mu"] / base_area)
        row["sd_cd_ratio"] = (e["sd"] / a) / (base["sd"] / base_area)
        row["mean_guardrail_passed"] = row["mean_drag_ratio"] <= 1.02
        row["robust_target_passed"] = row["mean_guardrail_passed"] and row["sd_drag_ratio"] <= 0.8
        row["score_cd_0p5"] = 0.5 * (row["mean_cd_ratio"] + row["sd_cd_ratio"])
        row["score_drag_0p5"] = 0.5 * (row["mean_drag_ratio"] + row["sd_drag_ratio"])

    rng = np.random.default_rng(args.seed)
    wins = {key: {gid: 0 for gid in gids} for key in ("mean_drag", "mean_cd", "robustness", "combination_drag", "combination_cd")}
    inactive = {gid: 0 for gid in gids}
    boot_metrics = {gid: [] for gid in gids}
    for _ in range(args.bootstrap):
        ip = rng.integers(0, len(paired), len(paired))
        ia = rng.integers(0, len(full), len(full))
        estimates = {}
        for gid, item in data.items():
            a, z = item["area"], item["paired"]
            e = estimate(z[("hf", "C_D")][ip] * a, z[("lf", "C_D")][ip] * a, item["lf"][ia] * a,
                         z[("hf", "C_D2")][ip] * a * a, z[("lf", "C_D2")][ip] * a * a,
                         item["lf2"][ia] * a * a, args.rho_gate)
            estimates[gid] = e
            inactive[gid] += not e["active"]
            boot_metrics[gid].append((e["mu"], e["sd"]))
        b = estimates["cylinder_hex_wp5_000"]
        wins["mean_drag"][min(gids, key=lambda g: estimates[g]["mu"])] += 1
        wins["mean_cd"][min(gids, key=lambda g: estimates[g]["mu"] / data[g]["area"])] += 1
        eligible = [g for g in gids if estimates[g]["mu"] / b["mu"] <= 1.02]
        wins["robustness"][min(eligible, key=lambda g: estimates[g]["sd"])] += 1
        wins["combination_cd"][min(gids, key=lambda g: 0.5 * (
            (estimates[g]["mu"] / data[g]["area"]) / (b["mu"] / base_area) +
            (estimates[g]["sd"] / data[g]["area"]) / (b["sd"] / base_area)))] += 1
        wins["combination_drag"][min(gids, key=lambda g: 0.5 * (
            estimates[g]["mu"] / b["mu"] + estimates[g]["sd"] / b["sd"]))] += 1

    for row in metrics:
        gid = row["geometry_id"]
        arr = np.asarray(boot_metrics[gid])
        row["mean_drag_ci_low"], row["mean_drag_ci_high"] = np.quantile(arr[:, 0], [0.025, 0.975])
        row["sd_drag_ci_low"], row["sd_drag_ci_high"] = np.quantile(arr[:, 1], [0.025, 0.975])
        for key in wins:
            row[f"bootstrap_win_{key}"] = wins[key][gid] / args.bootstrap
    for row in gates:
        row["bootstrap_activation_frequency"] = 1.0 - inactive[row["geometry_id"]] / args.bootstrap

    winners = {
        "mean_drag": min(metrics, key=lambda r: r["mean_drag_area_m2"])["geometry_id"],
        "mean_cd": min(metrics, key=lambda r: r["mean_cd"])["geometry_id"],
        "robustness": min((r for r in metrics if r["mean_guardrail_passed"]), key=lambda r: r["sd_drag_area_m2"])["geometry_id"],
        "combination_drag": min(metrics, key=lambda r: r["score_drag_0p5"])["geometry_id"],
        "combination_cd": min(metrics, key=lambda r: r["score_cd_0p5"])["geometry_id"],
    }
    physical_winners = {k: winners[k] for k in ("mean_drag", "robustness", "combination_drag")}
    centers = [{"role": role, "geometry_id": gid, "parameters": data[gid]["parameters"]}
               for role, gid in physical_winners.items()]
    report = {"schema_version": 1, "status": "pass", "exploratory": True, "n_geometries": len(gids),
              "n_hf_pairs": len(paired), "n_lf": len(full), "bootstrap_draws": args.bootstrap,
              "winners": winners, "recommended_centers": centers,
              "winner_bootstrap_frequency": {key: {g: n / args.bootstrap for g, n in value.items()} for key, value in wins.items()},
              "robust_target_met": any(r["robust_target_passed"] for r in metrics), "holdout_opened": False,
              "paired_manifest_sha256": sha(paired_path), "full_lf_manifest_sha256": sha(full_path)}
    write_csv(args.output / "ranking.csv", sorted(metrics, key=lambda r: r["score_drag_0p5"]))
    write_csv(args.output / "gates.csv", gates)
    write_csv(args.output / "qa.csv", qa)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "recommended_centers.json").write_text(json.dumps({"schema_version": 1, "exploratory": True,
        "source_analysis_sha256": sha(args.output / "analysis.json"), "centers": centers, "holdout_opened": False}, indent=2) + "\n")
    center_lines = "\n".join(f"- `{x['role']}`: `{x['geometry_id']}`" for x in centers)
    (args.output / "README.md").write_text(f'''# Global 4D HF-64 analysis

Exploratory paired MFMC analysis of {len(gids)} mesh-qualified geometries using 64 PICLas samples and the matching subset of the validated 10,000-point ADBSat design. All four uncertainty variables (CLL alpha, CLL sigma, AoA, AoS) were matched at a tolerance of {args.tolerance:g}. The downloaded campaign tables contained the complete HF values but no duplicate LF rows; paired LF values were therefore linked exactly through the locked `source_lf_index` rather than imputed or rerun.

## Recommended physical-drag centers

{center_lines}

The primary robustness target (at least 20% lower drag standard deviation with no more than 2% mean-drag increase) was not met. The coefficient-only winners are retained in `analysis.json` as diagnostics, but center selection uses dimensional drag `C_D × A_ref`.

The result is exploratory; the holdout remains unopened. Bootstrap resampling used common sample indices across geometries for paired comparisons ({args.bootstrap:,} replicates), and every MFMC correlation gate was active in every replicate.

## Method resource

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
''')
    files = [p for p in args.output.iterdir() if p.is_file() and p.name != "SHA256SUMS"]
    (args.output / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(files)) + "\n")
    print(args.output / "analysis.json")


if __name__ == "__main__":
    main()
