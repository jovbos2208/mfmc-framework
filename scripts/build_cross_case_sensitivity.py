#!/usr/bin/env python3
"""Build paper-facing Cube/GOCE sensitivity outputs from existing artifacts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values or [{"status": "no rows"}])


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def goce_missing_payload(goce_root: Path, legacy_root: Path | None) -> dict[str, Any]:
    state = json_file(goce_root / "production" / "state.json")
    availability = json_file(goce_root / "inspection" / "data_availability_report.json")
    roles_path = goce_root / "production" / "roles.json"
    required = {
        "snapshots/prepared_field_snapshots.npz": False,
        "snapshots/field_snapshot_metadata.json": False,
        "pilot/field_pilot_statistics.npz": False,
        "pilot/field_pilot_statistics.json": False,
        "production/roles.json": False,
        "allocation/optimal_allocation.json": False,
        "benchmark/benchmark_summary.csv": False,
    }
    for relative in required:
        required[relative] = (goce_root / relative).is_file()
    pilot_counts = {
        name: int(value.get("successful_results", value.get("requested", 0)))
        for name, value in (state.get("pilot_runs") or {}).items()
    }
    legacy = {}
    if legacy_root is not None:
        legacy_state = json_file(legacy_root / "production" / "state.json")
        legacy_availability = json_file(
            legacy_root / "inspection" / "data_availability_report.json"
        )
        legacy = {
            "case_directory": str(legacy_root),
            "completed_stages": legacy_state.get("completed_stages", []),
            "reported_archive_counts": {
                "DSMC": legacy_availability.get("n_dsmc", 0),
                "TPMC": legacy_availability.get("n_tpmc", 0),
                "SENTMAN": legacy_availability.get("n_sentman", 0),
                "paired": legacy_availability.get("n_paired", 0),
            },
            "reported_archive_paths": {
                name: details.get("path")
                for name, details in (legacy_availability.get("fidelities") or {}).items()
            },
            "prepared_snapshots_present": (
                legacy_root / "snapshots" / "prepared_field_snapshots.npz"
            ).is_file(),
            "benchmark_present": (
                legacy_root / "benchmark" / "benchmark_summary.csv"
            ).is_file(),
        }
    missing_roles = [
        role
        for role in ("pilot", "reference_DSMC", "production")
        if not roles_path.is_file()
    ]
    return {
        "status": "pending GOCE production",
        "requested_case_directory": str(goce_root),
        "state_completed_stages": state.get("completed_stages", []),
        "required_production_counts_from_state": state.get(
            "required_production_counts", {}
        ),
        "available_pilot_sample_counts": pilot_counts,
        "locally_reported_archive_counts": {
            "DSMC": availability.get("n_dsmc", 0) or 0,
            "TPMC": availability.get("n_tpmc", 0) or 0,
            "SENTMAN": availability.get("n_sentman", 0) or 0,
            "paired": availability.get("n_paired", 0) or 0,
        },
        "missing_fields": availability.get("missing_fields", []),
        "required_analysis_artifacts": required,
        "missing_files": [name for name, present in required.items() if not present],
        "missing_production_roles": missing_roles,
        "legacy_two_fidelity_evidence": legacy,
        "protocol_requirements": {
            "minimum_DSMC_values": [2, 4, 6, 8, 10, 12, 16, 20],
            "reference_DSMC_values": [10, 20, 30, 40, 50],
            "repetitions": 30,
            "random_seed": 20260727,
            "note": "Exact production maxima depend on the pilot/model-optimal allocation for each m0; no counts are inferred from Cube.",
        },
        "next_safe_analysis_command": (
            "python -m mfmc_campaign.field_mfpod.cli sensitivity "
            "--config configs/mfpod/goce_tpmc_sentman.yaml "
            f"--results-dir {goce_root} --repetitions 30 --random-seed 20260727"
        ),
        "jobs_submitted": False,
    }


def build(cube_root: Path, goce_root: Path, output: Path, legacy_root: Path | None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    goce_sensitivity = goce_root / "sensitivity"
    goce_sensitivity.mkdir(parents=True, exist_ok=True)
    missing = goce_missing_payload(goce_root, legacy_root)
    missing_path = goce_sensitivity / "goce_missing_artifacts.json"
    missing_path.write_text(json.dumps(missing, indent=2), encoding="utf-8")

    cube_sensitivity = cube_root / "sensitivity"
    allocations = read_csv(cube_sensitivity / "m0_allocations.csv")
    summaries = read_csv(cube_sensitivity / "m0_summary.csv")
    reference = read_csv(cube_sensitivity / "reference_convergence_summary.csv")
    reference_50 = [
        row
        for row in summaries
        if row["reference_sample_count"] == "50"
        and row["method"].startswith("field-aware-m0-")
    ]
    allocation_rows = [{"case": "Cube", "status": "complete", **row} for row in allocations]
    allocation_rows.append({"case": "GOCE", "status": "pending GOCE production"})
    metric_rows = [{"case": "Cube", "status": "complete", **row} for row in reference_50]
    metric_rows.append({"case": "GOCE", "status": "pending GOCE production"})
    reference_rows = [{"case": "Cube", "status": "complete", **row} for row in reference]
    reference_rows.append({"case": "GOCE", "status": "pending GOCE production"})
    write_csv(output / "cube_goce_allocation_comparison.csv", allocation_rows)
    write_csv(output / "cube_goce_metric_comparison.csv", metric_rows)
    write_csv(output / "cube_goce_reference_convergence.csv", reference_rows)

    x = [int(row["minimum_target"]) for row in allocations]
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    for column, label in (("n_DSMC", "Cube DSMC"), ("n_TPMC", "Cube TPMC"), ("n_SENTMAN", "Cube Sentman")):
        axis.plot(x, [int(row[column]) for row in allocations], marker="o", label=label)
    axis.text(0.98, 0.95, "GOCE pending production", transform=axis.transAxes, ha="right", va="top")
    axis.set_xlabel(r"minimum paired DSMC count $m_0$")
    axis.set_ylabel("selected sample count")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cube_goce_m0_comparison.{suffix}", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.scatter(
        [float(row["mean_field_relative_error_median"]) for row in reference_50],
        [float(row["projector_distance_fro_median"]) for row in reference_50],
        c=[int(row["minimum_target"]) for row in reference_50],
        cmap="viridis",
    )
    for row in reference_50:
        axis.annotate(
            f"m0={row['minimum_target']}",
            (float(row["mean_field_relative_error_median"]), float(row["projector_distance_fro_median"])),
            fontsize=7,
        )
    axis.set_xlabel("median mean-field relative error")
    axis.set_ylabel("median top-5 projector distance")
    axis.set_title("Cube trade-off; GOCE pending production")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cube_goce_pod_tradeoff.{suffix}", dpi=220)
    plt.close(fig)

    best_mean = min(reference_50, key=lambda row: float(row["mean_field_relative_error_median"]))
    best_cov = min(reference_50, key=lambda row: float(row["covariance_probe_relative_error_median"]))
    best_eig = min(reference_50, key=lambda row: float(row["leading_eigenvalue_mean_relative_error_median"]))
    best_pod = min(reference_50, key=lambda row: float(row["projector_distance_fro_median"]))
    all_reference_50 = [
        row for row in summaries if row["reference_sample_count"] == "50"
    ]
    baseline_rows = {
        name: next(
            row
            for row in all_reference_50
            if row["minimum_target"] == "2" and row["method"] == name
        )
        for name in (
            "DSMC-only",
            "fixed-ratios",
            "two-fidelity-TPMC",
            "scalar-drag-allocation",
        )
    }
    m02_field = next(
        row
        for row in all_reference_50
        if row["minimum_target"] == "2" and row["method"] == "field-aware-m0-2"
    )
    pod_diagnostics = json_file(cube_sensitivity / "pod_subspace_diagnostics.json")
    reference_pod = pod_diagnostics.get("reference", {})
    m02_diagnostics = (pod_diagnostics.get("methods") or {}).get(
        "field-aware-m0-2", {}
    )
    cube_state = json_file(cube_root / "production" / "state.json")
    configured_costs = cube_state.get("allocation_costs", {})
    measured_runs = {
        row.get("fidelity"): row.get("mean_cost")
        for row in cube_state.get("production_runs", [])
        if row.get("fidelity") != "DSMC" or row.get("requested") == 20
    }
    dsmc_measured = float(measured_runs.get("DSMC", 1.0))
    measured_ratios = {
        name: float(value) / dsmc_measured
        for name, value in measured_runs.items()
        if value is not None
    }
    reference_by_count = {row["reference_sample_count"]: row for row in reference}

    def interval(row: Mapping[str, str], metric: str) -> str:
        return (
            f"{float(row[f'{metric}_median']):.6g} "
            f"[IQR {float(row[f'{metric}_q25']):.6g}, "
            f"{float(row[f'{metric}_q75']):.6g}]"
        )

    findings = f"""# Cube--GOCE field-MFPOD sensitivity

## 1. Data and validation status

Cube is complete and uses disjoint pilot, production, and independent DSMC-reference roles. GOCE is **pending GOCE production**; no GOCE sensitivity metric is estimated.

## 2. Allocation behavior

Cube allocations are bootstrap-robust selected under the configured scenario costs. Sentman has zero selected production samples for all tested Cube m0 values. This is conditional evidence, not a general statement about Sentman.

## 3. Cube sensitivity results

Against the 50-DSMC numerical reference, the metric-specific optima are m0={best_mean['minimum_target']} for mean-field error ({interval(best_mean, 'mean_field_relative_error')}), m0={best_cov['minimum_target']} for covariance-probe error ({interval(best_cov, 'covariance_probe_relative_error')}), m0={best_eig['minimum_target']} for leading-eigenvalue error ({interval(best_eig, 'leading_eigenvalue_mean_relative_error')}), and m0={best_pod['minimum_target']} for projector distance ({interval(best_pod, 'projector_distance_fro')}).

At m0=2, the field-aware and two-fidelity-TPMC mean errors are respectively {interval(m02_field, 'mean_field_relative_error')} and {interval(baseline_rows['two-fidelity-TPMC'], 'mean_field_relative_error')}; their near identity is consistent with zero selected Sentman samples. The scalar-drag allocation gives {interval(baseline_rows['scalar-drag-allocation'], 'mean_field_relative_error')}, while fixed ratios give {interval(baseline_rows['fixed-ratios'], 'mean_field_relative_error')}.

## 4. GOCE sensitivity results

Pending GOCE production. The exact missing files, roles, and reported sample counts are recorded in `goce_missing_artifacts.json`.

## 5. Cross-geometry comparison

Not yet decidable. In particular, m0=6 cannot be called transferable until GOCE has passed the identical protocol.

## 6. POD-subspace diagnosis

Moment/eigenvalue accuracy and top-five POD-subspace accuracy rank allocations differently. The 50-field reference has lambda5-lambda6={float(reference_pod.get('lambda5_minus_lambda6', float('nan'))):.6g}. For m0=2, the normalized top-five/complement coupling is {float(m02_diagnostics.get('top5_to_complement_coupling_fro', float('nan'))):.6g}, corresponding to a coupling/gap ratio of {float(m02_diagnostics.get('coupling_to_lambda5_lambda6_gap', float('nan'))):.6g}; this supports sensitivity of the retained subspace even when moment metrics improve. Global scalar weights cannot correct local or direction-dependent TPMC errors.

## 7. Reference convergence

The nested-prefix analysis measures convergence toward the finite 50-field numerical reference. At 10 fields, mean error is {interval(reference_by_count['10'], 'mean_field_relative_error')} and projector distance is {interval(reference_by_count['10'], 'projector_distance_fro')}; at 40 fields these become {interval(reference_by_count['40'], 'mean_field_relative_error')} and {interval(reference_by_count['40'], 'projector_distance_fro')}. This does not establish convergence of 50 fields to the infinite DSMC population.

## 8. Cost interpretation

Allocation counts use configured scenario costs {json.dumps(configured_costs, sort_keys=True)}. Separately, the production-state mean CPU-hour ratios relative to the 20-sample DSMC production block are {json.dumps(measured_ratios, sort_keys=True)}. No measured-cost sensitivity allocation was run, so whether the allocation conclusion changes under measured ratios remains undecidable.

## 9. Supported findings

- **Robustly supported:** Cube allocation and metric summaries are reproducible under seed 20260727 and 30 permutations; every tested m0 selected Sentman in 0/500 bootstrap resamples.
- **Partly supported:** Under the prescribed Cube field objective and cost scenario, Sentman provides no selected conditional benefit after TPMC. This is case- and objective-conditional.
- **Not yet decidable:** Transfer to GOCE and sensitivity to measured rather than configured costs.

## 10. Findings not supported

- Global optimality, universal MFMC superiority, universal Sentman uselessness, and physical-ground-truth language are not supported.

## 11. Limitations

Finite pilot and reference ensembles, one Cube cost scenario, global scalar control weights, and missing GOCE production limit generalization.

## 12. Recommended manuscript claims

"Under the prescribed Cube field objective and cost scenario, Sentman provided no selected conditional variance-reduction benefit after TPMC. The allocation improved selected moment metrics, while POD-subspace metrics exhibited a distinct paired-sample trade-off."

## 13. Claims that must not be made

Do not claim global optimality, 50 DSMC as ground truth, universal superiority over DSMC-only, or equivalence of configured costs and measured CPU-hours.

## 14. Recommended tables and figures

Use the generated allocation/metric/reference CSVs, the m0 comparison, the POD trade-off, and Cube's operator diagnostics. Mark every GOCE cell as pending.

## 15. Remaining experiments

Complete GOCE production and independent reference archives, prepare snapshots with disjoint roles, then run the exact predeclared command in `goce_missing_artifacts.json`. A separate measured-cost sensitivity can then be compared with the scenario-cost result.
"""
    (output / "paper_findings.md").write_text(findings, encoding="utf-8")
    tex_lines = []
    for line in findings.splitlines():
        if line.startswith("## "):
            tex_lines.append(f"\\subsection*{{{line[3:]}}}")
        elif line.startswith("# "):
            tex_lines.append(f"\\section*{{{line[2:]}}}")
        else:
            tex_lines.append(
                line.replace("%", "\\%")
                .replace("_", "\\_")
                .replace("m0=", "$m_0=$")
                .replace("lambda5--lambda6", "$\\lambda_5-\\lambda_6$")
            )
    (output / "paper_findings.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )
    table_lines = [
        r"\begin{tabular}{rrrrrr}",
        r"$m_0$ & $n_H$ & $n_T$ & $n_S$ & mean error & projector distance \\",
        r"\hline",
    ]
    by_m0 = {row["minimum_target"]: row for row in reference_50}
    for row in allocations:
        metric = by_m0[row["minimum_target"]]
        table_lines.append(
            f"{row['minimum_target']} & {row['n_DSMC']} & {row['n_TPMC']} & {row['n_SENTMAN']} & "
            f"{float(metric['mean_field_relative_error_median']):.5g} & "
            f"{float(metric['projector_distance_fro_median']):.5g} \\\\"
        )
    table_lines.extend([r"\hline", r"\multicolumn{6}{l}{GOCE: pending production} \\", r"\end{tabular}"])
    (output / "generated_results_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    inputs = [
        cube_sensitivity / "m0_allocations.csv",
        cube_sensitivity / "m0_summary.csv",
        cube_sensitivity / "reference_convergence_summary.csv",
        cube_sensitivity / "pod_subspace_diagnostics.json",
        goce_root / "production" / "state.json",
        goce_root / "inspection" / "data_availability_report.json",
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cube_status": "complete",
        "goce_status": "pending GOCE production",
        "numbers_generated_from_artifacts": True,
        "jobs_submitted": False,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output": str(output), "goce_status": missing["status"], "files": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-root", type=Path, required=True)
    parser.add_argument("--goce-root", type=Path, required=True)
    parser.add_argument("--legacy-goce-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.cube_root.resolve(), args.goce_root.resolve(), args.output.resolve(), args.legacy_goce_root.resolve() if args.legacy_goce_root else None), indent=2))


if __name__ == "__main__":
    main()
