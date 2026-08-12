"""Offline DSMC--TPMC-only MFPOD sensitivity analysis.

This module deliberately names every accepted array.  It never discovers or
loads a third model from an archive and never invokes a solver or scheduler.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .allocation import AllocationOptions, AllocationResult, optimize_allocation, optimize_field_allocation
from .covariance_operator import (
    covariance_probe_error,
    estimate_full_field_mfmc,
    explicit_full_field_covariance,
    solve_full_field_pod,
)
from .field_validation import leading_eigenvalue_error, pod_validation_metrics, relative_field_error
from .models import MFPODError, jsonable
from .sensitivity import _nested_prefix_indices, _permuted_nested_fields


M0_VALUES = (2, 4, 6, 8, 10, 12, 16, 20)
REFERENCE_SIZES = (10, 20, 30, 40, 50)
METRICS = (
    "mean_field_relative_error",
    "covariance_probe_relative_error",
    "leading_eigenvalue_mean_relative_error",
    "maximum_principal_angle_rad",
    "rms_principal_angle_rad",
    "projector_distance_fro",
    "heldout_projection_error",
    "reference_projection_error",
    "minimum_ritz_eigenvalue",
    "negative_eigenvalue_count",
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no rows"}])


def _hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def budget_tpmc_count(m0: int, *, budget: float, dsmc_cost: float, tpmc_cost: float, pool: int) -> int:
    """Budget-filling control count, capped by the available pool."""
    if m0 < 1 or budget <= 0 or dsmc_cost <= 0 or tpmc_cost <= 0 or pool < 0:
        raise MFPODError("Invalid count, budget, cost, or pool")
    remaining = budget - m0 * dsmc_cost
    return min(pool, max(0, int(np.floor((remaining + 1.0e-12) / tpmc_cost))))


def _required_paths(root: Path) -> dict[str, bool]:
    names = (
        "production/state.json",
        "production/roles.json",
        "snapshots/prepared_field_snapshots.npz",
        "snapshots/field_snapshot_metadata.json",
        "pilot/field_pilot_statistics.npz",
        "pilot/field_pilot_statistics.json",
        "inspection/data_availability_report.json",
        "benchmark/benchmark_summary.csv",
        "resolved_config.yaml",
    )
    return {name: (root / name).is_file() for name in names}


def missing_artifacts(root: str | Path, *, requested_m0: Sequence[int] = M0_VALUES) -> dict[str, Any]:
    root = Path(root)
    paths = _required_paths(root)
    counts = {"pilot_pairs": 0, "production_DSMC": 0, "production_TPMC": 0, "reference_DSMC": 0}
    count_sources = {name: None for name in counts}
    prepared = root / "snapshots/prepared_field_snapshots.npz"
    if prepared.is_file():
        with np.load(prepared, allow_pickle=False) as data:
            counts = {
                "pilot_pairs": min(len(data["pilot_DSMC"]), len(data["pilot_TPMC"])),
                "production_DSMC": len(data["production_DSMC"]),
                "production_TPMC": len(data["production_TPMC"]),
                "reference_DSMC": len(data["reference_DSMC"]),
            }
        count_sources = {name: str(prepared.resolve()) for name in counts}
    else:
        pilot_fields = root / "production/pilot_fields.npz"
        if pilot_fields.is_file():
            with np.load(pilot_fields, allow_pickle=False) as data:
                if "DSMC" in data.files and "TPMC" in data.files:
                    counts["pilot_pairs"] = min(len(data["DSMC"]), len(data["TPMC"]))
                    count_sources["pilot_pairs"] = str(pilot_fields.resolve())

    declared_roles: dict[str, int] = {}
    roles_path = root / "production/roles.json"
    if roles_path.is_file():
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
        production_roles = roles.get("production", {})
        declared_roles = {
            "pilot_pairs": len(roles.get("pilot", [])),
            "production_DSMC": len(production_roles.get("DSMC", [])),
            "production_TPMC": len(production_roles.get("TPMC", [])),
            "reference_DSMC": len(roles.get("reference_DSMC", [])),
        }

    inaccessible_archives: dict[str, str] = {}
    availability_path = root / "inspection/data_availability_report.json"
    if availability_path.is_file():
        availability = json.loads(availability_path.read_text(encoding="utf-8"))
        for model in ("DSMC", "TPMC"):
            archive = Path(str((availability.get("fidelities", {}).get(model, {}) or {}).get("path", "")))
            if str(archive) and not archive.is_file():
                inaccessible_archives[model] = str(archive)
    feasible = []
    missing_by_m0 = {}
    declared_count_feasible = []
    declared_missing_by_m0: dict[str, dict[str, int]] = {}
    for m0 in requested_m0:
        need_tpmc = budget_tpmc_count(m0, budget=20.0, dsmc_cost=1.0, tpmc_cost=0.1, pool=180)
        absent = {}
        if counts["production_DSMC"] < m0:
            absent["production_DSMC"] = m0 - counts["production_DSMC"]
        if counts["production_TPMC"] < need_tpmc:
            absent["production_TPMC"] = need_tpmc - counts["production_TPMC"]
        if counts["reference_DSMC"] < 50:
            absent["reference_DSMC"] = 50 - counts["reference_DSMC"]
        if counts["pilot_pairs"] < 30:
            absent["pilot_pairs"] = 30 - counts["pilot_pairs"]
        missing_by_m0[str(m0)] = absent
        if not absent and all(paths.values()):
            feasible.append(m0)
        if declared_roles:
            declared_absent = {}
            for name, needed in (
                ("production_DSMC", m0),
                ("production_TPMC", need_tpmc),
                ("reference_DSMC", 50),
                ("pilot_pairs", 30),
            ):
                if declared_roles.get(name, 0) < needed:
                    declared_absent[name] = needed - declared_roles.get(name, 0)
            declared_missing_by_m0[str(m0)] = declared_absent
            if not declared_absent:
                declared_count_feasible.append(m0)
    return {
        "case_root": str(root.resolve()),
        "status": "ready" if len(feasible) == len(tuple(requested_m0)) else "incomplete",
        "required_artifacts": paths,
        "available_counts": counts,
        "available_count_sources": count_sources,
        "declared_role_counts": declared_roles,
        "declared_count_feasible_if_archives_restored": declared_count_feasible,
        "declared_count_missing_by_m0": declared_missing_by_m0,
        "inaccessible_field_archives": inaccessible_archives,
        "feasible_m0": feasible,
        "missing_by_m0": missing_by_m0,
        "action": "No simulation was started; supply the listed existing artifacts before postprocessing.",
    }


def _load(root: Path) -> dict[str, Any]:
    missing = [name for name, present in _required_paths(root).items() if not present]
    if missing:
        raise MFPODError("Missing required artifacts: " + ", ".join(missing))
    prepared = root / "snapshots/prepared_field_snapshots.npz"
    pilot_stats = root / "pilot/field_pilot_statistics.npz"
    allowed = (
        "pilot_DSMC", "pilot_TPMC", "production_DSMC", "production_TPMC",
        "production_ids_DSMC", "production_ids_TPMC", "reference_DSMC",
    )
    with np.load(prepared, allow_pickle=False) as data:
        absent = [name for name in allowed if name not in data.files]
        if absent:
            raise MFPODError("Prepared snapshot archive lacks: " + ", ".join(absent))
        values = {name: np.asarray(data[name]) for name in allowed}
        values["pilot_drag"] = {
            name: np.asarray(data[f"pilot_CD_{name}"], dtype=float)
            for name in ("DSMC", "TPMC") if f"pilot_CD_{name}" in data.files
        }
        pilot_ids = np.asarray(data["pilot_sample_ids"]).astype(str)
        reference_ids = np.asarray(data["reference_sample_ids"]).astype(str)
    with np.load(pilot_stats, allow_pickle=False) as data:
        values["reference_field"] = np.asarray(data["reference_field"], dtype=float)
    values["pilot"] = {name: np.asarray(values[f"pilot_{name}"], dtype=float) for name in ("DSMC", "TPMC")}
    values["production"] = {name: np.asarray(values[f"production_{name}"], dtype=float) for name in ("DSMC", "TPMC")}
    values["ids"] = {name: np.asarray(values[f"production_ids_{name}"]).astype(str) for name in ("DSMC", "TPMC")}
    values["reference"] = np.asarray(values["reference_DSMC"], dtype=float)
    roles = json.loads((root / "production/roles.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "snapshots/field_snapshot_metadata.json").read_text(encoding="utf-8"))
    availability = json.loads((root / "inspection/data_availability_report.json").read_text(encoding="utf-8"))
    if min(map(len, values["pilot"].values())) < 30 or len(values["production"]["DSMC"]) < 20 or len(values["reference"]) != 50:
        raise MFPODError("Need >=30 pilot pairs, >=20 production DSMC, and exactly 50 reference DSMC fields")
    if len(set(values["ids"]["DSMC"])) != len(values["ids"]["DSMC"]) or len(set(values["ids"]["TPMC"])) != len(values["ids"]["TPMC"]):
        raise MFPODError("Duplicate production sample IDs")
    pilot_set, ref_set = set(pilot_ids), set(reference_ids)
    prod_set = set(values["ids"]["DSMC"]) | set(values["ids"]["TPMC"])
    if pilot_set & ref_set or pilot_set & prod_set or ref_set & prod_set:
        raise MFPODError("Pilot, production, and reference roles overlap")
    topo = (availability.get("topology") or availability.get("topology_by_model") or {}).get("TPMC", {})
    checks = ("same_face_count", "same_geometry_id", "same_coordinate_frame", "same_component_order", "same_face_areas", "same_face_centers")
    if any(topo.get(name) is not True for name in checks):
        raise MFPODError("DSMC/TPMC geometry or ordering validation failed")
    dimension = values["production"]["DSMC"].shape[1]
    arrays = (*values["pilot"].values(), *values["production"].values(), values["reference"])
    if any(a.ndim != 2 or a.shape[1] != dimension for a in arrays):
        raise MFPODError("Full fields do not share one state dimension")
    values.update({"roles": roles, "metadata": metadata, "availability": availability, "prepared": prepared})
    return values


def _locked(pilot: Mapping[str, np.ndarray], m0: int, n_tpmc: int, costs: Mapping[str, float]) -> AllocationResult:
    if n_tpmc == 0:
        return optimize_field_allocation(
            {"DSMC": pilot["DSMC"]}, {"DSMC": costs["DSMC"]},
            AllocationOptions(budget=m0 * costs["DSMC"], minimum_target=m0, maximum_counts={"DSMC": m0}, max_ratios={}, mode="enumeration"),
        )
    budget = m0 * costs["DSMC"] + n_tpmc * costs["TPMC"]
    return optimize_field_allocation(
        pilot, costs,
        AllocationOptions(
            budget=budget, minimum_target=m0, minimum_counts={"TPMC": n_tpmc},
            maximum_counts={"DSMC": m0, "TPMC": n_tpmc}, min_ratios={}, max_ratios={},
            mode="enumeration", bootstrap_repeats=0, random_seed=20260727,
        ),
    )


def _comparators(pools: Mapping[str, Any], costs: Mapping[str, float], budget: float) -> dict[str, AllocationResult]:
    pilot, max_counts = pools["pilot"], {name: len(pools["production"][name]) for name in ("DSMC", "TPMC")}
    result = {"DSMC-only": _locked(pilot, min(20, max_counts["DSMC"]), 0, costs)}
    fixed_dsmc = min(max_counts["DSMC"], int(np.floor(budget / (costs["DSMC"] + 5 * costs["TPMC"]))))
    result["fixed-ratio-5"] = _locked(pilot, fixed_dsmc, min(max_counts["TPMC"], 5 * fixed_dsmc), costs)
    options = AllocationOptions(
        budget=budget, minimum_target=2, maximum_counts=max_counts, min_ratios={}, max_ratios={},
        mode="continuous_round", bootstrap_repeats=0, random_seed=20260727,
    )
    result["model-selected"] = optimize_field_allocation(pilot, costs, options)
    if set(pools["pilot_drag"]) == {"DSMC", "TPMC"}:
        scalar = optimize_allocation(pools["pilot_drag"], costs, options)
        result["scalar-drag-allocation"] = _locked(pilot, scalar.counts["DSMC"], scalar.counts.get("TPMC", 0), costs)
    return result


def _evaluate(fields: Mapping[str, np.ndarray], allocation: AllocationResult, ref_stats: Any, ref_pod: Any, full_ref: np.ndarray, reference_field: np.ndarray, seed: int, probe_count: int = 100) -> dict[str, Any]:
    counts = {name: int(count) for name, count in allocation.counts.items() if count > 0}
    estimate = estimate_full_field_mfmc(
        {name: fields[name] for name in counts}, counts, reference_field=reference_field,
        mean_weights=(allocation.control_weights or {}).get("mean", {}),
        second_moment_weights=(allocation.control_weights or {}).get("second_moment", {}),
    )
    pod = solve_full_field_pod(estimate, n_modes=5, random_seed=seed)
    sub = pod_validation_metrics(pod.modes, ref_pod.modes, full_ref, estimated_mean=estimate.mean_field, reference_mean=ref_stats.mean_field)
    eig = leading_eigenvalue_error(pod.eigenvalues, ref_pod.eigenvalues)
    return {
        "mean_field_relative_error": relative_field_error(estimate.mean_field, ref_stats.mean_field),
        "covariance_probe_relative_error": covariance_probe_error(estimate.covariance, ref_stats.covariance, probe_count=probe_count, random_seed=4401),
        "leading_eigenvalue_mean_relative_error": eig["mean_relative_error"],
        "maximum_principal_angle_rad": sub["maximum_principal_angle_rad"],
        "rms_principal_angle_rad": sub["rms_principal_angle_rad"],
        "projector_distance_fro": sub["projector_distance_fro"],
        "heldout_projection_error": sub["projection_error"],
        "reference_projection_error": sub["reference_projection_error"],
        "minimum_ritz_eigenvalue": pod.diagnostics["minimum_computed_ritz_eigenvalue"],
        "negative_eigenvalue_count": pod.diagnostics["negative_eigenvalue_count"],
    }


def _tpmc_only(fields: np.ndarray, count: int, ref_stats: Any, ref_pod: Any, full_ref: np.ndarray, reference_field: np.ndarray, seed: int) -> dict[str, Any]:
    pseudo = estimate_full_field_mfmc({"DSMC": fields}, {"DSMC": count}, reference_field=reference_field)
    allocation = AllocationResult({"DSMC": count}, float(count), 0.0, "non-target", True, {}, [])
    return _evaluate({"DSMC": fields}, allocation, ref_stats, ref_pod, full_ref, reference_field, seed)


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["m0"]), int(row["reference_sample_count"]), str(row["method"])), []).append(row)
    output = []
    for key, values in sorted(grouped.items()):
        m0, nref, method = key
        first = values[0]
        summary = {name: first[name] for name in ("case", "m0", "reference_sample_count", "method", "method_role", "n_DSMC", "n_TPMC", "configured_cost", "measured_cost_cpu_hours", "beta_mean_TPMC", "beta_second_moment_TPMC")}
        summary["repetitions"] = len(values)
        for metric in METRICS:
            data = np.asarray([float(row[metric]) for row in values])
            for label, value in (("median", np.median(data)), ("q25", np.quantile(data, .25)), ("q75", np.quantile(data, .75)), ("minimum", np.min(data)), ("maximum", np.max(data)), ("mean", np.mean(data)), ("standard_deviation", np.std(data, ddof=1))):
                summary[f"{metric}_{label}"] = float(value)
            for baseline, label in (("DSMC-only", "dsmc_only"), ("fixed-ratio-5", "fixed_ratio_5")):
                peers = {int(row["repetition"]): row for row in grouped.get((m0, nref, baseline), [])}
                paired = [float(row[metric]) < float(peers[int(row["repetition"])][metric]) for row in values if int(row["repetition"]) in peers and method != baseline]
                summary[f"{metric}_paired_win_rate_vs_{label}"] = float(np.mean(paired)) if paired else None
        output.append(summary)
    return output


def _reference_convergence(reference: np.ndarray, reference_field: np.ndarray, repetitions: int, seed: int) -> tuple[list[dict], list[dict]]:
    full = estimate_full_field_mfmc({"DSMC": reference}, {"DSMC": 50}, reference_field=reference_field)
    full_pod = solve_full_field_pod(full, n_modes=5, random_seed=seed)
    rows = []
    for rep in range(repetitions):
        prefixes = _nested_prefix_indices(50, REFERENCE_SIZES, random_seed=seed + 100003 + rep)
        for count, indices in prefixes.items():
            subset = reference[indices]
            stats = estimate_full_field_mfmc({"DSMC": subset}, {"DSMC": count}, reference_field=reference_field)
            pod = solve_full_field_pod(stats, n_modes=5, random_seed=seed + rep)
            sub = pod_validation_metrics(pod.modes, full_pod.modes, reference, estimated_mean=stats.mean_field, reference_mean=full.mean_field)
            eig = leading_eigenvalue_error(pod.eigenvalues, full_pod.eigenvalues)
            rows.append({
                "repetition": rep, "reference_sample_count": count,
                "mean_field_relative_error": relative_field_error(stats.mean_field, full.mean_field),
                "covariance_probe_relative_error": covariance_probe_error(stats.covariance, full.covariance, probe_count=100, random_seed=4401),
                "leading_eigenvalue_mean_relative_error": eig["mean_relative_error"],
                **{name: sub[name] for name in ("maximum_principal_angle_rad", "rms_principal_angle_rad", "projector_distance_fro")},
                "heldout_projection_error": sub["projection_error"], "reference_projection_error": sub["reference_projection_error"],
                "minimum_ritz_eigenvalue": pod.diagnostics["minimum_computed_ritz_eigenvalue"], "negative_eigenvalue_count": pod.diagnostics["negative_eigenvalue_count"],
            })
    summaries = []
    for count in REFERENCE_SIZES:
        vals = [row for row in rows if row["reference_sample_count"] == count]
        item = {"reference_sample_count": count, "repetitions": len(vals)}
        for metric in METRICS:
            data = np.asarray([float(row[metric]) for row in vals])
            for label, value in (("median", np.median(data)), ("q25", np.quantile(data,.25)), ("q75",np.quantile(data,.75)), ("minimum",np.min(data)), ("maximum",np.max(data)), ("mean",np.mean(data)), ("standard_deviation",np.std(data,ddof=1))):
                item[f"{metric}_{label}"] = float(value)
        summaries.append(item)
    return rows, summaries


def _diagnostics(pools: Mapping[str, Any], allocations: Mapping[int, AllocationResult], seed: int) -> dict[str, Any]:
    reference = estimate_full_field_mfmc({"DSMC": pools["reference"]}, {"DSMC": 50}, reference_field=pools["reference_field"])
    ref_pod = solve_full_field_pod(reference, n_modes=6, random_seed=seed)
    eigenvalues = np.asarray(ref_pod.eigenvalues)
    payload = {"reference_eigenvalues": eigenvalues, "reference_eigenvalue_gaps": eigenvalues[:-1]-eigenvalues[1:], "methods": {}}
    for m0, allocation in allocations.items():
        fields = _permuted_nested_fields(pools["production"], pools["ids"], allocation.counts, random_seed=seed)
        estimate = estimate_full_field_mfmc(fields, allocation.counts, reference_field=pools["reference_field"], mean_weights=(allocation.control_weights or {}).get("mean",{}), second_moment_weights=(allocation.control_weights or {}).get("second_moment",{}))
        pod = solve_full_field_pod(estimate, n_modes=5, random_seed=seed)
        angles = pod_validation_metrics(pod.modes, ref_pod.modes[:,:5], pools["reference"], estimated_mean=estimate.mean_field, reference_mean=reference.mean_field)
        terms = estimate.diagnostics["operator_terms"]
        item = {"signed_operator_coefficients": terms, "principal_angles_rad": angles["principal_angles_rad"]}
        if estimate.covariance.shape[0] <= 2000:
            cov, ref_cov = explicit_full_field_covariance(estimate), explicit_full_field_covariance(reference)
            error = cov-ref_cov; basis=ref_pod.modes[:,:5]; block=basis.T@error@basis; coupling=error@basis-basis@block
            gap=float(eigenvalues[4]-eigenvalues[5])
            centered = {name: fields[name][: allocation.counts[name]] - pools["reference_field"] for name in allocation.counts}
            n_h = allocation.counts["DSMC"]
            contributions = [("DSMC paired", centered["DSMC"][:n_h].T @ centered["DSMC"][:n_h] / n_h)]
            beta = (allocation.control_weights or {}).get("second_moment", {}).get("TPMC", 0.0)
            if allocation.counts.get("TPMC", 0) and beta:
                n_l = allocation.counts["TPMC"]
                contributions.extend((("TPMC all", beta * centered["TPMC"].T @ centered["TPMC"] / n_l), ("TPMC paired", -beta * centered["TPMC"][:n_h].T @ centered["TPMC"][:n_h] / n_h)))
            contributions.append(("estimated mean outer product", -np.outer(estimate.centered_mean, estimate.centered_mean)))
            contribution_norms = {label: float(np.linalg.norm(value, "fro")) for label, value in contributions}
            norm_sum = sum(contribution_norms.values())
            item.update({
                "explicit_matrix_formed": True,
                "relative_frobenius_covariance_error": float(np.linalg.norm(error,"fro")/np.linalg.norm(ref_cov,"fro")),
                "relative_spectral_covariance_error": float(np.linalg.norm(error,2)/np.linalg.norm(ref_cov,2)),
                "top5_block_error_fro": float(np.linalg.norm(block,"fro")/np.linalg.norm(ref_cov,"fro")),
                "top5_off_diagonal_error_fro": float(np.linalg.norm(block-np.diag(np.diag(block)),"fro")/np.linalg.norm(ref_cov,"fro")),
                "top5_to_complement_coupling_fro": float(np.linalg.norm(coupling,"fro")/np.linalg.norm(ref_cov,"fro")),
                "lambda5_minus_lambda6": gap, "coupling_to_gap": float(np.linalg.norm(coupling,2)/gap) if gap>0 else None,
                "signed_contribution_frobenius_norms": contribution_norms,
                "signed_operator_cancellation_ratio_fro": float(np.linalg.norm(cov,"fro")/norm_sum) if norm_sum else None,
            })
        else:
            item.update({"explicit_matrix_formed": False, "matrix_free": True, "covariance_probe_seed": 4401})
        payload["methods"][f"fixed-m0-{m0}"] = item
    return payload


def _figures(out: Path, case: str, allocations: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], refs: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt
    paths=[]; x=np.asarray([row["m0"] for row in allocations])
    fig,ax=plt.subplots(figsize=(6.4,3.8)); ax.plot(x,[row["n_DSMC"] for row in allocations],"o-",label="DSMC"); ax.plot(x,[row["n_TPMC"] for row in allocations],"o-",label="TPMC"); ax.set(xlabel="$m_0$",ylabel="production count",title=f"{case}: fixed two-fidelity allocation counts"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    for ext in ("png","pdf"): p=out/f"m0_allocation_counts.{ext}"; fig.savefig(p,dpi=220); paths.append(str(p))
    plt.close(fig)
    selected=sorted([r for r in summaries if r["reference_sample_count"]==50 and r["method"].startswith("fixed-m0-")],key=lambda r:r["m0"])
    plot_metrics=("mean_field_relative_error","covariance_probe_relative_error","leading_eigenvalue_mean_relative_error","maximum_principal_angle_rad","projector_distance_fro","heldout_projection_error")
    fig,axes=plt.subplots(2,3,figsize=(12,6.4),sharex=True)
    for ax,metric in zip(axes.flat,plot_metrics):
        med=np.asarray([r[f"{metric}_median"] for r in selected]); lo=np.asarray([r[f"{metric}_q25"] for r in selected]); hi=np.asarray([r[f"{metric}_q75"] for r in selected]); ax.plot(x,med,"o-"); ax.fill_between(x,lo,hi,alpha=.2); ax.set_title(metric.replace("_"," ")); ax.grid(alpha=.25)
    fig.suptitle(f"{case}: validation against the independent 50-DSMC numerical reference"); fig.tight_layout()
    for ext in ("png","pdf"): p=out/f"m0_validation_metrics_reference_50.{ext}"; fig.savefig(p,dpi=220); paths.append(str(p))
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8.4,6.4),sharex=True); rx=np.asarray([r["reference_sample_count"] for r in refs])
    for ax,metric in zip(axes.flat,plot_metrics[:3]+("projector_distance_fro",)):
        med=np.asarray([r[f"{metric}_median"] for r in refs]); lo=np.asarray([r[f"{metric}_q25"] for r in refs]); hi=np.asarray([r[f"{metric}_q75"] for r in refs]); ax.plot(rx,med,"o-"); ax.fill_between(rx,lo,hi,alpha=.2); ax.set_title(metric.replace("_"," ")); ax.grid(alpha=.25)
    fig.suptitle(f"{case}: independent DSMC-reference convergence against all 50 fields"); fig.tight_layout()
    for ext in ("png","pdf"): p=out/f"reference_convergence.{ext}"; fig.savefig(p,dpi=220); paths.append(str(p))
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4.2)); ax.plot(x,[r["leading_eigenvalue_mean_relative_error_median"] for r in selected],"o-",label="eigenvalue error"); ax.plot(x,[r["projector_distance_fro_median"] for r in selected],"s-",label="projector distance"); ax.set(xlabel="$m_0$",title=f"{case}: moment versus POD-subspace trade-off"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    for ext in ("png","pdf"): p=out/f"pod_subspace_tradeoff.{ext}"; fig.savefig(p,dpi=220); paths.append(str(p))
    plt.close(fig); return paths


def run_two_fidelity_sensitivity(root: str | Path, *, case_name: str, repetitions: int = 30, random_seed: int = 20260727, costs: Mapping[str,float] | None = None, budget: float = 20.0) -> dict[str, Any]:
    root=Path(root).resolve(); pools=_load(root); out=root/"sensitivity_two_fidelity"; out.mkdir(parents=True,exist_ok=True)
    costs={"DSMC":1.0,"TPMC":0.1,**(costs or {})}; available_tpmc=len(pools["production"]["TPMC"])
    allocations={m0:_locked(pools["pilot"],m0,budget_tpmc_count(m0,budget=budget,dsmc_cost=costs["DSMC"],tpmc_cost=costs["TPMC"],pool=available_tpmc),costs) for m0 in M0_VALUES}
    expected={m0:(m0,max(0,200-10*m0)) for m0 in M0_VALUES}
    if costs=={"DSMC":1.0,"TPMC":0.1} and available_tpmc>=180 and any((a.counts.get("DSMC",0),a.counts.get("TPMC",0))!=expected[m0] for m0,a in allocations.items()):
        raise MFPODError("Programmatic allocation-count verification failed")
    comparators=_comparators(pools,costs,budget); rows=[]
    for rep in range(repetitions):
        ref_prefixes=_nested_prefix_indices(50,REFERENCE_SIZES,random_seed=random_seed+50021+rep)
        refs={}
        for n,idx in ref_prefixes.items():
            fields=pools["reference"][idx]; stats=estimate_full_field_mfmc({"DSMC":fields},{"DSMC":n},reference_field=pools["reference_field"]); pod=solve_full_field_pod(stats,n_modes=5,random_seed=random_seed+rep); refs[n]=(stats,pod)
        method_alloc={**{f"fixed-m0-{m0}":a for m0,a in allocations.items()},**comparators}
        evaluated={}
        for method,allocation in method_alloc.items():
            fields=_permuted_nested_fields(pools["production"],pools["ids"],allocation.counts,random_seed=random_seed+rep)
            for n,(ref_stats,ref_pod) in refs.items(): evaluated[(method,n)]=_evaluate(fields,allocation,ref_stats,ref_pod,pools["reference"],pools["reference_field"],random_seed+rep)
        rng=np.random.default_rng(random_seed+rep); tpmc_order=rng.permutation(available_tpmc); tpmc_count=min(available_tpmc,int(np.floor(budget/costs["TPMC"])))
        for n,(ref_stats,ref_pod) in refs.items(): evaluated[("TPMC-only-non-target",n)]=_tpmc_only(pools["production"]["TPMC"][tpmc_order],tpmc_count,ref_stats,ref_pod,pools["reference"],pools["reference_field"],random_seed+rep)
        for m0 in M0_VALUES:
            for method in (f"fixed-m0-{m0}", *comparators, "TPMC-only-non-target"):
                allocation=method_alloc.get(method); n_dsmc=0 if allocation is None else allocation.counts.get("DSMC",0); n_tpmc=tpmc_count if allocation is None else allocation.counts.get("TPMC",0)
                for n in REFERENCE_SIZES:
                    rows.append({"case":case_name,"m0":m0,"repetition":rep,"production_seed":random_seed+rep,"reference_sample_count":n,"method":method,"method_role":"non-DSMC-target baseline" if method=="TPMC-only-non-target" else "DSMC-target estimator","n_DSMC":n_dsmc,"n_TPMC":n_tpmc,"configured_cost":n_dsmc*costs["DSMC"]+n_tpmc*costs["TPMC"],"measured_cost_cpu_hours":None,"beta_mean_TPMC":None if allocation is None else (allocation.control_weights or {}).get("mean",{}).get("TPMC",0.0),"beta_second_moment_TPMC":None if allocation is None else (allocation.control_weights or {}).get("second_moment",{}).get("TPMC",0.0),**evaluated[(method,n)]})
    summaries=_summaries(rows); ref_rows,ref_summaries=_reference_convergence(pools["reference"],pools["reference_field"],repetitions,random_seed)
    allocation_rows=[{"case":case_name,"m0":m0,"n_DSMC":a.counts.get("DSMC",0),"n_TPMC":a.counts.get("TPMC",0),"configured_cost":a.total_cost,"measured_cost_cpu_hours":None,"beta_mean_TPMC":(a.control_weights or {}).get("mean",{}).get("TPMC",0.0),"beta_second_moment_TPMC":(a.control_weights or {}).get("second_moment",{}).get("TPMC",0.0),"allocation_verified":(a.counts.get("DSMC",0)==m0)} for m0,a in allocations.items()]
    _csv(out/"m0_allocations.csv",allocation_rows); _csv(out/"m0_repetitions.csv",rows); _csv(out/"m0_summary.csv",summaries); _csv(out/"reference_convergence_repetitions.csv",ref_rows); _csv(out/"reference_convergence_summary.csv",ref_summaries)
    diagnostics=_diagnostics(pools,allocations,random_seed); _json(out/"pod_subspace_diagnostics.json",diagnostics); figures=_figures(out,case_name,allocation_rows,summaries,ref_summaries)
    selected=[r for r in summaries if r["reference_sample_count"]==50 and r["method"].startswith("fixed-m0-")]; labels={"mean field":"mean_field_relative_error","covariance":"covariance_probe_relative_error","eigenvalues":"leading_eigenvalue_mean_relative_error","POD subspace":"projector_distance_fro","projection":"heldout_projection_error"}
    findings=[f"# {case_name}: two-fidelity findings","","The 50-field independent DSMC ensemble is a numerical reference, not ground truth.","","## Metric-specific preferred counts",""]
    for label,metric in labels.items():
        best=min(selected,key=lambda r:r[f"{metric}_median"]); findings.append(f"- {label}: m0={best['m0']}, median {best[f'{metric}_median']:.6g} (IQR {best[f'{metric}_q25']:.6g}--{best[f'{metric}_q75']:.6g}).")
    findings += ["","Configured scenario costs are reported separately from measured CPU-hours. Comparable measured CPU-hours were unavailable, so no measured-cost sweep was performed.","","Moment accuracy and POD-subspace stability are different objectives; cancellation in the signed covariance correction and a small retained/complement eigengap can rotate modes without producing equally large mean errors."]
    (out/"case_findings.md").write_text("\n".join(findings)+"\n",encoding="utf-8")
    source_paths=[pools["prepared"],root/"pilot/field_pilot_statistics.npz",root/"production/roles.json",root/"production/state.json",root/"snapshots/field_snapshot_metadata.json",root/"inspection/data_availability_report.json",root/"resolved_config.yaml",root/"benchmark/benchmark_summary.csv"]
    metadata={"case":case_name,"protocol":"DSMC target with TPMC full-field control only","m0_values":list(M0_VALUES),"reference_sizes":list(REFERENCE_SIZES),"repetitions":repetitions,"random_seed":random_seed,"costs":{"configured":costs,"measured_cpu_hours":"unavailable"},"available_counts":{"pilot_pairs":min(map(len,pools["pilot"].values())),"production_DSMC":len(pools["production"]["DSMC"]),"production_TPMC":available_tpmc,"reference_DSMC":len(pools["reference"])},"role_separation_verified":True,"topology_verified":True,"source_sha256":{str(p):_hash(p) for p in source_paths},"figures":figures}
    _json(out/"sensitivity_metadata.json",metadata)
    return {"output":str(out),"rows":len(rows),"summary_rows":len(summaries),"allocations":allocation_rows,"figures":figures}
