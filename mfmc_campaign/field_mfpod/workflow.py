from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .allocation import (
    AllocationOptions,
    allocate_counts,
    compare_field_allocation_strategies,
    optimize_allocation,
    optimize_field_allocation,
    select_empirical_allocation,
)
from .config import MFPODConfig
from .covariance_operator import covariance_probe_error, estimate_full_field_mfmc, solve_full_field_pod
from .field_validation import leading_eigenvalue_error, pod_validation_metrics, relative_field_error
from .field_statistics import compute_field_pilot_statistics
from .metrics import evaluate_subspace
from .models import MFPODError, jsonable
from .operator import compute_adaptive_mfpod, compute_mfpod
from .pod import compute_hf_pod, compute_lf_pod, select_dimensions
from .snapshots import PICLASIdentitySurfaceAdapter, drag_functional, inspect_surface_data, validate_surface_topology
from .weights import estimate_global_mfpod_weight
from .visualization import export_surface_modes, generate_report_figures


def _write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(jsonable(data), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows or [{"status":"no feasible rows"}])


def inspect(cfg: MFPODConfig) -> dict:
    out = cfg.output_dir / "inspection"
    report = inspect_surface_data(cfg.case_name, cfg.archives, out, cfg.raw.get("topology_tolerance"))
    cost_stats={"cost_unit":"configured relative model-evaluation cost","measured":False,"models":{}}
    for fidelity,path in cfg.archives.items():
        if path.is_file():
            with np.load(path,allow_pickle=False) as npz:
                if "cpu_hours" in npz.files:
                    values=np.asarray(npz["cpu_hours"],dtype=float); values=values[np.isfinite(values)&(values>0)]
                    if values.size: cost_stats["models"][fidelity]={"median":float(np.median(values)),"mean":float(np.mean(values)),"standard_deviation":float(np.std(values,ddof=1)) if values.size>1 else 0.0,"successful_runs":int(values.size)}; cost_stats["measured"]=True
        if fidelity not in cost_stats["models"]: cost_stats["models"][fidelity]={"configured_cost":float(cfg.raw.get("costs",{}).get(fidelity,float("nan"))),"warning":"measured run cost unavailable"}
    _write_json(out/"cost_statistics.json",cost_stats)
    (out / "piclas_surface_export_specification.md").write_text(
        "# Required PICLAS surface export\n\nOne NPZ per fidelity with `force_per_area[n,face,3]`, `sample_id`, `face_area`, `A_ref` or `A_ref_per_sample`, `q_inf`, `u_hat_inf`, `fidelity`, `case_name`, `geometry_id`, and preferably `face_center`, `face_normal`, `reference_point`, `C_D`, `cpu_hours`, and `hardware`. DSMC and TPMC arrays must retain identical face ordering, scale, component ordering, and body-fixed frame.\n", encoding="utf-8")
    return report


def _coupled_batches(cfg: MFPODConfig):
    adapter = PICLASIdentitySurfaceAdapter(cfg.archives, cfg.raw.get("topology_tolerance"))
    h=adapter.load_batch(cfg.case_name, "DSMC", cfg.snapshot_type); l=adapter.load_batch(cfg.case_name, "TPMC", cfg.snapshot_type)
    topology=validate_surface_topology(h.geometry,l.geometry,cfg.raw.get("topology_tolerance"))
    if not topology["identity_mapping_allowed"]: raise MFPODError(topology["reason"])
    li={sid:i for i,sid in enumerate(l.sample_ids)}; hi=[]; lj=[]; ids=[]
    for i,sid in enumerate(h.sample_ids):
        if sid in li: hi.append(i); lj.append(li[sid]); ids.append(sid)
    if len(ids)<4: raise MFPODError("At least four paired DSMC/TPMC samples are required")
    return h,l,np.asarray(hi),np.asarray(lj),np.asarray(ids),topology


def _split_indices(n: int, cfg: MFPODConfig):
    pilot_n=int(cfg.raw.get("pilot",{}).get("paired_samples", min(12,max(2,n//4))))
    test_n=int(cfg.raw.get("reference_samples", max(2,n//3)))
    if pilot_n+test_n+2>n:
        policy=cfg.raw.get("infeasible_policy","fail")
        if policy=="fail": raise MFPODError(f"Need pilot({pilot_n}) + reference({test_n}) + >=2 production samples, but only {n} paired samples exist")
        pilot_n=max(2,min(pilot_n,n//4)); test_n=max(2,min(test_n,n//3))
    rng=np.random.default_rng(cfg.random_seed); order=rng.permutation(n)
    return order[:pilot_n],order[pilot_n:pilot_n+test_n],order[pilot_n+test_n:]


def prepare_snapshots(cfg: MFPODConfig) -> dict:
    h,l,hi,lj,ids,topology=_coupled_batches(cfg); pilot,test,production=_split_indices(len(ids),cfg)
    out=cfg.output_dir/"snapshots"; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"prepared_snapshots.npz",hf=h.values[hi],lf=l.values[lj],sample_ids=ids,pilot_indices=pilot,test_indices=test,production_indices=production,face_area=h.geometry.face_area,A_ref=np.asarray([h.geometry.A_ref]),face_center=np.empty((0,3)) if h.geometry.face_center is None else h.geometry.face_center,face_normal=np.empty((0,3)) if h.geometry.face_normal is None else h.geometry.face_normal)
    _write_json(cfg.output_dir/"inspection"/"topology_report.json",topology)
    metadata={"case":cfg.case_name,"snapshot_type":cfg.snapshot_type,"coordinate_frame":cfg.coordinate_frame,"centering_mode":cfg.centering_mode,"normalization":"dimensionless traction/q_inf","area_weighting":"sqrt(A_j/A_ref)","sample_roles":{"pilot":ids[pilot].tolist(),"reference_test":ids[test].tolist(),"production":ids[production].tolist()},"random_seed":cfg.random_seed,"disjoint_roles":True}
    _write_json(out/"snapshot_metadata.json",metadata); return metadata


def _configured_models(cfg: MFPODConfig) -> tuple[str, ...]:
    controls = tuple(str(name).upper() for name in cfg.raw.get("control_variates", ["TPMC"]))
    return ("DSMC", *controls)


def _load_field_batches(cfg: MFPODConfig):
    models = _configured_models(cfg)
    missing = [name for name in models if name not in cfg.archives or not cfg.archives[name].is_file()]
    if missing:
        raise MFPODError(f"Missing full-field archives for {missing}")
    adapter = PICLASIdentitySurfaceAdapter(cfg.archives, cfg.raw.get("topology_tolerance"))
    batches = {name: adapter.load_batch(cfg.case_name, name, "full_traction") for name in models}
    topology = {
        name: validate_surface_topology(
            batches["DSMC"].geometry, batches[name].geometry, cfg.raw.get("topology_tolerance")
        )
        for name in models[1:]
    }
    invalid = [name for name, report in topology.items() if not report["identity_mapping_allowed"]]
    if invalid:
        raise MFPODError(f"Models do not share the ordered DSMC surface Hilbert space: {invalid}")
    return models, batches, topology


def prepare_field_snapshots(cfg: MFPODConfig) -> dict:
    """Create disjoint paired pilot/reference roles and nested production pools."""

    models, batches, topology = _load_field_batches(cfg)
    id_lookup = {
        name: {str(sample_id): index for index, sample_id in enumerate(batch.sample_ids)}
        for name, batch in batches.items()
    }
    common_ids = [
        str(sample_id)
        for sample_id in batches["DSMC"].sample_ids
        if all(str(sample_id) in id_lookup[name] for name in models[1:])
    ]
    if len(common_ids) < 6:
        raise MFPODError("At least six all-model paired fields are required for disjoint roles")
    for sample_id in common_ids:
        target_index = id_lookup["DSMC"][sample_id]
        for name in models[1:]:
            control_index = id_lookup[name][sample_id]
            if not np.isclose(
                batches["DSMC"].q_inf[target_index], batches[name].q_inf[control_index], rtol=1.0e-12, atol=0.0
            ) or not np.isclose(
                batches["DSMC"].A_ref_per_sample[target_index],
                batches[name].A_ref_per_sample[control_index],
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise MFPODError(
                    f"Paired normalization mismatch for sample_id={sample_id!r} between DSMC and {name}"
                )
    pilot_indices, reference_indices, production_indices = _split_indices(len(common_ids), cfg)
    pilot_ids = [common_ids[index] for index in pilot_indices]
    reference_ids = [common_ids[index] for index in reference_indices]
    production_ids = [common_ids[index] for index in production_indices]
    reserved = set(pilot_ids) | set(reference_ids)

    payload: dict[str, np.ndarray] = {
        "pilot_sample_ids": np.asarray(pilot_ids),
        "reference_sample_ids": np.asarray(reference_ids),
        "production_paired_sample_ids": np.asarray(production_ids),
    }
    pool_counts = {}
    for name in models:
        pilot_rows = [id_lookup[name][sample_id] for sample_id in pilot_ids]
        payload[f"pilot_{name}"] = batches[name].values[pilot_rows]
        payload[f"pilot_CD_{name}"] = np.asarray([
            batches[name].values[index] @ drag_functional(
                batches[name].geometry, batches[name].u_hat_inf[index]
            )
            for index in pilot_rows
        ])
        paired_rows = [id_lookup[name][sample_id] for sample_id in production_ids]
        ordered_ids = list(production_ids)
        ordered_rows = list(paired_rows)
        if name != "DSMC":
            for index, sample_id in enumerate(batches[name].sample_ids):
                text_id = str(sample_id)
                if text_id not in reserved and text_id not in set(ordered_ids):
                    ordered_ids.append(text_id)
                    ordered_rows.append(index)
        payload[f"production_{name}"] = batches[name].values[ordered_rows]
        payload[f"production_ids_{name}"] = np.asarray(ordered_ids)
        payload[f"production_CD_{name}"] = np.asarray([
            batches[name].values[index] @ drag_functional(
                batches[name].geometry, batches[name].u_hat_inf[index]
            )
            for index in ordered_rows
        ])
        pool_counts[name] = len(ordered_rows)
    reference_rows = [id_lookup["DSMC"][sample_id] for sample_id in reference_ids]
    payload["reference_DSMC"] = batches["DSMC"].values[reference_rows]
    geometry = batches["DSMC"].geometry
    payload.update(
        {
            "face_area": geometry.face_area,
            "A_ref": np.asarray([geometry.A_ref]),
            "face_center": np.empty((0, 3)) if geometry.face_center is None else geometry.face_center,
            "face_normal": np.empty((0, 3)) if geometry.face_normal is None else geometry.face_normal,
        }
    )
    out = cfg.output_dir / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "prepared_field_snapshots.npz", **payload)
    metadata = {
        "case": cfg.case_name,
        "models": models,
        "snapshot_type": "full_traction",
        "coordinate_frame": cfg.coordinate_frame,
        "centering": "pilot_dsmc_mean",
        "normalization": "dimensionless traction/q_inf",
        "area_weighting": "sqrt(A_j/A_ref)",
        "sample_roles": {
            "pilot": pilot_ids,
            "dsmc_reference": reference_ids,
            "production_paired_prefix": production_ids,
        },
        "production_pool_counts": pool_counts,
        "topology": topology,
        "random_seed": cfg.random_seed,
        "disjoint_roles": True,
    }
    _write_json(out / "field_snapshot_metadata.json", metadata)
    return metadata


def _load_prepared_fields(cfg: MFPODConfig):
    path = cfg.output_dir / "snapshots" / "prepared_field_snapshots.npz"
    if not path.exists():
        prepare_field_snapshots(cfg)
    return np.load(path, allow_pickle=False)


def _load_prepared(cfg):
    path=cfg.output_dir/"snapshots"/"prepared_snapshots.npz"
    if not path.exists(): prepare_snapshots(cfg)
    return np.load(path,allow_pickle=False)


def _center(cfg,z):
    h,l=z["hf"],z["lf"]; p=z["pilot_indices"]
    if cfg.centering_mode=="none": mh=np.zeros(h.shape[1]); ml=np.zeros(l.shape[1])
    elif cfg.centering_mode=="per_fidelity_pilot_mean": mh=np.mean(h[p],axis=0); ml=np.mean(l[p],axis=0)
    else:
        ref=cfg.raw.get("common_fixed_reference")
        if ref is None: raise MFPODError("common_fixed_reference centering requires common_fixed_reference NPZ path")
        with np.load(Path(ref),allow_pickle=False) as r: mh=ml=np.asarray(r["z_ref"])
    return h-mh,l-ml,mh,ml


def pilot(cfg: MFPODConfig) -> dict:
    z=_load_prepared(cfg); hc,lc,mh,ml=_center(cfg,z); p=z["pilot_indices"]; settings=cfg.raw.get("pilot",{})
    result=estimate_global_mfpod_weight(hc[p],lc[p],bootstrap_repeats=int(settings.get("bootstrap_repeats",500)),random_seed=int(settings.get("random_seed",1101)),alpha_bounds=tuple(cfg.raw["alpha_bounds"]) if cfg.raw.get("alpha_bounds") is not None else None)
    out=cfg.output_dir/"pilot"; out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/"centering_fields.npz",mu_H_pilot=mh,mu_L_pilot=ml); _write_json(out/"pilot_statistics.json",result)
    return result


def allocation_sweep(cfg: MFPODConfig) -> dict:
    z=_load_prepared(cfg); hc,lc,_,_=_center(cfg,z); p=z["pilot_indices"]; ps=pilot(cfg); a=cfg.raw.get("allocation",{}); costs=cfg.raw.get("costs",{"DSMC":1.0,"TPMC":0.05}); budgets=cfg.raw.get("budgets_hf_equivalent",[5,10]); budget=float(max(budgets))*float(costs.get("DSMC",1.0)); fractions=a.get("candidate_hf_budget_fractions",[.2,.35,.5,.65,.8]); targets=a.get("target_r",[5]); target=int(targets[0] if isinstance(targets,list) else targets)
    try:
        result=select_empirical_allocation(hc[p],lc[p],budget=budget,hf_cost=float(costs["DSMC"]),lf_cost=float(costs["TPMC"]),candidate_fractions=fractions,alpha=ps["alpha"],target_r=target,validation_fraction=float(cfg.raw.get("pilot",{}).get("validation_fraction",.4)),repeats=min(20,int(cfg.raw.get("repeats",5))),random_seed=int(cfg.raw.get("pilot",{}).get("random_seed",1101)))
    except MFPODError as exc:
        result={"description":"pilot-selected empirical allocation","feasible":False,"reason":str(exc),"selected":{"fraction":.5}}
    out=cfg.output_dir/"allocation"; _write_json(out/"selected_allocation.json",result); _write_csv(out/"allocation_sweep.csv",result.get("candidate_results",[])); return result


def _field_pilot_arrays(cfg: MFPODConfig) -> tuple[dict[str, np.ndarray], dict]:
    settings = cfg.raw.get("field_allocation", cfg.raw.get("allocation_optimization", {})) or {}
    archive = settings.get("pilot_field_archive", settings.get("pilot_response_archive"))
    if archive:
        path = Path(archive)
        if not path.is_absolute():
            path = (cfg.path.parent / path).resolve()
        with np.load(path, allow_pickle=False) as data:
            fields = {
                name: np.asarray(data[name], dtype=float)
                for name in ("DSMC", "TPMC", "SENTMAN")
                if name in data.files
            }
        fields = {name: value[:, None] if value.ndim == 1 else value for name, value in fields.items()}
        if "DSMC" not in fields:
            raise MFPODError("pilot_field_archive must contain DSMC full fields")
        return fields, {"source": str(path), "external": True}
    prepared = _load_prepared_fields(cfg)
    fields = {
        name: np.asarray(prepared[f"pilot_{name}"], dtype=float)
        for name in _configured_models(cfg)
    }
    return fields, {"source": str(cfg.output_dir / "snapshots" / "prepared_field_snapshots.npz"), "external": False}


def field_pilot(cfg: MFPODConfig) -> dict:
    """Compute allocation statistics directly from paired complete fields."""

    fields, source = _field_pilot_arrays(cfg)
    settings = cfg.raw.get("field_allocation", cfg.raw.get("allocation_optimization", {})) or {}
    statistics = compute_field_pilot_statistics(
        fields,
        target="DSMC",
        covariance_ridge=float(settings.get("covariance_ridge", 1.0e-10)),
        psd_floor=float(settings.get("psd_floor", 0.0)),
    )
    out = cfg.output_dir / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "field_pilot_statistics.npz",
        models=np.asarray(statistics.models),
        reference_field=statistics.reference_field,
        mean_covariance_raw=statistics.mean_covariance_raw,
        mean_covariance=statistics.mean_covariance,
        second_moment_covariance_raw=statistics.second_moment_covariance_raw,
        second_moment_covariance=statistics.second_moment_covariance,
    )
    summary = {
        "models": statistics.models,
        "source": source,
        "diagnostics": statistics.diagnostics,
        "mean_covariance_raw": statistics.mean_covariance_raw,
        "mean_covariance": statistics.mean_covariance,
        "second_moment_covariance_raw": statistics.second_moment_covariance_raw,
        "second_moment_covariance": statistics.second_moment_covariance,
        "allocation_features": "complete_fields",
        "tpmc_basis_used": False,
    }
    _write_json(out / "field_pilot_statistics.json", summary)
    return jsonable(summary)


def optimal_allocation(cfg: MFPODConfig) -> dict:
    """Run field-aware integer allocation without a TPMC working basis."""

    settings = cfg.raw.get("field_allocation", cfg.raw.get("allocation_optimization", {})) or {}
    constraints = cfg.raw.get("allocation_constraints", settings) or {}
    fields, source = _field_pilot_arrays(cfg)
    costs = {str(name).upper(): float(value) for name, value in (cfg.raw.get("costs", {}) or {}).items()}
    budget = float(constraints.get("budget", settings.get("budget", max(cfg.raw.get("budgets_hf_equivalent", [10])) * costs.get("DSMC", 1.0))))
    maximum_counts = {str(k).upper(): int(v) for k, v in (constraints.get("maximum_counts", {}) or {}).items()}
    if not source["external"] and not maximum_counts:
        prepared = _load_prepared_fields(cfg)
        maximum_counts = {
            name: int(prepared[f"production_{name}"].shape[0]) for name in fields
        }
    options = AllocationOptions(
        budget=budget,
        target="DSMC",
        minimum_target=int(constraints.get("minimum_target", 2)),
        minimum_counts={str(k).upper(): int(v) for k, v in (constraints.get("minimum_counts", {}) or {}).items()},
        maximum_counts=maximum_counts,
        min_ratios={str(k).upper(): float(v) for k, v in (constraints.get("min_ratios", {}) or {}).items()},
        max_ratios={str(k).upper(): float(v) for k, v in (constraints.get("max_ratios", {"TPMC": 10.0}) or {}).items()},
        mode=str(settings.get("mode", "bootstrap_robust")),
        bootstrap_repeats=int(settings.get("bootstrap_repeats", 200)),
        robust_quantile=float(settings.get("robust_quantile", 0.90)),
        random_seed=int(settings.get("random_seed", cfg.random_seed)),
        covariance_ridge=float(settings.get("covariance_ridge", 1.0e-10)),
        psd_floor=float(settings.get("psd_floor", 0.0)),
        max_enumeration_candidates=int(settings.get("max_enumeration_candidates", 250000)),
        mean_weight=float(settings.get("mean_weight", 0.25)),
        second_moment_weight=float(settings.get("second_moment_weight", 0.75)),
    )
    result = optimize_field_allocation(fields, costs, options)
    comparisons = compare_field_allocation_strategies(fields, costs, options)
    out = cfg.output_dir / "allocation"
    _write_json(out / "optimal_allocation.json", result.as_dict())
    _write_csv(out / "optimal_allocation_candidates.csv", result.candidate_table)
    _write_csv(out / "allocation_strategy_comparison.csv", comparisons)
    field_pilot(cfg)
    return result.as_dict()


def _build_full_field_estimate(cfg: MFPODConfig):
    allocation_path = cfg.output_dir / "allocation" / "optimal_allocation.json"
    if not allocation_path.exists():
        optimal_allocation(cfg)
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    prepared = _load_prepared_fields(cfg)
    counts = {str(name).upper(): int(value) for name, value in allocation["counts"].items() if int(value) > 0}
    fields = {
        name: np.asarray(prepared[f"production_{name}"], dtype=float)
        for name in counts
    }
    pilot_path = cfg.output_dir / "pilot" / "field_pilot_statistics.npz"
    if not pilot_path.exists():
        field_pilot(cfg)
    with np.load(pilot_path, allow_pickle=False) as pilot_data:
        reference = np.asarray(pilot_data["reference_field"], dtype=float)
    weights = allocation.get("control_weights") or {}
    return estimate_full_field_mfmc(
        fields,
        counts,
        reference_field=reference,
        mean_weights=weights.get("mean", {}),
        second_moment_weights=weights.get("second_moment", {}),
    )


def field_estimate(cfg: MFPODConfig) -> dict:
    """Persist the full-field DSMC-target MFMC mean and operator metadata."""

    statistics = _build_full_field_estimate(cfg)
    out = cfg.output_dir / "estimator"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "mean_field.npz",
        mean_field=statistics.mean_field,
        centered_mean=statistics.centered_mean,
        reference_field=statistics.reference_field,
    )
    metadata = statistics.metadata()
    metadata.pop("mean_field", None)
    metadata.pop("centered_mean", None)
    metadata.pop("reference_field", None)
    metadata["operator_reconstruction"] = "prepared_field_snapshots + counts + separate control weights"
    _write_json(out / "covariance_operator_metadata.json", metadata)
    _write_json(
        out / "control_weights.json",
        {
            "mean": statistics.mean_weights,
            "second_moment": statistics.second_moment_weights,
        },
    )
    return metadata


def field_pod(cfg: MFPODConfig) -> dict:
    """Solve the leading DSMC-target POD eigenpairs from the matrix-free operator."""

    statistics = _build_full_field_estimate(cfg)
    settings = cfg.raw.get("pod", {}) or {}
    result = solve_full_field_pod(
        statistics,
        n_modes=int(settings.get("number_of_modes", cfg.raw.get("pod_modes_r", 5))),
        tolerance=float(settings.get("eigensolver_tolerance", 1.0e-8)),
        max_iterations=int(settings.get("max_iterations", 5000)),
        negative_eigenvalue_tolerance=float(settings.get("negative_eigenvalue_tolerance", 1.0e-10)),
        clip_small_negative_eigenvalues=bool(settings.get("clip_small_negative_eigenvalues", False)),
        random_seed=int(settings.get("random_seed", cfg.random_seed)),
    )
    out = cfg.output_dir / "pod"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "full_field_modes.npz",
        modes=result.modes,
        eigenvalues=result.eigenvalues,
        raw_eigenvalues=np.asarray(result.diagnostics["raw_ritz_values"]),
    )
    _write_json(out / "eigensolver_diagnostics.json", result.diagnostics)
    return {
        "modes": int(result.modes.shape[1]),
        "state_dimension": int(result.modes.shape[0]),
        "backend": result.backend,
        "minimum_computed_ritz_eigenvalue": result.diagnostics["minimum_computed_ritz_eigenvalue"],
        "negative_eigenvalue_count": result.diagnostics["negative_eigenvalue_count"],
    }


def field_benchmark(cfg: MFPODConfig) -> dict:
    """Run the predeclared equal-cost full-field allocation comparison."""

    prepared = _load_prepared_fields(cfg)
    pilot_fields, _ = _field_pilot_arrays(cfg)
    configured_models = tuple(name for name in _configured_models(cfg) if name in pilot_fields)
    production_fields = {
        name: np.asarray(prepared[f"production_{name}"], dtype=float) for name in configured_models
    }
    costs = {str(name).upper(): float(value) for name, value in (cfg.raw.get("costs", {}) or {}).items()}
    settings = cfg.raw.get("field_allocation", {}) or {}
    constraints = cfg.raw.get("allocation_constraints", {}) or {}
    maximum_counts = {name: production_fields[name].shape[0] for name in configured_models}
    base_options = dict(
        budget=float(constraints.get("budget", 20.0)),
        target="DSMC",
        minimum_target=int(constraints.get("minimum_target", 2)),
        minimum_counts={str(k).upper(): int(v) for k, v in (constraints.get("minimum_counts", {}) or {}).items()},
        maximum_counts=maximum_counts,
        min_ratios={str(k).upper(): float(v) for k, v in (constraints.get("min_ratios", {}) or {}).items()},
        max_ratios={str(k).upper(): float(v) for k, v in (constraints.get("max_ratios", {"TPMC": 10.0}) or {}).items()},
        bootstrap_repeats=int(settings.get("bootstrap_repeats", 200)),
        robust_quantile=float(settings.get("robust_quantile", 0.90)),
        random_seed=int(settings.get("random_seed", cfg.random_seed)),
        covariance_ridge=float(settings.get("covariance_ridge", 1.0e-10)),
        psd_floor=float(settings.get("psd_floor", 0.0)),
        max_enumeration_candidates=int(settings.get("max_enumeration_candidates", 250000)),
        mean_weight=float(settings.get("mean_weight", 0.25)),
        second_moment_weight=float(settings.get("second_moment_weight", 0.75)),
    )

    allocations: dict[str, Any] = {}
    configured_mode = str(settings.get("mode", "bootstrap_robust"))
    for label, mode in (
        (f"field-aware-{configured_mode}", configured_mode),
        ("greedy", "greedy"),
        ("continuous-round", "continuous_round"),
        ("direct-enumeration", "enumeration"),
    ):
        try:
            allocations[label] = optimize_field_allocation(
                {name: pilot_fields[name] for name in configured_models},
                {name: costs[name] for name in configured_models},
                AllocationOptions(**base_options, mode=mode),
            )
        except MFPODError:
            if mode != "enumeration":
                raise
    if "TPMC" in configured_models:
        two_maximum = {name: maximum_counts[name] for name in ("DSMC", "TPMC")}
        allocations["two-fidelity-TPMC"] = optimize_field_allocation(
            {name: pilot_fields[name] for name in ("DSMC", "TPMC")},
            {name: costs[name] for name in ("DSMC", "TPMC")},
            AllocationOptions(
                **{
                    **base_options,
                    "maximum_counts": two_maximum,
                    "minimum_counts": {k: v for k, v in base_options["minimum_counts"].items() if k == "TPMC"},
                    "min_ratios": {k: v for k, v in base_options["min_ratios"].items() if k == "TPMC"},
                    "max_ratios": {k: v for k, v in base_options["max_ratios"].items() if k == "TPMC"},
                    "mode": "continuous_round",
                    "bootstrap_repeats": 0,
                }
            ),
        )

    n_h_only = min(
        maximum_counts["DSMC"], int(np.floor(base_options["budget"] / costs["DSMC"]))
    )
    exact_options = AllocationOptions(
        **{
            **base_options,
            "minimum_target": n_h_only,
            "maximum_counts": {"DSMC": n_h_only},
            "minimum_counts": {},
            "min_ratios": {},
            "max_ratios": {},
            "mode": "enumeration",
            "bootstrap_repeats": 0,
        }
    )
    allocations["DSMC-only"] = optimize_field_allocation(
        {"DSMC": pilot_fields["DSMC"]}, {"DSMC": costs["DSMC"]}, exact_options
    )

    configured_ratios = (cfg.raw.get("validation", {}) or {}).get(
        "fixed_ratios", {name: 1.0 for name in configured_models[1:]}
    )
    fixed_counts = None
    for n_h in range(base_options["minimum_target"], maximum_counts["DSMC"] + 1):
        trial = {"DSMC": n_h}
        for name in configured_models[1:]:
            trial[name] = max(n_h, int(np.ceil(float(configured_ratios.get(name, 1.0)) * n_h)))
        trial_cost = sum(trial[name] * costs[name] for name in trial)
        if trial_cost <= base_options["budget"] and all(trial[name] <= maximum_counts[name] for name in trial):
            fixed_counts = trial
    if fixed_counts is not None:
        fixed_cost = sum(fixed_counts[name] * costs[name] for name in fixed_counts)
        allocations["fixed-ratios"] = optimize_field_allocation(
            {name: pilot_fields[name] for name in fixed_counts},
            {name: costs[name] for name in fixed_counts},
            AllocationOptions(
                **{
                    **base_options,
                    "budget": fixed_cost,
                    "minimum_target": fixed_counts["DSMC"],
                    "minimum_counts": {name: count for name, count in fixed_counts.items() if name != "DSMC"},
                    "maximum_counts": fixed_counts,
                    "min_ratios": {},
                    "max_ratios": {},
                    "mode": "enumeration",
                    "bootstrap_repeats": 0,
                }
            ),
        )

    if bool((cfg.raw.get("validation", {}) or {}).get("compare_scalar_drag_allocation", True)):
        drag_fields = {
            name: np.asarray(prepared[f"pilot_CD_{name}"], dtype=float) for name in configured_models
        }
        drag_result = optimize_allocation(
            drag_fields,
            {name: costs[name] for name in configured_models},
            AllocationOptions(**{**base_options, "mode": "enumeration", "bootstrap_repeats": 0}),
        )
        locked = {
            **base_options,
            "minimum_target": drag_result.counts["DSMC"],
            "minimum_counts": {name: count for name, count in drag_result.counts.items() if name != "DSMC" and count > 0},
            "maximum_counts": drag_result.counts,
            "min_ratios": {},
            "max_ratios": {},
            "budget": drag_result.total_cost,
            "mode": "enumeration",
            "bootstrap_repeats": 0,
        }
        allocations["scalar-drag-allocation"] = optimize_field_allocation(
            {name: pilot_fields[name] for name in drag_result.counts},
            {name: costs[name] for name in drag_result.counts},
            AllocationOptions(**locked),
        )

    reference_fields = np.asarray(prepared["reference_DSMC"], dtype=float)
    pilot_statistics_path = cfg.output_dir / "pilot" / "field_pilot_statistics.npz"
    if not pilot_statistics_path.exists():
        field_pilot(cfg)
    with np.load(pilot_statistics_path, allow_pickle=False) as pilot_data:
        reference_field = np.asarray(pilot_data["reference_field"], dtype=float)
    reference_statistics = estimate_full_field_mfmc(
        {"DSMC": reference_fields},
        {"DSMC": reference_fields.shape[0]},
        reference_field=reference_field,
    )
    pod_settings = cfg.raw.get("pod", {}) or {}
    n_modes = int(pod_settings.get("number_of_modes", 5))
    reference_pod = solve_full_field_pod(
        reference_statistics,
        n_modes=n_modes,
        tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
        max_iterations=int(pod_settings.get("max_iterations", 5000)),
        random_seed=cfg.random_seed,
    )
    validation_settings = cfg.raw.get("validation", {}) or {}
    rows = []
    for method, allocation in allocations.items():
        active_fields = {name: production_fields[name] for name in allocation.counts}
        estimate = estimate_full_field_mfmc(
            active_fields,
            allocation.counts,
            reference_field=reference_field,
            mean_weights=(allocation.control_weights or {}).get("mean", {}),
            second_moment_weights=(allocation.control_weights or {}).get("second_moment", {}),
        )
        pod = solve_full_field_pod(
            estimate,
            n_modes=n_modes,
            tolerance=float(pod_settings.get("eigensolver_tolerance", 1.0e-8)),
            max_iterations=int(pod_settings.get("max_iterations", 5000)),
            negative_eigenvalue_tolerance=float(pod_settings.get("negative_eigenvalue_tolerance", 1.0e-10)),
            clip_small_negative_eigenvalues=bool(pod_settings.get("clip_small_negative_eigenvalues", False)),
            random_seed=cfg.random_seed,
        )
        subspace = pod_validation_metrics(
            pod.modes,
            reference_pod.modes,
            reference_fields,
            estimated_mean=estimate.mean_field,
            reference_mean=reference_statistics.mean_field,
        )
        eigenvalue = leading_eigenvalue_error(pod.eigenvalues, reference_pod.eigenvalues)
        rows.append(
            {
                "method": method,
                **{f"n_{name}": count for name, count in allocation.counts.items()},
                "total_cost": allocation.total_cost,
                "unused_budget": base_options["budget"] - allocation.total_cost,
                "mean_field_relative_error": relative_field_error(estimate.mean_field, reference_statistics.mean_field),
                "covariance_probe_relative_error": covariance_probe_error(
                    estimate.covariance,
                    reference_statistics.covariance,
                    probe_count=int(validation_settings.get("covariance_probe_count", 100)),
                    random_seed=int(validation_settings.get("covariance_probe_seed", 4401)),
                ),
                "leading_eigenvalue_mean_relative_error": eigenvalue["mean_relative_error"],
                "maximum_principal_angle_rad": subspace["maximum_principal_angle_rad"],
                "projector_distance_fro": subspace["projector_distance_fro"],
                "heldout_projection_error": subspace["projection_error"],
                "minimum_ritz_eigenvalue": pod.diagnostics["minimum_computed_ritz_eigenvalue"],
                "negative_eigenvalue_count": pod.diagnostics["negative_eigenvalue_count"],
            }
        )
    out = cfg.output_dir / "benchmark"
    _write_csv(out / "benchmark_summary.csv", rows)
    _write_json(
        out / "benchmark_metadata.json",
        {
            "equal_configured_budget": base_options["budget"],
            "pilot_cost_included": False,
            "reference_sample_count": int(reference_fields.shape[0]),
            "methods": list(allocations),
            "result_status": "methodological comparison; physical claims require measured costs and independent data",
        },
    )
    return {"methods": len(rows), "rows": rows}


def benchmark(cfg: MFPODConfig) -> dict:
    z=_load_prepared(cfg); hc,lc,_,_=_center(cfg,z); prod=z["production_indices"]; test=z["test_indices"]; ps=pilot(cfg); selected=allocation_sweep(cfg); costs=cfg.raw.get("costs",{"DSMC":1.,"TPMC":.05}); max_r=max(cfg.raw.get("reduced_dimensions",[1,2,5])); repeats=int(cfg.raw.get("repeats",5)); rows=[]; corrections=[]; spectra=[]; rng=np.random.default_rng(cfg.random_seed)
    for budget_eq in cfg.raw.get("budgets_hf_equivalent",[5,10]):
        budget=float(budget_eq)*float(costs["DSMC"]); fraction=float(selected.get("selected",{}).get("fraction",.5))
        try: allocation=allocate_counts(budget,float(costs["DSMC"]),float(costs["TPMC"]),hf_budget_fraction=fraction)
        except MFPODError: continue
        requested_hf_only=int(np.floor(budget/float(costs["DSMC"])))
        requested_lf_only=int(np.floor(budget/float(costs["TPMC"])))
        requested_max=max(allocation["m_L"],requested_hf_only,requested_lf_only)
        if requested_max>len(prod):
            if cfg.raw.get("infeasible_policy","fail")=="fail":
                raise MFPODError(f"Budget {budget_eq} requires up to {requested_max} production samples but only {len(prod)} are available")
        for rep in range(repeats):
            order=rng.permutation(prod); mh=min(allocation["m_H"],max(1,len(order)-1)); ml=min(allocation["m_L"],len(order))
            if ml<=mh: continue
            htrain=hc[order[:mh]]; lp=lc[order[:mh]]; lextra=lc[order[mh:ml]]
            hf_count=min(requested_hf_only,len(order))
            lf_count=min(requested_lf_only,len(order))
            hf=compute_hf_pod(hc[order[:hf_count]],max_r); lf=compute_lf_pod(lc[order[:lf_count]],max_r); mf=compute_mfpod(htrain,lp,lextra,ps["alpha"],backend=cfg.raw.get("eigensolver",{}).get("backend","auto"),n_modes=max_r,negative_eigenvalue_handling=cfg.raw.get("negative_eigenvalue_handling","published_hf_mc_correction"))
            methods={"HF-only POD":(hf,hf_count,0),"LF-only POD":(lf,0,lf_count),"global MFPOD":(mf,mh,ml)}
            if cfg.raw.get("include_adaptive",False): methods["adaptive MFPOD"]=(compute_adaptive_mfpod(htrain,lp,lextra,max_modes=max_r),mh,ml)
            reference=compute_hf_pod(hc[test],max_r)
            if rep == 0 and cfg.snapshot_type == "full_traction":
                for method_name, result_for_export in {"reference_DSMC":reference,"HF-only":hf,"TPMC-only":lf,"global_MFPOD":mf}.items():
                    export_surface_modes(cfg.output_dir/"modes",result_for_export.modes[:,:min(5,result_for_export.modes.shape[1])],result_for_export.eigenvalues[:5],face_area=z["face_area"],A_ref=float(z["A_ref"][0]),face_center=z["face_center"],face_normal=z["face_normal"],method=method_name,case=cfg.case_name,centering_mode=cfg.centering_mode,coordinate_frame=cfg.coordinate_frame,budget=budget_eq)
                    spectra.extend({"budget_hf_equivalent":budget_eq,"method":method_name,"mode":j+1,"eigenvalue":float(value)} for j,value in enumerate(result_for_export.eigenvalues))
            for method,(res,method_mh,method_ml) in methods.items():
                for r in cfg.raw.get("reduced_dimensions",[1,2,5]):
                    if r<=res.modes.shape[1] and r<=reference.modes.shape[1]: rows.append({"case":cfg.case_name,"budget_hf_equivalent":budget_eq,"repeat":rep,"method":method,"r":r,"m_H":method_mh,"m_L":method_ml,"production_cost":float(costs["DSMC"])*method_mh+float(costs["TPMC"])*method_ml,"total_cost":float(costs["DSMC"])*method_mh+float(costs["TPMC"])*method_ml+len(z["pilot_indices"])*(float(costs["DSMC"])+float(costs["TPMC"])),**evaluate_subspace(res.modes[:,:r],hc[test],reference.modes[:,:r])})
            corrections.append({"budget_hf_equivalent":budget_eq,"repeat":rep,"corrected_count":int(mf.corrected_mask.sum()),"corrected_fraction":float(mf.corrected_mask.mean()),"alpha":ps["alpha"]})
    out=cfg.output_dir/"benchmark"; _write_csv(out/"benchmark_repetitions.csv",rows); _write_csv(out/"eigenvalue_corrections.csv",corrections); _write_csv(out/"eigenvalue_spectra.csv",spectra)
    if not rows: raise MFPODError("No feasible benchmark repetition; reduce requested counts/budgets or add archive samples")
    summary=[]
    for key in sorted({(r["budget_hf_equivalent"],r["method"],r["r"]) for r in rows}):
        vals=[r for r in rows if (r["budget_hf_equivalent"],r["method"],r["r"])==key]; summary.append({"budget_hf_equivalent":key[0],"method":key[1],"r":key[2],"median_projection_error":float(np.median([x["projection_error"] for x in vals])),"median_captured_energy":float(np.median([x["captured_energy"] for x in vals])),"repetitions":len(vals)})
    _write_csv(out/"benchmark_summary.csv",summary); return {"rows":len(rows),"summary":summary}


def report(cfg: MFPODConfig) -> dict:
    summary_path=cfg.output_dir/"benchmark"/"benchmark_summary.csv"
    if not summary_path.exists(): benchmark(cfg)
    out=cfg.output_dir/"report"; out.mkdir(parents=True,exist_ok=True)
    text=f"# MFPOD report: {cfg.case_name}\n\nMethod pinned to Aretz and Willcox, arXiv:2605.29213v1. DSMC is the high-fidelity target; TPMC is only a control variate. Results use {cfg.snapshot_type}, {cfg.centering_mode}, and the body-fixed frame. Allocation is pilot-selected empirical allocation, not a theoretical optimum. The archived DSMC test POD is an internal numerical reference, not physical truth.\n"
    (out/"report.md").write_text(text,encoding="utf-8"); (out/"generated_mfpod_results_macros.tex").write_text("% Generated; no hand-entered numerical claims.\n",encoding="utf-8"); (out/"generated_mfpod_tables.tex").write_text("% Tables are generated from benchmark_summary.csv.\n",encoding="utf-8"); figures=generate_report_figures(cfg.output_dir,cfg.case_name,cfg.centering_mode,cfg.coordinate_frame); return {"report":str(out/"report.md"),"figures":figures}


def generate_campaign_manifests(cfg: MFPODConfig, count: int = 100, hf_count: int = 30):
    out=cfg.output_dir/"campaign"; rows=[{"sample_id":f"mfpod_{i:05d}","ordered_index":i,"random_seed":cfg.random_seed+i} for i in range(count)]; _write_csv(out/"campaign_samples.csv",rows); _write_csv(out/"tpmc_run_manifest.csv",[{**r,"fidelity":"TPMC"} for r in rows]); _write_csv(out/"sentman_run_manifest.csv",[{**r,"fidelity":"SENTMAN"} for r in rows]); _write_csv(out/"dsmc_run_manifest.csv",[{**r,"fidelity":"DSMC"} for r in rows[:hf_count]]); _write_csv(out/"campaign_cost_plan.csv",[{"n_DSMC":hf_count,"n_TPMC":count,"n_SENTMAN":count,"note":"replace scenario costs with measured comparable CPU-hour costs"}]); return {"n_DSMC":hf_count,"n_TPMC":count,"n_SENTMAN":count}


def run_all(cfg: MFPODConfig):
    availability=inspect(cfg)
    if not availability["ready"]:
        generate_campaign_manifests(cfg); raise MFPODError("Real surface data are incomplete. Inspection report and non-submitted campaign manifests were generated.")
    if cfg.raw.get("field_allocation", {}).get("enabled", False):
        prepare_field_snapshots(cfg)
        field_pilot(cfg)
        allocation = optimal_allocation(cfg)
        field_estimate(cfg)
        pod_result = field_pod(cfg)
        benchmark_result = field_benchmark(cfg)
        resolved = dict(cfg.raw)
        resolved["resolved_archive_counts"] = {
            name: availability.get(f"n_{name.lower()}", 0) for name in _configured_models(cfg)
        }
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(jsonable(resolved), sort_keys=False), encoding="utf-8"
        )
        return {"allocation": allocation, "pod": pod_result, "benchmark": benchmark_result}
    prepare_snapshots(cfg); pilot(cfg); allocation_sweep(cfg)
    if cfg.raw.get("allocation_optimization", {}).get("enabled", False): optimal_allocation(cfg)
    result=benchmark(cfg); report(cfg)
    resolved=dict(cfg.raw); resolved["resolved_archive_counts"]={"DSMC":availability.get("n_dsmc",0),"TPMC":availability.get("n_tpmc",0)}
    cfg.output_dir.mkdir(parents=True,exist_ok=True); (cfg.output_dir/"resolved_config.yaml").write_text(yaml.safe_dump(jsonable(resolved),sort_keys=False),encoding="utf-8")
    costs=cfg.raw.get("costs",{"DSMC":1.0,"TPMC":0.05}); feasible=[]; production_available=max(0,availability.get("n_paired",0)-int(cfg.raw.get("pilot",{}).get("paired_samples",0))-int(cfg.raw.get("reference_samples",0))); selected=allocation_sweep(cfg); fraction=float(selected.get("selected",{}).get("fraction",.5))
    for b in cfg.raw.get("budgets_hf_equivalent",[]):
        try:
            budget=float(b)*float(costs["DSMC"]); alloc=allocate_counts(budget,float(costs["DSMC"]),float(costs["TPMC"]),hf_budget_fraction=fraction); hf_only=int(np.floor(budget/float(costs["DSMC"]))); lf_only=int(np.floor(budget/float(costs["TPMC"])))
            feasible.append({"budget":b,"m_H_HF_only":hf_only,"m_L_LF_only":lf_only,"m_H_MFPOD":alloc["m_H"],"m_L_MFPOD":alloc["m_L"],"available_production_HF":production_available,"available_production_LF":production_available,"feasible":max(hf_only,lf_only,alloc["m_L"])<=production_available})
        except MFPODError as exc: feasible.append({"budget":b,"feasible":False,"reason":str(exc)})
    _write_csv(cfg.output_dir/"feasibility_report.csv",feasible)
    return result
