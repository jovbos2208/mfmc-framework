from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .allocation import allocate_counts, select_empirical_allocation
from .config import MFPODConfig
from .metrics import evaluate_subspace
from .models import MFPODError, jsonable
from .operator import compute_adaptive_mfpod, compute_mfpod
from .pod import compute_hf_pod, compute_lf_pod, select_dimensions
from .snapshots import PICLASIdentitySurfaceAdapter, inspect_surface_data, validate_surface_topology
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
    out=cfg.output_dir/"campaign"; rows=[{"sample_id":f"mfpod_{i:05d}","ordered_index":i,"random_seed":cfg.random_seed+i} for i in range(count)]; _write_csv(out/"campaign_samples.csv",rows); _write_csv(out/"tpmc_run_manifest.csv",[{**r,"fidelity":"TPMC"} for r in rows]); _write_csv(out/"dsmc_run_manifest.csv",[{**r,"fidelity":"DSMC"} for r in rows[:hf_count]]); _write_csv(out/"campaign_cost_plan.csv",[{"m_H":hf_count,"m_L":count,"note":"replace with measured CPU-hour costs"}]); return {"m_H":hf_count,"m_L":count}


def run_all(cfg: MFPODConfig):
    availability=inspect(cfg)
    if not availability["ready"]:
        generate_campaign_manifests(cfg); raise MFPODError("Real surface data are incomplete. Inspection report and non-submitted campaign manifests were generated.")
    prepare_snapshots(cfg); pilot(cfg); allocation_sweep(cfg); result=benchmark(cfg); report(cfg)
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
