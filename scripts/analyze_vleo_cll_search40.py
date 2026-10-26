#!/usr/bin/env python3
"""Audit and rank the common-sample nine-geometry VLEO CLL Search-40 set."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

def read(p):
    with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument("--results-root",type=Path,required=True);p.add_argument("--geometry-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=50000);p.add_argument("--seed",type=int,default=20261027);a=p.parse_args()
    data={}; contracts={}; provenance={}
    for d in sorted(x for x in a.results_root.iterdir() if x.is_dir()):
        gid=d.name; rows=read(d/"model_evaluations.csv"); q={}
        for r in rows:q.setdefault((r["model_id"],r["qoi"]),{})[r["sample_id"]]=float(r["value"])
        hf=next(k[0] for k in q if k[0].startswith("PICLas"));lf=next(k[0] for k in q if k[0].startswith("ADBSat"))
        ids=sorted(q[(hf,"C_D")]);
        if len(ids)!=40 or any(sorted(q[k])!=ids for k in ((hf,"C_D2"),(lf,"C_D"),(lf,"C_D2"))):raise ValueError(f"unpaired {gid}")
        inp=read(d/"sample_inputs.csv"); contract={r["sample_id"]:(r.get("input__cll_alpha_n"),r.get("input__cll_sigma_t")) for r in inp if r["model_id"].startswith("PICLas") and r["qoi"]=="C_D"};contracts[gid]=contract
        m=json.loads((a.geometry_root/gid/f"{gid}.manifest.json").read_text());area=.5*float(m["validation"]["surface_area_m2"])
        h=np.array([q[(hf,"C_D")][i] for i in ids])*area;l=np.array([q[(lf,"C_D")][i] for i in ids])*area
        h2=np.array([q[(hf,"C_D2")][i] for i in ids])*area**2;l2=np.array([q[(lf,"C_D2")][i] for i in ids])*area**2
        data[gid]={"ids":ids,"area":area,"hf":h,"lf":l,"hf2":h2,"lf2":l2,"cd":h/area}
        provenance[gid]={"n":40,"rho_Y":float(np.corrcoef(h,l)[0,1]),"rho_Y2":float(np.corrcoef(h2,l2)[0,1]),"lf_hf_mean_ratio":float(l.mean()/h.mean()),"raw_sha256":hashlib.sha256((d/"model_evaluations.csv").read_bytes()).hexdigest()}
    base="cylinder_hex_wp5_000"; base_contract=contracts[base]
    if any(v!=base_contract for v in contracts.values()):raise ValueError("CLL sample contract differs across geometries")
    b=data[base]; metrics=[]
    for gid,x in data.items():
        metrics.append({"geometry_id":gid,"area_ref_m2":x["area"],"mean_cd":float(x["cd"].mean()),"sd_cd":float(x["cd"].std(ddof=1)),"mean_drag_area_m2":float(x["hf"].mean()),"sd_drag_area_m2":float(x["hf"].std(ddof=1)),"mean_drag_ratio":float(x["hf"].mean()/b["hf"].mean()),"sd_drag_ratio":float(x["hf"].std(ddof=1)/b["hf"].std(ddof=1)),"mean_cd_ratio":float(x["cd"].mean()/b["cd"].mean()),"sd_cd_ratio":float(x["cd"].std(ddof=1)/b["cd"].std(ddof=1))})
    by={r["geometry_id"]:r for r in metrics}
    for r in metrics:r["combined_cd_score_0p5"]=.5*(r["mean_cd_ratio"]+r["sd_cd_ratio"]);r["robust_guardrail_passed"]=r["mean_drag_ratio"]<=1.02
    for r in metrics:
        r["pareto_cd"]=not any((s["mean_cd"]<=r["mean_cd"] and s["sd_cd"]<=r["sd_cd"] and (s["mean_cd"]<r["mean_cd"] or s["sd_cd"]<r["sd_cd"])) for s in metrics)
        r["pareto_drag_area"]=not any((s["mean_drag_area_m2"]<=r["mean_drag_area_m2"] and s["sd_drag_area_m2"]<=r["sd_drag_area_m2"] and (s["mean_drag_area_m2"]<r["mean_drag_area_m2"] or s["sd_drag_area_m2"]<r["sd_drag_area_m2"])) for s in metrics)
    ci_rng=np.random.default_rng(a.seed+1); ci_idx=ci_rng.integers(0,40,(a.bootstrap,40)); bb=b["hf"][ci_idx]
    for r in metrics:
        y=data[r["geometry_id"]]["hf"][ci_idx]
        r["mean_drag_ratio_bootstrap_95"]=[float(v) for v in np.quantile(y.mean(1)/bb.mean(1),[.025,.975])]
        r["sd_drag_ratio_bootstrap_95"]=[float(v) for v in np.quantile(y.std(1,ddof=1)/bb.std(1,ddof=1),[.025,.975])]
    mean_w=min(metrics,key=lambda r:r["mean_cd"])["geometry_id"]
    robust_pool=[r for r in metrics if r["robust_guardrail_passed"]];robust_w=min(robust_pool,key=lambda r:r["sd_drag_area_m2"])["geometry_id"]
    combo_w=min(metrics,key=lambda r:r["combined_cd_score_0p5"])["geometry_id"]
    rng=np.random.default_rng(a.seed); wins={k:{g:0 for g in data} for k in ("minimum_mean_cd","minimum_gsi_sd","mean_sd_compromise")}
    for _ in range(a.bootstrap):
        ix=rng.integers(0,40,40); bm=b["hf"][ix]; bcd=b["cd"][ix]; vals={}
        for gid,x in data.items():
            y=x["hf"][ix];cd=x["cd"][ix];vals[gid]=(cd.mean(),y.mean()/bm.mean(),y.std(ddof=1),.5*(cd.mean()/bcd.mean()+cd.std(ddof=1)/bcd.std(ddof=1)))
        wins["minimum_mean_cd"][min(vals,key=lambda g:vals[g][0])]+=1
        eligible=[g for g in vals if vals[g][1]<=1.02]
        if eligible:wins["minimum_gsi_sd"][min(eligible,key=lambda g:vals[g][2])]+=1
        wins["mean_sd_compromise"][min(vals,key=lambda g:vals[g][3])]+=1
    stability={k:{g:v/a.bootstrap for g,v in d.items()} for k,d in wins.items()}
    report={"schema_version":1,"n_geometries":len(data),"n_common_samples":40,"sample_contract_equal":True,"bootstrap_draws":a.bootstrap,"analysis_seed":a.seed,"winners":{"minimum_mean_cd":mean_w,"minimum_gsi_sd":robust_w,"mean_sd_compromise":combo_w},"winner_bootstrap_frequency":stability,"metrics":sorted(metrics,key=lambda r:r["geometry_id"]),"provenance":provenance,"interpretation_limits":["Search results are exploratory and cannot support the confirmatory claim.","Robust winner uses the predeclared mean-drag guardrail of 1.02.","Combined winner uses normalized C_D mean and SD; physical drag-area metrics are reported in parallel."]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n");print(a.output)
if __name__=="__main__":main()
