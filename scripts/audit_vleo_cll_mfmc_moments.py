#!/usr/bin/env python3
"""Audit CLL HF/LF correlations for drag-area first and second moments."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def rows(path):
    with Path(path).open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def model_cd(path):
    d={}
    for r in rows(Path(path)/"model_evaluations.csv"):
        if r["qoi"]=="C_D": d.setdefault(r["model_id"],{})[r["sample_id"]]=float(r["value"])
    hf=next(v for k,v in d.items() if k.startswith("PICLas")); lf=next(v for k,v in d.items() if k.startswith("ADBSat"))
    ids=sorted(hf); return ids,np.array([hf[i] for i in ids]),np.array([lf[i] for i in ids])
def rho_boot(x,y,rng,draws=50000):
    idx=rng.integers(0,len(x),(draws,len(x))); a=x[idx]; b=y[idx]
    a-=a.mean(1,keepdims=True); b-=b.mean(1,keepdims=True)
    r=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1)); r=r[np.isfinite(r)]
    return float(np.corrcoef(x,y)[0,1]),[float(z) for z in np.quantile(r,[.025,.975])]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--baseline",type=Path,required=True); p.add_argument("--transfer",type=Path,required=True)
    p.add_argument("--baseline-lf-recovery",type=Path,required=True); p.add_argument("--baseline-area",type=float,required=True); p.add_argument("--transfer-area",type=float,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    ids,bh,bl_bad=model_cd(a.baseline); ids2,ch,cl=model_cd(a.transfer)
    rec={r["sample_id"]:float(r["C_D"]) for r in rows(a.baseline_lf_recovery)}
    if ids!=ids2 or sorted(rec)!=ids: raise ValueError("sample mismatch")
    bl=np.array([rec[i] for i in ids]); bh*=a.baseline_area; bl*=a.baseline_area; ch*=a.transfer_area; cl*=a.transfer_area
    rng=np.random.default_rng(20261023); out={"schema_version":1,"n_pairs":len(ids),"gate_min_abs_rho":.5,"metrics":{}}
    for name,x,y in (("baseline_Y",bh,bl),("baseline_Y2",bh**2,bl**2),("transfer_Y",ch,cl),("transfer_Y2",ch**2,cl**2),("geometry_delta_Y",ch-bh,cl-bl),("geometry_delta_Y2",ch**2-bh**2,cl**2-bl**2)):
        rho,ci=rho_boot(x,y,rng); out["metrics"][name]={"rho":rho,"bootstrap_95":ci,"point_gate_passed":abs(rho)>=.5,"lower_abs_ci_gate_passed":min(abs(ci[0]),abs(ci[1]))>=.5 and ci[0]*ci[1]>0}
    out["variance_mfmc_gate_passed_on_pilot"]=all(out["metrics"][k]["lower_abs_ci_gate_passed"] for k in ("baseline_Y","baseline_Y2","transfer_Y","transfer_Y2","geometry_delta_Y","geometry_delta_Y2"))
    out["limitation"]="Pilot gate only; activation must be repeated within each geometry/bootstrap analysis."
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n"); print(a.output)
if __name__=="__main__": main()
