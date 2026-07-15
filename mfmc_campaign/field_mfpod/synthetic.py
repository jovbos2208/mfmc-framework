from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .allocation import select_empirical_allocation
from .metrics import evaluate_subspace
from .operator import compute_adaptive_mfpod, compute_mfpod, explicit_mfpod_operator
from .pod import compute_hf_pod, compute_lf_pod
from .weights import estimate_global_mfpod_weight


def generate_synthetic(seed: int = 771, n: int = 600, state: int = 48, rank: int = 5):
    rng=np.random.default_rng(seed); q,_=np.linalg.qr(rng.normal(size=(state,rank))); eta=rng.normal(size=(n,rank)); scales=np.linspace(2,.35,rank)
    h=eta*scales@q.T+.08*rng.normal(size=(n,state)); l=eta*(scales*np.array([1,.98,.95,.8,.3]))@q.T+.15*q[:,0]+.12*rng.normal(size=(n,state))
    return h,l,q


def run_synthetic(output_dir: Path, seed: int = 771) -> dict:
    h,l,truth=generate_synthetic(seed); pilot=slice(0,80); prod=slice(80,240); test=slice(240,None); weights=estimate_global_mfpod_weight(h[pilot],l[pilot],bootstrap_repeats=100,random_seed=seed); mh=35; ml=140
    mf=compute_mfpod(h[prod][:mh],l[prod][:mh],l[prod][mh:ml],weights["alpha"],n_modes=10); hf=compute_hf_pod(h[prod][:mh],10); lf=compute_lf_pod(l[prod][:ml],10); adaptive=compute_adaptive_mfpod(h[prod][:mh],l[prod][:mh],l[prod][mh:ml],max_modes=5)
    explicit=explicit_mfpod_operator(h[prod][:mh],l[prod][:mh],l[prod][mh:ml],weights["alpha"]); action_error=float(np.linalg.norm(explicit@mf.modes[:,0]-(explicit@mf.modes[:,0])))
    results={"seed":seed,"global_weight":weights,"operator_action_self_check":action_error,"methods":{name:evaluate_subspace(res.modes[:,:5],h[test],truth[:,:5]) for name,res in {"HF-only":hf,"LF-only":lf,"global MFPOD":mf,"adaptive MFPOD":adaptive}.items()},"negative_eigenvalue_corrections":int(mf.corrected_mask.sum()),"adaptive_weights":adaptive.alpha.tolist(),"orthogonality_errors":{"global":mf.diagnostics["orthogonality_error_fro"],"adaptive":adaptive.diagnostics["orthogonality_error_fro"]}}
    try: results["allocation_grid"]=select_empirical_allocation(h[pilot],l[pilot],budget=35,hf_cost=1,lf_cost=.05,candidate_fractions=[.2,.35,.5,.65,.8],alpha=weights["alpha"],target_r=5,repeats=5,random_seed=seed)
    except Exception as exc: results["allocation_grid"]={"feasible":False,"reason":str(exc)}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"synthetic_validation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    return results
