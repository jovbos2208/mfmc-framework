#!/usr/bin/env python3
"""Create reproducible Search configs/manifests and stage VLEO CLL assets."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np, yaml

def lhs(n,seed):
    r=np.random.default_rng(seed); x=np.empty((n,2))
    for j in range(2): x[:,j]=(r.permutation(n)+r.random(n))/n
    return x
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--framework-root",type=Path,default=Path(".")); p.add_argument("--bundle",type=Path,required=True); a=p.parse_args()
    root=a.framework_root.resolve(); bundle=a.bundle.resolve(); georoot=bundle/"geometries"; cfgroot=bundle/"configs"; manroot=bundle/"manifests"
    cfgroot.mkdir(parents=True,exist_ok=True); manroot.mkdir(parents=True,exist_ok=True)
    search=lhs(40,20261025); hold=lhs(138,20261026)
    for name,arr,prefix,role in (("search_40",search,"search","search"),("holdout_120_plus18",hold,"holdout","sealed_holdout")):
        payload={"schema_version":1,"role":role,"seed":20261025 if role=="search" else 20261026,"n":len(arr),"analysis_count":120 if role=="sealed_holdout" else 40,"reserve_count":18 if role=="sealed_holdout" else 0,"samples":[{"sample_id":f"{prefix}_{i:03d}","cll_alpha_n":float(x[0]),"cll_sigma_t":float(x[1]),"random_seed":int((202610250000 if role=="search" else 202610260000)+i)} for i,x in enumerate(arr)]}
        (manroot/f"{name}.json").write_text(json.dumps(payload,indent=2)+"\n")
    samples=json.loads((manroot/"search_40.json").read_text())["samples"]
    search_manifest_sha=digest(manroot/"search_40.json")
    template=yaml.safe_load((root/"configs/studies/vleo_cll/cylinder_hex_nose_0p30_cll_transfer_20.yaml").read_text())
    assets_obj=bundle/"adbsat_assets/obj_files"; assets_mat=bundle/"adbsat_assets/models"; assets_obj.mkdir(parents=True,exist_ok=True); assets_mat.mkdir(parents=True,exist_ok=True)
    geoms=[]
    for manifest_path in sorted(georoot.glob("*/*.manifest.json")):
        m=json.loads(manifest_path.read_text()); gid=m["geometry_id"]; d=manifest_path.parent
        h5=d/f"{gid}_mesh.h5"; obj=d/f"{gid}.obj"; mat=d/f"{gid}.mat"
        if not all(x.exists() for x in (h5,obj,mat)): raise FileNotFoundError(gid)
        shutil.copy2(obj,assets_obj/obj.name); shutil.copy2(mat,assets_mat/mat.name)
        c=json.loads(json.dumps(template)); c["study"]["id"]=f"vleo_cll_search40_{gid}"; c["geometries"]=[{"id":gid,"name":gid,"characteristic_length":0.1,"geometry_class":"parametric_cylinder_hex","metadata":{"hf_mesh":f"../outputs/vleo_cll_server_run_bundle/geometries/{gid}/{gid}_mesh.h5","lf_model":gid,"piclas_object_boundary_name":"CYLINDER_HEX"}}]
        c["sampling"]={"method":"explicit","sample_count":40,"max_production_samples":40,"source_contract":f"locked search_40.json SHA-256 {search_manifest_sha}","explicit_samples":[{"cll_alpha_n":x["cll_alpha_n"],"cll_sigma_t":x["cll_sigma_t"]} for x in samples]}
        c["pilot"].update({"size":40,"sizes":[40]}); c["outputs"]["dir"]=f"campaign_outputs/vleo_cll/search40/{gid}"; c["seeds"]={"global":20261025}
        (cfgroot/f"{gid}_cll_search40.yaml").write_text(yaml.safe_dump(c,sort_keys=False))
        geoms.append({"geometry_id":gid,"manifest":str(manifest_path.relative_to(bundle)),"h5":str(h5.relative_to(bundle)),"obj":str(obj.relative_to(bundle)),"mat":str(mat.relative_to(bundle)),"n_elements":m["pyhope_conversion"]["validation"]["nElems"]})
    summary={"schema_version":1,"status":"search_ready_holdout_manifest_sealed","search_samples":40,"search_manifest_sha256":search_manifest_sha,"holdout_valid":120,"holdout_reserve":18,"holdout_manifest_sha256":digest(manroot/"holdout_120_plus18.json"),"geometries":geoms,"notes":["Run each Search geometry once; all three objectives reuse the same raw results.","Do not run or inspect the holdout before finalist lock.","Install staged OBJ/MAT files into ADBSat runtime before Search."]}
    (bundle/"bundle_manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    checks=[]
    for f in sorted(x for x in bundle.rglob("*") if x.is_file() and x.name != "SHA256SUMS"): checks.append(f"{digest(f)}  {f.relative_to(bundle)}")
    (bundle/"SHA256SUMS").write_text("\n".join(checks)+"\n")
    print(bundle/"bundle_manifest.json")
if __name__=="__main__": main()
