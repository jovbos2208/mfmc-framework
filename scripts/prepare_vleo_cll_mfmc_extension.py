#!/usr/bin/env python3
"""Prepare ten new paired HF/LF samples and a 5000-point LF MFMC design."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, yaml

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lhs(n,rng):
    a=np.empty((n,2))
    for j in range(2):a[:,j]=(rng.permutation(n)+rng.random(n))/n
    return a
def maximin_add(existing,n,seed,trials=200000):
    rng=np.random.default_rng(seed); chosen=[]
    for _ in range(n):
        c=rng.random((trials,2)); anchors=np.vstack([existing,*chosen]) if chosen else existing
        # Chunk-free here: 200k x <=49 x 2 is bounded and deterministic.
        dist=np.sqrt(((c[:,None,:]-anchors[None,:,:])**2).sum(2)).min(1)
        chosen.append(c[int(np.argmax(dist))])
    return np.asarray(chosen)
def main():
    p=argparse.ArgumentParser();p.add_argument('--framework-root',type=Path,default=Path('.'));p.add_argument('--base-bundle',type=Path,required=True);p.add_argument('--output-bundle',type=Path,required=True);a=p.parse_args()
    root=a.framework_root.resolve();base=a.base_bundle.resolve();out=a.output_bundle.resolve();(out/'configs').mkdir(parents=True,exist_ok=True);(out/'manifests').mkdir(parents=True,exist_ok=True)
    old=json.loads((base/'manifests/search_40.json').read_text());x40=np.array([[s['cll_alpha_n'],s['cll_sigma_t']] for s in old['samples']]);x10=maximin_add(x40,10,20261028)
    paired=[]
    for i,x in enumerate(x40):paired.append({'sample_id':f'search_{i:03d}','cll_alpha_n':float(x[0]),'cll_sigma_t':float(x[1]),'random_seed':202610250000+i,'role':'existing_paired'})
    for j,x in enumerate(x10,40):paired.append({'sample_id':f'search_{j:03d}','cll_alpha_n':float(x[0]),'cll_sigma_t':float(x[1]),'random_seed':202610280000+j,'role':'new_paired'})
    extra=lhs(4950,np.random.default_rng(20261029));lf=paired+[{ 'sample_id':f'lf_extra_{i:04d}','cll_alpha_n':float(x[0]),'cll_sigma_t':float(x[1]),'random_seed':202610290000+i,'role':'lf_only'} for i,x in enumerate(extra)]
    ext={'schema_version':1,'role':'search_extension','seed':20261028,'n':10,'samples':paired[40:]};(out/'manifests/search_extension_10.json').write_text(json.dumps(ext,indent=2)+'\n')
    p50={'schema_version':1,'role':'paired_search_50','n':50,'samples':paired};(out/'manifests/paired_search_50.json').write_text(json.dumps(p50,indent=2)+'\n')
    l5={'schema_version':1,'role':'mfmc_lf_design','n':5000,'paired_prefix_count':50,'samples':lf};(out/'manifests/mfmc_lf_5000.json').write_text(json.dumps(l5,indent=2)+'\n')
    template=yaml.safe_load((base/'configs/cylinder_hex_wp5_000_cll_search40.yaml').read_text())
    gids=[x['geometry_id'] for x in json.loads((base/'bundle_manifest.json').read_text())['geometries']]
    for gid in gids:
        c=json.loads(json.dumps(template));c['study']['id']=f'vleo_cll_search50_extension10_{gid}';c['geometries'][0]['id']=gid;c['geometries'][0]['name']=gid;c['geometries'][0]['metadata']['hf_mesh']=f'../outputs/vleo_cll_server_run_bundle/geometries/{gid}/{gid}_mesh.h5';c['geometries'][0]['metadata']['lf_model']=gid
        c['sampling']={'method':'explicit','sample_count':10,'max_production_samples':10,'source_contract':f"search_extension_10.json SHA-256 {sha(out/'manifests/search_extension_10.json')}",'explicit_samples':[{'cll_alpha_n':s['cll_alpha_n'],'cll_sigma_t':s['cll_sigma_t']} for s in paired[40:]]}
        c['pilot'].update({'size':10,'sizes':[10]});c['outputs']['dir']=f'campaign_outputs/vleo_cll/search50_extension10/{gid}';c['seeds']={'global':20261028}
        (out/'configs'/f'{gid}_cll_extension10.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    req={'schema_version':1,'requires_existing_bundle':'outputs/vleo_cll_server_run_bundle','geometry_ids':gids,'notes':['Existing 40 HF results must not be rerun.','Run extension10 once per geometry.','Run LF5000 once per geometry with run_vleo_cll_adbsat_lf_manifest.py.']};(out/'requirements.json').write_text(json.dumps(req,indent=2)+'\n')
    sums=[]
    for f in sorted(x for x in out.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):sums.append(f'{sha(f)}  {f.relative_to(out)}')
    (out/'SHA256SUMS').write_text('\n'.join(sums)+'\n');print(out)
if __name__=='__main__':main()
