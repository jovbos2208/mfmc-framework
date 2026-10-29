#!/usr/bin/env python3
"""Rank the locked local-search candidates using ADBSat-CLL screening only."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def area(root,g):return .5*json.loads((root/g/f'{g}.manifest.json').read_text())['validation']['surface_area_m2']
def values(path):return np.array([float(r['C_D']) for r in read(path)])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/vleo_cll/local_search_round1'));p.add_argument('--existing-root',type=Path,default=Path('outputs/vleo_cll'));p.add_argument('--existing-geometry-root',type=Path,default=Path('outputs/vleo_cll_server_run_bundle/geometries'));a=p.parse_args()
 design=json.loads((a.root/'local_search_design.json').read_text()); centres={}
 for branch,g in design['centres'].items():
  x=values(a.existing_root/'mfmc_lf5000'/g/'adbsat_lf_5000.csv'); ar=area(a.existing_geometry_root,g);centres[branch]={'mean_cd':x.mean(),'sd_cd':x.std(ddof=1),'mean_drag':x.mean()*ar,'sd_drag':x.std(ddof=1)*ar}
 rows=[]
 for c in design['candidates']:
  g=c['geometry_id'];x=values(a.root/'lf5000'/g/'adbsat_lf_5000.csv');ar=area(a.root/'geometries',g); z=centres[c['branch']]
  r={k:c[k] for k in ('geometry_id','branch','centre_geometry_id','perturbed_parameter','direction','step')}
  r.update({'mean_cd':x.mean(),'sd_cd':x.std(ddof=1),'mean_drag_area_m2':x.mean()*ar,'sd_drag_area_m2':x.std(ddof=1)*ar})
  r.update({'mean_cd_ratio':r['mean_cd']/z['mean_cd'],'sd_cd_ratio':r['sd_cd']/z['sd_cd'],
    'mean_drag_ratio_to_centre':r['mean_drag_area_m2']/z['mean_drag'],'sd_drag_ratio_to_centre':r['sd_drag_area_m2']/z['sd_drag']})
  r['score_cd_0p5']=.5*(r['mean_cd_ratio']+r['sd_cd_ratio']);rows.append(r)
 fields=list(rows[0]);
 with (a.root/'screening.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 selected=[]
 rules={'mean':lambda r:r['mean_cd'],'robustness':lambda r:r['sd_drag_area_m2'],'combination':lambda r:r['score_cd_0p5']}
 for branch,key in rules.items():
  pool=[r for r in rows if r['branch']==branch]
  # Exclude the strong SD-only direction if it would exceed the global 1.02 guardrail according to the LF screen.
  if branch=='robustness':
   baseline=centres['robustness']['mean_drag']/centres['robustness']['mean_drag']
   pool=[r for r in pool if r['mean_drag_ratio_to_centre']*1.0062477805762977<=1.02]
  for rank,r in enumerate(sorted(pool,key=key)[:2],1): selected.append({'branch':branch,'screen_rank':rank,**r})
 out={'schema_version':1,'role':'LF-only exploratory screening; not an HF decision','holdout_used':False,
  'selection_rule':'top two per branch; robustness also enforces projected global R_mu <= 1.02',
  'selected':selected,'screening_csv_sha256':sha(a.root/'screening.csv')}
 target=a.root/'hf_selection.json';target.write_text(json.dumps(out,indent=2)+'\n');
 (a.root/'HF_SELECTION_LOCK.json').write_text(json.dumps({'hf_selection_sha256':sha(target),'n':len(selected),'holdout_opened':False},indent=2)+'\n');print(target)
if __name__=='__main__':main()
