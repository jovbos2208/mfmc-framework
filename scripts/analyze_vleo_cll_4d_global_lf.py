#!/usr/bin/env python3
"""QA, rank, and select a diverse HF shortlist from the global 4D LF screen."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

PARAMS=('nose_length_fraction','tail_length_fraction','front_width_height_ratio',
        'rear_width_height_ratio','nose_end_scale','tail_end_scale')
def read(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
 fields=list(rows[0]);
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def area(root,g):return .5*float(json.loads((root/g/f'{g}.manifest.json').read_text())['validation']['surface_area_m2'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=Path('outputs/vleo_cll_4d_global_search_bundle'));p.add_argument('--results',type=Path,default=Path('outputs/vleo_cll/global_4d_lf10000'));p.add_argument('--output',type=Path,default=Path('outputs/vleo_cll/global_4d_lf_analysis'));p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=20261103);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 design=json.loads((a.bundle/'design.json').read_text());manifest_path=a.bundle/'manifests/joint_gsi_attitude_lf10000.json';manifest=json.loads(manifest_path.read_text());expected=manifest['samples'];ids=[s['sample_id'] for s in expected];core=np.array([s['off_axis_deg']<=5 for s in expected]);geoinfo={g['geometry_id']:g for g in design['geometries']};values={};qa=[]
 for g in geoinfo:
  path=a.results/g/'adbsat_lf_5000.csv';rows=read(path)
  if len(rows)!=10000 or [r['sample_id'] for r in rows]!=ids:raise ValueError(f'count/order mismatch {g}')
  arr=np.array([[float(r[k]) for k in ('cll_alpha_n','cll_sigma_t','aoa_deg','aos_deg','C_D','C_D2','runtime_ms')] for r in rows])
  exp=np.array([[s[k] for k in ('cll_alpha_n','cll_sigma_t','aoa_deg','aos_deg')] for s in expected],float)
  if not np.allclose(arr[:,:4],exp,rtol=0,atol=1e-12):raise ValueError(f'manifest mismatch {g}')
  if not np.isfinite(arr).all() or not np.allclose(arr[:,5],arr[:,4]**2,rtol=0,atol=1e-12):raise ValueError(f'invalid values {g}')
  values[g]=arr[:,4]*area(a.bundle/'geometries',g);qa.append({'geometry_id':g,'n':len(rows),'finite':True,'manifest_match':True,'raw_sha256':sha(path),'status':'pass'})
 gids=list(values);Y=np.vstack([values[g] for g in gids]);base_i=gids.index('cylinder_hex_wp5_000');B=Y[base_i];rows=[]
 for i,g in enumerate(gids):
  cd=Y[i]/area(a.bundle/'geometries',g);bcd=B/area(a.bundle/'geometries','cylinder_hex_wp5_000');r={'geometry_id':g,'role':geoinfo[g]['role'],'mean_cd':cd.mean(),'sd_cd':cd.std(ddof=1),'mean_drag_area_m2':Y[i].mean(),'sd_drag_area_m2':Y[i].std(ddof=1),'core_mean_drag_area_m2':Y[i,core].mean(),'core_sd_drag_area_m2':Y[i,core].std(ddof=1),'mean_drag_ratio':Y[i].mean()/B.mean(),'sd_drag_ratio':Y[i].std(ddof=1)/B.std(ddof=1),'core_mean_drag_ratio':Y[i,core].mean()/B[core].mean(),'core_sd_drag_ratio':Y[i,core].std(ddof=1)/B[core].std(ddof=1),'mean_cd_ratio':cd.mean()/bcd.mean(),'sd_cd_ratio':cd.std(ddof=1)/bcd.std(ddof=1)};r['mean_guardrail_passed']=r['mean_drag_ratio']<=1.02;r['robust_target_passed']=r['mean_guardrail_passed'] and r['sd_drag_ratio']<=.8;r['score_cd_0p5']=.5*(r['mean_cd_ratio']+r['sd_cd_ratio']);r['score_drag_0p5']=.5*(r['mean_drag_ratio']+r['sd_drag_ratio']);r.update({f'geometry_{k}':geoinfo[g]['parameters'][k] for k in PARAMS});rows.append(r)
 rng=np.random.default_rng(a.seed);wins={k:np.zeros(len(gids),int) for k in ('mean','robustness','combination')}
 for _ in range(a.bootstrap):
  ix=rng.integers(0,len(ids),len(ids));z=Y[:,ix];mu=z.mean(1);sd=z.std(1,ddof=1);wins['mean'][np.argmin(mu)]+=1;eligible=np.flatnonzero(mu/mu[base_i]<=1.02);wins['robustness'][eligible[np.argmin(sd[eligible])]]+=1;score=.5*(mu/mu[base_i]+sd/sd[base_i]);wins['combination'][np.argmin(score)]+=1
 for r in rows:
  i=gids.index(r['geometry_id']);r.update({f'bootstrap_win_{k}':wins[k][i]/a.bootstrap for k in wins})
 objectives={'mean':lambda r:r['mean_drag_area_m2'],'robustness':lambda r:r['sd_drag_area_m2'],'combination':lambda r:r['score_drag_0p5']};selected=[{'geometry_id':'cylinder_hex_wp5_000','selection_role':'baseline_control','rank':0}]
 bounds=np.asarray([design['geometry_bounds'][name] for name in PARAMS],dtype=float)
 for name,key in objectives.items():
  pool=[r for r in rows if name!='robustness' or r['mean_guardrail_passed']];pool=sorted(pool,key=key);picked=[]
  for r in pool:
   if r['geometry_id']=='cylinder_hex_wp5_000':continue
   v=np.array([r[f'geometry_{k}'] for k in PARAMS]);vn=(v-bounds[:,0])/(bounds[:,1]-bounds[:,0])
   if all(np.linalg.norm(vn-p[1])>=.12 for p in picked):picked.append((r,vn))
   if len(picked)==3:break
  selected.extend({'geometry_id':r['geometry_id'],'selection_role':name,'rank':i+1,'screen_metric':key(r)} for i,(r,_) in enumerate(picked))
 shortlist=[];seen=set()
 for x in selected:
  if x['geometry_id'] not in seen:shortlist.append(x);seen.add(x['geometry_id'])
 report={'schema_version':1,'status':'pass','exploratory':True,'n_geometries':len(gids),'n_samples':len(ids),'n_core_samples':int(core.sum()),'bootstrap_draws':a.bootstrap,'winners':{k:gids[int(np.argmax(v))] for k,v in wins.items()},'robust_target_met':any(r['robust_target_passed'] for r in rows),'shortlist':shortlist,'holdout_opened':False,'manifest_sha256':sha(manifest_path)};write_csv(a.output/'ranking.csv',sorted(rows,key=lambda r:r['score_drag_0p5']));write_csv(a.output/'qa.csv',qa);(a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'hf_shortlist.json').write_text(json.dumps({'schema_version':1,'source_analysis_sha256':sha(a.output/'analysis.json'),'selection':shortlist,'holdout_opened':False},indent=2)+'\n');print(a.output/'analysis.json')
if __name__=='__main__':main()
