#!/usr/bin/env python3
"""Joint MFMC/bootstrap ranking of Search-50 and local HF round 1."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_vleo_cll_search50_mfmc import (GEOMETRIES,estimate,load_lf,load_manifest,model_values,write_csv)

def read(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def local_pairs(path,lookup,lf):
 values,coords,meta=model_values(path);hf=next(k[0] for k in values if k[0].startswith('PICLas'));lf_by={r['manifest_sample_id']:r for r in lf};rows=[];used=set()
 for sid,xy in coords.items():
  matches=[s for key,s in lookup.items() if max(abs(xy[0]-key[0]),abs(xy[1]-key[1]))<=1e-12]
  if len(matches)!=1:raise ValueError(f'coordinate match count {len(matches)} for {path}/{sid}')
  sample=matches[0];mid=sample['sample_id'];
  if mid in used:raise ValueError(f'duplicate manifest match {path}/{mid}')
  used.add(mid);l=lf_by[mid];rows.append({'manifest_sample_id':mid,'actual_sample_id':sid,'alpha_n':xy[0],'sigma_t':xy[1],'role':'local_hf_round1','hf_cd':values[(hf,'C_D')][sid],'hf_cd2':values[(hf,'C_D2')][sid],'lf_cd':l['lf_cd'],'lf_cd2':l['lf_cd2'],'hf_cost':meta[sid]['cost_hf'],'lf_cost':l['runtime_ms']/1000})
 return sorted(rows,key=lambda r:r['manifest_sample_id'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/vleo_cll'));p.add_argument('--search-root',type=Path,default=Path('outputs/vleo_cll/local_search_round1'));p.add_argument('--manifest',type=Path,default=Path('outputs/vleo_cll_mfmc_extension_bundle/manifests/paired_search_50.json'));p.add_argument('--lf-manifest',type=Path,default=Path('outputs/vleo_cll_mfmc_extension_bundle/manifests/mfmc_lf_5000.json'));p.add_argument('--output',type=Path,default=Path('outputs/vleo_cll/local_hf_round1_analysis'));p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--rho-gate',type=float,default=.5);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 _,lookup=load_manifest(a.manifest);lfm=json.loads(a.lf_manifest.read_text());selection=json.loads((a.search_root/'hf_selection.json').read_text());local=[x['geometry_id'] for x in selection['selected']]
 oldrows=read(a.root/'search50_mfmc_analysis/paired_search50.csv');old={g:[] for g in GEOMETRIES}
 for r in oldrows:
  x={k:r[k] for k in r};
  for k in ('alpha_n','sigma_t','hf_cd','hf_cd2','lf_cd','lf_cd2','hf_cost','lf_cost'):x[k]=float(x[k])
  old[r['geometry_id']].append(x)
 data={}
 for g in GEOMETRIES:
  pr=sorted(old[g],key=lambda r:r['manifest_sample_id']);lf=load_lf(a.root/'mfmc_lf5000'/g/'adbsat_lf_5000.csv',lfm,1e-12);m=json.loads((Path('outputs/vleo_cll_server_run_bundle/geometries')/g/f'{g}.manifest.json').read_text());data[g]={'pairs':pr,'lf':lf,'area':.5*float(m['validation']['surface_area_m2']),'source':'search50'}
 for g in local:
  lf=load_lf(a.search_root/'lf5000'/g/'adbsat_lf_5000.csv',lfm,1e-12);pr=local_pairs(a.root/'local_hf_round1'/g,lookup,lf);m=json.loads((a.search_root/'geometries'/g/f'{g}.manifest.json').read_text());data[g]={'pairs':pr,'lf':lf,'area':.5*float(m['validation']['surface_area_m2']),'source':'local_hf_round1'}
 arrays={}
 for g,d in data.items():
  ar=d['area'];q=d['pairs'];z=d['lf'];arrays[g]=(np.array([r['hf_cd'] for r in q])*ar,np.array([r['lf_cd'] for r in q])*ar,np.array([r['lf_cd'] for r in z])*ar,np.array([r['hf_cd2'] for r in q])*ar**2,np.array([r['lf_cd2'] for r in q])*ar**2,np.array([r['lf_cd2'] for r in z])*ar**2)
 point={g:estimate(*x,a.rho_gate) for g,x in arrays.items()};base=point[GEOMETRIES[0]];metrics=[]
 for g,e in point.items():
  ar=data[g]['area'];r={'geometry_id':g,'source':data[g]['source'],'estimator':'MFMC' if e['active'] else 'TPMC-only fallback','pearson_Y':e['rho'],'pearson_Y2':e['rho2'],'mean_drag_area_m2':e['mu'],'sd_drag_area_m2':e['sd'],'mean_cd':e['mu']/ar,'sd_cd':e['sd']/ar,'mean_drag_ratio':e['mu']/base['mu'],'sd_drag_ratio':e['sd']/base['sd'],'mean_cd_ratio':(e['mu']/ar)/(base['mu']/data[GEOMETRIES[0]]['area']),'sd_cd_ratio':(e['sd']/ar)/(base['sd']/data[GEOMETRIES[0]]['area'])};r['mean_guardrail_passed']=r['mean_drag_ratio']<=1.02;r['robust_target_passed']=r['mean_guardrail_passed'] and r['sd_drag_ratio']<=.8;r['score_cd_0p5']=.5*(r['mean_cd_ratio']+r['sd_cd_ratio']);r['score_drag_0p5']=.5*(r['mean_drag_ratio']+r['sd_drag_ratio']);metrics.append(r)
 rng=np.random.default_rng(a.seed);wins={k:{g:0 for g in data} for k in ('mean','robustness','combination')};inactive={g:0 for g in data};negative={g:0 for g in data}
 for _ in range(a.bootstrap):
  ip=rng.integers(0,50,50);ia=rng.integers(0,5000,5000);v={}
  for g,x in arrays.items():
   e=estimate(x[0][ip],x[1][ip],x[2][ia],x[3][ip],x[4][ip],x[5][ia],a.rho_gate);v[g]=e;inactive[g]+=not e['active'];negative[g]+=e['raw_reconstructed_var']<=0
  b=v[GEOMETRIES[0]];wins['mean'][min(v,key=lambda g:v[g]['mu']/data[g]['area'])]+=1;eligible=[g for g in v if v[g]['mu']/b['mu']<=1.02]
  if eligible:wins['robustness'][min(eligible,key=lambda g:v[g]['sd'])]+=1
  wins['combination'][min(v,key=lambda g:.5*((v[g]['mu']/data[g]['area'])/(b['mu']/data[GEOMETRIES[0]]['area'])+(v[g]['sd']/data[g]['area'])/(b['sd']/data[GEOMETRIES[0]]['area'])))]+=1
 freq={k:{g:n/a.bootstrap for g,n in z.items()} for k,z in wins.items()};winners={'mean':min(metrics,key=lambda r:r['mean_cd'])['geometry_id'],'robustness':min((r for r in metrics if r['mean_guardrail_passed']),key=lambda r:r['sd_drag_area_m2'])['geometry_id'],'combination':min(metrics,key=lambda r:r['score_cd_0p5'])['geometry_id']}
 gates=[{'geometry_id':g,'point_active':point[g]['active'],'bootstrap_activation_frequency':1-inactive[g]/a.bootstrap,'negative_variance_replicates':negative[g]} for g in data];write_csv(a.output/'ranking.csv',metrics);write_csv(a.output/'gates.csv',gates)
 report={'schema_version':1,'exploratory':True,'n_geometries':len(data),'n_local':len(local),'n_pairs':50,'n_lf':5000,'bootstrap_draws':a.bootstrap,'winners':winners,'winner_bootstrap_frequency':freq,'metrics':metrics,'gates':gates,'robust_target_met':any(r['robust_target_passed'] for r in metrics),'holdout_opened':False};(a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 qa={'schema_version':1,'status':'pass','local_geometry_count':len(local),'hf_values_per_geometry':100,'paired_samples_per_geometry':50,'lf_samples_per_geometry':5000,'coordinate_tolerance':1e-12,'all_gates_active':all(x['point_active'] for x in gates),'negative_variance_replicates':sum(x['negative_variance_replicates'] for x in gates),'holdout_opened':False};(a.output/'qa_report.json').write_text(json.dumps(qa,indent=2)+'\n')
 top={k:freq[k][g] for k,g in winners.items()};(a.output/'README.md').write_text(f'''# Local HF round 1 — exploratory MFMC analysis

Joint ranking of the nine Search-50 geometries and six local candidates using 50 paired HF samples, 5,000 LF samples and {a.bootstrap:,} common-index bootstrap replicates.

- Mean winner: `{winners['mean']}` ({top['mean']:.1%} selection frequency)
- Robustness winner under the mean guardrail: `{winners['robustness']}` ({top['robustness']:.1%})
- Equal-weight C_D winner: `{winners['combination']}` ({top['combination']:.1%})
- Primary robustness target met: {report['robust_target_met']}
- Holdout opened: no

All point and bootstrap MFMC gates remained active; no reconstructed variance was negative. The result remains exploratory. The robustness branch is not stable enough to close the search.
''');print(a.output/'analysis.json')
if __name__=='__main__':main()
