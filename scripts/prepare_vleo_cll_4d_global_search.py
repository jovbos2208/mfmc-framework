#!/usr/bin/env python3
"""Prepare a global geometry DOE and locked 4D GSI/attitude LF screen."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mfmc_campaign.parametric_geometry import CylinderHexSpec,ParametricGeometryError,build_geometry_assets,generate_cylinder_hex

SEED_GEOMETRY=20261101;SEED_UNCERTAINTY=20261102
BOUNDS={
 'nose_length_fraction':(.05,.40),'tail_length_fraction':(.05,.40),
 'front_width_height_ratio':(.50,2.0),'rear_width_height_ratio':(.50,2.0),
 'nose_end_scale':(.10,.50),'tail_end_scale':(.20,.80),
}
ANGLE_ANCHORS=[(0,0),(5,0),(-5,0),(0,5),(0,-5),(10,0),(-10,0),(0,10),(0,-10),(5,5),(5,-5),(-5,5),(-5,-5)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lhs(n,d,seed):
 r=np.random.default_rng(seed);x=np.empty((n,d))
 for j in range(d):x[:,j]=(r.permutation(n)+r.random(n))/n
 return x
def candidate_specs(n):
 x=lhs(10000,len(BOUNDS),SEED_GEOMETRY);valid=[]
 for row in x:
  vals={k:lo+row[i]*(hi-lo) for i,(k,(lo,hi)) in enumerate(BOUNDS.items())};spec=CylinderHexSpec(**vals).scaled(.1)
  try:generate_cylinder_hex(spec)
  except ParametricGeometryError:continue
  valid.append((row,spec))
 z=np.array([r for r,_ in valid]); d=z.shape[1]; bins1=np.minimum((8*z).astype(int),7)
 pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
 bins2=np.array([[4*bins1[k,i]//8*4 + 4*bins1[k,j]//8 for i,j in pairs] for k in range(len(z))])
 chosen=[int(np.argmin(np.sum((z-.5)**2,axis=1)))]; dist=np.linalg.norm(z-z[chosen[0]],axis=1)
 count1=np.zeros((d,8),int); count2=np.zeros((len(pairs),16),int)
 def add_counts(index):
  for j in range(d):count1[j,bins1[index,j]]+=1
  for j in range(len(pairs)):count2[j,bins2[index,j]]+=1
 add_counts(chosen[0]); target1=n/8; target2=n/16
 while len(chosen)<n:
  one=np.sum([np.maximum(target1-count1[j,bins1[:,j]],0)/target1 for j in range(d)],axis=0)
  two=np.sum([np.maximum(target2-count2[j,bins2[:,j]],0)/target2 for j in range(len(pairs))],axis=0)
  score=one+0.30*two+0.30*dist; score[chosen]=-np.inf
  i=int(np.argmax(score));chosen.append(i);add_counts(i);dist=np.minimum(dist,np.linalg.norm(z-z[i],axis=1))
 return [valid[i][1] for i in chosen]
def uncertainty_samples(n):
 x=lhs(n,4,SEED_UNCERTAINTY);rows=[]
 for i,r in enumerate(x):
  radius=10*np.sqrt(r[2]);phi=2*np.pi*r[3];aoa=radius*np.cos(phi);aos=radius*np.sin(phi);role='space_filling_4d'
  if i<len(ANGLE_ANCHORS):aoa,aos=ANGLE_ANCHORS[i];role='attitude_anchor'
  rows.append({'sample_id':f'joint4d_{i:05d}','role':role,'cll_alpha_n':float(r[0]),'cll_sigma_t':float(r[1]),'aoa_deg':float(aoa),'aos_deg':float(aos),'off_axis_deg':float(np.hypot(aoa,aos)),'random_seed':202611020000+i})
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/vleo_cll_4d_global_search_bundle'));p.add_argument('--anchor-root',type=Path,default=Path('outputs/vleo_cll_server_run_bundle/geometries'));p.add_argument('--n-new-geometries',type=int,default=40);p.add_argument('--n-lf',type=int,default=10000);a=p.parse_args();geo=a.output/'geometries';cfg=a.output/'configs';man=a.output/'manifests';geo.mkdir(parents=True,exist_ok=True);cfg.mkdir(exist_ok=True);man.mkdir(exist_ok=True)
 geometries=[]
 for i,spec in enumerate(candidate_specs(a.n_new_geometries)):
  gid=f'cylinder_hex_4d_global_{i:03d}';m=build_geometry_assets(geo/gid,spec,design_id=gid,uniform_scale_factor=.1,body_mesh_size_m=.03,farfield_mesh_size_m=.15);geometries.append({'geometry_id':gid,'role':'new_global_doe','design_fingerprint':m['design_fingerprint'],'parameters':{k:getattr(spec,k) for k in BOUNDS}})
 for src in sorted(a.anchor_root.glob('cylinder_hex_wp5_00[0-8]')):
  gid=src.name;dst=geo/gid
  if dst.exists():shutil.rmtree(dst)
  dst.mkdir();
  for suffix in ('obj','mat','manifest.json','surface.npz'):shutil.copy2(src/f'{gid}.{suffix}',dst/f'{gid}.{suffix}')
  m=json.loads((dst/f'{gid}.manifest.json').read_text());s=m['specification']; geometries.append({'geometry_id':gid,'role':'historical_global_anchor','design_fingerprint':m['design_fingerprint'],'parameters':{k:(s.get(k) if s.get(k) is not None else s.get('width_height_ratio')) for k in BOUNDS}})
 samples=uncertainty_samples(a.n_lf);mp=man/'joint_gsi_attitude_lf10000.json';mp.write_text(json.dumps({'schema_version':1,'role':'global_4d_lf_screen','seed':SEED_UNCERTAINTY,'n':len(samples),'angle_domain':{'shape':'disk','radius_deg':10.0,'core_radius_deg':5.0},'samples':samples},indent=2)+'\n')
 for g in geometries:
  payload={'schema_version':1,'geometry_id':g['geometry_id'],'role':g['role'],'manifest':'manifests/joint_gsi_attitude_lf10000.json','manifest_sha256':sha(mp),'output_dir':f'campaign_outputs/vleo_cll/global_6d_geometry_lf10000/{g["geometry_id"]}','reference_area_convention':'A_total/2'};(cfg/f'{g["geometry_id"]}_lf4d.json').write_text(json.dumps(payload,indent=2)+'\n')
 bundle_path=a.output.as_posix(); gids=' '.join(g['geometry_id'] for g in geometries);(a.output/'run_lf4d_array.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=vleo-4d-lf
#SBATCH --array=0-{len(geometries)-1}%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output={bundle_path}/logs/lf_%A_%a.out
#SBATCH --error={bundle_path}/logs/lf_%A_%a.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source .venv/bin/activate
mkdir -p {bundle_path}/logs
geometries=({gids})
gid="${{geometries[$SLURM_ARRAY_TASK_ID]}}"
python scripts/run_vleo_cll_adbsat_lf_manifest.py --framework-root . --manifest {bundle_path}/manifests/joint_gsi_attitude_lf10000.json --geometry-id "$gid" --output-dir "campaign_outputs/vleo_cll/global_6d_geometry_lf10000/$gid" --quiet-adbsat --resume
''')
 design={'schema_version':1,'status':'ready_for_global_4d_lf_screen','geometry_seed':SEED_GEOMETRY,'uncertainty_seed':SEED_UNCERTAINTY,'n_new_geometries':a.n_new_geometries,'n_anchor_geometries':9,'n_total_geometries':len(geometries),'n_lf_samples':len(samples),'geometry_bounds':BOUNDS,'uncertainty':{'cll_alpha_n':[0,1],'cll_sigma_t':[0,1],'attitude_domain':'uniform-area disk','max_off_axis_deg':10,'reported_core_deg':5},'holdout_used':False,'geometries':geometries};dp=a.output/'design.json';dp.write_text(json.dumps(design,indent=2)+'\n')
 (a.output/'README.md').write_text('''# Global 4D VLEO CLL search — Stage A

This bundle restarts geometry selection over joint CLL and attitude uncertainty. It contains a six-dimensional geometry DOE (nose/tail length, separate front/rear width-height ratios, and separate nose/tail end scales), nine historical anchors, and one common 10,000-point LF manifest. The chamfer remains fixed at 0.15. AoA/AoS are sampled uniformly by area inside a 10-degree off-axis disk; the 5-degree core is reported separately.

This stage is LF-only. Do not run PICLas or open the holdout before the global LF screen has selected a diverse HF shortlist.

## Method resource

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
''')
 files=[f for f in a.output.rglob('*') if f.is_file() and f.name!='SHA256SUMS'];(a.output/'SHA256SUMS').write_text('\n'.join(f'{sha(f)}  {f.relative_to(a.output)}' for f in sorted(files))+'\n');print(dp)
if __name__=='__main__':main()
