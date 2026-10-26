#!/usr/bin/env python3
"""Evaluate one geometry on a locked ADBSat-CLL sample manifest."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--framework-root',type=Path,default=Path('.'));p.add_argument('--manifest',type=Path,required=True);p.add_argument('--geometry-id',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 root=a.framework_root.resolve();manifest=a.manifest if a.manifest.is_absolute() else root/a.manifest;out=a.output_dir if a.output_dir.is_absolute() else root/a.output_dir;out.mkdir(parents=True,exist_ok=True);payload=json.loads(manifest.read_text());samples=payload['samples'];csvp=out/'adbsat_lf_5000.csv';done={}
 if a.resume and csvp.exists():
  with csvp.open(newline='',encoding='utf-8') as f:done={r['sample_id']:r for r in csv.DictReader(f)}
 adbsat=root/'ADBSat-PyVersion';sys.path.insert(0,str(adbsat));import simulate
 rows=[]
 for i,s in enumerate(samples):
  if s['sample_id'] in done:rows.append(done[s['sample_id']]);continue
  env={'environment_model':'csv','geometry_id':a.geometry_id,'lf_model':a.geometry_id,'flow_zero_direction':[1.,0.,0.],'aos_deg':0.,'aoa_deg':0.,'adbsat_aos_offset_deg':0.,'cll_alpha_n':s['cll_alpha_n'],'cll_sigma_t':s['cll_sigma_t'],'wall_temperature_k':300.,'relative_speed_mps':7718.371,'atmosphere_row':[2.65669645265334e-11,0.,0.,1e15,0.,0.,0.,0.,0.,0.,961.],'write_input_audit':False,'write_panel_surface_field':False,'write_mat':False,'surface_sample_id':s['sample_id']}
  coeff,ms=simulate.run_simulation('CLL',930000+i,313.,0.,str(adbsat),env_payload_path=env);cd=float(coeff['C_D'] if isinstance(coeff,dict) else coeff[0]);rows.append({'sample_id':s['sample_id'],'sample_index':i,'role':s['role'],'cll_alpha_n':s['cll_alpha_n'],'cll_sigma_t':s['cll_sigma_t'],'C_D':cd,'C_D2':cd*cd,'runtime_ms':float(ms)})
  if (i+1)%100==0:
   with csvp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with csvp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report={'schema_version':1,'geometry_id':a.geometry_id,'manifest':str(manifest),'manifest_sha256':sha(manifest),'n':len(rows),'output':str(csvp),'reference_area_convention':'A_total/2'};(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(out/'manifest.json')
if __name__=='__main__':main()
