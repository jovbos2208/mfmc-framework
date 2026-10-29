#!/usr/bin/env python3
"""Build meshes and locked 50-point configs for selected local-search geometries."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mfmc_campaign.parametric_geometry import build_gmsh_exterior_mesh,build_piclas_hdf5_mesh

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--search-root',type=Path,default=Path('outputs/vleo_cll/local_search_round1'));p.add_argument('--manifest',type=Path,default=Path('outputs/vleo_cll_mfmc_extension_bundle/manifests/paired_search_50.json'));p.add_argument('--template',type=Path,default=Path('outputs/vleo_cll_mfmc_extension_bundle/configs/cylinder_hex_wp5_000_cll_extension10.yaml'));p.add_argument('--output',type=Path,default=Path('outputs/vleo_cll_local_hf_round1_bundle'));p.add_argument('--gmsh',default='gmsh');p.add_argument('--pyhope',default='pyhope');a=p.parse_args()
 selection=json.loads((a.search_root/'hf_selection.json').read_text()); samples=json.loads(a.manifest.read_text())['samples']; template=yaml.safe_load(a.template.read_text());(a.output/'geometries').mkdir(parents=True,exist_ok=True);(a.output/'configs').mkdir(exist_ok=True);(a.output/'manifests').mkdir(exist_ok=True)
 shutil.copy2(a.manifest,a.output/'manifests'/a.manifest.name); built=[]
 for item in selection['selected']:
  gid=item['geometry_id'];src=a.search_root/'geometries'/gid;dst=a.output/'geometries'/gid
  if dst.exists(): shutil.rmtree(dst)
  shutil.copytree(src,dst);mp=dst/f'{gid}.manifest.json'
  gm=build_gmsh_exterior_mesh(mp,gmsh_executable=a.gmsh);hm=build_piclas_hdf5_mesh(mp,pyhope_executable=a.pyhope)
  c=json.loads(json.dumps(template));c['study']['id']=f'vleo_cll_local_hf_r1_{gid}';c['geometries'][0]['id']=gid;c['geometries'][0]['name']=gid;c['geometries'][0]['metadata']['hf_mesh']=f'../outputs/vleo_cll_local_hf_round1_bundle/geometries/{gid}/{gid}_mesh.h5';c['geometries'][0]['metadata']['lf_model']=gid
  hf_kwargs=c['models']['hf']['kwargs'];hf_kwargs['mpi_procs']=36;hf_kwargs['submission_group_size']=10;hf_kwargs['simulator_module']='PICLas_prandtl'
  c['sampling']={'method':'explicit','sample_count':50,'max_production_samples':50,'source_contract':f'locked paired_search_50.json SHA-256 {sha(a.manifest)}','explicit_samples':[{'cll_alpha_n':s['cll_alpha_n'],'cll_sigma_t':s['cll_sigma_t']} for s in samples]};c['pilot'].update({'size':50,'sizes':[50]});c['outputs']['dir']=f'campaign_outputs/vleo_cll/local_hf_round1/{gid}';c['seeds']={'global':20261030}
  cp=a.output/'configs'/f'{gid}_cll_search50.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False));built.append({'geometry_id':gid,'branch':item['branch'],'config':str(cp.relative_to(a.output)),'n_elements':hm['nElems'],'gmsh_valid':gm['valid'],'h5_valid':hm['valid']})
 summary={'schema_version':1,'status':'ready_for_hf_execution','exploratory':True,'holdout_used':False,'paired_manifest_sha256':sha(a.manifest),'selection_sha256':sha(a.search_root/'hf_selection.json'),'geometries':built,'commands':['Validate all configs before submission.','Run each selected geometry once; do not run any holdout config.']};bp=a.output/'bundle_manifest.json';bp.write_text(json.dumps(summary,indent=2)+'\n')
 gids=' '.join(x['geometry_id'] for x in built)
 (a.output/'run_hf_array.slurm').write_text(f'''#!/bin/bash
#SBATCH --job-name=vleo-local-r1
#SBATCH --array=0-5%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=48:00:00
#SBATCH --output=outputs/vleo_cll_local_hf_round1_bundle/logs/hf_%A_%a.out
#SBATCH --error=outputs/vleo_cll_local_hf_round1_bundle/logs/hf_%A_%a.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source .venv/bin/activate
mkdir -p outputs/vleo_cll_local_hf_round1_bundle/logs
# Do not leak the launcher's memory mode into nested PICLas sbatch/srun jobs.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
geometries=({gids})
gid="${{geometries[$SLURM_ARRAY_TASK_ID]}}"
python -m mfmc_campaign.cli run-pilot-correlation "outputs/vleo_cll_local_hf_round1_bundle/configs/${{gid}}_cll_search50.yaml"
''')
 (a.output/'README.md').write_text('''# Exploratory VLEO CLL local HF round 1

This locked bundle contains six LF-screened geometries and the common 50-point Search manifest. It does not contain or use holdout samples. The SLURM array is throttled to one active geometry (`0-5%1`); within that geometry the campaign retains a submission batch size of 10.

1. Verify `sha256sum -c SHA256SUMS` inside this directory.
2. Install each selected OBJ/MAT asset in the server ADBSat runtime.
3. Validate every YAML config.
4. Submit `sbatch outputs/vleo_cll_local_hf_round1_bundle/run_hf_array.slurm`.

Do not submit any holdout job. Each selected geometry is new, so evaluating the common Search-50 coordinates does not rerun an existing geometry result.
''')
 files=[x for x in a.output.rglob('*') if x.is_file() and x.name!='SHA256SUMS'];(a.output/'SHA256SUMS').write_text('\n'.join(f'{sha(x)}  {x.relative_to(a.output)}' for x in sorted(files))+'\n');print(bp)
if __name__=='__main__':main()
