from pathlib import Path

import numpy as np
import pytest

from mfmc_campaign.field_mfpod.config import load_config
from mfmc_campaign.field_mfpod.models import SurfaceGeometry
from mfmc_campaign.field_mfpod.snapshots import inspect_surface_data, validate_surface_topology
from mfmc_campaign.field_mfpod.workflow import prepare_snapshots


def archive(path, fidelity, ids, area):
    n=len(ids); f=len(area); rng=np.random.default_rng(2)
    np.savez(path,force_per_area=rng.normal(size=(n,f,3)),sample_id=np.asarray(ids),face_area=area,A_ref=np.array([area.sum()]),q_inf=np.ones(n),u_hat_inf=np.tile([1.,0,0],(n,1)),fidelity=np.array([fidelity]),case_name=np.array(['Cube-300km']),geometry_id=np.array(['cube']),face_center=np.arange(f*3).reshape(f,3))


def test_topology_rejects_order_or_frame_mismatch():
    g=SurfaceGeometry(np.array([1.,2.]),3.,'cube',face_center=np.array([[0,0,0],[1,0,0]])); reordered=SurfaceGeometry(np.array([2.,1.]),3.,'cube',face_center=np.array([[1,0,0],[0,0,0]])); assert not validate_surface_topology(g,reordered)['identity_mapping_allowed']
    frame=SurfaceGeometry(g.face_area,3.,'cube',coordinate_frame='wind_aligned',face_center=g.face_center); assert not validate_surface_topology(g,frame)['identity_mapping_allowed']


def test_inspection_duplicates_and_disjoint_roles(tmp_path):
    hp=tmp_path/'h.npz'; lp=tmp_path/'l.npz'; ids=[f's{i}' for i in range(18)]; area=np.ones(4); archive(hp,'DSMC',ids,area); archive(lp,'TPMC',ids,area)
    report=inspect_surface_data('Cube-300km',{'DSMC':hp,'TPMC':lp},tmp_path/'inspection'); assert report['ready']; assert report['n_paired']==18
    cfgp=tmp_path/'config.yaml'; cfgp.write_text(f"""case_name: Cube-300km
geometry_id: cube
fidelity_archives:
  DSMC: {hp}
  TPMC: {lp}
output_root: {tmp_path}/out
pilot:
  paired_samples: 4
reference_samples: 6
infeasible_policy: fail
""")
    cfg=load_config(cfgp); meta=prepare_snapshots(cfg); roles=meta['sample_roles']; assert not (set(roles['pilot'])&set(roles['reference_test'])); assert not (set(roles['production'])&set(roles['reference_test']))


def test_inspection_reports_missing_without_fabrication(tmp_path):
    report=inspect_surface_data('Cube-300km',{'DSMC':tmp_path/'none.npz','TPMC':tmp_path/'also-none.npz'},tmp_path/'inspection'); assert not report['ready']; assert (tmp_path/'inspection'/'data_availability_report.md').exists()
