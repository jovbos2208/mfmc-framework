from pathlib import Path

import numpy as np
import pytest

from mfmc_campaign.field_mfpod.config import load_config
from mfmc_campaign.field_mfpod.models import SurfaceGeometry
from mfmc_campaign.field_mfpod.snapshots import inspect_surface_data, validate_surface_topology
from mfmc_campaign.field_mfpod.workflow import (
    field_benchmark,
    field_estimate,
    field_pilot,
    field_pod,
    optimal_allocation,
    prepare_field_snapshots,
    prepare_snapshots,
)


def archive(path, fidelity, ids, area):
    n=len(ids); f=len(area); rng=np.random.default_rng(2)
    np.savez(path,force_per_area=rng.normal(size=(n,f,3)),sample_id=np.asarray(ids),face_area=area,A_ref=np.array([area.sum()]),q_inf=np.ones(n),u_hat_inf=np.tile([1.,0,0],(n,1)),fidelity=np.array([fidelity]),case_name=np.array(['Cube-300km']),geometry_id=np.array(['cube']),coordinate_frame=np.array(['body_fixed']),component_order=np.asarray(['x','y','z']),face_center=np.arange(f*3).reshape(f,3))


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


def test_three_fidelity_field_workflow_writes_matrix_free_outputs(tmp_path):
    ids = [f"s{i}" for i in range(24)]
    area = np.ones(3)
    paths = {}
    for fidelity in ("DSMC", "TPMC", "SENTMAN"):
        paths[fidelity] = tmp_path / f"{fidelity}.npz"
        archive(paths[fidelity], fidelity, ids, area)
    config = tmp_path / "field.yaml"
    config.write_text(
        f"""case_name: Cube-300km
geometry_id: cube
high_fidelity: DSMC
control_variates: [TPMC, SENTMAN]
fidelity_archives:
  DSMC: {paths['DSMC']}
  TPMC: {paths['TPMC']}
  SENTMAN: {paths['SENTMAN']}
output_root: {tmp_path}/out
costs: {{DSMC: 1.0, TPMC: 0.2, SENTMAN: 0.05}}
field_representation:
  quantity: Total_ForcePerArea
  nondimensionalize: true
  area_weighted: true
  coordinate_frame: body_fixed
  centering: pilot_dsmc_mean
pilot:
  paired_samples: 6
reference_samples: 6
field_allocation:
  enabled: true
  mode: enumeration
  mean_weight: 0.25
  second_moment_weight: 0.75
  bootstrap_repeats: 0
allocation_constraints:
  budget: 5.0
  minimum_target: 2
  max_ratios: {{TPMC: 10.0}}
pod:
  number_of_modes: 2
  eigensolver_tolerance: 1.0e-8
validation:
  covariance_probe_count: 5
  compare_scalar_drag_allocation: true
infeasible_policy: fail
random_seed: 12
""",
        encoding="utf-8",
    )
    cfg = load_config(config)
    metadata = prepare_field_snapshots(cfg)
    assert metadata["disjoint_roles"]
    assert field_pilot(cfg)["tpmc_basis_used"] is False
    allocation_result = optimal_allocation(cfg)
    assert allocation_result["diagnostics"]["tpmc_basis_used"] is False
    field_estimate(cfg)
    field_pod(cfg)
    comparison = field_benchmark(cfg)
    assert comparison["methods"] >= 5
    assert (cfg.output_dir / "estimator" / "mean_field.npz").exists()
    assert (cfg.output_dir / "pod" / "full_field_modes.npz").exists()
    assert (cfg.output_dir / "benchmark" / "benchmark_summary.csv").exists()
