import numpy as np

from mfmc_campaign.field_mfpod.metrics import evaluate_subspace, principal_angles
from mfmc_campaign.field_mfpod.operator import (
    apply_published_eigenvalue_correction,
    build_mfpod_linear_operator,
    compute_adaptive_mfpod,
    compute_mfpod,
    explicit_mfpod_operator,
)
from mfmc_campaign.field_mfpod.pod import compute_hf_pod
from mfmc_campaign.field_mfpod.snapshots import build_drag_contribution_snapshots, build_full_traction_snapshots
from mfmc_campaign.field_mfpod.weights import estimate_global_mfpod_weight


def data(seed=5, m=18, n=12):
    rng=np.random.default_rng(seed); h=rng.normal(size=(m,n)); l=.9*h+.2*rng.normal(size=(m,n)); e=rng.normal(size=(25,n)); return h,l,e


def test_full_traction_norm_and_drag_contribution():
    rng=np.random.default_rng(1); force=rng.normal(size=(4,3,3)); area=np.array([1.,2.,4.]); q=np.array([2.,3.,4.,5.]); aref=7.; u=np.tile([0.,1.,0.],(4,1))
    z=build_full_traction_snapshots(force,area,q,aref)
    expected=np.sum(area[None,:,None]/aref*(force/q[:,None,None])**2,axis=(1,2))
    np.testing.assert_allclose(np.sum(z*z,axis=1),expected)
    drag=build_drag_contribution_snapshots(force,area,q,aref,u)
    np.testing.assert_allclose(np.sum(drag,axis=1),-np.einsum('nfc,nc,f->n',force,u,area)/(q*aref))


def test_operator_action_matches_explicit_and_backends_agree():
    h,l,e=data(); alpha=.72; explicit=explicit_mfpod_operator(h,l,e,alpha); op=build_mfpod_linear_operator(h,l,e,alpha); v=np.arange(h.shape[1],dtype=float)
    np.testing.assert_allclose(op@v,explicit@v,rtol=1e-12,atol=1e-12)
    reduced=compute_mfpod(h,l,e,alpha,backend='reduced_snapshot_span',n_modes=5)
    linear=compute_mfpod(h,l,e,alpha,backend='linear_operator',n_modes=5)
    np.testing.assert_allclose(reduced.eigenvalues,linear.eigenvalues,rtol=1e-8,atol=1e-10)
    assert np.max(principal_angles(reduced.modes,linear.modes))<1e-6


def test_global_alpha_is_sample_covariance_over_variance():
    h,l,_=data(); result=estimate_global_mfpod_weight(h,l,bootstrap_repeats=10)
    x=np.sum(h*h,axis=1); y=np.sum(l*l,axis=1)
    assert np.isclose(result['alpha_raw'],np.cov(x,y,ddof=1)[0,1]/np.var(y,ddof=1))


def test_published_nonpositive_correction_uses_hf_mc_energy():
    h,_,_=data(); q,_=np.linalg.qr(np.random.default_rng(4).normal(size=(h.shape[1],3))); raw=np.array([2.,0.,-1.])
    corrected,mask,replacements=apply_published_eigenvalue_correction(raw,q,h)
    np.testing.assert_allclose(corrected[mask],np.sum((h@q[:,mask])**2,axis=0)/h.shape[0]); assert np.all(corrected>=0)


def test_adaptive_update_returns_orthonormal_modes_and_weights():
    h,l,e=data(m=30,n=20); result=compute_adaptive_mfpod(h,l,e,max_modes=4)
    np.testing.assert_allclose(result.modes.T@result.modes,np.eye(4),atol=1e-9); assert len(result.alpha)==4


def test_metrics_energy_identity_and_known_angles():
    rng=np.random.default_rng(8); z=rng.normal(size=(100,10)); pod=compute_hf_pod(z,4); result=evaluate_subspace(pod.modes,z)
    assert abs(result['projection_error']+result['captured_energy']-1)<1e-12
    assert np.max(principal_angles(pod.modes,pod.modes*np.array([1,-1,1,-1])))<1e-7
