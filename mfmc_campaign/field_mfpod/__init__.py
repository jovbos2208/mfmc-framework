"""Published multifidelity POD for PICLAS surface-load fields.

The implementation is pinned to Aretz and Willcox, arXiv:2605.29213v1.
The older :mod:`mfmc_campaign.field_pod_mfmc` module remains available as the
experimental reduced-covariance estimator; it is not MFPOD.
"""

from .allocation import allocate_counts, select_empirical_allocation
from .metrics import evaluate_subspace, principal_angles
from .operator import (
    apply_published_eigenvalue_correction,
    build_mfpod_linear_operator,
    compute_adaptive_mfpod,
    compute_mfpod,
    solve_mfpod_eigenproblem,
)
from .pod import compute_hf_pod, compute_lf_pod
from .snapshots import (
    PICLASIdentitySurfaceAdapter,
    SurfaceSnapshotAdapter,
    build_drag_contribution_snapshots,
    build_full_traction_snapshots,
    inspect_surface_data,
    validate_surface_topology,
)
from .weights import estimate_global_mfpod_weight

__all__ = [
    "PICLASIdentitySurfaceAdapter",
    "SurfaceSnapshotAdapter",
    "allocate_counts",
    "apply_published_eigenvalue_correction",
    "build_drag_contribution_snapshots",
    "build_full_traction_snapshots",
    "build_mfpod_linear_operator",
    "compute_adaptive_mfpod",
    "compute_hf_pod",
    "compute_lf_pod",
    "compute_mfpod",
    "estimate_global_mfpod_weight",
    "evaluate_subspace",
    "inspect_surface_data",
    "principal_angles",
    "select_empirical_allocation",
    "solve_mfpod_eigenproblem",
    "validate_surface_topology",
]
