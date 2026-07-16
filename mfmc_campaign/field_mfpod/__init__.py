"""Field-aware MFMC allocation and multifidelity POD for surface loads.

The full-field APIs target DSMC statistics directly.  Published two-fidelity
MFPOD functions remain available as explicit legacy/reproducibility APIs.
"""

from .allocation import (
    AllocationOptions,
    AllocationResult,
    allocate_counts,
    build_moment_features,
    compare_allocation_strategies,
    compare_field_allocation_strategies,
    optimize_allocation,
    optimize_field_allocation,
    select_empirical_allocation,
)
from .covariance_operator import (
    FullFieldMFMCStatistics,
    covariance_probe_error,
    estimate_full_field_mfmc,
    explicit_full_field_covariance,
    solve_full_field_pod,
)
from .field_statistics import (
    FieldPilotStatistics,
    compute_field_pilot_statistics,
    mean_field_model_covariance,
    second_moment_model_covariance,
)
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
    "AllocationOptions",
    "AllocationResult",
    "FieldPilotStatistics",
    "FullFieldMFMCStatistics",
    "allocate_counts",
    "build_moment_features",
    "compare_allocation_strategies",
    "compare_field_allocation_strategies",
    "apply_published_eigenvalue_correction",
    "build_drag_contribution_snapshots",
    "build_full_traction_snapshots",
    "build_mfpod_linear_operator",
    "compute_adaptive_mfpod",
    "compute_field_pilot_statistics",
    "compute_hf_pod",
    "compute_lf_pod",
    "compute_mfpod",
    "covariance_probe_error",
    "estimate_full_field_mfmc",
    "estimate_global_mfpod_weight",
    "evaluate_subspace",
    "inspect_surface_data",
    "explicit_full_field_covariance",
    "mean_field_model_covariance",
    "principal_angles",
    "optimize_allocation",
    "optimize_field_allocation",
    "select_empirical_allocation",
    "solve_mfpod_eigenproblem",
    "solve_full_field_pod",
    "second_moment_model_covariance",
    "validate_surface_topology",
]
