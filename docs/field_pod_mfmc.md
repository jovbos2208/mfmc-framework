# Full-field, allocation-first MFMC/POD

This workflow estimates the DSMC mean field and covariance directly in the
complete, common PICLAS surface space. TPMC and Sentman are control variates;
neither model supplies a POD basis or constrains the final DSMC modes.

The production sequence is:

```text
paired DSMC/TPMC/Sentman pilot
  -> measured costs and full-field pilot covariance blocks
  -> robust integer allocation and equal-cost comparator allocations
  -> nested production streams plus an independent DSMC reference
  -> full-field MFMC mean and matrix-free covariance operator
  -> matrix-free POD and equal-cost validation
```

## Field representation

For face `j` and sample `i`, the stored traction is converted to

```text
z[i,j,:] = sqrt(A[j] / A_ref[i]) * force_per_area[i,j,:] / q_inf[i]
```

All fidelities must therefore use the same ordered reference surface,
body-fixed coordinate frame, component order, face areas and normalization.
Sentman fields are conservatively mapped to the PICLAS surface before they are
archived.

Each archive is an appendable NPZ with at least:

```text
force_per_area: (n_samples, n_faces, 3)
sample_id:      (n_samples,)
face_area:      (n_faces,)
A_ref or A_ref_per_sample
q_inf:          (n_samples,)
u_hat_inf:      (n_samples, 3)
fidelity, case_name, geometry_id
```

`coordinate_frame`, `component_order`, `face_center`, `face_normal`, `C_D`,
`cpu_hours` and `hardware` should also be stored.

## Pilot statistics and allocation

The pilot IDs are disjoint from production and reference IDs. For every pilot
ID all three models are evaluated. The code computes:

- `Gamma^(mu)` from complete-field Hilbert inner products;
- `Gamma^(M)` from squared snapshot cross-Gram matrices, without forming an
  `(n_state, n_state)` second-moment matrix;
- separate optimal control weights for the mean and second moment;
- measured mean CPU-hour costs when positive pilot costs are available;
- a bootstrap-robust integer allocation under the configured budget.

The default objective weights are `mean_weight: 0.25` and
`second_moment_weight: 0.75`.
The portable paper configs express the budget in DSMC equivalents. After the
pilot, `budget_hf_equivalent` is multiplied by the measured mean DSMC
CPU-hour cost, so allocation costs and budget use the same unit.

Before production, the framework also plans all predeclared equal-cost
comparators: DSMC-only, DSMC+TPMC, fixed ratios, scalar-drag allocation,
greedy, continuous-round and direct enumeration. Solver production evaluates
the component-wise union of their required nested counts. This prevents a
baseline from being silently evaluated with a smaller budget merely because
its samples were not generated.

## Matrix-free estimator and POD

The DSMC target mean and second moment are estimated with separate pilot-derived
control weights. The covariance action is assembled as sums of
`X.T @ (X @ vector)` terms and a mean outer-product action. The full covariance
matrix is not formed. Leading modes are computed with `scipy.sparse.linalg.eigsh`.
Raw Ritz values and negative-eigenvalue diagnostics are retained.

## Restartable production

Run commands from the repository root. The portable templates are:

```text
configs/mfpod/cube_tpmc_sentman.yaml
configs/mfpod/goce_tpmc_sentman.yaml
```

Create the deterministic plan without submitting jobs:

```bash
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --dry-run
```

Use explicit checkpoints on the cluster:

```bash
# Paired pilot only
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --stop-after pilot

# Pilot statistics, measured-cost allocation and comparator union
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --resume --stop-after allocation

# Independent DSMC reference and required nested production streams
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --resume --stop-after production

# MFMC mean/covariance, POD and comparisons; no new solver work
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --resume
```

At every checkpoint:

```bash
mfmc-mfpod production-status \
  --config configs/mfpod/cube_tpmc_sentman.yaml
```

Completed archive IDs are checked before submission, so `--resume` does not
resubmit successfully archived samples.

## Persistent outputs

Outputs live below
`outputs/field_pod_mfmc_three_fidelity/<case>/`.

Important files are:

```text
production/sample_plan.json
production/state.json
production/pilot_fields.npz
production/roles.json
pilot/field_pilot_statistics.json
allocation/optimal_allocation.json
allocation/optimal_allocation_candidates.csv
allocation/allocation_strategy_comparison.csv
estimator/mean_field.npz
estimator/covariance_operator_metadata.json
pod/full_field_modes.npz
pod/eigensolver_diagnostics.json
benchmark/benchmark_summary.csv
benchmark/benchmark_metadata.json
```

`state.json` records the primary field-aware allocation,
`comparison_allocations`, and the union in `required_production_counts`.
Review these counts after the allocation checkpoint and before allowing solver
production.

## Sentman mapping

If a mapping archive is absent, production first runs the PICLAS pilot and uses
one archived DSMC VTU to construct the canonical ADBSat OBJ/MAT and conservative
triangle-to-PICLAS mapping. Runtime ADBSat assets are written below
`outputs/mfpod_runtime/`; tracked source assets are not modified.

The conservative mapping satisfies

```text
t_reference[j] =
  sum(A_triangle[k] * t_triangle[k] for k mapped to j) / A_reference[j]
```

and therefore preserves integrated force.

## Interpretation

- Pilot, production and DSMC-reference IDs are disjoint.
- Pilot data select costs, weights and allocations; they are not reused in the
  final estimator.
- The reference archive is DSMC-only and is used only for validation.
- The main allocation and all baselines use the same configured production
  budget. Pilot and reference costs are reported separately from that budget.
- No TPMC POD basis or TPMC projection residual appears in the main method.
