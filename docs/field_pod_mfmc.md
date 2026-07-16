# Field POD/MFMC Demonstrator

This document describes the first narrow field-level POD/MFMC demonstrator for
archived surface-load fields. DSMC is the high-fidelity target, PICLAS TPMC is
the working-basis source, and ADBSat Sentman can be used as a second low-fidelity
control variate after conservative mapping to the PICLAS reference surface.

## Scientific Target

For one uncertainty realization xi, the archived DSMC surface-load snapshot is
`z_H(xi)`. The target statistics are the DSMC mean, second moment, and
covariance:

```text
mu_H    = E[z_H]
M_H     = E[z_H z_H^T]
Sigma_H = M_H - mu_H mu_H^T
```

The final POD modes come from the MFMC-estimated DSMC projected covariance. The
TPMC basis is only a computational working basis; it does not redefine the DSMC
target covariance.

## Snapshot Definitions

The default `full_traction` snapshot uses PICLAS `Total_ForcePerArea` values
and builds

```text
z_l[n, j, c] = sqrt(A_j / A_ref) * t_l[n, j, c] / q_inf[n]
```

for face `j` and component `c in {x, y, z}`. This weighting makes the Euclidean
norm correspond to a discrete area-weighted surface-integral norm of
dimensionless traction.

The secondary `drag_contribution` snapshot is

```text
c_D,j = -(t_j dot u_hat_inf) A_j / (q_inf A_ref)
```

and `sum_j c_D,j` must reproduce an archived scalar `C_D` when that scalar is
available and uses the same sign/reference-area convention.

## Required Archive Format

The first implementation reads one `.npz` file per fidelity:

```text
force_per_area: (n_samples, n_faces, 3)
sample_id:      (n_samples,)
face_area:      (n_faces,)
A_ref:          scalar or (1,)
q_inf:          scalar or (n_samples,)
u_hat_inf:      (3,) or (n_samples, 3)
```

Preferred optional fields are `face_center`, `face_normal`, `reference_point`,
and `C_D`.

## Workflow

Use the existing package CLI:

```bash
python3 -m mfmc_campaign.cli check-field-data configs/studies/vleo_mfmc_paper/field_pod_mfmc_cube_300km.yaml
python3 -m mfmc_campaign.cli run-field-pod-mfmc configs/studies/vleo_mfmc_paper/field_pod_mfmc_cube_300km.yaml
```

`check-field-data` writes `data_availability_report.json` and, when required
fields are missing, `field_pod_mfmc_missing_data_<case>.md`.

## Restartable Solver Production

The Cube template contains a production definition for DSMC, TPMC and Sentman.
From the repository root, first inspect the deterministic plan without
submitting jobs:

```bash
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --dry-run
```

Then launch all stages:

```bash
mfmc-mfpod production --config configs/mfpod/cube_tpmc_sentman.yaml
```

The command runs paired pilots at every fidelity, replaces the scenario costs
with finite positive pilot mean CPU-hour costs, solves the full-field integer
allocation, evaluates the allocated nested production streams, evaluates a
separate DSMC reference set, and runs the field estimator, matrix-free POD and
benchmark. Its persistent files are below
`outputs/field_pod_mfmc_three_fidelity/<case>/production/`:

- `sample_plan.json`: immutable samples, IDs and disjoint role streams
- `state.json`: completed stages, measured costs and submitted counts
- `roles.json`: exact pilot/reference/production IDs used by analysis

An interrupted run is resumed with:

```bash
mfmc-mfpod production \
  --config configs/mfpod/cube_tpmc_sentman.yaml \
  --resume
```

Existing archive IDs are checked before submission, so completed samples are
not submitted again. Use `--stop-after pilot`, `allocation`, or `production`
to create deliberate checkpoints, and inspect them with
`mfmc-mfpod production-status --config ...`.

Before the first real run, install `piclas` and `piclas2vtk` with
`scripts/configure_piclas.sh`. If the Sentman mapping is absent, production
runs the PICLAS pilot first, builds the mapping from its first DSMC VTU, and
then continues with Sentman. Generated OBJ/MAT assets live in the ignored
`outputs/mfpod_runtime/` copy, leaving tracked ADBSat assets unchanged. The
manual mapping procedure below remains useful when selecting a different
trusted canonical VTU.

## Canonical ADBSat Surface

Build the ADBSat OBJ, MAT and immutable triangle-to-PICLAS-cell mapping from one
trusted PICLAS output VTU. OBJ and MAT must always be replaced together.

```bash
python3 -m mfmc_campaign.cli build-adbsat-surface \
  --vtu /path/to/canonical/output0001.vtu \
  --obj external/ADBSat/inou/obj_files/Cube.obj \
  --mat external/ADBSat/inou/models/Cube.mat \
  --mapping data/field_inputs/cube_300km/Cube.surface_mapping.npz
```

Use `--length-scale-to-m` only when the VTU coordinates are not already in
metres. The generated MAT embeds a SHA-256 mesh fingerprint. ADBSat refuses to
write panel fields if the loaded MAT does not carry the expected fingerprint.

Enable the Sentman surface archive in the `legacy_adbsat` model configuration:

```yaml
- id: Sentman
  kind: legacy_adbsat
  method: Sentman
  kwargs:
    simulation_script: python {base_dir}/simulate.py
    base_dir: external/ADBSat
    surface_archive:
      enabled: true
      mapping_path: data/field_inputs/cube_300km/Cube.surface_mapping.npz
      path: data/field_inputs/cube_300km/SENTMAN_surface_loads.npz
      case_name: Cube-300km
      geometry_id: Cube
      regime_id: CUBE_300KM
      append: true
```

ADBSat writes local pressure/shear panel tractions. The adapter maps them back
to PICLAS cells using

```text
t_reference[j] = sum(k in triangles(j), A[k] * t_panel[k]) / A_reference[j]
```

which exactly preserves integrated force.

For three-fidelity MFPOD, add the Sentman archive and cost:

```yaml
high_fidelity: DSMC
low_fidelity_basis_source: TPMC
control_variates: [TPMC, SENTMAN]
fidelity_archives:
  DSMC: /path/to/DSMC_surface_loads.npz
  TPMC: /path/to/TPMC_surface_loads.npz
  SENTMAN: /path/to/SENTMAN_surface_loads.npz
costs:
  DSMC: 1.0
  TPMC: 0.05
  SENTMAN: 0.001
```

All controls use the TPMC working basis. Shared control weights are obtained
from the joint covariance of the selected paired DSMC/TPMC/Sentman responses.
After paying for the paired samples, additional LF samples are assigned
greedily according to estimated `beta^2 Var(response)` per unit cost.

## Estimator

The reduced coefficients are

```text
b_l(xi) = Psi_s^T (z_l(xi) - z_ref)
```

where `Psi_s` is the TPMC POD basis and `z_ref` is the mean TPMC snapshot used
to construct it. The demonstrator estimates `E[b_H]`, `E[b_H b_H^T]`, and the
reduced DSMC covariance. The default `shared_weights` mode uses one scalar
control-variate weight derived from `||b||_2` and applies that same weight to
all coefficient means and coefficient-product moments.

## Diagnostics

Projection residuals are gatekeeping diagnostics:

```text
eps_proj = ||z_H - z_ref - Psi_s Psi_s^T (z_H - z_ref)||_2
           / ||z_H - z_ref||_2
```

MFMC can reduce Monte Carlo error inside the reduced basis, but it cannot remove
projection/truncation error. If residuals are large, the LF basis does not span
important archived DSMC surface-load structure.

## Limitations

- The method estimates archived DSMC-target covariance in reduced space, not
  external physical truth.
- Projection error is not removed by MFMC.
- ADBSat CLL uses the same mapping machinery but has not yet been included in
  the production MFPOD configurations.
- Sobol indices are not implemented and require a different sampling design.
- No claim is made that TPMC modes are DSMC POD modes before the projected DSMC
  covariance is estimated.
