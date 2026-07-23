# Cluster runbook: pilot-allocated full-field MFMC/POD

This runbook deliberately separates solver submission into reviewable
checkpoints. Commands must be executed from the repository root.

## 1. Install the checkout

On a connected login node:

```bash
git clone <repository-url> mfmc-framework
cd mfmc-framework
./scripts/bootstrap_cluster.sh test
source .venv/bin/activate
```

On an offline cluster, use the wheelhouse procedure in
`docs/cluster_deployment.md`.

Install the two site-built PICLAS executables into the checkout:

```bash
PICLAS_BIN=/cluster/path/to/piclas \
PICLAS2VTK_BIN=/cluster/path/to/piclas2vtk \
  ./scripts/configure_piclas.sh
```

Run the non-production checks:

```bash
python scripts/check_production_runtime.py
./scripts/smoke_test.sh
python -m pytest -q tests/field_mfpod tests/test_field_pod_mfmc.py
```

Use a persistent login session such as `tmux` for production orchestration.
The command submits SLURM work, polls it and postprocesses completed jobs.

## 2. Select and freeze a case configuration

Start with Cube:

```bash
export MFPOD_CONFIG=configs/mfpod/cube_tpmc_sentman.yaml
```

After Cube is complete, GOCE uses:

```bash
export MFPOD_CONFIG=configs/mfpod/goce_tpmc_sentman.yaml
```

Before creating a plan, review:

- `pilot.paired_samples`;
- `reference_samples`;
- `allocation_constraints.budget_hf_equivalent`;
- `production.maximum_counts`;
- `validation.fixed_ratios`;
- SLURM resources under `production.models`;
- geometry, regime and uncertainty distributions.

Do not edit sampling, role counts or seeds after `sample_plan.json` has been
created. For a changed scientific design, use a new `case_name` and
`output_root` so the new campaign has a separate immutable plan.

## 3. Create the deterministic plan

This command does not submit jobs:

```bash
mfmc-mfpod production --config "$MFPOD_CONFIG" --dry-run
mfmc-mfpod production-status --config "$MFPOD_CONFIG"
```

Check that the printed pilot, reference and maximum counts match the intended
paper design.

## 4. Run only the paired pilot

```bash
mfmc-mfpod production \
  --config "$MFPOD_CONFIG" \
  --stop-after pilot
```

If the command or login session is interrupted:

```bash
mfmc-mfpod production \
  --config "$MFPOD_CONFIG" \
  --resume --stop-after pilot
```

The pilot stage evaluates the same IDs with DSMC, TPMC and Sentman. Existing
archive IDs are skipped on resume.

Verify:

```bash
mfmc-mfpod production-status --config "$MFPOD_CONFIG"
```

All three archive counts must be at least `pilot.paired_samples`.

## 5. Compute and review pilot-based allocations

This stage submits no production or reference jobs:

```bash
mfmc-mfpod production \
  --config "$MFPOD_CONFIG" \
  --resume --stop-after allocation
```

Inspect the state and scientific products:

```bash
mfmc-mfpod production-status --config "$MFPOD_CONFIG"
python -m json.tool \
  outputs/field_pod_mfmc_three_fidelity/Cube-300km-TPMC-Sentman/production/state.json
```

For GOCE, replace the case directory with
`GOCE-244km-TPMC-Sentman`.

For the selected case, verify these entries in its `state.json`:

- `allocation_costs`: positive comparable CPU-hour costs;
- `allocation_budget`: the HF-equivalent budget and its resolved CPU-hour
  value;
- `allocation`: the primary field-aware counts and separate mean/second-moment
  weights;
- `comparison_allocations`: all equal-cost baselines;
- `required_production_counts`: component-wise union that will actually run.

Also inspect:

```text
pilot/field_pilot_statistics.json
allocation/optimal_allocation.json
allocation/optimal_allocation_candidates.csv
allocation/allocation_strategy_comparison.csv
```

Stop here if costs have inconsistent units, an allocation hits an unintended
cap, the budget is infeasible, or bootstrap activation is unstable. Change the
design under a new case/output name rather than mutating an already executed
campaign.

## 6. Run the independent reference and production union

After approving `required_production_counts`:

```bash
mfmc-mfpod production \
  --config "$MFPOD_CONFIG" \
  --resume --stop-after production
```

This runs:

- the independent DSMC-only reference IDs;
- the DSMC prefix required by any declared comparison;
- the TPMC and Sentman nested prefixes required by any declared comparison.

It does not rerun pilot IDs. The exact disjoint roles are persisted in
`production/roles.json`.

After interruption, issue the same command again with `--resume`.

## 7. Run MFMC estimation, POD and validation

This final step uses the archives and submits no additional solver evaluations:

```bash
mfmc-mfpod production --config "$MFPOD_CONFIG" --resume
```

Confirm completion:

```bash
mfmc-mfpod production-status --config "$MFPOD_CONFIG"
```

The important final outputs are:

```text
estimator/mean_field.npz
estimator/covariance_operator_metadata.json
estimator/control_weights.json
pod/full_field_modes.npz
pod/eigensolver_diagnostics.json
benchmark/benchmark_summary.csv
benchmark/benchmark_metadata.json
```

The benchmark contains the field-aware method, DSMC-only, DSMC+TPMC,
fixed-ratio, scalar-drag and allocation-algorithm comparisons at the same
configured production budget.

## 8. Run the second geometry

Keep Cube and GOCE outputs separate. After Cube is complete:

```bash
export MFPOD_CONFIG=configs/mfpod/goce_tpmc_sentman.yaml
mfmc-mfpod production --config "$MFPOD_CONFIG" --dry-run
```

Then repeat Sections 4 through 7. Never copy Cube archives or role manifests
into the GOCE output directory.
