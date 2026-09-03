# Cylinder-hex TPMC--Sentman MFMC optimization

This WP6 workflow does not use a geometry response surrogate. The legacy mode
uses a cost budget of 20 measured PICLas-TPMC equivalents; the control-node
mode uses exactly 20 PICLas-TPMC runs per geometry.
PICLas TPMC is the optimization target, Sentman is a nested scalar control
variate, and DSMC is reserved for final validation.
All generated optimization and final-validation PICLas workflows use the
Prandtl runtime module `PICLas_prandtl` with exactly 36 MPI processes.

The estimator treats `drag_area_m2 = C_D * A_ref` as the geometry-invariant
quantity. It estimates the first and second raw moments separately and derives
the drag standard deviation from them. Five common-random-number pairs form an
independent pilot; production samples do not overlap the pilot. A nested
bootstrap propagates pilot-weight, first-moment and second-moment uncertainty.

## Symmetric control-node optimization

The production geometry-optimization mode starts from the cylinder-hex and
moves ten low-dimensional, symmetry-preserving surface controls: two axial
ring locations and independent width/height scales on the nose, shoulder,
tail shoulder, and tail rings. After every move, the transverse coordinates
are rescaled to restore the prescribed volume. The generated surface must be
finite, non-degenerate, watertight, outward oriented, inside the envelope and
ordered in the axial direction. The exact control-node coordinates before and
after deformation are stored in every geometry manifest.

This mode uses `budget_mode=target_run_count`: exactly 20 paired TPMC/Sentman
runs are retained for every geometry, while the remaining cheap Sentman states
estimate the LF expectation. Five-fold cross-fitting estimates the control
weights out of fold, so all 20 TPMC runs contribute to the final mean and
second-moment estimates. Mean and standard-deviation controls still have
independent correlation and bootstrap-improvement gates.

The 20-run constraint is a TPMC budget, not a combined CPU-hour cap. Sentman
cost is recorded separately and can use the full LF sample pool because it is
orders of magnitude cheaper.

Create a server configuration from
`configs/studies/cylinder_hex_control_node_mfmc_optimization.example.json`.
With `initial_bundle: null`, initialization creates an empty paired bundle from
the uncertainty samples in `lf_config`; no old geometry result is required.
The first `refine` batch includes the undeformed cylinder-hex as the statistical
baseline. A restartable run begins with:

```bash
STATE=outputs/cylinder_hex/control_node_mfmc_optimization/state.json

python scripts/run_cylinder_hex_mfmc_optimization.py initialize \
  --config configs/studies/cylinder_hex_control_node_mfmc_optimization.server.json \
  --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py refine --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py prepare --state "$STATE"

# Cheap LF branch; omit --execute for preflight only.
python scripts/run_cylinder_hex_mfmc_optimization.py sentman --state "$STATE" --execute

# PICLas TPMC branch: planning, submission, and collection are separate.
python scripts/run_cylinder_hex_mfmc_optimization.py submit --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py submit --state "$STATE" --execute
python scripts/run_cylinder_hex_mfmc_optimization.py collect --state "$STATE" --execute

python scripts/run_cylinder_hex_mfmc_optimization.py merge --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py analyze --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py status --state "$STATE"
```

If the confidence-aware incumbent improves, the normalized trust radius grows;
otherwise it contracts. Repeat `refine` through `analyze` until the minimum
radius or the configured iteration limit produces a stop decision. Common
random numbers are preserved across geometries. DSMC remains excluded until
`finalize`.

Sentman is not assumed to be useful. A moment control is activated only when
the pilot has `abs(correlation) >= 0.5`, and it is retained only when its nested
bootstrap standard error is lower than the equal-data TPMC-only estimator. The
manifest records `mfmc` or `tpmc_only` independently for mean and standard
deviation. This guard is essential: the completed six-geometry bundle has
full-sample TPMC--Sentman correlations ranging only from about `-0.60` to
`+0.09`, and a five-pair pilot did not produce a verified MFMC gain.

## Stage 1: existing optimization seeds

```bash
python scripts/analyze_cylinder_hex_tpmc_sentman_mfmc.py \
  --bundle outputs/cylinder_hex/wp5_mfmc_paired_bundle_round3_64proc.json \
  --output-dir outputs/cylinder_hex/wp6_mfmc_optimization/iteration_00 \
  --budget-hf-equivalent 20 \
  --pilot-count 5 \
  --bootstrap-repeats 1000
```

Review `geometry_mfmc_metrics.csv` before submitting more solver work. In
particular, inspect both estimator columns, correlations, standard errors and
quality flags.

## Stage 2: derivative-free batch

The selector operates only on the predeclared non-validation design points. It
combines the cheap Sentman robust objective, a local pattern step around the
current best geometry, and geometry-space coverage. It neither fits nor queries
a geometry surrogate.

```bash
python scripts/select_cylinder_hex_mfmc_optimization_batch.py \
  --design-manifest outputs/cylinder_hex/wp5_design/geometry_design_manifest.json \
  --sentman-metrics outputs/cylinder_hex/wp5_lf/lf_robust_metrics.csv \
  --bundle outputs/cylinder_hex/wp5_mfmc_paired_bundle_round3_64proc.json \
  --mfmc-metrics outputs/cylinder_hex/wp6_mfmc_optimization/iteration_00/geometry_mfmc_metrics.csv \
  --output outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/selection.json \
  --count 4 \
  --iteration 1

python scripts/build_cylinder_hex_round2_suite.py \
  --selection outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/selection.json \
  --design-manifest outputs/cylinder_hex/wp5_design/geometry_design_manifest.json \
  --lf-config outputs/cylinder_hex/wp5_lf_config.json \
  --config-output-dir configs/studies/cylinder_hex_wp6_mfmc_iteration_01 \
  --suite-output outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/suite.json \
  --n-dsmc 0 \
  --n-tpmc 20 \
  --round-number 4 \
  --mpi-procs 64
```

Submit and collect only TPMC:

```bash
python scripts/run_cylinder_hex_round2_suite.py submit --execute \
  --suite outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/suite.json \
  --run-root outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/runs \
  --fidelity tpmc

python scripts/run_cylinder_hex_round2_suite.py collect --execute \
  --suite outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/suite.json \
  --run-root outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/runs \
  --fidelity tpmc
```

Merge the new TPMC rows with the existing 256-state Sentman pool and recompute
the equal-budget objectives:

```bash
python scripts/merge_cylinder_hex_mfmc_optimization_results.py \
  --bundle outputs/cylinder_hex/wp5_mfmc_paired_bundle_round3_64proc.json \
  --suite outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/suite.json \
  --run-root outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/runs \
  --sentman-results outputs/cylinder_hex/wp5_lf/lf_results.csv \
  --output outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/paired_bundle.json

python scripts/analyze_cylinder_hex_tpmc_sentman_mfmc.py \
  --bundle outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/paired_bundle.json \
  --output-dir outputs/cylinder_hex/wp6_mfmc_optimization/iteration_01/analysis \
  --budget-hf-equivalent 20 \
  --pilot-count 5 \
  --bootstrap-repeats 1000
```

Repeat small batches until all non-validation design points are exhausted or
the Pareto set and confidence intervals stabilize. Only after that checkpoint
should local continuous refinements be generated around the best discrete
designs. The six predeclared validation geometries remain untouched throughout
optimization.

## Statistical decisions and output schema

The bootstrap arrays used for each objective are stored in
`geometry_mfmc_details.json` under `bootstrap_distributions`. Their pairing
record contains the seed, pilot IDs and production TPMC IDs. Geometry
comparisons use paired bootstrap differences only when those records are
identical; otherwise they use a deterministic independent bootstrap. Each
comparison reports the candidate-minus-incumbent difference, confidence
interval, probability of improvement and confidence-dominance decision.

The analysis directory contains:

- `geometry_mfmc_metrics.csv`: point estimates, standard errors, correlations,
  betas, estimator classes, costs and both Pareto memberships.
- `geometry_mfmc_details.json`: complete allocations, sample IDs, CIs,
  bootstrap arrays and fallback reasons.
- `geometry_mfmc_comparisons.json`: paired/independent comparison audit trail.
- `geometry_pareto.csv`: union of point and confidence-aware Pareto rows.
- `geometry_budget.csv`: the hard budget audit per geometry. Pilot runs are
  included and `allocated_cost_cpu_hours` is the sum of selected measured run
  costs, not a nominal count.
- `geometry_estimator_fallback.csv`: counts for `sentman_both_moments`,
  `sentman_mean_only`, `sentman_second_moment_only` and `tpmc_only`.

The state file has `schema_version`, immutable config, current iteration,
iteration artifacts, action log, stop decision and the `optimization_closed`
guard. Every iteration records its selection, suite, run root, merged bundle,
analysis artifacts, confidence Pareto set and incumbent uncertainty. Writes use
an atomic temporary-file replacement.

## Restartable driver

Copy the example configuration and change only server paths:

```bash
cp configs/studies/cylinder_hex_mfmc_optimization.example.json \
  configs/studies/cylinder_hex_mfmc_optimization.server.json

python scripts/run_cylinder_hex_mfmc_optimization.py initialize \
  --config configs/studies/cylinder_hex_mfmc_optimization.server.json \
  --state outputs/cylinder_hex/wp6_mfmc_optimization/state.json
```

The state-driven loop is:

```bash
STATE=outputs/cylinder_hex/wp6_mfmc_optimization/state.json
python scripts/run_cylinder_hex_mfmc_optimization.py analyze --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py status --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py select --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py prepare --state "$STATE"

# Planning only: creates/updates batch state but submits nothing.
python scripts/run_cylinder_hex_mfmc_optimization.py submit --state "$STATE"

# Explicit external action. Repetition is safe: submitted/collected states skip.
python scripts/run_cylinder_hex_mfmc_optimization.py submit --state "$STATE" --execute
python scripts/run_cylinder_hex_mfmc_optimization.py collect --state "$STATE" --execute
python scripts/run_cylinder_hex_mfmc_optimization.py merge --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py analyze --state "$STATE"
```

`prepare` always calls the existing Round-2 suite builder with `n_dsmc=0` and
`mpi_procs=36`, and forces the `PICLas_prandtl` simulator module. `submit` and
`collect` delegate to the existing Round-2 runner; there is no second
submission implementation. Inputs, paths, seeds, sample counts and thresholds
remain in the config embedded in the state manifest.

Stop decisions are one of `budget_exhausted`, `design_space_exhausted`,
`no_significant_improvement`, `pareto_set_stable`,
`objective_uncertainty_target_met` or `continue_optimization`. Defaults require
two stable iterations, 95% confidence and a 0.95 improvement probability.
A single unsuccessful iteration therefore cannot stop the workflow.

## Local refinement

After a discrete stop decision:

```bash
python scripts/run_cylinder_hex_mfmc_optimization.py refine --state "$STATE"
python scripts/run_cylinder_hex_mfmc_optimization.py prepare --state "$STATE"
```

The refinement is a bounded local pattern search around the minimum-mean
design, robust optimum and Pareto knee. It fits no surrogate. Duplicates and
invalid surfaces are rejected, generated surfaces are validated, and the
normalised/physical parameters plus reusable geometry manifests are persisted.
The subsequent `prepare` step performs the normal Gmsh/HDF5 build and HDF5 mesh
validation through the existing suite builder. The local batch has the same
20-HF, TPMC/Sentman and Prandtl 36-MPI contract and contains no DSMC
configuration.

## Final DSMC validation

`finalize` is rejected unless the state contains a stop decision. It closes the
optimization, selects baseline/minimum-mean/robust/Pareto-knee and optionally a
Pareto edge, and creates a Prandtl 36-MPI suite with identical TPMC/DSMC CRN
states:

```bash
python scripts/run_cylinder_hex_mfmc_optimization.py finalize --state "$STATE"

python scripts/run_cylinder_hex_round2_suite.py submit --execute \
  --suite outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/suite.json \
  --run-root outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/runs \
  --fidelity all

python scripts/run_cylinder_hex_round2_suite.py collect --execute \
  --suite outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/suite.json \
  --run-root outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/runs \
  --fidelity all

python scripts/merge_cylinder_hex_round2_results.py \
  --bundle outputs/cylinder_hex/wp6_mfmc_optimization/iteration_XX/paired_bundle.json \
  --suite outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/suite.json \
  --run-root outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/runs \
  --output outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/bundle.json
```

Replace `iteration_XX` with the final analyzed iteration recorded by `status`.
Then build the validation-only report:

```bash
python scripts/report_cylinder_hex_dsmc_validation.py \
  --bundle outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/bundle.json \
  --finalists outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/finalists.json \
  --output-dir outputs/cylinder_hex/wp6_mfmc_optimization/final_validation/report
```

The JSON/CSV report contains paired DSMC-minus-TPMC differences and intervals
for mean and standard deviation, both rankings, rank stability, and an explicit
`tpmc_optimum_confirmed` or `tpmc_optimum_not_confirmed` status. It records
`optimization_updated_from_dsmc=false`; DSMC results never feed selection,
tuning, local refinement or stopping.

## First server checkpoint: analyze all 12 geometries, submit nothing

From the server repository root, after activating the production environment:

```bash
test "$(git branch --show-current)" = agent/fix-allocation-control-activation
python -m py_compile \
  mfmc_campaign/geometry_mfmc_optimization.py \
  mfmc_campaign/geometry_local_validation.py \
  mfmc_campaign/geometry_optimization_workflow.py \
  scripts/run_cylinder_hex_mfmc_optimization.py

python scripts/analyze_cylinder_hex_tpmc_sentman_mfmc.py \
  --bundle outputs/cylinder_hex/wp5_mfmc_paired_bundle_round3_64proc.json \
  --output-dir outputs/cylinder_hex/wp6_mfmc_optimization/iteration_00 \
  --budget-hf-equivalent 20 \
  --pilot-count 5 \
  --bootstrap-repeats 2000 \
  --bootstrap-seed 20260822 \
  --minimum-abs-control-correlation 0.5 \
  --confidence-level 0.95 \
  --improvement-probability-threshold 0.95
```

Confirm `n_geometries == 12`, inspect the budget/fallback/Pareto tables, and
archive the analysis manifest before running `select`. Do not invoke `submit`
at this checkpoint. The expected negative evidence is legitimate: if no
Sentman control lowers nested-bootstrap SE, all affected objectives must remain
TPMC-only.

## Server synchronization

Local push after the documented commits:

```bash
git push -u origin agent/fix-allocation-control-activation
```

Server update when that branch is already checked out:

```bash
git fetch origin
git switch agent/fix-allocation-control-activation
git pull --ff-only origin agent/fix-allocation-control-activation
```
