# VLEO Multifidelity GSA and Robust Geometry Optimization

Status: WP0--WP4 complete; WP5 pilot surrogate and first sequential acquisition implemented

## Paper Through-Line

The two earlier plans form one coherent methods paper if sensitivity analysis is
used to construct and explain the robust design rather than presented as a
separate application:

1. learn a DSMC-target multifidelity surrogate over uncertainty and geometry;
2. quantify variable- and source-level sensitivities with uncertainty intervals;
3. optimize expected drag, dispersion, and upper-tail drag with that surrogate;
4. validate selected designs with fresh paired HF/LF samples; and
5. explain the deterministic/robust design difference using the GSA results.

The geometry-comparison objective must be dimensional drag force, `C_D A_ref`,
or ballistic coefficient under one documented reference-area convention. Raw
`C_D` is not a geometry-invariant optimization objective when projected or
reference area changes between designs.

## Implementation Sequence

### WP0 — Reproducible sample/output data contract (complete)

- Centralize stable sample and request fingerprints.
- Populate fingerprints in the normal campaign's `model_evaluations.csv`.
- Emit `sample_inputs.csv` for normal and pilot-correlation evaluations.
- Preserve numeric uncertainty inputs and numeric geometry metadata.
- Add `export-surrogate-dataset`, with an exact, disk-backed provenance join.
- Reject blank fingerprints, conflicting duplicate inputs, and incomplete joins
  by default; allow an explicit exploratory `--allow-incomplete` mode.

Implemented modules and tests:

- `mfmc_campaign/fingerprints.py`
- `mfmc_campaign/surrogate_dataset.py`
- campaign and pilot-correlation output integration
- toy end-to-end and strict-join regression tests

### WP1 — Existing-output inventory and auditable backfill (static cases complete)

- Use the first-paper archive at `campaign_outputs/vleo_mfmc_paper/` as the
  primary legacy dataset. It contains ten case directories, config snapshots,
  evaluation caches, and model-evaluation tables.
- Import its existing `uncertainty_sensitivity_analysis/reconstruction_audit.csv`
  rather than repeating the already validated replay audit. At inventory time it
  records 310 exact production request cache-hash matches, zero mismatches, and
  80 trajectory request sequences whose hashes could not be checked locally
  because HWM14 was unavailable.
- Continue excluding externally reused legacy pilot values unless their sampled
  inputs can be independently and exactly audited.
- Inventory each Cube/GOCE/SOAR/CHAMP output directory by case, model, QoI,
  phase, sample count, fingerprint coverage, and failed/non-finite fraction.
- Add a dry-run-first backfill command for runs whose inputs can be reconstructed
  exactly from `config_snapshot.json`, deterministic seed derivation, and sample
  IDs.
- Verify reconstructed request/sample fingerprints against any fingerprints
  already present. Never silently attach reconstructed inputs after a mismatch.
- Mark irrecoverable cases for selective rerun rather than rerunning the whole
  matrix.

Implemented status:

- `inventory-legacy-surrogate-data` inventories the archive and imports its
  existing reconstruction audit without loading all multi-GB CSVs into memory.
- `backfill-legacy-surrogate-data` replays the original sampling/RNG path,
  independently rechecks every admitted request against `evaluation_cache.json`,
  writes a matched production subset, guarantees that every selected HF index
  is retained by every corresponding LF model, and runs the strict surrogate
  join.
- The original archive is read-only; generated artifacts are written to a
  separate output root.
- Completed with a deterministic cap of 2500 samples per request for eight
  static cases: 710,624 model-evaluation rows and 710,624 surrogate rows, all
  finite and exactly joined.
- `champ_20080128_d3` and `goce_20131021_d3` remain quarantined until their
  HWM14/environment state can be hash-audited.

Acceptance: every row admitted to surrogate fitting has a numeric input vector,
non-empty fingerprints, and an audit status of native or verified-reconstructed.

### WP2 — Sparse multifidelity PCE core (implemented for all static cases)

- Implement distribution-aware transforms from physical variables to canonical
  independent coordinates.
- Generate a hyperbolically truncated polynomial basis with configurable total
  degree, interaction order, and sparsity norm. Do not use an unrestricted cubic
  tensor basis for the full VLEO input dimension.
- Fit LF, HF, and paired residual models:
  `HF(x) = LF_PCE(x) + delta_PCE(x)`.
- Select regularization by grouped cross-validation. Geometry-held-out validation
  is mandatory once geometry parameters enter the model.
- Select the LF correction model using paired validation performance and
  residual error, not correlation alone.
- Record basis, transforms, coefficients, split IDs, seeds, errors, and flags in
  machine-readable artifacts.

Acceptance: synthetic polynomials recover dominant terms; MF correction matches
or improves HF-only held-out error without validation leakage.

Implemented status:

- Hyperbolically truncated sparse polynomial basis with interaction limits.
- Standardized LassoCV fitting and serialized transforms/models.
- Repetition-held-out outer validation for LF-only, HF-only, and MF residual
  correction using exact sample fingerprints.
- `fit-surrogate-pce` CLI and coefficient/metric/model artifacts.
- Synthetic polynomial and multifidelity recovery tests pass.
- All eight auditable static cases have been fitted. PICLas-TPMC is selected in
  all eight. Four cases have at least 30 HF samples and are eligible for primary
  claims; their MF residual out-of-fold `R2` ranges from 0.902 to 0.998. The
  remaining four (10--27 HF samples) are retained as explicitly exploratory.
  For `soar_300km`, `R2 = 0.9978` and `RMSE = 0.0138` for `C_D`.

### WP3 — Surrogate GSA with uncertainty intervals (implemented for static cases)

- Estimate first-order and total-effect Sobol indices by Monte Carlo evaluation
  of the fitted surrogate, so clipped/nonstandard input distributions remain
  supported.
- Aggregate variable indices into the configured atmosphere, attitude, and GSI
  source blocks.
- Bootstrap the complete fit-and-estimate process, not only the final index
  calculation.
- Add `run-surrogate-gsa` and write metrics, coefficients, variable/source Sobol
  tables, intervals, and diagnostic plots.

Acceptance: known additive and interaction benchmarks reproduce expected
rankings and intervals; weak fits are flagged and excluded from paper claims.

Implemented status:

- `run-surrogate-gsa` draws independent pick-freeze matrices from the archived
  clipped input distributions and evaluates the selected MF-PCE.
- Centered Saltelli first-order and Jansen total-effect estimators are computed
  directly for variables and jointly replaced source blocks.
- Conditional Monte Carlo bootstrap intervals, deterministic seeds, quality
  flags, and the planned Sobol CSV outputs are written.
- A repetition-level block bootstrap now resamples archived HF blocks, refits
  both the selected LF PCE and paired discrepancy PCE, resamples the Sobol
  integration rows, and reports refit-aware 95% intervals. All eight runs used
  100 requested refits and completed 100/100 without failure. LF model selection
  is held fixed and is recorded as the remaining conditioning assumption.
- `aggregate-surrogate-gsa` writes cross-case case/source tables and separates
  four primary cases (`HF >= 30`, out-of-fold `R2 >= 0.7`) from four HF-poor
  exploratory cases.
- `soar_300km` point result ranks energy accommodation first (`ST = 0.8297`),
  followed by composition (`ST = 0.0848`) and density (`ST = 0.0546`).
- Outputs are under `outputs/vleo_wp1_backfill/*/surrogate_pce/`; the joint
  tables are under `outputs/vleo_wp1_backfill/gsa_cross_case/`.

### WP4 — Parameterized 3D cylinder-hex geometry family

- Implement the four-variable generator: nose/taper fraction, aft/tail fraction,
  cross-section width/height ratio, and chamfer/rounding fraction.
- Enforce fixed payload volume and maximum envelope explicitly.
- Produce canonical solver assets and geometry manifests containing parameters,
  mesh fingerprint, volume, area conventions, and validity checks.
- Validate PICLas and ADBSat ingestion on baseline, center, and boundary designs.

Acceptance: deterministic generation, watertight/non-self-intersecting geometry,
volume/envelope tolerances, and consistent reference areas across solvers.

Implemented status:

- The geometry is generated analytically from scratch; no legacy `.msh` or
  external-project geometry is imported.
- The four-variable fixed-volume loft, hard envelope checks, deterministic
  OBJ/STL/MAT/NPZ assets, fingerprints, manifest, and CLI are implemented.
- Watertightness, outward orientation, convexity/no-self-intersection, volume,
  envelope, boundary-design, and byte-reproducibility tests are implemented.
- A fresh analytical Gmsh exterior-flow domain with named inlet, outlet, and
  spacecraft boundaries is implemented. Its baseline has 8,361 positive-volume
  tetrahedra, the exact expected gas volume, and deterministic mesh fingerprint.
- HOPR/PICLas HDF5 conversion and actual PICLas ingestion are complete for the
  uniformly scaled baseline. The corrected smoke case produced
  `C_D = 2.37246` instead of the earlier zero-force result.
- A separate uniformly scaled `0.1` baseline is generated without overwriting
  the unit-scale reference. Length, area, and volume follow factors `0.1`,
  `0.01`, and `0.001`; its preliminary exterior mesh has 181 tetrahedra versus
  8,361. Density is multiplied by ten so the Knudsen number is preserved.
- Five-seed L0/L1/L2 runs contain 724, 1,648, and 5,448 hexahedra. Mean drag
  coefficients are 2.37019, 2.37058, and 2.37160; the total L0-to-L2 change is
  0.0599% and all confidence intervals overlap. L1 is the production level and
  L2 is retained for validation. Uniformly low positive scaled Jacobians remain
  a reported mesh-quality limitation.

### WP5 — Design-space campaign and sequential HF acquisition

- Generate the initial LF design over geometry and uncertainty.
- Choose initial HF geometries by space filling plus LF Pareto coverage.
- Add HF samples sequentially using predicted robust-objective improvement and
  surrogate uncertainty/disagreement diagnostics.
- Maintain a permanently untouched geometry-and-uncertainty validation set.

Acceptance: learning curves report held-out error and robust-metric stability as
HF cost increases.

Implemented status:

- A deterministic maximin Latin-hypercube design spans the four geometry
  variables and includes the validated baseline explicitly.
- Six of the default 32 geometries are fixed as a permanently untouched
  validation set before any LF or HF outcome is inspected; the training
  workflow does not evaluate them.
- Each design writes validated solver assets plus a portable ADBSat runtime
  bundle. ADBSat now resolves safe generated geometry IDs directly rather than
  silently falling back to a historical geometry.
- The restartable LF workflow deterministically thins the archived WP1
  uncertainty samples and reuses the identical draws for every geometry.
- LF robust metrics use `C_D * A_ref`, not raw coefficients. Initial HF
  selection combines LF Pareto coverage, geometry maximin coverage, and the
  baseline while excluding validation geometries.
- A reference-area consistency guard now requires both solvers to use the fixed
  canonical manifest area; legacy ADBSat projected-area normalization is
  rejected for WP5 metrics.
- The initial-HF suite builder creates L1 meshes only for selected designs and
  selects five common, space-filling WP1 uncertainty states per geometry (30
  initial DSMC runs). A restartable suite driver submits and collects them.
- A WP5 DSMC-target geometry/uncertainty surrogate workflow now consumes the
  auditable paired bundle, fits TPMC and DSMC-minus-TPMC sparse PCEs, and writes
  leave-one-geometry-out diagnostics. It also produces corrected 256-state
  robust metrics and a deterministic nested TPMC-state plan for the next HF
  acquisition. The pilot remains explicitly quality-flagged until those new HF
  samples are collected.
- HF learning curves and acquisition of entirely new geometry locations remain
  to be implemented.

### WP6 — Robust optimization and paper validation

- Optimize `E[D]`, `std(D)`, `q95(D)`, and a documented scalar robust objective;
  report the Pareto front rather than only one weight choice.
- Compare the original cylinder-hex, nominal optimum, robust optimum, and at
  least three Pareto designs.
- Validate all reported designs with fresh paired DSMC/LF uncertainty samples.
- Separate physical drag variability, surrogate error, and finite-validation
  estimator uncertainty.
- Use source-level GSA changes to explain why the robust optimum differs.

Acceptance: claims are based on fresh HF confidence intervals; LF-only and
nominal-only failure modes are reported even if the robust optimum is not
statistically superior.

## Immediate Next Implementation Milestone

Generate the WP5 geometry design, run the archived-WP1 common-random-number LF
campaign, and select the initial HF geometries. Then construct L1 PICLas meshes
only for those selected geometries and reserve L2 checks for the final reported
designs. The two trajectory cases remain quarantined and are not prerequisites.
