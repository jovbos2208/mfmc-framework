# Paper 2 research and implementation plan

## Research questions and boundary to Paper 1

Paper 2 asks: (1) What are E[Cd] and Var(Cd) for explicitly defined GOCE and CHAMP state populations? (2) Can paired lower-fidelity surface fields reduce the cost of estimating the HF mean field and HF covariance eigenspace? (3) At equal total cost, including mapping and offline work, how does the method compare with DSMC-only estimation?

Paper 1 validates solver coupling and MFMC behavior on controlled shapes. Paper 2 applies the validated conventions to mission geometries and adds field statistics. It must not inherit the 100/200/300/400 km matrix automatically.

## Statistical population decision

Recommended minimum study: one fixed mission state per spacecraft with uncertain environment, attitude, and surface interaction inputs. This makes randomness an input distribution conditional on the state and avoids mixing trajectory time variation with uncertainty. A later trajectory population is a separate estimand and must specify the time/state measure and whether conditional input uncertainty is nested within it.

GOCE already has a draft fixed state at 244.14592193 km. CHAMP state selection remains open. Production is blocked until original geometry scale, Aref, state provenance, and the intended population are fixed.

Cd is defined sample-wise as Fd divided by q-infinity times Aref in a documented body frame. Estimate E[Cd] and E[Cd squared] as separate MFMC targets and reconstruct Var(Cd). This is not E[Fd] divided by E[q-infinity] times Aref. Confidence intervals must retain covariance between the two moment estimators. Report separately (a) variance from physical input uncertainty and (b) any included conditional DSMC particle noise; use repeated seeds at fixed physical inputs to estimate the latter.

## Selected field method

Primary field: the complete surface force-density vector in body-fixed coordinates and physical units. Pressure is optional supplementary output. Map every HF and LF field to a single declared reference surface using a conservative face-to-face operator that preserves integrated force and moment within tolerance. Store mapping fingerprints, source/target areas, normals, and reference point.

Use the area-weighted inner product. For target-face areas Ai and three-component fields u and v, inner product(u,v) = sum_i Ai times dot(u_i,v_i). Estimate the HF mean with a multifidelity control-variate estimator from paired HF/LF fields plus additional LF fields. Estimate the centered HF covariance operator directly with multifidelity control variates and paired cross-statistics. Apply the operator matrix-free; do not assemble a dense 3N by 3N matrix. Symmetrize the estimated operator, diagnose negative Rayleigh quotients, and use a documented positive-semidefinite correction only when necessary, with correction magnitude reported.

Compute POD modes as eigenvectors of this estimated HF covariance operator. This targets the HF distribution. It is distinct from pooling HF and LF snapshots, which targets a mixture distribution, and from projecting HF fields onto an LF-only basis, which is a reconstruction strategy. Those two alternatives may be baselines but must not be called unbiased MFMC-POD of the HF distribution.

## Allocation

The global-drag allocation is not automatically valid for field covariance. Run a field pilot with disjoint roles: paired fields for cost/cross-covariance estimation, additional LF fields, and independent HF validation fields. Select counts by a weighted objective covering mean-field error and covariance-operator error, with bootstrap-robust constraints and minimum HF counts. Include solver CPU hours, field export, conservative mapping, storage, eigensolver, and repeated-validation costs.

## Minimal defensible matrix

- Spacecraft: GOCE and CHAMP.
- Population: one fixed, provenance-backed mission state per spacecraft.
- Models: DSMC target; TPMC primary control; add Sentman only after conservative field mapping and full vector traction are available.
- Data roles per spacecraft: at least 30 paired pilot fields, 50 independent HF validation fields, and allocation-selected production counts. These are starting minima, not power guarantees.
- Replication: repeated allocations/campaign resampling or bootstrap intervals; repeated DSMC seeds at a subset of fixed inputs for noise separation.
- Baselines: DSMC-only at identical realized cost; pooled-snapshot POD; LF-basis reconstruction.

## Metrics

Report mean-field weighted relative error; explained variance and eigenvalue spectra; area-weighted principal/subspace angles; independent-HF reconstruction error; errors in reconstructed integrated force and moment; covariance probe errors; and total realized cost. Individual mode signs are arbitrary. For clustered eigenvalues compare invariant subspaces, not one-to-one modes.

## Implementation order

1. Resolve CHAMP scale/Aref and select both mission states/populations.
2. Freeze coordinate, sign, Aref, reference point, units, and sample-fingerprint contracts.
3. Repair GOCE archive metadata and generate missing CHAMP full-field exports without changing legacy global-Cd data.
4. Verify conservative mapping by constant fields and integrated forces/moments.
5. Quantify DSMC conditional noise with repeated seeds.
6. Run a field-specific pilot and estimate robust allocation inputs.
7. Implement matrix-free multifidelity mean/covariance operators with symmetry/PSD diagnostics.
8. Compute modes and independent validation metrics.
9. Run equal-cost DSMC-only and methodological baselines.
10. Freeze analysis artifacts and write only claims supported by independent validation.

## Proposed paper structure

1. Introduction and questions. 2. Mission states and statistical populations. 3. Solvers, common inputs, Cd, and surface-field contract. 4. Multifidelity moment estimators. 5. Conservative mapping and area-weighted HF-covariance POD. 6. Pilot and allocation. 7. GOCE and CHAMP results. 8. Equal-budget validation. 9. Noise, limitations, and transfer from Paper 1. 10. Conclusions.

## Planned outputs

Figures: spacecraft/reference meshes; population schematic; field mapping conservation; cost/correlation pilot; mean force-density fields; eigenvalue spectra; leading modes; subspace angles; reconstruction/error versus cost; DSMC-noise decomposition. Tables: state/population definitions; geometry scale/Aref; archive contract; costs/counts; Cd moments; mapping conservation; POD/force/moment validation; risk register.

## Risks and open decisions

- CHAMP scale and Aref conflict is unresolved.
- GOCE archives lack required coordinate-frame/component metadata and preferred cost/normal/reference-point metadata.
- CHAMP has global Cd samples but no verified surface fields.
- Sentman may not provide a semantically comparable full traction vector without additional export work.
- GSI parameters with identical numbers can have different definitions across solvers.
- Field-optimal allocations can differ from drag-optimal allocations.
- Finite-sample covariance estimates can be indefinite; corrections can bias spectra.
- Particle noise can dominate high modes and falsely appear as physical variability.
