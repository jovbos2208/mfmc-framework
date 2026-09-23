# Paper 1 revision plan

## Scope and evidence boundary

Paper 1 becomes a controlled MFMC benchmark over Cube, sphere, and a finite-thickness open hemispherical shell at 100, 200, 300, and 400 km. SOAR, GOCE, CHAMP, and spatial POD move out of this paper. The repository contains no manuscript source, so this is a section-level change specification rather than a line-by-line edit. No solver result or convergence claim is made during preparation.

The direct QoIs remain Cd and Cd squared. Report E[Cd], E[Cd squared], and Var(Cd) = E[Cd squared] - E[Cd] squared. Every solver receives the same sample identity and physical inputs; solver-specific GSI parameter mappings require an explicit semantic table before production.

## Section changes

| Section | Retain | Remove or move | New evidence required before claims |
|---|---|---|---|
| Abstract | MFMC motivation and estimator definition | Mission-specific GOCE/CHAMP results and POD to Paper 2 | Quantified accuracy/cost results from the complete 12-case matrix |
| Introduction | Rarefied-flow UQ motivation | Broad spacecraft application claims not tested here | Explain the three controlled geometric challenges and limited benchmark scope |
| Contributions | Common-random-input MFMC framework | Production or generality claims unsupported by the matrix | Separate aerodynamic verification, numerical convergence, and MFMC efficiency contributions |
| Geometries | Existing Cube provenance | SOAR and mission spacecraft | Sphere D=0.1 m; shell Ro=0.05 m, Ri=0.048 m; reference areas, frames, regions, cup orientation |
| Input laws | Existing distributions that match the validated campaign family | Silent distribution changes | One table for common atmosphere, winds, attitude, GSI variables; explain removal of pure attitude variables for sphere |
| Numerical methods | PICLas DSMC/TPMC and ADBSat adapters | POD methods | Mesh/domain/time-step/particle/sampling settings and solver-specific GSI semantics |
| Verification | Existing equations and diagnostics | Mixed verification and efficiency narrative | Analytic/free-molecular sphere reference with sourced value; shell forward/backward and angle symmetry checks |
| Results | MFMC estimator definitions | GOCE/CHAMP and spatial fields | Separate tables for Cd moments, correlations, costs, allocations, errors, and uncertainty intervals |
| Figures | Method flow if still accurate | Mission geometry and POD figures | Mesh views with normals, Cd verification, convergence plots, correlation/allocation plots, equal-budget error plots |
| Discussion | Bias/variance and particle-noise caveats | Mission conclusions | Contrast planar attitude sensitivity, convex symmetry, and concave shadowing/multiple reflection |
| Conclusions | Evidence-bounded MFMC summary | Efficiency promises | State only observed regimes and geometries; list transfer limits to Paper 2 |

## Ordered computation plan

1. Geometry gate: accept only meshes whose manifest passes topology, normals, volume, physical-boundary, and nonnegative-Jacobian checks. Stop and remesh any failed level.
2. Solver-ingestion gate: one deterministic zero-angle smoke case per geometry and solver branch. Continue only if all branches use the declared mesh, Aref, Lref, frame, and identical state.
3. Aerodynamic verification: sphere in collisionless fully diffuse/full-accommodation state; establish a sourced reference and tolerance before judging agreement. Run shell cup-forward, cup-backward, and small signed-angle sweeps. Stop if symmetry, force sign, or cavity accessibility fails.
4. Numerical convergence: independently vary surface mesh, volume mesh, domain, time step, particle number, and sampling length. Do not combine changes in the primary convergence table. Continue when the chosen observable change and DSMC sampling interval meet predeclared tolerances.
5. Pilot gate: paired common samples for the DSMC target and the TPMC and Sentman controls in every geometry-altitude cell. Inspect finite values, sample fingerprints, Pearson/Spearman relationships, outliers, and measured CPU hours. Exclude a control variate in cells where robust variance reduction is not supported.
6. Production: freeze configurations and fingerprints, then allocate under equal budgets. Preserve independent repetitions or bootstrap intervals.
7. Analysis: report Cd and Cd-squared estimates separately, reconstructed variance with uncertainty, DSMC-only equal-budget baselines, and realized rather than nominal costs.

## Stop and continuation criteria

- Stop on inconsistent Aref, flow direction, units, sample IDs, nonfinite QoIs, negative cells, inaccessible cavity, or unmapped GSI definitions.
- Remesh rather than claim convergence if changes are dominated by mesh/domain effects.
- Increase DSMC sampling before MFMC allocation if conditional particle noise masks physical input variation.
- Continue to production only after the pilot demonstrates stable correlations and measured costs for the specific cell.
- A negative or unstable reconstructed variance triggers joint moment uncertainty analysis; it is not clipped without disclosure.

## Required tables and figures

Tables: geometry/normalization; baseline states and provisional Knudsen proxies; numerical settings; verification errors; per-cell costs/correlations; allocations; Cd moments and uncertainty; equal-budget error. Figures: three geometries and surface regions; sphere verification; shell orientation/sweep; convergence panels; correlations by altitude; allocation and equal-budget error. No aerodynamic convergence or MFMC superiority statement is permitted until the corresponding result exists.
