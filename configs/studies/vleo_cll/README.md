# VLEO CLL production staging

These local campaign configurations prepare the GSI-only geometry study. They
contain no private PICLas source. The private build remains outside Framework;
only its binaries are installed into the ignored Framework runtime directory
with `scripts/configure_piclas.sh`. Gate G5 must pass before production submit.

## Locked design intent

- Primary robustness endpoint: GSI-induced standard deviation of
  `drag_area_m2 = C_D * reference_area_m2`.
- Coefficient normalization: do not inject a configured projected area. PICLas
  and ADBSat both use `reference_area_m2 = A_total / 2`, computed from the
  surface used by the respective solver. The resulting dimensional drag area
  remains `C_D * (A_total / 2)`.
- Smallest relevant robustness improvement: 20% lower standard deviation than
  the baseline (`std_ratio <= 0.80`). This is a design SESOI, not a pilot estimate.
- Mean-performance guardrail: no more than 2% larger mean drag area than the
  baseline (`mean_ratio <= 1.02`). The absolute allowance is computed after the
  baseline pilot as `0.02 * baseline_mean_drag_area_m2`.
- Primary CLL design domain: `cll_alpha_n, cll_sigma_t in [0,1]`.
- Fixed state: pure O, `n=1e15 m^-3`, `T_inf=961 K`, `V=7718.371 m/s`,
  `T_w=300 K`, AoA=AoS=0 deg.
- All geometries in a stage use the same canonical samples and random seeds.

The 20-point pilot is ready to validate and dry-run. It pairs PICLas-TPMC-CLL
with ADBSat-CLL and measures both correlation and cost. It is exploratory and
must never be reused as a confirmatory holdout.

The pilot uses the scaled parametric Cylinder-Hex optimization baseline shown
schematically in Figure 1 of the manuscript: `L=0.1 m`, the converged L1 PICLas
mesh `geometry/cylinder_hex_convergence/L1/cylinder_hex_scale_0p1_l1_mesh.h5`,
and the matching ADBSat surface model `cylinder_hex_scale_0p1`. It must not be
run with the legacy `SpecPractOpt_cylinder_hex_mesh.h5` asset. Before execution,
install the matching `cylinder_hex_scale_0p1.obj` and `.mat` files as
`ADBSat-PyVersion/inou/obj_files/cylinder_hex_scale_0p1.obj` and
`ADBSat-PyVersion/inou/models/cylinder_hex_scale_0p1.mat`, respectively.
Before every campaign, verify that the PICLas object-boundary surface and the
ADBSat panel surface have matching `A_total` within the documented mesh
tolerance; otherwise equal normalization conventions would still produce
different numerical reference areas.

## Execution order

1. Validate: `python3 -m mfmc_campaign.cli validate-config configs/studies/vleo_cll/cylinder_hex_cll_pilot_20.yaml`
2. Install the private binaries with
   `PICLAS_BIN=/private/build/bin/piclas PICLAS2VTK_BIN=/private/build/bin/piclas2vtk scripts/configure_piclas.sh`.
3. Dry-run/preflight the generated PICLas requests before Slurm submission.
4. Run the 20-point pilot and compute paired geometry-difference variance and
   model costs.
5. Lock Search (`20`, `40`, or `80`), holdout, and DSMC sizes from the pilot.
6. Generate disjoint sample manifests and freeze their hashes before execution.

No holdout or DSMC production config is intentionally activated yet: assigning
their sample counts before the paired pilot would create an unjustified design.

