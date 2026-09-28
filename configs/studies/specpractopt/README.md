# SpecPractOpt cylinder: TPMC/Sentman MFMC angle grid

Run both stages from the Framework repository root. The pilot is evaluated once
at `AoS=45 deg`, `AoA=0 deg`. Its paired TPMC/Sentman samples and measured costs are then
reused for all 37 deterministic AoS cells. AoA remains fixed at `0 deg`; AoS
runs from `-90 deg` through `+90 deg` in increments of `5 deg`.

Only two GSI inputs are uncertain. Energy accommodation follows a bounded
normal distribution with mean `0.5`, standard deviation `0.1`, and bounds
`[0, 1]`. Wall temperature follows a bounded normal distribution with mean
`300 K`, standard deviation `25 K`, and bounds `[250, 400] K`.

The joint solver batch estimates `C_D`, `C_D2`, lateral `C_L`, and `C_L2`.
`Var_C_D` and `Var_C_L` are derived as second moment minus squared mean. For
this study only, the reported `C_L` is explicitly mapped to the solver's
side-force coefficient `C_Y`; the vertical lift coefficient is not requested.

```bash
cd public_repo/Framework

python3 -m mfmc_campaign.cli run-pilots \
  configs/studies/specpractopt/cylinder_hex_gsi_pilot.yaml

python3 -m mfmc_campaign.cli run \
  configs/studies/specpractopt/cylinder_hex_gsi_angle_grid_mfmc.yaml
```

The production config computes each cell's total budget as 20 times the mean
TPMC cost recorded in the pilot `model_evaluations.csv`. Run the pilot first;
the literal budget value in the production YAML is only a validation fallback
for the period before that CSV exists.

The angle-grid estimator assumes that the paired-pilot HF/LF correlations and
control-variate coefficients obtained at `AoS=45 deg`, `AoA=0 deg` remain
representative throughout the full `[-90, 90] deg` AoS sweep at fixed
`AoA=0 deg`.

## Production-derived MFMC weights

`cylinder_hex_maxwell_angle_grid_mfmc.yaml` uses
`estimator.weight_source: production` with `weight_sample_count: 32`. For every
angle and QoI, the first 32 fingerprint-coupled `prod_hf`/`prod_lf_pair`
evaluations determine the correlation and control-variate coefficient. Those
same evaluations remain in the complete production-pair mean used by the final
MFMC estimator; they are not discarded as a separate pilot. `prod_lf_full`
still supplies the enlarged ADBSat mean. External pilot statistics may still
drive the pre-run budget allocation, but they cannot override the final
production-derived coefficient in this mode. Omitting the `estimator` block
retains the legacy `pilot` behavior.

For the GSI angle grid, `reuse_fallback_pilot: true` additionally makes a missing
external pilot a reusable production prefix: the configured 50 coupled HF/LF
pairs are evaluated once, used for allocation and local weights, and retained
as production samples. Only samples beyond that prefix are submitted as new
production evaluations. Result flags record `reused_fallback_pilot=50`.
