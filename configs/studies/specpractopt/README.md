# SpecPractOpt cylinder: TPMC/ADBSat MFMC angle grid

Run both stages from the Framework repository root. The pilot is evaluated once
at `AoS=AoA=0 deg`. Its paired TPMC/ADBSat samples and measured costs are then
reused for all 37 deterministic AoS cells. AoA remains fixed at `0 deg`; AoS
runs from `-90 deg` through `+90 deg` in increments of `5 deg`.

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

The angle-grid estimator assumes that the paired-pilot HF/LF correlation and
control-variate coefficient obtained at zero attitude remain representative
throughout the full `[-90, 90] deg` AoS sweep at fixed `AoA=0 deg`.
