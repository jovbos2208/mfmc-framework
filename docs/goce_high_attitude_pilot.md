# GOCE high-attitude TPMC--Sentman pilot

This pilot evaluates 80 paired attitude samples with AoA and AoS independently
uniform on `[-15 deg, 15 deg]`. PICLAS TPMC is the target model and Sentman is
the low-fidelity control variate. Both `C_D` and `C_D2` are recorded.

From the repository root, validate and start the restartable run with:

```bash
source .venv/bin/activate
mfmc-campaign validate-config \
  configs/studies/pilot_correlation/GOCE_high_aoa_aos_moment_correlation.yaml
mfmc-campaign run-pilot-correlation \
  configs/studies/pilot_correlation/GOCE_high_aoa_aos_moment_correlation.yaml \
  --resume
```

Results are written to:

```text
campaign_outputs/pilot_correlation/goce_high_aoa_aos_tpmc_sentman_moments/
```

The checkout provides the Prandtl adapter classes, ADBSat runtime, GOCE mesh,
and parameter templates. The cluster-specific `piclas` and `piclas2vtk`
executables must first be installed with `scripts/configure_piclas.sh` as
described in `docs/cluster_deployment.md`.
