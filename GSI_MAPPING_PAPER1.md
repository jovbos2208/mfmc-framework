# Paper 1 GSI mapping

## Selected hierarchy

- High fidelity: PICLas DSMC.
- Low fidelity 1: PICLas TPMC.
- Low fidelity 2: ADBSat Sentman.

PICLas-Maxwell is not part of the active Paper 1 hierarchy.

## Shared uncertain inputs

Each canonical sample contains one translational/energy accommodation variable alpha and one wall temperature Tw. Atmosphere, species composition, relative velocity, winds, attitude, alpha, Tw, reference area, and sample identifier are shared before solver-specific translation.

| Canonical quantity | PICLas DSMC and TPMC | ADBSat Sentman |
|---|---|---|
| alpha | Part-Boundary3-TransACC = alpha | alpha = alpha |
| momentum accommodation | Part-Boundary3-MomentumACC = alpha squared unless introduced as an independent uncertainty | not an independent Sentman input |
| wall temperature | Part-Boundary3-WallTemp = Tw through the existing updater | Tw = Tw |
| rotational/vibrational accommodation | fixed at 1 unless a later study explicitly varies them | not represented separately |

This is a common physical accommodation axis, not a claim that both wall-scattering models are mathematically identical. Sentman is an analytical diffuse free-molecular panel model. PICLas applies its configured particle-wall reflection law. Differences caused by these model forms are part of the fidelity discrepancy.

## Required audit before production

For at least one deterministic sample and one random sample, retain the PICLas updated parameter file and the ADBSat input audit. Verify sample ID, alpha, alpha squared momentum value, Tw, atmosphere row, velocity, attitude, Aref, and flow direction. Shared scalar values must agree to relative tolerance 1e-12 before solver file formatting. Any additional solver-only parameter must be fixed and listed, not silently inferred.

## Recommended uncertainty treatment

Use alpha as the single active GSI uncertainty in the primary benchmark. Keep the alpha-to-momentum relation frozen as alpha squared. A two-dimensional translational/momentum accommodation study would answer a different scientific question and should be reported as sensitivity analysis rather than mixed into the primary MFMC population.
