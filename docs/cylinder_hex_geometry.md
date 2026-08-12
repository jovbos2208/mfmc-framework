# Parametric Cylinder-Hex Geometry

This geometry family is generated analytically from scratch. No vertices,
dimensions, topology, or mesh entities are imported from an existing `.msh`,
OBJ, STL, or solver project.

The body axis is positive `x`, from the nose cap center to the tail cap center.
Every cross-section is a scaled, convex, chamfered rectangle in the `y-z`
plane. Piecewise-linear scaling connects a smaller nose cap to the maximum
payload section and then to a smaller tail cap. The result is a closed convex
polyhedron.

## Design variables

- `nose_length_fraction`: fraction of total length occupied by the nose taper;
- `tail_length_fraction`: fraction occupied by the aft taper;
- `width_height_ratio`: maximum-section width divided by height;
- `chamfer_fraction`: normalized corner cut in both cross-section directions.

Total length, target enclosed volume, maximum width, maximum height, and the two
end scales are constraints or fixed study constants, not optimization
variables. For each design, width and height are solved analytically so the
enclosed volume equals the target volume. Designs that cannot satisfy the
external envelope are rejected.

## Reference area

The manifest records the maximum cross-section area normal to the body `+x`
axis as the canonical body-axis reference area. Solver-specific coefficients
may use a wind-projected area at nonzero attitude. Cross-geometry optimization
must therefore use dimensional drag, equivalently `C_D * A_ref`, rather than
compare raw drag coefficients.

## Generated assets

`generate-cylinder-hex` writes:

- an OBJ and MAT asset for ADBSat;
- an STL surface for the independent PICLas volume-meshing stage;
- an analytical Gmsh exterior-domain definition with `IN`, `OUT`, and
  `CYLINDER_HEX` physical boundaries;
- a canonical NPZ surface archive;
- a JSON manifest with parameters, fingerprints, reference-area convention,
  derived dimensions, provenance, and validity checks.

The generator verifies finite coordinates, nondegenerate triangles, a closed
two-manifold surface, outward orientation, convexity, fixed volume, and envelope
limits. Convexity plus the closed two-manifold check provides the no-self-
intersection certificate for this specific analytical family.

Example:

```bash
python -m mfmc_campaign.cli generate-cylinder-hex \
  --output-dir outputs/vleo_geometry/cylinder_hex_baseline \
  --design-id cylinder_hex_baseline
```

Uniform scaling is explicit. For example, `--uniform-scale 0.1` multiplies all
lengths by `0.1`, all areas by `0.01`, and the enclosed volume by `0.001`. The
manifest records the factor and scaling laws. This geometric operation changes
the Knudsen number unless the gas state is scaled consistently; it must not be
presented as an aerodynamically equivalent cost reduction without a similarity
and mesh-convergence study.

The generated `0.1` baseline has length `0.1 m`, volume `1.2e-4 m3`, maximum
width and height `0.0398105 m`, and body-axis reference area
`1.51356e-3 m2`. With the current absolute Gmsh target sizes its validated
exterior mesh contains 181 tetrahedra instead of 8,361 for the unit-scale
baseline (about 46 times fewer). This is a cost indicator, not yet a converged
DSMC mesh result.

The validated Gmsh exterior mesh is converted to PICLas/HOPR HDF5 with PyHOPE.
The conversion splits each tetrahedron into hexahedra and enables PyHOPE's
Jacobian, connectivity, watertightness, normal, and internal-boundary checks.
The manifest records the PyHOPE version, config, output fingerprint, boundary
names, and element counts. A successful conversion is still distinct from a
PICLas smoke run.

Where Gmsh is installed, the new exterior tetrahedral mesh can be generated and
checked with:

```bash
python -m mfmc_campaign.cli mesh-cylinder-hex-exterior \
  --manifest outputs/vleo_geometry/cylinder_hex_baseline/cylinder_hex_baseline.manifest.json
```

Then create the solver mesh in the PyHOPE-enabled environment:

```bash
python -m mfmc_campaign.cli convert-cylinder-hex-pyhope \
  --manifest outputs/vleo_geometry/cylinder_hex_baseline/cylinder_hex_baseline.manifest.json \
  --pyhope /home/jovan/venv/bin/pyhope
```

The validated smoke-run mesh and its conversion provenance are versioned under
`piclas/geometry`, so a Git checkout on the cluster does not depend on the
ignored local `outputs/` tree. For a restartable Slurm smoke run, edit the adapter paths in
`configs/studies/cylinder_hex_piclas_adapter_smoke.json`, then first run the
non-mutating preflight:

```bash
python scripts/run_piclas_adapter_workflow.py submit \
  --config configs/studies/cylinder_hex_piclas_adapter_smoke.json \
  --state outputs/vleo_geometry/cylinder_hex_scale_0p1/piclas_batch_state.json
```

On a Slurm login node, add `--execute` to submit. After submission, collect the
same handle and write QoIs and CPU-hours with:

```bash
python scripts/run_piclas_adapter_workflow.py collect --execute \
  --config configs/studies/cylinder_hex_piclas_adapter_smoke.json \
  --state outputs/vleo_geometry/cylinder_hex_scale_0p1/piclas_batch_state.json \
  --results outputs/vleo_geometry/cylinder_hex_scale_0p1/piclas_results.json
```
