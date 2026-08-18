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
  --pyhope pyhope
```

The validated smoke-run mesh and its conversion provenance are versioned under
`piclas/geometry`, so a Git checkout on the cluster does not depend on the
ignored local `outputs/` tree. For a restartable Slurm smoke run, edit the adapter paths in
`configs/studies/cylinder_hex_piclas_adapter_smoke.json`, then first run the
non-mutating preflight:

The generated farfield marks the `x_min` face as `IN`. The smoke configuration
therefore fixes `flow_zero_direction` to `[1, 0, 0]`, so injected particles move
into the gas domain and the same axis is used to project drag during collection.

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

## Mesh-convergence suite

The reproducible convergence builder creates L1 and L2 by halving the Gmsh
body and farfield characteristic lengths at each level. It validates Gmsh gas
volume and boundary groups, converts with PyHOPE, records HDF5 fingerprints and
scaled-Jacobian histograms, and writes five-seed PICLas configs:

```bash
python scripts/build_cylinder_hex_mesh_convergence.py --pyhope pyhope
```

The versioned solver meshes contain 724 (L0), 1,648 (L1), and 5,448 (L2)
hexahedra. Refinement removes no validity checks and introduces no negative
Jacobians, but tetrahedron-to-hexahedron splitting leaves every element in
PyHOPE's lowest positive scaled-Jacobian bin. Result convergence and element
quality must therefore be reported as separate limitations.

Each refined level is submitted and collected with its generated config. After
all three five-seed result files exist, produce the statistical convergence
report with:

```bash
python scripts/analyze_cylinder_hex_mesh_convergence.py \
  --suite piclas/geometry/cylinder_hex_convergence/mesh_convergence_suite.json \
  --l0-results outputs/cylinder_hex/piclas_results_5seeds.json \
  --l1-results outputs/cylinder_hex/mesh_convergence/l1_results.json \
  --l2-results outputs/cylinder_hex/mesh_convergence/l2_results.json \
  --output outputs/cylinder_hex/mesh_convergence/convergence_report.json
```

## WP5 geometry and low-fidelity campaign

The production design contains 32 fixed-volume geometries: the validated
baseline, 25 LF-training designs, and six validation geometries whose role is
fixed before solver results are observed. Generate it on the cluster and stage
the ADBSat assets directly into its runtime with:

```bash
python -m mfmc_campaign.cli generate-cylinder-hex-design \
  --output-dir outputs/cylinder_hex/wp5_design \
  --n-designs 32 --n-validation 6 --uniform-scale 0.1 \
  --adbsat-runtime-dir ADBSat-PyVersion
```

Prepare 256 common-random-number uncertainty samples from the first-paper
archive (the backfilled WP1 `sample_inputs.csv` is equivalent):

```bash
python scripts/run_cylinder_hex_lf_campaign.py prepare \
  --design-manifest outputs/cylinder_hex/wp5_design/geometry_design_manifest.json \
  --source-samples outputs/vleo_wp1_backfill/cube_300km/sample_inputs.csv \
  --config outputs/cylinder_hex/wp5_lf_config.json \
  --n-samples 256 --adbsat-runtime-dir ADBSat-PyVersion

python scripts/run_cylinder_hex_lf_campaign.py run \
  --config outputs/cylinder_hex/wp5_lf_config.json \
  --output-dir outputs/cylinder_hex/wp5_lf
```

The second command is a non-mutating preflight unless `--execute` is appended.
The executable run writes restart state after every geometry and produces
`lf_robust_metrics.csv`. It evaluates `C_D*A_ref`, because raw coefficients with
different reference areas are not a valid cross-geometry drag objective. Both
ADBSat and PICLas use the same fixed manifest reference area; legacy ADBSat's
attitude-dependent projected-area normalization is not used for this study.

After LF completion, select six initial HF geometries without touching the
reserved validation set:

```bash
python -m mfmc_campaign.cli select-cylinder-hex-initial-hf \
  --design-csv outputs/cylinder_hex/wp5_design/geometry_design.csv \
  --lf-metrics-csv outputs/cylinder_hex/wp5_lf/lf_robust_metrics.csv \
  --output outputs/cylinder_hex/wp5_initial_hf.json --count 6
```

Build L1 meshes and five common-random-number PICLas inputs for each of the six
selected geometries (30 initial HF runs):

```bash
python scripts/build_cylinder_hex_initial_hf_suite.py \
  --selection outputs/cylinder_hex/wp5_initial_hf.json \
  --design-manifest outputs/cylinder_hex/wp5_design/geometry_design_manifest.json \
  --lf-config outputs/cylinder_hex/wp5_lf_config.json \
  --gmsh gmsh --pyhope pyhope

python scripts/run_cylinder_hex_initial_hf_suite.py submit \
  --suite piclas/geometry/cylinder_hex_wp5/initial_hf_suite.json \
  --run-root outputs/cylinder_hex/wp5_initial_hf
```

The submit command above performs preflight only. Add `--execute` after all six
geometries report `ready: true`. Collection is likewise performed as one
restartable suite action:

```bash
python scripts/run_cylinder_hex_initial_hf_suite.py collect --execute \
  --suite piclas/geometry/cylinder_hex_wp5/initial_hf_suite.json \
  --run-root outputs/cylinder_hex/wp5_initial_hf
```
