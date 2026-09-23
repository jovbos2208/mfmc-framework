# Preparation status

## Prepared and statically validated

- Paper 1 active matrix: 12 configurations for Cube, sphere, and open hemispherical shell at 100, 200, 300, and 400 km. Each declares PICLas DSMC as target and PICLas TPMC and ADBSat Sentman as controls. Direct QoIs are Cd and Cd squared.
- SOAR is absent from the new active manifest. GOCE and CHAMP remain untouched in their legacy locations and are inventoried for Paper 2.
- Sphere and shell ADBSat assets: OBJ and MAT at coarse, medium, fine, and medium/large-domain levels; canonical NPZ and STL surfaces are retained for audit.
- PICLas assets: eight Gmsh volume meshes, PyHOPE input files, and HDF5 meshes. Standard domains extend 2.5 diameters in every signed coordinate direction; comparison domains extend 5 diameters.
- Shell regions are separately labeled in OBJ/MAT/NPZ as outer, inner, and rim. PICLas intentionally uses one reflective BENCHMARK_BODY boundary because the current adapter applies one wall model; region identity remains in the canonical surface archive.
- Cup-forward is the nominal shell orientation: molecular velocity +x and aperture outward normal -x. Cup-backward is prepared as a deterministic 180-degree orientation, not a thirteenth MFMC cell.

Mesh counts are recorded in piclas/geometry/paper1/MESH_MANIFEST.json. Sphere surface triangles are 224, 528, and 960; shell triangles are 512, 1152, and 2048. Standard PICLas HDF5 element counts range from 7,896 to 69,824; the two large-domain meshes contain 45,320 and 50,580 elements.

## Static checks performed

- Surface: SI units, dimensions, finite coordinates, no degenerate or duplicate triangles, two-manifold watertightness, Euler characteristic 2, positive signed material volume, unit normals, region counts, aperture accessibility, and projected x area versus pi D squared over four.
- Gmsh: positive tetrahedral volumes, expected gas volume, and nonempty IN, OUT, and BENCHMARK_BODY physical surfaces.
- PyHOPE: connectivity, watertightness, surface normals, internal boundaries, expected HDF5 datasets/boundary names, and zero negative scaled Jacobians.
- Configuration: all 12 YAML files pass the repository validate-config command; mesh/model paths exist; output paths are unique; seeds are explicit; sphere attitude-only random variables are omitted while winds remain.
- Paper 2: GOCE and CHAMP geometry/configuration/data provenance inventoried without deleting or moving legacy artifacts.

All split tetrahedra fall in PyHOPE scaled-Jacobian bin 0.0-0.1, matching the behavior of the existing project split-to-hex strategy. They are positive and structurally accepted, but this is a numerical-quality warning: solver ingestion and aerodynamic mesh convergence remain mandatory.

## Prepared but not simulation verified

- No PICLas, ADBSat, pilot, or production run was started.
- No aerodynamic reference value was invented or embedded.
- configs/paper1/verification_matrix.yaml defines fixed gas state, cup orientations, angle sweeps, and numerical variants; it is a study matrix, not a claim of completion.
- configs/paper2/champ_mfpod_DRAFT.yaml is disabled and not production-ready.

## Open prerequisites

1. Confirm solver-specific equivalence or mapping of accommodation/GSI parameters.
2. Source the collisionless diffuse sphere Cd reference and define acceptance tolerance.
3. Verify solver ingestion and numerical convergence of the uniformly scaled 10 cm Cube; its frozen normalization is Lref=0.1 m and Aref=0.01 square metres.
4. Perform PICLas ingestion smoke tests and independent convergence studies.
5. Resolve CHAMP original scale, Aref, mission state, and statistical population.
6. Add missing GOCE field metadata: component order and coordinate frame; preferably CPU hours, normals, hardware, and reference point.
7. Export CHAMP full vector surface-force fields before MFPOD work.
8. The manuscript source is not present in this repository, so the Paper 1 plan cannot cite exact source lines.

## Later commands; do not run during preparation

Static recheck:

    source /home/jovan/venv/bin/activate
    for f in configs/paper1/*km.yaml; do python -m mfmc_campaign.cli validate-config "$f" || exit 1; done

Four executable one-sample or angle-grid verification configs are already present under configs/paper1/verification. Run only those explicit configs, for example:

    python -m mfmc_campaign.cli run configs/paper1/verification/sphere_diffuse_smoke.yaml
    python -m mfmc_campaign.cli run-sweep configs/paper1/verification/shell_cup_forward.yaml configs/paper1/verification/shell_cup_backward.yaml

After all verification and convergence gates pass, pilot only:

    python -m mfmc_campaign.cli run-pilots-sweep configs/paper1/cube_100km.yaml configs/paper1/cube_200km.yaml configs/paper1/cube_300km.yaml configs/paper1/cube_400km.yaml

Repeat the pilot command by geometry; do not launch the 12-case production sweep until correlations, costs, GSI mapping, and numerical settings are frozen.

Paper 2 data checks:

    python -m mfmc_campaign.cli check-field-data configs/mfpod/goce_tpmc_sentman.yaml
    python -m mfmc_campaign.cli check-field-data configs/paper2/champ_mfpod_DRAFT.yaml

The CHAMP command is expected to fail until the listed prerequisites are resolved.

## Recommended next order

1. Confirm the documented GSI mapping with deterministic input audits. 2. Run deterministic sphere and shell ingestion/physics verification. 3. Run numerical convergence variants. 4. Freeze Paper 1 settings and execute paired pilots. 5. Decide production allocations. In parallel at the planning level only, resolve CHAMP scale/population and repair GOCE archive metadata before any Paper 2 field campaign.
