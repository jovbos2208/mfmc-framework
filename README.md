# MFMC Framework

MFMC Framework provides reusable workflows for scalar multifidelity Monte Carlo
(MFMC), field-level POD/MFMC, and multifidelity POD (MFPOD). It includes
campaign configuration, sampling, pilot diagnostics, cost-aware allocation,
restartable execution, surface-load archives, and PICLAS/ADBSat adapters.

The production checkout includes the Python solver adapters, the ADBSat Python
runtime and atmosphere tables, PICLAS parameter files, and the standard HDF5
meshes. Generated results remain outside Git. The site-specific PICLAS and
`piclas2vtk` binaries must be supplied by the cluster installation.

## Install

Core installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Installation with PICLAS/ADBSat surface and atmosphere support:

```bash
python3 -m pip install -e ".[production]"
```

For development and tests:

```bash
python3 -m pip install -e ".[dev,production]"
python3 -m pytest -q
```

## Commands

```bash
# Scalar MFMC campaign
mfmc-campaign validate-config examples/toy_mock_campaign.yaml
mfmc-campaign run examples/toy_mock_campaign.yaml
mfmc-campaign run-pilots-sweep config-a.yaml config-b.yaml --resume

# Archived DSMC + TPMC + Sentman field workflow
mfmc-campaign check-field-data configs/mfpod/cube_tpmc_sentman.yaml
mfmc-campaign run-field-pod-mfmc configs/mfpod/cube_tpmc_sentman.yaml

# Full-field, allocation-first MFMC/POD workflow
mfmc-mfpod inspect --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod production --config configs/mfpod/cube_tpmc_sentman.yaml --dry-run
mfmc-mfpod production --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod production-status --config configs/mfpod/cube_tpmc_sentman.yaml
# After an interruption, continue without resubmitting archived sample IDs:
mfmc-mfpod production --config configs/mfpod/cube_tpmc_sentman.yaml --resume

# Analysis-only commands for archives that already exist
mfmc-mfpod prepare-field-snapshots --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod field-pilot --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod optimal-allocation --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod field-estimate --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod field-pod --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod field-benchmark --config configs/mfpod/cube_tpmc_sentman.yaml
mfmc-mfpod run-all --config configs/mfpod/cube_tpmc_sentman.yaml

# Canonical PICLAS VTU to ADBSat OBJ/MAT and conservative face mapping
mfmc-campaign build-adbsat-surface \
  --vtu data/surface.vtu \
  --obj data/geometry.obj \
  --mat data/geometry.mat \
  --mapping data/surface_mapping.npz
```

The MFPOD templates expect archives below `data/field_inputs/<case>/` and
write generated results below `outputs/`. The archive format and conservative
Sentman mapping are documented in [docs/field_pod_mfmc.md](docs/field_pod_mfmc.md).
The checkpoint-by-checkpoint cluster procedure is documented in
[docs/mfpod_cluster_runbook.md](docs/mfpod_cluster_runbook.md).
The `production` command owns the persisted sample plan, paired pilots,
pilot-cost allocation, nested production streams, DSMC-only reference set,
and final analysis. Run it from the repository root because solver paths in
the production template are repository-relative.

## Cluster Deployment

```bash
git clone <repository-url> mfmc-framework
cd mfmc-framework
./scripts/bootstrap_cluster.sh production
source .venv/bin/activate
PICLAS_BIN=/path/to/piclas \
PICLAS2VTK_BIN=/path/to/piclas2vtk \
  ./scripts/configure_piclas.sh
python scripts/check_production_runtime.py
./scripts/smoke_test.sh
```

For an offline compute cluster, build a wheelhouse on a connected login node
and install from it as described in [docs/cluster_deployment.md](docs/cluster_deployment.md).

## Repository Boundary

Included:

- MFMC campaign engine and CLI
- pilot correlation, allocation, output, and plotting functions
- field POD/MFMC and MFPOD implementations
- PICLAS surface archive export
- VTU-to-ADBSat geometry conversion and conservative surface mapping
- campaign adapter interfaces for PICLAS DSMC/TPMC and ADBSat models
- bundled `PICLas*.py` and `ADBSat*.py` simulation adapter classes
- bundled ADBSat Python runtime, standard geometries, and atmosphere tables
- PICLAS parameter templates and standard Cube/GOCE/CHAMP/SOAR meshes
- focused unit, integration, and synthetic tests

Not included:

- site-specific PICLAS and `piclas2vtk` binaries
- PICLAS source code and generated solver output
- SLURM outputs, generated archives, and paper results
- machine-specific absolute paths

## License

This repository is distributed under the GNU General Public License v3.0 only
(`GPL-3.0-only`). It contains a modified Python implementation derived from
ADBSat, which is also GPLv3 software. See `LICENSE` and
`THIRD_PARTY_NOTICES.md` for details and upstream attribution.
