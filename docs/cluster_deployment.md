# Cluster Deployment

## Connected Login Node

```bash
git clone <repository-url> mfmc-framework
cd mfmc-framework
./scripts/bootstrap_cluster.sh production
source .venv/bin/activate
PICLAS_BIN=/cluster/path/to/piclas \
PICLAS2VTK_BIN=/cluster/path/to/piclas2vtk \
  ./scripts/configure_piclas.sh
python scripts/check_production_runtime.py
mfmc-campaign validate-config examples/toy_mock_campaign.yaml
```

`production` installs the optional HDF5, VTU, atmosphere, and tabular-data
dependencies. Use `test` to install both production and test dependencies.

## Offline Compute Nodes

On a connected machine with the same Python version and compatible platform:

```bash
python3 -m pip download -d wheelhouse ".[production]"
python3 -m build --wheel
python3 -m pip download -d wheelhouse dist/*.whl
```

Transfer the repository and `wheelhouse/` to the cluster, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --no-index --find-links wheelhouse ".[production]"
./scripts/smoke_test.sh
```

## Solver Runtime

ADBSat and all Python adapter classes are included in the checkout. PICLAS is
cluster-built software, so its two executables are installed into `piclas/`
with `scripts/configure_piclas.sh`. Campaign defaults then resolve
`PICLas.py`, `ADBSat.py`, `update_parameter_file/`, `piclas/`, and
`ADBSat-PyVersion/` directly from the repository root.

Production data should live outside Git or below the ignored `data/` folder:

```text
data/
  field_inputs/
    cube_300km/
      DSMC_surface_loads.npz
      TPMC_surface_loads.npz
      SENTMAN_surface_loads.npz
    goce_244km/
      DSMC_surface_loads.npz
      TPMC_surface_loads.npz
      SENTMAN_surface_loads.npz
```

Before submitting a large job, run configuration validation, the smoke test,
and a small pilot with `--resume` enabled.
