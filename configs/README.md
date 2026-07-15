# Configuration Templates

`mfpod/` contains portable three-fidelity DSMC/TPMC/Sentman templates. Paths
are resolved relative to each YAML file. Put local archives below
`data/field_inputs/` or adapt the paths in a cluster-specific copy.

Campaign configurations for external solvers must provide their own solver
locations. `examples/toy_mock_campaign.yaml` is the installation smoke test
and requires no external software.

