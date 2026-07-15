# Toy Mock Campaign Tutorial

This tutorial runs a complete MFMC campaign without external solvers or data files. Both the high-fidelity and low-fidelity models use deterministic mock evaluators, so the example is useful for checking installation, configuration, output generation, and CLI behavior.

## 1. Create An Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

## 2. Validate The Config

```bash
mfmc-campaign validate-config examples/toy_mock_campaign.yaml
```

Expected output includes:

```text
Config valid: examples/toy_mock_campaign.yaml
Study id: toy_mock_campaign
```

## 3. Run The Campaign

```bash
mfmc-campaign run examples/toy_mock_campaign.yaml
```

The example runs one geometry, one regime, one uncertain source block, one direct quantity of interest, and one low-fidelity model.

## 4. Inspect Outputs

The run writes to `outputs/toy_mock_campaign/`.

Important files:

- `results_long.csv` contains the MFMC estimates and diagnostics for each campaign cell.
- `model_evaluations.csv` contains the sampled high-fidelity and low-fidelity mock values.
- `pilot_robustness.csv` contains pilot-size correlation and gain diagnostics.
- `summary.json` records output paths and cell counts.
- `config_snapshot.json` stores the normalized config used for the run.

To regenerate summary tables from existing outputs:

```bash
mfmc-campaign postprocess --output-dir outputs/toy_mock_campaign
```

To enable plots for this example, set `outputs.plots: true` in `examples/toy_mock_campaign.yaml` and rerun the campaign.
