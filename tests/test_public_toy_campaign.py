from __future__ import annotations

import json
from pathlib import Path

import yaml

import mfmc_campaign
from mfmc_campaign.campaign import run_campaign
from mfmc_campaign.cli import run_cli
from mfmc_campaign.config import load_and_validate


ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG = ROOT / "examples" / "toy_mock_campaign.yaml"


def test_package_import_smoke() -> None:
    assert hasattr(mfmc_campaign, "run_campaign")


def test_toy_config_validates() -> None:
    cfg = load_and_validate(str(TOY_CONFIG))

    assert cfg["study"]["id"] == "toy_mock_campaign"
    assert cfg["execution"]["backend"] == "mock"


def test_cli_validate_config(capsys) -> None:
    exit_code = run_cli(["validate-config", str(TOY_CONFIG)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config valid:" in captured.out
    assert "toy_mock_campaign" in captured.out


def test_toy_campaign_run_writes_core_outputs(tmp_path: Path) -> None:
    cfg_data = yaml.safe_load(TOY_CONFIG.read_text(encoding="utf-8"))
    cfg_data["outputs"]["dir"] = str(tmp_path / "toy_outputs")

    config_path = tmp_path / "toy_mock_campaign.yaml"
    config_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding="utf-8")

    cfg = load_and_validate(str(config_path))
    summary = run_campaign(cfg, resume=False)

    output_dir = Path(summary["output_dir"])
    assert summary["study_id"] == "toy_mock_campaign"
    assert summary["n_cells_executed"] == 1
    assert (output_dir / "results_long.csv").exists()
    assert (output_dir / "model_evaluations.csv").exists()
    assert (output_dir / "pilot_robustness.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "config_snapshot.json").exists()

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["study_id"] == "toy_mock_campaign"
