import csv
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "estimate_anglewise_mfmc.py"
SPEC = importlib.util.spec_from_file_location("estimate_anglewise_mfmc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "phase", "fidelity", "model_id", "qoi", "regime_id", "repetition",
    "sample_id", "sample_index", "sample_fingerprint", "value",
]


def add(rows, *, phase, qoi, index, value, angle=45):
    fidelity = "hf" if phase.endswith("hf") else "lf"
    prefix = "pilot" if phase.startswith("pilot") else "prod"
    rows.append({
        "phase": phase,
        "fidelity": fidelity,
        "model_id": "PICLas_TPMC" if fidelity == "hf" else "Maxwell",
        "qoi": qoi,
        "regime_id": f"CASE_AOS_P{angle:03d}_AOA_P000",
        "repetition": "0",
        "sample_id": f"{prefix}_{index}",
        "sample_index": str(index),
        "sample_fingerprint": f"{prefix}-fingerprint-{index}",
        "value": str(value),
    })


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(reversed(rows))  # pairing must not depend on row order


def test_anglewise_weights_and_estimates(tmp_path):
    rows = []
    for qoi_offset, qoi in enumerate(MODULE.QOIS):
        for i, lf in enumerate((10.0, 12.0, 14.0, 16.0)):
            add(rows, phase="prod_hf", qoi=qoi, index=i, value=5.0 + 2.0 * lf)
            add(rows, phase="prod_lf_pair", qoi=qoi, index=i, value=lf)
        for i, lf in enumerate((10.0, 12.0, 14.0, 16.0, 18.0)):
            add(rows, phase="prod_lf_full", qoi=qoi, index=i, value=lf)
    source = tmp_path / "model_evaluations.csv"
    write_csv(source, rows)

    result = MODULE.estimate_file(source, pair_count=4, lf_model="Maxwell")

    assert len(result) == 4
    for row in result:
        assert row["beta"] == pytest.approx(2.0)
        assert row["rho"] == pytest.approx(1.0)
        assert row["estimate"] == pytest.approx(33.0)
        assert row["n_weight_pairs"] == 4
        assert row["n_estimate_pairs"] == 4
        assert row["n_lf_full"] == 5


def test_requires_requested_number_of_pairs(tmp_path):
    rows = []
    for qoi in MODULE.QOIS:
        for i in range(3):
            add(rows, phase="prod_hf", qoi=qoi, index=i, value=2 * i + 1)
            add(rows, phase="prod_lf_pair", qoi=qoi, index=i, value=i)
    source = tmp_path / "model_evaluations.csv"
    write_csv(source, rows)

    with pytest.raises(ValueError, match="found 3 paired production samples, need 4"):
        MODULE.estimate_file(source, pair_count=4)


def test_parse_negative_angle():
    assert MODULE.parse_angle("CASE_AOS_M090_AOA_P000") == -90.0
