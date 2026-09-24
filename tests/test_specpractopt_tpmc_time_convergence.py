from __future__ import annotations

import pytest

from scripts.analyze_specpractopt_tpmc_time_convergence import paired_stats, summary_stats
from scripts.build_specpractopt_tpmc_time_convergence import build_workflow_config


def test_workflow_uses_fixed_environment_and_common_seed_controls() -> None:
    config = build_workflow_config(1.0e-4, (101, 102, 103, 104, 105))
    defaults = config["adapter"]["kwargs"]["payload_defaults"]
    request = config["request"]

    assert defaults["t_end_s"] == pytest.approx(1.0e-4)
    assert defaults["momentum_accommodation"] == pytest.approx(1.0)
    assert defaults["sampling_iterations"] == 250
    assert config["adapter"]["kwargs"]["collision_mode"] == 0
    assert request["metadata"]["atmosphere_row"][3] == pytest.approx(1.0e15)
    assert request["metadata"]["aos_deg"] == pytest.approx(45.0)
    assert {row["energy_accommodation"] for row in request["samples"]} == {0.5}
    assert [row["random_seed"] for row in request["samples"]] == [101, 102, 103, 104, 105]


def test_summary_and_paired_statistics_preserve_sign() -> None:
    reference = [1.00, 1.02, 0.98, 1.01, 0.99]
    values = [value + 0.02 for value in reference]

    summary = summary_stats(values)
    paired = paired_stats(values, reference)

    assert summary["n"] == 5
    assert paired["mean"] == pytest.approx(0.02)
    assert paired["ci95"][0] == pytest.approx(0.02)
    assert paired["ci95"][1] == pytest.approx(0.02)
