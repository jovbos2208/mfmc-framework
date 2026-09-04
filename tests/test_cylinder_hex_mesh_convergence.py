from __future__ import annotations

import pytest

from scripts.analyze_cylinder_hex_mesh_convergence import _stats
from scripts.build_cylinder_hex_mesh_convergence import _five_seed_config


def test_convergence_statistics_report_sample_uncertainty() -> None:
    report = _stats([2.3710, 2.3702, 2.3688, 2.3725, 2.3685])

    assert report["n"] == 5
    assert report["mean_cd"] == pytest.approx(2.3702, abs=1.0e-4)
    assert report["sample_std_cd"] > 0.0
    assert report["ci95_cd"][0] < report["mean_cd"] < report["ci95_cd"][1]


def test_five_seed_config_preserves_cluster_and_flow_controls() -> None:
    base = {
        "adapter": {"kwargs": {"mpi_procs": 128}},
        "request": {
            "geometry": {"id": "old", "name": "old", "metadata": {"hf_mesh": "old.h5"}},
            "metadata": {
                "flow_zero_direction": [1.0, 0.0, 0.0],
                "case_name": "old",
            },
        },
    }

    config = _five_seed_config(
        base,
        level="L1",
        geometry_id="cylinder_hex_scale_0p1_l1",
        mesh_reference="geometry/convergence/L1/mesh.h5",
    )

    assert config["adapter"]["kwargs"]["mpi_procs"] == 128
    assert config["request"]["metadata"]["flow_zero_direction"] == [1.0, 0.0, 0.0]
    assert len(config["request"]["samples"]) == 5
    assert len({sample["random_seed"] for sample in config["request"]["samples"]}) == 5
