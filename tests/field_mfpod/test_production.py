import json

import yaml

from mfmc_campaign.field_mfpod.config import load_config
from mfmc_campaign.field_mfpod.production import production_status, run_production


def production_config(tmp_path):
    return {
        "case_name": "Cube-production-test",
        "geometry_id": "Cube",
        "output_root": str(tmp_path / "outputs"),
        "high_fidelity": "DSMC",
        "control_variates": ["TPMC"],
        "fidelity_archives": {
            "DSMC": str(tmp_path / "data" / "DSMC_surface_loads.npz"),
            "TPMC": str(tmp_path / "data" / "TPMC_surface_loads.npz"),
        },
        "costs": {"DSMC": 1.0, "TPMC": 0.1},
        "field_representation": {
            "quantity": "Total_ForcePerArea",
            "nondimensionalize": True,
            "area_weighted": True,
            "coordinate_frame": "body_fixed",
        },
        "pilot": {"paired_samples": 2},
        "reference_samples": 2,
        "field_allocation": {"enabled": True, "mean_weight": 0.25, "second_moment_weight": 0.75},
        "allocation_constraints": {"budget": 3.0},
        "production": {
            "enabled": True,
            "backend": "mock",
            "model_ids": {"DSMC": "Mock_HF", "TPMC": "Mock_TPMC"},
            "maximum_counts": {"DSMC": 3, "TPMC": 5},
            "geometry": {"id": "Cube", "name": "Cube"},
            "regime": {"id": "cube300", "descriptors": {"altitude_km": 300}},
            "active_source_blocks": ["environment.density"],
            "variables": [
                {
                    "name": "density_scale",
                    "source_block": "environment.density",
                    "distribution": {"kind": "fixed", "params": {"value": 1.0}},
                    "baseline": 1.0,
                }
            ],
            "sampling": {"method": "independent"},
            "models": {
                "hf": {"id": "Mock_HF", "kind": "mock"},
                "lf": [{"id": "Mock_TPMC", "kind": "mock"}],
                "available_qois": {"Mock_HF": ["C_D"], "Mock_TPMC": ["C_D"]},
            },
            "qois": ["C_D"],
            "random_seed": 41,
        },
    }


def test_production_dry_run_writes_deterministic_nested_plan_and_state(tmp_path):
    path = tmp_path / "production.yaml"
    path.write_text(yaml.safe_dump(production_config(tmp_path)), encoding="utf-8")
    cfg = load_config(path)

    first = run_production(cfg, dry_run=True)
    second = production_status(cfg)

    assert first["planned"] == second["planned"]
    assert first["planned"] == {
        "pilot": 2,
        "reference_DSMC": 2,
        "maximum_counts": {"DSMC": 3, "TPMC": 5},
    }
    plan_path = cfg.output_dir / "production" / "sample_plan.json"
    state_path = cfg.output_dir / "production" / "state.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(plan["sample_ids"]) == 9
    assert plan["roles"]["production_stream"][:3] != plan["roles"]["pilot"]
    assert state["status"] == "planned"

