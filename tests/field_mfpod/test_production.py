import json
import sys
import types

import numpy as np
import pytest
import yaml

from mfmc_campaign.adapters import LegacyPiclasAdapter
from mfmc_campaign.field_mfpod.config import load_config
from mfmc_campaign.field_mfpod.models import MFPODError
from mfmc_campaign.field_mfpod.production import (
    _runtime_config,
    _run_piclas_workloads,
    production_status,
    run_production,
)
from mfmc_campaign.types import EvaluationResult


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


def test_existing_sample_plan_rejects_changed_role_counts(tmp_path):
    path = tmp_path / "production.yaml"
    configured = production_config(tmp_path)
    path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    run_production(load_config(path), dry_run=True)
    configured["pilot"]["paired_samples"] = 3
    path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    with pytest.raises(MFPODError, match="immutable sample_plan"):
        production_status(load_config(path))


def test_hf_equivalent_budget_uses_measured_pilot_dsmc_cost(tmp_path):
    path = tmp_path / "production.yaml"
    configured = production_config(tmp_path)
    configured["allocation_constraints"] = {
        "budget_hf_equivalent": 20.0,
    }
    path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    cfg = load_config(path)
    runtime = _runtime_config(
        cfg,
        cfg.raw["production"],
        tmp_path / "pilot_fields.npz",
        measured_costs={"DSMC": 2.5, "TPMC": 0.1},
    )
    assert runtime.raw["allocation_constraints"]["budget"] == 50.0
    assert runtime.raw["costs"]["DSMC"] == 2.5


def test_piclas_workloads_are_all_submitted_before_sequential_collection(tmp_path):
    path = tmp_path / "production.yaml"
    path.write_text(yaml.safe_dump(production_config(tmp_path)), encoding="utf-8")
    cfg = load_config(path)
    events = []

    class FakeAdapter:
        def __init__(self, fidelity, archive):
            self.fidelity = fidelity
            self.archive = archive

        def submit(self, request):
            events.append(f"submit:{self.fidelity}")
            return request

        def collect(self, request):
            events.append(f"collect:{self.fidelity}")
            self.archive.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.archive, sample_id=np.asarray(request.sample_ids))
            return EvaluationResult(
                values_by_qoi={"C_D": [1.0] * len(request.sample_ids)},
                costs=[1.0] * len(request.sample_ids),
                sample_ids=request.sample_ids,
            )

    adapters = {
        "Mock_HF": FakeAdapter("DSMC", cfg.archives["DSMC"]),
        "Mock_TPMC": FakeAdapter("TPMC", cfg.archives["TPMC"]),
    }

    class Registry:
        def get(self, model_id):
            return adapters[model_id]

    sample_lookup = {"d0": {"database_index": 0}, "t0": {"database_index": 1}}
    result = _run_piclas_workloads(
        cfg,
        cfg.raw["production"],
        Registry(),
        {"DSMC": "Mock_HF", "TPMC": "Mock_TPMC"},
        [("DSMC", "DSMC", ["d0"], "pilot"), ("TPMC", "TPMC", ["t0"], "pilot")],
        sample_lookup,
    )

    assert events == ["submit:DSMC", "submit:TPMC", "collect:DSMC", "collect:TPMC"]
    assert result["DSMC"]["status"] == "complete"
    assert result["TPMC"]["status"] == "complete"


def test_piclas_default_group_sizes_are_one_for_dsmc_and_ten_for_tpmc(monkeypatch):
    module = types.ModuleType("fake_piclas_grouping")

    class Simulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.PiclasSimulator = Simulator
    monkeypatch.setitem(sys.modules, module.__name__, module)

    dsmc = LegacyPiclasAdapter("HF", ["C_D"], {"simulator_module": module.__name__}, fidelity="hf")
    tpmc = LegacyPiclasAdapter(
        "TPMC",
        ["C_D"],
        {"simulator_module": module.__name__, "piclas_mode": "tpmc"},
        fidelity="lf",
    )

    assert dsmc.sim.kwargs["submission_group_size"] == 1
    assert tpmc.sim.kwargs["submission_group_size"] == 10
