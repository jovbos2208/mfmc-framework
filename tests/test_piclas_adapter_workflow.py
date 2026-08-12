from __future__ import annotations

from pathlib import Path

from mfmc_campaign.piclas_adapter_workflow import plan_workflow, preflight, similarity_report


def _config(tmp_path: Path) -> dict:
    piclas_dir = tmp_path / "piclas"
    update_dir = tmp_path / "update"
    piclas_dir.mkdir()
    update_dir.mkdir()
    for name in ("piclas", "piclas2vtk", "DSMC1.ini"):
        (piclas_dir / name).touch()
    (update_dir / "update_parameter.py").touch()
    mesh = tmp_path / "scaled_mesh.h5"
    mesh.touch()
    return {
        "similarity": {"linear_scale": 0.1, "density_scale": 10.0},
        "adapter": {
            "model_id": "PICLas_DSMC",
            "kwargs": {
                "piclas_dir": str(piclas_dir),
                "update_dir": str(update_dir),
                "payload_defaults": {"density_scale": 10.0},
            },
        },
        "request": {
            "qois": ["C_D"],
            "geometry": {"id": "scaled", "metadata": {"hf_mesh": str(mesh)}},
            "regime": {"id": "vleo", "descriptors": {"altitude_km": 250.0}},
            "samples": [{"database_index": 0}],
        },
    }


def test_preflight_confirms_knudsen_similarity_and_required_paths(tmp_path: Path) -> None:
    report = preflight(_config(tmp_path))

    assert report["ready"]
    assert report["similarity"]["knudsen_number_preserved"]
    assert report["similarity"]["knudsen_ratio_scaled_to_reference"] == 1.0


def test_preflight_detects_declared_payload_density_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["adapter"]["kwargs"]["payload_defaults"]["density_scale"] = 9.0

    report = preflight(config)

    assert not report["ready"]
    assert not similarity_report(config)["declared_and_payload_density_match"]
    assert any("density_scale" in issue for issue in report["issues"])


def test_submit_plan_is_non_mutating(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"

    plan = plan_workflow(_config(tmp_path), state_path=state_path)

    assert plan["status"] == "dry_run"
    assert plan["n_samples"] == 1
    assert not state_path.exists()
