from pathlib import Path

from mfmc_campaign.config import load_and_validate


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "studies"
    / "pilot_correlation"
    / "GOCE_high_aoa_aos_moment_correlation.yaml"
)


def test_goce_high_attitude_uses_tpmc_as_hf_and_sentman_as_lf():
    config = load_and_validate(str(CONFIG))
    models = config["models"]

    assert models["hf"]["id"] == "PICLas_TPMC"
    assert models["hf"]["kind"] == "legacy_piclas"
    assert models["hf"]["kwargs"]["simulator_module"] == "PICLas_prandtl"
    assert models["hf"]["kwargs"]["piclas_mode"] == "tpmc"
    assert models["hf"]["kwargs"]["mpi_procs"] == 64
    assert models["hf"]["kwargs"]["submission_group_size"] == 10
    assert [model["id"] for model in models["lf"]] == ["Sentman"]
    assert models["lf"][0]["kwargs"]["simulator_module"] == "ADBSat_prandtl"
    assert config["pilot"]["size"] == 80
    assert config["qois"]["direct"] == ["C_D", "C_D2"]
