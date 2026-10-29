from pathlib import Path
from types import SimpleNamespace

import pytest

import ADBSat


def test_mpi_procs_controls_slurm_allocation_and_worker_environment(tmp_path, monkeypatch):
    base_dir = tmp_path / "ADBSat-PyVersion"
    base_dir.mkdir()
    work_dir = tmp_path / "request"

    monkeypatch.setattr(ADBSat.shutil, "which", lambda _name: "/usr/bin/sbatch")
    monkeypatch.setattr(
        ADBSat.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Submitted batch job 123\n", stderr=""),
    )

    simulator = ADBSat.ADBSatSimulator(
        method="CLL",
        base_dir=str(base_dir),
        mpi_procs=12,
    )
    job_id, _ = simulator.queue_simulation_job(300, 0, "inputs.txt", work_dir=str(work_dir))

    script = (work_dir / "job_CLL.sh").read_text(encoding="utf-8")
    assert job_id == "123"
    assert "#SBATCH --cpus-per-task=12" in script
    assert "export ADBSAT_NUM_WORKERS=12" in script


def test_mpi_procs_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="mpi_procs must be at least 1"):
        ADBSat.ADBSatSimulator(method="CLL", base_dir=str(tmp_path), mpi_procs=0)
