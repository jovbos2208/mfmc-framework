import pathlib
import subprocess
import tempfile
from unittest.mock import patch

from PICLas_prandtl import PiclasSimulator


def test_prandtl_defaults_to_four_concurrent_jobs():
    simulator = PiclasSimulator()
    assert simulator.max_concurrent_jobs == 4


def test_submission_slot_waits_until_one_tracked_job_finishes():
    simulator = PiclasSimulator(max_concurrent_jobs=2, submission_poll_interval_s=0)
    with patch.object(
        simulator,
        "_query_job_states",
        side_effect=[{"11": "R", "12": "PD"}, {"12": "R"}],
    ) as query, patch("PICLas_prandtl.time.sleep") as sleep:
        active = simulator._wait_for_submission_slot(["11", "12"])

    assert active == ["12"]
    assert query.call_count == 2
    sleep.assert_called_once_with(0.0)


def test_submission_slot_does_not_query_below_limit():
    simulator = PiclasSimulator(max_concurrent_jobs=4)
    with patch.object(simulator, "_query_job_states") as query:
        active = simulator._wait_for_submission_slot(["21", "22", "23"])

    assert active == ["21", "22", "23"]
    query.assert_not_called()


def test_prandtl_group_script_uses_slurm_native_mpi_launch():
    with tempfile.TemporaryDirectory() as td:
        cases = [pathlib.Path(td, "case one"), pathlib.Path(td, "case two")]
        for case in cases:
            case.mkdir()
        simulator = PiclasSimulator(piclas_dir=td, mpi_procs=36)
        script_path = simulator.create_simulation_group_job_script([str(x) for x in cases])
        script = pathlib.Path(script_path).read_text(encoding="utf-8")
        assert "#SBATCH --ntasks=36" in script
        assert "#SBATCH --cpus-per-task=1" in script
        assert 'srun $SRUN_MPI_OPT --kill-on-bad-exit=1 -n "$NTASKS"' in script
        assert "mpirun" not in script
        subprocess.run(["bash", "-n", script_path], check=True)
