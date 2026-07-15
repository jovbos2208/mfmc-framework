import json
import os
import pathlib
import shutil
import io
import sys
import tempfile
import types
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

meshio_stub = types.ModuleType("meshio")
sys.modules.setdefault("meshio", meshio_stub)
pyvista_stub = types.ModuleType("pyvista")
pyvista_stub.DataSet = object
pyvista_stub.read = lambda *args, **kwargs: None
sys.modules.setdefault("pyvista", pyvista_stub)
pandas_stub = types.ModuleType("pandas")
pandas_stub.read_csv = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pandas.read_csv stub called"))
sys.modules.setdefault("pandas", pandas_stub)


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADBSAT_PY_DIR = ROOT / "ADBSat-PyVersion"
UPDATE_PARAMETER_DIR = ROOT / "update_parameter_file"
if str(ADBSAT_PY_DIR) not in sys.path:
    sys.path.insert(0, str(ADBSAT_PY_DIR))
if str(UPDATE_PARAMETER_DIR) not in sys.path:
    sys.path.insert(0, str(UPDATE_PARAMETER_DIR))

from PICLas import PiclasSimulator
import update_parameter as piclas_update
from calc.environment import environment as adbsat_environment


class _FakeCenters:
    def __init__(self, points):
        self.points = points


class _FakeMesh:
    def __init__(self, force_per_area, centers):
        self.cell_data = {"Total_ForcePerArea": force_per_area}
        self.point_data = {}
        self._centers = centers

    def cell_centers(self):
        return _FakeCenters(self._centers)


class TestPiclasQoIAndEnvironmentConsistency(unittest.TestCase):
    def _assert_update_parameter_geometry(self, geometry_id: str, mesh_name: str, source_name: str):
        payload = {
            "geometry_id": geometry_id,
            "geometry_name": geometry_id,
            "hf_mesh": mesh_name,
            "rho": [1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e10, 6.0e9, 7.0e9, 8.0e9, 9.0e9, 1.0e-9],
            "Tinf": 900.0,
        }

        with tempfile.TemporaryDirectory() as td:
            ini_path = os.path.join(td, "parameter.ini")
            payload_path = os.path.join(td, "payload.json")
            shutil.copyfile(UPDATE_PARAMETER_DIR / "parameter.ini", ini_path)
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            piclas_update.update_ini_from_csv(200, 0.0, 0, ini_path, env_payload_path=payload_path)

            ini_text = pathlib.Path(ini_path).read_text(encoding="utf-8")

        self.assertIn(f"MeshFile = {mesh_name}", ini_text)
        self.assertIn(f"ProjectName     = {geometry_id}", ini_text)
        self.assertIn(f"Part-Boundary3-SourceName  = {source_name}", ini_text)

    def _assert_update_parameter_debug_json(self, geometry_id: str, mesh_name: str, source_name: str):
        payload = {
            "geometry_id": geometry_id,
            "geometry_name": geometry_id,
            "hf_mesh": mesh_name,
            "rho": [1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e10, 6.0e9, 7.0e9, 8.0e9, 9.0e9, 1.0e-9],
            "Tinf": 900.0,
        }

        with tempfile.TemporaryDirectory() as td:
            ini_path = os.path.join(td, "parameter.ini")
            payload_path = os.path.join(td, "payload.json")
            debug_path = os.path.join(td, "update_debug.json")
            shutil.copyfile(UPDATE_PARAMETER_DIR / "parameter.ini", ini_path)
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                piclas_update.update_ini_from_csv(
                    200,
                    0.0,
                    0,
                    ini_path,
                    env_payload_path=payload_path,
                    debug_print=True,
                    debug_json=debug_path,
                )

            debug_payload = json.loads(pathlib.Path(debug_path).read_text(encoding="utf-8"))

        self.assertIn("[update_parameter DEBUG]", stdout.getvalue())
        self.assertEqual(geometry_id, debug_payload["geometry_id"])
        self.assertEqual(mesh_name, debug_payload["resolved_mesh_file"])
        self.assertEqual(geometry_id, debug_payload["resolved_project_name"])
        self.assertEqual(source_name, debug_payload["resolved_boundary3_source_name"])

    def _assert_prepare_simulation_folder_geometry(self, geometry_id: str, mesh_name: str, source_name: str):
        with tempfile.TemporaryDirectory() as td:
            update_dir = os.path.join(td, "update_parameter_file")
            piclas_dir = os.path.join(td, "piclas")
            os.makedirs(update_dir, exist_ok=True)
            os.makedirs(piclas_dir, exist_ok=True)

            pathlib.Path(os.path.join(update_dir, "parameter.ini")).write_text(
                "MeshFile = Cube_mesh.h5  ! (relative) path to meshfile\n"
                "ProjectName     = Cube    ! Name of the current simulation\n"
                "Part-Boundary3-SourceName  = CUBE\n",
                encoding="utf-8",
            )
            pathlib.Path(os.path.join(update_dir, "dyn_p.txt")).write_text("1.0\n", encoding="utf-8")

            for filename in ["DSMC1.ini", "piclas", "piclas2vtk", mesh_name]:
                pathlib.Path(os.path.join(piclas_dir, filename)).write_text("stub\n", encoding="utf-8")

            sim = PiclasSimulator(
                update_script="python update_parameter.py",
                update_dir=update_dir,
                piclas_dir=piclas_dir,
                mpi_procs=1,
            )

            with patch("PICLas.subprocess.run", return_value=None):
                job_subdir = sim.prepare_simulation_folder(
                    200,
                    0.0,
                    0,
                    geometry_id=geometry_id,
                    geometry_mesh=mesh_name,
                )

            ini_text = pathlib.Path(job_subdir, "parameter.ini").read_text(encoding="utf-8")

            self.assertTrue(pathlib.Path(job_subdir, mesh_name).exists())
            self.assertFalse(pathlib.Path(job_subdir, "Cube_mesh.h5").exists() and mesh_name != "Cube_mesh.h5")
            self.assertIn(f"MeshFile = {mesh_name}", ini_text)
            self.assertIn(f"ProjectName     = {geometry_id}", ini_text)
            self.assertIn(f"Part-Boundary3-SourceName  = {source_name}", ini_text)
            self.assertEqual(f"{geometry_id}_DSMCSurfState_000.00*", sim._surface_state_glob_pattern(job_subdir=job_subdir))
            self.assertEqual(f"{geometry_id}_visuSurf_000.00*", sim._visu_surface_glob_pattern(job_subdir=job_subdir))

    def _assert_prepare_simulation_folder_debug_json(self, geometry_id: str, mesh_name: str, source_name: str):
        with tempfile.TemporaryDirectory() as td:
            update_dir = os.path.join(td, "update_parameter_file")
            piclas_dir = os.path.join(td, "piclas")
            os.makedirs(update_dir, exist_ok=True)
            os.makedirs(piclas_dir, exist_ok=True)

            pathlib.Path(os.path.join(update_dir, "parameter.ini")).write_text(
                "MeshFile = Cube_mesh.h5  ! (relative) path to meshfile\n"
                "ProjectName     = Cube    ! Name of the current simulation\n"
                "Part-Boundary3-SourceName  = CUBE\n",
                encoding="utf-8",
            )
            pathlib.Path(os.path.join(update_dir, "dyn_p.txt")).write_text("1.0\n", encoding="utf-8")

            for filename in ["DSMC1.ini", "piclas", "piclas2vtk", mesh_name]:
                pathlib.Path(os.path.join(piclas_dir, filename)).write_text("stub\n", encoding="utf-8")

            sim = PiclasSimulator(
                update_script="python update_parameter.py",
                update_dir=update_dir,
                piclas_dir=piclas_dir,
                mpi_procs=1,
                debug_geometry=True,
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout), patch("PICLas.subprocess.run", return_value=None):
                job_subdir = sim.prepare_simulation_folder(
                    200,
                    0.0,
                    0,
                    geometry_id=geometry_id,
                    geometry_mesh=mesh_name,
                )

            debug_dir = pathlib.Path(piclas_dir, "debug_geometry")
            debug_path = debug_dir / f"{pathlib.Path(job_subdir).name}_piclas_prepare.json"
            debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))

        self.assertIn("[PICLas DEBUG]", stdout.getvalue())
        self.assertEqual(geometry_id, debug_payload["geometry_id"])
        self.assertEqual(mesh_name, debug_payload["resolved_mesh_filename"])
        self.assertEqual(geometry_id, debug_payload["resolved_project_name"])
        self.assertEqual(source_name, debug_payload["resolved_boundary3_source_name"])
        self.assertTrue(debug_payload["debug_update_json"].endswith("_update_parameter.json"))

    def test_update_parameter_keeps_cube_boundary_source_for_cube_payload(self):
        self._assert_update_parameter_geometry("Cube", "Cube_mesh.h5", "CUBE")

    def test_update_parameter_sets_mesh_and_project_for_goce_payload(self):
        self._assert_update_parameter_geometry("GOCE", "GOCE_mesh.h5", "OBJ")

    def test_update_parameter_sets_mesh_and_project_for_soar_payload(self):
        self._assert_update_parameter_geometry("SOAR", "SOAR_mesh.h5", "OBJ")

    def test_update_parameter_sets_mesh_and_project_for_champ_payload(self):
        self._assert_update_parameter_geometry("CHAMP", "CHAMP_mesh.h5", "OBJ")

    def test_update_parameter_applies_numerical_controls_from_payload(self):
        payload = {
            "geometry_id": "Cube",
            "geometry_name": "Cube",
            "hf_mesh": "Cube_mesh.h5",
            "rho": [1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e10, 6.0e9, 7.0e9, 8.0e9, 9.0e9, 1.0e-9],
            "Tinf": 900.0,
            "manual_timestep_s": 2.5e-7,
            "macro_particle_factor": 12345.0,
            "sampling_iterations": 4321,
            "octree_part_num_node": 99,
            "octree_part_num_node_min": 77,
            "particles_mpi_weight": 2222,
        }

        with tempfile.TemporaryDirectory() as td:
            ini_path = os.path.join(td, "parameter.ini")
            payload_path = os.path.join(td, "payload.json")
            shutil.copyfile(UPDATE_PARAMETER_DIR / "parameter.ini", ini_path)
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            piclas_update.update_ini_from_csv(200, 0.0, 0, ini_path, env_payload_path=payload_path)
            ini_text = pathlib.Path(ini_path).read_text(encoding="utf-8")

        self.assertIn("ManualTimeStep        = 2.500000E-07", ini_text)
        self.assertIn("Part-Species$-MacroParticleFactor = 1.23450E+04", ini_text)
        self.assertIn("Part-IterationForMacroVal         = 4321", ini_text)
        self.assertIn("Particles-OctreePartNumNode        = 99", ini_text)
        self.assertIn("Particles-OctreePartNumNodeMin     = 77", ini_text)
        self.assertIn("Particles-MPIWeight                      = 2222", ini_text)

    def test_update_parameter_debug_json_for_soar(self):
        self._assert_update_parameter_debug_json("SOAR", "SOAR_mesh.h5", "OBJ")

    def test_update_parameter_debug_json_for_champ(self):
        self._assert_update_parameter_debug_json("CHAMP", "CHAMP_mesh.h5", "OBJ")

    def test_piclas_query_job_states_parses_exact_job_ids(self):
        sim = PiclasSimulator()
        squeue_result = subprocess.CompletedProcess(
            args=["squeue"],
            returncode=0,
            stdout="1234 R\n123 CG\n",
            stderr="",
        )

        with patch("PICLas.subprocess.run", return_value=squeue_result):
            states = sim._query_job_states(["123"])

        self.assertEqual("CG", states.get("123"))
        self.assertEqual("R", states.get("1234"))
        self.assertNotIn("12", states)

    def test_piclas_wait_for_job_completion_breaks_on_cg(self):
        sim = PiclasSimulator()
        squeue_result = subprocess.CompletedProcess(
            args=["squeue"],
            returncode=0,
            stdout="123 CG\n",
            stderr="",
        )

        with patch("PICLas.subprocess.run", return_value=squeue_result) as run_mock, patch(
            "PICLas.time.sleep", return_value=None
        ) as sleep_mock:
            sim._wait_for_job_completion("123", poll_interval=1)

        self.assertEqual(1, run_mock.call_count)
        sleep_mock.assert_not_called()

    def test_piclas_wait_for_job_completion_polls_through_pd_and_r_until_missing(self):
        sim = PiclasSimulator()
        squeue_results = [
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="123 PD\n", stderr=""),
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="123 R\n", stderr=""),
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="", stderr=""),
        ]

        with patch("PICLas.subprocess.run", side_effect=squeue_results) as run_mock, patch(
            "PICLas.time.sleep", return_value=None
        ) as sleep_mock:
            sim._wait_for_job_completion("123", poll_interval=1)

        self.assertEqual(3, run_mock.call_count)
        self.assertEqual(2, sleep_mock.call_count)

    def test_piclas_wait_for_job_completion_keeps_waiting_for_other_listed_states(self):
        sim = PiclasSimulator()
        squeue_results = [
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="123 CF\n", stderr=""),
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="", stderr=""),
        ]

        with patch("PICLas.subprocess.run", side_effect=squeue_results) as run_mock, patch(
            "PICLas.time.sleep", return_value=None
        ) as sleep_mock:
            sim._wait_for_job_completion("123", poll_interval=1)

        self.assertEqual(2, run_mock.call_count)
        self.assertEqual(1, sleep_mock.call_count)

    def test_piclas_wait_for_all_jobs_completion_handles_pd_r_cg_and_missing(self):
        sim = PiclasSimulator()
        squeue_results = [
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="1 PD\n2 R\n3 CG\n", stderr=""),
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="2 R\n", stderr=""),
            subprocess.CompletedProcess(args=["squeue"], returncode=0, stdout="", stderr=""),
        ]

        with patch("PICLas.subprocess.run", side_effect=squeue_results) as run_mock, patch(
            "PICLas.time.sleep", return_value=None
        ) as sleep_mock:
            sim._wait_for_all_jobs_completion(["1", "2", "3"], poll_interval=1)

        self.assertEqual(3, run_mock.call_count)
        self.assertEqual(2, sleep_mock.call_count)

    def test_piclas_retries_missing_h5_outputs_once_and_recovers(self):
        with tempfile.TemporaryDirectory() as td:
            ok_dir = os.path.join(td, "job_ok")
            retry_dir = os.path.join(td, "job_retry")
            os.makedirs(ok_dir, exist_ok=True)
            os.makedirs(retry_dir, exist_ok=True)

            for idx in range(4):
                open(os.path.join(ok_dir, f"Cube_DSMCSurfState_000.00{idx}.h5"), "w", encoding="utf-8").close()
            with open(os.path.join(retry_dir, "job_piclas.sh"), "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\n")

            sim = PiclasSimulator(job_template="job_piclas.sh")
            wait_calls = {"count": 0}

            def fake_wait(job_ids, poll_interval=60):
                wait_calls["count"] += 1
                if wait_calls["count"] == 2:
                    for idx in range(4):
                        open(
                            os.path.join(retry_dir, f"Cube_DSMCSurfState_000.00{idx}.h5"),
                            "w",
                            encoding="utf-8",
                        ).close()

            with patch.object(sim, "_wait_for_all_jobs_completion", side_effect=fake_wait), patch.object(
                sim, "submit_simulation_job", return_value="retry-1"
            ) as submit_mock, patch("PICLas.time.sleep", return_value=None):
                sim._wait_for_jobs_and_retry_failed_outputs([ok_dir, retry_dir], ["job-1", "job-2"], max_retries=2)

        self.assertEqual(2, wait_calls["count"])
        self.assertEqual(1, submit_mock.call_count)

    def test_piclas_tpmc_still_requires_configured_surface_state_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            job_dir = os.path.join(td, "job_tpmc")
            os.makedirs(job_dir, exist_ok=True)
            open(os.path.join(job_dir, "Cube_DSMCSurfState_000.000.h5"), "w", encoding="utf-8").close()

            sim = PiclasSimulator(job_template="job_piclas.sh", piclas_mode="tpmc")

            self.assertEqual(4, sim.required_surface_state_files)
            self.assertFalse(sim._has_required_surface_state_outputs(job_dir))

    def test_piclas_retries_missing_h5_outputs_at_most_twice(self):
        with tempfile.TemporaryDirectory() as td:
            retry_dir = os.path.join(td, "job_retry")
            os.makedirs(retry_dir, exist_ok=True)
            with open(os.path.join(retry_dir, "job_piclas.sh"), "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\n")

            sim = PiclasSimulator(job_template="job_piclas.sh")

            with patch.object(sim, "_wait_for_all_jobs_completion", return_value=None), patch.object(
                sim, "submit_simulation_job", side_effect=["retry-1", "retry-2"]
            ) as submit_mock, patch("PICLas.time.sleep", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Maximale Anzahl an PICLas-Retries erreicht"):
                    sim._wait_for_jobs_and_retry_failed_outputs([retry_dir], ["job-1"], max_retries=2)

        self.assertEqual(2, submit_mock.call_count)

    def test_pymsis_row_nans_are_zeroed_in_update_parameter(self):
        fake_pymsis = types.SimpleNamespace(
            calculate=lambda *args, **kwargs: np.asarray(
                [[1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0, 11.0]],
                dtype=float,
            )
        )
        with patch.dict(sys.modules, {"pymsis": fake_pymsis}):
            row = piclas_update._sample_pymsis_row(
                {"environment_model": "pymsis_hwm14", "altitude_km": 200, "datetime_utc": "2014-03-20T06:00:00"},
                200,
            )

        self.assertEqual(11, row.shape[0])
        self.assertEqual(0.0, float(row[1]))
        self.assertEqual(0.0, float(row[5]))

    def test_pymsis_submodule_run_api_is_supported_in_update_parameter(self):
        fake_pymsis = types.SimpleNamespace(
            msis=types.SimpleNamespace(
                run=lambda *args, **kwargs: np.asarray(
                    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
                    dtype=float,
                )
            )
        )
        with patch.dict(sys.modules, {"pymsis": fake_pymsis}):
            row = piclas_update._sample_pymsis_row(
                {"environment_model": "pymsis_hwm14", "altitude_km": 200, "datetime_utc": "2014-03-20T06:00:00"},
                200,
            )

        self.assertEqual(11, row.shape[0])
        self.assertEqual(2.0, float(row[1]))
        self.assertEqual(11.0, float(row[10]))

    def test_payload_rho_nans_are_zeroed_in_update_parameter(self):
        atmosphere = piclas_update._atmosphere_from_payload(
            {
                "rho": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
                "Tinf": 900.0,
            },
            200,
        )

        self.assertIsNotNone(atmosphere)
        self.assertEqual(0.0, float(atmosphere[3]))
        self.assertEqual(0.0, float(atmosphere[5]))

    def test_piclas_collect_results_qois_from_vector_force(self):
        force_per_area = np.asarray([[0.0, -1.0, 0.5], [0.0, -2.0, 1.5]], dtype=float)
        centers = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        expected_drag = 1.5
        expected_lift = 1.0
        expected_cm = np.asarray([0.5, -0.125, -0.25], dtype=float)

        with tempfile.TemporaryDirectory() as td:
            for name in ["output1.vtu", "output2.vtu", "output3.vtu", "output4.vtu"]:
                open(os.path.join(td, name), "w", encoding="utf-8").close()
            with open(os.path.join(td, "dyn_p.txt"), "w", encoding="utf-8") as f:
                f.write("2.0\n")
            with open(os.path.join(td, "cpu_time.txt"), "w", encoding="utf-8") as f:
                f.write("3600000\n")

            sim = PiclasSimulator(mpi_procs=4)
            with patch("PICLas.cell_areas_and_total", return_value=(np.asarray([2.0, 2.0], dtype=float), 4.0)), patch(
                "PICLas.pv.read", return_value=_FakeMesh(force_per_area, centers)
            ):
                qois, cpu_h = sim.collect_results_qois([td], AoS=[0.0], AoA=[0.0])

        self.assertAlmostEqual(expected_drag, qois["C_D"][0], places=12)
        self.assertAlmostEqual(expected_drag * expected_drag, qois["C_D2"][0], places=12)
        self.assertAlmostEqual(expected_lift, qois["C_L"][0], places=12)
        self.assertAlmostEqual(expected_cm[0], qois["C_Mx"][0], places=12)
        self.assertAlmostEqual(expected_cm[1], qois["C_My"][0], places=12)
        self.assertAlmostEqual(expected_cm[2], qois["C_Mz"][0], places=12)
        self.assertEqual([4.0], cpu_h)

    def test_prepare_simulation_folder_uses_geometry_specific_mesh_and_project(self):
        self._assert_prepare_simulation_folder_geometry("GOCE", "GOCE_mesh.h5", "OBJ")

    def test_prepare_simulation_folder_uses_soar_geometry_specific_mesh_and_project(self):
        self._assert_prepare_simulation_folder_geometry("SOAR", "SOAR_mesh.h5", "OBJ")

    def test_prepare_simulation_folder_uses_champ_geometry_specific_mesh_and_project(self):
        self._assert_prepare_simulation_folder_geometry("CHAMP", "CHAMP_mesh.h5", "OBJ")

    def test_prepare_simulation_folder_debug_json_for_goce(self):
        self._assert_prepare_simulation_folder_debug_json("GOCE", "GOCE_mesh.h5", "OBJ")

    def test_piclas_and_adbsat_use_same_payload_atmosphere_row(self):
        payload = {
            "environment_model": "pymsis_hwm14",
            "altitude_km": 200,
            "atmosphere_row": [
                9.1e-10,
                1.1e12,
                2.2e12,
                3.3e12,
                4.4e12,
                5.5e10,
                6.6e10,
                7.7e10,
                8.8e10,
                9.9e10,
                987.0,
            ],
        }

        atmosphere = piclas_update._resolve_atmosphere(200, 0, payload, "pymsis_hwm14")
        piclas_rho = np.asarray(
            [
                atmosphere[4],
                atmosphere[3],
                atmosphere[1],
                atmosphere[2],
                atmosphere[6],
                atmosphere[5],
                atmosphere[7],
                atmosphere[8],
                atmosphere[9],
                atmosphere[0],
            ],
            dtype=float,
        )
        adbsat_param = adbsat_environment({}, None, 0, 200000.0, env_payload=payload, env_model="pymsis_hwm14")

        np.testing.assert_allclose(piclas_rho, adbsat_param["rho"])
        self.assertAlmostEqual(float(atmosphere[10]), float(adbsat_param["Tinf"]), places=12)

        constants = piclas_update.ConstantsData()
        total_density = float(np.sum(piclas_rho[:9]))
        mmean = (
            piclas_rho[0] * constants.mHe
            + piclas_rho[1] * constants.mO
            + piclas_rho[2] * constants.mN2
            + piclas_rho[3] * constants.mO2
            + piclas_rho[4] * constants.mAr
            + piclas_rho[5] * constants.mH
            + piclas_rho[6] * constants.mN
            + piclas_rho[7] * constants.mAnO
            + piclas_rho[8] * constants.mNO
        ) / total_density
        expected_vinf = np.sqrt(constants.mu_E / (constants.R_E + 200000.0))
        expected_vth = np.sqrt(2 * constants.kb * float(atmosphere[10]) / (mmean / constants.NA / 1000))

        self.assertAlmostEqual(expected_vinf, float(adbsat_param["vinf"]), places=9)
        self.assertAlmostEqual(expected_vth, float(adbsat_param["vth"]), places=9)


if __name__ == "__main__":
    unittest.main()
