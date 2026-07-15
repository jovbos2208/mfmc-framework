import unittest
import json
from unittest.mock import patch

import numpy as np

from mfmc_campaign.adapters import (
    BaseModelAdapter,
    LegacyPiclasAdapter,
    _angles_from_flow_vector,
    _estimate_freestream_speed_mps,
    _flow_unit_from_angles,
    build_adapter_registry,
    make_request,
)


class TestPiclasAdapterAttitude(unittest.TestCase):
    def test_legacy_piclas_tpmc_can_be_registered_as_lf(self):
        captured_kwargs = []

        class DummyPiclasSim:
            def __init__(self, **kwargs):
                captured_kwargs.append(dict(kwargs))

        class DummyModule:
            PiclasSimulator = DummyPiclasSim

        cfg = {
            "execution": {"backend": "legacy_slurm"},
            "qois": {"direct": ["C_D", "C_D2"]},
            "models": {
                "hf": {"id": "PICLas_HF", "kind": "legacy_piclas", "kwargs": {"simulator_module": "dummy_piclas"}},
                "lf": [
                    {
                        "id": "PICLas_TPMC",
                        "kind": "legacy_piclas_tpmc",
                        "kwargs": {"simulator_module": "dummy_piclas", "mpi_procs": 12},
                    }
                ],
                "available_qois": {
                    "PICLas_HF": ["C_D", "C_D2"],
                    "PICLas_TPMC": ["C_D", "C_D2"],
                },
            },
        }

        with patch("mfmc_campaign.adapters.importlib.import_module", return_value=DummyModule):
            registry = build_adapter_registry(cfg)

        self.assertIsInstance(registry.lfs["PICLas_TPMC"], LegacyPiclasAdapter)
        self.assertAlmostEqual(1.0e-3, float(registry.hf.payload_defaults["t_end_s"]), places=12)
        self.assertEqual(2500, int(registry.hf.payload_defaults["sampling_iterations"]))
        self.assertEqual("lf", registry.lfs["PICLas_TPMC"].fidelity)
        self.assertEqual("tpmc", captured_kwargs[1]["piclas_mode"])
        self.assertEqual(12, captured_kwargs[1]["mpi_procs"])
        self.assertAlmostEqual(1.0e-4, float(registry.lfs["PICLas_TPMC"].payload_defaults["t_end_s"]), places=12)
        self.assertEqual(250, int(registry.lfs["PICLas_TPMC"].payload_defaults["sampling_iterations"]))
        self.assertNotIn("macro_particle_factor_scale", registry.lfs["PICLas_TPMC"].payload_defaults)

    def test_legacy_piclas_dsmc_payload_defaults_reset_tend_after_tpmc(self):
        captured_kwargs = []

        class DummyPiclasSim:
            def __init__(self, **kwargs):
                captured_kwargs.append(dict(kwargs))

        class DummyModule:
            PiclasSimulator = DummyPiclasSim

        with patch("mfmc_campaign.adapters.importlib.import_module", return_value=DummyModule):
            adapter = LegacyPiclasAdapter(
                model_id="PICLas_HF",
                available_qois=["C_D"],
                kwargs={"simulator_module": "dummy_piclas"},
                fidelity="hf",
            )

        self.assertEqual("dsmc", adapter.payload_defaults["piclas_mode"])
        self.assertAlmostEqual(1.0e-3, float(adapter.payload_defaults["t_end_s"]), places=12)
        self.assertEqual(2500, int(adapter.payload_defaults["sampling_iterations"]))
        self.assertNotIn("piclas_mode", captured_kwargs[0])

    def test_legacy_piclas_tpmc_payload_defaults_set_tend_and_sampling_iterations(self):
        adapter = object.__new__(LegacyPiclasAdapter)
        BaseModelAdapter.__init__(adapter, model_id="PICLas_TPMC", fidelity="lf", available_qois=["C_D"])
        adapter.submission_batch_size = None
        adapter.piclas_mode = "tpmc"
        adapter.payload_defaults = {
            "piclas_mode": "tpmc",
            "t_end_s": 1.0e-4,
            "sampling_iterations": 250,
        }
        captured = {}

        class DummyPiclasSim:
            def submit_batch_jobs(
                self,
                altitude,
                aos,
                indices,
                env_payload_paths=None,
                env_model=None,
                aos_values=None,
                aoa_values=None,
                random_seeds=None,
                geometry_id=None,
                geometry_mesh=None,
                flow_zero_direction=None,
            ):
                with open(env_payload_paths[0], "r", encoding="utf-8") as handle:
                    captured["payload"] = json.load(handle)
                return {"job_ids": [], "job_subdirs": []}

            def wait_for_batch_jobs(self, batch_handle, max_retries=2):
                return None

            def submit_batch_postprocessing(self, handles, random_seed, wait_for_completion=False):
                return {"job_id": "0"}

            def wait_for_postprocessing(self, postprocess_handle):
                return None

            def collect_batch_results(self, batch_handle, requested_qois=None):
                return {"C_D": [0.5]}, [1.0]

        adapter.sim = DummyPiclasSim()
        req = make_request(
            study_id="s",
            cell_id="c",
            model_id="PICLas_TPMC",
            fidelity="lf",
            qois=["C_D"],
            geometry={"id": "Cube", "name": "Cube"},
            regime={"id": "r", "label": "r", "descriptors": {"altitude_km": 200}},
            active_source_blocks=[],
            sample_ids=["a"],
            samples=[
                {
                    "database_index": 2,
                    "aos_deg": 0.0,
                    "aoa_deg": 0.0,
                    "t_end_s": 1.0e-36,
                    "sampling_iterations": 1,
                }
            ],
            seed=7,
            metadata={"aos_deg": 0.0},
        )

        adapter.evaluate(req)
        self.assertEqual("tpmc", captured["payload"]["piclas_mode"])
        self.assertAlmostEqual(1.0e-4, float(captured["payload"]["t_end_s"]), places=12)
        self.assertEqual(250, int(captured["payload"]["sampling_iterations"]))
        self.assertNotIn("macro_particle_factor_scale", captured["payload"])
        self.assertNotIn("t_end_scale", captured["payload"])
        self.assertNotIn("t_end_s", captured["payload"]["sample"])
        self.assertNotIn("sampling_iterations", captured["payload"]["sample"])

    def test_legacy_piclas_hf_ignores_sample_piclas_numerical_controls(self):
        adapter = object.__new__(LegacyPiclasAdapter)
        BaseModelAdapter.__init__(adapter, model_id="PICLas_HF", fidelity="hf", available_qois=["C_D"])
        adapter.submission_batch_size = None
        adapter.payload_defaults = {}
        captured = {}

        class DummyPiclasSim:
            def submit_batch_jobs(
                self,
                altitude,
                aos,
                indices,
                env_payload_paths=None,
                env_model=None,
                aos_values=None,
                aoa_values=None,
                random_seeds=None,
                geometry_id=None,
                geometry_mesh=None,
                flow_zero_direction=None,
            ):
                with open(env_payload_paths[0], "r", encoding="utf-8") as handle:
                    captured["payload"] = json.load(handle)
                return {"job_ids": [], "job_subdirs": []}

            def wait_for_batch_jobs(self, batch_handle, max_retries=2):
                return None

            def submit_batch_postprocessing(self, handles, random_seed, wait_for_completion=False):
                return {"job_id": "0"}

            def wait_for_postprocessing(self, postprocess_handle):
                return None

            def collect_batch_results(self, batch_handle, requested_qois=None):
                return {"C_D": [0.5]}, [1.0]

        adapter.sim = DummyPiclasSim()
        req = make_request(
            study_id="s",
            cell_id="c",
            model_id="PICLas_HF",
            fidelity="hf",
            qois=["C_D"],
            geometry={"id": "Cube", "name": "Cube"},
            regime={"id": "r", "label": "r", "descriptors": {"altitude_km": 200}},
            active_source_blocks=[],
            sample_ids=["a"],
            samples=[
                {
                    "database_index": 2,
                    "aos_deg": 0.0,
                    "aoa_deg": 0.0,
                    "t_end_s": 1.0e-36,
                    "t_end_scale": 1.0e-33,
                    "sampling_iterations": 1,
                }
            ],
            seed=7,
            metadata={"aos_deg": 0.0},
        )

        adapter.evaluate(req)
        self.assertNotIn("t_end_s", captured["payload"])
        self.assertNotIn("t_end_scale", captured["payload"])
        self.assertNotIn("sampling_iterations", captured["payload"])
        self.assertNotIn("t_end_s", captured["payload"]["sample"])
        self.assertNotIn("t_end_scale", captured["payload"]["sample"])
        self.assertNotIn("sampling_iterations", captured["payload"]["sample"])

    def test_passes_samplewise_aos_aoa_to_simulator(self):
        adapter = object.__new__(LegacyPiclasAdapter)
        BaseModelAdapter.__init__(adapter, model_id="PICLas_HF", fidelity="hf", available_qois=["C_D", "C_D2"])
        captured = {}

        class DummyPiclasSim:
            pass

        def fake_run_batch_qois(
            altitude,
            aos,
            indices,
            random_seed,
            requested_qois=None,
            env_payload_paths=None,
            env_model=None,
            aos_values=None,
            aoa_values=None,
            geometry_id=None,
            geometry_mesh=None,
        ):
            captured["aos_values"] = list(aos_values or [])
            captured["aoa_values"] = list(aoa_values or [])
            captured["geometry_id"] = geometry_id
            captured["geometry_mesh"] = geometry_mesh
            q = {"C_D": [0.5, 0.6], "C_D2": [0.25, 0.36]}
            return q, [1.0, 1.2]

        adapter.sim = DummyPiclasSim()
        adapter.sim.run_batch_qois = fake_run_batch_qois

        req = make_request(
            study_id="s",
            cell_id="c",
            model_id="PICLas_HF",
            fidelity="hf",
            qois=["C_D", "C_D2"],
            geometry={"id": "Cube", "name": "Cube", "metadata": {"hf_mesh": "Cube_mesh.h5"}},
            regime={"id": "r", "label": "r", "descriptors": {"altitude_km": 200, "surface_state": "nominal"}},
            active_source_blocks=["attitude.aos", "attitude.aoa"],
            sample_ids=["a", "b"],
            samples=[
                {"database_index": 2, "aos_deg": 3.0, "aoa_deg": -1.0},
                {"database_index": 4, "aos_deg": -2.0, "aoa_deg": 0.5, "jitter_aos_deg": 0.2},
            ],
            seed=7,
            metadata={"aos_deg": 0.0, "reference_area_m2": 1.0},
        )

        res = adapter.evaluate(req)
        self.assertEqual([0.5, 0.6], res.values_by_qoi["C_D"])
        self.assertEqual([0.25, 0.36], res.values_by_qoi["C_D2"])
        self.assertEqual([3.0, -1.8], captured["aos_values"])
        self.assertEqual([-1.0, 0.5], captured["aoa_values"])
        self.assertEqual("Cube", captured["geometry_id"])
        self.assertEqual("Cube_mesh.h5", captured["geometry_mesh"])

    def test_wind_adjusts_samplewise_aos_aoa_before_piclas_run(self):
        adapter = object.__new__(LegacyPiclasAdapter)
        BaseModelAdapter.__init__(adapter, model_id="PICLas_HF", fidelity="hf", available_qois=["C_D"])
        adapter.submission_batch_size = None
        captured = {}

        class DummyPiclasSim:
            pass

        def fake_run_batch_qois(
            altitude,
            aos,
            indices,
            random_seed,
            requested_qois=None,
            env_payload_paths=None,
            env_model=None,
            aos_values=None,
            aoa_values=None,
            geometry_id=None,
            geometry_mesh=None,
        ):
            captured["aos_values"] = list(aos_values or [])
            captured["aoa_values"] = list(aoa_values or [])
            return {"C_D": [0.5]}, [1.0]

        adapter.sim = DummyPiclasSim()
        adapter.sim.run_batch_qois = fake_run_batch_qois

        req = make_request(
            study_id="s",
            cell_id="c",
            model_id="PICLas_HF",
            fidelity="hf",
            qois=["C_D"],
            geometry={"id": "Cube", "name": "Cube"},
            regime={"id": "r", "label": "r", "descriptors": {"altitude_km": 200}},
            active_source_blocks=["environment.winds", "attitude.dispersion"],
            sample_ids=["a"],
            samples=[
                {
                    "database_index": 2,
                    "aos_deg": 0.0,
                    "aoa_deg": 0.0,
                    "wind_east_mps": 100.0,
                    "wind_up_mps": 50.0,
                }
            ],
            seed=7,
            metadata={"aos_deg": 0.0},
        )

        adapter.evaluate(req)
        rel = _estimate_freestream_speed_mps(200.0) * _flow_unit_from_angles(0.0, 0.0) - np.array([100.0, 0.0, 50.0])
        expected = _angles_from_flow_vector(rel)
        self.assertIsNotNone(expected)
        self.assertAlmostEqual(expected["aos_deg"], captured["aos_values"][0], places=6)
        self.assertAlmostEqual(expected["aoa_deg"], captured["aoa_values"][0], places=6)

    def test_submission_batch_size_chunks_piclas_batches(self):
        adapter = object.__new__(LegacyPiclasAdapter)
        BaseModelAdapter.__init__(adapter, model_id="PICLas_HF", fidelity="hf", available_qois=["C_D", "C_D2"])
        adapter.submission_batch_size = 8
        captured = {"calls": []}

        class DummyPiclasSim:
            pass

        def fake_run_batch_qois(
            altitude,
            aos,
            indices,
            random_seed,
            requested_qois=None,
            env_payload_paths=None,
            env_model=None,
            aos_values=None,
            aoa_values=None,
            geometry_id=None,
            geometry_mesh=None,
        ):
            captured["calls"].append(list(indices))
            cds = [0.5 + 0.01 * idx for idx in range(len(indices))]
            return {"C_D": cds, "C_D2": [cd * cd for cd in cds]}, [1.0] * len(indices)

        adapter.sim = DummyPiclasSim()
        adapter.sim.run_batch_qois = fake_run_batch_qois

        samples = [{"database_index": idx, "aos_deg": 0.0, "aoa_deg": 0.0} for idx in range(10)]
        req = make_request(
            study_id="s",
            cell_id="c",
            model_id="PICLas_HF",
            fidelity="hf",
            qois=["C_D", "C_D2"],
            geometry={"id": "Cube", "name": "Cube"},
            regime={"id": "r", "label": "r", "descriptors": {"altitude_km": 200}},
            active_source_blocks=[],
            sample_ids=[f"s_{idx}" for idx in range(10)],
            samples=samples,
            seed=7,
            metadata={"aos_deg": 0.0},
        )

        res = adapter.evaluate(req)
        self.assertEqual([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]], captured["calls"])
        self.assertEqual(10, len(res.values_by_qoi["C_D"]))
        self.assertEqual(10, len(res.values_by_qoi["C_D2"]))
        self.assertEqual(10, len(res.costs))


if __name__ == "__main__":
    unittest.main()
