import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from ADBSat import ADBSatSimulator
from ADBSat_prandtl import ADBSatSimulator as PrandtlADBSatSimulator
from mfmc_campaign.adapters import BaseModelAdapter, LegacyADBSatAdapter


class TestADBSatCoefficientQoIs(unittest.TestCase):
    def _write_results(self, base_dir, header, row):
        result_dir = pathlib.Path(base_dir, "MFMC_Jobs_SENTMAN")
        result_dir.mkdir(parents=True)
        pathlib.Path(result_dir, "all_results.txt").write_text(
            f"{header}\n{row}\n",
            encoding="utf-8",
        )

    def test_coefficient_output_is_returned_by_both_wrappers(self):
        header = "gsi_model idx C_D C_L C_Y C_Mx C_My C_Mz cpu_time_ms"
        row = "SENTMAN 7 2.0 -0.5 0.25 0.1 -0.2 0.3 3600000"

        for simulator_type in (ADBSatSimulator, PrandtlADBSatSimulator):
            with self.subTest(simulator=simulator_type.__module__), tempfile.TemporaryDirectory() as td:
                self._write_results(td, header, row)
                sim = simulator_type(method="SENTMAN", base_dir=td)
                qois, costs, indices = sim.analyze_simulation_results_qois(
                    [7], requested_qois=["C_D", "C_D2", "C_L", "C_Mz", "C_Y", "C_Y2"]
                )

                np.testing.assert_allclose(qois["C_D"], [2.0])
                np.testing.assert_allclose(qois["C_D2"], [4.0])
                np.testing.assert_allclose(qois["C_L"], [-0.5])
                np.testing.assert_allclose(qois["C_Mz"], [0.3])
                np.testing.assert_allclose(qois["C_Y"], [0.25])
                np.testing.assert_allclose(qois["C_Y2"], [0.0625])
                np.testing.assert_allclose(costs, [1.0])
                np.testing.assert_array_equal(indices, [7])

    def test_q_inf_column_does_not_change_coefficients(self):
        header = "gsi_model idx C_D C_L C_Y C_Mx C_My C_Mz q_inf cpu_time_ms"
        row = "SENTMAN 3 2.0 -0.5 0.25 0.1 -0.2 0.3 4.0 7200000"

        with tempfile.TemporaryDirectory() as td:
            self._write_results(td, header, row)
            sim = ADBSatSimulator(method="SENTMAN", base_dir=td)
            qois, costs, indices = sim.analyze_simulation_results_qois(
                [3], requested_qois=["C_D", "C_L", "C_Y", "C_Mz"]
            )

            np.testing.assert_allclose(qois["C_D"], [2.0])
            np.testing.assert_allclose(qois["C_L"], [-0.5])
            np.testing.assert_allclose(qois["C_Y"], [0.25])
            np.testing.assert_allclose(qois["C_Mz"], [0.3])
            np.testing.assert_allclose(costs, [2.0])
            np.testing.assert_array_equal(indices, [3])

    def test_explicit_result_directories_isolate_parallel_campaigns(self):
        header = "gsi_model idx C_D C_L C_Y C_Mx C_My C_Mz cpu_time_ms"
        with tempfile.TemporaryDirectory() as td:
            sim = ADBSatSimulator(method="SENTMAN", base_dir=td)
            result_dirs = [pathlib.Path(td, "campaign_a"), pathlib.Path(td, "campaign_b")]
            for result_dir, cd in zip(result_dirs, (1.25, 2.75)):
                result_dir.mkdir()
                pathlib.Path(result_dir, "all_results.txt").write_text(
                    f"{header}\nSENTMAN 0 {cd} 0 0 0 0 0 1000\n",
                    encoding="utf-8",
                )

            first, _, _ = sim.analyze_simulation_results([0], result_dir=result_dirs[0])
            second, _, _ = sim.analyze_simulation_results([0], result_dir=result_dirs[1])

            np.testing.assert_allclose(first, [1.25])
            np.testing.assert_allclose(second, [2.75])

    def test_legacy_drag_only_coefficients_are_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_results(
                td,
                "gsi_model idx C_D cpu_time_ms",
                "SENTMAN 1 2.0 1000",
            )
            sim = ADBSatSimulator(method="SENTMAN", base_dir=td)
            values, costs, indices = sim.analyze_simulation_results([1])
            np.testing.assert_allclose(values, [2.0])
            np.testing.assert_allclose(costs, [1000.0 / 3600000.0])
            np.testing.assert_array_equal(indices, [1])

    def test_lateral_coefficient_can_be_exposed_as_campaign_cl(self):
        class DummySimulator:
            def analyze_simulation_results_qois(self, run_ids, requested_qois=None):
                self.requested_qois = requested_qois
                return {"C_Y": [0.25], "C_Y2": [0.0625]}, [1.5], [run_ids[0]]

        adapter = object.__new__(LegacyADBSatAdapter)
        BaseModelAdapter.__init__(adapter, "Sentman", "lf", ["C_L", "C_L2"])
        adapter.method = "Sentman"
        adapter.qoi_aliases = {"C_L": "C_Y", "C_L2": "C_Y2"}
        adapter.surface_mapping = None
        adapter.sim = DummySimulator()

        result = adapter._collect_adbsat_results(
            SimpleNamespace(qois=["C_L", "C_L2"], sample_ids=["sample-7"]),
            [7],
        )

        self.assertEqual(["C_Y", "C_Y2"], adapter.sim.requested_qois)
        np.testing.assert_allclose(result.values_by_qoi["C_L"], [0.25])
        np.testing.assert_allclose(result.values_by_qoi["C_L2"], [0.0625])


if __name__ == "__main__":
    unittest.main()
