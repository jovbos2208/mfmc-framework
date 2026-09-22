import pathlib
import tempfile
import unittest

import numpy as np

from ADBSat import ADBSatSimulator
from ADBSat_prandtl import ADBSatSimulator as PrandtlADBSatSimulator


class TestADBSatPressureQoIs(unittest.TestCase):
    def _write_results(self, base_dir, header, row):
        result_dir = pathlib.Path(base_dir, "MFMC_Jobs_SENTMAN")
        result_dir.mkdir(parents=True)
        pathlib.Path(result_dir, "all_results.txt").write_text(
            f"{header}\n{row}\n",
            encoding="utf-8",
        )

    def test_pressure_output_is_returned_by_both_wrappers(self):
        header = "gsi_model idx P_D P_L P_Y P_Mx P_My P_Mz q_inf cpu_time_ms"
        row = "SENTMAN 7 6.0 -2.0 1.0 0.5 -0.25 0.125 3.0 3600000"

        for simulator_type in (ADBSatSimulator, PrandtlADBSatSimulator):
            with self.subTest(simulator=simulator_type.__module__), tempfile.TemporaryDirectory() as td:
                self._write_results(td, header, row)
                sim = simulator_type(method="SENTMAN", base_dir=td)
                qois, costs, indices = sim.analyze_simulation_results_qois(
                    [7], requested_qois=["P_D", "P_D2", "P_L", "P_Mz"]
                )

                np.testing.assert_allclose(qois["P_D"], [6.0])
                np.testing.assert_allclose(qois["P_D2"], [36.0])
                np.testing.assert_allclose(qois["P_L"], [-2.0])
                np.testing.assert_allclose(qois["P_Mz"], [0.125])
                np.testing.assert_allclose(costs, [1.0])
                np.testing.assert_array_equal(indices, [7])

    def test_coefficients_with_q_inf_are_converted_to_pressure(self):
        header = "gsi_model idx C_D C_L C_Y C_Mx C_My C_Mz q_inf cpu_time_ms"
        row = "SENTMAN 3 2.0 -0.5 0.25 0.1 -0.2 0.3 4.0 7200000"

        with tempfile.TemporaryDirectory() as td:
            self._write_results(td, header, row)
            sim = ADBSatSimulator(method="SENTMAN", base_dir=td)
            qois, costs, indices = sim.analyze_simulation_results_qois(
                [3], requested_qois=["P_D", "P_L", "P_Y", "P_Mz"]
            )

            np.testing.assert_allclose(qois["P_D"], [8.0])
            np.testing.assert_allclose(qois["P_L"], [-2.0])
            np.testing.assert_allclose(qois["P_Y"], [1.0])
            np.testing.assert_allclose(qois["P_Mz"], [1.2])
            np.testing.assert_allclose(costs, [2.0])
            np.testing.assert_array_equal(indices, [3])

    def test_legacy_coefficients_without_q_inf_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_results(
                td,
                "gsi_model idx C_D cpu_time_ms",
                "SENTMAN 1 2.0 1000",
            )
            sim = ADBSatSimulator(method="SENTMAN", base_dir=td)
            with self.assertRaisesRegex(ValueError, "pressure in Pa cannot be reconstructed"):
                sim.analyze_simulation_results([1])


if __name__ == "__main__":
    unittest.main()
