import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADBSAT_PY_DIR = ROOT / "ADBSat-PyVersion"
if str(ADBSAT_PY_DIR) not in sys.path:
    sys.path.insert(0, str(ADBSAT_PY_DIR))

mat2vtu_stub = types.ModuleType("mat2vtu")
mat2vtu_stub.mat2vtu = lambda *args, **kwargs: None
sys.modules.setdefault("mat2vtu", mat2vtu_stub)

postpro_stub = types.ModuleType("postpro")
plot_surfq_stub = types.ModuleType("postpro.plot_surfq")
plot_surfq_stub.plot_surfq = lambda *args, **kwargs: None
sys.modules.setdefault("postpro", postpro_stub)
sys.modules.setdefault("postpro.plot_surfq", plot_surfq_stub)

import simulate as adbsat_simulate
from calc.ADBSatConstants import ConstantsData


def _fake_mesh(n_elems: int = 4):
    return {
        "meshdata": np.array(
            [[(np.zeros((3, n_elems), dtype=float),)]],
            dtype=[("XData", "O")],
        )
    }


class TestADBSatPayloadIntegration(unittest.TestCase):
    def test_run_simulation_loads_payload_atmosphere_into_inparam(self):
        payload = {
            "environment_model": "csv",
            "geometry_id": "Cube",
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
            "sample": {
                "database_index": 7,
                "energy_accommodation": 0.81,
                "wall_temperature_k": 456.0,
            },
            "aos_deg": 12.5,
            "aoa_deg": -3.0,
        }

        expected_rho = np.array(
            [
                payload["atmosphere_row"][4],
                payload["atmosphere_row"][3],
                payload["atmosphere_row"][1],
                payload["atmosphere_row"][2],
                payload["atmosphere_row"][6],
                payload["atmosphere_row"][5],
                payload["atmosphere_row"][7],
                payload["atmosphere_row"][8],
                payload["atmosphere_row"][9],
                payload["atmosphere_row"][0],
            ],
            dtype=float,
        )

        captured = {}

        def fail_if_csv_database_is_used(*args, **kwargs):
            raise AssertionError("CSV atmosphere fallback should not be used when payload.atmosphere_row is provided")

        def fake_calc_coeff(
            mod_out,
            res_out,
            aoa_list,
            aos_list,
            inparam,
            shadow,
            solar,
            dyn_p,
            delete_temp_files,
            verbose,
            return_details=True,
        ):
            captured["aoa_list"] = aoa_list
            captured["aos_list"] = aos_list
            captured["inparam"] = inparam
            captured["dyn_p"] = dyn_p
            return {
                "C_D": 0.5,
                "C_L": 0.0,
                "C_Mx": 0.0,
                "C_My": 0.0,
                "C_Mz": 0.0,
            }

        with tempfile.TemporaryDirectory() as td:
            payload_path = os.path.join(td, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            with patch.object(adbsat_simulate, "loadmat", return_value=_fake_mesh()), patch.object(
                adbsat_simulate, "_load_atmosphere_database", side_effect=fail_if_csv_database_is_used
            ), patch.object(adbsat_simulate, "calc_coeff", side_effect=fake_calc_coeff):
                coeffs, runtime = adbsat_simulate.run_simulation(
                    gsi_model="Sentman",
                    run_id=7,
                    altitude=200,
                    aos_deg=1.0,
                    adbsat_path=td,
                    env_payload_path=payload_path,
                )

        self.assertEqual(0.5, coeffs["C_D"])
        self.assertGreaterEqual(runtime, 0.0)
        self.assertIn("inparam", captured)
        np.testing.assert_allclose(captured["inparam"]["rho"], expected_rho)
        self.assertAlmostEqual(987.0, float(captured["inparam"]["Tinf"]), places=12)
        self.assertAlmostEqual(456.0, float(captured["inparam"]["Tw"]), places=12)
        np.testing.assert_allclose(captured["inparam"]["alpha"], np.full(4, 0.81, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["trans_accommodation"], np.full(4, 0.81, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["momentum_accommodation"], np.full(4, 0.81 * 0.81, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["alphaN"], np.full(4, 0.81, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["sigmaT"], np.full(4, 0.7, dtype=float))
        self.assertAlmostEqual(np.radians(-3.0), float(captured["aoa_list"][0]), places=12)
        self.assertAlmostEqual(np.radians(102.5), float(captured["aos_list"][0]), places=12)
        constants = ConstantsData()
        expected_vinf = np.sqrt(constants.mu_E / (constants.R_E + 200000.0))
        self.assertAlmostEqual(expected_vinf, float(captured["inparam"]["vinf"]), places=9)
        self.assertGreater(float(captured["dyn_p"]), 0.0)

    def test_run_simulation_incorporates_wind_into_effective_angles(self):
        payload = {
            "environment_model": "csv",
            "geometry_id": "Cube",
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
            "aos_deg": 0.0,
            "aoa_deg": 0.0,
            "wind_enu_mps": [100.0, 0.0, 50.0],
        }
        captured = {}

        def fake_calc_coeff(
            mod_out,
            res_out,
            aoa_list,
            aos_list,
            inparam,
            shadow,
            solar,
            dyn_p,
            delete_temp_files,
            verbose,
            return_details=True,
        ):
            captured["aoa_list"] = aoa_list
            captured["aos_list"] = aos_list
            return {"C_D": 0.5, "C_L": 0.0, "C_Mx": 0.0, "C_My": 0.0, "C_Mz": 0.0}

        with tempfile.TemporaryDirectory() as td:
            payload_path = os.path.join(td, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            with patch.object(adbsat_simulate, "loadmat", return_value=_fake_mesh()), patch.object(
                adbsat_simulate, "_load_atmosphere_database", side_effect=AssertionError("CSV fallback should not be used")
            ), patch.object(adbsat_simulate, "calc_coeff", side_effect=fake_calc_coeff):
                adbsat_simulate.run_simulation(
                    gsi_model="Sentman",
                    run_id=7,
                    altitude=200,
                    aos_deg=0.0,
                    adbsat_path=td,
                    env_payload_path=payload_path,
                )

        constants = ConstantsData()
        base_speed = np.sqrt(constants.mu_E / (constants.R_E + 200000.0))
        rel = np.array([0.0, base_speed, 0.0]) - np.array([100.0, 0.0, 50.0])
        rel /= np.linalg.norm(rel)
        expected_aoa = np.degrees(np.arcsin(np.clip(rel[2], -1.0, 1.0)))
        expected_aos = np.degrees(np.arctan2(-rel[0], rel[1])) + 90.0
        self.assertAlmostEqual(np.radians(expected_aoa), float(captured["aoa_list"][0]), places=6)
        self.assertAlmostEqual(np.radians(expected_aos), float(captured["aos_list"][0]), places=6)

    def test_piclas_maxwell_keeps_trans_and_momentum_accommodation_separate(self):
        payload = {
            "environment_model": "csv",
            "geometry_id": "Cube",
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
            "sample": {
                "energy_accommodation": 0.74,
                "momentum_accommodation": 0.42,
                "wall_temperature_k": 321.0,
            },
        }
        captured = {}

        def fake_calc_coeff(
            mod_out,
            res_out,
            aoa_list,
            aos_list,
            inparam,
            shadow,
            solar,
            dyn_p,
            delete_temp_files,
            verbose,
            return_details=True,
        ):
            captured["inparam"] = inparam
            return {"C_D": 0.5, "C_L": 0.0, "C_Mx": 0.0, "C_My": 0.0, "C_Mz": 0.0}

        with tempfile.TemporaryDirectory() as td:
            payload_path = os.path.join(td, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            with patch.object(adbsat_simulate, "loadmat", return_value=_fake_mesh()), patch.object(
                adbsat_simulate, "_load_atmosphere_database", side_effect=AssertionError("CSV fallback should not be used")
            ), patch.object(adbsat_simulate, "calc_coeff", side_effect=fake_calc_coeff):
                adbsat_simulate.run_simulation(
                    gsi_model="PICLasMaxwell",
                    run_id=7,
                    altitude=200,
                    aos_deg=0.0,
                    adbsat_path=td,
                    env_payload_path=payload_path,
                )

        np.testing.assert_allclose(captured["inparam"]["alpha"], np.full(4, 0.74, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["trans_accommodation"], np.full(4, 0.74, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["momentum_accommodation"], np.full(4, 0.42, dtype=float))
        np.testing.assert_allclose(captured["inparam"]["sigmaT"], np.full(4, 0.7, dtype=float))
        self.assertAlmostEqual(321.0, float(captured["inparam"]["Tw"]), places=12)


if __name__ == "__main__":
    unittest.main()
