import unittest
import warnings
import csv
import os
import tempfile

import numpy as np

from mfmc_campaign.adapters import _build_environment_payload
from mfmc_campaign.estimator import derive_quantities, pilot_robustness_metrics, statistical_flags
from mfmc_campaign.sampling import InputModel, SamplingContext
from mfmc_campaign.types import RegimeDescriptor


class TestSamplingAndEstimator(unittest.TestCase):
    def test_source_freeze_behavior(self):
        variables = [
            {
                "name": "x",
                "source_block": "environment.density",
                "distribution": {"kind": "normal", "params": {"mean": 10.0, "std": 1.0}},
                "baseline": 10.0,
            },
            {
                "name": "y",
                "source_block": "attitude.aos",
                "distribution": {"kind": "normal", "params": {"mean": 0.0, "std": 5.0}},
                "baseline": 3.0,
            },
        ]
        model = InputModel(variables, {"method": "independent"})
        rng = np.random.default_rng(123)

        rows = model.sample(20, SamplingContext(regime_id="r1", active_source_blocks=["environment.density"]), rng)
        ys = [r["y"] for r in rows]
        self.assertTrue(all(abs(y - 3.0) < 1e-12 for y in ys))

    def test_regime_conditioned_distribution(self):
        variables = [
            {
                "name": "temp",
                "source_block": "environment.temperature",
                "distribution": {"kind": "fixed", "params": {"value": 900}},
                "baseline": 900,
                "regime_overrides": {
                    "storm": {"distribution": {"kind": "fixed", "params": {"value": 1200}}}
                },
            }
        ]
        model = InputModel(variables, {"method": "independent"})
        rng = np.random.default_rng(1)

        quiet = model.sample(3, SamplingContext(regime_id="quiet", active_source_blocks=["environment.temperature"]), rng)
        storm = model.sample(3, SamplingContext(regime_id="storm", active_source_blocks=["environment.temperature"]), rng)

        self.assertEqual([900, 900, 900], [int(r["temp"]) for r in quiet])
        self.assertEqual([1200, 1200, 1200], [int(r["temp"]) for r in storm])

    def test_blockwise_joint_sampling(self):
        variables = [
            {
                "name": "a",
                "source_block": "environment.block",
                "distribution": {"kind": "normal", "params": {"mean": 0.0, "std": 1.0}},
                "baseline": 0.0,
            },
            {
                "name": "b",
                "source_block": "environment.block",
                "distribution": {"kind": "normal", "params": {"mean": 0.0, "std": 1.0}},
                "baseline": 0.0,
            },
        ]
        model = InputModel(
            variables,
            {
                "method": "blockwise_joint",
                "block_covariances": {
                    "environment.block": {
                        "variables": ["a", "b"],
                        "means": [0.0, 0.0],
                        "matrix": [[1.0, 0.8], [0.8, 1.0]],
                    }
                },
            },
        )
        rng = np.random.default_rng(11)
        rows = model.sample(1000, SamplingContext(regime_id="r", active_source_blocks=["environment.block"]), rng)
        a = np.array([r["a"] for r in rows], dtype=float)
        b = np.array([r["b"] for r in rows], dtype=float)
        corr = np.corrcoef(a, b)[0, 1]
        self.assertGreater(corr, 0.6)

    def test_trajectory_sampling_injects_coupled_environment_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arc_environment.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "utc",
                        "geodetic_lat_deg",
                        "geodetic_lon_deg",
                        "altitude_km",
                        "relative_speed_m_s",
                        "density_kg_m3",
                        "temperature_K",
                        "x_o_fraction",
                        "x_n2_fraction",
                        "x_o2_fraction",
                        "x_he_fraction",
                        "wind_east_mps",
                        "wind_north_mps",
                        "wind_up_mps",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "utc": "2013-10-22T00:00:00Z",
                        "geodetic_lat_deg": 10.0,
                        "geodetic_lon_deg": 20.0,
                        "altitude_km": 230.0,
                        "relative_speed_m_s": 7800.0,
                        "density_kg_m3": 8.0e-11,
                        "temperature_K": 900.0,
                        "x_o_fraction": 0.7,
                        "x_n2_fraction": 0.27,
                        "x_o2_fraction": 0.02,
                        "x_he_fraction": 0.01,
                        "wind_east_mps": 11.0,
                        "wind_north_mps": -3.0,
                        "wind_up_mps": 1.0,
                    }
                )
                writer.writerow(
                    {
                        "utc": "2013-10-22T00:01:00Z",
                        "geodetic_lat_deg": 11.0,
                        "geodetic_lon_deg": 21.0,
                        "altitude_km": 231.0,
                        "relative_speed_m_s": 7810.0,
                        "density_kg_m3": 9.0e-11,
                        "temperature_K": 910.0,
                        "x_o_fraction": 0.71,
                        "x_n2_fraction": 0.26,
                        "x_o2_fraction": 0.02,
                        "x_he_fraction": 0.01,
                        "wind_east_mps": 12.0,
                        "wind_north_mps": -4.0,
                        "wind_up_mps": 2.0,
                    }
                )

            model = InputModel(
                [{"name": "energy_accommodation", "source_block": "gsi.energy_accommodation", "distribution": {"kind": "fixed", "params": {"value": 0.9}}, "baseline": 0.9}],
                {"method": "independent", "trajectory": {"enabled": True, "path": path, "sample": "sequential"}},
            )
            rows = model.sample(2, SamplingContext(regime_id="r", active_source_blocks=["gsi.energy_accommodation"]), np.random.default_rng(3))

        self.assertEqual([0, 1], [r["database_index"] for r in rows])
        self.assertEqual(["2013-10-22T00:00:00Z", "2013-10-22T00:01:00Z"], [r["datetime_utc"] for r in rows])
        self.assertEqual([11.0, 12.0], [r["wind_east_mps"] for r in rows])
        self.assertAlmostEqual(230.0, rows[0]["altitude_km"])
        self.assertAlmostEqual(7800.0, rows[0]["flow_speed_mps"])
        self.assertEqual(11, len(rows[0]["atmosphere_row"]))
        self.assertAlmostEqual(8.0e-11, rows[0]["atmosphere_row"][0])

    def test_environment_payload_allows_sample_trajectory_state_to_override_regime(self):
        regime = RegimeDescriptor(
            regime_id="r",
            label="r",
            descriptors={
                "altitude_km": 200.0,
                "datetime_utc": "2013-01-01T00:00:00Z",
                "lat_deg": 0.0,
                "lon_deg": 0.0,
            },
        )
        payload = _build_environment_payload(
            {
                "altitude_km": 231.5,
                "datetime_utc": "2013-10-22T00:00:00Z",
                "lat_deg": 10.0,
                "lon_deg": 20.0,
                "flow_speed_mps": 7800.0,
            },
            regime,
            {"environment_model": "csv"},
        )
        self.assertAlmostEqual(231.5, payload["altitude_km"])
        self.assertEqual("2013-10-22T00:00:00Z", payload["datetime_utc"])
        self.assertAlmostEqual(7800.0, payload["flow_speed_mps"])

    def test_derived_quantity_and_flags(self):
        derived = derive_quantities({"C_D": 0.5, "C_D2": 0.30})
        self.assertIn("Var_C_D", derived)
        self.assertAlmostEqual(0.05, derived["Var_C_D"]["value"], places=9)

        flags = statistical_flags(
            {
                "hf_variance": 0.0,
                "lf_variance": 0.0,
                "pearson_correlation": float("nan"),
                "control_variate_beta": float("nan"),
                "residual_variance": 0.0,
            }
        )
        self.assertIn("near_zero_hf_variance", flags)
        self.assertIn("ill_defined_correlation", flags)

    def test_pilot_robustness_metrics(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, size=200)
        y = x + rng.normal(0, 0.2, size=200)
        rows = pilot_robustness_metrics(x, y, [10, 20, 40], repetitions=15, rng=rng, hf_cost=5.0, lf_cost=0.25)
        self.assertTrue(rows)
        self.assertTrue(all("beta_std" in r for r in rows))
        self.assertTrue(all("allocation_ratio_std" in r for r in rows))
        self.assertTrue(all("underperform_frequency" in r for r in rows))

    def test_pilot_robustness_metrics_ignores_all_nan_without_runtime_warning(self):
        rng = np.random.default_rng(7)
        x = np.full(8, float("nan"))
        y = np.full(8, float("nan"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", RuntimeWarning)
            rows = pilot_robustness_metrics(x, y, [4], repetitions=3, rng=rng, hf_cost=5.0, lf_cost=0.25)
        self.assertTrue(rows)
        self.assertFalse(caught)
        self.assertTrue(np.isnan(rows[0]["beta_mean"]))
        self.assertTrue(np.isnan(rows[0]["gain_mean"]))


if __name__ == "__main__":
    unittest.main()
