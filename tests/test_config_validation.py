import unittest

from mfmc_campaign.config import validate_config, normalize_config



def base_config():
    return normalize_config(
        {
            "study": {"id": "t1", "mode": "baseline", "active_source_blocks": ["environment.density"]},
            "geometries": [{"id": "Cube", "name": "Cube"}],
            "regimes": [
                {
                    "id": "r1",
                    "label": "r1",
                    "descriptors": {
                        "altitude_km": 200,
                        "characteristic_length": 0.1,
                        "knudsen_number": 1.0,
                        "speed_ratio": 8.0,
                        "freestream_temperature": 900.0,
                        "composition_descriptor": "demo",
                        "solar_activity_state": "quiet",
                        "geomagnetic_activity_state": "quiet",
                        "wind_state": "low",
                        "geometry_class": "cube",
                        "surface_state": "clean",
                    },
                }
            ],
            "sources": {"blocks": [{"name": "environment.density"}, {"name": "attitude.aos"}]},
            "variables": [
                {
                    "name": "density",
                    "source_block": "environment.density",
                    "distribution": {"kind": "normal", "params": {"mean": 1.0, "std": 0.1}},
                    "baseline": 1.0,
                }
            ],
            "sampling": {"method": "independent", "sample_count": 8},
            "models": {
                "hf": {"id": "hf"},
                "lf": [{"id": "lf1"}],
                "available_qois": {"hf": ["C_D", "C_D2"], "lf1": ["C_D", "C_D2"]},
            },
            "qois": {"direct": ["C_D", "C_D2"], "derived": [{"name": "Var_C_D", "expression": "E[C_D2]-E[C_D]^2"}]},
            "pilot": {"size": 8, "sizes": [4, 8], "robustness_repetitions": 3},
            "budget": {"total": 20, "hf_fraction": 0.5},
            "repetitions": 1,
            "seeds": {"global": 123},
            "outputs": {"dir": "tmp", "write_parquet": False, "plots": False},
            "execution": {"backend": "mock"},
        }
    )


class TestConfigValidation(unittest.TestCase):
    def test_valid_config_has_no_errors(self):
        cfg = base_config()
        errors, _ = validate_config(cfg)
        self.assertEqual([], errors)

    def test_unknown_source_block(self):
        cfg = base_config()
        cfg["variables"][0]["source_block"] = "unknown.block"
        errors, _ = validate_config(cfg)
        self.assertTrue(any("unknown source block" in e.message for e in errors))

    def test_malformed_distribution(self):
        cfg = base_config()
        cfg["variables"][0]["distribution"] = {"kind": "bogus", "params": {}}
        errors, _ = validate_config(cfg)
        self.assertTrue(any("unsupported distribution" in e.message for e in errors))

    def test_qoi_unavailable(self):
        cfg = base_config()
        cfg["models"]["available_qois"]["lf1"] = ["C_D"]
        errors, _ = validate_config(cfg)
        self.assertTrue(any("unavailable" in e.message for e in errors))

    def test_pairwise_requires_two_sources(self):
        cfg = base_config()
        cfg["study"]["mode"] = "pairwise_interaction"
        cfg["study"]["pairwise_source_blocks"] = [["environment.density"]]
        errors, _ = validate_config(cfg)
        self.assertTrue(any("exactly 2" in e.message for e in errors))


if __name__ == "__main__":
    unittest.main()
