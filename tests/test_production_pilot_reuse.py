import unittest

from mfmc_campaign.campaign import _merge_reused_prefix
from mfmc_campaign.types import EvaluationResult


class TestProductionPilotReuse(unittest.TestCase):
    def test_merges_reused_prefix_and_new_suffix_in_order(self):
        prefix = EvaluationResult(
            values_by_qoi={"C_D": [1.0, 2.0, 99.0], "C_L": [-1.0, -2.0, -99.0]},
            costs=[10.0, 11.0, 99.0],
            sample_ids=["pilot_0", "pilot_1", "pilot_2"],
            metadata={"source": "pilot"},
        )
        suffix = EvaluationResult(
            values_by_qoi={"C_D": [3.0, 4.0], "C_L": [-3.0, -4.0]},
            costs=[12.0, 13.0],
            sample_ids=["prod_2", "prod_3"],
            metadata={"source": "production"},
        )
        merged = _merge_reused_prefix(
            prefix, suffix, 2, ["prod_0", "prod_1", "prod_2", "prod_3"], ["C_D", "C_L"]
        )
        self.assertEqual([1.0, 2.0, 3.0, 4.0], merged.values_by_qoi["C_D"])
        self.assertEqual([-1.0, -2.0, -3.0, -4.0], merged.values_by_qoi["C_L"])
        self.assertEqual([10.0, 11.0, 12.0, 13.0], merged.costs)
        self.assertEqual(["prod_0", "prod_1", "prod_2", "prod_3"], merged.sample_ids)
        self.assertEqual(2, merged.metadata["reused_pilot_prefix"])

    def test_rejects_incomplete_suffix(self):
        prefix = EvaluationResult({"C_D": [1.0]}, [1.0], ["pilot_0"], {})
        suffix = EvaluationResult({"C_D": []}, [], [], {})
        with self.assertRaisesRegex(ValueError, "inconsistent QoI lengths"):
            _merge_reused_prefix(prefix, suffix, 1, ["prod_0", "prod_1"], ["C_D"])


if __name__ == "__main__":
    unittest.main()
