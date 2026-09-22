import csv
import os
import tempfile
import unittest

from mfmc_campaign.campaign import _external_pilot_cost_array, _external_pilot_result
from mfmc_campaign.types import ExperimentCell


class TestExternalPilotRegimeReuse(unittest.TestCase):
    def test_external_pilot_can_be_reused_across_fixed_attitude_regimes(self):
        cell = ExperimentCell(
            study_id="production",
            mode="mixed_uncertainty",
            geometry_id="geometry",
            regime_id="AO_FIXED_AOS_P005_AOA_M010",
            active_source_blocks=["gsi.energy_accommodation"],
            qoi="C_D",
            hf_model_id="PICLas_TPMC",
            lf_model_id="ADBSat_PICLasMaxwell",
            repetition=0,
            seed=1,
            pilot_size=2,
            budget=20.0,
        )
        fieldnames = [
            "phase", "model_id", "qoi", "geometry_id", "regime_id",
            "hf_model_id", "lf_model_id", "active_sources", "study_id",
            "mode", "pilot_size", "repetition", "sample_index", "sample_id",
            "value", "cost",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model_evaluations.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(2):
                    writer.writerow(
                        {
                            "phase": "pilot_hf",
                            "model_id": "PICLas_TPMC",
                            "qoi": "C_D",
                            "geometry_id": "geometry",
                            "regime_id": "AO_FIXED",
                            "hf_model_id": "PICLas_TPMC",
                            "lf_model_id": "ADBSat_PICLasMaxwell",
                            "active_sources": "gsi.energy_accommodation",
                            "study_id": "pilot",
                            "mode": "mixed_uncertainty",
                            "pilot_size": 2,
                            "repetition": 0,
                            "sample_index": index,
                            "sample_id": f"pilot_{index}",
                            "value": 1.0 + index,
                            "cost": 0.25 + index,
                        }
                    )

            self.assertIsNone(
                _external_pilot_result(
                    path=path,
                    cell=cell,
                    phase="pilot_hf",
                    model_id="PICLas_TPMC",
                    qoi="C_D",
                    match_lf_model_id=False,
                )
            )
            result = _external_pilot_result(
                path=path,
                cell=cell,
                phase="pilot_hf",
                model_id="PICLas_TPMC",
                qoi="C_D",
                match_lf_model_id=False,
                match_regime_id=False,
            )
            costs = _external_pilot_cost_array(
                path=path,
                cell=cell,
                phase="pilot_hf",
                model_id="PICLas_TPMC",
                qoi="C_D",
                match_lf_model_id=False,
                match_regime_id=False,
            )

        self.assertIsNotNone(result)
        self.assertEqual([1.0, 2.0], result.values_by_qoi["C_D"])
        self.assertEqual([0.25, 1.25], costs.tolist())


if __name__ == "__main__":
    unittest.main()
