import tempfile
import unittest
from pathlib import Path

import numpy as np

from mfmc_campaign.piclas_surface_archive import write_surface_archive_npz


def _payload(sample_ids, offset=0.0):
    n = len(sample_ids)
    n_faces = 3
    return {
        "sample_id": np.asarray(sample_ids),
        "force_per_area": np.full((n, n_faces, 3), 1.0 + offset, dtype=float),
        "face_area": np.asarray([1.0, 2.0, 3.0], dtype=float),
        "A_ref": np.asarray([6.0 + offset], dtype=float),
        "A_ref_per_sample": np.full(n, 6.0 + offset, dtype=float),
        "q_inf": np.full(n, 2.0, dtype=float),
        "u_hat_inf": np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=float), (n, 1)),
        "C_D": np.full(n, 1.5 + offset, dtype=float),
        "job_subdir": np.asarray([f"job_{sid}" for sid in sample_ids]),
        "fidelity": np.asarray(["DSMC"]),
        "model_id": np.asarray(["PICLas_HF"]),
        "case_name": np.asarray(["Cube-300km"]),
    }


class TestPiclasSurfaceArchive(unittest.TestCase):
    def test_write_and_append_deduplicates_by_sample_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "DSMC_surface_loads.npz"
            first = write_surface_archive_npz(path, _payload(["s0", "s1"], offset=0.0))
            second = write_surface_archive_npz(path, _payload(["s1", "s2"], offset=10.0))

            self.assertEqual(2, first["n_samples"])
            self.assertEqual(3, second["n_samples"])
            with np.load(path, allow_pickle=False) as npz:
                self.assertEqual(["s0", "s1", "s2"], [str(v) for v in npz["sample_id"]])
                self.assertAlmostEqual(1.0, float(npz["force_per_area"][0, 0, 0]))
                self.assertAlmostEqual(11.0, float(npz["force_per_area"][1, 0, 0]))
                self.assertAlmostEqual(11.0, float(npz["force_per_area"][2, 0, 0]))
                np.testing.assert_allclose(npz["face_area"], [1.0, 2.0, 3.0])
                self.assertAlmostEqual(6.0, float(npz["A_ref"][0]))
                np.testing.assert_allclose(npz["A_ref_per_sample"], [6.0, 16.0, 16.0])

            self.assertTrue(path.with_suffix(".summary.json").exists())
            self.assertTrue(path.with_suffix(".manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
