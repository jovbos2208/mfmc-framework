import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADBSAT_PY_DIR = ROOT / "ADBSat-PyVersion"
if str(ADBSAT_PY_DIR) not in sys.path:
    sys.path.insert(0, str(ADBSAT_PY_DIR))

from calc.insidetri import insidetri
from calc.shadowAnaly import shadowAnaly


class TestADBSatMatlabShadowPort(unittest.TestCase):
    def test_insidetri_returns_only_paired_inside_indices(self):
        A = np.array([[-1.0, 2.0], [-1.0, 2.0]])
        B = np.array([[1.0, 3.0], [-1.0, 2.0]])
        C = np.array([[0.0, 2.0], [1.0, 3.0]])
        points = np.zeros((2, 2))
        np.testing.assert_array_equal(insidetri(A, B, C, points), np.array([0]))

    def test_projected_occluder_shadows_front_panel(self):
        # Panel 0 is the tested front panel at x=0. Panel 1 lies at x=1,
        # covers its centroid in projection, and has the MATLAB-compatible
        # normal/depth orientation required to cast a shadow.
        x = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        y = np.array([[-0.2, -1.0], [0.2, 1.0], [0.0, 0.0]])
        z = np.array([[-0.2, -1.0], [-0.2, -1.0], [0.2, 1.0]])
        barC = np.vstack((np.mean(x, axis=0), np.mean(y, axis=0), np.mean(z, axis=0)))
        delta = np.radians([0.0, 180.0])
        surfN = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(
            shadowAnaly(x, y, z, barC, delta, np.eye(3), surfN),
            np.array([0]),
        )

    def test_nonoverlapping_projection_does_not_shadow(self):
        x = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        y = np.array([[-0.2, 2.0], [0.2, 4.0], [0.0, 3.0]])
        z = np.array([[-0.2, -1.0], [-0.2, -1.0], [0.2, 1.0]])
        barC = np.vstack((np.mean(x, axis=0), np.mean(y, axis=0), np.mean(z, axis=0)))
        delta = np.radians([0.0, 180.0])
        surfN = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(shadowAnaly(x, y, z, barC, delta, np.eye(3), surfN).size, 0)


if __name__ == "__main__":
    unittest.main()
