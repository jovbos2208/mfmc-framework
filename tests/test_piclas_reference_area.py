import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPiclasReferenceArea(unittest.TestCase):
    def test_cube_h5_reference_area_accepts_cube_boundary(self):
        try:
            import h5py  # noqa: F401
        except Exception as exc:
            self.skipTest(f"h5py unavailable: {exc}")

        mesh_path = ROOT / "piclas" / "Cube_mesh.h5"
        if not mesh_path.exists():
            self.skipTest("Cube HDF5 mesh is unavailable")

        from mfmc_campaign.adapters import _piclas_h5_half_total_reference_area

        area = _piclas_h5_half_total_reference_area(str(mesh_path))

        self.assertIsNotNone(area)
        self.assertGreater(float(area), 0.0)

    def test_goce_h5_reference_area_uses_obj_boundary_surface(self):
        try:
            import h5py  # noqa: F401
            import pyvista  # noqa: F401
        except Exception as exc:
            self.skipTest(f"h5py/pyvista unavailable: {exc}")

        mesh_path = ROOT / "piclas" / "GOCE_mesh.h5"
        boundary_vtu = ROOT / "GOCE_2019" / "GOCE_Debugmesh_BC.vtu"
        if not mesh_path.exists() or not boundary_vtu.exists():
            self.skipTest("GOCE HDF5 mesh or HOPR boundary VTU is unavailable")

        from mfmc_campaign.adapters import _piclas_h5_half_total_reference_area

        area = _piclas_h5_half_total_reference_area(str(mesh_path))

        self.assertIsNotNone(area)
        self.assertAlmostEqual(0.0236561598414, float(area), places=12)
        self.assertNotAlmostEqual(0.129761684793, float(area), places=6)


if __name__ == "__main__":
    unittest.main()
