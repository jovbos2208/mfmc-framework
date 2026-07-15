import tempfile
import unittest
from pathlib import Path

import numpy as np

from mfmc_campaign.adbsat_surface_mapping import (
    CanonicalSurfaceMapping,
    aggregate_panel_traction_to_reference,
    build_and_write_adbsat_surface,
    load_surface_mapping,
    write_surface_mapping,
)
from mfmc_campaign.adbsat_surface_archive import export_adbsat_surface_archive


class TestADBSatSurfaceMapping(unittest.TestCase):
    def test_vtu_conversion_preserves_order_and_conserves_force(self):
        try:
            import pyvista as pv
            from scipy.io import loadmat
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        if not hasattr(pv, "UnstructuredGrid"):
            self.skipTest("A different test installed a minimal pyvista stub")

        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        cells = np.asarray([4, 0, 1, 2, 3, 3, 1, 4, 2], dtype=np.int64)
        cell_types = np.asarray([9, 5], dtype=np.uint8)  # VTK_QUAD, VTK_TRIANGLE
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vtu = root / "surface.vtu"
            obj = root / "Canonical.obj"
            mat = root / "Canonical.mat"
            mapping_path = root / "Canonical.mapping.npz"
            grid.save(vtu)
            summary = build_and_write_adbsat_surface(vtu, obj, mat, mapping_path)
            mapping = load_surface_mapping(mapping_path)
            mat_data = loadmat(mat)
            mat_mesh = mat_data["meshdata"][0, 0]

            self.assertEqual(2, summary["n_reference_faces"])
            self.assertEqual(3, summary["n_adbsat_triangles"])
            np.testing.assert_array_equal(mapping.triangle_to_reference_cell, [0, 0, 1])
            np.testing.assert_allclose(mapping.reference_face_area, [1.0, 0.5])
            self.assertEqual(3, mat_mesh["Areas"].size)
            self.assertEqual(mapping.mesh_fingerprint, str(mat_data["mesh_fingerprint"].reshape(-1)[0]))
            self.assertEqual(3, sum(1 for line in obj.read_text().splitlines() if line.startswith("f ")))

            panel_traction = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                ]
            )
            reference = aggregate_panel_traction_to_reference(panel_traction, mapping)
            np.testing.assert_allclose(reference, [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
            panel_force = np.sum(panel_traction * mapping.triangle_area[:, None], axis=0)
            reference_force = np.sum(reference * mapping.reference_face_area[:, None], axis=0)
            np.testing.assert_allclose(reference_force, panel_force)

    def test_panel_artifacts_export_to_reference_surface_archive(self):
        points = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mapping = CanonicalSurfaceMapping(
            source_vtu="surface.vtu",
            mesh_fingerprint="fingerprint",
            points=points,
            triangles=triangles,
            triangle_to_reference_cell=np.asarray([0, 0], dtype=np.int64),
            triangle_area=np.asarray([0.5, 0.5]),
            triangle_center=np.asarray([[2 / 3, 1 / 3, 0], [1 / 3, 2 / 3, 0]]),
            triangle_normal=np.asarray([[0, 0, 1], [0, 0, 1]], dtype=float),
            reference_face_area=np.asarray([1.0]),
            reference_face_center=np.asarray([[0.5, 0.5, 0.0]]),
            reference_face_normal=np.asarray([[0.0, 0.0, 1.0]]),
            length_scale_to_m=1.0,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mapping_path = root / "mapping.npz"
            result_dir = root / "MFMC_Jobs_Sentman"
            fields = result_dir / "surface_fields"
            fields.mkdir(parents=True)
            write_surface_mapping(mapping_path, mapping)
            for run_id, sample_id, scale in [(0, "s0", 1.0), (1, "s1", 2.0)]:
                np.savez_compressed(
                    fields / f"Sentman_{run_id}.npz",
                    sample_id=np.asarray([sample_id]),
                    mesh_fingerprint=np.asarray([mapping.mesh_fingerprint]),
                    panel_force_per_area=np.asarray([[scale, 0, 0], [3 * scale, 0, 0]], dtype=float),
                    panel_area=mapping.triangle_area,
                    panel_center=mapping.triangle_center,
                    panel_normal=mapping.triangle_normal,
                    q_inf=np.asarray([2.0]),
                    A_ref=np.asarray([1.0]),
                    u_hat_inf=np.asarray([1.0, 0.0, 0.0]),
                    C_D=np.asarray([scale]),
                )
            output = root / "SENTMAN_surface_loads.npz"
            summary = export_adbsat_surface_archive(
                result_dir=result_dir,
                method="Sentman",
                run_ids=[0, 1],
                sample_ids=["s0", "s1"],
                mapping_path=mapping_path,
                output_path=output,
                case_name="Cube-300km",
            )
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(2, summary["n_samples"])
                self.assertEqual((2, 1, 3), archive["force_per_area"].shape)
                np.testing.assert_allclose(archive["force_per_area"][:, 0, 0], [2.0, 4.0])
                np.testing.assert_allclose(archive["face_area"], [1.0])


if __name__ == "__main__":
    unittest.main()
