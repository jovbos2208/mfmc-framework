import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from mfmc_campaign.field_pod_mfmc import (
    FieldPodMfmcConfig,
    ProjectionWarningThreshold,
    TopologyTolerance,
    build_snapshots,
    build_tpmc_basis,
    check_topology,
    load_field_config,
    load_surface_archive,
    multi_control_mfmc_moments,
    moment_matrix,
    principal_angles,
    projection_diagnostics,
    reconstruct_cd_from_full_traction,
    run_data_check,
    run_field_workflow,
    shared_weight_mfmc_moments,
)


def _make_low_rank_fields(rng, n_samples=160, n_faces=20, rank=5):
    m = 3 * n_faces
    basis, _ = np.linalg.qr(rng.normal(size=(m, rank)))
    amplitudes = rng.normal(scale=np.linspace(1.4, 0.4, rank), size=(n_samples, rank))
    mu = rng.normal(scale=0.1, size=m)
    z_h = mu + amplitudes @ basis.T + rng.normal(scale=0.025, size=(n_samples, m))
    z_l = 0.98 * z_h + 0.08 + rng.normal(scale=0.02, size=(n_samples, m))
    return z_h, z_l


def _archive_from_z(
    path,
    z,
    sample_ids,
    areas,
    a_ref,
    q_inf=2.0,
    u=None,
    centers=None,
    normals=None,
    a_ref_per_sample=None,
    positive_cd=False,
):
    n_samples = z.shape[0]
    n_faces = areas.size
    if u is None:
        u = np.asarray([0.0, 1.0, 0.0], dtype=float)
    if a_ref_per_sample is None:
        a_ref_rows = np.full(n_samples, float(a_ref), dtype=float)
    else:
        a_ref_rows = np.asarray(a_ref_per_sample, dtype=float).reshape(-1)
    weights = np.sqrt(areas.reshape(1, n_faces, 1) / a_ref_rows.reshape(n_samples, 1, 1))
    force = z.reshape(n_samples, n_faces, 3) * q_inf / weights
    q = np.full(n_samples, q_inf)
    u_all = np.tile(u.reshape(1, 3), (n_samples, 1))
    cd = -np.einsum("nfc,nc,f->n", force, u_all, areas) / (q * a_ref_rows)
    if positive_cd:
        cd = np.abs(cd)
    payload = {
        "force_per_area": force,
        "sample_id": np.asarray(sample_ids),
        "face_area": areas,
        "A_ref": np.asarray([a_ref]),
        "A_ref_per_sample": a_ref_rows,
        "q_inf": q,
        "u_hat_inf": u_all,
        "C_D": cd,
    }
    if centers is not None:
        payload["face_center"] = centers
    if normals is not None:
        payload["face_normal"] = normals
    np.savez(path, **payload)
    return cd


class TestFieldPodMfmc(unittest.TestCase):
    def test_snapshot_weighting_and_drag_contribution_reconstruct_cd(self):
        rng = np.random.default_rng(10)
        n_faces = 4
        areas = np.asarray([1.0, 2.0, 3.0, 4.0])
        a_ref = 5.0
        z, _ = _make_low_rank_fields(rng, n_samples=3, n_faces=n_faces, rank=2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "DSMC_surface_loads.npz"
            cd = _archive_from_z(path, z, ["s0", "s1", "s2"], areas, a_ref)
            archive = load_surface_archive(path, "DSMC")

        full = build_snapshots(archive, "full_traction")
        self.assertEqual((3, 3 * n_faces), full.values.shape)
        np.testing.assert_allclose(full.values, z, atol=1.0e-12)

        drag = build_snapshots(archive, "drag_contribution")
        np.testing.assert_allclose(np.sum(drag.values, axis=1), cd, atol=1.0e-12)

    def test_per_sample_reference_area_preserves_snapshots_and_positive_cd_convention(self):
        rng = np.random.default_rng(16)
        n_faces = 5
        areas = 0.5 + rng.random(n_faces)
        a_ref_rows = np.asarray([2.0, 2.2, 1.8, 2.1], dtype=float)
        z, _ = _make_low_rank_fields(rng, n_samples=4, n_faces=n_faces, rank=2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "DSMC_surface_loads.npz"
            cd = _archive_from_z(
                path,
                z,
                ["s0", "s1", "s2", "s3"],
                areas,
                float(a_ref_rows[0]),
                a_ref_per_sample=a_ref_rows,
                positive_cd=True,
            )
            archive = load_surface_archive(path, "DSMC")

        full = build_snapshots(archive, "full_traction")
        np.testing.assert_allclose(full.values, z, atol=1.0e-12)
        signed_cd = reconstruct_cd_from_full_traction(full, archive)
        np.testing.assert_allclose(np.abs(signed_cd), cd, atol=1.0e-12)

    def test_topology_mismatch_fails_identity_mapping(self):
        rng = np.random.default_rng(11)
        areas = np.ones(5)
        z, _ = _make_low_rank_fields(rng, n_samples=2, n_faces=5, rank=2)
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "d.npz"
            p2 = Path(td) / "t.npz"
            _archive_from_z(p1, z, ["a", "b"], areas, 5.0)
            _archive_from_z(p2, z, ["a", "b"], areas * np.asarray([1, 1, 1, 1, 2]), 5.0)
            d = load_surface_archive(p1, "DSMC")
            t = load_surface_archive(p2, "TPMC")

        report = check_topology(d.geometry, t.geometry, TopologyTolerance())
        self.assertFalse(report["identity_mapping_allowed"])
        self.assertFalse(report["same_face_areas"])

    def test_projection_and_coefficients_are_finite(self):
        rng = np.random.default_rng(12)
        areas = np.ones(8)
        z_h, z_l = _make_low_rank_fields(rng, n_samples=60, n_faces=8, rank=3)
        ids = [f"s{i}" for i in range(60)]
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "d.npz"
            lp = Path(td) / "t.npz"
            _archive_from_z(hp, z_h, ids, areas, 8.0)
            _archive_from_z(lp, z_l, ids, areas, 8.0)
            h = build_snapshots(load_surface_archive(hp, "DSMC"), "full_traction")
            l = build_snapshots(load_surface_archive(lp, "TPMC"), "full_traction")
            z_ref, psi, *_ = build_tpmc_basis(l, 8)
            summary = projection_diagnostics(h, z_ref, psi, ids, ProjectionWarningThreshold())
            coeff = (h.values - z_ref) @ psi

        self.assertEqual((60, 8), coeff.shape)
        self.assertTrue(np.isfinite(summary["mean_projection_residual"]))
        self.assertLess(summary["mean_projection_residual"], 0.35)

    def test_shared_weight_mfmc_covariance_beats_dsmc_only_in_high_correlation_synthetic_case(self):
        rng = np.random.default_rng(13)
        z_h, z_l = _make_low_rank_fields(rng, n_samples=1200, n_faces=10, rank=4)
        _, _, sigma_ref = moment_matrix(z_h)
        n = 35
        mu_d, _, sigma_d = moment_matrix(z_h[:n])
        mu_m, _, sigma_m, diag = shared_weight_mfmc_moments(
            z_h[:n],
            z_l[:n],
            z_l,
            shared_weight_response="coefficient_norm",
        )
        err_d = np.linalg.norm(sigma_d - sigma_ref, ord="fro") / np.linalg.norm(sigma_ref, ord="fro")
        err_m = np.linalg.norm(sigma_m - sigma_ref, ord="fro") / np.linalg.norm(sigma_ref, ord="fro")
        self.assertLess(err_m, err_d)
        self.assertLess(np.linalg.norm(mu_m - np.mean(z_h, axis=0)), np.linalg.norm(mu_d - np.mean(z_h, axis=0)))
        self.assertIn("negative_eigenvalue_count_before_correction", diag)

    def test_multi_control_mfmc_accepts_tpmc_and_sentman(self):
        rng = np.random.default_rng(131)
        b_h = rng.normal(size=(4000, 5))
        b_t = b_h + rng.normal(scale=0.16, size=b_h.shape)
        b_s = 0.75 * b_h + rng.normal(scale=0.28, size=b_h.shape)
        n = 35
        mu_d, _, _ = moment_matrix(b_h[:n])
        mu_m, _, sigma_m, diag = multi_control_mfmc_moments(
            b_h[:n],
            [b_t[:n], b_s[:n]],
            [b_t, b_s],
            control_names=["TPMC", "SENTMAN"],
        )
        mu_ref, _, _ = moment_matrix(b_h)
        self.assertLess(np.linalg.norm(mu_m - mu_ref), np.linalg.norm(mu_d - mu_ref))
        self.assertEqual((5, 5), sigma_m.shape)
        self.assertEqual({"TPMC", "SENTMAN"}, set(diag["shared_betas"]))
        self.assertTrue(np.all(np.isfinite(list(diag["shared_betas"].values()))))

    def test_principal_angles_zero_for_same_subspace_with_sign_flip(self):
        rng = np.random.default_rng(14)
        q, _ = np.linalg.qr(rng.normal(size=(20, 4)))
        q2 = q.copy()
        q2[:, 1] *= -1.0
        angles = principal_angles(q[:, :3], q2[:, :3])
        self.assertLess(float(np.max(angles)), 1.0e-7)

    def test_data_checker_writes_missing_data_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "field.yaml"
            cfg_dict = {
                "case_name": "Cube-300km",
                "output_root": str(root / "out"),
                "fidelity_archives": {
                    "DSMC": str(root / "missing_dsmc.npz"),
                    "TPMC": str(root / "missing_tpmc.npz"),
                },
            }
            config_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
            cfg = load_field_config(config_path)
            report = run_data_check(cfg)

            self.assertTrue(report["missing_required"])
            self.assertTrue((cfg.output_dir / "data_availability_report.json").exists())
            self.assertTrue((cfg.output_dir / "field_pod_mfmc_missing_data_Cube-300km.md").exists())

    def test_synthetic_workflow_writes_machine_readable_outputs(self):
        rng = np.random.default_rng(15)
        n_faces = 12
        areas = 0.7 + rng.random(n_faces)
        centers = rng.normal(size=(n_faces, 3))
        normals = rng.normal(size=(n_faces, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        z_h, z_l_c = _make_low_rank_fields(rng, n_samples=90, n_faces=n_faces, rank=4)
        _, z_l_extra = _make_low_rank_fields(rng, n_samples=240, n_faces=n_faces, rank=4)
        z_l = np.vstack([z_l_c, z_l_extra])
        hf_ids = [f"s{i}" for i in range(90)]
        lf_ids = hf_ids + [f"lf_extra_{i}" for i in range(240)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d_path = root / "DSMC_surface_loads.npz"
            t_path = root / "TPMC_surface_loads.npz"
            _archive_from_z(d_path, z_h, hf_ids, areas, float(np.sum(areas)), centers=centers, normals=normals)
            _archive_from_z(t_path, z_l, lf_ids, areas, float(np.sum(areas)), centers=centers, normals=normals)
            cfg = FieldPodMfmcConfig(
                case_name="Cube-300km",
                output_root=root / "out",
                high_fidelity="DSMC",
                low_fidelity_basis_source="TPMC",
                snapshot_type="full_traction",
                basis_size_s=8,
                pod_modes_r=4,
                budgets=(18.0,),
                repeats=3,
                random_seed=99,
                include_pilot_cost=False,
                mfmc_weight_mode="shared_weights",
                shared_weight_response="coefficient_norm",
                psd_correction="none",
                cd_reconstruction_tolerance=1.0e-6,
                topology_tolerance=TopologyTolerance(),
                projection_residual_warning_threshold=ProjectionWarningThreshold(),
                fidelity_archives={"DSMC": d_path, "TPMC": t_path},
                hf_cost=1.0,
                lf_cost=0.05,
                mfmc_hf_fraction=0.45,
            )
            summary = run_field_workflow(cfg)
            with (cfg.output_dir / "comparison_metrics.csv").open(encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))

            self.assertEqual(90, summary["n_coupled_samples"])
            self.assertTrue((cfg.output_dir / "Psi_s.npz").exists())
            self.assertTrue((cfg.output_dir / "Sigma_b_mfmc_by_budget.npz").exists())
            self.assertTrue(metrics)

    def test_three_fidelity_workflow_writes_sentman_coefficients_and_metrics(self):
        rng = np.random.default_rng(151)
        n_faces = 8
        areas = 0.8 + rng.random(n_faces)
        centers = rng.normal(size=(n_faces, 3))
        normals = rng.normal(size=(n_faces, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        z_h, z_t_c = _make_low_rank_fields(rng, n_samples=60, n_faces=n_faces, rank=3)
        z_s_c = 0.82 * z_h + rng.normal(scale=0.08, size=z_h.shape)
        z_t = np.vstack([z_t_c, z_t_c + rng.normal(scale=0.03, size=z_t_c.shape)])
        z_s_extra = np.tile(z_s_c, (3, 1)) + rng.normal(scale=0.05, size=(180, z_s_c.shape[1]))
        z_s = np.vstack([z_s_c, z_s_extra])
        hf_ids = [f"s{i}" for i in range(60)]
        t_ids = hf_ids + [f"t{i}" for i in range(60)]
        s_ids = hf_ids + [f"sent{i}" for i in range(180)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {name: root / f"{name}_surface_loads.npz" for name in ("DSMC", "TPMC", "SENTMAN")}
            a_ref = float(np.sum(areas))
            _archive_from_z(paths["DSMC"], z_h, hf_ids, areas, a_ref, centers=centers, normals=normals)
            _archive_from_z(paths["TPMC"], z_t, t_ids, areas, a_ref, centers=centers, normals=normals)
            _archive_from_z(paths["SENTMAN"], z_s, s_ids, areas, a_ref, centers=centers, normals=normals)
            cfg = FieldPodMfmcConfig(
                case_name="three-fidelity",
                output_root=root / "out",
                high_fidelity="DSMC",
                low_fidelity_basis_source="TPMC",
                snapshot_type="full_traction",
                basis_size_s=6,
                pod_modes_r=3,
                budgets=(12.0,),
                repeats=2,
                random_seed=7,
                include_pilot_cost=False,
                mfmc_weight_mode="shared_weights",
                shared_weight_response="coefficient_norm",
                psd_correction="none",
                cd_reconstruction_tolerance=1.0e-6,
                topology_tolerance=TopologyTolerance(),
                projection_residual_warning_threshold=ProjectionWarningThreshold(),
                fidelity_archives=paths,
                hf_cost=1.0,
                lf_cost=0.05,
                mfmc_hf_fraction=0.5,
                control_variates=("TPMC", "SENTMAN"),
                control_costs={"TPMC": 0.05, "SENTMAN": 0.001},
            )
            summary = run_field_workflow(cfg)
            with (cfg.output_dir / "comparison_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(["TPMC", "SENTMAN"], summary["control_variates"])
            self.assertTrue((cfg.output_dir / "b_coefficients_SENTMAN.npz").exists())
            self.assertIn("n_lf_SENTMAN", rows[0])
            self.assertIn("shared_beta_SENTMAN", rows[0])


if __name__ == "__main__":
    unittest.main()
