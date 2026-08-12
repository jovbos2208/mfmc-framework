import numpy as np
import pytest

from mfmc_campaign.field_mfpod.models import MFPODError
from mfmc_campaign.field_mfpod.two_fidelity_sensitivity import M0_VALUES, budget_tpmc_count, missing_artifacts
from mfmc_campaign.field_mfpod.sensitivity import _nested_prefix_indices, _permuted_nested_fields


def test_budget_counts_match_protocol_and_fix_target():
    assert [(m0,budget_tpmc_count(m0,budget=20,dsmc_cost=1,tpmc_cost=.1,pool=180)) for m0 in M0_VALUES] == [(2,180),(4,160),(6,140),(8,120),(10,100),(12,80),(16,40),(20,0)]


def test_production_prefixes_are_nested_paired_and_reproducible():
    dsmc=np.arange(60.0).reshape(20,3); tpmc=np.arange(540.0).reshape(180,3); ids_d=np.array([f"s{i}" for i in range(20)]); ids_t=np.array([f"s{i}" for i in range(180)])
    a=_permuted_nested_fields({"DSMC":dsmc,"TPMC":tpmc},{"DSMC":ids_d,"TPMC":ids_t},{"DSMC":6,"TPMC":140},random_seed=20260727)
    b=_permuted_nested_fields({"DSMC":dsmc,"TPMC":tpmc},{"DSMC":ids_d,"TPMC":ids_t},{"DSMC":6,"TPMC":140},random_seed=20260727)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    dl={tuple(row):sid for row,sid in zip(dsmc,ids_d)}; tl={tuple(row):sid for row,sid in zip(tpmc,ids_t)}
    assert [dl[tuple(row)] for row in a["DSMC"][:6]] == [tl[tuple(row)] for row in a["TPMC"][:6]]
    order4=_permuted_nested_fields({"DSMC":dsmc,"TPMC":tpmc},{"DSMC":ids_d,"TPMC":ids_t},{"DSMC":4,"TPMC":160},random_seed=20260727)
    assert np.array_equal(order4["DSMC"][:4],a["DSMC"][:4])


def test_reference_prefixes_nested_and_seeded():
    a=_nested_prefix_indices(50,[10,20,30,40,50],random_seed=20260727); b=_nested_prefix_indices(50,[10,20,30,40,50],random_seed=20260727)
    assert all(np.array_equal(a[n],b[n]) for n in a)
    assert all(np.array_equal(a[n],a[m][:n]) for n,m in zip((10,20,30,40),(20,30,40,50)))


def test_invalid_budget_input_rejected():
    with pytest.raises(MFPODError): budget_tpmc_count(0,budget=20,dsmc_cost=1,tpmc_cost=.1,pool=180)


def test_missing_report_counts_local_pilot_fields_without_claiming_remote_fields(tmp_path):
    root = tmp_path / "GOCE-244km-TPMC"
    (root / "production").mkdir(parents=True)
    (root / "inspection").mkdir()
    np.savez_compressed(
        root / "production/pilot_fields.npz",
        DSMC=np.zeros((30, 6)),
        TPMC=np.zeros((30, 6)),
        SENTMAN=np.ones((30, 6)),
    )
    (root / "production/roles.json").write_text(
        '{"pilot":["p0"],"reference_DSMC":["r0"],"production":{"DSMC":["d0"],"TPMC":["t0"]}}',
        encoding="utf-8",
    )
    (root / "inspection/data_availability_report.json").write_text(
        '{"fidelities":{"DSMC":{"path":"/unmounted/DSMC.npz"},"TPMC":{"path":"/unmounted/TPMC.npz"},"SENTMAN":{"path":"/unmounted/SENTMAN.npz"}}}',
        encoding="utf-8",
    )

    report = missing_artifacts(root, requested_m0=[2])

    assert report["available_counts"] == {
        "pilot_pairs": 30,
        "production_DSMC": 0,
        "production_TPMC": 0,
        "reference_DSMC": 0,
    }
    assert report["declared_role_counts"] == {
        "pilot_pairs": 1,
        "production_DSMC": 1,
        "production_TPMC": 1,
        "reference_DSMC": 1,
    }
    assert set(report["inaccessible_field_archives"]) == {"DSMC", "TPMC"}
    assert "pilot_pairs" not in report["missing_by_m0"]["2"]
    assert report["declared_count_missing_by_m0"]["2"] == {
        "production_DSMC": 1,
        "production_TPMC": 179,
        "reference_DSMC": 49,
        "pilot_pairs": 29,
    }
