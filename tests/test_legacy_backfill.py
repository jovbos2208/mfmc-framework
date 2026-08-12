from __future__ import annotations

from mfmc_campaign.legacy_backfill import _selected_indices


def test_selected_indices_are_deterministic_and_include_boundaries() -> None:
    assert _selected_indices(3, 10) == {0, 1, 2}
    assert _selected_indices(10, 3) == {0, 4, 9}
    assert _selected_indices(10, 0) == set(range(10))
    assert _selected_indices(0, 3) == set()


def test_lf_selection_can_be_extended_with_all_paired_hf_indices() -> None:
    hf_indices = _selected_indices(6, 3)
    lf_indices = _selected_indices(100, 3)
    lf_indices.update(hf_indices)

    assert hf_indices.issubset(lf_indices)
    assert lf_indices == {0, 2, 5, 49, 99}
