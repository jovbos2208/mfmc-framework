import importlib.util
from pathlib import Path

P=Path(__file__).parents[1]/"scripts/prepare_vleo_cll_local_search.py"
S=importlib.util.spec_from_file_location("local_search",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)


def test_radius_is_five_percent_of_domains():
    assert M.STEPS=={"nose_length_fraction":.0175,"tail_length_fraction":.0175,
                     "width_height_ratio":.075,"chamfer_fraction":.0165}


def test_three_distinct_branch_centres():
    assert set(M.CENTRES)=={"mean","robustness","combination"}
    assert len(set(M.CENTRES.values()))==3
