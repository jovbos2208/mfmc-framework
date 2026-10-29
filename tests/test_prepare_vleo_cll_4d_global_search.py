import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'scripts/prepare_vleo_cll_4d_global_search.py';S=importlib.util.spec_from_file_location('g4d',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_joint_samples_obey_disk_and_bounds():
 rows=M.uncertainty_samples(1000)
 assert all(0<=r['cll_alpha_n']<=1 and 0<=r['cll_sigma_t']<=1 for r in rows)
 assert max(np.hypot(r['aoa_deg'],r['aos_deg']) for r in rows)<=10+1e-12
 assert len({r['sample_id'] for r in rows})==1000
def test_anchor_matrix_is_present():
 rows=M.uncertainty_samples(100)
 assert [(r['aoa_deg'],r['aos_deg']) for r in rows[:13]]==M.ANGLE_ANCHORS
