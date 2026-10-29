import importlib.util
from pathlib import Path

import numpy as np

P=Path(__file__).parents[1]/"scripts/analyze_vleo_cll_search50_mfmc.py"
S=importlib.util.spec_from_file_location("search50",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)


def test_fingerprint_is_coordinate_based():
    assert M.fingerprint(.1,.2)==M.fingerprint(.1,.2)
    assert M.fingerprint(.1,.2)!=M.fingerprint(.2,.1)


def test_mfmc_estimator_and_fallback():
    l=np.arange(1.,7.); h=2*l+1; la=np.arange(1.,101.)
    out=M.estimate(h,l,la,h*h,l*l,la*la,.5)
    assert out["active"] and np.isclose(out["mu"],2*la.mean()+1)
    bad=M.estimate(np.ones(6),l,la,np.ones(6),l*l,la*la,.5)
    assert not bad["active"] and bad["beta"]==0


def test_ci_uses_finite_values():
    lo,hi=M.ci(np.array([1.,2.,3.,np.nan]))
    assert lo<2<hi
