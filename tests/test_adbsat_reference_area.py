from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADBSAT_PY_DIR = ROOT / "ADBSat-PyVersion"
if str(ADBSAT_PY_DIR) not in sys.path:
    sys.path.insert(0, str(ADBSAT_PY_DIR))

from calc.calc_coeff import _reference_area


def test_campaign_reference_area_overrides_attitude_dependent_projection() -> None:
    area, source = _reference_area(
        {"reference_area_m2": 0.0017, "reference_area_source": "explicit_payload"},
        projected_area=0.0024,
    )
    assert area == 0.0017
    assert source == "explicit_payload"


def test_reference_area_falls_back_to_half_total_surface() -> None:
    area, source = _reference_area({}, half_total_area=0.0024)
    assert area == 0.0024
    assert source == "half_total_surface"
