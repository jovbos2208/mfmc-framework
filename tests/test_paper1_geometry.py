import math

import numpy as np
import pytest

from mfmc_campaign.paper1_geometry import make_hemisphere_shell, make_sphere, validate_geometry


def test_sphere_is_watertight_and_has_prescribed_projected_area() -> None:
    geometry = make_sphere(n_theta=8, n_phi=16)
    report = validate_geometry(geometry)
    assert report["valid"]
    assert report["euler_characteristic"] == 2
    assert report["projected_x_area_m2"] == pytest.approx(math.pi * 0.1**2 / 4.0, rel=0.03)


def test_shell_is_closed_but_cavity_has_no_closing_disk() -> None:
    geometry = make_hemisphere_shell(n_theta=8, n_phi=16)
    report = validate_geometry(geometry)
    assert report["valid"]
    assert report["region_triangle_counts"]["outer"] > 0
    assert report["region_triangle_counts"]["inner"] > 0
    assert report["region_triangle_counts"]["rim"] == 32
    rim = geometry.face_region == 3
    assert np.allclose(geometry.mesh.triangle_center[rim, 0], 0.0, atol=1.0e-12)


def test_shell_inner_and_outer_normals_have_opposite_radial_sense() -> None:
    geometry = make_hemisphere_shell(n_theta=8, n_phi=16)
    for region, sign in ((1, 1.0), (2, -1.0)):
        selected = geometry.face_region == region
        dots = np.einsum(
            "ij,ij->i", geometry.mesh.triangle_normal[selected], geometry.mesh.triangle_center[selected]
        )
        assert np.all(sign * dots > 0.0)
