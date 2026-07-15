import numpy as np
from scipy.special import erf


def coeff_piclas_maxwell(param_eq, delta):
    """
    PICLas-like extended Maxwellian wall approximation.

    PICLas uses MomentumACC to choose diffuse versus specular reflection and
    TransACC for translational energy accommodation. This deterministic panel
    model uses the same split in the mean coefficient formula.
    """
    trans_acc = np.asarray(param_eq.get("trans_accommodation", param_eq.get("alpha", 1.0)), dtype=float)
    momentum_acc = np.asarray(param_eq.get("momentum_accommodation", param_eq.get("alpha", 1.0)), dtype=float)
    Tw = np.asarray(param_eq["Tw"], dtype=float)
    Tinf = np.asarray(param_eq["Tinf"], dtype=float)
    s = np.asarray(param_eq["s"], dtype=float)

    trans_acc = np.clip(trans_acc, 0.0, 1.0)
    momentum_acc = np.clip(momentum_acc, 0.0, 1.0)

    specular_fraction = 1.0 - momentum_acc
    theta = np.pi / 2 - delta

    tref_over_tinf = 1.0 - trans_acc + trans_acc * (Tw / Tinf)
    tref_over_tinf = np.maximum(tref_over_tinf, 0.0)

    cd = (
        2
        * ((1 - specular_fraction * np.cos(2 * theta)) / (np.sqrt(np.pi) * s))
        * np.exp(-(s**2) * np.sin(theta) ** 2)
        + (np.sin(theta) / s**2)
        * (1 + 2 * s**2 + specular_fraction * (1 - 2 * s**2 * np.cos(2 * theta)))
        * erf(s * np.sin(theta))
        + ((1 - specular_fraction) / s)
        * np.sqrt(np.pi)
        * np.sin(theta) ** 2
        * np.sqrt(tref_over_tinf)
    )

    cl = (
        ((4 * specular_fraction) / (np.sqrt(np.pi) * s))
        * np.sin(theta)
        * np.cos(theta)
        * np.exp(-(s**2) * np.sin(theta) ** 2)
        + (np.cos(theta) / s**2)
        * (1 + specular_fraction * (1 + 4 * s**2 * np.sin(theta) ** 2))
        * erf(s * np.sin(theta))
        + ((1 - specular_fraction) / s)
        * np.sqrt(np.pi)
        * np.sin(theta)
        * np.cos(theta)
        * np.sqrt(tref_over_tinf)
    )

    cd[delta > np.pi / 2] = 0
    cl[delta > np.pi / 2] = 0

    cp = cd * np.cos(delta) + cl * np.sin(delta)
    ctau = cd * np.sin(delta) - cl * np.cos(delta)

    return cp, ctau, cd, cl
