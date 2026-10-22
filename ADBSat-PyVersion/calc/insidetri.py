"""Barycentric point-in-triangle test matching MATLAB ADBSat."""

import numpy as np


def insidetri(A, B, C, p):
    """Return indices whose paired point lies inside its paired triangle."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    p = np.asarray(p, dtype=float)
    if not (A.shape == B.shape == C.shape == p.shape):
        raise ValueError("A, B, C and p must have identical 2-by-N shapes")
    if A.ndim != 2 or A.shape[0] != 2:
        raise ValueError("A, B, C and p must be 2-by-N arrays")

    v0 = C - A
    v1 = B - A
    v2 = p - A
    dot00 = np.sum(v0 * v0, axis=0)
    dot01 = np.sum(v0 * v1, axis=0)
    dot02 = np.sum(v0 * v2, axis=0)
    dot11 = np.sum(v1 * v1, axis=0)
    dot12 = np.sum(v1 * v2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Boundary handling is intentionally identical to MATLAB.
    return np.flatnonzero((u >= 0.0) & (v >= 0.0) & ((u + v) < 1.0))




