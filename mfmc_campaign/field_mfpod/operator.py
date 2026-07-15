from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .models import MFPODError, MFPODResult
from .weights import residual_energy_weight


def _validate(hf: np.ndarray, lf_p: np.ndarray, lf_extra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, p, e = map(lambda x: np.asarray(x, dtype=np.float64), (hf, lf_p, lf_extra))
    if h.ndim != 2 or p.shape != h.shape or e.ndim != 2 or e.shape[1] != h.shape[1]:
        raise MFPODError("MFPOD expects HF and paired LF rows with equal shape and LF-extra rows in the same state space")
    if h.shape[0] < 1 or e.shape[0] < 1:
        raise MFPODError("Published two-fidelity MFPOD requires 1 <= m_H < m_L")
    return h, p, e


def build_mfpod_linear_operator(hf: np.ndarray, lf_paired: np.ndarray, lf_extra: np.ndarray, alpha: float) -> LinearOperator:
    h, p, e = _validate(hf, lf_paired, lf_extra)
    m_h, m_l = h.shape[0], p.shape[0] + e.shape[0]
    def matvec(v):
        return h.T @ (h @ v) / m_h + alpha * (1 / m_l - 1 / m_h) * (p.T @ (p @ v)) + alpha / m_l * (e.T @ (e @ v))
    return LinearOperator((h.shape[1], h.shape[1]), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def explicit_mfpod_operator(hf: np.ndarray, lf_paired: np.ndarray, lf_extra: np.ndarray, alpha: float) -> np.ndarray:
    h, p, e = _validate(hf, lf_paired, lf_extra)
    m_h, m_l = h.shape[0], p.shape[0] + e.shape[0]
    return h.T @ h / m_h + alpha * (1 / m_l - 1 / m_h) * (p.T @ p) + alpha * e.T @ e / m_l


def _reduced_span(h: np.ndarray, p: np.ndarray, e: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, dict]:
    m_h, m_l = h.shape[0], p.shape[0] + e.shape[0]
    snapshots = np.concatenate((h.T, p.T, e.T), axis=1)
    coeff = np.concatenate((np.full(m_h, 1 / m_h), np.full(m_h, alpha * (1 / m_l - 1 / m_h)), np.full(e.shape[0], alpha / m_l)))
    q, r = np.linalg.qr(snapshots, mode="reduced")
    small = (r * coeff[None, :]) @ r.T
    small = 0.5 * (small + small.T)
    vals, vecs = np.linalg.eigh(small)
    order = np.argsort(vals)[::-1]
    vals, modes = vals[order], q @ vecs[:, order]
    return vals, modes, {"snapshot_rank": int(np.linalg.matrix_rank(r)), "span_dimension": int(q.shape[1])}


def solve_mfpod_eigenproblem(
    hf: np.ndarray, lf_paired: np.ndarray, lf_extra: np.ndarray, alpha: float,
    *, backend: str = "auto", n_modes: Optional[int] = None, tolerance: float = 1e-10, max_iterations: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, p, e = _validate(hf, lf_paired, lf_extra)
    start = time.perf_counter()
    span = 2 * h.shape[0] + e.shape[0]
    chosen = "reduced_snapshot_span" if backend == "auto" and span <= min(h.shape[1], 2000) else backend
    if chosen == "auto": chosen = "linear_operator"
    if chosen == "reduced_snapshot_span":
        vals, modes, diagnostics = _reduced_span(h, p, e, alpha)
        if n_modes is not None: vals, modes = vals[:n_modes], modes[:, :n_modes]
        converged = True
    elif chosen == "linear_operator":
        k = min(n_modes or min(span, h.shape[1] - 1), h.shape[1] - 1)
        if k < 1: raise MFPODError("linear_operator backend needs state dimension >= 2")
        op = build_mfpod_linear_operator(h, p, e, alpha)
        vals, modes = eigsh(op, k=k, which="LA", tol=tolerance, maxiter=max_iterations)
        order = np.argsort(vals)[::-1]; vals, modes = vals[order], modes[:, order]
        diagnostics, converged = {"snapshot_rank": None, "span_dimension": span}, True
    else:
        raise MFPODError(f"Unknown eigensolver backend {backend!r}")
    op = build_mfpod_linear_operator(h, p, e, alpha)
    residuals = np.asarray([np.linalg.norm(op @ modes[:, j] - vals[j] * modes[:, j]) for j in range(vals.size)])
    diagnostics.update({"backend": chosen, "orthogonality_error_fro": float(np.linalg.norm(modes.T @ modes - np.eye(modes.shape[1]), ord="fro")), "eigenpair_residuals": residuals.tolist(), "max_eigenpair_residual": float(np.max(residuals)) if residuals.size else 0.0, "converged": converged, "runtime_seconds": time.perf_counter() - start})
    return vals, modes, diagnostics


def apply_published_eigenvalue_correction(raw_eigenvalues: np.ndarray, modes: np.ndarray, hf_snapshots: np.ndarray, handling: str = "published_hf_mc_correction") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_eigenvalues, dtype=float)
    replacements = np.einsum("ij,ij->j", hf_snapshots @ modes, hf_snapshots @ modes) / hf_snapshots.shape[0]
    mask = raw <= 0
    if handling == "published_hf_mc_correction": corrected = np.where(mask, replacements, raw)
    elif handling == "clip_zero": corrected = np.clip(raw, 0, None)
    elif handling == "none": corrected = raw.copy()
    else: raise MFPODError(f"Unknown negative_eigenvalue_handling={handling!r}")
    return corrected, mask, replacements


def compute_mfpod(hf: np.ndarray, lf_paired: np.ndarray, lf_extra: np.ndarray, alpha: float, *, backend: str = "auto", n_modes: Optional[int] = None, negative_eigenvalue_handling: str = "published_hf_mc_correction", tolerance: float = 1e-10, max_iterations: int = 5000) -> MFPODResult:
    raw, modes, diag = solve_mfpod_eigenproblem(hf, lf_paired, lf_extra, alpha, backend=backend, n_modes=n_modes, tolerance=tolerance, max_iterations=max_iterations)
    corrected, mask, replacements = apply_published_eigenvalue_correction(raw, modes, np.asarray(hf), negative_eigenvalue_handling)
    order = np.argsort(corrected)[::-1]
    diag.update({"negative_eigenvalue_handling": negative_eigenvalue_handling, "corrected_count": int(mask.sum()), "corrected_fraction": float(mask.mean()) if mask.size else 0.0})
    return MFPODResult(modes=modes[:, order], eigenvalues=corrected[order], backend=diag["backend"], diagnostics=diag, raw_eigenvalues=raw[order], corrected_mask=mask[order], hf_mc_replacements=replacements[order], alpha=float(alpha))


def compute_adaptive_mfpod(hf: np.ndarray, lf_paired: np.ndarray, lf_extra: np.ndarray, *, max_modes: int, backend: str = "auto", residual_tolerance: float = 1e-12, negative_eigenvalue_handling: str = "published_hf_mc_correction") -> MFPODResult:
    h, p, e = _validate(hf, lf_paired, lf_extra)
    basis = np.empty((h.shape[1], 0)); eigenvalues=[]; raw_values=[]; masks=[]; replacements=[]; history=[]; stopping="maximum_dimension"
    for _ in range(max_modes):
        wdiag = residual_energy_weight(h, p, basis)
        base_operator = build_mfpod_linear_operator(h, p, e, wdiag["alpha"])
        def project(v):
            return v - basis @ (basis.T @ v) if basis.size else v
        def projected_action(v):
            pv = project(v)
            return project(base_operator @ pv)
        projected = LinearOperator(base_operator.shape, matvec=projected_action, rmatvec=projected_action, dtype=np.float64)
        if h.shape[1] == 1:
            v = np.ones(1); rv = float(projected_action(v)[0])
        else:
            vals, vecs = eigsh(projected, k=1, which="LA", tol=1e-10, maxiter=5000)
            rv, v = float(vals[0]), vecs[:, 0]
        # Reorthogonalize twice to avoid loss of orthogonality when the
        # projected residual is small (modified Gram-Schmidt refinement).
        v = project(v); v = project(v)
        norm = np.linalg.norm(v)
        if norm <= max(residual_tolerance, 1e-10): stopping="no_stable_new_eigenpair"; break
        v = v / norm
        corrected, mask, repl = apply_published_eigenvalue_correction(np.asarray([rv]), v[:, None], h, negative_eigenvalue_handling)
        basis = np.column_stack((basis, v)); eigenvalues.append(corrected[0]); raw_values.append(rv); masks.append(mask[0]); replacements.append(repl[0]); history.append({**wdiag, "raw_eigenvalue": rv, "corrected_eigenvalue": float(corrected[0])})
        residual = h - (h @ basis) @ basis.T
        if np.linalg.norm(residual) <= residual_tolerance * max(np.linalg.norm(h), 1.0): stopping="hf_residual_tolerance"; break
    return MFPODResult(modes=basis, eigenvalues=np.asarray(eigenvalues), backend=backend, diagnostics={"adaptive_history": history, "stopping_reason": stopping, "orthogonality_error_fro": float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1])))}, raw_eigenvalues=np.asarray(raw_values), corrected_mask=np.asarray(masks), hf_mc_replacements=np.asarray(replacements), alpha=np.asarray([x["alpha"] for x in history]))
