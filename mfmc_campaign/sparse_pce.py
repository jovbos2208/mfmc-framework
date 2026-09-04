from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def hyperbolic_multi_indices(
    dimension: int,
    degree: int,
    *,
    q_norm: float = 0.75,
    max_interaction: int | None = 2,
) -> np.ndarray:
    """Generate non-constant multi-indices under hyperbolic truncation."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if degree <= 0:
        raise ValueError("degree must be positive")
    if not 0.0 < q_norm <= 1.0:
        raise ValueError("q_norm must be in (0, 1]")
    if max_interaction is not None and max_interaction <= 0:
        raise ValueError("max_interaction must be positive or None")

    selected: List[Tuple[int, ...]] = []
    threshold = float(degree) ** q_norm + 1.0e-12
    hyperbolic_interaction_limit = max(1, int(np.floor(threshold)))
    interaction_limit = min(dimension, hyperbolic_interaction_limit)
    if max_interaction is not None:
        interaction_limit = min(interaction_limit, max_interaction)
    for interaction_order in range(1, interaction_limit + 1):
        for active_dimensions in itertools.combinations(range(dimension), interaction_order):
            for active_powers in itertools.product(range(1, degree + 1), repeat=interaction_order):
                if sum(float(power) ** q_norm for power in active_powers) > threshold:
                    continue
                powers = [0] * dimension
                for column, power in zip(active_dimensions, active_powers):
                    powers[column] = int(power)
                selected.append(tuple(powers))
    selected.sort(key=lambda powers: (sum(powers), sum(power > 0 for power in powers), powers))
    return np.asarray(selected, dtype=int)


def evaluate_monomial_basis(canonical_x: np.ndarray, multi_indices: np.ndarray) -> np.ndarray:
    x = np.asarray(canonical_x, dtype=float)
    indices = np.asarray(multi_indices, dtype=int)
    if x.ndim != 2:
        raise ValueError("canonical_x must be a two-dimensional array")
    if indices.ndim != 2 or indices.shape[1] != x.shape[1]:
        raise ValueError("multi_indices must have one column per input dimension")
    basis = np.ones((x.shape[0], indices.shape[0]), dtype=float)
    for column, powers in enumerate(indices.T):
        active = np.flatnonzero(powers)
        for term in active:
            basis[:, term] *= x[:, column] ** int(powers[term])
    return basis


def grouped_cv_splits(groups: Sequence[Any], folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    group_values = np.asarray(groups)
    n_samples = len(group_values)
    if n_samples < 4:
        raise ValueError("At least four observations are required for cross-validation")
    unique_groups = np.unique(group_values)
    if len(unique_groups) >= 2:
        n_splits = min(int(folds), len(unique_groups))
        split_groups = [unique_groups[index::n_splits] for index in range(n_splits)]
        out = []
        all_indices = np.arange(n_samples, dtype=int)
        for test_groups in split_groups:
            test_mask = np.isin(group_values, test_groups)
            out.append((all_indices[~test_mask], all_indices[test_mask]))
        return out
    n_splits = min(int(folds), n_samples)
    shuffled = np.random.default_rng(0).permutation(n_samples)
    test_folds = np.array_split(shuffled, n_splits)
    all_indices = np.arange(n_samples, dtype=int)
    return [(all_indices[~np.isin(all_indices, test)], np.sort(test)) for test in test_folds]


@dataclass
class SparsePCEModel:
    input_names: List[str]
    active_input_indices: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    multi_indices: np.ndarray
    basis_mean: np.ndarray
    basis_scale: np.ndarray
    standardized_coefficients: np.ndarray
    output_mean: float
    output_scale: float
    alpha: float
    degree: int
    q_norm: float
    max_interaction: int | None

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.input_names):
            raise ValueError(f"Expected an (n, {len(self.input_names)}) input array")
        values = values[:, self.active_input_indices]
        canonical = (values - self.input_mean) / self.input_scale
        basis = evaluate_monomial_basis(canonical, self.multi_indices)
        standardized_basis = (basis - self.basis_mean) / self.basis_scale
        standardized_output = standardized_basis @ self.standardized_coefficients
        return self.output_mean + self.output_scale * standardized_output

    def effective_coefficients(self) -> np.ndarray:
        return self.output_scale * self.standardized_coefficients / self.basis_scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_names": self.input_names,
            "active_input_indices": self.active_input_indices.tolist(),
            "active_input_names": [self.input_names[index] for index in self.active_input_indices],
            "input_mean": self.input_mean.tolist(),
            "input_scale": self.input_scale.tolist(),
            "multi_indices": self.multi_indices.tolist(),
            "basis_mean": self.basis_mean.tolist(),
            "basis_scale": self.basis_scale.tolist(),
            "standardized_coefficients": self.standardized_coefficients.tolist(),
            "effective_coefficients": self.effective_coefficients().tolist(),
            "output_mean": self.output_mean,
            "output_scale": self.output_scale,
            "alpha": self.alpha,
            "degree": self.degree,
            "q_norm": self.q_norm,
            "max_interaction": self.max_interaction,
        }

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as target:
            json.dump(self.to_dict(), target, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SparsePCEModel":
        return cls(
            input_names=[str(name) for name in payload["input_names"]],
            active_input_indices=np.asarray(payload["active_input_indices"], dtype=int),
            input_mean=np.asarray(payload["input_mean"], dtype=float),
            input_scale=np.asarray(payload["input_scale"], dtype=float),
            multi_indices=np.asarray(payload["multi_indices"], dtype=int),
            basis_mean=np.asarray(payload["basis_mean"], dtype=float),
            basis_scale=np.asarray(payload["basis_scale"], dtype=float),
            standardized_coefficients=np.asarray(payload["standardized_coefficients"], dtype=float),
            output_mean=float(payload["output_mean"]),
            output_scale=float(payload["output_scale"]),
            alpha=float(payload["alpha"]),
            degree=int(payload["degree"]),
            q_norm=float(payload["q_norm"]),
            max_interaction=payload.get("max_interaction"),
        )

    @classmethod
    def read_json(cls, path: str) -> "SparsePCEModel":
        with open(path, "r", encoding="utf-8") as source:
            return cls.from_dict(json.load(source))


def fit_sparse_pce(
    x: np.ndarray,
    y: np.ndarray,
    input_names: Sequence[str],
    *,
    groups: Sequence[Any] | None = None,
    degree: int = 3,
    q_norm: float = 0.75,
    max_interaction: int | None = 2,
    cv_folds: int = 5,
    alphas: np.ndarray | None = None,
    max_iter: int = 10_000,
) -> SparsePCEModel:
    """Fit a standardized sparse polynomial model using grouped LassoCV."""
    values = np.asarray(x, dtype=float)
    targets = np.asarray(y, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] != len(targets):
        raise ValueError("x and y have incompatible shapes")
    if values.shape[1] != len(input_names):
        raise ValueError("input_names must match x columns")
    finite = np.isfinite(targets) & np.all(np.isfinite(values), axis=1)
    values = values[finite]
    targets = targets[finite]
    if groups is None:
        fit_groups = np.zeros(len(targets), dtype=int)
    else:
        group_values = np.asarray(groups)
        if len(group_values) != len(finite):
            raise ValueError("groups must contain one entry per original observation")
        fit_groups = group_values[finite]
    if len(targets) < 4:
        raise ValueError("At least four finite observations are required")

    input_mean = np.mean(values, axis=0)
    input_scale = np.std(values, axis=0)
    active_inputs = input_scale > 1.0e-14
    if not np.any(active_inputs):
        raise ValueError("All input columns are constant")
    values = values[:, active_inputs]
    active_input_indices = np.flatnonzero(active_inputs)
    input_mean = input_mean[active_inputs]
    input_scale = input_scale[active_inputs]
    canonical = (values - input_mean) / input_scale
    indices = hyperbolic_multi_indices(
        values.shape[1], degree, q_norm=q_norm, max_interaction=max_interaction
    )
    basis = evaluate_monomial_basis(canonical, indices)
    basis_mean = np.mean(basis, axis=0)
    basis_scale = np.std(basis, axis=0)
    active_terms = basis_scale > 1.0e-14
    if not np.any(active_terms):
        raise ValueError("Polynomial basis is constant on the training data")
    basis = basis[:, active_terms]
    indices = indices[active_terms]
    basis_mean = basis_mean[active_terms]
    basis_scale = basis_scale[active_terms]
    standardized_basis = (basis - basis_mean) / basis_scale

    output_mean = float(np.mean(targets))
    output_scale = float(np.std(targets))
    if output_scale <= 1.0e-14:
        raise ValueError("Output is constant")
    standardized_targets = (targets - output_mean) / output_scale
    cv = grouped_cv_splits(fit_groups, folds=cv_folds)
    pandas_module = sys.modules.get("pandas")
    invalid_pandas_stub = pandas_module is not None and not hasattr(pandas_module, "DataFrame")
    if invalid_pandas_stub:
        del sys.modules["pandas"]
    try:
        try:
            from sklearn.linear_model import LassoCV
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError("Sparse PCE fitting requires scikit-learn>=1.4,<1.8") from exc
        alpha_grid = np.asarray(alphas if alphas is not None else np.logspace(-4, -0.25, 24), dtype=float)
        estimator = LassoCV(
            alphas=alpha_grid,
            cv=cv,
            fit_intercept=False,
            max_iter=int(max_iter),
            n_jobs=1,
            selection="cyclic",
        )
        estimator.fit(standardized_basis, standardized_targets)
    finally:
        if invalid_pandas_stub:
            sys.modules["pandas"] = pandas_module
    return SparsePCEModel(
        input_names=[str(name) for name in input_names],
        active_input_indices=active_input_indices,
        input_mean=input_mean,
        input_scale=input_scale,
        multi_indices=indices,
        basis_mean=basis_mean,
        basis_scale=basis_scale,
        standardized_coefficients=np.asarray(estimator.coef_, dtype=float),
        output_mean=output_mean,
        output_scale=output_scale,
        alpha=float(estimator.alpha_),
        degree=int(degree),
        q_norm=float(q_norm),
        max_interaction=max_interaction,
    )


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(estimate)
    truth = truth[finite]
    estimate = estimate[finite]
    if not len(truth):
        return {"n": 0.0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    error = estimate - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0.0 else float("nan")
    return {
        "n": float(len(truth)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": r2,
    }
