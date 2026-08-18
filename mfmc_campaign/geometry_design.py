from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .parametric_geometry import CylinderHexSpec, build_geometry_assets


VARIABLES = (
    "nose_length_fraction",
    "tail_length_fraction",
    "width_height_ratio",
    "chamfer_fraction",
)


@dataclass(frozen=True)
class CylinderHexDesignSpace:
    nose_length_fraction: tuple[float, float] = (0.08, 0.32)
    tail_length_fraction: tuple[float, float] = (0.08, 0.32)
    width_height_ratio: tuple[float, float] = (0.65, 1.55)
    chamfer_fraction: tuple[float, float] = (0.05, 0.30)

    def bounds(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in VARIABLES], dtype=float)

    def validate(self) -> None:
        bounds = self.bounds()
        if bounds.shape != (len(VARIABLES), 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("Geometry design bounds must be finite lower/upper pairs")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Every geometry design lower bound must be below its upper bound")


def _lhs(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty((n, dimensions), dtype=float)
    for column in range(dimensions):
        values[:, column] = (rng.permutation(n) + rng.random(n)) / n
    return values


def _minimum_distance(points: np.ndarray, anchors: np.ndarray | None = None) -> float:
    distances: List[np.ndarray] = []
    if len(points) > 1:
        delta = points[:, None, :] - points[None, :, :]
        pairwise = np.sqrt(np.sum(delta * delta, axis=2))
        distances.append(pairwise[np.triu_indices(len(points), 1)])
    if anchors is not None and len(anchors):
        delta = points[:, None, :] - anchors[None, :, :]
        distances.append(np.sqrt(np.sum(delta * delta, axis=2)).reshape(-1))
    return float(np.min(np.concatenate(distances))) if distances else float("inf")


def maximin_latin_hypercube(
    n: int,
    dimensions: int,
    *,
    seed: int,
    trials: int = 256,
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    if n < 1 or dimensions < 1 or trials < 1:
        raise ValueError("n, dimensions, and trials must be positive")
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_score = -np.inf
    for _ in range(trials):
        candidate = _lhs(n, dimensions, rng)
        score = _minimum_distance(candidate, anchors)
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def _greedy_maximin_indices(points: np.ndarray, count: int, *, excluded: Iterable[int] = ()) -> List[int]:
    excluded_set = set(int(value) for value in excluded)
    candidates = [index for index in range(len(points)) if index not in excluded_set]
    if count < 0 or count > len(candidates):
        raise ValueError("Requested maximin subset is outside the available design count")
    if count == 0:
        return []
    center = np.full(points.shape[1], 0.5)
    first = max(candidates, key=lambda index: (float(np.linalg.norm(points[index] - center)), -index))
    selected = [first]
    while len(selected) < count:
        remaining = [index for index in candidates if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                float(np.min(np.linalg.norm(points[index] - points[selected], axis=1))),
                -index,
            ),
        )
        selected.append(next_index)
    return selected


def _stage_adbsat_assets(manifest: Mapping[str, Any], design_dir: Path, runtime_dir: Path) -> None:
    obj_target = runtime_dir / "inou" / "obj_files"
    mat_target = runtime_dir / "inou" / "models"
    obj_target.mkdir(parents=True, exist_ok=True)
    mat_target.mkdir(parents=True, exist_ok=True)
    geometry_id = str(manifest["geometry_id"])
    shutil.copy2(design_dir / manifest["assets"]["adbsat_obj"], obj_target / f"{geometry_id}.obj")
    shutil.copy2(design_dir / manifest["assets"]["adbsat_mat"], mat_target / f"{geometry_id}.mat")


def build_cylinder_hex_design(
    output_dir: str | Path,
    *,
    n_designs: int = 32,
    n_validation: int = 6,
    seed: int = 20260818,
    maximin_trials: int = 256,
    uniform_scale_factor: float = 0.1,
    design_space: CylinderHexDesignSpace | None = None,
    adbsat_runtime_dir: str | Path | None = None,
) -> Dict[str, Any]:
    if n_designs < 3:
        raise ValueError("At least three geometry designs are required")
    if n_validation < 1 or n_validation >= n_designs - 1:
        raise ValueError("Validation count must leave the baseline and at least one training design")
    space = design_space or CylinderHexDesignSpace()
    space.validate()
    bounds = space.bounds()
    baseline = np.asarray([0.20, 0.20, 1.0, 0.15], dtype=float)
    baseline_normalized = (baseline - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    normalized = np.vstack(
        [
            baseline_normalized,
            maximin_latin_hypercube(
                n_designs - 1,
                len(VARIABLES),
                seed=seed,
                trials=maximin_trials,
                anchors=baseline_normalized[None, :],
            ),
        ]
    )
    physical = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
    validation_indices = set(_greedy_maximin_indices(normalized, n_validation, excluded=[0]))

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = Path(adbsat_runtime_dir).resolve() if adbsat_runtime_dir else root / "adbsat_runtime_assets"
    rows: List[Dict[str, Any]] = []
    for index, values in enumerate(physical):
        geometry_id = f"cylinder_hex_wp5_{index:03d}"
        design_dir = root / "designs" / geometry_id
        spec = CylinderHexSpec(
            nose_length_fraction=float(values[0]),
            tail_length_fraction=float(values[1]),
            width_height_ratio=float(values[2]),
            chamfer_fraction=float(values[3]),
        ).scaled(uniform_scale_factor)
        manifest = build_geometry_assets(
            design_dir,
            spec,
            design_id=geometry_id,
            uniform_scale_factor=uniform_scale_factor,
            body_mesh_size_m=0.003,
            farfield_mesh_size_m=0.015,
        )
        _stage_adbsat_assets(manifest, design_dir, runtime)
        role = "validation" if index in validation_indices else "lf_training"
        if index == 0:
            role = "baseline_lf_training"
        row: Dict[str, Any] = {
            "design_index": index,
            "geometry_id": geometry_id,
            "role": role,
            "eligible_for_model_fitting": role != "validation",
            "manifest_path": str(Path("designs") / geometry_id / f"{geometry_id}.manifest.json"),
            "design_fingerprint": manifest["design_fingerprint"],
            "mesh_fingerprint": manifest["mesh_fingerprint"],
            "reference_area_m2": manifest["reference_area"]["value_m2"],
            "width_m": manifest["derived_geometry"]["width_m"],
            "height_m": manifest["derived_geometry"]["height_m"],
        }
        for column, name in enumerate(VARIABLES):
            row[name] = float(values[column])
            row[f"normalized_{name}"] = float(normalized[index, column])
        rows.append(row)

    csv_path = root / "geometry_design.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = root / "geometry_design_manifest.json"
    result = {
        "schema_version": 1,
        "design_family": "cylinder_hex",
        "seed": int(seed),
        "maximin_trials": int(maximin_trials),
        "n_designs": int(n_designs),
        "n_lf_training": int(n_designs - n_validation),
        "n_untouched_validation": int(n_validation),
        "uniform_linear_scale_factor": float(uniform_scale_factor),
        "density_scale_required_for_kn_similarity": float(1.0 / uniform_scale_factor),
        "variables": list(VARIABLES),
        "bounds": {name: list(getattr(space, name)) for name in VARIABLES},
        "baseline_geometry_id": rows[0]["geometry_id"],
        "validation_geometry_ids": [row["geometry_id"] for row in rows if row["role"] == "validation"],
        "adbsat_runtime_assets": str(runtime),
        "design_csv": str(csv_path),
        "designs": rows,
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "manifest_json": str(manifest_path)}


def _pareto_indices(objectives: np.ndarray) -> List[int]:
    finite = np.all(np.isfinite(objectives), axis=1)
    front: List[int] = []
    for index in np.where(finite)[0]:
        dominated = np.any(
            np.all(objectives[finite] <= objectives[index], axis=1)
            & np.any(objectives[finite] < objectives[index], axis=1)
        )
        if not dominated:
            front.append(int(index))
    return front


def select_initial_hf_designs(
    design_csv: str | Path,
    lf_metrics_csv: str | Path,
    output_json: str | Path,
    *,
    count: int = 6,
    objective_columns: Sequence[str] = ("mean_drag", "std_drag", "q95_drag"),
) -> Dict[str, Any]:
    with Path(design_csv).open(newline="", encoding="utf-8") as handle:
        designs = list(csv.DictReader(handle))
    with Path(lf_metrics_csv).open(newline="", encoding="utf-8") as handle:
        metrics = {row["geometry_id"]: row for row in csv.DictReader(handle)}
    eligible = [row for row in designs if str(row["eligible_for_model_fitting"]).lower() == "true"]
    missing = [row["geometry_id"] for row in eligible if row["geometry_id"] not in metrics]
    if missing:
        raise ValueError(f"LF metrics are missing for eligible geometries: {missing[:10]}")
    if count < 2 or count > len(eligible):
        raise ValueError("HF selection count must be between two and the eligible design count")
    points = np.asarray(
        [[float(row[f"normalized_{name}"]) for name in VARIABLES] for row in eligible], dtype=float
    )
    objectives = np.asarray(
        [[float(metrics[row["geometry_id"]][name]) for name in objective_columns] for row in eligible],
        dtype=float,
    )
    pareto = _pareto_indices(objectives)
    baseline = next(index for index, row in enumerate(eligible) if row["role"] == "baseline_lf_training")
    selected = [baseline]
    pareto_slots = min(max(1, count // 2), len(pareto))
    while len([index for index in selected if index in pareto]) < pareto_slots:
        remaining = [index for index in pareto if index not in selected]
        if not remaining:
            break
        selected.append(
            max(remaining, key=lambda index: (float(np.min(np.linalg.norm(points[index] - points[selected], axis=1))), -index))
        )
    while len(selected) < count:
        remaining = [index for index in range(len(eligible)) if index not in selected]
        selected.append(
            max(remaining, key=lambda index: (float(np.min(np.linalg.norm(points[index] - points[selected], axis=1))), -index))
        )
    rows = []
    for order, index in enumerate(selected):
        geometry_id = eligible[index]["geometry_id"]
        rows.append(
            {
                "selection_order": order,
                "geometry_id": geometry_id,
                "selection_basis": "baseline" if index == baseline else ("lf_pareto" if index in pareto else "geometry_space_filling"),
                "objectives": {name: float(metrics[geometry_id][name]) for name in objective_columns},
            }
        )
    result = {
        "schema_version": 1,
        "count": count,
        "objective_columns": list(objective_columns),
        "pareto_geometry_ids": [eligible[index]["geometry_id"] for index in pareto],
        "untouched_validation_geometry_ids": [row["geometry_id"] for row in designs if row["role"] == "validation"],
        "selected": rows,
    }
    target = Path(output_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "output_json": str(target)}
