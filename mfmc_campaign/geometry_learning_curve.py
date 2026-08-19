from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def _selection(manifest: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    if "model_selection" in manifest:
        return dict(manifest["model_selection"])
    cv = {row["method"]: row for row in manifest["geometry_held_out_summary"]}
    lf_rmse = float(cv["lf_pce"]["mean_geometry_rmse"])
    mf_rmse = float(cv["mf_pce"]["mean_geometry_rmse"])
    improvement = (lf_rmse - mf_rmse) / lf_rmse
    selected = "mf_pce" if improvement >= threshold else "lf_pce"
    return {
        "selected_surrogate": selected,
        "correction_applied": selected == "mf_pce",
        "lf_pce_mean_geometry_rmse": lf_rmse,
        "mf_pce_mean_geometry_rmse": mf_rmse,
        "relative_mf_improvement": improvement,
        "minimum_relative_improvement_required": threshold,
        "selected_geometry_rmse": mf_rmse if selected == "mf_pce" else lf_rmse,
    }


def build_geometry_learning_curve(
    manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    minimum_mf_relative_improvement: float = 0.01,
    target_geometry_rmse: float = 1.0e-4,
    minimum_geometry_count: int = 12,
) -> Dict[str, Any]:
    if len(manifest_paths) < 2:
        raise ValueError("At least two surrogate manifests are required for a learning curve")
    if minimum_geometry_count < 2 or target_geometry_rmse <= 0.0:
        raise ValueError("Learning-curve geometry count and RMSE targets must be positive")
    rows: list[Dict[str, Any]] = []
    for value in manifest_paths:
        path = Path(value).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        cv = {row["method"]: row for row in manifest["geometry_held_out_summary"]}
        selection = _selection(manifest, minimum_mf_relative_improvement)
        rows.append({
            "n_geometries": int(manifest["n_geometries"]),
            "n_hf": int(manifest["n_hf"]),
            "n_lf": int(manifest["n_lf"]),
            "lf_pce_mean_geometry_rmse": float(cv["lf_pce"]["mean_geometry_rmse"]),
            "mf_pce_mean_geometry_rmse": float(cv["mf_pce"]["mean_geometry_rmse"]),
            "lf_pce_median_geometry_rmse": float(cv["lf_pce"]["median_geometry_rmse"]),
            "mf_pce_median_geometry_rmse": float(cv["mf_pce"]["median_geometry_rmse"]),
            "selected_surrogate": selection["selected_surrogate"],
            "relative_mf_improvement": float(selection["relative_mf_improvement"]),
            "selected_geometry_rmse": float(selection["selected_geometry_rmse"]),
            "target_geometry_rmse": float(target_geometry_rmse),
            "accuracy_target_met": float(selection["selected_geometry_rmse"]) <= float(target_geometry_rmse),
            "geometry_count_target_met": int(manifest["n_geometries"]) >= int(minimum_geometry_count),
            "manifest_json": str(path),
        })
    rows.sort(key=lambda row: (int(row["n_geometries"]), int(row["n_hf"])))
    if len({int(row["n_geometries"]) for row in rows}) != len(rows):
        raise ValueError("Learning-curve manifests must have distinct geometry counts")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "geometry_learning_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt

        x = [int(row["n_geometries"]) for row in rows]
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(x, [float(row["lf_pce_mean_geometry_rmse"]) for row in rows], "o-", label="TPMC PCE")
        axis.plot(x, [float(row["mf_pce_mean_geometry_rmse"]) for row in rows], "s-", label="TPMC + DSMC discrepancy")
        axis.axhline(float(target_geometry_rmse), color="black", linestyle="--", linewidth=1, label="target")
        axis.set(xlabel="HF training geometries", ylabel="geometry-held-out RMSE [m²]")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(target / "geometry_learning_curve.png", dpi=220)
        fig.savefig(target / "geometry_learning_curve.pdf")
        plt.close(fig)
    except ImportError:  # pragma: no cover - plotting dependency is part of normal install
        pass
    ready = bool(rows[-1]["accuracy_target_met"] and rows[-1]["geometry_count_target_met"])
    summary = {
        "schema_version": 1,
        "status": "target_met" if ready else "more_geometry_acquisition_required",
        "minimum_mf_relative_improvement": float(minimum_mf_relative_improvement),
        "target_geometry_rmse": float(target_geometry_rmse),
        "minimum_geometry_count": int(minimum_geometry_count),
        "rows": rows,
        "learning_curve_csv": str(csv_path),
    }
    manifest_path = target / "geometry_learning_curve.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "manifest_json": str(manifest_path)}
