from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_surrogate_gsa(campaign_root: str, output_dir: str | None = None) -> Dict[str, Any]:
    root = Path(campaign_root).resolve()
    target = Path(output_dir).resolve() if output_dir else root / "gsa_cross_case"
    target.mkdir(parents=True, exist_ok=True)
    case_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for case_path in sorted(path for path in root.iterdir() if path.is_dir()):
        pce = case_path / "surrogate_pce"
        manifest_path = pce / "surrogate_pce_manifest.json"
        summary_path = pce / "gsa_summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            if case_path != target:
                skipped.append({"case": case_path.name, "reason": "missing PCE manifest or GSA summary"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best_lf = str(manifest["best_lf_model_id"])
        loaded = dict(manifest.get("load_summary", {}).get("original_model_counts", {}))
        hf_counts = [
            int(count)
            for model_id, count in loaded.items()
            if model_id not in {best_lf}
            and any(marker in model_id.lower() for marker in ("_hf", "dsmc", "high_fidelity"))
        ]
        if not hf_counts:
            dataset_counts = manifest.get("load_summary", {}).get("loaded_model_counts", {})
            hf_counts = [min(int(count) for count in dataset_counts.values())] if dataset_counts else [0]
        hf_samples = min(hf_counts)
        metrics = _read_csv(Path(manifest["metrics_csv"]))
        selected_metrics = [
            row
            for row in metrics
            if row["fold"] == "aggregate"
            and row["method"] == "mf_residual"
            and row["lf_model_id"] == best_lf
        ]
        if len(selected_metrics) != 1:
            raise ValueError(f"Expected one selected aggregate MF metric row for {case_path.name}")
        metric = selected_metrics[0]
        r2 = float(metric["r2"])
        eligible = hf_samples >= 30 and r2 >= 0.7
        if hf_samples < 30:
            tier = "insufficient_hf_exploratory"
        elif r2 < 0.7:
            tier = "weak_surrogate_exploratory"
        elif r2 < 0.9:
            tier = "moderate_primary_with_caution"
        else:
            tier = "strong_primary"

        sobol = _read_csv(Path(summary["source_sobol_csv"]))
        intervals: Dict[str, Dict[str, str]] = {}
        interval_file = summary.get("refit_bootstrap_intervals_csv")
        if interval_file and Path(interval_file).exists():
            intervals = {row["name"]: row for row in _read_csv(Path(interval_file))}
        top = max(sobol, key=lambda row: float(row["total_effect"]))
        common = {
            "case": case_path.name,
            "hf_samples": hf_samples,
            "best_lf_model_id": best_lf,
            "mf_oof_rmse": float(metric["rmse"]),
            "mf_oof_r2": r2,
            "quality_tier": tier,
            "eligible_primary_claims": eligible,
            "gsa_status": summary["status"],
            "quality_flags": ";".join(summary.get("quality_flags", [])),
        }
        top_interval = intervals.get(top["name"], {})
        case_rows.append(
            {
                **common,
                "top_source": top["name"],
                "top_first_order": float(top["first_order"]),
                "top_total_effect": float(top["total_effect"]),
                "top_total_q025_refit": top_interval.get("total_q025", ""),
                "top_total_q975_refit": top_interval.get("total_q975", ""),
            }
        )
        for row in sobol:
            interval = intervals.get(row["name"], {})
            source_rows.append(
                {
                    **common,
                    "source": row["name"],
                    "first_order": float(row["first_order"]),
                    "total_effect": float(row["total_effect"]),
                    "first_q025_refit": interval.get("first_q025", ""),
                    "first_q975_refit": interval.get("first_q975", ""),
                    "total_q025_refit": interval.get("total_q025", ""),
                    "total_q975_refit": interval.get("total_q975", ""),
                    "refit_bootstrap_successful": interval.get("bootstrap_repetitions_successful", ""),
                }
            )

    cases_path = target / "gsa_case_summary.csv"
    sources_path = target / "gsa_source_all_cases.csv"
    _write_csv(cases_path, case_rows)
    _write_csv(sources_path, source_rows)
    summary = {
        "status": "complete" if case_rows else "no_complete_cases",
        "campaign_root": str(root),
        "case_count": len(case_rows),
        "primary_case_count": sum(bool(row["eligible_primary_claims"]) for row in case_rows),
        "exploratory_case_count": sum(not bool(row["eligible_primary_claims"]) for row in case_rows),
        "skipped": skipped,
        "case_summary_csv": str(cases_path),
        "source_all_cases_csv": str(sources_path),
    }
    summary_path = target / "gsa_cross_case_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
