from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _case_directories(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "config_snapshot.json").exists() and (path / "model_evaluations.csv").exists():
            yield path


def _read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return next(csv.reader(source), [])


def _load_audit(path: Path | None) -> Dict[str, List[Dict[str, str]]]:
    by_case: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if path is None or not path.exists():
        return by_case
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            case = str(row.get("case", "")).strip()
            if case:
                by_case[case].append(row)
    return by_case


def _integer(row: Dict[str, str], name: str) -> int:
    try:
        return int(float(row.get(name, "0") or 0))
    except (TypeError, ValueError):
        return 0


def inventory_legacy_surrogate_data(
    campaign_root: str,
    output_dir: str,
    *,
    reconstruction_audit_csv: str | None = None,
) -> Dict[str, Any]:
    """Inventory legacy campaign artifacts without loading multi-GB result CSVs."""
    root = Path(campaign_root).resolve()
    target = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Campaign root does not exist: {root}")
    target.mkdir(parents=True, exist_ok=True)

    audit_path = (
        Path(reconstruction_audit_csv).resolve()
        if reconstruction_audit_csv
        else root / "uncertainty_sensitivity_analysis" / "reconstruction_audit.csv"
    )
    audit_by_case = _load_audit(audit_path)
    rows: List[Dict[str, Any]] = []
    total_status = Counter()

    for case_dir in _case_directories(root):
        evaluation_path = case_dir / "model_evaluations.csv"
        sample_inputs_path = case_dir / "sample_inputs.csv"
        cache_path = case_dir / "evaluation_cache.json"
        header = _read_header(evaluation_path)
        audit_rows = audit_by_case.get(case_dir.name, [])
        statuses = Counter(str(row.get("status", "")) for row in audit_rows)
        total_status.update(statuses)
        mismatch_count = sum(count for status, count in statuses.items() if "mismatch" in status)
        exact_count = statuses.get("cache_hash_match", 0)
        sequence_only_count = sum(
            count for status, count in statuses.items() if status.startswith("sequence_reconstructed")
        )
        if mismatch_count:
            eligibility = "blocked_hash_mismatch"
        elif exact_count and sequence_only_count:
            eligibility = "mixed_exact_and_sequence_only"
        elif exact_count:
            eligibility = "exact_reconstruction_available"
        elif sequence_only_count:
            eligibility = "sequence_only_requires_environment_audit"
        else:
            eligibility = "not_audited"

        rows.append(
            {
                "case": case_dir.name,
                "model_evaluations_bytes": evaluation_path.stat().st_size,
                "has_evaluation_cache": int(cache_path.exists()),
                "has_sample_inputs": int(sample_inputs_path.exists()),
                "has_sample_fingerprint_column": int("sample_fingerprint" in header),
                "has_request_fingerprint_column": int("request_fingerprint" in header),
                "audit_requests": len(audit_rows),
                "cache_hash_matches": exact_count,
                "sequence_only_requests": sequence_only_count,
                "cache_hash_mismatches": mismatch_count,
                "audit_production_samples_sum": sum(_integer(row, "production_samples") for row in audit_rows),
                "audit_retained_samples_sum": sum(_integer(row, "retained_samples") for row in audit_rows),
                "backfill_eligibility": eligibility,
            }
        )

    if not rows:
        raise RuntimeError(f"No legacy campaign cases found below {root}")

    csv_path = target / "legacy_case_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, Any] = {
        "campaign_root": str(root),
        "reconstruction_audit_csv": str(audit_path) if audit_path.exists() else None,
        "case_count": len(rows),
        "model_evaluations_bytes": sum(int(row["model_evaluations_bytes"]) for row in rows),
        "cases_with_sample_inputs": sum(int(row["has_sample_inputs"]) for row in rows),
        "audit_status_counts": dict(sorted(total_status.items())),
        "backfill_eligibility_counts": dict(Counter(str(row["backfill_eligibility"]) for row in rows)),
        "pilot_policy": "exclude legacy externally reused pilot rows unless exact inputs can be independently audited",
        "inventory_csv": str(csv_path),
    }
    json_path = target / "legacy_inventory_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json"] = str(json_path)
    return summary
