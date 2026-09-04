from __future__ import annotations

import csv
import json
from pathlib import Path

from mfmc_campaign.legacy_inventory import inventory_legacy_surrogate_data


def test_inventory_uses_existing_reconstruction_audit(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    case = root / "cube_300km"
    case.mkdir(parents=True)
    (case / "config_snapshot.json").write_text("{}", encoding="utf-8")
    (case / "evaluation_cache.json").write_text("{}", encoding="utf-8")
    with (case / "model_evaluations.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["study_id", "sample_fingerprint", "request_fingerprint", "value"])
        writer.writerow(["study", "", "", "1.0"])

    audit_dir = root / "uncertainty_sensitivity_analysis"
    audit_dir.mkdir()
    with (audit_dir / "reconstruction_audit.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["case", "status", "production_samples", "retained_samples"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": "cube_300km",
                "status": "cache_hash_match",
                "production_samples": 100,
                "retained_samples": 80,
            }
        )

    output = tmp_path / "inventory"
    summary = inventory_legacy_surrogate_data(str(root), str(output))

    assert summary["case_count"] == 1
    assert summary["audit_status_counts"] == {"cache_hash_match": 1}
    payload = json.loads((output / "legacy_inventory_summary.json").read_text(encoding="utf-8"))
    assert payload["cases_with_sample_inputs"] == 0
    with (output / "legacy_case_inventory.csv").open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert row["backfill_eligibility"] == "exact_reconstruction_available"
    assert row["audit_retained_samples_sum"] == "80"
