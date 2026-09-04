from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from typing import Dict, List, Sequence, Tuple


JOIN_COLUMNS = [
    "study_id",
    "cell_id",
    "phase",
    "qoi",
    "model_id",
    "fidelity",
    "sample_index",
    "sample_fingerprint",
    "request_fingerprint",
]


def _require_columns(header: Sequence[str], required: Sequence[str], path: str) -> None:
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _join_key(row: Dict[str, str], path: str, row_number: int) -> str:
    for name in ("sample_fingerprint", "request_fingerprint"):
        if not str(row.get(name, "")).strip():
            raise ValueError(f"{path}:{row_number} has a blank {name}")
    return json.dumps([row.get(name, "") for name in JOIN_COLUMNS], separators=(",", ":"))


def export_surrogate_dataset(
    sample_inputs_csv: str,
    model_evaluations_csv: str,
    target_csv: str,
    *,
    allow_incomplete: bool = False,
) -> Dict[str, int | str]:
    """Build a provenance-safe, row-level surrogate dataset.

    The join is disk-backed so campaign-sized CSVs do not need to fit in memory.
    Strict mode rejects missing matches in either direction and conflicting input
    rows. The target is atomically replaced only after validation succeeds.
    """
    target_parent = os.path.dirname(os.path.abspath(target_csv))
    os.makedirs(target_parent, exist_ok=True)
    database_fd, database_path = tempfile.mkstemp(prefix="surrogate_join_", suffix=".sqlite3", dir=target_parent)
    os.close(database_fd)
    output_fd, output_path = tempfile.mkstemp(prefix="surrogate_dataset_", suffix=".csv", dir=target_parent)
    os.close(output_fd)

    input_rows = 0
    duplicate_input_rows = 0
    evaluation_rows = 0
    written_rows = 0
    missing_evaluations: List[Tuple[int, str]] = []
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        connection.execute("CREATE TABLE inputs (join_key TEXT PRIMARY KEY, payload TEXT NOT NULL, matched INTEGER NOT NULL DEFAULT 0)")

        with open(sample_inputs_csv, "r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            input_header = list(reader.fieldnames or [])
            _require_columns(input_header, JOIN_COLUMNS, sample_inputs_csv)
            for row_number, row in enumerate(reader, start=2):
                input_rows += 1
                key = _join_key(row, sample_inputs_csv, row_number)
                payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO inputs(join_key, payload) VALUES (?, ?)",
                    (key, payload),
                )
                if cursor.rowcount == 0:
                    duplicate_input_rows += 1
                    existing = connection.execute(
                        "SELECT payload FROM inputs WHERE join_key = ?", (key,)
                    ).fetchone()
                    if existing is None or existing[0] != payload:
                        raise ValueError(
                            f"{sample_inputs_csv}:{row_number} conflicts with an earlier row for the same provenance key"
                        )
        connection.commit()

        with open(model_evaluations_csv, "r", encoding="utf-8", newline="") as evaluations:
            reader = csv.DictReader(evaluations)
            evaluation_header = list(reader.fieldnames or [])
            _require_columns(evaluation_header, JOIN_COLUMNS, model_evaluations_csv)
            appended_columns = [name for name in input_header if name not in evaluation_header]
            output_header = evaluation_header + appended_columns
            with open(output_path, "w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=output_header)
                writer.writeheader()
                for row_number, evaluation in enumerate(reader, start=2):
                    evaluation_rows += 1
                    key = _join_key(evaluation, model_evaluations_csv, row_number)
                    matched = connection.execute(
                        "SELECT payload FROM inputs WHERE join_key = ?", (key,)
                    ).fetchone()
                    if matched is None:
                        if len(missing_evaluations) < 5:
                            missing_evaluations.append((row_number, key))
                        if allow_incomplete:
                            continue
                        continue
                    input_row = json.loads(matched[0])
                    output = dict(evaluation)
                    output.update({name: input_row.get(name, "") for name in appended_columns})
                    writer.writerow(output)
                    written_rows += 1
                    connection.execute("UPDATE inputs SET matched = 1 WHERE join_key = ?", (key,))
        connection.commit()

        unmatched_inputs = int(connection.execute("SELECT COUNT(*) FROM inputs WHERE matched = 0").fetchone()[0])
        if not allow_incomplete and (missing_evaluations or unmatched_inputs):
            details = []
            if missing_evaluations:
                rows = ", ".join(str(row_number) for row_number, _ in missing_evaluations)
                details.append(f"model evaluations without inputs (first rows: {rows})")
            if unmatched_inputs:
                details.append(f"{unmatched_inputs} unique input rows without model evaluations")
            raise ValueError("Incomplete surrogate dataset join: " + "; ".join(details))

        os.replace(output_path, target_csv)
        return {
            "sample_input_rows": input_rows,
            "duplicate_sample_input_rows": duplicate_input_rows,
            "model_evaluation_rows": evaluation_rows,
            "written_rows": written_rows,
            "unmatched_input_rows": unmatched_inputs,
            "target_csv": target_csv,
        }
    finally:
        if connection is not None:
            connection.close()
        for path in (database_path, output_path):
            if os.path.exists(path):
                os.remove(path)
