from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mfmc_campaign.fingerprints import request_fingerprints, sample_fingerprints
from mfmc_campaign.output import ResultStore
from mfmc_campaign.surrogate_dataset import export_surrogate_dataset
from mfmc_campaign.types import EvaluationRequest, GeometryDescriptor, RegimeDescriptor


def _request(samples: list[dict[str, float]]) -> EvaluationRequest:
    return EvaluationRequest(
        study_id="study",
        cell_id="cell",
        model_id="hf",
        fidelity="high",
        qois=["drag"],
        geometry=GeometryDescriptor(
            geometry_id="shape",
            name="Shape",
            characteristic_length=1.5,
            metadata={"nose_radius": 0.2},
        ),
        regime=RegimeDescriptor(regime_id="vleo", label="VLEO", descriptors={"altitude_km": 250.0}),
        active_source_blocks=["atmosphere"],
        sample_ids=[f"sample-{index}" for index in range(len(samples))],
        samples=samples,
        seed=42,
    )


def _base_row(request: EvaluationRequest, index: int) -> dict[str, object]:
    sample_hash = sample_fingerprints(request.samples)[index]
    request_hash = request_fingerprints(request)[index]
    return {
        "study_id": "study",
        "cell_id": "cell",
        "phase": "pilot_hf",
        "mode": "baseline",
        "geometry_id": "shape",
        "regime_id": "vleo",
        "active_sources": ["atmosphere"],
        "qoi": "drag",
        "model_id": "hf",
        "fidelity": "high",
        "hf_model_id": "hf",
        "lf_model_id": "lf",
        "pilot_size": 2,
        "budget": 10,
        "repetition": 0,
        "seed": 42,
        "sample_id": request.sample_ids[index],
        "sample_index": index,
        "sample_fingerprint": sample_hash,
        "request_fingerprint": request_hash,
    }


def test_fingerprints_are_stable_and_separate_sample_from_request() -> None:
    samples = [{"density": 1.0, "temperature": 900.0}]
    first = _request(samples)
    second = _request([{"temperature": 900.0, "density": 1.0}])

    assert sample_fingerprints(first.samples) == sample_fingerprints(second.samples)
    assert request_fingerprints(first) == request_fingerprints(second)

    second.model_id = "lf"
    assert sample_fingerprints(first.samples) == sample_fingerprints(second.samples)
    assert request_fingerprints(first) != request_fingerprints(second)


def test_export_surrogate_dataset_joins_inputs_and_outputs(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path))
    request = _request([{"density": 1.0}, {"density": 2.0}])
    for index, sample in enumerate(request.samples):
        row = _base_row(request, index)
        store.append_sample_inputs(
            [
                {
                    **row,
                    "geometry_characteristic_length": 1.5,
                    "input__density": sample["density"],
                    "geometry__nose_radius": 0.2,
                }
            ]
        )
        store.append_model_evaluation({**row, "value": 10.0 + index, "cost": 2.0})

    target = tmp_path / "surrogate_dataset.csv"
    summary = export_surrogate_dataset(store.sample_inputs_csv, store.model_evaluations_csv, str(target))

    assert summary["written_rows"] == 2
    with target.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["input__density"] for row in rows] == ["1.0", "2.0"]
    assert all(row["geometry__nose_radius"] == "0.2" for row in rows)


def test_export_surrogate_dataset_rejects_blank_fingerprints(tmp_path: Path) -> None:
    store = ResultStore(str(tmp_path))
    request = _request([{"density": 1.0}])
    row = _base_row(request, 0)
    store.append_sample_inputs([{**row, "geometry_characteristic_length": 1.5, "input__density": 1.0}])
    store.append_model_evaluation({**row, "request_fingerprint": "", "value": 10.0, "cost": 2.0})

    with pytest.raises(ValueError, match="blank request_fingerprint"):
        export_surrogate_dataset(
            store.sample_inputs_csv,
            store.model_evaluations_csv,
            str(tmp_path / "surrogate_dataset.csv"),
        )
