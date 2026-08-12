from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

import numpy as np

from .types import EvaluationRequest


def fingerprint_payload(value: Any) -> Any:
    """Return a deterministic, JSON-serializable representation of *value*."""
    if isinstance(value, dict):
        return {
            str(key): fingerprint_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [fingerprint_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return fingerprint_payload(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def hash_payload(value: Any) -> str:
    payload = json.dumps(
        fingerprint_payload(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_fingerprints(samples: Sequence[Dict[str, Any]]) -> List[str]:
    """Fingerprint physical/sample inputs independently of model and fidelity."""
    return [hash_payload(sample) for sample in samples]


def request_fingerprints(request: EvaluationRequest) -> List[str]:
    """Fingerprint complete model requests, one hash per requested sample."""
    geometry_payload = {
        "geometry_id": request.geometry.geometry_id,
        "name": request.geometry.name,
        "characteristic_length": request.geometry.characteristic_length,
        "geometry_class": request.geometry.geometry_class,
        "tags": request.geometry.tags,
        "metadata": request.geometry.metadata,
    }
    regime_payload = {
        "regime_id": request.regime.regime_id,
        "label": request.regime.label,
        "descriptors": request.regime.descriptors,
        "metadata": request.regime.metadata,
    }
    common = {
        "study_id": request.study_id,
        "model_id": request.model_id,
        "fidelity": request.fidelity,
        "qois": list(request.qois),
        "geometry": geometry_payload,
        "regime": regime_payload,
        "active_source_blocks": sorted(request.active_source_blocks),
        "seed": int(request.seed),
        "metadata": request.metadata,
    }
    return [
        hash_payload(
            {
                **common,
                "sample_id": request.sample_ids[index] if index < len(request.sample_ids) else "",
                "sample": sample,
            }
        )
        for index, sample in enumerate(request.samples)
    ]
