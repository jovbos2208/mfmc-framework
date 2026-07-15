from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class QoIEntry:
    name: str
    quantity_kind: str  # direct | derived
    expression: str = ""
    metadata: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "quantity_kind": self.quantity_kind,
            "expression": self.expression,
            "metadata": dict(self.metadata or {}),
        }


class QoIRegistry:
    def __init__(self, direct: List[str], derived: List[Dict[str, Any]], model_availability: Dict[str, List[str]]):
        self._direct: List[QoIEntry] = [QoIEntry(name=q, quantity_kind="direct", expression="", metadata={}) for q in direct]
        self._derived: List[QoIEntry] = [
            QoIEntry(
                name=str(d.get("name")),
                quantity_kind="derived",
                expression=str(d.get("expression", "")),
                metadata={k: v for k, v in d.items() if k not in {"name", "expression"}},
            )
            for d in derived
            if isinstance(d, dict) and d.get("name")
        ]
        self._model_availability = {str(k): [str(v) for v in vals] for k, vals in model_availability.items()}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QoIRegistry":
        qois = config.get("qois", {}) if isinstance(config.get("qois"), dict) else {}
        direct = [str(q) for q in qois.get("direct", []) if isinstance(q, str)]
        derived = [d for d in qois.get("derived", []) if isinstance(d, dict)]
        model_availability = config.get("models", {}).get("available_qois", {})
        if not isinstance(model_availability, dict):
            model_availability = {}
        return cls(direct=direct, derived=derived, model_availability=model_availability)

    def direct_names(self) -> List[str]:
        return [q.name for q in self._direct]

    def derived_names(self) -> List[str]:
        return [q.name for q in self._derived]

    def all_names(self) -> List[str]:
        return self.direct_names() + self.derived_names()

    def quantity_kind(self, qoi: str) -> str:
        for item in self._direct + self._derived:
            if item.name == qoi:
                return item.quantity_kind
        return "unknown"

    def expression(self, qoi: str) -> str:
        for item in self._derived:
            if item.name == qoi:
                return item.expression
        return ""

    def is_available_for_model(self, model_id: str, qoi: str) -> bool:
        available = self._model_availability.get(model_id)
        if available is None:
            # fallback: direct QoIs are assumed available unless restricted
            return qoi in set(self.direct_names())
        return qoi in set(available)

    def validate(self, hf_model_id: str, lf_model_ids: List[str]) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        if not self._direct:
            errors.append("No direct QoIs defined")

        direct_set = set(self.direct_names())
        if len(direct_set) != len(self._direct):
            errors.append("Duplicate direct QoI names found")

        derived_set = set(self.derived_names())
        if len(derived_set) != len(self._derived):
            errors.append("Duplicate derived QoI names found")

        overlap = direct_set & derived_set
        if overlap:
            errors.append(f"QoIs defined as both direct and derived: {sorted(overlap)}")

        for q in self.direct_names():
            if not self.is_available_for_model(hf_model_id, q):
                errors.append(f"Requested QoI '{q}' unavailable for HF model '{hf_model_id}'")

        for lf_id in lf_model_ids:
            for q in self.direct_names():
                if not self.is_available_for_model(lf_id, q):
                    errors.append(f"Requested QoI '{q}' unavailable for LF model '{lf_id}'")

        for item in self._derived:
            if not item.expression:
                warnings.append(f"Derived QoI '{item.name}' has empty expression metadata")

        return errors, warnings


def build_qoi_registry(config: Dict[str, Any]) -> QoIRegistry:
    return QoIRegistry.from_config(config)
