from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StudyMode(str, Enum):
    BASELINE = "baseline"
    SOURCE_ISOLATION = "source_isolation"
    PAIRWISE_INTERACTION = "pairwise_interaction"
    MIXED_UNCERTAINTY = "mixed_uncertainty"
    REGIME_SWEEP = "regime_sweep"
    GEOMETRY_SWEEP = "geometry_sweep"
    PILOT_ROBUSTNESS = "pilot_robustness"
    PREDICTIVE_DATASET_EXPORT = "predictive_dataset_export"


@dataclass
class SourceBlock:
    name: str
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionDef:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomVariableDef:
    name: str
    source_block: str
    units: str = ""
    distribution: DistributionDef = field(default_factory=lambda: DistributionDef(kind="fixed"))
    transform: Optional[str] = None
    bounds: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    baseline: Any = None
    regime_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RegimeDescriptor:
    regime_id: str
    label: str
    descriptors: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryDescriptor:
    geometry_id: str
    name: str
    characteristic_length: Optional[float] = None
    geometry_class: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QoIDef:
    name: str
    derived: bool = False
    expression: Optional[str] = None
    available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentCell:
    study_id: str
    mode: str
    geometry_id: str
    regime_id: str
    active_source_blocks: List[str]
    qoi: str
    hf_model_id: str
    lf_model_id: str
    repetition: int
    seed: int
    pilot_size: int
    budget: float

    def cell_id(self) -> str:
        sources = "+".join(sorted(self.active_source_blocks)) if self.active_source_blocks else "none"
        return (
            f"{self.study_id}|{self.mode}|{self.geometry_id}|{self.regime_id}|"
            f"{self.qoi}|{self.hf_model_id}|{self.lf_model_id}|src={sources}|"
            f"rep={self.repetition}|pilot={self.pilot_size}|budget={self.budget}"
        )


@dataclass
class EvaluationRequest:
    study_id: str
    cell_id: str
    model_id: str
    fidelity: str
    qois: List[str]
    geometry: GeometryDescriptor
    regime: RegimeDescriptor
    active_source_blocks: List[str]
    sample_ids: List[str]
    samples: List[Dict[str, Any]]
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    values_by_qoi: Dict[str, List[float]]
    costs: List[float]
    sample_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    level: str
    message: str
    path: str
