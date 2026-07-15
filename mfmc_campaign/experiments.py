from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List

from .reproducibility import derive_seed
from .types import ExperimentCell, StudyMode


def _source_names(config: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for block in config.get("sources", {}).get("blocks", []):
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict):
            out.append(str(block.get("name")))
    return out


def _direct_qois(config: Dict[str, Any]) -> List[str]:
    return [q for q in config.get("qois", {}).get("direct", []) if isinstance(q, str)]


def _budget_values(config: Dict[str, Any]) -> List[float]:
    total = config.get("budget", {}).get("total", 100.0)
    if isinstance(total, list):
        return [float(x) for x in total]
    return [float(total)]


def _mode_specific_active_sources(config: Dict[str, Any]) -> Iterable[List[str]]:
    mode = str(config.get("study", {}).get("mode", StudyMode.BASELINE.value))
    source_names = _source_names(config)
    selected = config.get("study", {}).get("active_source_blocks", [])
    selected = selected if selected else source_names

    if mode == StudyMode.BASELINE.value:
        # Baseline: no random source by default unless explicitly requested.
        if config.get("study", {}).get("active_source_blocks"):
            yield list(selected)
        else:
            yield []
        return

    if mode == StudyMode.SOURCE_ISOLATION.value:
        for src in selected:
            yield [src]
        return

    if mode == StudyMode.PAIRWISE_INTERACTION.value:
        pairs = config.get("study", {}).get("pairwise_source_blocks", [])
        if pairs:
            for pair in pairs:
                yield list(pair)
        else:
            for a, b in combinations(selected, 2):
                yield [a, b]
        return

    if mode == StudyMode.MIXED_UNCERTAINTY.value:
        mixed = config.get("study", {}).get("mixed_source_blocks", [])
        if mixed:
            for block in mixed:
                yield list(block)
        else:
            yield list(selected)
        return

    if mode == StudyMode.REGIME_SWEEP.value:
        yield list(selected)
        return

    if mode == StudyMode.GEOMETRY_SWEEP.value:
        yield list(selected)
        return

    if mode == StudyMode.PILOT_ROBUSTNESS.value:
        yield list(selected)
        return

    if mode == StudyMode.PREDICTIVE_DATASET_EXPORT.value:
        yield list(selected)
        return

    yield list(selected)


def generate_experiment_cells(config: Dict[str, Any]) -> List[ExperimentCell]:
    study_id = str(config.get("study", {}).get("id", "mfmc_study"))
    mode = str(config.get("study", {}).get("mode", StudyMode.BASELINE.value))
    repetitions = int(config.get("repetitions", 1))
    global_seed = int(config.get("seeds", {}).get("global", 12345))
    pilot_size_default = int(config.get("pilot", {}).get("size", 32))
    pilot_sizes = [int(x) for x in config.get("pilot", {}).get("sizes", [pilot_size_default])]
    hf_id = str(config.get("models", {}).get("hf", {}).get("id", "hf"))
    lf_models = config.get("models", {}).get("lf", [])
    lf_strategy = str(config.get("models", {}).get("lf_strategy", "separate")).lower()
    use_multi_lf = bool(config.get("models", {}).get("use_multi_lf", False)) or lf_strategy in {
        "multi",
        "multi_lf",
        "paper_mfmc",
        "peherstorfer",
        "optimal_model_management",
        "nested_mfmc",
    }
    if use_multi_lf:
        lf_ids = [str(lf.get("id", "lf")) for lf in lf_models if isinstance(lf, dict)]
        lf_models = [{"id": "+".join(lf_ids)}] if lf_ids else []

    qois = _direct_qois(config)
    budgets = _budget_values(config)
    geometries = config.get("geometries", [])
    regimes = config.get("regimes", [])

    if mode != StudyMode.PILOT_ROBUSTNESS.value:
        pilot_sizes = [pilot_size_default]

    # Explicit sweep behavior: all provided geometries/regimes are iterated in every mode,
    # but these modes are kept explicit for readability and future policy hooks.
    if mode == StudyMode.BASELINE.value:
        budgets = budgets[:1]

    cells: List[ExperimentCell] = []
    for geometry in geometries:
        geometry_id = str(geometry.get("id", geometry.get("name", "geometry")))
        for regime in regimes:
            regime_id = str(regime.get("id", regime.get("label", "regime")))
            for active_sources in _mode_specific_active_sources(config):
                for qoi in qois:
                    for lf in lf_models:
                        lf_id = str(lf.get("id", "lf"))
                        for budget in budgets:
                            for pilot_size in pilot_sizes:
                                for rep in range(repetitions):
                                    seed = derive_seed(
                                        global_seed,
                                        study_id,
                                        mode,
                                        geometry_id,
                                        regime_id,
                                        "+".join(sorted(active_sources)),
                                        qoi,
                                        hf_id,
                                        lf_id,
                                        budget,
                                        pilot_size,
                                        rep,
                                    )
                                    cells.append(
                                        ExperimentCell(
                                            study_id=study_id,
                                            mode=mode,
                                            geometry_id=geometry_id,
                                            regime_id=regime_id,
                                            active_source_blocks=list(active_sources),
                                            qoi=qoi,
                                            hf_model_id=hf_id,
                                            lf_model_id=lf_id,
                                            repetition=rep,
                                            seed=seed,
                                            pilot_size=pilot_size,
                                            budget=float(budget),
                                        )
                                    )

    return cells
