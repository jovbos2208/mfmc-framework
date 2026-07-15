from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from .qoi_registry import build_qoi_registry
from .registries import build_regime_label_map, hydrate_split_registries, validate_regime_labels
from .types import StudyMode, ValidationIssue


ALLOWED_DISTRIBUTIONS = {
    "fixed",
    "uniform",
    "int_uniform",
    "normal",
    "lognormal",
    "choice",
    "empirical",
}

REQUIRED_TOP_LEVEL = [
    "study",
    "geometries",
    "regimes",
    "sources",
    "variables",
    "sampling",
    "models",
    "qois",
    "pilot",
    "budget",
    "repetitions",
    "seeds",
    "outputs",
]

REQUIRED_REGIME_KEYS = {
    "altitude_km",
    "characteristic_length",
    "speed_ratio",
    "freestream_temperature",
    "composition_descriptor",
    "solar_activity_state",
    "geomagnetic_activity_state",
    "wind_state",
    "geometry_class",
    "surface_state",
}


class ConfigValidationError(ValueError):
    pass


def _load_raw(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigValidationError(f"Unsupported config extension: {suffix}")

    if not isinstance(data, dict):
        raise ConfigValidationError("Config root must be a mapping/object")
    return data


def load_config(path: str) -> Dict[str, Any]:
    raw = _load_raw(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    hydrated = hydrate_split_registries(raw, base_dir)
    hydrated["_config_dir"] = base_dir
    return normalize_config(hydrated)


def _resolve_config_path(config: Dict[str, Any], path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    base_dir = config.get("_config_dir")
    if base_dir:
        return os.path.abspath(os.path.join(str(base_dir), path))
    return path


def _optional_str(value: Any) -> str:
    return "" if value is None or value == "" else str(value)


def _mean_hf_cost_from_model_evaluations(config: Dict[str, Any], spec: Dict[str, Any]) -> float:
    path = _optional_str(spec.get("path", spec.get("model_evaluations_csv", "")))
    if not path:
        pilot_dir = config.get("pilot", {}).get("dir", config.get("pilot", {}).get("pilot_dir"))
        if pilot_dir:
            path = os.path.join(str(pilot_dir), "model_evaluations.csv")
    path = _resolve_config_path(config, path)
    if not path or not os.path.exists(path):
        return float("nan")

    model_id = _optional_str(spec.get("model_id", config.get("models", {}).get("hf", {}).get("id", "")))
    phase = _optional_str(spec.get("phase", "pilot_hf"))
    qoi = _optional_str(spec.get("qoi", (config.get("qois", {}).get("direct", [""])[0] or "")))
    geometry_id = _optional_str(spec.get("geometry_id", ""))
    regime_id = _optional_str(spec.get("regime_id", ""))
    cost_columns = [item for item in [_optional_str(spec.get("cost_column", "")), "cost", "HF_cost", "hf_cost"] if item]

    values: List[float] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if phase and _optional_str(row.get("phase")) != phase:
                continue
            if model_id and _optional_str(row.get("model_id")) != model_id:
                continue
            if qoi and _optional_str(row.get("qoi")) != qoi:
                continue
            if geometry_id and _optional_str(row.get("geometry_id")) != geometry_id:
                continue
            if regime_id and _optional_str(row.get("regime_id")) != regime_id:
                continue
            cost = float("nan")
            for cost_column in cost_columns:
                try:
                    cost = float(row.get(cost_column, "nan"))
                except Exception:
                    cost = float("nan")
                if np.isfinite(cost):
                    break
            if np.isfinite(cost) and cost > 0.0:
                values.append(cost)

    return float(np.mean(values)) if values else float("nan")


def _apply_budget_from_model_evaluations(config: Dict[str, Any]) -> None:
    budget = config.get("budget", {})
    if not isinstance(budget, dict):
        return

    spec = budget.get("from_model_evaluations")
    if not isinstance(spec, dict):
        return

    hf_cost = _mean_hf_cost_from_model_evaluations(config, spec)
    if not np.isfinite(hf_cost) or hf_cost <= 0.0:
        return

    metadata = budget.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["measured_mean_hf_cost"] = hf_cost
        metadata["source_model_evaluations_csv"] = spec.get("path", spec.get("model_evaluations_csv", ""))

    multiples = spec.get("multiples")
    if multiples is None and isinstance(metadata, dict):
        multiples = metadata.get("multiples")
    if multiples is None:
        multiples = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0]

    budget["total"] = [float(hf_cost) * float(mult) for mult in multiples]


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)

    cfg.setdefault("study", {})
    cfg["study"].setdefault("id", "mfmc_study")
    cfg["study"].setdefault("mode", StudyMode.BASELINE.value)
    cfg["study"].setdefault("active_source_blocks", [])
    cfg["study"].setdefault("pairwise_source_blocks", [])
    cfg["study"].setdefault("mixed_source_blocks", [])

    cfg.setdefault("geometries", [])
    cfg.setdefault("regimes", [])
    cfg.setdefault("sources", {"blocks": []})
    if isinstance(cfg["sources"], list):
        cfg["sources"] = {"blocks": cfg["sources"]}
    cfg["sources"].setdefault("blocks", [])

    cfg.setdefault("variables", [])
    cfg.setdefault("sampling", {})
    cfg["sampling"].setdefault("method", "independent")
    cfg["sampling"].setdefault("sample_count", 32)
    cfg["sampling"].setdefault("max_production_samples", 2000)
    cfg["sampling"].setdefault("block_covariances", {})

    cfg.setdefault("models", {})
    cfg["models"].setdefault("hf", {})
    cfg["models"].setdefault("lf", [])
    cfg["models"].setdefault("available_qois", {})

    cfg.setdefault("qois", {})
    cfg["qois"].setdefault("direct", ["C_D", "C_D2"])
    cfg["qois"].setdefault("derived", [{"name": "Var_C_D", "expression": "E[C_D2]-E[C_D]^2"}])

    cfg.setdefault("pilot", {})
    cfg["pilot"].setdefault("size", 32)
    cfg["pilot"].setdefault("sizes", [8, 16, 32, 64])
    cfg["pilot"].setdefault("robustness_repetitions", 20)

    cfg.setdefault("budget", {})
    cfg["budget"].setdefault("total", 100.0)
    cfg["budget"].setdefault("hf_fraction", 0.25)
    _apply_budget_from_model_evaluations(cfg)

    cfg.setdefault("repetitions", 1)

    cfg.setdefault("seeds", {})
    cfg["seeds"].setdefault("global", 12345)

    cfg.setdefault("outputs", {})
    cfg["outputs"].setdefault("dir", "campaign_outputs/default")
    cfg["outputs"].setdefault("write_parquet", False)
    cfg["outputs"].setdefault("write_model_evaluations", True)
    cfg["outputs"].setdefault("write_config_snapshot", True)
    cfg["outputs"].setdefault("plots", True)

    cfg.setdefault("execution", {})
    cfg["execution"].setdefault("backend", "mock")
    cfg["execution"].setdefault("resume", False)
    if not isinstance(cfg["execution"].get("environment"), dict):
        cfg["execution"]["environment"] = {}
    cfg["execution"]["environment"].setdefault("model", "csv")

    # Helper map used by study selection and validation.
    label_map = build_regime_label_map(cfg.get("regimes", []))
    cfg["regime_label_map"] = {
        k: str(v.get("id", v.get("label", ""))) for k, v in label_map.items() if isinstance(v, dict)
    }

    return cfg


def _source_block_names(config: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for block in config.get("sources", {}).get("blocks", []):
        if isinstance(block, str):
            names.append(block)
        elif isinstance(block, dict) and "name" in block:
            names.append(str(block["name"]))
    return names


def _validate_distribution_shape(kind: str, params: Dict[str, Any]) -> bool:
    if kind == "fixed":
        return "value" in params
    if kind in {"uniform", "int_uniform"}:
        return "low" in params and "high" in params
    if kind == "normal":
        return "mean" in params and "std" in params
    if kind == "lognormal":
        return "mean" in params and "sigma" in params
    if kind == "choice":
        return "values" in params
    if kind == "empirical":
        return "values" in params
    return True


def _validate_covariances(config: Dict[str, Any]) -> List[ValidationIssue]:
    errors: List[ValidationIssue] = []
    cov_cfg = config.get("sampling", {}).get("block_covariances", {})
    if not isinstance(cov_cfg, dict):
        return [ValidationIssue("error", "sampling.block_covariances must be a mapping", "sampling.block_covariances")]

    for block_name, entry in cov_cfg.items():
        path = f"sampling.block_covariances.{block_name}"
        if not isinstance(entry, dict):
            errors.append(ValidationIssue("error", "block covariance entry must be a mapping", path))
            continue

        var_names = entry.get("variables", [])
        matrix = entry.get("matrix", [])
        if not isinstance(var_names, list) or len(var_names) == 0:
            errors.append(ValidationIssue("error", "covariance variables must be non-empty list", path + ".variables"))
            continue

        arr = np.asarray(matrix, dtype=float)
        n = len(var_names)
        if arr.shape != (n, n):
            errors.append(
                ValidationIssue(
                    "error",
                    f"covariance matrix shape must be ({n},{n}), got {arr.shape}",
                    path + ".matrix",
                )
            )
            continue

        if not np.allclose(arr, arr.T, atol=1e-10):
            errors.append(ValidationIssue("error", "covariance matrix must be symmetric", path + ".matrix"))
            continue

        eigvals = np.linalg.eigvalsh(arr)
        if np.min(eigvals) < -1e-10:
            errors.append(
                ValidationIssue(
                    "error",
                    f"covariance matrix must be PSD; minimum eigenvalue is {np.min(eigvals):.3e}",
                    path + ".matrix",
                )
            )

    return errors


def validate_config(config: Dict[str, Any]) -> Tuple[List[ValidationIssue], List[ValidationIssue]]:
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            errors.append(ValidationIssue("error", f"Missing top-level block '{key}'", key))

    mode = str(config.get("study", {}).get("mode", ""))
    if mode not in {m.value for m in StudyMode}:
        errors.append(ValidationIssue("error", f"Unknown study mode '{mode}'", "study.mode"))

    source_names = set(_source_block_names(config))
    if not source_names:
        warnings.append(ValidationIssue("warning", "No source blocks defined", "sources.blocks"))

    selected_sources = config.get("study", {}).get("active_source_blocks", [])
    for i, src in enumerate(selected_sources):
        if src not in source_names:
            errors.append(
                ValidationIssue(
                    "error",
                    f"Unknown active source block '{src}'",
                    f"study.active_source_blocks[{i}]",
                )
            )

    regimes = config.get("regimes", [])
    label_errors, label_warnings = validate_regime_labels(regimes)
    errors.extend(ValidationIssue("error", msg, "regimes") for msg in label_errors)
    warnings.extend(ValidationIssue("warning", msg, "regimes") for msg in label_warnings)

    regime_ids = {
        str(r.get("id", r.get("label", ""))): r
        for r in regimes
        if isinstance(r, dict)
    }

    for i, regime in enumerate(regimes):
        if not isinstance(regime, dict):
            errors.append(ValidationIssue("error", "regime must be mapping", f"regimes[{i}]"))
            continue

        descriptors = regime.get("descriptors", {})
        if not isinstance(descriptors, dict):
            errors.append(
                ValidationIssue(
                    "error",
                    f"Regime '{regime.get('id', i)}' descriptors must be mapping",
                    f"regimes[{i}].descriptors",
                )
            )
            continue

        missing = sorted(REQUIRED_REGIME_KEYS - set(descriptors.keys()))
        if missing:
            errors.append(
                ValidationIssue(
                    "error",
                    f"Regime '{regime.get('id', i)}' missing required descriptors: {', '.join(missing)}",
                    f"regimes[{i}].descriptors",
                )
            )
        if "knudsen_number" not in descriptors and "knudsen_proxy" not in descriptors:
            errors.append(
                ValidationIssue(
                    "error",
                    f"Regime '{regime.get('id', i)}' must include 'knudsen_number' or 'knudsen_proxy'",
                    f"regimes[{i}].descriptors",
                )
            )

    variables = config.get("variables", [])
    seen_names: set[str] = set()
    for i, var in enumerate(variables):
        if not isinstance(var, dict):
            errors.append(ValidationIssue("error", "Variable must be mapping", f"variables[{i}]"))
            continue

        name = str(var.get("name", ""))
        if not name:
            errors.append(ValidationIssue("error", "Variable missing name", f"variables[{i}].name"))
        elif name in seen_names:
            errors.append(ValidationIssue("error", f"Duplicate variable name '{name}'", f"variables[{i}].name"))
        else:
            seen_names.add(name)

        src = var.get("source_block")
        if src not in source_names:
            errors.append(
                ValidationIssue(
                    "error",
                    f"Variable '{name}' references unknown source block '{src}'",
                    f"variables[{i}].source_block",
                )
            )

        dist = var.get("distribution", {})
        kind = dist.get("kind") if isinstance(dist, dict) else None
        params = dist.get("params", {}) if isinstance(dist, dict) else {}
        if kind not in ALLOWED_DISTRIBUTIONS:
            errors.append(
                ValidationIssue(
                    "error",
                    f"Variable '{name}' uses unsupported distribution '{kind}'",
                    f"variables[{i}].distribution.kind",
                )
            )
        elif not _validate_distribution_shape(str(kind), params if isinstance(params, dict) else {}):
            errors.append(
                ValidationIssue(
                    "error",
                    f"Variable '{name}' has malformed distribution parameters for '{kind}'",
                    f"variables[{i}].distribution.params",
                )
            )

        bounds = var.get("bounds")
        if bounds is not None:
            if not (isinstance(bounds, list) and len(bounds) == 2 and bounds[0] <= bounds[1]):
                errors.append(
                    ValidationIssue(
                        "error",
                        f"Variable '{name}' has malformed bounds {bounds}",
                        f"variables[{i}].bounds",
                    )
                )

        overrides = var.get("regime_overrides", {})
        if overrides and not isinstance(overrides, dict):
            errors.append(
                ValidationIssue(
                    "error",
                    f"Variable '{name}' regime_overrides must be mapping",
                    f"variables[{i}].regime_overrides",
                )
            )
            continue

        for reg_key, override in overrides.items():
            if reg_key not in regime_ids and reg_key not in config.get("regime_label_map", {}):
                errors.append(
                    ValidationIssue(
                        "error",
                        f"Variable '{name}' override references unknown regime '{reg_key}'",
                        f"variables[{i}].regime_overrides.{reg_key}",
                    )
                )
                continue
            if not isinstance(override, dict):
                errors.append(
                    ValidationIssue(
                        "error",
                        f"Variable '{name}' override for regime '{reg_key}' must be mapping",
                        f"variables[{i}].regime_overrides.{reg_key}",
                    )
                )
                continue
            if "distribution" not in override and "baseline" not in override:
                errors.append(
                    ValidationIssue(
                        "error",
                        f"Variable '{name}' override for regime '{reg_key}' must define distribution and/or baseline",
                        f"variables[{i}].regime_overrides.{reg_key}",
                    )
                )
            if "distribution" in override:
                od = override.get("distribution", {})
                if not isinstance(od, dict):
                    errors.append(
                        ValidationIssue(
                            "error",
                            f"Variable '{name}' override distribution for regime '{reg_key}' must be mapping",
                            f"variables[{i}].regime_overrides.{reg_key}.distribution",
                        )
                    )
                else:
                    okind = od.get("kind")
                    oparams = od.get("params", {}) if isinstance(od.get("params", {}), dict) else {}
                    if okind not in ALLOWED_DISTRIBUTIONS:
                        errors.append(
                            ValidationIssue(
                                "error",
                                f"Variable '{name}' override distribution kind '{okind}' is unsupported",
                                f"variables[{i}].regime_overrides.{reg_key}.distribution.kind",
                            )
                        )
                    elif not _validate_distribution_shape(str(okind), oparams):
                        errors.append(
                            ValidationIssue(
                                "error",
                                f"Variable '{name}' override has malformed parameters for '{okind}'",
                                f"variables[{i}].regime_overrides.{reg_key}.distribution.params",
                            )
                        )

    errors.extend(_validate_covariances(config))

    hf_model = config.get("models", {}).get("hf", {})
    hf_id = str(hf_model.get("id", "hf"))
    lf_models = config.get("models", {}).get("lf", [])
    lf_ids = [str(m.get("id", "lf")) for m in lf_models if isinstance(m, dict)]

    qoi_registry = build_qoi_registry(config)
    q_errors, q_warnings = qoi_registry.validate(hf_model_id=hf_id, lf_model_ids=lf_ids)
    errors.extend(ValidationIssue("error", msg, "qois") for msg in q_errors)
    warnings.extend(ValidationIssue("warning", msg, "qois") for msg in q_warnings)

    env_cfg = config.get("execution", {}).get("environment", {})
    if env_cfg and not isinstance(env_cfg, dict):
        errors.append(ValidationIssue("error", "execution.environment must be a mapping", "execution.environment"))
    elif isinstance(env_cfg, dict):
        model = str(env_cfg.get("model", "csv"))
        if model not in {"csv", "pymsis_hwm14"}:
            errors.append(
                ValidationIssue(
                    "error",
                    f"Unsupported execution.environment.model '{model}' (expected 'csv' or 'pymsis_hwm14')",
                    "execution.environment.model",
                )
            )

    if mode == StudyMode.PAIRWISE_INTERACTION.value:
        pairs = config.get("study", {}).get("pairwise_source_blocks", [])
        if not pairs and len(selected_sources) < 2:
            errors.append(
                ValidationIssue(
                    "error",
                    "pairwise_interaction mode requires pairwise_source_blocks or at least 2 active sources",
                    "study",
                )
            )
        for i, pair in enumerate(pairs):
            if not isinstance(pair, list) or len(pair) != 2:
                errors.append(
                    ValidationIssue(
                        "error",
                        f"Pairwise entry {pair} must contain exactly 2 source blocks",
                        f"study.pairwise_source_blocks[{i}]",
                    )
                )
            else:
                for src in pair:
                    if src not in source_names:
                        errors.append(
                            ValidationIssue(
                                "error",
                                f"Pairwise source '{src}' not in source taxonomy",
                                f"study.pairwise_source_blocks[{i}]",
                            )
                        )

    if mode == StudyMode.MIXED_UNCERTAINTY.value:
        mixed_blocks = config.get("study", {}).get("mixed_source_blocks", [])
        if mixed_blocks:
            if not isinstance(mixed_blocks, list):
                errors.append(
                    ValidationIssue(
                        "error",
                        "study.mixed_source_blocks must be a list of source-block lists",
                        "study.mixed_source_blocks",
                    )
                )
            else:
                for i, block in enumerate(mixed_blocks):
                    if not isinstance(block, list) or len(block) == 0:
                        errors.append(
                            ValidationIssue(
                                "error",
                                "Each mixed-source entry must be a non-empty list",
                                f"study.mixed_source_blocks[{i}]",
                            )
                        )
                        continue
                    for src in block:
                        if src not in source_names:
                            errors.append(
                                ValidationIssue(
                                    "error",
                                    f"Mixed source '{src}' not in source taxonomy",
                                    f"study.mixed_source_blocks[{i}]",
                                )
                            )

    if mode == StudyMode.REGIME_SWEEP.value and len(regimes) < 2:
        warnings.append(ValidationIssue("warning", "regime_sweep mode configured with <2 regimes", "regimes"))
    if mode == StudyMode.GEOMETRY_SWEEP.value and len(config.get("geometries", [])) < 2:
        warnings.append(ValidationIssue("warning", "geometry_sweep mode configured with <2 geometries", "geometries"))

    if not qoi_registry.all_names():
        errors.append(ValidationIssue("error", "No QoIs requested", "qois"))

    return errors, warnings


def validate_or_raise(config: Dict[str, Any]) -> Dict[str, Any]:
    errors, warnings = validate_config(config)
    if errors:
        details = "\n".join(f"- [{e.path}] {e.message}" for e in errors)
        raise ConfigValidationError(f"Config validation failed:\n{details}")

    if warnings:
        _ = warnings

    return config


def load_and_validate(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    return validate_or_raise(cfg)
