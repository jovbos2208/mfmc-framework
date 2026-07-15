from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import yaml


def _load_data_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported registry file extension: {suffix}")


def _resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _load_list_from_ref(base_dir: str, ref_path: str, expected_key: str) -> List[Dict[str, Any]]:
    path = _resolve_path(base_dir, ref_path)
    data = _load_data_file(path)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        if expected_key in data and isinstance(data[expected_key], list):
            return [x for x in data[expected_key] if isinstance(x, dict)]
        if "items" in data and isinstance(data["items"], list):
            return [x for x in data["items"] if isinstance(x, dict)]

    raise ValueError(f"Registry file '{path}' has unsupported structure for key '{expected_key}'")


def _select_registry_items(
    items: List[Dict[str, Any]],
    selectors: Any,
    *,
    key_candidates: List[str],
    label: str,
) -> List[Dict[str, Any]]:
    if not isinstance(selectors, list) or not selectors:
        return items

    def _selector_key(value: Any) -> str | None:
        if isinstance(value, str):
            token = value.strip()
            return token or None
        if isinstance(value, dict):
            for key in key_candidates:
                token = str(value.get(key, "")).strip()
                if token:
                    return token
        return None

    lookup: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in key_candidates:
            token = str(item.get(key, "")).strip()
            if token and token not in lookup:
                lookup[token] = item

    selected: List[Dict[str, Any]] = []
    missing: List[str] = []
    for selector in selectors:
        sel_key = _selector_key(selector)
        if not sel_key:
            continue
        if sel_key not in lookup:
            missing.append(sel_key)
            continue

        base = dict(lookup[sel_key])
        if isinstance(selector, dict):
            merged = dict(base)
            merged.update(selector)
            if isinstance(base.get("metadata"), dict) or isinstance(selector.get("metadata"), dict):
                merged["metadata"] = dict(base.get("metadata", {}))
                merged["metadata"].update(selector.get("metadata", {}))
            selected.append(merged)
        else:
            selected.append(base)

    if missing:
        raise ValueError(f"Unknown {label} selector(s): {', '.join(missing)}")
    return selected


def hydrate_split_registries(config: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    cfg = dict(config)
    geometry_selectors = cfg.get("geometries")

    refs = cfg.get("registries", {}) if isinstance(cfg.get("registries"), dict) else {}

    sources_ref = cfg.get("sources_ref") or refs.get("sources")
    if sources_ref:
        blocks = _load_list_from_ref(base_dir, str(sources_ref), expected_key="blocks")
        cfg.setdefault("sources", {})
        if isinstance(cfg["sources"], list):
            cfg["sources"] = {"blocks": cfg["sources"]}
        cfg["sources"]["blocks"] = blocks

    regimes_ref = cfg.get("regimes_ref") or refs.get("regimes")
    if regimes_ref:
        cfg["regimes"] = _load_list_from_ref(base_dir, str(regimes_ref), expected_key="regimes")

    geometries_ref = cfg.get("geometries_ref") or refs.get("geometries")
    if geometries_ref:
        geometries = _load_list_from_ref(base_dir, str(geometries_ref), expected_key="geometries")
        cfg["geometries"] = _select_registry_items(
            geometries,
            geometry_selectors,
            key_candidates=["id", "name"],
            label="geometry",
        )

    return cfg


def build_regime_label_map(regimes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for regime in regimes:
        if not isinstance(regime, dict):
            continue
        label = str(regime.get("label", "")).strip()
        rid = str(regime.get("id", "")).strip()
        if label:
            out[label] = regime
        if rid:
            out[rid] = regime
    return out


def validate_regime_labels(regimes: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    seen_label_to_id: Dict[str, str] = {}
    for i, regime in enumerate(regimes):
        if not isinstance(regime, dict):
            errors.append(f"regimes[{i}] is not an object")
            continue

        rid = str(regime.get("id", "")).strip()
        label = str(regime.get("label", "")).strip()

        if not rid:
            errors.append(f"regimes[{i}] missing 'id'")
        if not label:
            warnings.append(f"regimes[{i}] missing 'label' (id will be used as label)")
            label = rid

        if label in seen_label_to_id and seen_label_to_id[label] != rid:
            errors.append(
                f"regime label '{label}' maps to multiple ids ('{seen_label_to_id[label]}' and '{rid}')"
            )
        else:
            seen_label_to_id[label] = rid

    return errors, warnings
