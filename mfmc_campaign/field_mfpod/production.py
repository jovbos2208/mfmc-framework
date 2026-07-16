"""Restartable solver-production orchestration for full-field MFMC/POD."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters import build_adapter_registry, make_request
from ..adbsat_surface_mapping import (
    build_and_write_adbsat_surface,
    load_surface_mapping,
    write_adbsat_mat,
    write_adbsat_obj,
)
from ..sampling import InputModel, SamplingContext
from .config import MFPODConfig
from .models import MFPODError, jsonable
from .snapshots import PICLASIdentitySurfaceAdapter
from .workflow import field_pilot, optimal_allocation, run_all


_STAGES = ("plan", "pilot", "allocation", "production", "analysis")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def _production_settings(cfg: MFPODConfig) -> dict[str, Any]:
    settings = cfg.raw.get("production", {}) or {}
    if not bool(settings.get("enabled", False)):
        raise MFPODError("Set production.enabled=true before running solver production")
    required = ("geometry", "regime", "variables", "models")
    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise MFPODError(f"production configuration is missing {missing}")
    return settings


def _model_ids(cfg: MFPODConfig, settings: dict[str, Any]) -> dict[str, str]:
    configured = ("DSMC", *tuple(str(x).upper() for x in cfg.raw.get("control_variates", ["TPMC"])))
    explicit = {str(k).upper(): str(v) for k, v in (settings.get("model_ids", {}) or {}).items()}
    models = settings["models"]
    explicit.setdefault("DSMC", str(models.get("hf", {}).get("id", "DSMC")))
    low_fidelity = list(models.get("lf", []))
    for fidelity in configured[1:]:
        if fidelity in explicit:
            continue
        matches = [str(item.get("id")) for item in low_fidelity if str(item.get("id", "")).upper() == fidelity]
        if len(matches) != 1:
            raise MFPODError(f"Set production.model_ids.{fidelity} to one configured LF model id")
        explicit[fidelity] = matches[0]
    missing = [name for name in configured if name not in explicit]
    if missing:
        raise MFPODError(f"No production model id configured for {missing}")
    return {name: explicit[name] for name in configured}


def _maximum_counts(cfg: MFPODConfig, settings: dict[str, Any]) -> dict[str, int]:
    configured = _model_ids(cfg, settings)
    raw = settings.get("maximum_counts", {}) or {}
    counts = {name: int(raw.get(name, raw.get(name.lower(), 0))) for name in configured}
    if any(value <= 0 for value in counts.values()):
        constraints = cfg.raw.get("allocation_constraints", {}) or {}
        fallback = constraints.get("maximum_counts", {}) or {}
        counts = {name: max(counts[name], int(fallback.get(name, 0))) for name in configured}
    if any(value <= 0 for value in counts.values()):
        raise MFPODError("production.maximum_counts must provide positive caps for every configured fidelity")
    return counts


def _sample_plan(cfg: MFPODConfig, settings: dict[str, Any]) -> dict[str, Any]:
    production_dir = cfg.output_dir / "production"
    path = production_dir / "sample_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pilot_count = int(cfg.raw.get("pilot", {}).get("paired_samples", 12))
    reference_count = int(cfg.raw.get("reference_samples", 16))
    maximum = _maximum_counts(cfg, settings)
    stream_count = max(maximum.values())
    total = pilot_count + reference_count + stream_count
    rng = np.random.default_rng(int(settings.get("random_seed", cfg.random_seed)))
    regime = settings["regime"]
    regime_id = str(regime.get("id", regime.get("label", "regime")))
    context = SamplingContext(
        regime_id=regime_id,
        active_source_blocks=list(settings.get("active_source_blocks", [])),
    )
    model = InputModel(
        list(settings.get("variables", [])),
        dict(settings.get("sampling", {"method": "independent"})),
    )
    samples = model.sample(total, context, rng)
    for index, sample in enumerate(samples):
        # The legacy solver launchers use this value in their job-directory
        # names.  Independent sampling does not otherwise create it.
        sample["database_index"] = index
    ids = [f"mfpod_{index:05d}" for index in range(total)]
    plan = {
        "random_seed": int(settings.get("random_seed", cfg.random_seed)),
        "sample_ids": ids,
        "samples": samples,
        "roles": {
            "pilot": ids[:pilot_count],
            "reference_DSMC": ids[pilot_count:pilot_count + reference_count],
            "production_stream": ids[pilot_count + reference_count:],
        },
        "maximum_counts": maximum,
        "nested_stream": True,
    }
    _write_json(path, plan)
    return plan


def _archive_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with np.load(path, allow_pickle=False) as data:
        return {str(value) for value in np.asarray(data["sample_id"]).tolist()}


def _adapter_config(settings: dict[str, Any]) -> dict[str, Any]:
    execution = deepcopy(settings.get("execution", {}) or {})
    execution.setdefault("backend", str(settings.get("backend", "legacy_slurm")))
    return {
        "execution": execution,
        "models": deepcopy(settings["models"]),
        "qois": {"direct": list(settings.get("qois", ["C_D", "C_D2"]))},
    }


def _prepare_adbsat_runtime(settings: dict[str, Any]) -> None:
    runtime = settings.get("adbsat_runtime", {}) or {}
    if not runtime:
        return
    source = Path(str(runtime.get("source_base_dir", "ADBSat-PyVersion")))
    destination = Path(str(runtime.get("base_dir", "outputs/mfpod_runtime/ADBSat-PyVersion")))
    if not source.is_dir():
        raise MFPODError(f"ADBSat source runtime is missing: {source}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("MFMC_Jobs_*", "__pycache__", "*.pyc"),
        )
    geometry_name = str(settings["geometry"].get("id", settings["geometry"].get("name", "Cube")))
    for model in settings["models"].get("lf", []):
        if str(model.get("kind", "")) != "legacy_adbsat":
            continue
        kwargs = model.get("kwargs", {}) or {}
        surface = kwargs.get("surface_archive", {}) or {}
        mapping_path = Path(str(surface.get("mapping_path", "")))
        if not mapping_path.is_file():
            continue
        mapping = load_surface_mapping(mapping_path)
        base_dir = Path(str(kwargs.get("base_dir", destination)))
        write_adbsat_obj(base_dir / "inou" / "obj_files" / f"{geometry_name}.obj", mapping)
        write_adbsat_mat(base_dir / "inou" / "models" / f"{geometry_name}.mat", mapping)


def _unmapped_adbsat_models(settings: dict[str, Any]) -> set[str]:
    missing = set()
    for model in settings["models"].get("lf", []):
        if str(model.get("kind", "")) != "legacy_adbsat":
            continue
        surface = (model.get("kwargs", {}) or {}).get("surface_archive", {}) or {}
        mapping = surface.get("mapping_path")
        if not mapping or not Path(str(mapping)).is_file():
            missing.add(str(model.get("id")))
    return missing


def _validate_solver_inputs(settings: dict[str, Any]) -> None:
    missing: list[str] = []
    for executable in (Path("piclas/piclas"), Path("piclas/piclas2vtk")):
        if not executable.is_file():
            missing.append(f"PICLAS executable: {executable}")
    model_configs = [settings["models"].get("hf", {}), *settings["models"].get("lf", [])]
    for model in model_configs:
        if str(model.get("kind", "")) != "legacy_adbsat":
            continue
        kwargs = model.get("kwargs", {}) or {}
        base_dir = Path(str(kwargs.get("base_dir", "ADBSat-PyVersion")))
        if not base_dir.is_dir():
            missing.append(f"ADBSat base directory: {base_dir}")
    if missing:
        details = "\n  - ".join(missing)
        raise MFPODError(
            "Production solver prerequisites are missing:\n  - " + details +
            "\nRun scripts/configure_piclas.sh before resuming."
        )


def _registry_without_models(settings: dict[str, Any], excluded_ids: set[str]):
    adapter_cfg = _adapter_config(settings)
    adapter_cfg["models"]["lf"] = [
        model for model in adapter_cfg["models"].get("lf", [])
        if str(model.get("id")) not in excluded_ids
    ]
    return build_adapter_registry(adapter_cfg)


def _build_missing_adbsat_mappings(
    cfg: MFPODConfig,
    settings: dict[str, Any],
    model_ids: set[str],
) -> None:
    with np.load(cfg.archives["DSMC"], allow_pickle=False) as archive:
        job_subdirs = [Path(str(value)) for value in np.asarray(archive.get("job_subdir", [])).tolist()]
    canonical_vtu = next(
        (path for directory in job_subdirs for path in sorted(directory.glob("output*.vtu"))),
        None,
    )
    if canonical_vtu is None:
        raise MFPODError(
            "Cannot build the Sentman mapping automatically: no archived DSMC job_subdir contains output*.vtu"
        )
    geometry_name = str(settings["geometry"].get("id", settings["geometry"].get("name", "Cube")))
    for model in settings["models"].get("lf", []):
        if str(model.get("id")) not in model_ids:
            continue
        kwargs = model.get("kwargs", {}) or {}
        base_dir = Path(str(kwargs.get("base_dir", "ADBSat-PyVersion")))
        surface = kwargs.get("surface_archive", {}) or {}
        mapping_path = Path(str(surface["mapping_path"]))
        build_and_write_adbsat_surface(
            canonical_vtu,
            base_dir / "inou" / "obj_files" / f"{geometry_name}.obj",
            base_dir / "inou" / "models" / f"{geometry_name}.mat",
            mapping_path,
        )


def _request_metadata(cfg: MFPODConfig, settings: dict[str, Any]) -> dict[str, Any]:
    execution = settings.get("execution", {}) or {}
    metadata = {
        "case_name": cfg.case_name,
        "environment_model": (execution.get("environment", {}) or {}).get("model", "csv"),
    }
    for key in ("flow_zero_direction", "flow_zero_direction_xyz", "hf_mesh", "aos_deg", "aoa_deg"):
        if key in execution:
            metadata[key] = execution[key]
    return metadata


def _run_fidelity(
    cfg: MFPODConfig,
    settings: dict[str, Any],
    registry: Any,
    model_ids: dict[str, str],
    fidelity: str,
    sample_ids: list[str],
    sample_lookup: dict[str, dict[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    archive = cfg.archives[fidelity]
    complete = _archive_ids(archive)
    pending = [sample_id for sample_id in sample_ids if sample_id not in complete]
    if not pending:
        return {"fidelity": fidelity, "requested": len(sample_ids), "submitted": 0, "status": "already-complete"}
    model_id = model_ids[fidelity]
    adapter = registry.get(model_id)
    request = make_request(
        study_id=str(settings.get("study_id", "field_mfpod_production")),
        cell_id=f"{cfg.case_name}|{phase}|{fidelity}",
        model_id=model_id,
        fidelity="hf" if fidelity == "DSMC" else "lf",
        qois=list(settings.get("qois", ["C_D", "C_D2"])),
        geometry=dict(settings["geometry"]),
        regime=dict(settings["regime"]),
        active_source_blocks=list(settings.get("active_source_blocks", [])),
        sample_ids=pending,
        samples=[sample_lookup[sample_id] for sample_id in pending],
        seed=int(settings.get("random_seed", cfg.random_seed)),
        metadata=_request_metadata(cfg, settings),
    )
    result = adapter.evaluate(request)
    missing = set(pending) - _archive_ids(archive)
    if missing:
        raise MFPODError(f"{fidelity} completed without archiving sample ids {sorted(missing)[:5]}")
    finite_costs = np.asarray(result.costs, dtype=float)
    finite_costs = finite_costs[np.isfinite(finite_costs) & (finite_costs > 0.0)]
    return {
        "fidelity": fidelity,
        "requested": len(sample_ids),
        "submitted": len(pending),
        "successful_results": len(result.sample_ids),
        "mean_cost": float(np.mean(finite_costs)) if finite_costs.size else None,
        "status": "complete",
    }


def _pilot_archive(cfg: MFPODConfig, pilot_ids: list[str]) -> Path:
    models = ("DSMC", *tuple(str(x).upper() for x in cfg.raw.get("control_variates", ["TPMC"])))
    adapter = PICLASIdentitySurfaceAdapter(cfg.archives, cfg.raw.get("topology_tolerance"))
    payload: dict[str, np.ndarray] = {}
    for fidelity in models:
        batch = adapter.load_batch(cfg.case_name, fidelity, "full_traction")
        lookup = {str(sample_id): index for index, sample_id in enumerate(batch.sample_ids)}
        missing = [sample_id for sample_id in pilot_ids if sample_id not in lookup]
        if missing:
            raise MFPODError(f"{fidelity} pilot archive is missing {missing[:5]}")
        payload[fidelity] = batch.values[[lookup[sample_id] for sample_id in pilot_ids]]
    path = cfg.output_dir / "production" / "pilot_fields.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _runtime_config(
    cfg: MFPODConfig,
    settings: dict[str, Any],
    pilot_archive: Path,
    roles_path: Path | None = None,
    measured_costs: dict[str, float] | None = None,
) -> MFPODConfig:
    raw = deepcopy(cfg.raw)
    allocation = raw.setdefault("field_allocation", {})
    allocation["pilot_field_archive"] = str(pilot_archive)
    constraints = raw.setdefault("allocation_constraints", {})
    constraints["maximum_counts"] = _maximum_counts(cfg, settings)
    if measured_costs:
        raw["costs"] = {
            **{str(name).upper(): float(value) for name, value in (raw.get("costs", {}) or {}).items()},
            **{str(name).upper(): float(value) for name, value in measured_costs.items()},
        }
    production = raw.setdefault("production", {})
    if roles_path is not None:
        production["roles_manifest"] = str(roles_path)
    return replace(cfg, raw=raw)


def production_status(cfg: MFPODConfig) -> dict[str, Any]:
    settings = _production_settings(cfg)
    plan = _sample_plan(cfg, settings)
    state_path = cfg.output_dir / "production" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    return {
        "case": cfg.case_name,
        "state": state,
        "archives": {name: {"path": str(path), "samples": len(_archive_ids(path))} for name, path in cfg.archives.items()},
        "planned": {"pilot": len(plan["roles"]["pilot"]), "reference_DSMC": len(plan["roles"]["reference_DSMC"]), "maximum_counts": plan["maximum_counts"]},
    }


def run_production(
    cfg: MFPODConfig,
    *,
    resume: bool = False,
    dry_run: bool = False,
    stop_after: str = "analysis",
) -> dict[str, Any]:
    """Run or resume pilot -> allocation -> production -> full-field analysis."""

    if stop_after not in _STAGES:
        raise MFPODError(f"stop_after must be one of {_STAGES}")
    settings = _production_settings(cfg)
    plan = _sample_plan(cfg, settings)
    production_dir = cfg.output_dir / "production"
    state_path = production_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed_stages": []}
    if state_path.exists() and not resume and not dry_run:
        is_untouched_plan = not state.get("completed_stages") and state.get("status") == "planned"
        if not is_untouched_plan:
            raise MFPODError("Production state already exists; pass --resume to continue safely")
    state.update({"case": cfg.case_name, "dry_run": bool(dry_run), "plan": str(production_dir / "sample_plan.json")})
    _write_json(state_path, state)
    if dry_run or stop_after == "plan":
        state["status"] = "planned"
        _write_json(state_path, state)
        return production_status(cfg)

    _prepare_adbsat_runtime(settings)
    _validate_solver_inputs(settings)
    model_ids = _model_ids(cfg, settings)
    unmapped_models = _unmapped_adbsat_models(settings)
    registry = _registry_without_models(settings, unmapped_models)
    sample_lookup = dict(zip(plan["sample_ids"], plan["samples"]))
    pilot_ids = list(plan["roles"]["pilot"])
    if "pilot" not in state["completed_stages"]:
        pilot_runs = dict(state.get("pilot_runs", {}))
        for fidelity in model_ids:
            if model_ids[fidelity] in unmapped_models:
                continue
            if pilot_runs.get(fidelity, {}).get("status") == "complete":
                continue
            pilot_runs[fidelity] = _run_fidelity(
                cfg, settings, registry, model_ids, fidelity, pilot_ids, sample_lookup, phase="pilot"
            )
            state["pilot_runs"] = pilot_runs
            _write_json(state_path, state)
        if unmapped_models:
            _build_missing_adbsat_mappings(cfg, settings, unmapped_models)
            registry = build_adapter_registry(_adapter_config(settings))
            for fidelity in model_ids:
                if model_ids[fidelity] not in unmapped_models:
                    continue
                pilot_runs[fidelity] = _run_fidelity(
                    cfg, settings, registry, model_ids, fidelity, pilot_ids, sample_lookup, phase="pilot"
                )
                state["pilot_runs"] = pilot_runs
                _write_json(state_path, state)
        state["completed_stages"].append("pilot")
        _write_json(state_path, state)
    if stop_after == "pilot":
        return production_status(cfg)

    # A resumed run may begin after all pilot archives and the mapping exist.
    if unmapped_models:
        registry = build_adapter_registry(_adapter_config(settings))

    pilot_archive = _pilot_archive(cfg, pilot_ids)
    measured_costs = {
        name: float(run["mean_cost"])
        for name, run in (state.get("pilot_runs", {}) or {}).items()
        if run.get("mean_cost") is not None and float(run["mean_cost"]) > 0.0
    }
    state["allocation_costs"] = {
        **{str(name).upper(): float(value) for name, value in cfg.raw.get("costs", {}).items()},
        **measured_costs,
    }
    runtime_cfg = _runtime_config(cfg, settings, pilot_archive, measured_costs=measured_costs)
    if "allocation" not in state["completed_stages"]:
        field_pilot(runtime_cfg)
        allocation = optimal_allocation(runtime_cfg)
        state["allocation"] = allocation
        state["completed_stages"].append("allocation")
        _write_json(state_path, state)
    else:
        allocation = state["allocation"]
    if stop_after == "allocation":
        return production_status(cfg)

    counts = {str(name).upper(): int(value) for name, value in allocation["counts"].items() if int(value) > 0}
    production_stream = list(plan["roles"]["production_stream"])
    if "production" not in state["completed_stages"]:
        runs = []
        reference_ids = list(plan["roles"]["reference_DSMC"])
        runs.append(_run_fidelity(cfg, settings, registry, model_ids, "DSMC", reference_ids, sample_lookup, phase="reference"))
        for fidelity, count in counts.items():
            runs.append(_run_fidelity(cfg, settings, registry, model_ids, fidelity, production_stream[:count], sample_lookup, phase="production"))
        roles = {
            "pilot": pilot_ids,
            "reference_DSMC": reference_ids,
            "production": {fidelity: production_stream[:count] for fidelity, count in counts.items()},
            "nested_stream": True,
        }
        roles_path = production_dir / "roles.json"
        _write_json(roles_path, roles)
        state["production_runs"] = runs
        state["roles_manifest"] = str(roles_path)
        state["completed_stages"].append("production")
        _write_json(state_path, state)
    if stop_after == "production":
        return production_status(cfg)

    roles_path = Path(state["roles_manifest"])
    runtime_cfg = _runtime_config(
        cfg, settings, pilot_archive, roles_path, measured_costs=state.get("allocation_costs")
    )
    result = run_all(runtime_cfg)
    if "analysis" not in state["completed_stages"]:
        state["completed_stages"].append("analysis")
    state["status"] = "complete"
    state["analysis"] = result
    _write_json(state_path, state)
    return production_status(cfg)
