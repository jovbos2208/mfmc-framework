from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

from .adapters import LegacyPiclasAdapter, make_request
from .types import EvaluationRequest, EvaluationResult


class PiclasWorkflowError(ValueError):
    pass


def load_workflow_config(path: str | Path) -> Dict[str, Any]:
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - JSON needs no optional dependency
            raise PiclasWorkflowError("PyYAML is required for non-JSON workflow configs") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise PiclasWorkflowError("Workflow config must contain a mapping at its root")
    raw["_config_path"] = str(source)
    return raw


def _require_mapping(parent: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise PiclasWorkflowError(f"'{key}' must be a mapping")
    return dict(value)


def build_adapter_and_request(config: Mapping[str, Any]) -> tuple[LegacyPiclasAdapter, EvaluationRequest]:
    adapter_cfg = _require_mapping(config, "adapter")
    request_cfg = _require_mapping(config, "request")
    model_id = str(adapter_cfg.get("model_id", "PICLas_HF"))
    fidelity = str(adapter_cfg.get("fidelity", "hf"))
    qois = list(request_cfg.get("qois", adapter_cfg.get("available_qois", ["C_D"])))
    available_qois = list(adapter_cfg.get("available_qois", qois))
    unsupported = sorted(set(qois) - set(available_qois))
    if unsupported:
        raise PiclasWorkflowError(f"Requested QoIs are unavailable from the adapter: {unsupported}")

    samples = request_cfg.get("samples")
    if not isinstance(samples, list) or not samples or not all(isinstance(row, dict) for row in samples):
        raise PiclasWorkflowError("request.samples must be a non-empty list of mappings")
    sample_ids = list(request_cfg.get("sample_ids", [f"sample-{index:06d}" for index in range(len(samples))]))
    if len(sample_ids) != len(samples) or len(set(sample_ids)) != len(sample_ids):
        raise PiclasWorkflowError("sample_ids must be unique and have the same length as samples")

    geometry = _require_mapping(request_cfg, "geometry")
    regime = _require_mapping(request_cfg, "regime")
    if "altitude_km" not in regime.get("descriptors", {}):
        raise PiclasWorkflowError("request.regime.descriptors.altitude_km is required")

    kwargs = dict(adapter_cfg.get("kwargs", {}))
    adapter = LegacyPiclasAdapter(model_id, available_qois, kwargs, fidelity=fidelity)
    request = make_request(
        study_id=str(request_cfg.get("study_id", "piclas_adapter_workflow")),
        cell_id=str(request_cfg.get("cell_id", "piclas_adapter_batch")),
        model_id=model_id,
        fidelity=fidelity,
        qois=qois,
        geometry=geometry,
        regime=regime,
        active_source_blocks=list(request_cfg.get("active_source_blocks", [])),
        sample_ids=[str(value) for value in sample_ids],
        samples=[dict(row) for row in samples],
        seed=int(request_cfg.get("seed", 1)),
        metadata=dict(request_cfg.get("metadata", {})),
    )
    return adapter, request


def similarity_report(config: Mapping[str, Any]) -> Dict[str, Any]:
    similarity = dict(config.get("similarity", {}))
    linear_scale = float(similarity.get("linear_scale", 1.0))
    configured_density_scale = float(similarity.get("density_scale", 1.0))
    adapter_cfg = dict(config.get("adapter", {}))
    kwargs = dict(adapter_cfg.get("kwargs", {}))
    payload_defaults = dict(kwargs.get("payload_defaults", kwargs.get("environment_payload_defaults", {})))
    payload_density_scale = float(payload_defaults.get("density_scale", 1.0))
    density_scale = payload_density_scale
    if linear_scale <= 0.0 or density_scale <= 0.0:
        raise PiclasWorkflowError("similarity linear_scale and density_scale must be positive")
    product = linear_scale * density_scale
    return {
        "linear_scale": linear_scale,
        "density_scale": density_scale,
        "declared_density_scale": configured_density_scale,
        "payload_density_scale": payload_density_scale,
        "declared_and_payload_density_match": abs(configured_density_scale - payload_density_scale) <= 1.0e-12,
        "knudsen_ratio_scaled_to_reference": 1.0 / product,
        "knudsen_number_preserved": abs(product - 1.0) <= 1.0e-10,
        "expected_density_scale_for_knudsen_similarity": 1.0 / linear_scale,
    }


def preflight(config: Mapping[str, Any], *, require_slurm: bool = False) -> Dict[str, Any]:
    adapter_cfg = _require_mapping(config, "adapter")
    request_cfg = _require_mapping(config, "request")
    kwargs = dict(adapter_cfg.get("kwargs", {}))
    piclas_dir = Path(str(kwargs.get("piclas_dir", "piclas"))).resolve()
    update_dir = Path(str(kwargs.get("update_dir", "update_parameter_file"))).resolve()
    geometry = _require_mapping(request_cfg, "geometry")
    metadata = dict(geometry.get("metadata", {}))
    metadata.update(dict(request_cfg.get("metadata", {})))
    mesh_value = metadata.get("hf_mesh")
    mesh_path = None
    if mesh_value:
        mesh_path = Path(str(mesh_value))
        if not mesh_path.is_absolute():
            mesh_path = piclas_dir / mesh_path
        mesh_path = mesh_path.resolve()

    required_paths = {
        "piclas_dir": piclas_dir,
        "update_dir": update_dir,
        "piclas_executable": piclas_dir / "piclas",
        "piclas2vtk_executable": piclas_dir / "piclas2vtk",
        "low_fidelity_ini": piclas_dir / str(kwargs.get("ini_low", "DSMC1.ini")),
    }
    if mesh_path is not None:
        required_paths["geometry_mesh"] = mesh_path
    update_command = shlex.split(str(kwargs.get("update_script", "python update_parameter.py")))
    if update_command and not Path(update_command[-1]).is_absolute():
        required_paths["update_script"] = update_dir / update_command[-1]

    path_checks = {name: path.exists() for name, path in required_paths.items()}
    executable_checks = {
        "sbatch": shutil.which("sbatch") is not None,
        "squeue": shutil.which("squeue") is not None,
    }
    similarity = similarity_report(config)
    issues = [f"missing {name}: {required_paths[name]}" for name, exists in path_checks.items() if not exists]
    if mesh_path is None:
        issues.append("request geometry/metadata does not define hf_mesh")
    if not similarity["knudsen_number_preserved"]:
        issues.append(
            "Knudsen similarity is not preserved: linear_scale * density_scale must equal 1"
        )
    if not similarity["declared_and_payload_density_match"]:
        issues.append("similarity.density_scale does not match adapter kwargs.payload_defaults.density_scale")
    if require_slurm:
        issues.extend(f"Slurm command unavailable: {name}" for name, exists in executable_checks.items() if not exists)
    return {
        "ready": not issues,
        "require_slurm": bool(require_slurm),
        "paths": {name: str(path) for name, path in required_paths.items()},
        "path_checks": path_checks,
        "slurm_commands": executable_checks,
        "similarity": similarity,
        "issues": issues,
    }


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "_config_path"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def plan_workflow(config: Mapping[str, Any], *, state_path: str | Path) -> Dict[str, Any]:
    report = preflight(config, require_slurm=False)
    request_cfg = _require_mapping(config, "request")
    return {
        "status": "dry_run",
        "state_path": str(Path(state_path).resolve()),
        "n_samples": len(request_cfg.get("samples", [])),
        "qois": list(request_cfg.get("qois", [])),
        "preflight": report,
        "next_command": "rerun the submit action with --execute on a Slurm login node after preflight issues are resolved",
    }


def submit_workflow(config: Mapping[str, Any], *, state_path: str | Path) -> Dict[str, Any]:
    report = preflight(config, require_slurm=True)
    if not report["ready"]:
        raise PiclasWorkflowError("PICLas submit preflight failed: " + "; ".join(report["issues"]))
    adapter, request = build_adapter_and_request(config)
    batch_handle = adapter.submit(request)
    state = {
        "schema_version": 1,
        "status": "submitted",
        "config_path": config.get("_config_path"),
        "config_fingerprint": _config_fingerprint(config),
        "batch_handle": batch_handle,
        "request": {
            "model_id": request.model_id,
            "sample_ids": request.sample_ids,
            "qois": request.qois,
        },
    }
    _write_json_atomic(state_path, state)
    return state


def collect_workflow(
    config: Mapping[str, Any],
    *,
    state_path: str | Path,
    results_path: str | Path,
) -> Dict[str, Any]:
    report = preflight(config, require_slurm=True)
    if not report["ready"]:
        raise PiclasWorkflowError("PICLas collect preflight failed: " + "; ".join(report["issues"]))
    state_file = Path(state_path).resolve()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("config_fingerprint") != _config_fingerprint(config):
        raise PiclasWorkflowError("Workflow config changed after submission; refusing to collect with mismatched settings")
    if state.get("status") == "collected":
        existing = state.get("results_path")
        if existing and Path(existing).is_file():
            return json.loads(Path(existing).read_text(encoding="utf-8"))
    if state.get("status") != "submitted" or not isinstance(state.get("batch_handle"), dict):
        raise PiclasWorkflowError("State file does not contain a submitted PICLas batch handle")

    adapter, _request = build_adapter_and_request(config)
    result: EvaluationResult = adapter.collect(dict(state["batch_handle"]))
    payload = {
        "schema_version": 1,
        "status": "collected",
        "values_by_qoi": result.values_by_qoi,
        "costs_cpu_hours": result.costs,
        "sample_ids": result.sample_ids,
        "metadata": result.metadata,
    }
    target = _write_json_atomic(results_path, payload)
    state["status"] = "collected"
    state["results_path"] = str(target)
    _write_json_atomic(state_file, state)
    return payload
