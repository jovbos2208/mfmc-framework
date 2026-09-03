from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from .geometry_control_nodes import generate_control_node_refinement_manifest
from .geometry_lf_workflow import run_lf_campaign
from .geometry_local_validation import generate_local_refinement_manifest, select_dsmc_finalists
from .geometry_mfmc_optimization import (
    analyze_geometry_mfmc_bundle,
    compare_geometry_estimates,
    decide_optimization_stop,
    merge_geometry_mfmc_tpmc_results,
    select_geometry_mfmc_optimization_batch,
)
from .geometry_round2 import build_round2_piclas_suite
from .parametric_geometry import CylinderHexSpec


PRANDTL_MPI_PROCESSES = 36
PRANDTL_PICLAS_MODULE = "PICLas_prandtl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_successful_subprocess_output(stdout: str, *, action: str, execute: bool) -> Dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {
            "action": action,
            "execute": execute,
            "status": "completed_without_json_summary",
            "warning": "successful subprocess returned empty stdout",
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Older simulator modules may write operational messages before the
        # final JSON document. Recover that document after a successful
        # subprocess; the jobs have already been queued at this point.
        decoder = json.JSONDecoder()
        for offset, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, end = decoder.raw_decode(text, offset)
            except json.JSONDecodeError:
                continue
            if not text[end:].strip():
                return candidate
        return {
            "action": action,
            "execute": execute,
            "status": "completed_without_json_summary",
            "stdout_tail": text[-4000:],
        }


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_state(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    target = Path(path).resolve()
    state = json.loads(target.read_text(encoding="utf-8"))
    return target, state


def initialize_workflow(config_json: str | Path, state_path: str | Path) -> Dict[str, Any]:
    target = Path(state_path).resolve()
    config_path = Path(config_json).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if float(config.get("budget_hf_equivalent", 20.0)) > 20.0:
        raise ValueError("Per-geometry budget cannot exceed 20 TPMC equivalents")
    if int(config.get("mpi_processes", PRANDTL_MPI_PROCESSES)) != PRANDTL_MPI_PROCESSES:
        raise ValueError(
            f"All optimization PICLas runs must use exactly {PRANDTL_MPI_PROCESSES} MPI processes"
        )
    parameterization = str(config.get("geometry_parameterization", "legacy_four_parameter"))
    if parameterization not in {"legacy_four_parameter", "symmetric_control_nodes"}:
        raise ValueError("geometry_parameterization must be legacy_four_parameter or symmetric_control_nodes")
    if parameterization == "symmetric_control_nodes":
        if str(config.get("budget_mode", "target_run_count")) != "target_run_count":
            raise ValueError("symmetric_control_nodes requires budget_mode=target_run_count")
        if int(config.get("tpmc_samples_per_geometry", 20)) != 20:
            raise ValueError("symmetric_control_nodes requires exactly 20 TPMC runs per geometry")
        if not config.get("initial_bundle"):
            lf_config = json.loads(Path(config["lf_config"]).resolve().read_text(encoding="utf-8"))
            sample_ids = list(map(str, lf_config["sample_ids"]))
            samples = list(lf_config["samples"])
            if len(sample_ids) != len(samples) or len(sample_ids) < 22:
                raise ValueError("The LF config must contain at least 22 aligned uncertainty samples")
            bootstrap_path = Path(config["output_root"]).resolve() / "control_node_bootstrap_bundle.json"
            bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
            bootstrap_path.write_text(json.dumps({
                "schema_version": 1,
                "study_id": str(config.get("study_id", "cylinder_hex_control_node_mfmc")),
                "selected_geometry_ids": [],
                "uncertainty_samples": dict(zip(sample_ids, samples)),
                "geometries": {},
                "evaluations": [],
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            config["initial_bundle"] = str(bootstrap_path)
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError("Existing workflow state belongs to a different configuration")
        return {**existing, "state_json": str(target), "idempotent_reuse": True}
    state = {
        "schema_version": 1,
        "study_id": str(config.get("study_id", "cylinder_hex_tpmc_sentman_mfmc")),
        "created_at": _now(),
        "updated_at": _now(),
        "config_json": str(config_path),
        "config": config,
        "current_iteration": int(config.get("initial_iteration", 0)),
        "iterations": {},
        "optimization_closed": False,
        "stop_decision": {"decision": "continue_optimization", "stop": False},
        "action_log": [],
        "control_node_trust_region": {
            "radius": float(config.get("control_node_initial_radius", 0.12)),
            "minimum_radius": float(config.get("control_node_minimum_radius", 0.015)),
            "manifests": [],
        },
    }
    _write_state(target, state)
    return {**state, "state_json": str(target)}


def _iteration(state: Dict[str, Any]) -> Dict[str, Any]:
    key = f"{int(state['current_iteration']):02d}"
    return state["iterations"].setdefault(key, {"iteration": int(state["current_iteration"]), "status": "initialized", "artifacts": {}})


def _record(state_path: Path, state: Dict[str, Any], action: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    state["updated_at"] = _now()
    state["action_log"].append({"timestamp": state["updated_at"], "action": action})
    _iteration(state)["last_action"] = action
    _write_state(state_path, state)
    return {"action": action, "state_json": str(state_path), **dict(result)}


def analyze_iteration(state_path_value: str | Path) -> Dict[str, Any]:
    state_path, state = load_state(state_path_value)
    config = state["config"]
    iteration = _iteration(state)
    bundle = iteration["artifacts"].get("merged_bundle", config["initial_bundle"])
    output_dir = Path(config["output_root"]).resolve() / f"iteration_{iteration['iteration']:02d}" / "analysis"
    result = analyze_geometry_mfmc_bundle(
        bundle,
        output_dir,
        budget_hf_equivalent=float(config.get("budget_hf_equivalent", 20.0)),
        pilot_count=int(config.get("pilot_count", 5)),
        bootstrap_repeats=int(config.get("bootstrap_repeats", 1000)),
        bootstrap_seed=int(config.get("bootstrap_seed", 20260822)),
        mean_objective_weight=float(config.get("mean_objective_weight", 0.5)),
        std_objective_weight=float(config.get("std_objective_weight", 0.5)),
        minimum_abs_control_correlation=float(config.get("minimum_abs_control_correlation", 0.5)),
        confidence_level=float(config.get("confidence_level", 0.95)),
        improvement_probability_threshold=float(config.get("improvement_probability_threshold", 0.95)),
        target_run_count=(
            int(config.get("tpmc_samples_per_geometry", 20))
            if str(config.get("budget_mode", "hf_equivalent_cost")) == "target_run_count"
            else None
        ),
        crossfit_folds=int(config.get("crossfit_folds", 5)),
        baseline_geometry_id=state.get("optimization_baseline_geometry_id"),
    )
    iteration["artifacts"].update({
        "analysis_manifest": result["manifest_json"],
        "metrics_csv": result["metrics_csv"],
        "details_json": result["details_json"],
        "comparisons_json": result["comparisons_json"],
    })
    iteration["status"] = "analyzed"
    iteration["confidence_pareto_geometry_ids"] = result["confidence_pareto_geometry_ids"]
    details_payload = json.loads(Path(result["details_json"]).read_text(encoding="utf-8"))
    detail_by_id = {str(row["geometry_id"]): row for row in details_payload["geometries"]}
    best_id = str(result["best_robust_geometry_id"])
    best = detail_by_id[best_id]
    iteration.update({
        "best_geometry_id": best_id,
        "best_mean_standard_error": float(best["mean_standard_error"]),
        "best_std_standard_error": float(best["std_standard_error"]),
        "best_improvement_probability": 0.0,
    })
    previous_analyzed = [
        state["iterations"][key] for key in sorted(state["iterations"])
        if state["iterations"][key] is not iteration and state["iterations"][key].get("best_geometry_id")
    ]
    if previous_analyzed:
        previous_best_id = str(previous_analyzed[-1]["best_geometry_id"])
        if previous_best_id in detail_by_id and previous_best_id != best_id:
            comparison = compare_geometry_estimates(
                best, detail_by_id[previous_best_id],
                confidence_level=float(config.get("confidence_level", 0.95)),
                improvement_probability_threshold=float(config.get("improvement_probability_threshold", 0.95)),
                seed=int(config.get("bootstrap_seed", 20260822)),
            )
            iteration["best_improvement_probability"] = min(
                float(objective["probability_candidate_improves"])
                for objective in comparison["objectives"].values()
            )
    control_mode = (
        str(config.get("geometry_parameterization", "legacy_four_parameter"))
        == "symmetric_control_nodes"
    )
    analysis_design = (
        iteration["artifacts"].get("design_manifest")
        if control_mode
        else config["design_manifest"]
    )
    if not analysis_design:
        raise ValueError("The analyzed iteration has no geometry design manifest")
    design = json.loads(Path(analysis_design).resolve().read_text(encoding="utf-8"))
    completed_ids = set(detail_by_id)
    eligible_ids = {
        str(row["geometry_id"]) for row in design["designs"]
        if bool(row["eligible_for_model_fitting"])
        and str(row["geometry_id"]) not in set(map(str, design["validation_geometry_ids"]))
    }
    design_space_exhausted = eligible_ids <= completed_ids
    maximum_evaluated = config.get("maximum_evaluated_geometries")
    budget_exhausted = maximum_evaluated is not None and len(completed_ids) >= int(maximum_evaluated)
    history = [state["iterations"][key] for key in sorted(state["iterations"]) if state["iterations"][key].get("status") == "analyzed"]
    state["stop_decision"] = decide_optimization_stop(
        history,
        stable_iterations_required=int(config.get("stable_iterations_required", 2)),
        minimum_improvement_probability=float(config.get("improvement_probability_threshold", 0.95)),
        objective_uncertainty_target=config.get("objective_uncertainty_target"),
        design_space_exhausted=design_space_exhausted,
        budget_exhausted=budget_exhausted,
    )
    if (
        control_mode
        and iteration.get("phase") == "control_node_refinement"
    ):
        trust = state["control_node_trust_region"]
        control_iterations = sum(
            row.get("phase") == "control_node_refinement" for row in state["iterations"].values()
        )
        maximum = int(config.get("maximum_control_node_iterations", 12))
        radius_met = float(trust["radius"]) <= float(trust["minimum_radius"])
        maximum_met = control_iterations >= maximum
        decision = (
            "control_node_trust_region_converged"
            if radius_met
            else "maximum_control_node_iterations"
            if maximum_met
            else "continue_optimization"
        )
        state["stop_decision"] = {
            "decision": decision,
            "stop": decision != "continue_optimization",
            "criteria": {
                "minimum_trust_region_radius": radius_met,
                "maximum_control_node_iterations": maximum_met,
            },
            "radius": float(trust["radius"]),
            "control_node_iterations": int(control_iterations),
        }
    history_path = Path(config["output_root"]).resolve() / "optimization_history.csv"
    history_rows = [
        {
            "iteration": row["iteration"],
            "phase": row.get("phase", "discrete"),
            "status": row.get("status", ""),
            "best_geometry_id": row.get("best_geometry_id", ""),
            "best_mean_standard_error": row.get("best_mean_standard_error", ""),
            "best_std_standard_error": row.get("best_std_standard_error", ""),
            "best_improvement_probability": row.get("best_improvement_probability", ""),
            "confidence_pareto_geometry_ids": ";".join(row.get("confidence_pareto_geometry_ids", [])),
        }
        for row in history
    ]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    state["optimization_history_csv"] = str(history_path)
    return _record(state_path, state, "analyze", result)


def select_iteration(state_path_value: str | Path) -> Dict[str, Any]:
    state_path, state = load_state(state_path_value)
    if state.get("optimization_closed"):
        raise ValueError("Optimization is closed; no further candidates may be selected")
    if state.get("stop_decision", {}).get("stop"):
        raise ValueError("Optimization has stopped; use refine or finalize instead of select")
    config = state["config"]
    iteration = _iteration(state)
    metrics = iteration["artifacts"].get("metrics_csv")
    if not metrics:
        raise ValueError("Analyze the current iteration before selecting a batch")
    output = Path(config["output_root"]).resolve() / f"iteration_{iteration['iteration'] + 1:02d}" / "selection.json"
    bundle = iteration["artifacts"].get("merged_bundle", config["initial_bundle"])
    result = select_geometry_mfmc_optimization_batch(
        config["design_manifest"], config["sentman_metrics"], bundle, metrics, output,
        count=int(config.get("batch_size", 4)), iteration=int(iteration["iteration"]) + 1,
        mean_objective_weight=float(config.get("mean_objective_weight", 0.5)),
        std_objective_weight=float(config.get("std_objective_weight", 0.5)),
    )
    state["current_iteration"] = int(iteration["iteration"]) + 1
    next_iteration = _iteration(state)
    next_iteration["artifacts"]["selection_json"] = result["output_json"]
    next_iteration["artifacts"]["selection_csv"] = result["output_csv"]
    next_iteration["status"] = "selected"
    return _record(state_path, state, "select", result)


def prepare_iteration(state_path_value: str | Path) -> Dict[str, Any]:
    state_path, state = load_state(state_path_value)
    config = state["config"]
    iteration = _iteration(state)
    selection = iteration["artifacts"].get("selection_json")
    if not selection:
        raise ValueError("Select a batch before preparing it")
    design_manifest = iteration["artifacts"].get("design_manifest") or config.get(
        "design_manifest"
    )
    if not design_manifest:
        raise ValueError("The selected iteration has no geometry design manifest")
    root = Path(config["output_root"]).resolve() / f"iteration_{iteration['iteration']:02d}"
    root.mkdir(parents=True, exist_ok=True)
    result = build_round2_piclas_suite(
        selection, design_manifest, config["lf_config"],
        output_root=config.get("geometry_output_root", "piclas/geometry/cylinder_hex_mfmc/L1"),
        config_output_dir=config.get("config_output_root", "configs/studies/cylinder_hex_mfmc"),
        suite_output_json=root / "suite.json", base_config_json=config["base_piclas_config"],
        n_dsmc=0, n_tpmc=int(config.get("tpmc_samples_per_geometry", 20)),
        round_number=int(iteration["iteration"]) + 3,
        mpi_procs=PRANDTL_MPI_PROCESSES,
        simulator_module=PRANDTL_PICLAS_MODULE,
    )
    if (
        result["mpi_procs"] != PRANDTL_MPI_PROCESSES
        or result["simulator_module"] != PRANDTL_PICLAS_MODULE
        or result["total_dsmc_runs"] != 0
    ):
        raise AssertionError("Optimization suite violated the Prandtl 36-MPI TPMC-only contract")
    iteration["artifacts"]["suite_json"] = result["suite_manifest"]
    lf_config = json.loads(Path(config["lf_config"]).resolve().read_text(encoding="utf-8"))
    lf_config["design_manifest"] = str(Path(design_manifest).resolve())
    lf_config["study_id"] = f"{state['study_id']}_iteration_{int(iteration['iteration']):03d}_sentman"
    lf_config_path = root / "sentman_config.json"
    lf_config_path.write_text(json.dumps(lf_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    iteration["artifacts"]["sentman_config"] = str(lf_config_path)
    iteration["status"] = "prepared"
    return _record(state_path, state, "prepare", result)


def run_iteration_sentman(state_path_value: str | Path, *, execute: bool) -> Dict[str, Any]:
    """Evaluate the cheap Sentman branch for the current generated batch."""
    state_path, state = load_state(state_path_value)
    iteration = _iteration(state)
    config_path = iteration["artifacts"].get("sentman_config")
    if not config_path:
        raise ValueError("Prepare the iteration before running Sentman")
    output = Path(state["config"]["output_root"]).resolve() / f"iteration_{iteration['iteration']:02d}" / "sentman"
    result = run_lf_campaign(config_path, output, execute=execute)
    if execute:
        iteration["artifacts"]["sentman_results"] = result["results_csv"]
        iteration["artifacts"]["sentman_metrics"] = result["metrics_csv"]
        iteration["status"] = "sentman_completed"
    return _record(state_path, state, "sentman", result)


def run_iteration_jobs(state_path_value: str | Path, action: str, *, execute: bool) -> Dict[str, Any]:
    if action not in {"submit", "collect"}:
        raise ValueError("action must be submit or collect")
    if action == "collect" and not execute:
        raise ValueError("collect requires execute=True")
    state_path, state = load_state(state_path_value)
    config = state["config"]
    iteration = _iteration(state)
    suite = iteration["artifacts"].get("suite_json")
    if not suite:
        raise ValueError("Prepare the iteration before running jobs")
    run_root = Path(config["output_root"]).resolve() / f"iteration_{iteration['iteration']:02d}" / "runs"
    command = [
        str(config.get("python_executable", "python3")),
        "scripts/run_cylinder_hex_round2_suite.py", action,
        "--suite", str(suite), "--run-root", str(run_root), "--fidelity", "tpmc",
    ]
    if execute:
        command.append("--execute")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        diagnostic = "\n".join(
            value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
        )
        raise RuntimeError(
            f"Cylinder-hex {action} subprocess failed with exit code "
            f"{completed.returncode}:\n{diagnostic}"
        )
    result = _decode_successful_subprocess_output(
        completed.stdout, action=action, execute=execute
    )
    iteration["artifacts"]["run_root"] = str(run_root)
    iteration["status"] = "submitted" if action == "submit" and execute else "planned" if action == "submit" else "collected"
    return _record(state_path, state, action, result)


def merge_iteration(state_path_value: str | Path) -> Dict[str, Any]:
    state_path, state = load_state(state_path_value)
    config = state["config"]
    iteration = _iteration(state)
    artifacts = iteration["artifacts"]
    previous_key = f"{int(iteration['iteration']) - 1:02d}"
    previous = state["iterations"].get(previous_key, {}).get("artifacts", {})
    bundle = previous.get("merged_bundle", config["initial_bundle"])
    output = Path(config["output_root"]).resolve() / f"iteration_{iteration['iteration']:02d}" / "paired_bundle.json"
    sentman_results = artifacts.get("sentman_results", config.get("sentman_results"))
    if not sentman_results:
        raise ValueError("Run the iteration Sentman campaign before merging")
    result = merge_geometry_mfmc_tpmc_results(
        bundle, artifacts["suite_json"], artifacts["run_root"], sentman_results, output
    )
    artifacts["merged_bundle"] = result["output_json"]
    iteration["status"] = "merged"
    return _record(state_path, state, "merge", result)


def refine_iteration(state_path_value: str | Path) -> Dict[str, Any]:
    state_path, state = load_state(state_path_value)
    iteration = _iteration(state)
    control_mode = (
        str(state["config"].get("geometry_parameterization", "legacy_four_parameter"))
        == "symmetric_control_nodes"
    )
    if not control_mode and not state["stop_decision"].get("stop"):
        raise ValueError("Local refinement begins only after the discrete stop decision")
    output = Path(state["config"]["output_root"]).resolve() / "local_refinement" / "candidates.json"
    if control_mode:
        trust = state["control_node_trust_region"]
        if iteration.get("phase") == "control_node_refinement":
            threshold = float(state["config"].get("improvement_probability_threshold", 0.95))
            if float(iteration.get("best_improvement_probability", 0.0)) >= threshold:
                trust["radius"] = min(
                    0.5,
                    float(trust["radius"]) * float(state["config"].get("trust_region_expand", 1.25)),
                )
            else:
                trust["radius"] = float(trust["radius"]) * float(
                    state["config"].get("trust_region_contract", 0.5)
                )
            trust["radius"] = max(float(trust["minimum_radius"]), float(trust["radius"]))
        center: Mapping[str, float] | None = None
        best_id = str(iteration.get("best_geometry_id", ""))
        for manifest_name in reversed(trust["manifests"]):
            manifest = json.loads(Path(manifest_name).read_text(encoding="utf-8"))
            match = next((row for row in manifest["designs"] if row["geometry_id"] == best_id), None)
            if match is not None:
                center = match["parameters"]
                break
        next_number = int(iteration["iteration"]) + 1
        output = Path(state["config"]["output_root"]).resolve() / f"control_node_iteration_{next_number:02d}" / "candidates.json"
        result = generate_control_node_refinement_manifest(
            output,
            center_parameters=center,
            baseline_spec=CylinderHexSpec(**state["config"].get("control_node_baseline_spec", {})),
            count=int(state["config"].get("local_batch_size", 6)),
            normalized_radius=float(trust["radius"]),
            iteration=next_number,
            uniform_scale_factor=float(state["config"].get("uniform_scale_factor", 0.1)),
            include_center=not bool(trust["manifests"]),
        )
        trust["manifests"].append(result["output_json"])
        if "optimization_baseline_geometry_id" not in state:
            state["optimization_baseline_geometry_id"] = str(result["candidates"][0]["geometry_id"])
        state["stop_decision"] = {"decision": "continue_optimization", "stop": False}
    else:
        result = generate_local_refinement_manifest(
            state["config"]["design_manifest"], iteration["artifacts"]["details_json"], output,
            count=int(state["config"].get("local_batch_size", 4)),
            normalized_radius=float(state["config"].get("local_normalized_radius", 0.08)),
            seed=int(state["config"].get("local_seed", 20260901)),
        )
    state["local_refinement_manifest"] = result["output_json"]
    selection_path = output.parent / "selection.json"
    selection = {
        "schema_version": 1,
        "iteration": int(iteration["iteration"]) + 1,
        "selected": [
            {
                "geometry_id": candidate["geometry_id"],
                "selection_order": index,
                "selection_basis": candidate.get(
                    "center_role", candidate.get("selection_basis", result["method"])
                ),
            }
            for index, candidate in enumerate(result["candidates"])
        ],
        "untouched_validation_geometry_ids": result.get("validation_geometry_ids_excluded", []),
        "method": result["method"],
    }
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state["current_iteration"] = int(iteration["iteration"]) + 1
    local_iteration = _iteration(state)
    local_iteration["phase"] = "control_node_refinement" if control_mode else "local_refinement"
    local_iteration["status"] = "selected"
    local_iteration["artifacts"].update({
        "selection_json": str(selection_path),
        "design_manifest": result["output_json"],
    })
    return _record(state_path, state, "refine", result)


def finalize_workflow(state_path_value: str | Path) -> Dict[str, Any]:
    state_path, state = load_state(state_path_value)
    if state.get("optimization_closed") and state.get("finalists_json"):
        return {
            "action": "finalize", "state_json": str(state_path), "idempotent_reuse": True,
            "output_json": state["finalists_json"], "suite_manifest": state.get("final_validation_suite"),
        }
    if not state["stop_decision"].get("stop"):
        raise ValueError("DSMC finalization is forbidden before an optimization stop decision")
    iteration = _iteration(state)
    output = Path(state["config"]["output_root"]).resolve() / "final_validation" / "finalists.json"
    primary_design_value = iteration["artifacts"].get(
        "design_manifest", state["config"].get("design_manifest")
    )
    if not primary_design_value:
        raise ValueError("The final optimization iteration has no geometry design manifest")
    primary_design_path = Path(primary_design_value).resolve()
    result = select_dsmc_finalists(
        primary_design_path, iteration["artifacts"]["details_json"], output,
        maximum_finalists=int(state["config"].get("maximum_dsmc_finalists", 5)),
        baseline_geometry_id=state.get("optimization_baseline_geometry_id"),
    )
    validation_root = output.parent
    manifests = [(
        primary_design_path,
        json.loads(primary_design_path.read_text(encoding="utf-8")),
    )]
    original_design_value = state["config"].get("design_manifest")
    if original_design_value:
        original_design_path = Path(original_design_value).resolve()
        if original_design_path != primary_design_path:
            manifests.append((
                original_design_path,
                json.loads(original_design_path.read_text(encoding="utf-8")),
            ))
    if state.get("local_refinement_manifest"):
        local_path = Path(state["local_refinement_manifest"]).resolve()
        manifests.append((local_path, json.loads(local_path.read_text(encoding="utf-8"))))
    for manifest_name in state.get("control_node_trust_region", {}).get("manifests", []):
        control_path = Path(manifest_name).resolve()
        if all(control_path != existing[0] for existing in manifests):
            manifests.append((control_path, json.loads(control_path.read_text(encoding="utf-8"))))
    design_rows: Dict[str, Any] = {}
    for manifest_path, manifest in manifests:
        for row in manifest["designs"]:
            copied = dict(row)
            copied["manifest_path"] = str((manifest_path.parent / row["manifest_path"]).resolve())
            design_rows[str(row["geometry_id"])] = copied
    missing = [row["geometry_id"] for row in result["finalists"] if row["geometry_id"] not in design_rows]
    if missing:
        raise ValueError(f"Finalist geometry manifests are missing: {missing}")
    combined_design = {
        "schema_version": 1,
        "designs": [design_rows[row["geometry_id"]] for row in result["finalists"]],
        "validation_geometry_ids": [],
    }
    combined_design_path = validation_root / "finalist_design_manifest.json"
    combined_design_path.write_text(json.dumps(combined_design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection = {
        "schema_version": 1,
        "selected": [
            {
                "geometry_id": row["geometry_id"],
                "selection_order": index,
                "selection_basis": row["validation_role"],
            }
            for index, row in enumerate(result["finalists"])
        ],
        "untouched_validation_geometry_ids": [],
        "final_validation_only": True,
    }
    selection_path = validation_root / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sample_count = int(state["config"].get("final_validation_samples", 5))
    suite = build_round2_piclas_suite(
        selection_path, combined_design_path, state["config"]["lf_config"],
        output_root=state["config"].get("final_geometry_output_root", "piclas/geometry/cylinder_hex_final/L1"),
        config_output_dir=state["config"].get("final_config_output_root", "configs/studies/cylinder_hex_final_validation"),
        suite_output_json=validation_root / "suite.json",
        base_config_json=state["config"]["base_piclas_config"],
        n_dsmc=sample_count,
        n_tpmc=sample_count,
        round_number=999,
        mpi_procs=PRANDTL_MPI_PROCESSES,
        simulator_module=PRANDTL_PICLAS_MODULE,
    )
    if (
        suite["mpi_procs"] != PRANDTL_MPI_PROCESSES
        or suite["simulator_module"] != PRANDTL_PICLAS_MODULE
        or suite["total_dsmc_runs"] == 0
    ):
        raise AssertionError("Final validation suite must contain Prandtl 36-MPI DSMC workflows")
    state["optimization_closed"] = True
    state["finalists_json"] = result["output_json"]
    state["final_validation_suite"] = suite["suite_manifest"]
    return _record(state_path, state, "finalize", {**result, "suite_manifest": suite["suite_manifest"]})
