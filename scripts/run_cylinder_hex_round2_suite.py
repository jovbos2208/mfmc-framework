#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.piclas_adapter_workflow import collect_workflow, load_workflow_config, plan_workflow, submit_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit or collect WP5 round-2 DSMC/TPMC workflows")
    parser.add_argument("action", choices=("submit", "collect"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--run-root", default="outputs/cylinder_hex/wp5_round2/runs")
    parser.add_argument("--fidelity", choices=("all", "dsmc", "tpmc"), default="all")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "collect" and not args.execute:
        parser.error("collect requires --execute")
    repository_root = Path.cwd().resolve()
    suite = json.loads(Path(args.suite).resolve().read_text(encoding="utf-8"))
    run_root = Path(args.run_root).resolve()
    summaries = []
    for geometry in suite["geometries"]:
        geometry_id = geometry["geometry_id"]
        available = tuple(
            fidelity for fidelity in ("dsmc", "tpmc")
            if f"{fidelity}_workflow_config" in geometry
        )
        fidelities = available if args.fidelity == "all" else (args.fidelity,)
        for fidelity in fidelities:
            if fidelity not in available:
                raise ValueError(f"Suite has no {fidelity} workflow for {geometry_id}")
            model_id = "PICLas_DSMC" if fidelity == "dsmc" else "PICLas_TPMC"
            config = load_workflow_config(repository_root / geometry[f"{fidelity}_workflow_config"])
            work = run_root / geometry_id / model_id
            state_path = work / "piclas_batch_state.json"
            results_path = work / "piclas_results.json"
            if args.action == "submit":
                if state_path.is_file():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    if state.get("status") in {"submitted", "collected"}:
                        summaries.append({"geometry_id": geometry_id, "model_id": model_id, "status": state["status"], "skipped": True})
                        continue
                with redirect_stdout(sys.stderr):
                    output = (
                        submit_workflow(config, state_path=state_path)
                        if args.execute
                        else plan_workflow(config, state_path=state_path)
                    )
            else:
                with redirect_stdout(sys.stderr):
                    output = collect_workflow(
                        config, state_path=state_path, results_path=results_path
                    )
            summaries.append({"geometry_id": geometry_id, "model_id": model_id, **output})
    print(json.dumps({"action": args.action, "execute": args.execute, "workflows": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
