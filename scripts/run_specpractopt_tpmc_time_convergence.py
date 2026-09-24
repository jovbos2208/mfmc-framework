#!/usr/bin/env python3
"""Plan, submit, or collect the SpecPractOpt TPMC end-time convergence suite."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.piclas_adapter_workflow import (  # noqa: E402
    collect_workflow,
    load_workflow_config,
    plan_workflow,
    submit_workflow,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "collect"))
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--run-root",
        default="outputs/specpractopt_tpmc_time_convergence/runs",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "collect" and not args.execute:
        parser.error("collect requires --execute")

    suite = json.loads(Path(args.suite).resolve().read_text(encoding="utf-8"))
    run_root = Path(args.run_root).resolve()
    summaries = []
    for level in suite["levels"]:
        config = load_workflow_config(level["workflow_config"])
        work = run_root / level["id"]
        state_path = work / "piclas_batch_state.json"
        results_path = work / "piclas_results.json"
        if args.action == "submit":
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("status") in {"submitted", "collected"}:
                    summaries.append({"level": level["id"], "status": state["status"], "skipped": True})
                    continue
            with redirect_stdout(sys.stderr):
                output = (
                    submit_workflow(config, state_path=state_path)
                    if args.execute
                    else plan_workflow(config, state_path=state_path)
                )
        else:
            with redirect_stdout(sys.stderr):
                output = collect_workflow(config, state_path=state_path, results_path=results_path)
        summaries.append({"level": level["id"], "t_end_s": level["t_end_s"], **output})

    print(json.dumps({"action": args.action, "execute": args.execute, "workflows": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
