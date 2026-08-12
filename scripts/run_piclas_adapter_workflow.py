#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.piclas_adapter_workflow import (
    PiclasWorkflowError,
    collect_workflow,
    load_workflow_config,
    plan_workflow,
    submit_workflow,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit and collect restartable Slurm batches through LegacyPiclasAdapter."
    )
    parser.add_argument("action", choices=("submit", "collect"))
    parser.add_argument("--config", required=True, help="PICLas adapter workflow JSON/YAML")
    parser.add_argument("--state", required=True, help="Persistent JSON batch-handle state")
    parser.add_argument("--results", default="piclas_results.json", help="Collected result JSON")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform Slurm actions; without this flag submit is a non-mutating dry-run",
    )
    args = parser.parse_args()
    config = load_workflow_config(args.config)
    try:
        if args.action == "submit":
            output = submit_workflow(config, state_path=args.state) if args.execute else plan_workflow(
                config, state_path=args.state
            )
        else:
            if not args.execute:
                parser.error("collect requires --execute because it waits for jobs and submits postprocessing")
            output = collect_workflow(config, state_path=args.state, results_path=args.results)
    except PiclasWorkflowError as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
