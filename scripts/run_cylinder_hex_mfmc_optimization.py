#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.geometry_optimization_workflow import (
    analyze_iteration,
    finalize_workflow,
    initialize_workflow,
    load_state,
    merge_iteration,
    prepare_iteration,
    refine_iteration,
    run_iteration_jobs,
    run_iteration_sentman,
    select_iteration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Restartable cylinder-hex TPMC/Sentman optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--config", required=True)
    initialize.add_argument("--state", required=True)
    for name in ("analyze", "select", "prepare", "merge", "status", "refine", "finalize"):
        command = subparsers.add_parser(name)
        command.add_argument("--state", required=True)
    for name in ("submit", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--state", required=True)
        command.add_argument("--execute", action="store_true")
    sentman = subparsers.add_parser("sentman")
    sentman.add_argument("--state", required=True)
    sentman.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize_workflow(args.config, args.state)
    elif args.command == "analyze":
        result = analyze_iteration(args.state)
    elif args.command == "select":
        result = select_iteration(args.state)
    elif args.command == "prepare":
        result = prepare_iteration(args.state)
    elif args.command in {"submit", "collect"}:
        result = run_iteration_jobs(args.state, args.command, execute=args.execute)
    elif args.command == "sentman":
        result = run_iteration_sentman(args.state, execute=args.execute)
    elif args.command == "merge":
        result = merge_iteration(args.state)
    elif args.command == "refine":
        result = refine_iteration(args.state)
    elif args.command == "finalize":
        result = finalize_workflow(args.state)
    else:
        _path, result = load_state(args.state)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
