#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.selected_hf_geometry import build_sequential_hf_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the first sequential WP5 DSMC acquisition suite")
    parser.add_argument("--initial-suite", required=True)
    parser.add_argument("--acquisition", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--config-output-dir", default="configs/studies/cylinder_hex_wp5_sequential_hf_round1")
    parser.add_argument("--suite-output", default="outputs/cylinder_hex/wp5_sequential_hf_round1/suite.json")
    args = parser.parse_args()
    result = build_sequential_hf_suite(
        args.initial_suite,
        args.acquisition,
        args.bundle,
        config_output_dir=args.config_output_dir,
        suite_output_json=args.suite_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
