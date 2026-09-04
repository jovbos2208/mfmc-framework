#!/usr/bin/env python3
"""Run the offline DSMC--TPMC sensitivity protocol or emit a missing-data report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mfmc_campaign.field_mfpod.two_fidelity_sensitivity import missing_artifacts, run_two_fidelity_sensitivity


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--case-root",required=True)
    parser.add_argument("--case-name",required=True)
    parser.add_argument("--missing-only",action="store_true")
    args=parser.parse_args(); root=Path(args.case_root)
    status=missing_artifacts(root)
    if args.missing_only or status["status"]!="ready":
        target=root/"sensitivity_two_fidelity"/"goce_missing_artifacts.json"; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(status,indent=2),encoding="utf-8"); print(json.dumps({"status":status["status"],"output":str(target)},indent=2)); return 0
    print(json.dumps(run_two_fidelity_sensitivity(root,case_name=args.case_name),indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
