from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Optional

from .campaign import postprocess_outputs, run_campaign_from_path
from .adbsat_surface_mapping import build_and_write_adbsat_surface
from .config import load_and_validate
from .field_pod_mfmc import run_data_check, run_field_workflow, load_field_config
from .output import export_predictive_dataset
from .pilot_correlation import run_pilot_correlation_from_path
from .plotting import generate_plots


def _cmd_run(args: argparse.Namespace) -> int:
    summary = run_campaign_from_path(args.config, resume=args.resume, pilots_only=False)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_run_sweep(args: argparse.Namespace) -> int:
    summaries = []
    for config_path in args.configs:
        summaries.append(run_campaign_from_path(config_path, resume=args.resume, pilots_only=False))
    print(json.dumps({"runs": summaries}, indent=2))
    return 0


def _cmd_run_pilots(args: argparse.Namespace) -> int:
    summary = run_campaign_from_path(args.config, resume=args.resume, pilots_only=True)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_run_pilots_sweep(args: argparse.Namespace) -> int:
    summaries = []
    for config_path in args.configs:
        summaries.append(run_campaign_from_path(config_path, resume=args.resume, pilots_only=True))
    print(json.dumps({"runs": summaries}, indent=2))
    return 0


def _cmd_run_pilot_correlation(args: argparse.Namespace) -> int:
    summary = run_pilot_correlation_from_path(args.config, resume=args.resume)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_run_pilot_correlation_sweep(args: argparse.Namespace) -> int:
    summaries = []
    for config_path in args.configs:
        summaries.append(run_pilot_correlation_from_path(config_path, resume=args.resume))
    print(json.dumps({"runs": summaries}, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_and_validate(args.config)
    print(f"Config valid: {args.config}")
    print(f"Study id: {cfg.get('study', {}).get('id')}")
    return 0


def _cmd_check_field_data(args: argparse.Namespace) -> int:
    cfg = load_field_config(args.config)
    report = run_data_check(cfg)
    print(json.dumps(report, indent=2, default=str))
    return 1 if report.get("missing_required") else 0


def _cmd_run_field_pod_mfmc(args: argparse.Namespace) -> int:
    cfg = load_field_config(args.config)
    summary = run_field_workflow(cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cmd_build_adbsat_surface(args: argparse.Namespace) -> int:
    summary = build_and_write_adbsat_surface(
        args.vtu,
        args.obj,
        args.mat,
        args.mapping,
        length_scale_to_m=args.length_scale_to_m,
        material_id=args.material_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    summary = run_campaign_from_path(args.config, resume=True, pilots_only=False)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    source_csv = args.results_csv
    if source_csv is None:
        source_csv = os.path.join(args.output_dir, "results_long.csv")
    robustness_csv = args.robustness_csv
    if robustness_csv is None:
        robustness_csv = os.path.join(args.output_dir, "pilot_robustness.csv")
    target_csv = args.target_csv
    if target_csv is None:
        target_csv = os.path.join(args.output_dir, "predictive_dataset.csv")

    export_predictive_dataset(source_csv, target_csv, robustness_csv)
    print(target_csv)
    return 0


def _cmd_postprocess(args: argparse.Namespace) -> int:
    summary = postprocess_outputs(args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    files = generate_plots(args.output_dir)
    print(json.dumps({"plots": files}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MFMC campaign CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute campaign")
    p_run.add_argument("config", help="Path to YAML/JSON config")
    p_run.add_argument("--resume", action="store_true", help="Skip completed cells")
    p_run.set_defaults(func=_cmd_run)

    p_run_sweep = sub.add_parser("run-sweep", help="Execute multiple campaign configs")
    p_run_sweep.add_argument("configs", nargs="+", help="Paths to YAML/JSON configs")
    p_run_sweep.add_argument("--resume", action="store_true", help="Resume each run")
    p_run_sweep.set_defaults(func=_cmd_run_sweep)

    p_run_pilots = sub.add_parser("run-pilots", help="Execute pilot-only campaign diagnostics")
    p_run_pilots.add_argument("config", help="Path to YAML/JSON config")
    p_run_pilots.add_argument("--resume", action="store_true", help="Resume pilot run")
    p_run_pilots.set_defaults(func=_cmd_run_pilots)

    p_run_pilots_sweep = sub.add_parser("run-pilots-sweep", help="Execute multiple pilot-only campaign diagnostics")
    p_run_pilots_sweep.add_argument("configs", nargs="+", help="Paths to YAML/JSON configs")
    p_run_pilots_sweep.add_argument("--resume", action="store_true", help="Resume each pilot run")
    p_run_pilots_sweep.set_defaults(func=_cmd_run_pilots_sweep)

    p_run_pilot_corr = sub.add_parser(
        "run-pilot-correlation",
        help="Execute shared-sample HF/LF pilot correlations without production MFMC runs",
    )
    p_run_pilot_corr.add_argument("config", help="Path to YAML/JSON config")
    p_run_pilot_corr.add_argument("--resume", action="store_true", help="Resume pilot-correlation run")
    p_run_pilot_corr.set_defaults(func=_cmd_run_pilot_correlation)

    p_run_pilot_corr_sweep = sub.add_parser(
        "run-pilot-correlation-sweep",
        help="Execute multiple shared-sample pilot-correlation configs",
    )
    p_run_pilot_corr_sweep.add_argument("configs", nargs="+", help="Paths to YAML/JSON configs")
    p_run_pilot_corr_sweep.add_argument("--resume", action="store_true", help="Resume each pilot-correlation run")
    p_run_pilot_corr_sweep.set_defaults(func=_cmd_run_pilot_correlation_sweep)

    p_val = sub.add_parser("validate-config", help="Validate campaign config")
    p_val.add_argument("config", help="Path to YAML/JSON config")
    p_val.set_defaults(func=_cmd_validate)

    p_check_field = sub.add_parser(
        "check-field-data",
        help="Validate one case for PICLAS field-level POD/MFMC inputs",
    )
    p_check_field.add_argument("config", help="Path to field POD/MFMC YAML config")
    p_check_field.set_defaults(func=_cmd_check_field_data)

    p_run_field = sub.add_parser(
        "run-field-pod-mfmc",
        help="Run DSMC-target field POD/MFMC with TPMC and optional Sentman controls",
    )
    p_run_field.add_argument("config", help="Path to field POD/MFMC YAML config")
    p_run_field.set_defaults(func=_cmd_run_field_pod_mfmc)

    p_adbsat_surface = sub.add_parser(
        "build-adbsat-surface",
        help="Build a canonical ADBSat OBJ, MAT and PICLAS-face mapping from one VTU",
    )
    p_adbsat_surface.add_argument("--vtu", required=True, help="Canonical PICLAS surface VTU")
    p_adbsat_surface.add_argument("--obj", required=True, help="Output ADBSat OBJ path")
    p_adbsat_surface.add_argument("--mat", required=True, help="Output ADBSat MAT path")
    p_adbsat_surface.add_argument("--mapping", required=True, help="Output triangle-to-reference-cell NPZ")
    p_adbsat_surface.add_argument("--length-scale-to-m", type=float, default=1.0)
    p_adbsat_surface.add_argument("--material-id", type=int, default=1)
    p_adbsat_surface.set_defaults(func=_cmd_build_adbsat_surface)

    p_res = sub.add_parser("resume", help="Resume campaign")
    p_res.add_argument("config", help="Path to YAML/JSON config")
    p_res.set_defaults(func=_cmd_resume)

    p_exp = sub.add_parser("export-predictive-dataset", help="Export predictive dataset CSV")
    p_exp.add_argument("--output-dir", required=True, help="Campaign output directory")
    p_exp.add_argument("--results-csv", default=None, help="Optional source results CSV")
    p_exp.add_argument("--robustness-csv", default=None, help="Optional pilot robustness CSV")
    p_exp.add_argument("--target-csv", default=None, help="Optional target CSV path")
    p_exp.set_defaults(func=_cmd_export)

    p_post = sub.add_parser("postprocess", help="Generate summary tables and plots from saved outputs")
    p_post.add_argument("--output-dir", required=True, help="Campaign output directory")
    p_post.set_defaults(func=_cmd_postprocess)

    p_plot = sub.add_parser("plot", help="Generate plots from structured outputs")
    p_plot.add_argument("--output-dir", required=True, help="Campaign output directory")
    p_plot.set_defaults(func=_cmd_plot)

    return parser


def run_cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


def main() -> int:
    return run_cli(None)


if __name__ == "__main__":
    raise SystemExit(main())
