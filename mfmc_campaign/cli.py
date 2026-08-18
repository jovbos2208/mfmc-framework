from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Optional

from .campaign import postprocess_outputs, run_campaign_from_path
from .adbsat_surface_mapping import build_and_write_adbsat_surface
from .config import load_and_validate
from .field_pod_mfmc import run_data_check, run_field_workflow, load_field_config
from .legacy_inventory import inventory_legacy_surrogate_data
from .legacy_backfill import backfill_legacy_surrogate_data
from .output import export_predictive_dataset
from .pilot_correlation import run_pilot_correlation_from_path
from .plotting import generate_plots
from .surrogate_dataset import export_surrogate_dataset
from .surrogate_pce_analysis import fit_multifidelity_pce_analysis
from .surrogate_gsa import run_surrogate_gsa
from .gsa_aggregate import aggregate_surrogate_gsa
from .parametric_geometry import (
    CylinderHexSpec,
    build_geometry_assets,
    build_gmsh_exterior_mesh,
    build_piclas_hdf5_mesh,
)
from .geometry_design import build_cylinder_hex_design, select_initial_hf_designs


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


def _cmd_export_surrogate(args: argparse.Namespace) -> int:
    sample_inputs_csv = args.sample_inputs_csv or os.path.join(args.output_dir, "sample_inputs.csv")
    model_evaluations_csv = args.model_evaluations_csv or os.path.join(args.output_dir, "model_evaluations.csv")
    target_csv = args.target_csv or os.path.join(args.output_dir, "surrogate_dataset.csv")
    summary = export_surrogate_dataset(
        sample_inputs_csv,
        model_evaluations_csv,
        target_csv,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_inventory_legacy(args: argparse.Namespace) -> int:
    summary = inventory_legacy_surrogate_data(
        args.campaign_root,
        args.output_dir,
        reconstruction_audit_csv=args.reconstruction_audit_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_backfill_legacy(args: argparse.Namespace) -> int:
    summary = backfill_legacy_surrogate_data(
        args.campaign_root,
        args.output_root,
        reconstruction_audit_csv=args.reconstruction_audit_csv,
        cases=args.cases,
        cap_per_request=args.cap_per_request,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_fit_surrogate_pce(args: argparse.Namespace) -> int:
    dataset = args.surrogate_dataset_csv or os.path.join(args.case_dir, "surrogate_dataset.csv")
    output_dir = args.output_dir or os.path.join(args.case_dir, "surrogate_pce")
    summary = fit_multifidelity_pce_analysis(
        dataset,
        output_dir,
        qoi=args.qoi,
        degree=args.degree,
        q_norm=args.q_norm,
        max_interaction=args.max_interaction,
        cv_folds=args.cv_folds,
        max_rows_per_model=args.max_rows_per_model,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_run_surrogate_gsa(args: argparse.Namespace) -> int:
    summary = run_surrogate_gsa(
        args.case_dir,
        pce_dir=args.pce_dir,
        output_dir=args.output_dir,
        qoi=args.qoi,
        mc_samples=args.mc_samples,
        bootstrap=args.bootstrap,
        refit_bootstrap=args.refit_bootstrap,
        refit_mc_samples=args.refit_mc_samples,
        refit_max_rows_per_model=args.refit_max_rows_per_model,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_aggregate_surrogate_gsa(args: argparse.Namespace) -> int:
    summary = aggregate_surrogate_gsa(args.campaign_root, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_generate_cylinder_hex(args: argparse.Namespace) -> int:
    reference_spec = CylinderHexSpec(
        nose_length_fraction=args.nose_fraction,
        tail_length_fraction=args.tail_fraction,
        width_height_ratio=args.width_height_ratio,
        chamfer_fraction=args.chamfer_fraction,
        total_length_m=args.length_m,
        target_volume_m3=args.volume_m3,
        max_width_m=args.max_width_m,
        max_height_m=args.max_height_m,
        nose_end_scale=args.nose_end_scale,
        tail_end_scale=args.tail_end_scale,
        axial_segments_per_taper=args.axial_segments,
    )
    spec = reference_spec.scaled(args.uniform_scale)
    summary = build_geometry_assets(
        args.output_dir,
        spec,
        design_id=args.design_id,
        uniform_scale_factor=args.uniform_scale,
        body_mesh_size_m=args.body_mesh_size_m,
        farfield_mesh_size_m=args.farfield_mesh_size_m,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_generate_cylinder_hex_design(args: argparse.Namespace) -> int:
    summary = build_cylinder_hex_design(
        args.output_dir,
        n_designs=args.n_designs,
        n_validation=args.n_validation,
        seed=args.seed,
        maximin_trials=args.maximin_trials,
        uniform_scale_factor=args.uniform_scale,
        adbsat_runtime_dir=args.adbsat_runtime_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_select_cylinder_hex_initial_hf(args: argparse.Namespace) -> int:
    summary = select_initial_hf_designs(
        args.design_csv,
        args.lf_metrics_csv,
        args.output,
        count=args.count,
        objective_columns=args.objectives,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_mesh_cylinder_hex_exterior(args: argparse.Namespace) -> int:
    summary = build_gmsh_exterior_mesh(args.manifest, gmsh_executable=args.gmsh)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_convert_cylinder_hex_pyhope(args: argparse.Namespace) -> int:
    summary = build_piclas_hdf5_mesh(args.manifest, pyhope_executable=args.pyhope)
    print(json.dumps(summary, indent=2))
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

    p_surrogate = sub.add_parser(
        "export-surrogate-dataset",
        help="Join sample inputs and model evaluations using strict provenance fingerprints",
    )
    p_surrogate.add_argument("--output-dir", required=True, help="Campaign output directory")
    p_surrogate.add_argument("--sample-inputs-csv", default=None, help="Optional sample inputs CSV")
    p_surrogate.add_argument("--model-evaluations-csv", default=None, help="Optional model evaluations CSV")
    p_surrogate.add_argument("--target-csv", default=None, help="Optional target CSV path")
    p_surrogate.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip unmatched evaluations and permit unused input rows",
    )
    p_surrogate.set_defaults(func=_cmd_export_surrogate)

    p_inventory = sub.add_parser(
        "inventory-legacy-surrogate-data",
        help="Inventory legacy campaign outputs and existing reconstruction audits",
    )
    p_inventory.add_argument("--campaign-root", required=True, help="Legacy campaign root")
    p_inventory.add_argument("--output-dir", required=True, help="Directory for inventory reports")
    p_inventory.add_argument(
        "--reconstruction-audit-csv",
        default=None,
        help="Optional reconstruction audit; defaults to the campaign sensitivity-analysis audit",
    )
    p_inventory.set_defaults(func=_cmd_inventory_legacy)

    p_backfill = sub.add_parser(
        "backfill-legacy-surrogate-data",
        help="Reconstruct hash-verified sample inputs from legacy production campaigns",
    )
    p_backfill.add_argument("--campaign-root", required=True, help="Read-only legacy campaign root")
    p_backfill.add_argument("--output-root", required=True, help="Separate root for reconstructed outputs")
    p_backfill.add_argument("--reconstruction-audit-csv", default=None, help="Optional existing audit CSV")
    p_backfill.add_argument("--cases", nargs="+", default=None, help="Optional case-name subset")
    p_backfill.add_argument(
        "--cap-per-request",
        type=int,
        default=2500,
        help="Deterministic evenly spaced sample cap per audited request; <=0 retains all",
    )
    p_backfill.add_argument("--overwrite", action="store_true", help="Replace generated files in existing case targets")
    p_backfill.set_defaults(func=_cmd_backfill_legacy)

    p_pce = sub.add_parser(
        "fit-surrogate-pce",
        help="Fit and repetition-wise validate HF, LF and MF-residual sparse PCE models",
    )
    p_pce.add_argument("--case-dir", required=True, help="Backfilled case directory")
    p_pce.add_argument("--surrogate-dataset-csv", default=None, help="Optional strict surrogate dataset")
    p_pce.add_argument("--output-dir", default=None, help="Optional PCE analysis output directory")
    p_pce.add_argument("--qoi", default="C_D")
    p_pce.add_argument("--degree", type=int, default=3)
    p_pce.add_argument("--q-norm", type=float, default=0.75)
    p_pce.add_argument("--max-interaction", type=int, default=2)
    p_pce.add_argument("--cv-folds", type=int, default=5)
    p_pce.add_argument("--max-rows-per-model", type=int, default=0)
    p_pce.set_defaults(func=_cmd_fit_surrogate_pce)

    p_gsa = sub.add_parser(
        "run-surrogate-gsa",
        help="Estimate variable- and source-level Sobol indices from a fitted MF-PCE",
    )
    p_gsa.add_argument("--case-dir", required=True, help="Backfilled case directory")
    p_gsa.add_argument("--pce-dir", default=None, help="Optional fitted PCE directory")
    p_gsa.add_argument("--output-dir", default=None, help="Optional GSA output directory")
    p_gsa.add_argument("--qoi", default="C_D")
    p_gsa.add_argument("--mc-samples", type=int, default=20_000)
    p_gsa.add_argument("--bootstrap", type=int, default=200)
    p_gsa.add_argument(
        "--refit-bootstrap",
        type=int,
        default=100,
        help="Repetition-block bootstrap fits for surrogate-training uncertainty; 0 disables",
    )
    p_gsa.add_argument("--refit-mc-samples", type=int, default=5_000)
    p_gsa.add_argument(
        "--refit-max-rows-per-model",
        type=int,
        default=0,
        help="Refit data cap; <=0 reuses the fitted PCE manifest cap",
    )
    p_gsa.add_argument("--seed", type=int, default=20260317)
    p_gsa.set_defaults(func=_cmd_run_surrogate_gsa)

    p_gsa_aggregate = sub.add_parser(
        "aggregate-surrogate-gsa",
        help="Build cross-case GSA tables and separate primary from exploratory cases",
    )
    p_gsa_aggregate.add_argument("--campaign-root", required=True)
    p_gsa_aggregate.add_argument("--output-dir", default=None)
    p_gsa_aggregate.set_defaults(func=_cmd_aggregate_surrogate_gsa)

    p_geometry = sub.add_parser(
        "generate-cylinder-hex",
        help="Generate a new fixed-volume parametric cylinder-hex surface and manifest",
    )
    p_geometry.add_argument("--output-dir", required=True)
    p_geometry.add_argument("--design-id", default=None)
    p_geometry.add_argument("--nose-fraction", type=float, default=0.20)
    p_geometry.add_argument("--tail-fraction", type=float, default=0.20)
    p_geometry.add_argument("--width-height-ratio", type=float, default=1.0)
    p_geometry.add_argument("--chamfer-fraction", type=float, default=0.15)
    p_geometry.add_argument("--length-m", type=float, default=1.0)
    p_geometry.add_argument("--volume-m3", type=float, default=0.12)
    p_geometry.add_argument("--max-width-m", type=float, default=0.60)
    p_geometry.add_argument("--max-height-m", type=float, default=0.60)
    p_geometry.add_argument("--nose-end-scale", type=float, default=0.20)
    p_geometry.add_argument("--tail-end-scale", type=float, default=0.45)
    p_geometry.add_argument("--axial-segments", type=int, default=1)
    p_geometry.add_argument(
        "--uniform-scale",
        type=float,
        default=1.0,
        help="Uniform linear scale; area follows factor^2 and volume factor^3",
    )
    p_geometry.add_argument(
        "--body-mesh-size-m",
        type=float,
        default=0.06,
        help="Gmsh characteristic length on spacecraft points",
    )
    p_geometry.add_argument(
        "--farfield-mesh-size-m",
        type=float,
        default=0.30,
        help="Gmsh characteristic length on outer-domain points",
    )
    p_geometry.set_defaults(func=_cmd_generate_cylinder_hex)

    p_geometry_design = sub.add_parser(
        "generate-cylinder-hex-design",
        help="Generate the reproducible WP5 LF geometry design and untouched validation split",
    )
    p_geometry_design.add_argument("--output-dir", required=True)
    p_geometry_design.add_argument("--n-designs", type=int, default=32)
    p_geometry_design.add_argument("--n-validation", type=int, default=6)
    p_geometry_design.add_argument("--seed", type=int, default=20260818)
    p_geometry_design.add_argument("--maximin-trials", type=int, default=256)
    p_geometry_design.add_argument("--uniform-scale", type=float, default=0.1)
    p_geometry_design.add_argument(
        "--adbsat-runtime-dir",
        default=None,
        help="Optional ADBSat-PyVersion directory; otherwise write a portable asset bundle",
    )
    p_geometry_design.set_defaults(func=_cmd_generate_cylinder_hex_design)

    p_hf_select = sub.add_parser(
        "select-cylinder-hex-initial-hf",
        help="Select initial HF geometries from LF Pareto coverage and geometry-space filling",
    )
    p_hf_select.add_argument("--design-csv", required=True)
    p_hf_select.add_argument("--lf-metrics-csv", required=True)
    p_hf_select.add_argument("--output", required=True)
    p_hf_select.add_argument("--count", type=int, default=6)
    p_hf_select.add_argument(
        "--objectives",
        nargs="+",
        default=["mean_drag", "std_drag", "q95_drag"],
        help="LF metric columns minimized for Pareto coverage",
    )
    p_hf_select.set_defaults(func=_cmd_select_cylinder_hex_initial_hf)

    p_geometry_mesh = sub.add_parser(
        "mesh-cylinder-hex-exterior",
        help="Generate and validate a new Gmsh tetrahedral exterior-flow mesh",
    )
    p_geometry_mesh.add_argument("--manifest", required=True)
    p_geometry_mesh.add_argument("--gmsh", default="gmsh")
    p_geometry_mesh.set_defaults(func=_cmd_mesh_cylinder_hex_exterior)

    p_geometry_pyhope = sub.add_parser(
        "convert-cylinder-hex-pyhope",
        help="Convert the Gmsh exterior mesh to validated PICLas HDF5 using PyHOPE",
    )
    p_geometry_pyhope.add_argument("--manifest", required=True)
    p_geometry_pyhope.add_argument("--pyhope", default="pyhope")
    p_geometry_pyhope.set_defaults(func=_cmd_convert_cylinder_hex_pyhope)

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
