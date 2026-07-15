from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from .campaign import run_campaign


def _base_sources() -> List[Dict[str, Any]]:
    return [
        {"name": "environment.density"},
        {"name": "environment.composition"},
        {"name": "environment.temperature"},
        {"name": "attitude.aos"},
        {"name": "attitude.aoa"},
        {"name": "gsi.model"},
        {"name": "numerical.database_index"},
        {"name": "operations.seed"},
    ]


def _regime(altitude: int, label: str) -> Dict[str, Any]:
    return {
        "id": label,
        "label": label,
        "descriptors": {
            "altitude_km": altitude,
            "characteristic_length": 0.1,
            "knudsen_number": 1.0,
            "speed_ratio": 8.0,
            "freestream_temperature": 900.0,
            "composition_descriptor": "msis_sampled",
            "solar_activity_state": "unknown",
            "geomagnetic_activity_state": "unknown",
            "wind_state": "unknown",
            "geometry_class": "cube",
            "surface_state": "default",
        },
    }


def _geometry(name: str = "Cube") -> Dict[str, Any]:
    return {
        "id": name,
        "name": name,
        "characteristic_length": 0.1,
        "geometry_class": "cube",
        "tags": ["legacy"],
    }


def _default_variables(db_index_max: int) -> List[Dict[str, Any]]:
    return [
        {
            "name": "database_index",
            "source_block": "numerical.database_index",
            "distribution": {"kind": "int_uniform", "params": {"low": 0, "high": max(0, db_index_max - 1)}},
            "baseline": 0,
            "bounds": [0, max(0, db_index_max - 1)],
        },
        {
            "name": "aos_deg",
            "source_block": "attitude.aos",
            "distribution": {"kind": "fixed", "params": {"value": 0}},
            "baseline": 0,
            "bounds": [-180, 180],
        },
        {
            "name": "aoa_deg",
            "source_block": "attitude.aoa",
            "distribution": {"kind": "fixed", "params": {"value": 0}},
            "baseline": 0,
            "bounds": [-180, 180],
        },
    ]


def build_legacy_mfmc_config(
    altitude: int,
    aos: int,
    db_index_max: int,
    budget: float,
    repetitions: int,
    candidate_methods: List[str],
    backend: str,
    output_dir: str,
) -> Dict[str, Any]:
    lf_models = []
    for method in candidate_methods:
        lf_models.append(
            {
                "id": method,
                "kind": "legacy_adbsat" if backend == "legacy_slurm" else "mock",
                "method": method,
                "kwargs": {
                    "simulation_script": f"python {os.path.abspath(os.path.join('ADBSat-PyVersion', 'simulate.py'))}",
                    "base_dir": "ADBSat-PyVersion",
                },
            }
        )

    return {
        "study": {
            "id": f"legacy_mfmc_{altitude}km",
            "mode": "mixed_uncertainty",
            "active_source_blocks": ["numerical.database_index"],
        },
        "geometries": [_geometry("Cube")],
        "regimes": [_regime(altitude, f"legacy_{altitude}km")],
        "sources": {"blocks": _base_sources()},
        "variables": _default_variables(db_index_max),
        "sampling": {"method": "independent", "sample_count": 64, "max_production_samples": 5000},
        "models": {
            "hf": {
                "id": "PICLas_HF",
                "kind": "legacy_piclas" if backend == "legacy_slurm" else "mock",
                "kwargs": {},
            },
            "lf": lf_models,
            "available_qois": {
                "PICLas_HF": ["C_D", "C_D2"],
                **{m: ["C_D", "C_D2"] for m in candidate_methods},
            },
        },
        "qois": {"direct": ["C_D", "C_D2"], "derived": [{"name": "Var_C_D", "expression": "E[C_D2]-E[C_D]^2"}]},
        "pilot": {"size": 50, "sizes": [10, 20, 50, 100], "robustness_repetitions": 20},
        "budget": {"total": budget, "hf_fraction": 0.25},
        "repetitions": repetitions,
        "seeds": {"global": 123},
        "outputs": {"dir": output_dir, "write_parquet": False, "write_config_snapshot": True, "plots": True},
        "execution": {"backend": backend, "resume": False, "aos_deg": aos},
    }


def write_legacy_mfmc_csv(summary: Dict[str, Any], altitude: int, budget: float) -> str:
    src = summary["results_csv"]
    dst = f"mfmc_results_{altitude}km_{int(budget)}.csv"

    rows: List[Dict[str, Any]] = []
    with open(src, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("qoi") != "C_D":
                continue
            if str(row.get("mode")) != "mixed_uncertainty":
                continue
            rows.append(row)

    by_rep: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_rep.setdefault(str(row.get("repetition")), []).append(row)

    with open(dst, "w", encoding="utf-8", newline="") as f:
        fields = [
            "repeat",
            "s_hat",
            "estimated_variance",
            "std_error",
            "total_cost",
            "m_list",
            "method_list",
            "alpha",
            "y0",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rep, grp in sorted(by_rep.items(), key=lambda kv: int(kv[0])):
            mfmc_vals = [float(g.get("mfmc_estimate", "nan")) for g in grp]
            hf_vals = [float(g.get("hf_only_estimate", "nan")) for g in grp]
            hf_vars = [float(g.get("hf_variance", "nan")) for g in grp]
            methods = sorted({str(g.get("lf_model_id")) for g in grp})
            total_cost = sum(float(g.get("cost_ratio", 0.0)) for g in grp)

            s_hat = sum(mfmc_vals) / len(mfmc_vals) if mfmc_vals else float("nan")
            y0 = sum(hf_vals) / len(hf_vals) if hf_vals else float("nan")
            est_var = sum(hf_vars) / len(hf_vars) if hf_vars else float("nan")
            std_error = est_var ** 0.5 if est_var >= 0 else float("nan")

            writer.writerow(
                {
                    "repeat": rep,
                    "s_hat": s_hat,
                    "estimated_variance": est_var,
                    "std_error": std_error,
                    "total_cost": total_cost,
                    "m_list": "[]",
                    "method_list": str(methods),
                    "alpha": "[]",
                    "y0": y0,
                }
            )

    return dst


def run_mfmc_legacy_wrapper(
    altitude: int = 200,
    aos: int = 0,
    db_index_max: int = 183300,
    budget_values: Optional[List[float]] = None,
    repetitions: int = 10,
    candidate_methods: Optional[List[str]] = None,
    backend: str = "legacy_slurm",
) -> Dict[str, Any]:
    if budget_values is None:
        budget_values = [100.0]
    if candidate_methods is None:
        candidate_methods = ["CLL", "Sentman", "Maxwell", "DRIA", "Schaaf"]

    all_summaries: Dict[str, Any] = {}
    for budget in budget_values:
        output_dir = os.path.join("campaign_outputs", "mfmc_test", f"{altitude}km_budget_{int(budget)}")
        config = build_legacy_mfmc_config(
            altitude=altitude,
            aos=aos,
            db_index_max=db_index_max,
            budget=float(budget),
            repetitions=repetitions,
            candidate_methods=candidate_methods,
            backend=backend,
            output_dir=output_dir,
        )
        summary = run_campaign(config, resume=False)
        legacy_csv = write_legacy_mfmc_csv(summary, altitude=altitude, budget=budget)
        summary["legacy_csv"] = legacy_csv
        all_summaries[str(int(budget))] = summary
    return all_summaries


def build_legacy_correlation_config(
    altitude: int,
    aos: int,
    db_index_max: int,
    methods: List[str],
    n_runs: int,
    backend: str,
    output_dir: str,
) -> Dict[str, Any]:
    lf_models = []
    for m in methods:
        lf_models.append(
            {
                "id": m,
                "kind": "legacy_adbsat" if backend == "legacy_slurm" else "mock",
                "method": m,
                "kwargs": {
                    "simulation_script": f"python {os.path.abspath(os.path.join('ADBSat-PyVersion', 'simulate.py'))}",
                    "base_dir": "ADBSat-PyVersion",
                },
            }
        )

    return {
        "study": {
            "id": f"legacy_corr_{altitude}km",
            "mode": "baseline",
            "active_source_blocks": ["numerical.database_index"],
        },
        "geometries": [_geometry("Cube")],
        "regimes": [_regime(altitude, f"legacy_corr_{altitude}km")],
        "sources": {"blocks": _base_sources()},
        "variables": _default_variables(db_index_max),
        "sampling": {"method": "independent", "sample_count": n_runs, "max_production_samples": max(200, n_runs)},
        "models": {
            "hf": {
                "id": "PICLas_HF",
                "kind": "legacy_piclas" if backend == "legacy_slurm" else "mock",
                "kwargs": {},
            },
            "lf": lf_models,
            "available_qois": {
                "PICLas_HF": ["C_D", "C_D2"],
                **{m: ["C_D", "C_D2"] for m in methods},
            },
        },
        "qois": {"direct": ["C_D"], "derived": []},
        "pilot": {"size": n_runs, "sizes": [n_runs], "robustness_repetitions": 10},
        "budget": {"total": float(max(50, n_runs)), "hf_fraction": 0.5},
        "repetitions": 1,
        "seeds": {"global": 42},
        "outputs": {"dir": output_dir, "write_parquet": False, "write_config_snapshot": True, "plots": False},
        "execution": {"backend": backend, "resume": False, "aos_deg": aos},
    }


def write_legacy_correlation_text(summary: Dict[str, Any], altitude: int, aos: int, methods: List[str], n_runs: int) -> str:
    src = summary["results_csv"]
    output_file = f"Correlations_{altitude}km_Sentman_CLL_{n_runs}runs.txt"

    method_corr = {m: float("nan") for m in methods}
    with open(src, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for m in methods:
        vals = [float(r.get("pearson_correlation", "nan")) for r in rows if r.get("lf_model_id") == m and r.get("qoi") == "C_D"]
        if vals:
            method_corr[m] = sum(vals) / len(vals)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Initializing Simulators done!\n")
        f.write(f"Altitude: {altitude} km\n")
        f.write(f"AoS: {aos} deg\n")
        f.write("AoA: 0 deg (fixed)\n\n")

        for m in methods:
            corr = method_corr[m]
            f.write(f"[{m}] n={n_runs}, matched_samples={n_runs}, Pearson-R: {corr:.4f}\n\n")

        f.write("Summary:\n")
        for m in methods:
            corr = method_corr[m]
            f.write(f"{m}:\n")
            f.write(f"  n={n_runs}: r = {corr:.4f} (matched={n_runs})\n")

    return output_file


def run_correlation_legacy_wrapper(
    altitudes_km: List[int],
    n_runs: int,
    aos: int,
    methods: List[str],
    db_index_max_map: Dict[int, int],
    backend: str = "legacy_slurm",
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for altitude in altitudes_km:
        db_max = db_index_max_map.get(altitude, 1000)
        output_dir = os.path.join("campaign_outputs", "correlation_study", f"{altitude}km")
        config = build_legacy_correlation_config(
            altitude=altitude,
            aos=aos,
            db_index_max=db_max,
            methods=methods,
            n_runs=n_runs,
            backend=backend,
            output_dir=output_dir,
        )
        summary = run_campaign(config, resume=False)
        legacy_text = write_legacy_correlation_text(summary, altitude=altitude, aos=aos, methods=methods, n_runs=n_runs)
        summary["legacy_text"] = legacy_text
        summaries[str(altitude)] = summary
    return summaries
