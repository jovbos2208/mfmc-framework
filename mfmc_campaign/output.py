from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .reproducibility import write_json


RESULT_COLUMNS = [
    "study_id",
    "cell_id",
    "mode",
    "geometry_id",
    "geometry_name",
    "geometry_class",
    "geometry_characteristic_length",
    "regime_id",
    "regime_label",
    "regime_altitude_km",
    "regime_knudsen_number",
    "regime_speed_ratio",
    "regime_freestream_temperature",
    "regime_composition_descriptor",
    "regime_solar_activity_state",
    "regime_geomagnetic_activity_state",
    "regime_wind_state",
    "regime_surface_state",
    "active_sources",
    "qoi",
    "quantity_kind",
    "qoi_expression",
    "hf_model_id",
    "lf_model_id",
    "pilot_size",
    "budget",
    "repetition",
    "seed",
    "rho_hat",
    "beta_hat",
    "r2_lin_hat",
    "residual_var_hat",
    "residual_ratio_hat",
    "beta_sign_flip_prob",
    "beta_cv",
    "pearson_correlation",
    "spearman_correlation",
    "control_variate_beta",
    "paper_mfmc_sample_counts",
    "paper_mfmc_betas",
    "hf_mean",
    "lf_mean",
    "hf_variance",
    "lf_variance",
    "covariance_hf_lf",
    "residual_variance",
    "cost_ratio",
    "predicted_variance_reduction",
    "mfmc_estimate",
    "hf_only_estimate",
    "reference_estimate",
    "realized_mfmc_error",
    "realized_hf_error",
    "cost_normalized_gain",
    "unstable_weight",
    "underperform_hf",
    "flags",
]

MODEL_EVALUATION_COLUMNS = [
    "study_id",
    "cell_id",
    "phase",
    "mode",
    "geometry_id",
    "regime_id",
    "active_sources",
    "qoi",
    "model_id",
    "fidelity",
    "hf_model_id",
    "lf_model_id",
    "pilot_size",
    "budget",
    "repetition",
    "seed",
    "sample_id",
    "sample_index",
    "sample_fingerprint",
    "request_fingerprint",
    "value",
    "cost",
]

ROBUSTNESS_COLUMNS = [
    "study_id",
    "cell_id",
    "mode",
    "geometry_id",
    "geometry_class",
    "regime_id",
    "active_sources",
    "qoi",
    "hf_model_id",
    "lf_model_id",
    "repetition",
    "pilot_size",
    "correlation_mean",
    "correlation_std",
    "beta_mean",
    "beta_std",
    "allocation_ratio_mean",
    "allocation_ratio_std",
    "negative_weight_frequency",
    "unstable_weight_frequency",
    "underperform_frequency",
    "gain_mean",
    "gain_p10",
    "gain_p50",
    "gain_p90",
]


class EvaluationCache:
    def __init__(self, path: str):
        self.path = path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    self._cache = data
            except json.JSONDecodeError:
                self._cache = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._cache[key] = value
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        self._dirty = False


class ResultStore:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.results_csv = os.path.join(self.output_dir, "results_long.csv")
        self.model_evaluations_csv = os.path.join(self.output_dir, "model_evaluations.csv")
        self.robustness_csv = os.path.join(self.output_dir, "pilot_robustness.csv")
        self.summary_json = os.path.join(self.output_dir, "summary.json")
        self.config_json = os.path.join(self.output_dir, "config_snapshot.json")
        self.cache_json = os.path.join(self.output_dir, "evaluation_cache.json")
        self.source_ranking_csv = os.path.join(self.output_dir, "summary_source_ranking.csv")
        self.regime_dependence_csv = os.path.join(self.output_dir, "summary_regime_dependence.csv")
        self.interaction_deltas_csv = os.path.join(self.output_dir, "summary_interaction_deltas.csv")
        self.pilot_robustness_summary_csv = os.path.join(self.output_dir, "summary_pilot_robustness.csv")
        self.geometry_summary_csv = os.path.join(self.output_dir, "summary_geometry_class.csv")
        self.budget_quantiles_csv = os.path.join(self.output_dir, "summary_budget_gain_quantiles.csv")

    def reset_outputs(self, keep_cache: bool = True) -> None:
        removable = [
            self.results_csv,
            self.model_evaluations_csv,
            self.robustness_csv,
            self.summary_json,
            os.path.join(self.output_dir, "predictive_dataset.csv"),
            os.path.join(self.output_dir, "results_long.parquet"),
            self.source_ranking_csv,
            self.regime_dependence_csv,
            self.interaction_deltas_csv,
            self.pilot_robustness_summary_csv,
            self.geometry_summary_csv,
            self.budget_quantiles_csv,
        ]
        if not keep_cache:
            removable.append(self.cache_json)

        for path in removable:
            if os.path.exists(path):
                os.remove(path)

        for fn in os.listdir(self.output_dir):
            if fn.endswith(".png"):
                os.remove(os.path.join(self.output_dir, fn))

    def load_completed_cell_ids(self) -> Set[str]:
        done: Set[str] = set()
        if not os.path.exists(self.results_csv):
            return done

        with open(self.results_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("cell_id")
                if cid:
                    done.add(cid)
        return done

    def _rewrite_without_cell_id(self, path: str, fieldnames: List[str], cell_id: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", newline="") as src:
            rows = [row for row in csv.DictReader(src) if row.get("cell_id") != cell_id]
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, path)

    def remove_cell_outputs(self, cell_id: str) -> None:
        self._rewrite_without_cell_id(self.results_csv, RESULT_COLUMNS, cell_id)
        self._rewrite_without_cell_id(self.robustness_csv, ROBUSTNESS_COLUMNS, cell_id)

    def _ensure_csv_header(self, path: str, fieldnames: List[str]) -> bool:
        file_exists = os.path.exists(path)
        if file_exists:
            with open(path, "r", encoding="utf-8", newline="") as existing:
                reader = csv.reader(existing)
                try:
                    existing_header = next(reader)
                except StopIteration:
                    existing_header = []
            if not existing_header:
                file_exists = False
            if existing_header and existing_header != fieldnames and set(existing_header).issubset(set(fieldnames)):
                with open(path, "r", encoding="utf-8", newline="") as existing:
                    rows = list(csv.DictReader(existing))
                tmp_path = f"{path}.tmp"
                with open(tmp_path, "w", encoding="utf-8", newline="") as tmp:
                    writer = csv.DictWriter(tmp, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for existing_row in rows:
                        writer.writerow(existing_row)
                os.replace(tmp_path, path)
        return file_exists

    def _append_csv_row(self, path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
        file_exists = self._ensure_csv_header(path, fieldnames)
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _append_csv_rows(self, path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        file_exists = self._ensure_csv_header(path, fieldnames)
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def append_result(self, row: Dict[str, Any]) -> None:
        payload = dict(row)
        if isinstance(payload.get("active_sources"), list):
            payload["active_sources"] = "+".join(payload["active_sources"])
        if isinstance(payload.get("flags"), list):
            payload["flags"] = "+".join(payload["flags"])
        self._append_csv_row(self.results_csv, RESULT_COLUMNS, payload)

    def append_model_evaluation(self, row: Dict[str, Any]) -> None:
        payload = dict(row)
        if isinstance(payload.get("active_sources"), list):
            payload["active_sources"] = "+".join(payload["active_sources"])
        self._append_csv_row(self.model_evaluations_csv, MODEL_EVALUATION_COLUMNS, payload)

    def append_model_evaluations(self, rows: List[Dict[str, Any]]) -> None:
        payloads = []
        for row in rows:
            payload = dict(row)
            if isinstance(payload.get("active_sources"), list):
                payload["active_sources"] = "+".join(payload["active_sources"])
            payloads.append(payload)
        self._append_csv_rows(self.model_evaluations_csv, MODEL_EVALUATION_COLUMNS, payloads)

    def append_robustness(self, row: Dict[str, Any]) -> None:
        payload = dict(row)
        if isinstance(payload.get("active_sources"), list):
            payload["active_sources"] = "+".join(payload["active_sources"])
        self._append_csv_row(self.robustness_csv, ROBUSTNESS_COLUMNS, payload)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        write_json(self.summary_json, summary)

    def write_config_snapshot(self, config: Dict[str, Any], fingerprint: Dict[str, Any]) -> None:
        payload = {"config": config, "fingerprint": fingerprint}
        write_json(self.config_json, payload)

    def write_optional_parquet(self, source_csv: str, target_name: str = "results_long.parquet") -> Optional[str]:
        target = os.path.join(self.output_dir, target_name)
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(source_csv)
            df.to_parquet(target, index=False)
            return target
        except Exception:
            return None


def export_predictive_dataset(results_csv: str, target_csv: str, robustness_csv: Optional[str] = None) -> None:
    keep_cols = [
        "study_id",
        "mode",
        "geometry_id",
        "geometry_class",
        "geometry_characteristic_length",
        "regime_id",
        "hf_model_id",
        "lf_model_id",
        "regime_altitude_km",
        "regime_knudsen_number",
        "regime_speed_ratio",
        "regime_freestream_temperature",
        "regime_composition_descriptor",
        "regime_solar_activity_state",
        "regime_geomagnetic_activity_state",
        "regime_wind_state",
        "regime_surface_state",
        "active_sources",
        "qoi",
        "quantity_kind",
        "pilot_size",
        "budget",
        "pearson_correlation",
        "rho_hat",
        "spearman_correlation",
        "control_variate_beta",
        "beta_hat",
        "r2_lin_hat",
        "residual_var_hat",
        "residual_ratio_hat",
        "residual_variance",
        "cost_ratio",
        "predicted_variance_reduction",
        "cost_normalized_gain",
        "underperform_hf",
        "unstable_weight",
        "beta_sign_flip_prob",
        "beta_cv",
        "flags",
        "robust_beta_std",
        "robust_correlation_std",
        "robust_allocation_ratio_std",
        "robust_negative_weight_frequency",
        "robust_unstable_weight_frequency",
        "robust_underperform_frequency",
        "robust_gain_mean",
        "robust_gain_p50",
    ]

    with open(results_csv, "r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        rows = list(reader)

    robust_by_cell: Dict[str, Dict[str, str]] = {}
    if robustness_csv and os.path.exists(robustness_csv):
        with open(robustness_csv, "r", encoding="utf-8", newline="") as src:
            for row in csv.DictReader(src):
                cell_id = str(row.get("cell_id", ""))
                if not cell_id:
                    continue
                # Keep the largest pilot-size entry per cell when multiple rows exist.
                old = robust_by_cell.get(cell_id)
                if old is None:
                    robust_by_cell[cell_id] = row
                    continue
                old_n = float(old.get("pilot_size", "nan"))
                new_n = float(row.get("pilot_size", "nan"))
                if new_n >= old_n:
                    robust_by_cell[cell_id] = row

    os.makedirs(os.path.dirname(target_csv), exist_ok=True)
    with open(target_csv, "w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=keep_cols)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in keep_cols}
            robust = robust_by_cell.get(str(row.get("cell_id", "")), {})
            out["robust_beta_std"] = robust.get("beta_std")
            out["robust_correlation_std"] = robust.get("correlation_std")
            out["robust_allocation_ratio_std"] = robust.get("allocation_ratio_std")
            out["robust_negative_weight_frequency"] = robust.get("negative_weight_frequency")
            out["robust_unstable_weight_frequency"] = robust.get("unstable_weight_frequency")
            out["robust_underperform_frequency"] = robust.get("underperform_frequency")
            out["robust_gain_mean"] = robust.get("gain_mean")
            out["robust_gain_p50"] = robust.get("gain_p50")
            writer.writerow(out)


def _read_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Optional[str]) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def _write_rows(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_means(rows: List[Dict[str, str]], key_fields: List[str], value_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        groups.setdefault(key, {v: [] for v in value_fields})
        for vf in value_fields:
            val = _to_float(row.get(vf))
            if val == val:  # not NaN
                groups[key][vf].append(val)

    out: List[Dict[str, Any]] = []
    for key, vals in groups.items():
        item = {k: v for k, v in zip(key_fields, key)}
        for vf in value_fields:
            arr = vals[vf]
            item[vf + "_mean"] = sum(arr) / len(arr) if arr else float("nan")
        out.append(item)
    return out


def _group_gain_quantiles(rows: List[Dict[str, str]], key_fields: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        slot = groups.setdefault(key, {"gain": [], "fail": []})
        gain = _to_float(row.get("cost_normalized_gain"))
        if gain == gain:
            slot["gain"].append(gain)
        fail = _to_float(row.get("underperform_hf"))
        if fail == fail:
            slot["fail"].append(fail)

    out: List[Dict[str, Any]] = []
    for key, vals in groups.items():
        gains = vals["gain"]
        fails = vals["fail"]
        item = {k: v for k, v in zip(key_fields, key)}
        item["n_rows"] = len(gains)
        if gains:
            item["gain_p05"] = float(np.nanpercentile(np.asarray(gains, dtype=float), 5.0))
            item["gain_p50"] = float(np.nanpercentile(np.asarray(gains, dtype=float), 50.0))
            item["gain_p95"] = float(np.nanpercentile(np.asarray(gains, dtype=float), 95.0))
        else:
            item["gain_p05"] = float("nan")
            item["gain_p50"] = float("nan")
            item["gain_p95"] = float("nan")
        item["mfmc_fail_rate"] = (sum(fails) / len(fails)) if fails else float("nan")
        out.append(item)
    return out


def build_interaction_deltas(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, ...], Dict[str, Dict[str, List[float]]]] = {}

    for row in rows:
        if row.get("quantity_kind") != "direct":
            continue
        active = str(row.get("active_sources", ""))
        if not active:
            continue

        base_key = (
            str(row.get("study_id", "")),
            str(row.get("geometry_id", "")),
            str(row.get("regime_id", "")),
            str(row.get("qoi", "")),
            str(row.get("hf_model_id", "")),
            str(row.get("lf_model_id", "")),
        )
        slot = by_key.setdefault(base_key, {})
        vals = slot.setdefault(active, {"corr": [], "gain": []})

        corr = _to_float(row.get("pearson_correlation"))
        gain = _to_float(row.get("cost_normalized_gain"))
        if corr == corr:
            vals["corr"].append(corr)
        if gain == gain:
            vals["gain"].append(gain)

    deltas: List[Dict[str, Any]] = []
    for base_key, source_map in by_key.items():
        for active, metrics in source_map.items():
            parts = [p for p in active.split("+") if p]
            if len(parts) != 2:
                continue
            a, b = sorted(parts)
            pair_corr = sum(metrics["corr"]) / len(metrics["corr"]) if metrics["corr"] else float("nan")
            pair_gain = sum(metrics["gain"]) / len(metrics["gain"]) if metrics["gain"] else float("nan")

            a_m = source_map.get(a, {"corr": [], "gain": []})
            b_m = source_map.get(b, {"corr": [], "gain": []})
            iso_corr_vals = a_m["corr"] + b_m["corr"]
            iso_gain_vals = a_m["gain"] + b_m["gain"]
            iso_corr_mean = sum(iso_corr_vals) / len(iso_corr_vals) if iso_corr_vals else float("nan")
            iso_gain_mean = sum(iso_gain_vals) / len(iso_gain_vals) if iso_gain_vals else float("nan")

            deltas.append(
                {
                    "study_id": base_key[0],
                    "geometry_id": base_key[1],
                    "regime_id": base_key[2],
                    "qoi": base_key[3],
                    "hf_model_id": base_key[4],
                    "lf_model_id": base_key[5],
                    "pair_sources": f"{a}+{b}",
                    "isolated_corr_mean": iso_corr_mean,
                    "pair_corr_mean": pair_corr,
                    "corr_degradation": pair_corr - iso_corr_mean if pair_corr == pair_corr and iso_corr_mean == iso_corr_mean else float("nan"),
                    "isolated_gain_mean": iso_gain_mean,
                    "pair_gain_mean": pair_gain,
                    "gain_delta": pair_gain - iso_gain_mean if pair_gain == pair_gain and iso_gain_mean == iso_gain_mean else float("nan"),
                }
            )

    return deltas


def write_summary_tables(output_dir: str, results_csv: str, robustness_csv: str) -> Dict[str, str]:
    rows = _read_rows(results_csv)
    robust_rows = _read_rows(robustness_csv)

    source_rows = _group_means(
        rows=[r for r in rows if r.get("quantity_kind") == "direct"],
        key_fields=["study_id", "lf_model_id", "active_sources", "qoi"],
        value_fields=["pearson_correlation", "cost_normalized_gain", "residual_variance"],
    )
    source_csv = os.path.join(output_dir, "summary_source_ranking.csv")
    _write_rows(
        source_csv,
        [
            "study_id",
            "lf_model_id",
            "active_sources",
            "qoi",
            "pearson_correlation_mean",
            "cost_normalized_gain_mean",
            "residual_variance_mean",
        ],
        source_rows,
    )

    regime_rows = _group_means(
        rows=[r for r in rows if r.get("quantity_kind") == "direct"],
        key_fields=["study_id", "lf_model_id", "regime_id", "regime_altitude_km", "regime_knudsen_number", "qoi"],
        value_fields=["pearson_correlation", "cost_normalized_gain", "residual_variance"],
    )
    regime_csv = os.path.join(output_dir, "summary_regime_dependence.csv")
    _write_rows(
        regime_csv,
        [
            "study_id",
            "lf_model_id",
            "regime_id",
            "regime_altitude_km",
            "regime_knudsen_number",
            "qoi",
            "pearson_correlation_mean",
            "cost_normalized_gain_mean",
            "residual_variance_mean",
        ],
        regime_rows,
    )

    interaction_rows = build_interaction_deltas(rows)
    interaction_csv = os.path.join(output_dir, "summary_interaction_deltas.csv")
    _write_rows(
        interaction_csv,
        [
            "study_id",
            "geometry_id",
            "regime_id",
            "qoi",
            "hf_model_id",
            "lf_model_id",
            "pair_sources",
            "isolated_corr_mean",
            "pair_corr_mean",
            "corr_degradation",
            "isolated_gain_mean",
            "pair_gain_mean",
            "gain_delta",
        ],
        interaction_rows,
    )

    pilot_summary = _group_means(
        rows=robust_rows,
        key_fields=["study_id", "lf_model_id", "qoi", "pilot_size"],
        value_fields=[
            "correlation_std",
            "beta_std",
            "allocation_ratio_std",
            "negative_weight_frequency",
            "unstable_weight_frequency",
            "underperform_frequency",
            "gain_mean",
        ],
    )
    pilot_csv = os.path.join(output_dir, "summary_pilot_robustness.csv")
    _write_rows(
        pilot_csv,
        [
            "study_id",
            "lf_model_id",
            "qoi",
            "pilot_size",
            "correlation_std_mean",
            "beta_std_mean",
            "allocation_ratio_std_mean",
            "negative_weight_frequency_mean",
            "unstable_weight_frequency_mean",
            "underperform_frequency_mean",
            "gain_mean_mean",
        ],
        pilot_summary,
    )

    geometry_rows = _group_means(
        rows=[r for r in rows if r.get("quantity_kind") == "direct"],
        key_fields=["study_id", "lf_model_id", "geometry_class", "qoi"],
        value_fields=["cost_normalized_gain", "pearson_correlation"],
    )
    geometry_csv = os.path.join(output_dir, "summary_geometry_class.csv")
    _write_rows(
        geometry_csv,
        ["study_id", "lf_model_id", "geometry_class", "qoi", "cost_normalized_gain_mean", "pearson_correlation_mean"],
        geometry_rows,
    )

    budget_rows = _group_gain_quantiles(
        rows=[r for r in rows if r.get("quantity_kind") == "direct"],
        key_fields=[
            "study_id",
            "mode",
            "geometry_id",
            "regime_id",
            "active_sources",
            "qoi",
            "hf_model_id",
            "lf_model_id",
            "budget",
        ],
    )
    budget_csv = os.path.join(output_dir, "summary_budget_gain_quantiles.csv")
    _write_rows(
        budget_csv,
        [
            "study_id",
            "mode",
            "geometry_id",
            "regime_id",
            "active_sources",
            "qoi",
            "hf_model_id",
            "lf_model_id",
            "budget",
            "n_rows",
            "gain_p05",
            "gain_p50",
            "gain_p95",
            "mfmc_fail_rate",
        ],
        budget_rows,
    )

    return {
        "source_ranking": source_csv,
        "regime_dependence": regime_csv,
        "interaction_deltas": interaction_csv,
        "pilot_robustness": pilot_csv,
        "geometry_class": geometry_csv,
        "budget_gain_quantiles": budget_csv,
    }
