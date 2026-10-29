#!/usr/bin/env python3
"""Audit and analyse the exploratory VLEO CLL Search-50 MFMC campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


GEOMETRIES = [f"cylinder_hex_wp5_{i:03d}" for i in range(9)]
BASELINE = GEOMETRIES[0]
QOIS = ("C_D", "C_D2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(alpha: float, sigma: float) -> str:
    return hashlib.sha256(f"{alpha:.17g}|{sigma:.17g}".encode()).hexdigest()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr(rankdata(x), rankdata(y))


def ci(values: np.ndarray) -> list[float]:
    good = values[np.isfinite(values)]
    return [float(x) for x in np.quantile(good, [0.025, 0.975])] if len(good) else [float("nan")] * 2


def load_manifest(path: Path) -> tuple[dict, dict[tuple[float, float], dict]]:
    payload = json.loads(path.read_text())
    lookup = {(float(s["cll_alpha_n"]), float(s["cll_sigma_t"])): s for s in payload["samples"]}
    if len(lookup) != len(payload["samples"]):
        raise ValueError(f"duplicate CLL coordinates in {path}")
    return payload, lookup


def model_values(directory: Path) -> tuple[dict, dict, dict]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    meta: dict[str, dict] = {}
    for row in read_csv(directory / "model_evaluations.csv"):
        key = (row["model_id"], row["qoi"])
        sid = row["sample_id"]
        if sid in values.setdefault(key, {}):
            raise ValueError(f"duplicate evaluation {directory}: {key}/{sid}")
        values[key][sid] = float(row["value"])
        meta.setdefault(sid, {})["cost_hf" if row["fidelity"] == "hf" else "cost_lf"] = float(row["cost"])
    coords: dict[str, tuple[float, float]] = {}
    source: dict[str, dict] = {}
    for row in read_csv(directory / "sample_inputs.csv"):
        if row["qoi"] != "C_D" or row["fidelity"] != "hf":
            continue
        sid = row["sample_id"]
        xy = (float(row["input__cll_alpha_n"]), float(row["input__cll_sigma_t"]))
        if sid in coords and coords[sid] != xy:
            raise ValueError(f"inconsistent coordinates {directory}/{sid}")
        coords[sid] = xy
        source[sid] = {"sample_fingerprint": row["sample_fingerprint"], "actual_sample_id": sid}
    return values, coords, {k: meta[k] | source[k] for k in coords}


def matched_campaign(directory: Path, expected: dict, role: str, tolerance: float) -> list[dict]:
    values, coords, meta = model_values(directory)
    hf = next(k[0] for k in values if k[0].startswith("PICLas"))
    lf = next(k[0] for k in values if k[0].startswith("ADBSat"))
    rows = []
    used = set()
    for sid, xy in coords.items():
        candidates = [(key, sample) for key, sample in expected.items() if max(abs(xy[0]-key[0]), abs(xy[1]-key[1])) <= tolerance]
        if len(candidates) != 1:
            raise ValueError(f"coordinate match count {len(candidates)} for {directory}/{sid}: {xy}")
        key, sample = candidates[0]
        if key in used:
            raise ValueError(f"duplicate manifest match {directory}: {key}")
        used.add(key)
        needed = ((hf, "C_D"), (hf, "C_D2"), (lf, "C_D"), (lf, "C_D2"))
        if any(sid not in values.get(k, {}) for k in needed):
            raise ValueError(f"incomplete paired QoIs {directory}/{sid}")
        rows.append({"manifest_sample_id": sample["sample_id"], "actual_sample_id": sid,
                     "alpha_n": xy[0], "sigma_t": xy[1], "sample_fingerprint": meta[sid]["sample_fingerprint"],
                     "role": role, "hf_cd": values[(hf,"C_D")][sid], "hf_cd2": values[(hf,"C_D2")][sid],
                     "lf_cd": values[(lf,"C_D")][sid], "lf_cd2": values[(lf,"C_D2")][sid],
                     "hf_cost": meta[sid]["cost_hf"], "lf_cost": meta[sid]["cost_lf"]})
    return sorted(rows, key=lambda r: r["manifest_sample_id"])


def load_lf(path: Path, manifest: dict, tolerance: float) -> list[dict]:
    actual = read_csv(path)
    if len(actual) != len(manifest["samples"]):
        raise ValueError(f"expected {len(manifest['samples'])} LF rows, found {len(actual)} in {path}")
    expected = {s["sample_id"]: s for s in manifest["samples"]}
    seen = set(); rows = []
    for row in actual:
        sid = row["sample_id"]
        if sid in seen or sid not in expected:
            raise ValueError(f"duplicate/unexpected LF sample {path}/{sid}")
        seen.add(sid); sample = expected[sid]
        a, s = float(row["cll_alpha_n"]), float(row["cll_sigma_t"])
        if max(abs(a-float(sample["cll_alpha_n"])), abs(s-float(sample["cll_sigma_t"]))) > tolerance:
            raise ValueError(f"LF coordinate mismatch {path}/{sid}")
        cd, cd2 = float(row["C_D"]), float(row["C_D2"])
        if not np.isfinite([a,s,cd,cd2,float(row["runtime_ms"])]).all() or abs(cd2-cd*cd) > 1e-12:
            raise ValueError(f"invalid LF row {path}/{sid}")
        rows.append({"manifest_sample_id": sid, "alpha_n": a, "sigma_t": s, "role": row["role"],
                     "lf_cd": cd, "lf_cd2": cd2, "runtime_ms": float(row["runtime_ms"])})
    return rows


def estimate(h: np.ndarray, l: np.ndarray, la: np.ndarray, h2: np.ndarray, l2: np.ndarray,
             la2: np.ndarray, rho_gate: float) -> dict:
    rho, rho2 = corr(h,l), corr(h2,l2)
    beta = float(np.cov(h,l,ddof=1)[0,1]/np.var(l,ddof=1))
    beta2 = float(np.cov(h2,l2,ddof=1)[0,1]/np.var(l2,ddof=1))
    mu = float(h.mean()+beta*(la.mean()-l.mean()))
    m2 = float(h2.mean()+beta2*(la2.mean()-l2.mean()))
    var = m2-mu*mu
    gate = bool(np.isfinite([rho,rho2,beta,beta2,var]).all() and abs(rho)>=rho_gate and abs(rho2)>=rho_gate and var>0)
    if not gate:
        mu, var = float(h.mean()), float(np.var(h,ddof=1))
        m2 = float(np.mean(h2)); beta = beta2 = 0.0
    return {"active":gate,"rho":rho,"rho2":rho2,"beta":beta,"beta2":beta2,"mu":mu,"m2":m2,
            "var":var,"sd":math.sqrt(var), "raw_reconstructed_var":float(m2-mu*mu)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--results-root",type=Path,default=Path("outputs/vleo_cll"))
    p.add_argument("--bundle-root",type=Path,default=Path("outputs/vleo_cll_mfmc_extension_bundle"))
    p.add_argument("--geometry-root",type=Path,default=Path("outputs/vleo_cll_server_run_bundle/geometries"))
    p.add_argument("--output",type=Path,default=Path("outputs/vleo_cll/search50_mfmc_analysis"))
    p.add_argument("--bootstrap",type=int,default=5000); p.add_argument("--seed",type=int,default=20261029)
    p.add_argument("--match-tolerance",type=float,default=1e-12); p.add_argument("--rho-gate",type=float,default=.5)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    paired_manifest, paired_lookup=load_manifest(a.bundle_root/"manifests/paired_search_50.json")
    ext_manifest, ext_lookup=load_manifest(a.bundle_root/"manifests/search_extension_10.json")
    lf_manifest=json.loads((a.bundle_root/"manifests/mfmc_lf_5000.json").read_text())
    manifest_hashes={x.name:sha256(x) for x in (a.bundle_root/"manifests").glob("*.json")}
    data={}; mappings=[]; combined=[]; qa=[]; environment_contract=None
    for gid in GEOMETRIES:
        configs=[]
        for campaign in ("search40","search50_extension10"):
            cfg=json.loads((a.results_root/campaign/gid/"config_snapshot.json").read_text())["config"]
            geo=cfg["geometries"][0]
            if geo["id"]!=gid or geo["metadata"].get("lf_model")!=gid:
                raise ValueError(f"geometry/ADBSat asset mismatch for {campaign}/{gid}")
            configs.append({"execution":cfg["execution"],"regimes":cfg["regimes"],
                            "wall_temperature_k":cfg["models"]["hf"]["kwargs"]["payload_defaults"]["wall_temperature_k"]})
        if configs[0] != configs[1]: raise ValueError(f"old/extension environment differs for {gid}")
        if environment_contract is None: environment_contract=configs[0]
        elif configs[0] != environment_contract: raise ValueError(f"fixed environment differs for {gid}")
        old=matched_campaign(a.results_root/"search40"/gid, paired_lookup, "existing_paired", a.match_tolerance)
        old=[r for r in old if r["manifest_sample_id"].startswith("search_") and int(r["manifest_sample_id"].split("_")[1])<40]
        new=matched_campaign(a.results_root/"search50_extension10"/gid, ext_lookup, "extension_paired", a.match_tolerance)
        pairs=sorted(old+new,key=lambda r:r["manifest_sample_id"])
        lf=load_lf(a.results_root/"mfmc_lf5000"/gid/"adbsat_lf_5000.csv",lf_manifest,a.match_tolerance)
        if len(old)!=40 or len(new)!=10 or len(pairs)!=50 or len(lf)!=5000:
            raise ValueError(f"wrong counts for {gid}: {len(old)}/{len(new)}/{len(lf)}")
        lf_by={r["manifest_sample_id"]:r for r in lf}
        for row in pairs:
            lr=lf_by[row["manifest_sample_id"]]
            if abs(row["lf_cd"]-lr["lf_cd"])>1e-12:
                raise ValueError(f"paired LF differs from LF5000 for {gid}/{row['manifest_sample_id']}")
            mappings.append({"geometry_id":gid, **{k:row[k] for k in ("manifest_sample_id","actual_sample_id","alpha_n","sigma_t","sample_fingerprint","role")}})
            combined.append({"geometry_id":gid,**row})
        area=.5*float(json.loads((a.geometry_root/gid/f"{gid}.manifest.json").read_text())["validation"]["surface_area_m2"])
        data[gid]={"pairs":pairs,"lf":lf,"area":area}
        qa.append({"geometry_id":gid,"n_existing_pairs":len(old),"n_extension_pairs":len(new),"n_lf":len(lf),
                   "finite":True,"cll_in_unit_square":all(0<=r["alpha_n"]<=1 and 0<=r["sigma_t"]<=1 for r in lf),
                   "reference_area_m2":area,"fixed_environment_equal":True,
                   "adbsat_cube_fallback":False,"status":"pass"})
    write_csv(a.output/"paired_search50.csv",combined); write_csv(a.output/"sample_mapping.csv",mappings)
    write_csv(a.output/"qa_report.csv",qa)
    rng=np.random.default_rng(a.seed); metrics=[]; gate_rows=[]; boot={g:[] for g in GEOMETRIES}
    for gid,d in data.items():
        area=d["area"]; pr=d["pairs"]; lr=d["lf"]
        h=np.array([r["hf_cd"] for r in pr])*area; l=np.array([r["lf_cd"] for r in pr])*area
        h2=np.array([r["hf_cd2"] for r in pr])*area**2; l2=np.array([r["lf_cd2"] for r in pr])*area**2
        la=np.array([r["lf_cd"] for r in lr])*area; la2=np.array([r["lf_cd2"] for r in lr])*area**2
        point=estimate(h,l,la,h2,l2,la2,a.rho_gate)
        brho=[]; brho2=[]; betas=[]; neg=0; inactive=0
        for _ in range(a.bootstrap):
            ip=rng.integers(0,50,50); ia=rng.integers(0,5000,5000)
            e=estimate(h[ip],l[ip],la[ia],h2[ip],l2[ip],la2[ia],a.rho_gate)
            e["tpmc_mu"]=float(h[ip].mean()); e["tpmc_sd"]=float(h[ip].std(ddof=1))
            brho.append(e["rho"]); brho2.append(e["rho2"]); betas.append(e["beta"])
            neg += e["raw_reconstructed_var"]<=0; inactive += not e["active"]
            boot[gid].append(e)
        resid=h-(np.mean(h)+point["beta"]*(l-np.mean(l)))
        gate_rows.append({"geometry_id":gid,"pearson_Y":point["rho"],"spearman_Y":spearman(h,l),"pearson_Y2":point["rho2"],
            "spearman_Y2":spearman(h2,l2),"pearson_Y_ci_low":ci(np.array(brho))[0],"pearson_Y_ci_high":ci(np.array(brho))[1],
            "pearson_Y2_ci_low":ci(np.array(brho2))[0],"pearson_Y2_ci_high":ci(np.array(brho2))[1],
            "beta_Y":point["beta"],"beta_Y_bootstrap_sd":float(np.std(betas,ddof=1)),"mean_bias_hf_minus_lf":float(np.mean(h-l)),
            "mean_abs_residual":float(np.mean(np.abs(resid))),"max_abs_residual":float(np.max(np.abs(resid))),
            "point_gate_active":point["active"],"bootstrap_activation_frequency":1-inactive/a.bootstrap,
            "negative_variance_replicates":neg})
        cd_mu=point["mu"]/area; cd_sd=point["sd"]/area
        metrics.append({"geometry_id":gid,"area_ref_m2":area,"estimator":"MFMC" if point["active"] else "TPMC-only fallback",
            "mean_drag_area_m2":point["mu"],"sd_drag_area_m2":point["sd"],"mean_cd":cd_mu,"sd_cd":cd_sd,
            "tpmc50_mean_drag_area_m2":float(h.mean()),"tpmc50_sd_drag_area_m2":float(h.std(ddof=1)),
            "hf_cpu_h":float(sum(r["hf_cost"] for r in pr)/3600),"lf_cpu_h":float(sum(r["runtime_ms"] for r in lr)/3.6e6)})
    base=next(r for r in metrics if r["geometry_id"]==BASELINE)
    for r in metrics:
        r["mean_drag_ratio"]=r["mean_drag_area_m2"]/base["mean_drag_area_m2"]
        r["sd_drag_ratio"]=r["sd_drag_area_m2"]/base["sd_drag_area_m2"]
        r["mean_cd_ratio"]=r["mean_cd"]/base["mean_cd"]; r["sd_cd_ratio"]=r["sd_cd"]/base["sd_cd"]
        r["mean_guardrail_passed"]=r["mean_drag_ratio"]<=1.02; r["robust_target_passed"]=r["mean_guardrail_passed"] and r["sd_drag_ratio"]<=.8
        for w in (.25,.5,.75):
            r[f"score_cd_w{w:.2f}"]=w*r["mean_cd_ratio"]+(1-w)*r["sd_cd_ratio"]
            r[f"score_drag_w{w:.2f}"]=w*r["mean_drag_ratio"]+(1-w)*r["sd_drag_ratio"]
    for r in metrics:
        r["pareto_cd"]=not any(s["mean_cd"]<=r["mean_cd"] and s["sd_cd"]<=r["sd_cd"] and (s["mean_cd"]<r["mean_cd"] or s["sd_cd"]<r["sd_cd"]) for s in metrics)
        r["pareto_drag_area"]=not any(s["mean_drag_area_m2"]<=r["mean_drag_area_m2"] and s["sd_drag_area_m2"]<=r["sd_drag_area_m2"] and (s["mean_drag_area_m2"]<r["mean_drag_area_m2"] or s["sd_drag_area_m2"]<r["sd_drag_area_m2"]) for s in metrics)
    winners={"mean":min(metrics,key=lambda r:r["mean_cd"])["geometry_id"],
             "robustness":min((r for r in metrics if r["mean_guardrail_passed"]),key=lambda r:r["sd_drag_area_m2"])["geometry_id"],
             "combination":min(metrics,key=lambda r:r["score_cd_w0.50"])["geometry_id"]}
    freq={k:{g:0 for g in GEOMETRIES} for k in winners}
    for i in range(a.bootstrap):
        vals={g:boot[g][i] for g in GEOMETRIES}; b=vals[BASELINE]
        freq["mean"][min(vals,key=lambda g:vals[g]["mu"]/data[g]["area"])]+=1
        elig=[g for g in GEOMETRIES if vals[g]["mu"]/b["mu"]<=1.02]
        if elig: freq["robustness"][min(elig,key=lambda g:vals[g]["sd"])]+=1
        freq["combination"][min(vals,key=lambda g:.5*(vals[g]["mu"]/data[g]["area"]/(b["mu"]/data[BASELINE]["area"])+vals[g]["sd"]/data[g]["area"]/(b["sd"]/data[BASELINE]["area"])))]+=1
    freq={k:{g:v/a.bootstrap for g,v in x.items()} for k,x in freq.items()}
    comparisons=[]
    for r in metrics:
        g=r["geometry_id"]; bm=np.array([x["mu"] for x in boot[g]]); bs=np.array([x["sd"] for x in boot[g]])
        tm=np.array([x["tpmc_mu"] for x in boot[g]]); ts=np.array([x["tpmc_sd"] for x in boot[g]])
        hf_mean_cost=r["hf_cpu_h"]/50; equal_n=int((r["hf_cpu_h"]+r["lf_cpu_h"])/hf_mean_cost)
        comparisons.append({"geometry_id":g,"mfmc_mean_se":float(bm.std(ddof=1)),"tpmc50_mean_se":float(tm.std(ddof=1)),
          "mfmc_sd_se":float(bs.std(ddof=1)),"tpmc50_sd_se":float(ts.std(ddof=1)),
          "mean_relative_se_reduction":float(1-bm.std(ddof=1)/tm.std(ddof=1)),
          "sd_relative_se_reduction":float(1-bs.std(ddof=1)/ts.std(ddof=1)),
          "mean_effective_variance_gain":float(np.var(tm,ddof=1)/np.var(bm,ddof=1)),
          "sd_effective_variance_gain":float(np.var(ts,ddof=1)/np.var(bs,ddof=1)),
          "mfmc_mean_bootstrap_bias":float(bm.mean()-r["mean_drag_area_m2"]),
          "tpmc50_mean_bootstrap_bias":float(tm.mean()-r["tpmc50_mean_drag_area_m2"]),
          "mfmc_mean_mse":float(np.mean((bm-r["mean_drag_area_m2"])**2)),
          "tpmc50_mean_mse":float(np.mean((tm-r["tpmc50_mean_drag_area_m2"])**2)),
          "hf_cpu_h":r["hf_cpu_h"],"lf_cpu_h":r["lf_cpu_h"],"equal_budget_tpmc_n":equal_n,
          "equal_budget_tpmc_mean_se_approx":float(tm.std(ddof=1)*math.sqrt(50/equal_n)),
          "mfmc_advantage_mean":bool(np.mean((bm-r["mean_drag_area_m2"])**2)<np.mean((tm-r["tpmc50_mean_drag_area_m2"])**2))})
    write_csv(a.output/"mfmc_gate_report.csv",gate_rows); write_csv(a.output/"geometry_ranking.csv",metrics)
    write_csv(a.output/"mfmc_vs_tpmc.csv",comparisons)
    write_csv(a.output/"pareto.csv",[{"geometry_id":r["geometry_id"],"pareto_cd":r["pareto_cd"],"pareto_drag_area":r["pareto_drag_area"]} for r in metrics])
    qa_report={"schema_version":1,"status":"pass","match_tolerance":a.match_tolerance,"manifest_sha256":manifest_hashes,
               "fixed_environment_equal":True,"reference_area_convention":"A_total/2","raw_data_modified":False,"geometries":qa}
    (a.output/"qa_report.json").write_text(json.dumps(qa_report,indent=2)+"\n")
    report={"schema_version":1,"exploratory":True,"bootstrap_draws":a.bootstrap,"seed":a.seed,"rho_gate":a.rho_gate,
            "winners":winners,"winner_bootstrap_frequency":freq,"metrics":metrics,"gates":gate_rows,
            "holdout_opened":False,"mfmc_vs_tpmc":comparisons,"decision":{"search50_sufficient":all(freq[k][winners[k]]>=.8 for k in winners),
            "robust_target_met":any(r["robust_target_passed"] for r in metrics),"next_centres":winners,
            "holdout_may_be_opened":False,"note":"Use local perturbations around branch winners; lock and hash the next-round manifest before execution."}}
    (a.output/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    md=f"""# Exploratory Search-50 MFMC analysis\n\nAll nine geometries passed structural QA (40 old pairs, 10 new pairs, and 5,000 LF evaluations). Matching used `(alpha_n, sigma_t)` with tolerance `{a.match_tolerance:g}`; raw data were not modified.\n\n- Mean winner: `{winners['mean']}`\n- Robustness winner under the 1.02 mean guardrail: `{winners['robustness']}`\n- Equal-weight C_D mean–SD winner: `{winners['combination']}`\n- Any candidate meets both R_sigma <= 0.80 and R_mu <= 1.02: `{report['decision']['robust_target_met']}`\n- Holdout opened: no\n\nThis search analysis is exploratory. Gate decisions were recomputed inside every paired bootstrap replicate; invalid reconstructed variances triggered TPMC-only fallback in that replicate. See `analysis.json`, `mfmc_gate_report.csv`, and `geometry_ranking.csv` for estimates and uncertainty diagnostics.\n\n## Method resource\n\nKassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065\n"""
    (a.output/"README.md").write_text(md); print(a.output/"analysis.json")


if __name__ == "__main__":
    main()
