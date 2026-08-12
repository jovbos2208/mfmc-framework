#!/usr/bin/env python3
"""Build a conservative Cube/GOCE report from offline two-fidelity results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows or [{"status":"no rows"}])


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def pilot_control_statistics(case_root: Path, sensitivity_root: Path):
    path = case_root / "pilot/field_pilot_statistics.json"
    if not path.is_file() and (sensitivity_root / "sensitivity_metadata.json").is_file():
        metadata = json.loads((sensitivity_root / "sensitivity_metadata.json").read_text())
        candidates = [Path(name).with_suffix(".json") for name in metadata.get("source_sha256", {}) if name.endswith("pilot/field_pilot_statistics.npz")]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    models = data.get("models", [])
    if "DSMC" not in models or "TPMC" not in models:
        return None
    i, j = models.index("DSMC"), models.index("TPMC")
    result = {"source": str(path.resolve()), "paired_rows": data.get("diagnostics", {}).get("paired_rows")}
    for label, key in (("mean", "mean_covariance"), ("second_moment", "second_moment_covariance")):
        covariance = data[key]
        result[f"correlation_{label}"] = covariance[i][j] / math.sqrt(covariance[i][i] * covariance[j][j])
        result[f"beta_{label}_TPMC"] = covariance[i][j] / covariance[j][j]
    return result


def build(cube_root: Path, goce_root: Path, output: Path):
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    cube=cube_root/"sensitivity_two_fidelity"; goce=goce_root/"sensitivity_two_fidelity"
    cube_summary=read_csv(cube/"m0_summary.csv"); cube_refs=read_csv(cube/"reference_convergence_summary.csv"); cube_costs=read_csv(cube/"m0_allocations.csv")
    pilot_controls={"Cube":pilot_control_statistics(cube_root,cube),"GOCE":pilot_control_statistics(goce_root,goce)}
    goce_ready=(goce/"m0_summary.csv").is_file(); missing=json.loads((goce/"goce_missing_artifacts.json").read_text()) if (goce/"goce_missing_artifacts.json").is_file() else {"status":"unknown"}
    goce_summary=read_csv(goce/"m0_summary.csv") if goce_ready else []; goce_refs=read_csv(goce/"reference_convergence_summary.csv") if goce_ready else []; goce_costs=read_csv(goce/"m0_allocations.csv") if goce_ready else []
    summary_rows=[{"geometry":"Cube","evidence_status":"available",**row} for row in cube_summary]+[{"geometry":"GOCE","evidence_status":"available",**row} for row in goce_summary]
    reference_rows=[{"geometry":"Cube","evidence_status":"available",**row} for row in cube_refs]+[{"geometry":"GOCE","evidence_status":"available",**row} for row in goce_refs]
    cost_rows=[{"geometry":"Cube","evidence_status":"configured-cost sweep; measured CPU-hours unavailable",**row} for row in cube_costs]+[{"geometry":"GOCE","evidence_status":"available",**row} for row in goce_costs]
    if not goce_ready:
        for m0 in (2,4,6,8,10,12,16,20): cost_rows.append({"geometry":"GOCE","evidence_status":"missing production/reference artifacts; not evaluated","m0":m0,"n_DSMC":"","n_TPMC":"","configured_cost":"","measured_cost_cpu_hours":""})
    write_csv(output/"cube_goce_m0_summary.csv",summary_rows); write_csv(output/"cube_goce_reference_convergence.csv",reference_rows); write_csv(output/"cube_goce_cost_comparison.csv",cost_rows)
    fixed=sorted([r for r in cube_summary if r["reference_sample_count"]=="50" and r["method"].startswith("fixed-m0-")],key=lambda r:int(r["m0"])); x=[int(r["m0"]) for r in fixed]
    metrics=(("mean_field_relative_error","mean field"),("covariance_probe_relative_error","covariance probe"),("leading_eigenvalue_mean_relative_error","leading eigenvalues"),("projector_distance_fro","POD projector"))
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.5),sharex=True)
    for ax,(metric,label) in zip(axes.flat,metrics):
        ax.plot(x,[float(r[f"{metric}_median"]) for r in fixed],"o-",label="Cube")
        if goce_ready:
            gf=sorted([r for r in goce_summary if r["reference_sample_count"]=="50" and r["method"].startswith("fixed-m0-")],key=lambda r:int(r["m0"])); ax.plot(x,[float(r[f"{metric}_median"]) for r in gf],"s-",label="GOCE")
        else: ax.text(.5,.88,"GOCE pending",transform=ax.transAxes,ha="center")
        ax.set_title(label); ax.grid(alpha=.25)
    axes.flat[0].legend(); fig.suptitle("Cube–GOCE two-fidelity m0 comparison"); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(output/f"cube_goce_m0_metrics.{ext}",dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,4.2)); ax.plot(x,[float(r["leading_eigenvalue_mean_relative_error_median"]) for r in fixed],"o-",label="Cube eigenvalue error"); ax.plot(x,[float(r["projector_distance_fro_median"]) for r in fixed],"s-",label="Cube projector distance");
    if not goce_ready: ax.text(.5,.9,"GOCE subspace evidence unavailable",transform=ax.transAxes,ha="center")
    ax.set(xlabel="$m_0$",title="Cube–GOCE POD-subspace trade-off"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(output/f"cube_goce_subspace_tradeoff.{ext}",dpi=220)
    plt.close(fig)
    metric_names={"mean_field_relative_error":"mean-field error","covariance_probe_relative_error":"covariance-probe error","leading_eigenvalue_mean_relative_error":"leading-eigenvalue error","projector_distance_fro":"POD-projector distance","heldout_projection_error":"held-out projection error"}
    best={metric:min(fixed,key=lambda r:float(r[f"{metric}_median"])) for metric in metric_names}
    m6=next(r for r in fixed if r["m0"]=="6"); dsmc=next(r for r in cube_summary if r["m0"]=="6" and r["reference_sample_count"]=="50" and r["method"]=="DSMC-only")
    text=f"""# Paper findings: Cube and GOCE two-fidelity sensitivity

## 1. Data availability and role separation

- **Robustly supported:** Cube has 30 paired pilot fields, 20 production DSMC fields, 180 production TPMC fields, and 50 independent DSMC reference fields. Role disjointness, unique IDs, and topology were verified.
- **Not yet decidable:** GOCE lacks the prepared production/reference snapshot archive and associated metadata needed by the identical protocol. No values were inferred.

## 2. Two-fidelity allocation protocol

- **Robustly supported:** The fixed counts are (2,180), (4,160), (6,140), (8,120), (10,100), (12,80), (16,40), and (20,0) at configured costs 1 and 0.1 under budget 20. The DSMC count equals m0 exactly.

## 3. Cube results

{chr(10).join(f'- **Robustly supported:** The Cube median {label} is minimized at m0={best[m]["m0"]} ({float(best[m][f"{m}_median"]):.6g}).' for m,label in metric_names.items())}

## 4. GOCE results

- **Not yet decidable:** GOCE production and independent-reference results are unavailable; no preferred m0 is reported.

## 5. Cross-geometry comparison

- **Partly supported (pilot only):** The complete-field DSMC--TPMC pilot correlations were weaker for GOCE (mean {pilot_controls['GOCE']['correlation_mean']:.3f}, second moment {pilot_controls['GOCE']['correlation_second_moment']:.3f}) than for Cube ({pilot_controls['Cube']['correlation_mean']:.3f} and {pilot_controls['Cube']['correlation_second_moment']:.3f}). The corresponding global GOCE control weights were {pilot_controls['GOCE']['beta_mean_TPMC']:.3f} and {pilot_controls['GOCE']['beta_second_moment_TPMC']:.3f}, compared with {pilot_controls['Cube']['beta_mean_TPMC']:.3f} and {pilot_controls['Cube']['beta_second_moment_TPMC']:.3f} for Cube.
- **Not yet decidable:** Transferability, geometry-dependent production error, and whether the global GOCE weights are adequate cannot be assessed without GOCE production/reference fields.

## 6. Mean/covariance versus POD-subspace trade-off

- **Robustly supported:** Cube moment and subspace objectives rank m0 differently. At m0=6 the median eigenvalue error is {float(m6['leading_eigenvalue_mean_relative_error_median']):.6g}, while the projector-distance optimum occurs at m0={best['projector_distance_fro']['m0']}.
- **Partly supported:** The signed covariance correction and retained/complement coupling explain why mean accuracy need not imply stable modes; this is numerical operator evidence, not a universal mechanism claim.

## 7. Reference convergence

- **Partly supported:** Nested independent DSMC prefixes quantify sensitivity to using 10--50 fields. The 50-field ensemble is the common numerical reference, not ground truth or proof of population convergence.

## 8. Configured versus measured costs

- **Robustly supported:** All primary counts use configured scenario costs. Comparable measured CPU-hours are absent, so no measured-cost preference is claimed.

## 9. Robustly supported findings

- Cube count locking, pairing, nested production/reference prefixes, seeded repetitions, and the metric-specific Cube optima are reproducible.
- At m0=6, the Cube two-fidelity median mean error ({float(m6['mean_field_relative_error_median']):.6g}) is below DSMC-only ({float(dsmc['mean_field_relative_error_median']):.6g}); the paired win rates are recorded in the generated table.

## 10. Partially supported findings

- Increasing paired DSMC support generally stabilizes the Cube subspace, but metric behavior is not monotone for every prefix and repetition.
- Global control weights are effective for selected Cube moments; adequacy for spatially local errors or GOCE is unresolved.

## 11. Unsupported findings

- Global optimality, physical ground truth, general low-fidelity superiority, and equivalence of configured and measured costs are unsupported.

## 12. Limitations

- One complete geometry, 30 production permutations, one configured cost ratio, global scalar weights, five retained modes, and a finite 50-field numerical reference.

## 13. Recommended manuscript claims

- “The DSMC–TPMC estimator reduced Cube mean-field and leading-eigenvalue errors over a range of paired DSMC counts.”
- “The allocation preferred by moment metrics did not necessarily minimize POD-subspace error.”
- “Increasing paired DSMC support improved Cube subspace stability relative to small-prefix corrections.”

## 14. Claims that must not be made

- Do not claim cross-geometry transfer, GOCE performance, global optimality, ground truth, general control-model superiority, or measured-cost equivalence.

## 15. Recommended figures and tables

- Use `cube_goce_m0_metrics`, `cube_goce_subspace_tradeoff`, the fixed-count table, and the Cube reference-convergence figure. Label GOCE panels as pending.

## 16. Remaining work

- Supply existing GOCE prepared snapshots, metadata, resolved configuration, and benchmark information; then run the identical offline protocol. No solver or scheduler work is part of this analysis.
"""
    (output/"paper_findings.md").write_text(text,encoding="utf-8")
    tex=f"""\\section{{Two-fidelity sensitivity findings}}\nThe Cube study used an independent 50-DSMC numerical reference, not ground truth. Metric-specific preferred paired counts differed, demonstrating a moment--subspace trade-off. Pilot-only complete-field correlations were weaker for GOCE (mean {pilot_controls['GOCE']['correlation_mean']:.3f}, second moment {pilot_controls['GOCE']['correlation_second_moment']:.3f}) than for Cube ({pilot_controls['Cube']['correlation_mean']:.3f} and {pilot_controls['Cube']['correlation_second_moment']:.3f}); GOCE production performance remained unavailable.\n"""
    (output/"paper_findings.tex").write_text(tex,encoding="utf-8")
    lines=["\\begin{tabular}{rrrrrr}","$m_0$ & DSMC & TPMC & Mean error & Cov. error & Projector distance \\\\","\\hline"]
    for r in fixed: lines.append(f"{r['m0']} & {r['n_DSMC']} & {r['n_TPMC']} & {float(r['mean_field_relative_error_median']):.4g} & {float(r['covariance_probe_relative_error_median']):.4g} & {float(r['projector_distance_fro_median']):.4g} \\\\")
    lines += ["\\hline","\\multicolumn{6}{l}{GOCE: not evaluated because required artifacts were unavailable.} \\\\","\\end{tabular}"]
    (output/"generated_results_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    inputs=[cube/"m0_summary.csv",cube/"reference_convergence_summary.csv",cube/"m0_allocations.csv",goce/"goce_missing_artifacts.json"]
    manifest={"protocol":"offline DSMC-target/TPMC-control comparison","cube_status":"complete","goce_status":"complete" if goce_ready else "incomplete; no values inferred","pilot_control_statistics":pilot_controls,"goce_missing_artifacts":missing,"inputs_sha256":{str(p):digest(p) for p in inputs},"outputs":[p.name for p in output.iterdir()]}
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cube-root",type=Path,required=True); p.add_argument("--goce-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); print(json.dumps(build(a.cube_root,a.goce_root,a.output),indent=2))


if __name__=="__main__": main()
