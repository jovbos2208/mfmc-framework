from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def export_surface_modes(output_dir: Path, modes: np.ndarray, eigenvalues: np.ndarray, *, face_area: np.ndarray, A_ref: float, face_center: np.ndarray, face_normal: np.ndarray, method: str, case: str, centering_mode: str, coordinate_frame: str, budget: float | None = None) -> list[Path]:
    output_dir.mkdir(parents=True,exist_ok=True); n_faces=len(face_area); centers=np.asarray(face_center); normals=np.asarray(face_normal); centers=centers if centers.shape==(n_faces,3) else np.full((n_faces,3),np.nan); normals=normals if normals.shape==(n_faces,3) else np.full((n_faces,3),np.nan); paths=[]
    for k in range(modes.shape[1]):
        weighted=modes[:,k].reshape(n_faces,3); traction=weighted/np.sqrt(np.asarray(face_area)[:,None]/A_ref); rows=[]
        for j in range(n_faces): rows.append({"face_index":j,"center_x":centers[j,0],"center_y":centers[j,1],"center_z":centers[j,2],"normal_x":normals[j,0],"normal_y":normals[j,1],"normal_z":normals[j,2],"mode_tx":traction[j,0],"mode_ty":traction[j,1],"mode_tz":traction[j,2],"magnitude":np.linalg.norm(traction[j]),"mode_index":k+1,"eigenvalue":eigenvalues[k],"method":method,"case":case,"centering_mode":centering_mode,"coordinate_frame":coordinate_frame,"budget_hf_equivalent":budget})
        slug=method.replace(" ","_").replace("/","_"); path=output_dir/f"{slug}_budget_{budget}_mode_{k+1}.csv"
        with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        np.savez_compressed(path.with_suffix(".npz"),face_center=centers,face_normal=normals,mode_traction=traction,magnitude=np.linalg.norm(traction,axis=1),mode_index=k+1,eigenvalue=eigenvalues[k],method=method,case=case,centering_mode=centering_mode,coordinate_frame=coordinate_frame)
        paths.append(path)
    return paths


def _read_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(encoding="utf-8") as f: return list(csv.DictReader(f))


def generate_report_figures(root: Path, case: str, centering_mode: str, coordinate_frame: str) -> list[str]:
    """Generate reproducible diagnostics supported by the available archive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=root/"figures"; figures.mkdir(parents=True,exist_ok=True); generated=[]
    benchmark=_read_csv(root/"benchmark"/"benchmark_repetitions.csv"); spectra=_read_csv(root/"benchmark"/"eigenvalue_spectra.csv"); corrections=_read_csv(root/"benchmark"/"eigenvalue_corrections.csv"); allocation=_read_csv(root/"allocation"/"allocation_sweep.csv")
    def save(name,title,xlabel,ylabel):
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); path=figures/name; plt.savefig(path,dpi=180); plt.close(); generated.append(str(path))
    if benchmark:
        methods=sorted({r["method"] for r in benchmark}); max_budget=max(float(r["budget_hf_equivalent"]) for r in benchmark); max_r=max(int(r["r"]) for r in benchmark)
        for metric,name,ylabel in (("captured_energy","captured_energy_vs_dimension.png","held-out DSMC captured energy"),("projection_error","projection_error_vs_dimension.png","held-out DSMC projection error")):
            plt.figure()
            for method in methods:
                xs=sorted({int(r["r"]) for r in benchmark if r["method"]==method and float(r["budget_hf_equivalent"])==max_budget}); ys=[np.median([float(r[metric]) for r in benchmark if r["method"]==method and float(r["budget_hf_equivalent"])==max_budget and int(r["r"])==x]) for x in xs]; plt.plot(xs,ys,marker="o",label=method)
            plt.legend(); save(name,f"{case}: {metric.replace('_',' ')}", "reduced dimension",ylabel)
        for metric,name,ylabel in (("captured_energy","captured_energy_vs_budget.png","captured energy"),("projection_error","projection_error_vs_budget.png","projection error"),("maximum_principal_angle_rad","principal_angle_vs_budget.png","maximum principal angle [rad]"),("projector_distance_fro","projector_distance_vs_budget.png","projector distance")):
            plt.figure()
            for method in methods:
                xs=sorted({float(r["budget_hf_equivalent"]) for r in benchmark if r["method"]==method and int(r["r"])==max_r}); ys=[np.median([float(r[metric]) for r in benchmark if r["method"]==method and int(r["r"])==max_r and float(r["budget_hf_equivalent"])==x]) for x in xs]; plt.plot(xs,ys,marker="o",label=method)
            plt.legend(); save(name,f"{case}: {metric.replace('_',' ')}", "HF-equivalent production budget",ylabel)
        plt.figure()
        for method in methods:
            subset=[r for r in benchmark if r["method"]==method and int(r["r"])==max_r]; xs=[float(r["production_cost"]) for r in subset]; ys=[float(r["total_cost"]) for r in subset]; plt.scatter(xs,ys,label=method,alpha=.6)
        plt.legend(); save("production_vs_total_cost.png",f"{case}: production and pilot-inclusive cost","production cost","total cost including pilot")
    if spectra:
        plt.figure()
        max_budget=max(float(r["budget_hf_equivalent"]) for r in spectra)
        for method in sorted({r["method"] for r in spectra}):
            rows=[r for r in spectra if r["method"]==method and float(r["budget_hf_equivalent"])==max_budget]; plt.semilogy([int(r["mode"]) for r in rows],[max(float(r["eigenvalue"]),1e-18) for r in rows],marker="o",label=method)
        plt.legend(); save("eigenvalue_spectra.png",f"{case}: POD spectra","mode","eigenvalue")
    if corrections:
        plt.figure(); xs=sorted({float(r["budget_hf_equivalent"]) for r in corrections}); ys=[np.mean([float(r["corrected_fraction"]) for r in corrections if float(r["budget_hf_equivalent"])==x]) for x in xs]; plt.plot(xs,ys,marker="o"); save("eigenvalue_correction_fraction.png",f"{case}: published HF-MC correction frequency","HF-equivalent budget","corrected fraction")
    if allocation:
        plt.figure(); xs=sorted({float(r["fraction"]) for r in allocation}); data=[[float(r["heldout_hf_projection_error"]) for r in allocation if float(r["fraction"])==x] for x in xs]; plt.boxplot(data,positions=xs,widths=.04); save("allocation_metric_vs_hf_fraction.png",f"{case}: empirical allocation pilot","HF budget fraction","held-out HF projection error")
    prepared=root/"snapshots"/"prepared_snapshots.npz"
    pilot_stats=root/"pilot"/"pilot_statistics.json"
    if prepared.exists():
        with np.load(prepared,allow_pickle=False) as z:
            p=z["pilot_indices"]; x=np.sum(z["hf"][p]**2,axis=1); y=np.sum(z["lf"][p]**2,axis=1)
        plt.figure(); plt.scatter(y,x); save("hf_lf_squared_energy_scatter.png",f"{case}: paired pilot field energy","TPMC squared norm","DSMC squared norm")
    if pilot_stats.exists():
        import json
        values=json.loads(pilot_stats.read_text())["bootstrap_values"]
        if values: plt.figure(); plt.hist(values,bins=min(30,max(5,len(values)//10))); save("alpha_bootstrap_distribution.png",f"{case}: global alpha bootstrap","alpha","count")
    caption=f"All figures: {case}; dimensionless full traction normalized by q_inf and sqrt(A_j/A_ref); {centering_mode}; {coordinate_frame}; five smoke repetitions where applicable; costs distinguish production and pilot-inclusive totals; reference curves use a disjoint archived DSMC numerical reference, not physical truth.\n"
    (figures/"captions.md").write_text(caption,encoding="utf-8")
    return generated
