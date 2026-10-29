#!/usr/bin/env python3
"""Build the locked paired HF-64 bundle selected by the global 4D LF screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mfmc_campaign.parametric_geometry import (
    CylinderHexSpec,
    ParametricGeometryError,
    build_geometry_assets,
    build_gmsh_exterior_mesh,
    build_piclas_hdf5_mesh,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_hf_samples(samples: list[dict], n: int) -> list[dict]:
    """Keep all attitude anchors, then maximin-fill in normalized joint 4D space."""
    if n < sum(s["role"] == "attitude_anchor" for s in samples):
        raise ValueError("HF size is smaller than the number of mandatory attitude anchors")
    x = np.asarray(
        [[s["cll_alpha_n"], s["cll_sigma_t"], (s["aoa_deg"] + 10.0) / 20.0,
          (s["aos_deg"] + 10.0) / 20.0] for s in samples],
        dtype=float,
    )
    selected = [i for i, s in enumerate(samples) if s["role"] == "attitude_anchor"]
    distance = np.min(np.linalg.norm(x[:, None, :] - x[selected][None, :, :], axis=2), axis=1)
    distance[selected] = -1.0
    while len(selected) < n:
        idx = int(np.argmax(distance))
        selected.append(idx)
        distance = np.minimum(distance, np.linalg.norm(x - x[idx], axis=1))
        distance[selected] = -1.0
    return [dict(samples[i], hf_index=j, source_lf_index=i) for j, i in enumerate(selected)]


def variable(name: str, source: str) -> dict:
    low, high = (-10.0, 10.0) if name in {"aoa_deg", "aos_deg"} else (0.0, 1.0)
    return {
        "name": name,
        "source_block": source,
        "distribution": {"kind": "uniform", "params": {"low": low, "high": high}},
        "baseline": 0.0 if low < 0 else 0.5,
        "bounds": [low, high],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=Path("outputs/vleo_cll_6d_global_lf_analysis"))
    parser.add_argument("--global-bundle", type=Path, default=Path("outputs/vleo_cll_6d_global_search_bundle"))
    parser.add_argument("--anchor-root", type=Path, default=Path("outputs/vleo_cll_server_run_bundle/geometries"))
    parser.add_argument("--template", type=Path, default=Path("outputs/vleo_cll_mfmc_extension_bundle/configs/cylinder_hex_wp5_000_cll_extension10.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vleo_cll_6d_hf_pilot_bundle"))
    parser.add_argument("--n-hf", type=int, default=64)
    parser.add_argument("--gmsh", default="gmsh")
    parser.add_argument("--pyhope", default="pyhope")
    parser.add_argument("--defer-pyhope", action="store_true",
                        help="Build the portable bundle now and convert Gmsh meshes on the execution host")
    args = parser.parse_args()

    shortlist_path = args.analysis / "hf_shortlist.json"
    source_manifest = args.global_bundle / "manifests/joint_gsi_attitude_lf10000.json"
    shortlist = json.loads(shortlist_path.read_text())["selection"]
    all_samples = json.loads(source_manifest.read_text())["samples"]
    samples = select_hf_samples(all_samples, args.n_hf)
    template = yaml.safe_load(args.template.read_text())

    for name in ("geometries", "configs", "manifests"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    hf_manifest = args.output / "manifests/joint_gsi_attitude_hf64.json"
    hf_manifest.write_text(json.dumps({
        "schema_version": 1,
        "role": "paired_4d_hf_pilot",
        "selection_method": "all 13 attitude anchors plus deterministic maximin fill in normalized joint 4D space",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha(source_manifest),
        "n": len(samples),
        "angle_domain": {"shape": "disk", "radius_deg": 10.0, "core_radius_deg": 5.0},
        "samples": samples,
    }, indent=2) + "\n")

    built = []
    rejected = []
    for item in shortlist:
        gid = item["geometry_id"]
        new_geometry = (args.global_bundle / "geometries" / gid / f"{gid}.exterior.geo").is_file()
        src = (args.global_bundle / "geometries" / gid) if new_geometry else (args.anchor_root / gid)
        dst = args.output / "geometries" / gid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        manifest_path = dst / f"{gid}.manifest.json"
        if new_geometry:
            # Extreme global designs can contain axial features shorter than the
            # coarse 30-mm LF asset setting. A 10-mm body size avoids invalid
            # split-to-hex connectivity while preserving the exact CAD surface.
            source_geometry = json.loads(manifest_path.read_text())
            build_geometry_assets(
                dst,
                CylinderHexSpec(**source_geometry["specification"]),
                design_id=gid,
                uniform_scale_factor=0.1,
                body_mesh_size_m=0.01,
                farfield_mesh_size_m=0.05,
            )
        try:
            gmsh = build_gmsh_exterior_mesh(manifest_path, gmsh_executable=args.gmsh)
            h5 = ({"nElems": None, "valid": False, "status": "pending_server_pyhope_conversion"}
                  if args.defer_pyhope else
                  build_piclas_hdf5_mesh(manifest_path, pyhope_executable=args.pyhope))
        except ParametricGeometryError as exc:
            rejected.append({**item, "reason": f"PICLas mesh gate failed: {exc}"})
            shutil.rmtree(dst)
            continue

        cfg = json.loads(json.dumps(template))
        cfg["study"]["id"] = f"vleo_cll_6d_hf64_{gid}"
        cfg["study"]["active_source_blocks"] = ["gsi.cll", "attitude.aoa", "attitude.aos"]
        cfg["sources"]["blocks"] = [{"name": "gsi.cll"}, {"name": "attitude.aoa"}, {"name": "attitude.aos"}]
        cfg["variables"] = [
            variable("cll_alpha_n", "gsi.cll"),
            variable("cll_sigma_t", "gsi.cll"),
            variable("aoa_deg", "attitude.aoa"),
            variable("aos_deg", "attitude.aos"),
        ]
        cfg["geometries"][0]["id"] = gid
        cfg["geometries"][0]["name"] = gid
        cfg["geometries"][0]["metadata"]["hf_mesh"] = f"../{args.output.as_posix()}/geometries/{gid}/{gid}_mesh.h5"
        cfg["geometries"][0]["metadata"]["lf_model"] = gid
        cfg["regimes"][0]["id"] = "AO_1E15_JOINT_GSI_ATTITUDE"
        cfg["regimes"][0]["label"] = "Fixed environment; joint CLL and AoA/AoS uncertainty inside 10 deg disk"
        explicit = [{k: s[k] for k in ("cll_alpha_n", "cll_sigma_t", "aoa_deg", "aos_deg")} for s in samples]
        cfg["sampling"] = {
            "method": "explicit", "sample_count": args.n_hf, "max_production_samples": args.n_hf,
            "source_contract": f"locked joint_gsi_attitude_hf64.json SHA-256 {sha(hf_manifest)}",
            "explicit_samples": explicit,
        }
        kwargs = cfg["models"]["hf"]["kwargs"]
        kwargs.update({"mpi_procs": 36, "submission_group_size": 10, "simulator_module": "PICLas_prandtl"})
        cfg["pilot"].update({"size": args.n_hf, "sizes": [args.n_hf]})
        cfg["outputs"]["dir"] = f"campaign_outputs/vleo_cll/global_6d_hf64/{gid}"
        cfg["seeds"] = {"global": 20261104}
        config_path = args.output / "configs" / f"{gid}_joint4d_hf64.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        built.append({
            **item, "config": str(config_path.relative_to(args.output)),
            "n_elements": h5["nElems"], "gmsh_valid": gmsh["valid"], "h5_valid": h5["valid"],
            "mesh_status": h5.get("status", "piclas_hdf5_mesh_validated"),
        })

    gids = " ".join(x["geometry_id"] for x in built)
    last = len(built) - 1
    (args.output / "run_hf_array.slurm").write_text(f'''#!/bin/bash
#SBATCH --job-name=vleo-6d-hf64
#SBATCH --array=0-{last}%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=72:00:00
#SBATCH --output={args.output.as_posix()}/logs/hf_%A_%a.out
#SBATCH --error={args.output.as_posix()}/logs/hf_%A_%a.err
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source .venv/bin/activate
mkdir -p {args.output.as_posix()}/logs
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
geometries=({gids})
gid="${{geometries[$SLURM_ARRAY_TASK_ID]}}"
python -m mfmc_campaign.cli run-pilot-correlation "{args.output.as_posix()}/configs/${{gid}}_joint4d_hf64.yaml"
''')
    bundle_manifest = {
        "schema_version": 1, "status": ("pending_server_pyhope_conversion" if args.defer_pyhope else "ready_for_hf_execution"), "exploratory": True,
        "holdout_used": False, "n_hf": args.n_hf, "paired_manifest_sha256": sha(hf_manifest),
        "source_shortlist_sha256": sha(shortlist_path), "execution_contract": {
            "mpi_procs": 36, "submission_group_size": 10, "simulator_module": "PICLas_prandtl",
            "geometry_concurrency": 1, "slurm_memory_directive": "omitted",
        }, "geometries": built, "mesh_gate_rejections": rejected,
    }
    (args.output / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n")
    (args.output / "finalize_meshes_on_server.sh").write_text(f'''#!/bin/bash
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"
source .venv/bin/activate
PYHOPE_EXECUTABLE="${{PYHOPE_EXECUTABLE:-pyhope}}"
BUNDLE="{args.output.as_posix()}"
for manifest in "$BUNDLE"/geometries/*/*.manifest.json; do
  python -m mfmc_campaign.cli convert-cylinder-hex-pyhope \
    --manifest "$manifest" --pyhope "$PYHOPE_EXECUTABLE"
done
for config in "$BUNDLE"/configs/*.yaml; do
  python -m mfmc_campaign.cli validate-config "$config" >/dev/null
done
python - "$BUNDLE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); path=root/'bundle_manifest.json'; payload=json.loads(path.read_text())
for item in payload['geometries']:
    gid=item['geometry_id']; manifest=json.loads((root/'geometries'/gid/f'{{gid}}.manifest.json').read_text())
    validation=manifest['pyhope_conversion']['validation']; item['n_elements']=validation['nElems']
    item['h5_valid']=validation['valid']; item['mesh_status']='piclas_hdf5_mesh_validated'
if not payload['geometries'] or not all(x['gmsh_valid'] and x['h5_valid'] for x in payload['geometries']):
    raise SystemExit('mesh validation incomplete')
payload['status']='ready_for_hf_execution'; path.write_text(json.dumps(payload,indent=2)+'\\n')
files=[p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
(root/'SHA256SUMS').write_text('\\n'.join(f"{{hashlib.sha256(p.read_bytes()).hexdigest()}}  {{p.relative_to(root)}}" for p in sorted(files))+'\\n')
PY
echo "HF bundle finalized and ready for sbatch."
''')
    (args.output / "README.md").write_text(f'''# Global 6-D geometry VLEO CLL — paired HF-64 pilot

This exploratory bundle contains {len(built)} mesh-qualified LF-screened geometries and one common, locked 64-point manifest over CLL, AoA, and AoS. All 13 attitude anchors are retained; the remaining points are a deterministic maximin subset of the completed LF-10,000 design. The holdout remains unopened. Candidates rejected by the independent PICLas mesh gate are documented in `bundle_manifest.json` and are not submitted.

The SLURM array is throttled to one geometry (`0-{last}%1`). Each HF simulation uses 36 MPI processes, submissions are grouped in batches of 10, and the simulator module is `PICLas_prandtl`. No SLURM memory directive is set on the launcher, and inherited SLURM memory variables are removed before nested submissions.

The LF screen is a selection device, not confirmatory evidence. Interpret HF correlations and drag statistics jointly before defining new local centers.

## Method resource

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
''')
    files = [p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.relative_to(args.output)}" for p in sorted(files)) + "\n")
    print(args.output / "bundle_manifest.json")


if __name__ == "__main__":
    main()
