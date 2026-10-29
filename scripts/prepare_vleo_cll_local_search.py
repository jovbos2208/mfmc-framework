#!/usr/bin/env python3
"""Create a locked one-factor local geometry search around Search-50 winners."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.parametric_geometry import CylinderHexSpec, build_geometry_assets


CENTRES = {
    "mean": "cylinder_hex_wp5_008",
    "robustness": "cylinder_hex_wp5_007",
    "combination": "cylinder_hex_wp5_003",
}
PARAMETERS = ("nose_length_fraction", "tail_length_fraction", "front_width_height_ratio",
              "rear_width_height_ratio", "nose_end_scale", "tail_end_scale")
# Five per cent of each predeclared admissible interval.
STEPS = {"nose_length_fraction": .0175, "tail_length_fraction": .0175,
         "front_width_height_ratio": .075, "rear_width_height_ratio": .075,
         "nose_end_scale": .020, "tail_end_scale": .030}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--geometry-root",type=Path,default=Path("outputs/vleo_cll_server_run_bundle/geometries"))
    p.add_argument("--output",type=Path,default=Path("outputs/vleo_cll/local_search_round1"))
    p.add_argument("--install-adbsat",type=Path)
    a=p.parse_args(); (a.output/"geometries").mkdir(parents=True,exist_ok=True)
    rows=[]; seen=set(); index=0
    for branch, centre_id in CENTRES.items():
        source=json.loads((a.geometry_root/centre_id/f"{centre_id}.manifest.json").read_text())
        centre=CylinderHexSpec(**source["specification"])
        for parameter in PARAMETERS:
            for direction in (-1,1):
                value=getattr(centre,parameter)+direction*STEPS[parameter]
                spec=replace(centre,**{parameter:value}); spec.validate()
                signature=tuple(asdict(spec).items())
                if signature in seen: continue
                seen.add(signature); gid=f"cylinder_hex_local_r1_{index:03d}"; index+=1
                manifest=build_geometry_assets(a.output/"geometries"/gid,spec,design_id=gid,
                    uniform_scale_factor=.1,body_mesh_size_m=.03,farfield_mesh_size_m=.15)
                rows.append({"geometry_id":gid,"branch":branch,"centre_geometry_id":centre_id,
                    "perturbed_parameter":parameter,"direction":"minus" if direction<0 else "plus",
                    "step":STEPS[parameter],"parameters":{k:getattr(spec,k) for k in PARAMETERS},
                    "design_fingerprint":manifest["design_fingerprint"],"surface_valid":manifest["validation"]["valid"],
                    "reference_area_m2":manifest["reference_area"]["value_m2"]})
                if a.install_adbsat:
                    for sub,suffix in (("obj_files","obj"),("models","mat")):
                        target=a.install_adbsat/"inou"/sub; target.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(a.output/"geometries"/gid/f"{gid}.{suffix}",target/f"{gid}.{suffix}")
    payload={"schema_version":1,"role":"exploratory_local_search","holdout_used":False,
        "centres":CENTRES,"radius_policy":"one-at-a-time +/- 5% of each admissible parameter interval",
        "steps":STEPS,"n_candidates":len(rows),"candidates":rows}
    design=a.output/"local_search_design.json"; design.write_text(json.dumps(payload,indent=2)+"\n")
    lock={"schema_version":1,"design_manifest":str(design),"design_manifest_sha256":digest(design),
          "n_candidates":len(rows),"holdout_opened":False}
    (a.output/"LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
    print(design)


if __name__=="__main__": main()
