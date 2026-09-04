#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfmc_campaign.parametric_geometry import (
    CylinderHexSpec,
    build_geometry_assets,
    build_gmsh_exterior_mesh,
    build_piclas_hdf5_mesh,
    validate_piclas_hdf5_mesh,
)


LEVELS = {
    "L1": {"body_mesh_size_m": 0.030, "farfield_mesh_size_m": 0.150},
    "L2": {"body_mesh_size_m": 0.015, "farfield_mesh_size_m": 0.075},
}


def _five_seed_config(base: dict, *, level: str, geometry_id: str, mesh_reference: str) -> dict:
    config = deepcopy(base)
    request = config["request"]
    request["study_id"] = "vleo_cylinder_hex_mesh_convergence"
    request["cell_id"] = f"scale_0p1_dsmc_{level.lower()}_5seeds"
    request["geometry"]["id"] = geometry_id
    request["geometry"]["name"] = geometry_id
    request["geometry"]["metadata"]["hf_mesh"] = mesh_reference
    request["sample_ids"] = [f"{level.lower()}-seed-{index:03d}" for index in range(5)]
    request["samples"] = [
        {
            "database_index": 0,
            "aos_deg": 0.0,
            "aoa_deg": 0.0,
            "random_seed": 20260830 + index,
        }
        for index in range(5)
    ]
    request["metadata"]["case_name"] = f"cylinder_hex_scale_0p1_{level.lower()}_dsmc"
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build L1/L2 Cylinder-Hex PICLas meshes and run configs.")
    parser.add_argument(
        "--output-root",
        default="piclas/geometry/cylinder_hex_convergence",
        help="Versioned mesh root, normally below piclas/geometry",
    )
    parser.add_argument(
        "--base-config",
        default="configs/studies/cylinder_hex_piclas_adapter_smoke.json",
    )
    parser.add_argument("--config-output-dir", default="configs/studies")
    parser.add_argument("--gmsh", default="gmsh")
    parser.add_argument("--pyhope", default="pyhope")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    repository_root = Path.cwd().resolve()
    piclas_root = Path("piclas").resolve()
    config_output = Path(args.config_output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_output.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    spec = CylinderHexSpec().scaled(0.1)
    baseline_manifest_path = piclas_root / "geometry/cylinder_hex_scale_0p1.manifest.json"
    baseline_mesh_path = piclas_root / "geometry/cylinder_hex_scale_0p1_mesh.h5"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    baseline_h5 = validate_piclas_hdf5_mesh(baseline_mesh_path)
    baseline_gmsh = baseline_manifest["gmsh_exterior_mesh"]["validation"]
    baseline_workflow = _five_seed_config(
        base_config,
        level="L0",
        geometry_id="cylinder_hex_scale_0p1",
        mesh_reference="geometry/cylinder_hex_scale_0p1_mesh.h5",
    )
    baseline_workflow_path = config_output / "cylinder_hex_piclas_adapter_l0_5seeds.json"
    baseline_workflow_path.write_text(json.dumps(baseline_workflow, indent=2) + "\n", encoding="utf-8")
    summaries = [
        {
            "level": "L0",
            "body_mesh_size_m": 0.060,
            "farfield_mesh_size_m": 0.300,
            "geometry_id": "cylinder_hex_scale_0p1",
            "mesh_path": str(baseline_mesh_path.relative_to(repository_root)),
            "mesh_reference": "geometry/cylinder_hex_scale_0p1_mesh.h5",
            "workflow_config": str(baseline_workflow_path.relative_to(repository_root)),
            "n_tetrahedra": baseline_gmsh["n_tetrahedra"],
            "n_hexahedra": baseline_h5["nElems"],
            "gas_volume_m3": baseline_gmsh["gas_volume_m3"],
            "characteristic_cell_size_m": (
                baseline_gmsh["gas_volume_m3"] / baseline_h5["nElems"]
            ) ** (1.0 / 3.0),
            "gmsh_fingerprint": baseline_gmsh["mesh_fingerprint"],
            "hdf5_fingerprint": baseline_h5["mesh_fingerprint"],
            "scaled_jacobian_histogram": {"<0.0": 0, "0.0-0.1": baseline_h5["nElems"]},
        }
    ]

    for level, controls in LEVELS.items():
        geometry_id = f"cylinder_hex_scale_0p1_{level.lower()}"
        level_dir = output_root / level
        manifest = build_geometry_assets(
            level_dir,
            spec,
            design_id=geometry_id,
            uniform_scale_factor=0.1,
            **controls,
        )
        gmsh_summary = build_gmsh_exterior_mesh(manifest["manifest_json"], gmsh_executable=args.gmsh)
        h5_summary = build_piclas_hdf5_mesh(
            manifest["manifest_json"], pyhope_executable=args.pyhope
        )
        mesh_path = Path(h5_summary["mesh_path"])
        try:
            mesh_reference = str(mesh_path.relative_to(piclas_root))
        except ValueError as exc:
            raise ValueError("output-root must be located below the configured piclas directory") from exc
        workflow = _five_seed_config(
            base_config,
            level=level,
            geometry_id=geometry_id,
            mesh_reference=mesh_reference,
        )
        workflow_path = config_output / f"cylinder_hex_piclas_adapter_{level.lower()}_5seeds.json"
        workflow_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "level": level,
                **controls,
                "geometry_id": geometry_id,
                "mesh_path": str(mesh_path.relative_to(repository_root)),
                "mesh_reference": mesh_reference,
                "workflow_config": str(workflow_path.relative_to(repository_root)),
                "n_tetrahedra": gmsh_summary["n_tetrahedra"],
                "n_hexahedra": h5_summary["nElems"],
                "gas_volume_m3": gmsh_summary["gas_volume_m3"],
                "characteristic_cell_size_m": (
                    gmsh_summary["gas_volume_m3"] / h5_summary["nElems"]
                ) ** (1.0 / 3.0),
                "gmsh_fingerprint": gmsh_summary["mesh_fingerprint"],
                "hdf5_fingerprint": h5_summary["mesh_fingerprint"],
                "scaled_jacobian_histogram": h5_summary.get("scaled_jacobian_histogram", {}),
            }
        )

    suite = {
        "schema_version": 1,
        "geometry_family": "cylinder_hex_scale_0p1",
        "levels": summaries,
    }
    suite_path = output_root / "mesh_convergence_suite.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"suite_manifest": str(suite_path.relative_to(repository_root)), **suite},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
