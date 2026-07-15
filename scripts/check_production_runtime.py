#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the bundled MFMC solver runtime")
    parser.add_argument("--allow-missing-piclas-binaries", action="store_true")
    args = parser.parse_args()

    required = [
        ROOT / "PICLas.py",
        ROOT / "PICLas_prandtl.py",
        ROOT / "ADBSat.py",
        ROOT / "ADBSat_prandtl.py",
        ROOT / "update_parameter_file" / "update_parameter.py",
        ROOT / "piclas" / "parameter.ini",
        ROOT / "piclas" / "DSMC1.ini",
        ROOT / "piclas" / "Cube_mesh.h5",
        ROOT / "piclas" / "GOCE_mesh.h5",
        ROOT / "piclas" / "CHAMP_mesh.h5",
        ROOT / "piclas" / "SOAR_mesh.h5",
        ROOT / "ADBSat-PyVersion" / "simulate.py",
        ROOT / "ADBSat-PyVersion" / "calc" / "calc_coeff.py",
        ROOT / "ADBSat-PyVersion" / "atmos_data" / "database_300km.csv",
        ROOT / "ADBSat-PyVersion" / "inou" / "models" / "Cube.mat",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]

    for module in ("mfmc_campaign.adapters", "PICLas", "PICLas_prandtl", "ADBSat", "ADBSat_prandtl"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"import {module}: {exc}")

    for name in ("piclas", "piclas2vtk"):
        binary = ROOT / "piclas" / name
        if not binary.is_file() or not os.access(binary, os.X_OK):
            if not args.allow_missing_piclas_binaries:
                missing.append(f"executable piclas/{name}")

    if missing:
        print("Production runtime is incomplete:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("MFMC, MFPOD, PICLas adapters, and ADBSat runtime are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
