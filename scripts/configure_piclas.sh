#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
piclas_dir="${repo_root}/piclas"

piclas_source="${PICLAS_BIN:-$(command -v piclas || true)}"
piclas2vtk_source="${PICLAS2VTK_BIN:-$(command -v piclas2vtk || true)}"

if [[ -z "${piclas_source}" || ! -x "${piclas_source}" ]]; then
  echo "Set PICLAS_BIN to an executable PICLas binary." >&2
  exit 2
fi
if [[ -z "${piclas2vtk_source}" || ! -x "${piclas2vtk_source}" ]]; then
  echo "Set PICLAS2VTK_BIN to an executable piclas2vtk binary." >&2
  exit 2
fi

install -m 755 "${piclas_source}" "${piclas_dir}/piclas"
install -m 755 "${piclas2vtk_source}" "${piclas_dir}/piclas2vtk"
echo "PICLas runtime installed in ${piclas_dir}."

