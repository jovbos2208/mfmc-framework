#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
piclas_dir="${repo_root}/piclas"

piclas_source="${PICLAS_BIN:-$(command -v piclas || true)}"
piclas2vtk_source="${PICLAS2VTK_BIN:-$(command -v piclas2vtk || true)}"
piclas_lib_source="${PICLAS_LIB:-}"

if [[ -z "${piclas_source}" || ! -x "${piclas_source}" ]]; then
  echo "Set PICLAS_BIN to an executable PICLas binary." >&2
  exit 2
fi
if [[ -z "${piclas2vtk_source}" || ! -x "${piclas2vtk_source}" ]]; then
  echo "Set PICLAS2VTK_BIN to an executable piclas2vtk binary." >&2
  exit 2
fi

# PICLas links its executable against libpiclas.so. CMake normally places the
# executable in build/bin and the shared library in the sibling build/lib
# directory, with an $ORIGIN/../lib runpath. The campaign copies the executable
# into piclas/job_*/, so installing the library into piclas/lib preserves that
# relative layout for every generated job directory.
if [[ -z "${piclas_lib_source}" ]]; then
  candidate="$(dirname "$(dirname "${piclas_source}")")/lib/libpiclas.so"
  if [[ -f "${candidate}" ]]; then
    piclas_lib_source="${candidate}"
  fi
fi
if [[ -z "${piclas_lib_source}" || ! -f "${piclas_lib_source}" ]]; then
  echo "Set PICLAS_LIB to the libpiclas.so built with PICLAS_BIN." >&2
  exit 2
fi

install -m 755 "${piclas_source}" "${piclas_dir}/piclas"
install -m 755 "${piclas2vtk_source}" "${piclas_dir}/piclas2vtk"
install -d -m 755 "${piclas_dir}/lib"
install -m 755 "${piclas_lib_source}" "${piclas_dir}/lib/libpiclas.so"
echo "PICLas runtime and libpiclas.so installed in ${piclas_dir}."

