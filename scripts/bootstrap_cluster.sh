#!/usr/bin/env bash
set -euo pipefail

mode="${1:-production}"
python_bin="${PYTHON:-python3}"
venv_dir="${VENV:-.venv}"

case "${mode}" in
  core) extras="" ;;
  production) extras="production" ;;
  test) extras="dev,production" ;;
  *)
    echo "Usage: $0 [core|production|test]" >&2
    exit 2
    ;;
esac

"${python_bin}" -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip

if [[ -n "${extras}" ]]; then
  python -m pip install -e ".[${extras}]"
else
  python -m pip install -e .
fi

mfmc-campaign validate-config examples/toy_mock_campaign.yaml
python scripts/check_production_runtime.py --allow-missing-piclas-binaries
echo "Framework environment ready in ${venv_dir} (${mode})."
