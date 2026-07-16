#!/usr/bin/env bash
set -euo pipefail

python -m py_compile mfmc_campaign/*.py mfmc_campaign/field_mfpod/*.py
mfmc-campaign validate-config examples/toy_mock_campaign.yaml

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
python -m mfmc_campaign.field_mfpod synthetic --output "${tmp_dir}/mfpod"

echo "MFMC and MFPOD smoke tests passed."

