#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: run_manifest_row.sh MANIFEST ROW_INDEX}"
ROW_INDEX="${2:?usage: run_manifest_row.sh MANIFEST ROW_INDEX}"

python -m clarifysae_llama.experiments.run_manifest \
  --manifest "$MANIFEST" \
  --row-index "$ROW_INDEX"
