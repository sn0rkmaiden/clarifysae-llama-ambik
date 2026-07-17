#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
  echo "usage: $0 PATH_TO_LEGACY_SWEEP_CONFIG" >&2
  echo "For new experiments use docs/EXPERIMENTS_V1.md." >&2
  exit 2
fi
python -m clarifysae_llama.runners.sweep --config "$CONFIG"
