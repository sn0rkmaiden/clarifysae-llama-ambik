#!/usr/bin/env bash
set -euo pipefail

CALIB100="${1:-data/raw/ambik/ambik_calib_100.csv}"
TEST900="${2:-data/raw/ambik/ambik_test_900.csv}"
FULL="${3:-data/processed/ambik/ambik_full_1000.csv}"
SPLIT="${4:-splits/ambik_v1.csv}"

python -m clarifysae_llama.experiments.prepare_ambik \
  --calib100 "$CALIB100" \
  --test900 "$TEST900" \
  --output "$FULL" \
  --expected-rows 1000

python -m clarifysae_llama.experiments.split_ambik \
  --dataset "$FULL" \
  --output "$SPLIT" \
  --explore-size 400 \
  --confirm-size 600 \
  --smoke-size 20 \
  --pilot-size 100 \
  --seed 20260717
