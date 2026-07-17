#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/ambik_explore_v1.yaml}"
PART_DIR="${2:-artifacts/feature_scales/ambik_v1_parts}"
FINAL_OUTPUT="${3:-artifacts/feature_scales/ambik_v1_feature_scales.csv}"
mkdir -p "$PART_DIR"

# The part files are safe to run as independent server jobs. This loop runs
# them serially; submit the same commands separately when parallelizing.
while read -r model layer; do
  python -m clarifysae_llama.experiments.collect_feature_scales \
    --experiment-config "$CONFIG" \
    --model-key "$model" \
    --layer "$layer" \
    --output "$PART_DIR/${model}_l${layer}.csv" \
    --quantile 0.95
done <<'EOF'
gemma2b 12
gemma9b 20
gemma9b 31
llama1b 10
llama1b 11
llama1b 12
llama1b 13
llama8b 23
llama8b 27
EOF

python -m clarifysae_llama.experiments.merge_feature_scales \
  --inputs "$PART_DIR/*.csv" \
  --output "$FINAL_OUTPUT"
