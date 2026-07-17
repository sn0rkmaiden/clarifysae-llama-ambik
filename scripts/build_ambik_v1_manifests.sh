#!/usr/bin/env bash
set -euo pipefail

python -m clarifysae_llama.experiments.build_manifest \
  --config configs/experiments/ambik_explore_v1.yaml \
  --output manifests/generated/ambik_explore_v1.csv

echo "Explore manifest created. After the explore runs complete, freeze alphas:"
echo "python -m clarifysae_llama.experiments.select_alpha --manifest manifests/generated/ambik_explore_v1.csv"
echo "Then build confirm manifest:"
echo "python -m clarifysae_llama.experiments.build_manifest --config configs/experiments/ambik_confirm_v1.yaml --output manifests/generated/ambik_confirm_v1.csv"
