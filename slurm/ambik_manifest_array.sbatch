#!/usr/bin/env bash
#SBATCH --job-name=clarifysae-ambik
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-555%16
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
mkdir -p slurm/logs

MANIFEST="${MANIFEST:-manifests/generated/ambik_explore_v1.csv}"
python -m clarifysae_llama.experiments.run_manifest \
  --manifest "$MANIFEST" \
  --row-index "$SLURM_ARRAY_TASK_ID"
