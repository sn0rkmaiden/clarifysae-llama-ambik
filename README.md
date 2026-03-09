# clarifysae-llama-ambik

Clean repo for running **Llama-3.2-1B-Instruct** baseline and **pretrained SAE steering with `sparsify`** on **AmbiK**.

This repo is intentionally narrow:
- one model family: Hugging Face causal LMs
- one steering method: pretrained `sparsify` SAE
- one dataset target: AmbiK
- one evaluation path first: the `no_help` prompting setup

## What is included

- AmbiK loader for the `ambik_test_400.csv` style split
- prompt builder matching your earlier `no_help` setup
- plain HF backend
- steered HF backend with a hook-based `sparsify` steerer
- metrics for clarification rate, correct clarification rate, and task success proxy
- JSONL prediction logging and CSV metric aggregation
- baseline and steering config templates

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For gated Meta checkpoints, authenticate with Hugging Face before running.

## Data preparation

Put the AmbiK split at:

```text
data/raw/ambik/ambik_test_400.csv
```

or point `dataset.path` in the config to another CSV.

## Baseline run

```bash
python -m clarifysae_llama.runners.run_eval --config configs/base_llama32_1b.yaml
```

## Steered run

Update `steering.feature_indices` in the config first.

```bash
python -m clarifysae_llama.runners.run_eval --config configs/steer_llama32_1b.yaml
```

## Strength sweep

```bash
python -m clarifysae_llama.runners.sweep --config configs/sweep_strength.yaml
```

## Outputs

Each run writes into `outputs/`:
- `predictions/<experiment_name>/predictions.jsonl`
- `metrics/<experiment_name>/example_metrics.csv`
- `tables/<experiment_name>/aggregate_metrics.csv`
- `logs/runs.jsonl`

## Notes

- The steered config ships with a placeholder feature index. Replace it with a real one before running.
- Hookpoint-to-module mapping is implemented for Llama-style architectures and can be extended later.
- The runner starts with the `no_help` setup because it is the cleanest path for Phase 1 replication.
