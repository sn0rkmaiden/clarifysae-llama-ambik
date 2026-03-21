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


## Feature discovery with C / Q vocabularies

The repo includes a discovery runner for computing generic vocab-based
feature scores.

Expected vocabulary format:
- JSON list of strings, for example `["?", " clarify", " which"]`
- or JSON dict from a label to one or more string variants

Example config:
- `configs/discovery/discover_llama32_1b_vocab_scores.yaml`

Run it with:

```bash
python -m clarifysae_llama.runners.discover_features --config configs/discovery/discover_llama32_1b_vocab_scores.yaml
```

Outputs are written under:
- `outputs/discovery/<experiment_name>/<vocab_name>/feature_scores.pt`
- `outputs/discovery/<experiment_name>/<vocab_name>/feature_scores.csv`
- `outputs/discovery/<experiment_name>/<vocab_name>/top_100_features.csv`

Notes:
- Replace the example vocab paths with your real C and Q vocab files.
- This runner computes a generic vocab score from positive-token activation minus background activation, with an entropy term across vocab groups.


## OutputScores

The repo includes an OutputScore runner.
It is intended to mirror the old two-stage workflow:
1. discover C / Q features from vocab scores
2. select top features from a score file
3. compute `a_max` on a text corpus
4. intervene on one feature at a time and score the output distribution

Example config:
- `configs/discovery/compute_output_scores_llama32_1b.yaml`

Run it with:

```bash
python -m clarifysae_llama.runners.compute_output_scores --config configs/discovery/output_scores_llama_Cvocab_10mlp.yaml
```

Outputs are written under:
- `outputs/discovery/<experiment_name>/output_scores/<name>/a_max.pt`
- `outputs/discovery/<experiment_name>/output_scores/<name>/output_scores.csv`
- `outputs/discovery/<experiment_name>/output_scores/<name>/output_scores.json`

Notes:
- `feature_scores_path` should point to a `.pt` file from vocab discovery.
- By default the runner reads the `scores` tensor from that file.
- You can either use `top_k_features` or provide explicit `feature_indices`.

## Inspecting / merging score tables

The repo includes a small inspection utility for merging C / Q / OutputScore tables.

Example config:
- `configs/discovery/inspect_scores.yaml`

Run it with:

```bash
python -m clarifysae_llama.runners.inspect_scores --config configs/discovery/inspect_scores.yaml
```

This writes:
- `outputs/discovery/inspection/merged_scores.csv`

and prints a merged table to the console.
