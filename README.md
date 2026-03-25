# clarifysae-llama-ambik

Clean repo for running **Llama-3.2-1B-Instruct** baseline and **pretrained SAE steering with `sparsify`** on **AmbiK**.

This repo is intentionally narrow:
- one model family: Hugging Face causal LMs
- one steering method: pretrained `sparsify` SAE
- one dataset target: AmbiK
- one evaluation path first: the `no_help` prompting setup
- evaluation supports both the old single JSON prompt and a separated 3-stage protocol (ambiguity label, question generation, JSON compliance)

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

The patched configs now default to the separated 3-stage evaluation protocol:
1. ambiguity classification
2. question generation (up to 3 questions)
3. JSON compliance


```bash
python -m clarifysae_llama.runners.run_eval --config configs/base_llama32_1b.yaml
```

## Steered run

Update `steering.feature_indices` in the config first.

```bash
python -m clarifysae_llama.runners.run_eval --config configs/steer_llama32_1b.yaml
```

## Strength sweep

Legacy one-parameter sweep:

```bash
python -m clarifysae_llama.runners.sweep --config configs/sweep_strength.yaml
```

## Automated single-feature steering sweeps

Use this mode when you want to try many features and many strengths but keep every actual run as a **single-feature intervention**.

### Config format

```yaml
experiment_name: llama32_1b_single_feature_strength_sweep
base_config: configs/steer_llama32_1b.yaml
sweep:
  mode: single_feature_strength
  strengths: [1.0, 3.0, 5.0, 10.0]
  groups:
    - vocab: C
      hookpoint: layers.10.mlp
      features: [11288, 93503]
    - vocab: Q
      hookpoint: layers.12.mlp
      features: [3557, 130523]
```

Each generated run overrides the base config with:
- one `steering.hookpoint`
- one `steering.feature_indices: [feature_id]`
- one `steering.strength`

`vocab` is metadata for tracking your C and Q selections. It is included in generated run names, the sweep manifest, and `outputs/logs/runs.jsonl`.
This means the same feature ID can appear under both vocab groups without collisions.

### Recommended workflow

1. Edit `configs/sweep_single_feature_strengths_chosen.yaml`.
2. Change only:
   - `sweep.strengths`
   - the `features` lists inside each `(vocab, hookpoint)` group
3. Run:

```bash
python -m clarifysae_llama.runners.sweep --config configs/sweep_single_feature_strengths_chosen.yaml
```

Or use the helper script:

```bash
./scripts/sweep_single_feature_strengths.sh
```

The helper script defaults to `configs/sweep_single_feature_strengths_chosen.yaml`, and you can pass another config path as its first argument.

### Generated run names

Run names are created automatically from:
- sweep name
- vocab
- hookpoint
- feature index
- strength

Example:

```text
llama32_1b_chosen_single_feature_strength_sweep__C__l10_mlp__feat11288__str5p0
```

This removes the need to manage one config per feature by hand.

## Outputs

Each run writes into `outputs/`:
- `predictions/<experiment_name>/predictions.jsonl`
- `<experiment_name>/metrics/example_metrics.csv`
- `<experiment_name>/tables/aggregate_metrics.csv`
- `logs/runs.jsonl`

Each automated sweep also writes into `outputs/sweeps/<sweep_name>/`:
- `generated_configs/<run_name>.yaml`
- `manifest.csv`
- `manifest.jsonl`
- `source_sweep_config.yaml`

The sweep manifest includes:
- `run_name`
- `vocab`
- `hookpoint`
- `feature_index`
- `strength`
- `config_path`
- `predictions_path`
- `example_metrics_path`
- `aggregate_metrics_path`

## Notes

- The steered config ships with a placeholder feature index. Replace it with a real one before using `configs/steer_llama32_1b.yaml` directly.
- `strength: 0.0` is **not** a true no-steering baseline in this repo, because the SAE reconstruction path is still used when steering is enabled. Use `configs/base_llama32_1b.yaml` for the real baseline.
- `steer_generated_tokens_only` is now applied in the hook itself, and `preserve_unsteered_residual` now preserves the original residual by adding only the steered SAE-space delta back to the hidden state.
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
python -m clarifysae_llama.runners.discover_features --config configs/discovery/discover_llama_vocab_layer12mlp.yaml
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
python -m clarifysae_llama.runners.inspect_scores --config configs/discovery/inspect_scores_layer10mlp_Cvocab.yaml
```

This writes:
- `outputs/discovery/inspection/merged_scores.csv`

and prints a merged table to the console.
