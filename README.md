# ClarifySAE

Inference-time clarification steering with Sparse Autoencoders for Gemma and
Llama instruction-tuned models.

This repository now has two experiment paths:

1. **Manifest-driven AmbiK v1** — the primary path for new experiments.
2. **Legacy sweep runners** — retained to reproduce the earlier AmbiK-100 and
   ClarQ-LLM task-0 experiments.

The new path uses a small set of readable component configs and generated,
self-contained execution manifests. It does **not** create one hand-written YAML
per feature and strength.

## Current AmbiK design

- Full dataset: 1,000 paired ambiguous/clear scenarios.
- `explore400`: normalized strength sweep and feature selection.
- `confirm600`: untouched held-out evaluation.
- `smoke20` and `pilot100`: nested subsets of `explore400`.
- 92 unique shortlisted model-layer-feature interventions.
- Feature-specific activation scale: 95th percentile of positive SAE
  activations on `explore400`.
- Dimensionless normalized strengths: `[-2, -1, -0.5, 0.5, 1, 2]`.
- Ambiguous and paired-clear instructions are evaluated together.

Detailed instructions are in [docs/EXPERIMENTS_V1.md](docs/EXPERIMENTS_V1.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Authenticate with Hugging Face for gated model or SAE checkpoints.

The four backends require the packages listed in `requirements.txt`, including
`eai-sparsify`, `sae-lens`, `dictionary-learning`, and `bitsandbytes`.

## Required data

Place the full paired AmbiK CSV at:

```text
data/raw/ambik/ambik_full_1000.csv
```

or edit `configs/datasets/ambik_v1.yaml`.

The CSV must contain at least:

```text
id
environment_full
ambiguity_type
ambiguous_task
unambiguous_direct
question
answer
plan_for_clear_task
```

## Quick start: restarted AmbiK evaluation

### 1. Create the deterministic split

```bash
./scripts/create_ambik_v1_split.sh
```

### 2. Collect robust feature scales

```bash
./scripts/collect_ambik_v1_scales.sh
```

The nine model/layer commands can be submitted as separate server jobs. Use a
distinct part file for each job and merge the parts after completion, as shown
in `docs/EXPERIMENTS_V1.md`.

### 3. Build the exploration manifest

```bash
python -m clarifysae_llama.experiments.build_manifest \
  --config configs/experiments/ambik_explore_v1.yaml \
  --output manifests/generated/ambik_explore_v1.csv
```

This creates 552 steered rows plus four baselines. Each row contains a complete
resolved config as JSON.

### 4. Execute one manifest row

```bash
python -m clarifysae_llama.experiments.run_manifest \
  --manifest manifests/generated/ambik_explore_v1.csv \
  --row-index 0
```

A Slurm array template is available at `slurm/ambik_manifest_array.sbatch`.

### 5. Freeze one alpha per feature

```bash
python -m clarifysae_llama.experiments.select_alpha \
  --manifest manifests/generated/ambik_explore_v1.csv \
  --output artifacts/selection/ambik_explore_v1_selected_alpha.csv
```

### 6. Build confirm600

```bash
python -m clarifysae_llama.experiments.build_manifest \
  --config configs/experiments/ambik_confirm_v1.yaml \
  --output manifests/generated/ambik_confirm_v1.csv
```

The confirmatory manifest contains 92 frozen steering conditions and four
baselines.

## Repository structure

```text
configs/models/              model and runtime profiles
configs/datasets/            dataset paths and split manifest
configs/decoding/            shared decoding settings
configs/metrics/             shared metric settings
configs/experiments/         compact experiment descriptions
manifests/feature_catalog.csv
manifests/generated/         generated execution manifests
splits/                      deterministic dataset splits
artifacts/feature_scales/    per-feature activation statistics
artifacts/selection/         frozen explore-to-confirm selections
src/clarifysae_llama/experiments/
```

## Reproducibility protections

The new AmbiK runner:

- does not use the old prompt-only cache;
- saves and validates a complete run fingerprint;
- sets Python, NumPy, Torch, and CUDA seeds;
- defaults to deterministic decoding;
- loads semantic metric backends before generation and fails loudly;
- appends raw output after every example and resumes by ID;
- rejects a mismatched config in an existing result directory;
- saves resolved config, command, Git metadata, environment, checksums, raw
  generations, scored predictions, and aggregate tables;
- separates raw generation from scoring so metrics can be recomputed;
- rejects the old misleading `evaluation.protocol: separated` label. The
  currently supported AmbiK protocol is `combined_json`.

## Feature catalog

`manifests/feature_catalog.csv` is the canonical 92-feature inventory. A feature
selected by both C and Q is stored once with `vocab_membership=C+Q`.

Do not duplicate compute merely because a feature was found by both
vocabularies.

## ClarQ-LLM

The existing ClarQ runner and sweep files remain under `configs/clarq/`. They
reproduce the earlier task-type-0 experiments. Existing attached results cover
10 dialogues from one task type; they are not a full evaluation of the 26
English test task types.

The manifest framework is intentionally dataset-agnostic enough to be extended
to ClarQ after the restarted AmbiK pipeline is validated. That extension should
preserve the official five-development / 26-test task-type division rather than
randomly splitting ClarQ.

## Historical sweep files

The older `configs/steering/`, `configs/clarq/`, and
`clarifysae_llama.runners.sweep` code are kept for historical reproducibility.
Broken or version-mismatched top-level sweep files were moved to
`configs/legacy/unverified/`. They should not be used as the primary interface
for new AmbiK v1 experiments.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts
```
