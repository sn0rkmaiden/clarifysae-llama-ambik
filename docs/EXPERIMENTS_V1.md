# AmbiK v1: manifest-driven evaluation

This is the primary workflow for the restarted AmbiK experiment. The older
`sweep.py` YAML files remain available for reproducing historical runs, but new
experiments should use this workflow.

## Design

- Full AmbiK: 1,000 paired scenarios.
- `explore400`: feature-strength selection and diagnostics.
- `confirm600`: untouched held-out evaluation.
- `smoke20` and `pilot100` are nested flags inside `explore400`.
- All ambiguous and paired-clear variants are evaluated together.
- Feature activation scale is fixed per feature using the 95th percentile of
  positive baseline SAE activations on `explore400`.
- Normalized alpha grid: `[-2, -1, -0.5, 0.5, 1, 2]`.
- C/Q duplicates are one intervention with `vocab_membership=C+Q` metadata.

## Repository structure

```text
configs/models/              four model profiles
configs/datasets/            dataset and split location
configs/decoding/            shared decoding profile
configs/metrics/             shared evaluation profile
configs/experiments/         explore and confirm experiment descriptions
manifests/feature_catalog.csv
manifests/generated/         self-contained execution manifests
splits/                      deterministic split manifests
artifacts/feature_scales/    robust per-feature activation scales
artifacts/selection/         frozen alpha chosen on explore400
```

There is no hand-written YAML per feature or per strength. A generated manifest
contains one self-contained JSON config per row.

## Required local data

Place the full paired AmbiK CSV at:

```text
data/raw/ambik/ambik_full_1000.csv
```

It must contain at least:

```text
id, environment_full, ambiguity_type, ambiguous_task,
unambiguous_direct, question, answer, plan_for_clear_task
```

Change `configs/datasets/ambik_v1.yaml` when using a different location. You can also set:

```bash
export CLARIFYSAE_AMBIK_PATH=/absolute/path/to/ambik_full_1000.csv
export CLARIFYSAE_RESULTS_ROOT=/absolute/path/to/server/results
```


## Step 1: create the split

```bash
./scripts/create_ambik_v1_split.sh
```

This produces `splits/ambik_v1.csv` and a metadata JSON with the source dataset
checksum and category counts.

## Step 2: collect feature scales

Run the nine model/layer tasks, either serially:

```bash
./scripts/collect_ambik_v1_scales.sh
```

or as separate server jobs using commands from that script. All tasks merge into:

```text
artifacts/feature_scales/ambik_v1_feature_scales.csv
```

A non-firing feature is marked and prevents manifest creation until reviewed.

## Step 3: build the explore manifest

```bash
python -m clarifysae_llama.experiments.build_manifest \
  --config configs/experiments/ambik_explore_v1.yaml \
  --output manifests/generated/ambik_explore_v1.csv
```

Expected rows: 552 steered conditions plus four baselines = 556.

## Step 4: run the manifest

One row locally:

```bash
./scripts/run_manifest_row.sh manifests/generated/ambik_explore_v1.csv 0
```

A Slurm template is in `slurm/ambik_manifest_array.sbatch`. Adjust resources and
array concurrency for the cluster.

Every run uses an immutable run ID and writes:

```text
resolved_config.yaml
run_fingerprint.json
command.txt
input_checksums.json
status.json
predictions/raw_outputs.jsonl
predictions/predictions.jsonl
predictions/predictions_full.jsonl
metrics/example_metrics.csv
tables/aggregate_metrics.csv
tables/category_metrics.csv
```

Raw generation is appended after every example, and rerunning the same row resumes
from missing IDs. A mismatched config cannot reuse the directory.

## Step 5: freeze one alpha per feature

After all explore rows complete:

```bash
python -m clarifysae_llama.experiments.select_alpha \
  --manifest manifests/generated/ambik_explore_v1.csv \
  --output artifacts/selection/ambik_explore_v1_selected_alpha.csv \
  --clear-tolerance 0.05
```

The implemented selection rule is:

1. maximize macro first-question proxy resolution;
2. require paired-clear overasking no more than baseline + 0.05;
3. tie-break by first-question similarity;
4. then fewer questions;
5. then smaller absolute alpha.

When no alpha meets the clear constraint, clear overasking excess is minimized
before the quality tie-breaks.

## Step 6: build and run confirm600

```bash
python -m clarifysae_llama.experiments.build_manifest \
  --config configs/experiments/ambik_confirm_v1.yaml \
  --output manifests/generated/ambik_confirm_v1.csv
```

Expected rows: 92 frozen feature conditions plus four baselines = 96.

## Reproducibility protections

- The old prompt-only cache is not used.
- Python, NumPy, Torch, and CUDA seeds are set.
- Deterministic decoding is the default profile.
- The semantic metric backend is loaded before generation and errors are fatal.
- The misleading `separated` AmbiK protocol now raises an error.
- Model revisions can be pinned in `configs/models/*.yaml`.
- Generation and scoring artifacts are separate, so metrics can be recomputed
  without rerunning the model.

## Preparing AmbiK from the available exports

The historical data is distributed as `ambik_calib_100.csv` and
`ambik_test_900.csv`. Together they contain 1,000 disjoint scenarios. The
`ambik_test_400.csv` export is a redundant historical copy of the beginning of
the 900-row export and must not be added as a third source.

Place the two source files under `data/raw/ambik/`, then run:

```bash
./scripts/create_ambik_v1_split.sh \
  data/raw/ambik/ambik_calib_100.csv \
  data/raw/ambik/ambik_test_900.csv
```

This creates:

- `data/processed/ambik/ambik_full_1000.csv`
- `data/processed/ambik/ambik_full_1000.metadata.json`
- `splits/ambik_v1.csv`
- `splits/ambik_v1.metadata.json`

The canonical example IDs are derived from content hashes rather than the
unstable dataframe-index columns in the historical CSV exports. The loader also
accepts `smoke20` and `pilot100` as split names in addition to `explore400` and
`confirm600`.

## Residual-relative steering scale

AmbiK prompt activations are retained as diagnostics, but they are not used as
the primary steering scale. Several shortlisted OutputScore features may never
fire naturally on an AmbiK prompt, so a positive-activation quantile can be
undefined or estimated from only a handful of observations.

The primary scale is now `relative_residual_l2`. For feature decoder direction
`d_j` at layer `l`, the collector computes

```text
scale_j = median_token_l2(h_l) / l2(d_j)
```

The effective residual-stream intervention is therefore approximately

```text
l2(delta_h) = abs(alpha) * median_token_l2(h_l)
```

so `alpha` is interpreted as a fraction of a typical residual-stream norm. This
scale exists for every valid feature, does not depend on whether the feature
fires naturally, and is comparable across features and model layers.

Collect all nine model-layer scale groups with:

```bash
./scripts/collect_ambik_v1_residual_scales.sh
```

Before running the full explore400 manifest, validate the provisional alpha grid
on `smoke20`. The existing positive-q95 collector remains available for domain
activation diagnostics; it now labels fewer than 20 positive observations as
`very_sparse` and fewer than 200 as `rare`.
