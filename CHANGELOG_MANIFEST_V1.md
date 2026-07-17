# Manifest v1 implementation changes

## Added

- Deterministic, category-stratified AmbiK 400/600 split generation with nested
  smoke20 and pilot100 flags.
- A canonical 92-feature catalog with C/Q deduplication.
- Robust per-feature activation scale collection using the positive-activation
  95th percentile.
- Scale-aware steering for both decoder-vector and latent-additive backends.
- Compact human experiment configs and self-contained generated manifests.
- Manifest row/shard execution, resumable per-example generation, immutable run
  fingerprints, provenance, checksums, and status files.
- Explore-to-confirm alpha selection with paired-clear overasking constraint.
- Model, dataset, decoding, metric, explore, and confirm profiles.
- Slurm array and shell-script templates.
- Unit tests for splitting, paired loading, and manifest cardinality.

## Corrected

- The old `separated` AmbiK protocol label now fails instead of silently running
  the combined prompt.
- Semantic and NLI metric backend failures now raise an error instead of
  returning zero for every example.
- Python, NumPy, Torch, and CUDA seeds can be deterministic.
- Model revisions are accepted by tokenizer and model loading.
- SAE feature scales are fixed per feature rather than recomputed from each
  prompt.
- CLAM dataset loading can use the same split manifest as ClarifySAE.
- The ClarQ legacy invalid escape warning was removed.

## Deliberately retained

Older sweep configs and runners remain in place for reproducing historical
experiments. New AmbiK work should use the manifest workflow documented in
`docs/EXPERIMENTS_V1.md`.
