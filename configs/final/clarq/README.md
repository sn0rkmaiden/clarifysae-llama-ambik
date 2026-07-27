# Reviewer-focused ClarQ evaluation

These eight configs evaluate the four configurations selected on ClarQ task
type 0 without further tuning.

`evaluation_set: 0-5` means:

- index 0: the original ten-dialogue selection/pilot task;
- indices 1-5: five additional task types used for the cross-task check.

Report index 0 separately. Treat the macro-average over indices 1-5 as the
new cross-task result.

## Frozen steering conditions

| Model | Layer | Feature | Strength |
|---|---:|---:|---:|
| Gemma-2B | 12 | 538 | -5 |
| Gemma-9B | 20 | 344 | 3 |
| Llama-1B | 12 | 6230 | -5 |
| Llama-8B | 27 | 62124 | 10 |

The files are generated from the exact model-specific ClarQ sweep bases:

```bash
PYTHONPATH=src python scripts/build_final_clarq_configs.py
```

The generator enforces identical model, decoding, prompting, provider, and
judge sections within every baseline/steered pair.

## Smoke test

Run one baseline first:

```bash
python -m clarifysae_llama.runners.run_clarq_eval \
  --config configs/final/clarq/gemma2b_baseline_eval0to5.yaml
```

The runner writes:

- `resolved_config.yaml` and `run_fingerprint.json`;
- `checkpoints/dialogues.jsonl` after every dialogue;
- dialogue-level metrics;
- micro summary;
- task-type summary with task names;
- macro summary across task types.

Rerunning the same config resumes saved dialogues. Reusing an experiment name
with a different resolved config fails instead of overwriting the run.

## Full run

```bash
for config in configs/final/clarq/*_eval0to5.yaml; do
  python -m clarifysae_llama.runners.run_clarq_eval --config "$config"
done
```

Run the four model pairs independently when separate GPUs are available. Do
not change features or strengths after inspecting task types 1-5.
