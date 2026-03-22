from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.backends.steered_hf_backend import SteeredHFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.data.ambik_loader import load_ambik_no_help_dataset
from clarifysae_llama.data.prompting import build_no_help_prompt, build_plan_prefix
from clarifysae_llama.eval.metrics import aggregate_metrics, compute_example_metrics
from clarifysae_llama.eval.reporting import save_metric_tables
from clarifysae_llama.utils.io import ensure_dir, write_jsonl
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.seed import set_seed



def _configure_console(config: dict[str, Any]) -> dict[str, Any]:
    console_cfg = config.get('console', {})
    suppress_tf_warnings = bool(console_cfg.get('suppress_transformers_warnings', True))
    show_progress = bool(console_cfg.get('show_progress', True))

    if suppress_tf_warnings:
        os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        hf_logging.set_verbosity_error()
        warnings.filterwarnings('ignore', message=r'.*`torch_dtype` is deprecated! Use `dtype` instead!.*')
        warnings.filterwarnings(
            'ignore',
            message=r'.*Both `max_new_tokens` .* and `max_length`.*',
        )
        warnings.filterwarnings(
            'ignore',
            message=r'.*The following generation flags are not valid and may be ignored:.*',
        )
    else:
        hf_logging.set_verbosity_warning()

    return {
        'show_progress': show_progress,
        'suppress_transformers_warnings': suppress_tf_warnings,
    }



def build_backend(config: dict):
    backend_name = config['model'].get('backend', 'hf')
    steering_enabled = config.get('steering', {}).get('enabled', False)

    if backend_name != 'hf':
        raise ValueError(f'Only hf backend is supported in this repo, got: {backend_name}')
    if steering_enabled:
        return SteeredHFCausalBackend(config)
    return HFCausalBackend(config)



def build_prompts(dataset: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in dataset.iterrows():
        plan_lines = str(row['plan']).split('\n')
        prefix, action = build_plan_prefix(plan_lines, int(row['end_of_ambiguity']))
        prompt = build_no_help_prompt(
            description=str(row['environment_full']),
            task=str(row['task']),
            prefix=prefix,
            action=action,
        )
        rows.append({
            'id': row['id'],
            'ambiguity_type': row['ambiguity_type'],
            'user_intent': row['user_intent'],
            'prompt': prompt,
        })
    return rows



def _print_run_header(config: dict[str, Any], n_examples: int, batch_size: int) -> None:
    experiment_name = config['experiment_name']
    dataset_path = config['dataset']['path']
    n_batches = math.ceil(n_examples / batch_size) if n_examples else 0

    print(f"\n=== run_eval :: {experiment_name} ===")
    print(f"dataset: {dataset_path}")
    print(f"eval examples: {n_examples} | batch_size: {batch_size} | batches: {n_batches}")

    steering_cfg = config.get('steering', {})
    if steering_cfg.get('enabled', False):
        print(
            'steering: '
            f"hookpoint={steering_cfg.get('hookpoint')} "
            f"features={steering_cfg.get('feature_indices')} "
            f"strength={steering_cfg.get('strength')}"
        )
    else:
        print('steering: disabled')



def run_eval(config: dict) -> dict[str, Any]:
    console_cfg = _configure_console(config)
    set_seed(int(config.get('seed', 42)))

    experiment_name = config['experiment_name']
    root_dir = Path(config['output']['root_dir'])
    pred_dir = ensure_dir(root_dir / 'predictions' / experiment_name)
    run_dir = ensure_dir(root_dir / experiment_name)
    ensure_dir(root_dir / 'logs')

    dataset = load_ambik_no_help_dataset(
        path=config['dataset']['path'],
        limit=config['dataset'].get('limit'),
    )
    prompt_rows = build_prompts(dataset)

    backend = build_backend(config)
    batch_size = int(config.get('batching', {}).get('batch_size', 1))
    _print_run_header(config, n_examples=len(prompt_rows), batch_size=batch_size)

    started_at = time.perf_counter()
    prediction_rows = []
    iterator = range(0, len(prompt_rows), batch_size)
    if console_cfg['show_progress']:
        iterator = tqdm(
            iterator,
            desc=f"{experiment_name} | generating",
            unit='batch',
            dynamic_ncols=True,
        )

    for start in iterator:
        chunk = prompt_rows[start:start + batch_size]
        prompts = [row['prompt'] for row in chunk]
        predictions = backend.generate_batch(prompts)
        for row, prediction in zip(chunk, predictions):
            metrics = compute_example_metrics(
                prediction_text=prediction,
                ambiguity_type=row['ambiguity_type'],
                user_intent=row['user_intent'],
            )
            prediction_rows.append({
                'id': row['id'],
                'user_intent': row['user_intent'],
                'prompt': row['prompt'],
                **metrics,
            })

    predictions_path = pred_dir / 'predictions.jsonl'
    write_jsonl(predictions_path, prediction_rows)

    raw_df = pd.DataFrame(prediction_rows)

    example_metrics = raw_df[
        ['id', 'y_amb_type', 'prediction_text', 'SR', 'help_rate', 'correct_help_rate']
    ].copy()

    agg_input = raw_df[
        ['y_amb_type', 'SR', 'help_rate', 'correct_help_rate']
    ].copy()

    agg = aggregate_metrics(agg_input)
    save_metric_tables(example_metrics, agg, run_dir)

    example_metrics_path = run_dir / 'metrics' / 'example_metrics.csv'
    aggregate_metrics_path = run_dir / 'tables' / 'aggregate_metrics.csv'

    elapsed_sec = time.perf_counter() - started_at
    print(f"completed: {experiment_name} in {elapsed_sec:.1f}s")
    print(f"  predictions: {predictions_path}")
    print(f"  example metrics: {example_metrics_path}")
    print(f"  aggregate metrics: {aggregate_metrics_path}")

    log_payload = {
        'experiment_name': experiment_name,
        'dataset_path': config['dataset']['path'],
        'n_examples': len(prompt_rows),
        'model_name': config['model']['name'],
        'steering_enabled': config.get('steering', {}).get('enabled', False),
        'sae_repo': config.get('steering', {}).get('sae_repo'),
        'hookpoint': config.get('steering', {}).get('hookpoint'),
        'feature_indices': config.get('steering', {}).get('feature_indices'),
        'strength': config.get('steering', {}).get('strength'),
        'predictions_path': str(predictions_path),
        'example_metrics_path': str(example_metrics_path),
        'aggregate_metrics_path': str(aggregate_metrics_path),
        'elapsed_sec': elapsed_sec,
    }
    if 'run_metadata' in config:
        log_payload['run_metadata'] = config['run_metadata']

    log_run(root_dir / 'logs' / 'runs.jsonl', log_payload)

    return {
        'experiment_name': experiment_name,
        'predictions_path': str(predictions_path),
        'example_metrics_path': str(example_metrics_path),
        'aggregate_metrics_path': str(aggregate_metrics_path),
        'run_metadata': config.get('run_metadata'),
        'elapsed_sec': elapsed_sec,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(load_yaml(args.config))
