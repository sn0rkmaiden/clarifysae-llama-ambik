from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

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



def run_eval(config: dict) -> dict[str, Any]:
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

    prediction_rows = []
    for start in tqdm(range(0, len(prompt_rows), batch_size), desc='Generating'):
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
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(load_yaml(args.config))
