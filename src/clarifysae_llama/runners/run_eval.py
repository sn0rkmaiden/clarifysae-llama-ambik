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
from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset
from clarifysae_llama.data.prompting import build_clarification_prompt
from clarifysae_llama.eval.metrics import aggregate_metrics, compute_example_metrics, normalize_questions
from clarifysae_llama.eval.reporting import save_metric_tables
from clarifysae_llama.utils.io import ensure_dir, write_json, write_jsonl
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.parsing import parse_model_json
from clarifysae_llama.utils.seed import set_seed



def _configure_console(config: dict[str, Any]) -> dict[str, Any]:
    console_cfg = config.get('console', {})
    suppress_tf_warnings = bool(console_cfg.get('suppress_transformers_warnings', True))
    show_progress = bool(console_cfg.get('show_progress', True))

    if suppress_tf_warnings:
        os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        hf_logging.set_verbosity_error()
        warnings.filterwarnings('ignore', message=r'.*Both `max_new_tokens` .* and `max_length`.*')
        warnings.filterwarnings('ignore', message=r'.*The following generation flags are not valid and may be ignored:.*')
    else:
        hf_logging.set_verbosity_warning()

    return {
        'show_progress': show_progress,
        'suppress_transformers_warnings': suppress_tf_warnings,
    }



def _evaluation_settings(config: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = config.get('evaluation', {})
    return {
        'embed_threshold': float(eval_cfg.get('embed_threshold', 0.75)),
        'nli_threshold': eval_cfg.get('nli_threshold'),
        'enable_nli': bool(eval_cfg.get('enable_nli', False)),
        'brevity_max': int(eval_cfg.get('brevity_max', 1)),
    }



def build_backend(config: dict):
    backend_name = config['model'].get('backend', 'hf')
    steering_enabled = config.get('steering', {}).get('enabled', False)

    if backend_name != 'hf':
        raise ValueError(f'Only hf backend is supported in this repo, got: {backend_name}')
    if steering_enabled:
        return SteeredHFCausalBackend(config)
    return HFCausalBackend(config)



def build_prompts(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in dataset.iterrows():
        prompt = build_clarification_prompt(
            description=str(row['environment_full']),
            task=str(row['ambiguous_task']),
        )
        rows.append({
            'id': int(row['id']),
            'ambiguity_type': str(row['ambiguity_type']),
            'environment': str(row['environment_full']),
            'ambiguous_instruction': str(row['ambiguous_task']),
            'gold_question': str(row.get('question', '') or ''),
            'gold_answer': str(row.get('answer', '') or ''),
            'gold_plan_for_clear': str(row.get('plan_for_clear_task', '') or ''),
            'prompt': prompt,
        })
    return rows



def _print_run_header(config: dict[str, Any], n_examples: int, batch_size: int, eval_settings: dict[str, Any]) -> None:
    experiment_name = config['experiment_name']
    dataset_path = config['dataset']['path']
    n_batches = math.ceil(n_examples / batch_size) if n_examples else 0

    print(f"\n=== run_eval :: {experiment_name} ===")
    print(f"dataset: {dataset_path}")
    print(f"eval examples: {n_examples} | batch_size: {batch_size} | batches: {n_batches}")
    print(
        'evaluation: '
        f"embed_threshold={eval_settings['embed_threshold']} "
        f"brevity_max={eval_settings['brevity_max']} "
        f"enable_nli={eval_settings['enable_nli']}"
    )

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
    eval_settings = _evaluation_settings(config)
    set_seed(int(config.get('seed', 42)))

    experiment_name = config['experiment_name']
    root_dir = Path(config['output']['root_dir'])
    pred_dir = ensure_dir(root_dir / 'predictions' / experiment_name)
    run_dir = ensure_dir(root_dir / experiment_name)
    ensure_dir(root_dir / 'logs')

    dataset = load_ambik_clarification_dataset(
        path=config['dataset']['path'],
        limit=config['dataset'].get('limit'),
    )
    prompt_rows = build_prompts(dataset)

    backend = build_backend(config)
    batch_size = int(config.get('batching', {}).get('batch_size', 1))
    _print_run_header(config, n_examples=len(prompt_rows), batch_size=batch_size, eval_settings=eval_settings)

    started_at = time.perf_counter()
    prediction_rows: list[dict[str, Any]] = []
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

        for row, raw_output in zip(chunk, predictions):
            parsed = parse_model_json(raw_output)
            parsed = parsed if isinstance(parsed, dict) else {}
            predicted_ambiguous = parsed.get('ambiguous')
            if isinstance(predicted_ambiguous, str):
                lowered = predicted_ambiguous.strip().lower()
                if lowered in {'true', 'false'}:
                    predicted_ambiguous = lowered == 'true'
                else:
                    predicted_ambiguous = None
            elif not isinstance(predicted_ambiguous, bool):
                predicted_ambiguous = None

            questions_field = parsed.get('question', parsed.get('questions', []))
            model_questions = normalize_questions(questions_field)
            metrics = compute_example_metrics(
                ambiguity_type=row['ambiguity_type'],
                gold_question=row['gold_question'],
                model_questions=model_questions,
                predicted_ambiguous=predicted_ambiguous,
                embed_threshold=eval_settings['embed_threshold'],
                nli_threshold=eval_settings['nli_threshold'],
                enable_nli=eval_settings['enable_nli'],
            )

            prediction_rows.append({
                'id': row['id'],
                'ambiguity_type': row['ambiguity_type'],
                'environment': row['environment'],
                'ambiguous_instruction': row['ambiguous_instruction'],
                'gold_question': row['gold_question'],
                'gold_answer': row['gold_answer'],
                'gold_plan_for_clear': row['gold_plan_for_clear'],
                'prompt': row['prompt'],
                'raw_model_output': raw_output,
                'parsed_output': parsed,
                **metrics,
            })

    predictions_path = pred_dir / 'predictions.jsonl'
    results_path = pred_dir / 'results.json'
    write_jsonl(predictions_path, prediction_rows)

    run_info = {
        'dataset_csv': config['dataset']['path'],
        'output_json': str(results_path),
        'seed': int(config.get('seed', 42)),
        'num_examples': len(prompt_rows),
        'model_name': config['model']['name'],
        'steering_enabled': config.get('steering', {}).get('enabled', False),
        'steering_cfg': config.get('steering') if config.get('steering', {}).get('enabled', False) else None,
        'evaluation': eval_settings,
    }
    if 'run_metadata' in config:
        run_info['run_metadata'] = config['run_metadata']

    write_json(results_path, {'run_info': run_info, 'examples': prediction_rows})

    raw_df = pd.DataFrame(prediction_rows)
    example_metrics = raw_df[
        [
            'id',
            'ambiguity_type',
            'gold_question',
            'gold_answer',
            'predicted_ambiguous',
            'ambiguity_decision_correct',
            'model_questions',
            'num_questions',
            'asked_question',
            'model_question_best_similarity',
            'resolved_proxy',
            'model_question_best_nli_similarity',
            'resolved_nli',
            'raw_model_output',
        ]
    ].copy()

    aggregate_df, category_df = aggregate_metrics(
        example_metrics,
        embed_threshold=eval_settings['embed_threshold'],
        brevity_max=eval_settings['brevity_max'],
        nli_threshold=eval_settings['nli_threshold'],
        enable_nli=eval_settings['enable_nli'],
    )
    save_metric_tables(example_metrics, aggregate_df, category_df, run_dir)

    example_metrics_path = run_dir / 'metrics' / 'example_metrics.csv'
    aggregate_metrics_path = run_dir / 'tables' / 'aggregate_metrics.csv'
    category_metrics_path = run_dir / 'tables' / 'category_metrics.csv'

    elapsed_sec = time.perf_counter() - started_at
    print(f"completed: {experiment_name} in {elapsed_sec:.1f}s")
    print(f"  predictions: {predictions_path}")
    print(f"  results json: {results_path}")
    print(f"  example metrics: {example_metrics_path}")
    print(f"  aggregate metrics: {aggregate_metrics_path}")
    print(f"  category metrics: {category_metrics_path}")

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
        'results_path': str(results_path),
        'example_metrics_path': str(example_metrics_path),
        'aggregate_metrics_path': str(aggregate_metrics_path),
        'category_metrics_path': str(category_metrics_path),
        'evaluation': eval_settings,
        'elapsed_sec': elapsed_sec,
    }
    if 'run_metadata' in config:
        log_payload['run_metadata'] = config['run_metadata']

    log_run(root_dir / 'logs' / 'runs.jsonl', log_payload)

    return {
        'experiment_name': experiment_name,
        'predictions_path': str(predictions_path),
        'results_path': str(results_path),
        'example_metrics_path': str(example_metrics_path),
        'aggregate_metrics_path': str(aggregate_metrics_path),
        'category_metrics_path': str(category_metrics_path),
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
