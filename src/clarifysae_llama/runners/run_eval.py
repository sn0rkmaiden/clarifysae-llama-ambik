from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.backends.steered_hf_backend import SteeredHFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset
from clarifysae_llama.data.prompting import build_clarification_prompt
from clarifysae_llama.eval.metrics import aggregate_metrics, compute_example_metrics, normalize_questions
from clarifysae_llama.eval.reporting import save_metric_tables
from clarifysae_llama.eval.text_matching import initialize_metric_backends
from clarifysae_llama.experiments.fingerprint import fingerprint_payload
from clarifysae_llama.experiments.provenance import write_run_provenance
from clarifysae_llama.utils.io import append_jsonl, ensure_dir, write_json, write_jsonl
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.parsing import assess_json_output, parse_model_json
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
    protocol = str(eval_cfg.get('protocol', 'combined_json'))
    if protocol != 'combined_json':
        raise ValueError(
            f"Unsupported AmbiK evaluation.protocol={protocol!r}. The previous 'separated' "
            "label was not implemented. Use combined_json unless a real multi-stage runner is added."
        )
    return {
        'protocol': protocol,
        'max_questions': int(eval_cfg.get('max_questions', 3)),
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


def build_prompts(dataset: pd.DataFrame, eval_settings: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_questions = int(eval_settings['max_questions'])

    for _, row in dataset.iterrows():
        description = str(row['environment_full'])
        task = str(row.get('task', row['ambiguous_task']))
        prompt_row = {
            'id': str(row['id']),
            'source_id': str(row.get('source_id', row['id'])),
            'variant': str(row.get('variant', 'ambiguous')),
            'ambiguity_type': str(row['ambiguity_type']),
            'environment': description,
            'instruction': task,
            'ambiguous_instruction': task,
            'gold_question': str(row.get('question', '') or ''),
            'gold_answer': str(row.get('answer', '') or ''),
            'gold_plan_for_clear': str(row.get('plan_for_clear_task', '') or ''),
            'prompt': build_clarification_prompt(
                description=description,
                task=task,
                max_questions=max_questions,
            ),
        }
        rows.append(prompt_row)
    return rows


def _print_run_header(config: dict[str, Any], n_examples: int, batch_size: int, eval_settings: dict[str, Any]) -> None:
    experiment_name = config['experiment_name']
    dataset_cfg = config['dataset']
    n_batches = math.ceil(n_examples / batch_size) if n_examples else 0

    print(f"\n=== run_eval :: {experiment_name} ===")
    print(f"dataset: {dataset_cfg['path']}")
    print(
        f"split={dataset_cfg.get('split_name')} variant={dataset_cfg.get('instruction_variant', 'ambiguous')} "
        f"examples={n_examples} batch_size={batch_size} batches={n_batches}"
    )
    print(
        'evaluation: '
        f"protocol={eval_settings['protocol']} "
        f"max_questions={eval_settings['max_questions']} "
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
            f"alpha={steering_cfg.get('strength')} "
            f"feature_scales={steering_cfg.get('feature_scales')} "
            f"scale_method={steering_cfg.get('scale_method')}"
        )
    else:
        print('steering: disabled')


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL in {path} at line {line_number}.') from exc
    return rows


def _run_generation_stage(
    *,
    backend,
    prompt_rows: list[dict[str, Any]],
    batch_size: int,
    console_cfg: dict[str, Any],
    experiment_name: str,
    raw_checkpoint_path: Path,
    resume: bool,
) -> None:
    existing_rows = _read_jsonl(raw_checkpoint_path) if resume else []
    existing_by_id: dict[str, str] = {}
    for row in existing_rows:
        row_id = str(row.get('id'))
        if row_id in existing_by_id:
            raise ValueError(f'Duplicate checkpoint ID {row_id!r} in {raw_checkpoint_path}.')
        existing_by_id[row_id] = str(row.get('raw_model_output', ''))

    expected_ids = {str(row['id']) for row in prompt_rows}
    unexpected = sorted(set(existing_by_id).difference(expected_ids))
    if unexpected:
        raise ValueError(
            f'Checkpoint {raw_checkpoint_path} contains IDs not present in this run: {unexpected[:10]}'
        )

    for row in prompt_rows:
        if str(row['id']) in existing_by_id:
            row['raw_model_output'] = existing_by_id[str(row['id'])]

    pending = [row for row in prompt_rows if 'raw_model_output' not in row]
    iterator = range(0, len(pending), batch_size)
    if console_cfg['show_progress']:
        iterator = tqdm(
            iterator,
            desc=f"{experiment_name} | generating",
            unit='batch',
            dynamic_ncols=True,
        )

    for start in iterator:
        chunk = pending[start:start + batch_size]
        prompts = [row['prompt'] for row in chunk]
        predictions = backend.generate_batch(prompts)
        if len(predictions) != len(chunk):
            raise RuntimeError(
                f'Backend returned {len(predictions)} predictions for a batch of {len(chunk)} prompts.'
            )
        for row, raw_output in zip(chunk, predictions):
            row['raw_model_output'] = raw_output
            append_jsonl(raw_checkpoint_path, {
                'id': str(row['id']),
                'source_id': str(row.get('source_id', row['id'])),
                'variant': row.get('variant', 'ambiguous'),
                'raw_model_output': raw_output,
            })


def _coerce_predicted_ambiguous(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', 'false'}:
            return lowered == 'true'
    return None


def _compact_prediction_row(row: dict[str, Any], *, enable_nli: bool) -> dict[str, Any]:
    keep = [
        'id', 'source_id', 'variant', 'ambiguity_type', 'ambiguous_instruction',
        'gold_question', 'gold_ambiguous', 'predicted_ambiguous',
        'ambiguity_decision_correct', 'model_questions', 'num_questions',
        'asked_question', 'model_question_first_similarity',
        'model_question_best_similarity', 'resolved_proxy_first',
        'resolved_proxy_any', 'json_parsed_output', 'json_exact_valid',
        'json_schema_valid', 'json_recoverable_parse',
    ]
    if enable_nli:
        keep.extend([
            'model_question_first_nli_similarity',
            'model_question_best_nli_similarity',
            'resolved_nli_first', 'resolved_nli_any',
        ])
    return {key: row[key] for key in keep if key in row and row[key] is not None}


def _select_example_metric_columns(raw_df: pd.DataFrame, *, enable_nli: bool) -> list[str]:
    columns = [
        'id', 'source_id', 'variant', 'ambiguity_type', 'gold_ambiguous',
        'predicted_ambiguous', 'ambiguity_decision_correct', 'model_questions',
        'num_questions', 'asked_question', 'model_question_first_similarity',
        'model_question_best_similarity', 'resolved_proxy_first',
        'resolved_proxy_any', 'json_exact_valid', 'json_schema_valid',
        'json_recoverable_parse',
    ]
    if enable_nli:
        columns.extend([
            'model_question_first_nli_similarity',
            'model_question_best_nli_similarity',
            'resolved_nli_first', 'resolved_nli_any',
        ])
    return [column for column in columns if column in raw_df.columns]


def _finalize_prediction_rows(prompt_rows: list[dict[str, Any]], eval_settings: dict[str, Any]) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        prediction_row = {
            'id': row['id'],
            'source_id': row.get('source_id', row['id']),
            'variant': row.get('variant', 'ambiguous'),
            'ambiguity_type': row['ambiguity_type'],
            'environment': row['environment'],
            'ambiguous_instruction': row['ambiguous_instruction'],
            'gold_question': row['gold_question'],
            'gold_answer': row['gold_answer'],
            'gold_plan_for_clear': row['gold_plan_for_clear'],
        }

        raw_output = row.get('raw_model_output', '')
        parsed = parse_model_json(raw_output)
        parsed = parsed if isinstance(parsed, dict) else None
        predicted_ambiguous = _coerce_predicted_ambiguous(parsed.get('ambiguous')) if parsed else None
        model_questions = normalize_questions(parsed.get('question', parsed.get('questions', []))) if parsed else []
        json_metrics = assess_json_output(raw_output)

        prediction_row.update({
            'prompt': row['prompt'],
            'raw_model_output': raw_output,
            'parsed_output': parsed,
            'json_parsed_output': json_metrics['json_parsed_output'],
            'json_exact_valid': json_metrics['json_exact_valid'],
            'json_schema_valid': json_metrics['json_schema_valid'],
            'json_recoverable_parse': json_metrics['json_recoverable_parse'],
        })
        prediction_row.update(compute_example_metrics(
            ambiguity_type=row['ambiguity_type'],
            gold_question=row['gold_question'],
            model_questions=model_questions,
            predicted_ambiguous=predicted_ambiguous,
            embed_threshold=eval_settings['embed_threshold'],
            nli_threshold=eval_settings['nli_threshold'],
            enable_nli=eval_settings['enable_nli'],
        ))
        prediction_rows.append(prediction_row)
    return prediction_rows


def _cleanup_backend(backend) -> None:
    if backend is None:
        return
    try:
        if hasattr(backend, 'steering') and getattr(backend, 'steering', None) is not None:
            try:
                backend.steering.detach()
            except Exception:
                pass
            for attribute in ('sae', 'target_module'):
                try:
                    delattr(backend.steering, attribute)
                except Exception:
                    pass
            try:
                del backend.steering
            except Exception:
                pass
        for attribute in ('model', 'tokenizer', 'generation_kwargs'):
            try:
                delattr(backend, attribute)
            except Exception:
                pass
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _validate_or_create_provenance(run_dir: Path, config: dict[str, Any]) -> None:
    fingerprint_path = run_dir / 'run_fingerprint.json'
    expected = fingerprint_payload(config)
    if fingerprint_path.exists():
        existing = json.loads(fingerprint_path.read_text(encoding='utf-8'))
        actual = existing.get('config_fingerprint')
        if actual != expected:
            raise RuntimeError(
                f'Run directory {run_dir} already belongs to a different config: {actual} != {expected}'
            )
        return
    extra_files = [config['dataset']['path']]
    for key in ('split_manifest',):
        value = config.get('dataset', {}).get(key)
        if value:
            extra_files.append(value)
    scale_artifact = config.get('steering', {}).get('scale_artifact')
    if scale_artifact:
        extra_files.append(scale_artifact)
    write_run_provenance(
        run_dir,
        config,
        repo_root=config.get('repo_root', Path.cwd()),
        command=sys.argv,
        extra_files=extra_files,
    )


def run_eval(config: dict) -> dict[str, Any]:
    console_cfg = _configure_console(config)
    eval_settings = _evaluation_settings(config)
    seed = int(config.get('seed', 42))
    deterministic = bool(config.get('reproducibility', {}).get('deterministic', False))
    set_seed(seed, deterministic=deterministic)

    experiment_name = config['experiment_name']
    root_dir = Path(config['output']['root_dir'])
    run_dir = ensure_dir(root_dir / experiment_name)
    pred_dir = ensure_dir(run_dir / 'predictions')
    ensure_dir(root_dir / 'logs')
    _validate_or_create_provenance(run_dir, config)
    write_json(run_dir / 'status.json', {'status': 'running', 'started_at': time.time()})

    dataset_cfg = config['dataset']
    dataset = load_ambik_clarification_dataset(
        path=dataset_cfg['path'],
        limit=dataset_cfg.get('limit'),
        split_manifest=dataset_cfg.get('split_manifest'),
        split_name=dataset_cfg.get('split_name'),
        instruction_variant=dataset_cfg.get('instruction_variant', 'ambiguous'),
    )
    prompt_rows = build_prompts(dataset, eval_settings)
    if not prompt_rows:
        raise ValueError('The selected dataset split contains no examples.')

    batch_size = int(config.get('batching', {}).get('batch_size', 1))
    _print_run_header(config, len(prompt_rows), batch_size, eval_settings)

    # Fail before expensive generation if a requested scoring backend is unavailable.
    metric_backend_info = initialize_metric_backends(enable_nli=eval_settings['enable_nli'])

    backend = None
    started_at = time.perf_counter()
    raw_checkpoint_path = pred_dir / 'raw_outputs.jsonl'
    resume = bool(config.get('output', {}).get('resume', True))
    if raw_checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f'{raw_checkpoint_path} exists and output.resume=false. Use a new immutable run ID.'
        )

    try:
        backend = build_backend(config)
        _run_generation_stage(
            backend=backend,
            prompt_rows=prompt_rows,
            batch_size=batch_size,
            console_cfg=console_cfg,
            experiment_name=experiment_name,
            raw_checkpoint_path=raw_checkpoint_path,
            resume=resume,
        )

        prediction_rows = _finalize_prediction_rows(prompt_rows, eval_settings)
        compact_rows = [
            _compact_prediction_row(row, enable_nli=eval_settings['enable_nli'])
            for row in prediction_rows
        ]

        predictions_path = pred_dir / 'predictions.jsonl'
        predictions_full_path = pred_dir / 'predictions_full.jsonl'
        results_path = pred_dir / 'results.json'
        results_full_path = pred_dir / 'results_full.json'
        write_jsonl(predictions_path, compact_rows)
        write_jsonl(predictions_full_path, prediction_rows)

        run_info = {
            'dataset': dataset_cfg,
            'seed': seed,
            'deterministic': deterministic,
            'num_examples': len(prompt_rows),
            'model': config['model'],
            'generation': config['generation'],
            'steering_enabled': config.get('steering', {}).get('enabled', False),
            'steering_cfg': config.get('steering') if config.get('steering', {}).get('enabled', False) else None,
            'evaluation': eval_settings,
            'metric_backends': metric_backend_info,
            'backend_provenance': backend.provenance_metadata() if hasattr(backend, 'provenance_metadata') else None,
            'cache_enabled': False,
            'run_metadata': config.get('run_metadata'),
        }
        write_json(results_path, {'run_info': run_info, 'examples': compact_rows})
        write_json(results_full_path, {'run_info': run_info, 'examples': prediction_rows})

        raw_df = pd.DataFrame(prediction_rows)
        example_metrics = raw_df[
            _select_example_metric_columns(raw_df, enable_nli=eval_settings['enable_nli'])
        ].copy()
        aggregate_df, category_df = aggregate_metrics(
            example_metrics,
            embed_threshold=eval_settings['embed_threshold'],
            brevity_max=eval_settings['brevity_max'],
            nli_threshold=eval_settings['nli_threshold'],
            enable_nli=eval_settings['enable_nli'],
        )
        save_metric_tables(example_metrics, aggregate_df, category_df, run_dir)

        elapsed_sec = time.perf_counter() - started_at
        write_json(run_dir / 'status.json', {
            'status': 'complete',
            'completed_at': time.time(),
            'elapsed_sec': elapsed_sec,
            'num_examples': len(prompt_rows),
        })

        log_payload = {
            'experiment_name': experiment_name,
            'run_dir': str(run_dir),
            'dataset': dataset_cfg,
            'n_examples': len(prompt_rows),
            'model_name': config['model']['name'],
            'steering_enabled': config.get('steering', {}).get('enabled', False),
            'hookpoint': config.get('steering', {}).get('hookpoint'),
            'feature_indices': config.get('steering', {}).get('feature_indices'),
            'strength': config.get('steering', {}).get('strength'),
            'feature_scales': config.get('steering', {}).get('feature_scales'),
            'evaluation': eval_settings,
            'elapsed_sec': elapsed_sec,
        }
        log_run(root_dir / 'logs' / 'runs.jsonl', log_payload)
        print(f'completed: {experiment_name} in {elapsed_sec:.1f}s')
        return {
            'experiment_name': experiment_name,
            'run_dir': str(run_dir),
            'predictions_path': str(predictions_path),
            'results_path': str(results_path),
            'example_metrics_path': str(run_dir / 'metrics' / 'example_metrics.csv'),
            'aggregate_metrics_path': str(run_dir / 'tables' / 'aggregate_metrics.csv'),
            'category_metrics_path': str(run_dir / 'tables' / 'category_metrics.csv'),
            'elapsed_sec': elapsed_sec,
        }
    except Exception as exc:
        write_json(run_dir / 'status.json', {
            'status': 'failed',
            'failed_at': time.time(),
            'error_type': type(exc).__name__,
            'error': str(exc),
        })
        raise
    finally:
        _cleanup_backend(backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(load_yaml(args.config))
