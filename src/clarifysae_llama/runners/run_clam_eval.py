from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.data.ambik_loader import load_ambik_selective_dataset
from clarifysae_llama.data.clam_prompting import (
    build_clam_classification_prompt,
    build_clam_question_prompt,
)
from clarifysae_llama.eval.clam_metrics import add_clam_aggregate_metrics, clean_single_question
from clarifysae_llama.eval.metrics import aggregate_metrics, compute_example_metrics
from clarifysae_llama.eval.reporting import save_metric_tables
from clarifysae_llama.utils.io import ensure_dir, write_json, write_jsonl
from clarifysae_llama.utils.seed import set_seed


def _load_demonstrations(path: str | Path) -> list[dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError('CLAM demonstrations file must contain a JSON list.')
    return [dict(item) for item in payload]


def _softmax_pair(ambiguous_score: float, clear_score: float) -> float:
    values = torch.tensor([ambiguous_score, clear_score], dtype=torch.float64)
    return float(torch.softmax(values, dim=0)[0].item())


def _build_rows(dataset: pd.DataFrame, demonstrations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in dataset.iterrows():
        environment = str(item['environment_full'])
        task = str(item['task'])
        rows.append({
            **item.to_dict(),
            'classification_prompt': build_clam_classification_prompt(
                environment=environment,
                task=task,
                demonstrations=demonstrations,
            ),
            'question_prompt': build_clam_question_prompt(
                environment=environment,
                task=task,
                demonstrations=demonstrations,
            ),
        })
    return rows


def _batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def run_clam_eval(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get('seed', 42)))
    experiment_name = str(config['experiment_name'])
    root_dir = Path(config['output']['root_dir'])
    run_dir = ensure_dir(root_dir / experiment_name)
    pred_dir = ensure_dir(run_dir / 'predictions')

    clam_cfg = config.get('clam', {})
    demonstrations = _load_demonstrations(clam_cfg['demonstrations_path'])
    include_pairs = bool(config.get('dataset', {}).get('include_unambiguous_pairs', True))
    dataset = load_ambik_selective_dataset(
        config['dataset']['path'],
        limit_pairs=config['dataset'].get('limit'),
        include_unambiguous_pairs=include_pairs,
    )
    rows = _build_rows(dataset, demonstrations)

    config = dict(config)
    config['steering'] = {'enabled': False}
    backend = HFCausalBackend(config)

    batch_size = int(config.get('batching', {}).get('batch_size', 1))
    threshold = float(clam_cfg.get('decision_threshold', 0.5))
    candidate_text = clam_cfg.get('candidate_text', {})
    ambiguous_candidate = str(candidate_text.get('ambiguous', ' AMBIGUOUS'))
    clear_candidate = str(candidate_text.get('clear', ' CLEAR'))
    candidates = [ambiguous_candidate, clear_candidate]
    length_normalize = bool(clam_cfg.get('length_normalize_scores', True))

    print(f'\n=== run_clam_eval :: {experiment_name} ===')
    print(f'pairs: {len(dataset) // 2 if include_pairs else len(dataset)} | examples: {len(dataset)}')
    print(f'decision threshold: {threshold} | candidates: {candidates}')

    classification_batches = list(_batched(rows, batch_size))
    for chunk in tqdm(classification_batches, desc='CLAM stage 1: classify', unit='batch'):
        prompts = [row['classification_prompt'] for row in chunk]
        scores = backend.score_candidate_continuations_batch(
            prompts,
            candidates,
            length_normalize=length_normalize,
        )
        for row, (ambiguous_score, clear_score) in zip(chunk, scores):
            probability = _softmax_pair(ambiguous_score, clear_score)
            row['ambiguity_score_ambiguous'] = ambiguous_score
            row['ambiguity_score_clear'] = clear_score
            row['ambiguity_probability'] = probability
            row['predicted_ambiguous'] = bool(probability >= threshold)

    # Generate stage-2 questions for every gold-ambiguous example. This gives
    # both (a) the end-to-end CLAM result, gated by the predicted label, and
    # (b) an oracle-gated question-generation diagnostic that is directly
    # comparable to ClarifySAE's ambiguity-known evaluation setting.
    question_rows = [row for row in rows if bool(row['gold_ambiguous']) or row['predicted_ambiguous']]
    question_batches = list(_batched(question_rows, batch_size))
    for chunk in tqdm(question_batches, desc='CLAM stage 2: ask', unit='batch'):
        raw_outputs = backend.generate_batch([row['question_prompt'] for row in chunk])
        for row, raw_output in zip(chunk, raw_outputs):
            cleaned = clean_single_question(raw_output)
            row['raw_oracle_question_output'] = raw_output if bool(row['gold_ambiguous']) else ''
            row['oracle_generated_question'] = cleaned if bool(row['gold_ambiguous']) else ''
            row['raw_question_output'] = raw_output if row['predicted_ambiguous'] else ''
            row['generated_question'] = cleaned if row['predicted_ambiguous'] else ''

    for row in rows:
        row.setdefault('raw_question_output', '')
        row.setdefault('generated_question', '')
        row.setdefault('raw_oracle_question_output', '')
        row.setdefault('oracle_generated_question', '')

    embed_threshold = float(config.get('evaluation', {}).get('embed_threshold', 0.75))
    nli_threshold = config.get('evaluation', {}).get('nli_threshold')
    enable_nli = bool(config.get('evaluation', {}).get('enable_nli', False))
    brevity_max = int(config.get('evaluation', {}).get('brevity_max', 1))

    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        questions = [row['generated_question']] if row['generated_question'] else []
        metrics = compute_example_metrics(
            ambiguity_type=str(row['ambiguity_type']),
            gold_question=str(row['gold_question']),
            model_questions=questions,
            predicted_ambiguous=bool(row['predicted_ambiguous']),
            embed_threshold=embed_threshold,
            nli_threshold=nli_threshold,
            enable_nli=enable_nli,
        )
        # The explicit pair label is authoritative for this runner.
        metrics['gold_ambiguous'] = bool(row['gold_ambiguous'])
        metrics['ambiguity_decision_correct'] = bool(
            row['predicted_ambiguous'] == row['gold_ambiguous']
        )
        oracle_questions = [row['oracle_generated_question']] if row['oracle_generated_question'] else []
        oracle_metrics = compute_example_metrics(
            ambiguity_type=str(row['ambiguity_type']),
            gold_question=str(row['gold_question']),
            model_questions=oracle_questions,
            predicted_ambiguous=True if bool(row['gold_ambiguous']) else False,
            embed_threshold=embed_threshold,
            nli_threshold=nli_threshold,
            enable_nli=enable_nli,
        )

        prediction_rows.append({
            'id': row['id'],
            'source_id': row['source_id'],
            'variant': row['variant'],
            'ambiguity_type': row['ambiguity_type'],
            'environment': row['environment_full'],
            'instruction': row['task'],
            'gold_question': row['gold_question'],
            'gold_ambiguous': bool(row['gold_ambiguous']),
            'classification_prompt': row['classification_prompt'],
            'question_prompt': row['question_prompt'] if row['predicted_ambiguous'] else None,
            'ambiguity_score_ambiguous': row['ambiguity_score_ambiguous'],
            'ambiguity_score_clear': row['ambiguity_score_clear'],
            'ambiguity_probability': row['ambiguity_probability'],
            'predicted_ambiguous': bool(row['predicted_ambiguous']),
            'raw_question_output': row['raw_question_output'],
            'generated_question': row['generated_question'],
            'raw_oracle_question_output': row['raw_oracle_question_output'],
            'oracle_generated_question': row['oracle_generated_question'],
            'oracle_gate_question_similarity': oracle_metrics['model_question_best_similarity'],
            'oracle_gate_resolved_proxy_any': oracle_metrics['resolved_proxy_any'],
            **metrics,
        })

    predictions_path = pred_dir / 'predictions.jsonl'
    results_path = pred_dir / 'results.json'
    write_jsonl(predictions_path, prediction_rows)
    write_json(results_path, {
        'run_info': {
            'experiment_name': experiment_name,
            'model_name': config['model']['name'],
            'dataset_path': config['dataset']['path'],
            'include_unambiguous_pairs': include_pairs,
            'decision_threshold': threshold,
            'candidates': candidates,
            'demonstrations_path': clam_cfg['demonstrations_path'],
        },
        'examples': prediction_rows,
    })

    example_metrics = pd.DataFrame(prediction_rows)
    aggregate_df, category_df = aggregate_metrics(
        example_metrics,
        embed_threshold=embed_threshold,
        brevity_max=brevity_max,
        nli_threshold=nli_threshold,
        enable_nli=enable_nli,
    )
    aggregate_df = add_clam_aggregate_metrics(aggregate_df, example_metrics)
    save_metric_tables(example_metrics, aggregate_df, category_df, run_dir)

    print(f'completed: {experiment_name}')
    print(f'  predictions: {predictions_path}')
    print(f'  aggregate metrics: {run_dir / "tables" / "aggregate_metrics.csv"}')
    return {
        'experiment_name': experiment_name,
        'predictions_path': str(predictions_path),
        'results_path': str(results_path),
        'aggregate_metrics_path': str(run_dir / 'tables' / 'aggregate_metrics.csv'),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_clam_eval(load_yaml(args.config))
