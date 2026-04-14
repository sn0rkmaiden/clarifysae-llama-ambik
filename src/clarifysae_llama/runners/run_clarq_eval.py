from __future__ import annotations

import argparse
import copy
import gc
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.backends.steered_hf_backend import SteeredHFCausalBackend
from clarifysae_llama.clarq_legacy.backend_adapter import BackendLLMAdapter
from clarifysae_llama.clarq_legacy.multi_info_provider_agent import helpers_m as MultiInfoProvider
from clarifysae_llama.clarq_legacy.provider_agent import helpers as GeneralProvider
from clarifysae_llama.clarq_legacy.seeker_agent import player as SeekerPlayer
from clarifysae_llama.clarq_legacy.utils import data_combination, read_path
from clarifysae_llama.config import load_yaml
from clarifysae_llama.eval.clarq_metrics import compute_metrics_for_payload, metrics_to_dataframes, parse_evaluation_set
from clarifysae_llama.utils.io import ensure_dir, write_csv, write_json
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.seed import set_seed


def _cleanup_backend(backend) -> None:
    if backend is None:
        return
    try:
        if hasattr(backend, 'steering') and getattr(backend, 'steering', None) is not None:
            try:
                backend.steering.detach()
            except Exception:
                pass
            try:
                del backend.steering
            except Exception:
                pass
        try:
            del backend.model
        except Exception:
            pass
        try:
            del backend.tokenizer
        except Exception:
            pass
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _build_unsteered_backend(model_cfg: dict[str, Any], generation_cfg: dict[str, Any], prompting_cfg: dict[str, Any]) -> HFCausalBackend:
    return HFCausalBackend({
        'model': model_cfg,
        'generation': generation_cfg,
        'prompting': prompting_cfg,
    })


def _build_seeker_backend(config: dict[str, Any]):
    steering_enabled = bool(config.get('steering', {}).get('enabled', False))
    if steering_enabled:
        return SteeredHFCausalBackend(config)
    return HFCausalBackend(config)


def _conversation_meta(config: dict[str, Any], clarq_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        'task_data_path': clarq_cfg['dataset_path'],
        'language': 'En' if 'English' in clarq_cfg['dataset_path'] else 'Ch',
        'mode': 'Chat' if clarq_cfg.get('player_chat_mode', False) else 'Comp',
        'evaluation_set_arg': clarq_cfg.get('evaluation_set', '0-25'),
        'evaluation_set': parse_evaluation_set(clarq_cfg.get('evaluation_set', '0-25')),
        'seeker_agent_llm': config['model']['name'],
        'provider_agent_llm': config['provider_model']['name'],
        'multi_info_provider_agent': bool(clarq_cfg.get('multi_info_provider_agent', False)),
        'steering': {
            'feature': (config.get('steering') or {}).get('feature_indices', [None])[0],
            'strength': (config.get('steering') or {}).get('strength'),
            'hookpoint': (config.get('steering') or {}).get('hookpoint'),
            'sae_repo': (config.get('steering') or {}).get('sae_repo'),
        } if config.get('steering', {}).get('enabled', False) else None,
        'judge_model': (config.get('judge_model') or {}).get('name'),
    }


def run_clarq_eval(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get('seed', 42)))

    clarq_cfg = config['clarq']
    eval_indices = parse_evaluation_set(str(clarq_cfg.get('evaluation_set', '0-25')))
    max_turns_cap = int(clarq_cfg.get('max_turns_cap', 22))
    show_progress = bool(config.get('console', {}).get('show_progress', True))

    experiment_name = config['experiment_name']
    root_dir = Path(config['output']['root_dir'])
    run_dir = ensure_dir(root_dir / experiment_name)
    ensure_dir(root_dir / 'logs')

    all_conv = data_combination(read_path(clarq_cfg['dataset_path']))
    provider_cls = MultiInfoProvider if clarq_cfg.get('multi_info_provider_agent', False) else GeneralProvider

    seeker_backend = provider_backend = judge_backend = None
    started_at = time.perf_counter()

    try:
        seeker_backend = _build_seeker_backend(config)
        provider_backend = _build_unsteered_backend(
            config['provider_model'],
            config['provider_generation'],
            config.get('provider_prompting', {}),
        )
        seeker_llm = BackendLLMAdapter(seeker_backend)
        provider_llm = BackendLLMAdapter(provider_backend)

        outer_iter = enumerate(all_conv)
        if show_progress:
            outer_iter = tqdm(outer_iter, total=len(all_conv), desc=f'{experiment_name} | ClarQ types', dynamic_ncols=True)

        for i, one_type in outer_iter:
            if i not in eval_indices:
                continue
            for j, conv in enumerate(one_type):
                gold_r = conv['all_response'].strip().split('\n')
                provider = provider_cls(gold_r, conv['background_splitted'], conv['gold_structure'], conv, provider_llm)
                seeker = SeekerPlayer(conv['background_splitted'], seeker_llm, clarq_cfg.get('player_chat_mode', False))
                l2l_conv: list[str] = []
                while True:
                    l2l_conv.append(provider.generate_response(l2l_conv))
                    l2l_conv.append(seeker.generate_response(l2l_conv))
                    if provider.is_conv_end(l2l_conv) or len(l2l_conv) > max_turns_cap:
                        break
                conv['l2l'][0] = l2l_conv

        payload = {
            'meta': _conversation_meta(config, clarq_cfg),
            'data': all_conv,
        }
        results_path = pred_dir / 'clarq_results.json'
        write_json(results_path, payload)

        metrics_path = None
        summary_path = None
        metrics_df = summary_df = None

        if config.get('judge_model'):
            judge_backend = _build_unsteered_backend(
                config['judge_model'],
                config['judge_generation'],
                config.get('judge_prompting', {}),
            )
            judge_llm = BackendLLMAdapter(judge_backend)
            metrics = compute_metrics_for_payload(payload, judge_llm, eval_indices)
            metrics_df, summary_df = metrics_to_dataframes(metrics)
            metrics_path = run_dir / 'tables' / 'clarq_metrics.csv'
            summary_path = run_dir / 'tables' / 'clarq_summary.csv'
            write_csv(metrics_path, metrics_df)
            write_csv(summary_path, summary_df)

        elapsed_sec = time.perf_counter() - started_at
        log_payload = {
            'experiment_name': experiment_name,
            'task_data_path': clarq_cfg['dataset_path'],
            'evaluation_set': eval_indices,
            'seeker_model': config['model']['name'],
            'provider_model': config['provider_model']['name'],
            'judge_model': (config.get('judge_model') or {}).get('name'),
            'steering_enabled': config.get('steering', {}).get('enabled', False),
            'results_path': str(results_path),
            'metrics_path': str(metrics_path) if metrics_path else None,
            'summary_path': str(summary_path) if summary_path else None,
            'elapsed_sec': elapsed_sec,
        }
        log_run(root_dir / 'logs' / 'runs.jsonl', log_payload)

        return {
            'experiment_name': experiment_name,
            'results_path': str(results_path),
            'metrics_path': str(metrics_path) if metrics_path else None,
            'summary_path': str(summary_path) if summary_path else None,
            'elapsed_sec': elapsed_sec,
        }
    finally:
        _cleanup_backend(judge_backend)
        _cleanup_backend(provider_backend)
        _cleanup_backend(seeker_backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = run_clarq_eval(load_yaml(args.config))
    print(result)
