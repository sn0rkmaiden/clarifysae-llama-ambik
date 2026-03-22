from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.config import dump_yaml, load_yaml, set_by_dotted_path
from clarifysae_llama.runners.run_eval import run_eval
from clarifysae_llama.utils.io import ensure_dir, write_csv, write_jsonl


LEGACY_MANIFEST_COLUMNS = [
    'run_name',
    'parameter',
    'value',
    'config_path',
    'predictions_path',
    'example_metrics_path',
    'aggregate_metrics_path',
]

SINGLE_FEATURE_MANIFEST_COLUMNS = [
    'run_name',
    'vocab',
    'hookpoint',
    'feature_index',
    'strength',
    'config_path',
    'predictions_path',
    'example_metrics_path',
    'aggregate_metrics_path',
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to sweep YAML config')
    return parser.parse_args()



def _sanitize_token(value: Any) -> str:
    token = str(value).strip()
    token = token.replace(' ', '')
    token = token.replace('[', '')
    token = token.replace(']', '')
    token = token.replace(',', '-')
    token = token.replace('.', 'p')
    token = re.sub(r'[^A-Za-z0-9_\-]+', '_', token)
    return token.strip('_') or 'value'



def _short_hookpoint(hookpoint: str) -> str:
    match = re.fullmatch(r'layers\.(\d+)\.(.+)', hookpoint)
    if match:
        layer_idx, suffix = match.groups()
        suffix = suffix.replace('.', '_')
        return f'l{layer_idx}_{suffix}'
    return _sanitize_token(hookpoint)



def _build_legacy_run_name(experiment_prefix: str, parameter: str, value: Any) -> str:
    suffix = _sanitize_token(value)
    return f"{experiment_prefix}__{parameter.replace('.', '_')}__{suffix}"



def _build_single_feature_run_name(
    experiment_prefix: str,
    vocab: str | None,
    hookpoint: str,
    feature_index: int,
    strength: Any,
) -> str:
    parts = [experiment_prefix]
    if vocab:
        parts.append(_sanitize_token(vocab))
    parts.extend([
        _short_hookpoint(hookpoint),
        f'feat{feature_index}',
        f'str{_sanitize_token(strength)}',
    ])
    return '__'.join(parts)



def _prepare_sweep_dirs(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> tuple[str, Path, Path]:
    sweep_name = str(sweep_cfg.get('experiment_name') or f"{base_cfg['experiment_name']}__sweep")
    root_dir = Path(base_cfg['output']['root_dir'])
    sweep_dir = ensure_dir(root_dir / 'sweeps' / sweep_name)
    generated_cfg_dir = ensure_dir(sweep_dir / 'generated_configs')
    dump_yaml(sweep_dir / 'source_sweep_config.yaml', sweep_cfg)
    return sweep_name, sweep_dir, generated_cfg_dir



def _validate_legacy_sweep_config(sweep_cfg: dict[str, Any]) -> None:
    sweep_section = sweep_cfg.get('sweep', {})
    if 'parameter' not in sweep_section or 'values' not in sweep_section:
        raise ValueError('Legacy sweep configs must define sweep.parameter and sweep.values.')
    if not isinstance(sweep_section['values'], list) or not sweep_section['values']:
        raise ValueError('sweep.values must be a non-empty list.')



def _validate_single_feature_sweep_config(sweep_cfg: dict[str, Any]) -> None:
    sweep_section = sweep_cfg.get('sweep', {})
    strengths = sweep_section.get('strengths')
    groups = sweep_section.get('groups')

    if not isinstance(strengths, list) or not strengths:
        raise ValueError('single_feature_strength sweep requires a non-empty sweep.strengths list.')
    if not isinstance(groups, list) or not groups:
        raise ValueError('single_feature_strength sweep requires a non-empty sweep.groups list.')

    for group_idx, group in enumerate(groups):
        if 'hookpoint' not in group:
            raise ValueError(f'sweep.groups[{group_idx}] is missing hookpoint.')
        if 'features' not in group:
            raise ValueError(f'sweep.groups[{group_idx}] is missing features.')
        if not isinstance(group['features'], list) or not group['features']:
            raise ValueError(f'sweep.groups[{group_idx}].features must be a non-empty list.')



def _run_legacy_sweep(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    _validate_legacy_sweep_config(sweep_cfg)
    sweep_name, sweep_dir, generated_cfg_dir = _prepare_sweep_dirs(sweep_cfg, base_cfg)
    parameter = sweep_cfg['sweep']['parameter']
    values = sweep_cfg['sweep']['values']

    manifest_rows: list[dict[str, Any]] = []
    for value in values:
        run_cfg = copy.deepcopy(base_cfg)
        set_by_dotted_path(run_cfg, parameter, value)
        run_name = _build_legacy_run_name(sweep_name, parameter, value)
        run_cfg['experiment_name'] = run_name
        cfg_path = generated_cfg_dir / f'{run_name}.yaml'
        dump_yaml(cfg_path, run_cfg)
        result = run_eval(run_cfg)
        manifest_rows.append({
            'run_name': run_name,
            'parameter': parameter,
            'value': value,
            'config_path': str(cfg_path),
            'predictions_path': result['predictions_path'],
            'example_metrics_path': result['example_metrics_path'],
            'aggregate_metrics_path': result['aggregate_metrics_path'],
        })

    write_jsonl(sweep_dir / 'manifest.jsonl', manifest_rows)
    write_csv(sweep_dir / 'manifest.csv', pd.DataFrame(manifest_rows, columns=LEGACY_MANIFEST_COLUMNS))



def _run_single_feature_strength_sweep(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    _validate_single_feature_sweep_config(sweep_cfg)
    sweep_name, sweep_dir, generated_cfg_dir = _prepare_sweep_dirs(sweep_cfg, base_cfg)
    strengths = sweep_cfg['sweep']['strengths']
    groups = sweep_cfg['sweep']['groups']

    manifest_rows: list[dict[str, Any]] = []
    seen_run_names: set[str] = set()

    for group_idx, group in enumerate(groups):
        vocab = group.get('vocab')
        hookpoint = str(group['hookpoint'])
        features = group['features']

        for feature_index in features:
            feature_index = int(feature_index)
            for strength in strengths:
                run_cfg = copy.deepcopy(base_cfg)
                set_by_dotted_path(run_cfg, 'steering.hookpoint', hookpoint)
                set_by_dotted_path(run_cfg, 'steering.feature_indices', [feature_index])
                set_by_dotted_path(run_cfg, 'steering.strength', strength)

                run_name = _build_single_feature_run_name(
                    experiment_prefix=sweep_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_index=feature_index,
                    strength=strength,
                )
                if run_name in seen_run_names:
                    raise ValueError(
                        'Generated duplicate run name. Add a vocab label or adjust your sweep config: '
                        f'{run_name}'
                    )
                seen_run_names.add(run_name)

                run_cfg['experiment_name'] = run_name
                run_cfg['run_metadata'] = {
                    'sweep_name': sweep_name,
                    'sweep_mode': 'single_feature_strength',
                    'group_index': group_idx,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'feature_index': feature_index,
                    'strength': strength,
                }

                cfg_path = generated_cfg_dir / f'{run_name}.yaml'
                dump_yaml(cfg_path, run_cfg)
                result = run_eval(run_cfg)
                manifest_rows.append({
                    'run_name': run_name,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'feature_index': feature_index,
                    'strength': strength,
                    'config_path': str(cfg_path),
                    'predictions_path': result['predictions_path'],
                    'example_metrics_path': result['example_metrics_path'],
                    'aggregate_metrics_path': result['aggregate_metrics_path'],
                })

    manifest_df = pd.DataFrame(manifest_rows, columns=SINGLE_FEATURE_MANIFEST_COLUMNS)
    write_jsonl(sweep_dir / 'manifest.jsonl', manifest_rows)
    write_csv(sweep_dir / 'manifest.csv', manifest_df)


if __name__ == '__main__':
    args = parse_args()
    sweep_cfg = load_yaml(args.config)
    base_cfg = load_yaml(sweep_cfg['base_config'])

    sweep_section = sweep_cfg.get('sweep', {})
    sweep_mode = sweep_section.get('mode', 'legacy')
    if 'parameter' in sweep_section and 'values' in sweep_section:
        _run_legacy_sweep(sweep_cfg, base_cfg)
    elif sweep_mode == 'single_feature_strength':
        _run_single_feature_strength_sweep(sweep_cfg, base_cfg)
    else:
        raise ValueError(
            'Unsupported sweep config. Use either legacy sweep.parameter/sweep.values '
            'or sweep.mode=single_feature_strength with sweep.strengths and sweep.groups.'
        )
