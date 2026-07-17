from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.config import load_yaml
from clarifysae_llama.experiments.fingerprint import fingerprint_payload
from clarifysae_llama.utils.io import ensure_dir, write_json


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_feature_catalog(path: str | Path, *, enabled_only: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        'model_key', 'layer', 'feature_id', 'vocab_membership', 'loader',
        'sae_repo', 'hookpoint', 'module_path', 'mode',
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Feature catalog is missing columns: {sorted(missing)}')
    if enabled_only and 'enabled' in frame.columns:
        enabled = frame['enabled'].astype(str).str.lower().isin({'1', 'true', 'yes', 'on'})
        frame = frame.loc[enabled].copy()
    frame['layer'] = frame['layer'].astype(int)
    frame['feature_id'] = frame['feature_id'].astype(int)
    if frame.duplicated(['model_key', 'layer', 'feature_id']).any():
        duplicates = frame.loc[
            frame.duplicated(['model_key', 'layer', 'feature_id'], keep=False),
            ['model_key', 'layer', 'feature_id'],
        ].drop_duplicates()
        raise ValueError(f'Feature catalog has duplicate interventions:\n{duplicates.head(20)}')
    return frame.reset_index(drop=True)


def _load_scales(path: str | Path, method: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Feature scale artifact does not exist: {path}. Run collect_feature_scales first.'
        )
    frame = pd.read_csv(path)
    required = {'model_key', 'layer', 'feature_id', 'selected_scale'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Feature scale artifact is missing columns: {sorted(missing)}')
    if 'scale_method' in frame.columns:
        invalid = frame.loc[frame['scale_method'].astype(str) != str(method)]
        if not invalid.empty:
            raise ValueError(
                f'Scale artifact contains methods other than {method!r}: '
                f"{sorted(invalid['scale_method'].astype(str).unique())}"
            )
    frame['layer'] = frame['layer'].astype(int)
    frame['feature_id'] = frame['feature_id'].astype(int)
    frame['selected_scale'] = pd.to_numeric(frame['selected_scale'], errors='raise')
    if (frame['selected_scale'] <= 0).any():
        bad = frame.loc[frame['selected_scale'] <= 0, ['model_key', 'layer', 'feature_id', 'selected_scale']]
        raise ValueError(f'Feature scales must be positive:\n{bad.head(20)}')
    return frame


def _model_profiles(experiment_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for path in experiment_cfg['experiment']['model_profiles']:
        profile = load_yaml(path)
        model_key = str(profile['model_key'])
        if model_key in profiles:
            raise ValueError(f'Duplicate model profile for {model_key}.')
        profiles[model_key] = profile
    return profiles


def _base_config(experiment_cfg: dict[str, Any], model_profile: dict[str, Any]) -> dict[str, Any]:
    exp = experiment_cfg['experiment']
    dataset_profile = load_yaml(exp['dataset_profile'])
    decoding_profile = load_yaml(exp['decoding_profile'])
    metrics_profile = load_yaml(exp['metrics_profile'])

    base: dict[str, Any] = {}
    for payload in (model_profile, dataset_profile, decoding_profile, metrics_profile):
        base = _deep_merge(base, payload)
    base.pop('model_key', None)
    base.pop('dataset_key', None)
    base.pop('steering_runtime', None)
    dataset_path_env = base.get('dataset', {}).pop('path_env', None)
    if dataset_path_env and os.environ.get(str(dataset_path_env)):
        base['dataset']['path'] = os.environ[str(dataset_path_env)]
    base['seed'] = int(experiment_cfg.get('execution', {}).get('generation_seed', exp.get('seed', 0)))
    base['dataset']['split_name'] = str(exp['split_name'])
    base['dataset']['instruction_variant'] = str(exp.get('instruction_variant', 'paired'))
    output_root = os.environ.get(
        'CLARIFYSAE_RESULTS_ROOT',
        str(exp.get('output_root', 'outputs/manifest_runs')),
    )
    base['output'] = {
        'root_dir': output_root,
        'resume': bool(experiment_cfg.get('execution', {}).get('resume', True)),
    }
    base.setdefault('console', {'show_progress': True, 'suppress_transformers_warnings': True})
    return base


def _steering_config(row: pd.Series, *, alpha: float, scale: float, scale_method: str, scale_artifact: str) -> dict[str, Any]:
    runtime = row.get('_steering_runtime') or {}
    return {
        'enabled': True,
        'loader': str(row['loader']),
        'sae_repo': str(row['sae_repo']),
        'sae_id': None if pd.isna(row.get('sae_id')) or str(row.get('sae_id', '')).strip() == '' else str(row['sae_id']),
        'sae_file': None if pd.isna(row.get('sae_file')) or str(row.get('sae_file', '')).strip() == '' else str(row['sae_file']),
        'hookpoint': str(row['hookpoint']),
        'module_path': str(row['module_path']),
        'feature_indices': [int(row['feature_id'])],
        'feature_scales': [float(scale)],
        'scale_method': str(scale_method),
        'scale_artifact': str(scale_artifact),
        'strength': float(alpha),
        'mode': str(row['mode']),
        'apply_to': str(runtime.get('apply_to', 'all_positions')),
        'steer_generated_tokens_only': bool(runtime.get('steer_generated_tokens_only', False)),
        'runtime': {
            'normalize_reconstruction': bool(runtime.get('normalize_reconstruction', False)),
            'preserve_unsteered_residual': bool(runtime.get('preserve_unsteered_residual', False)),
            'clamp_latents': runtime.get('clamp_latents'),
            'log_feature_acts': bool(runtime.get('log_feature_acts', False)),
        },
    }


def _run_id(
    *,
    experiment_name: str,
    model_key: str,
    layer: int | None,
    feature_id: int | None,
    alpha: float | None,
    baseline: bool,
) -> str:
    if baseline:
        return f'{experiment_name}__{model_key}__baseline'
    alpha_token = str(alpha).replace('-', 'm').replace('.', 'p')
    return f'{experiment_name}__{model_key}__l{layer}__f{feature_id}__a{alpha_token}'


def build_manifest(config_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    experiment_cfg = load_yaml(config_path)
    exp = experiment_cfg['experiment']
    phase = str(exp['phase'])
    if phase not in {'explore', 'confirm'}:
        raise ValueError('experiment.phase must be explore or confirm.')

    profiles = _model_profiles(experiment_cfg)
    scale_cfg = experiment_cfg['steering']['scale']
    scale_method = str(scale_cfg['method'])
    scale_artifact = str(scale_cfg['artifact'])
    scales = _load_scales(scale_artifact, scale_method)

    if phase == 'explore':
        features = _load_feature_catalog(
            experiment_cfg['features']['catalog'],
            enabled_only=bool(experiment_cfg['features'].get('enabled_only', True)),
        )
        alphas = [float(value) for value in experiment_cfg['steering']['alpha']]
        condition_rows: list[tuple[pd.Series, float]] = [
            (row, alpha)
            for _, row in features.iterrows()
            for alpha in alphas
        ]
    else:
        selected_path = Path(experiment_cfg['features']['selected_conditions'])
        if not selected_path.exists():
            raise FileNotFoundError(
                f'Selected condition artifact is missing: {selected_path}. Run select_alpha first.'
            )
        selected = pd.read_csv(selected_path)
        required = {'model_key', 'layer', 'feature_id', 'alpha'}
        missing = required.difference(selected.columns)
        if missing:
            raise ValueError(f'Selected-condition artifact is missing columns: {sorted(missing)}')
        catalog_path = experiment_cfg['features'].get('catalog', 'manifests/feature_catalog.csv')
        catalog = _load_feature_catalog(catalog_path, enabled_only=False)
        features = selected.merge(
            catalog,
            on=['model_key', 'layer', 'feature_id'],
            how='left',
            validate='one_to_one',
        )
        if features['loader'].isna().any():
            raise ValueError('Selected conditions reference features missing from the feature catalog.')
        condition_rows = [(row, float(row['alpha'])) for _, row in features.iterrows()]

    rows: list[dict[str, Any]] = []
    for row, alpha in condition_rows:
        model_key = str(row['model_key'])
        if model_key not in profiles:
            raise ValueError(f'No model profile configured for feature model_key={model_key!r}.')
        profile = profiles[model_key]
        scale_match = scales.loc[
            (scales['model_key'].astype(str) == model_key)
            & (scales['layer'].astype(int) == int(row['layer']))
            & (scales['feature_id'].astype(int) == int(row['feature_id']))
        ]
        if len(scale_match) != 1:
            raise ValueError(
                f'Expected one feature scale for {(model_key, int(row["layer"]), int(row["feature_id"]))}, '
                f'found {len(scale_match)}.'
            )
        scale = float(scale_match.iloc[0]['selected_scale'])
        row = row.copy()
        row['_steering_runtime'] = profile.get('steering_runtime', {})
        config = _base_config(experiment_cfg, profile)
        run_id = _run_id(
            experiment_name=str(exp['name']), model_key=model_key,
            layer=int(row['layer']), feature_id=int(row['feature_id']),
            alpha=alpha, baseline=False,
        )
        config['experiment_name'] = run_id
        config['steering'] = _steering_config(
            row, alpha=alpha, scale=scale,
            scale_method=scale_method, scale_artifact=scale_artifact,
        )
        config['run_metadata'] = {
            'manifest_experiment': str(exp['name']),
            'phase': phase,
            'model_key': model_key,
            'layer': int(row['layer']),
            'feature_id': int(row['feature_id']),
            'vocab_membership': str(row['vocab_membership']),
            'alpha': alpha,
            'feature_scale': scale,
            'scale_method': scale_method,
        }
        fingerprint = fingerprint_payload(config)
        rows.append({
            'run_id': run_id,
            'phase': phase,
            'model_key': model_key,
            'layer': int(row['layer']),
            'feature_id': int(row['feature_id']),
            'vocab_membership': str(row['vocab_membership']),
            'alpha': alpha,
            'feature_scale': scale,
            'scale_method': scale_method,
            'split_name': str(exp['split_name']),
            'config_fingerprint': fingerprint,
            'config_json': json.dumps(config, ensure_ascii=False, sort_keys=True),
            'status': 'pending',
        })

    if bool(experiment_cfg.get('execution', {}).get('include_baselines', True)):
        for model_key, profile in profiles.items():
            config = _base_config(experiment_cfg, profile)
            run_id = _run_id(
                experiment_name=str(exp['name']), model_key=model_key,
                layer=None, feature_id=None, alpha=None, baseline=True,
            )
            config['experiment_name'] = run_id
            config['steering'] = {'enabled': False}
            config['run_metadata'] = {
                'manifest_experiment': str(exp['name']),
                'phase': phase,
                'model_key': model_key,
                'condition': 'baseline',
            }
            rows.append({
                'run_id': run_id,
                'phase': phase,
                'model_key': model_key,
                'layer': None,
                'feature_id': None,
                'vocab_membership': None,
                'alpha': None,
                'feature_scale': None,
                'scale_method': None,
                'split_name': str(exp['split_name']),
                'config_fingerprint': fingerprint_payload(config),
                'config_json': json.dumps(config, ensure_ascii=False, sort_keys=True),
                'status': 'pending',
            })

    manifest = pd.DataFrame(rows).sort_values(
        ['model_key', 'layer', 'feature_id', 'alpha'], na_position='first'
    ).reset_index(drop=True)
    if manifest['run_id'].duplicated().any():
        raise AssertionError('Generated duplicate run IDs.')

    if output_path is None:
        output_path = Path('manifests/generated') / f"{exp['name']}.csv"
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    manifest.to_csv(output_path, index=False)
    write_json(output_path.with_suffix('.metadata.json'), {
        'source_config': str(config_path),
        'experiment_name': str(exp['name']),
        'phase': phase,
        'num_rows': len(manifest),
        'num_steered_rows': int(manifest['feature_id'].notna().sum()),
        'num_baselines': int(manifest['feature_id'].isna().sum()),
        'manifest_fingerprint': fingerprint_payload(manifest.drop(columns=['status']).to_dict(orient='records')),
    })
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a self-contained execution manifest.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.config, args.output)
    print(manifest.drop(columns=['config_json']).to_string(index=False))
    print(f'Generated {len(manifest)} manifest rows.')


if __name__ == '__main__':
    main()
