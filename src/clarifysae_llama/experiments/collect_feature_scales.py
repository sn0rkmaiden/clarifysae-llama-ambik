from __future__ import annotations

import argparse
import copy
import gc
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset
from clarifysae_llama.data.prompting import build_clarification_prompt
from clarifysae_llama.discovery.sae_utils import SparseLatents, encode_sparse
from clarifysae_llama.experiments.fingerprint import sha256_file
from clarifysae_llama.steering.config import SteeringConfig
from clarifysae_llama.steering.sparsify_steerer import SparsifySteerer
from clarifysae_llama.utils.io import ensure_dir
from clarifysae_llama.utils.seed import set_seed


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_model_profile(experiment_cfg: dict[str, Any], model_key: str) -> dict[str, Any]:
    for path in experiment_cfg['experiment']['model_profiles']:
        profile = load_yaml(path)
        if str(profile['model_key']) == model_key:
            return profile
    raise ValueError(f'No model profile configured for {model_key!r}.')


def _group_catalog(catalog_path: str | Path, model_key: str, layer: int) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    enabled = catalog.get('enabled', pd.Series(True, index=catalog.index)).astype(str).str.lower().isin(
        {'true', '1', 'yes', 'on'}
    )
    subset = catalog.loc[
        enabled
        & (catalog['model_key'].astype(str) == model_key)
        & (catalog['layer'].astype(int) == int(layer))
    ].copy()
    if subset.empty:
        raise ValueError(f'No enabled features for model={model_key}, layer={layer}.')
    steering_columns = ['loader', 'sae_repo', 'sae_id', 'sae_file', 'hookpoint', 'module_path', 'mode']
    for column in steering_columns:
        if subset[column].fillna('').astype(str).nunique() > 1:
            raise ValueError(f'Feature group has inconsistent {column} values.')
    return subset.sort_values('feature_id').reset_index(drop=True)


def _prompt_texts(experiment_cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    exp = experiment_cfg['experiment']
    dataset_profile = load_yaml(exp['dataset_profile'])
    dataset_cfg = dict(dataset_profile['dataset'])
    path_env = dataset_cfg.pop('path_env', None)
    if path_env and os.environ.get(str(path_env)):
        dataset_cfg['path'] = os.environ[str(path_env)]
    split_name = str(exp['split_name'])
    variant = str(exp.get('instruction_variant', 'paired'))
    frame = load_ambik_clarification_dataset(
        dataset_cfg['path'],
        split_manifest=dataset_cfg.get('split_manifest'),
        split_name=split_name,
        instruction_variant=variant,
    )
    prompts = [
        build_clarification_prompt(
            description=str(row['environment_full']),
            task=str(row.get('task', row['ambiguous_task'])),
            max_questions=3,
        )
        for _, row in frame.iterrows()
    ]
    return prompts, {
        'dataset_path': dataset_cfg['path'],
        'split_manifest': dataset_cfg.get('split_manifest'),
        'split_name': split_name,
        'instruction_variant': variant,
        'num_prompts': len(prompts),
    }


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == '':
        return None
    return str(value)


def collect_feature_scales(
    *,
    experiment_config: str | Path,
    model_key: str,
    layer: int,
    output_path: str | Path,
    quantile: float = 0.95,
    max_prompts: int | None = None,
) -> pd.DataFrame:
    if not 0 < quantile < 1:
        raise ValueError('quantile must be between 0 and 1.')
    experiment_cfg = load_yaml(experiment_config)
    profile = _load_model_profile(experiment_cfg, model_key)
    catalog_path = experiment_cfg['features']['catalog']
    features = _group_catalog(catalog_path, model_key, layer)
    prompts, dataset_metadata = _prompt_texts(experiment_cfg)
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]

    decoding_profile = load_yaml(experiment_cfg['experiment']['decoding_profile'])
    backend_cfg = _deep_merge(profile, decoding_profile)
    backend_cfg.pop('model_key', None)
    backend_cfg.setdefault('generation', {'max_new_tokens': 1, 'do_sample': False})
    set_seed(int(experiment_cfg['experiment'].get('seed', 0)), deterministic=True)
    backend = HFCausalBackend(backend_cfg)

    first = features.iloc[0]
    steering_runtime = profile.get('steering_runtime', {})
    steerer = SparsifySteerer(
        model=backend.model,
        model_device=next(backend.model.parameters()).device,
        dtype=backend.dtype,
        config=SteeringConfig(
            sae_repo=str(first['sae_repo']),
            hookpoint=str(first['hookpoint']),
            feature_indices=[int(value) for value in features['feature_id'].tolist()],
            strength=0.0,
            loader=str(first['loader']),
            sae_file=_optional_text(first.get('sae_file')),
            sae_id=_optional_text(first.get('sae_id')),
            module_path=str(first['module_path']),
            mode=str(first['mode']),
            apply_to=str(steering_runtime.get('apply_to', 'all_positions')),
        ),
    )

    feature_ids = [int(value) for value in features['feature_id'].tolist()]
    positive_values: dict[int, list[torch.Tensor]] = defaultdict(list)
    total_tokens = 0

    @torch.inference_mode()
    def capture_hook(_module, _inputs, output):
        nonlocal total_tokens
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None or hidden.ndim != 3:
            return None
        flat = hidden.reshape(-1, hidden.shape[-1]).to(
            device=steerer.sae_device,
            dtype=steerer.sae_dtype,
        )
        sparse = encode_sparse(steerer.sae, flat)
        if not isinstance(sparse, SparseLatents):
            raise TypeError('Expected sparse SAE latents.')
        total_tokens += int(flat.shape[0])
        for feature_id in feature_ids:
            mask = sparse.top_indices == feature_id
            values = sparse.top_acts[mask]
            values = values[torch.isfinite(values) & (values > 0)]
            if values.numel():
                positive_values[feature_id].append(values.detach().float().cpu())
        return None

    handle = steerer.target_module.register_forward_hook(capture_hook)
    try:
        for index, prompt in enumerate(prompts, start=1):
            formatted = backend._format_prompt(prompt)
            tokenized = backend.tokenizer(formatted, return_tensors='pt', truncation=True)
            tokenized = backend._inputs_to_model_device(tokenized)
            backend.model(**tokenized, use_cache=False)
            if index % 25 == 0 or index == len(prompts):
                print(f'[{model_key} layer {layer}] scale prompts: {index}/{len(prompts)}')
    finally:
        handle.remove()

    rows: list[dict[str, Any]] = []
    method_name = f'positive_q{int(round(quantile * 100))}'
    for _, feature_row in features.iterrows():
        feature_id = int(feature_row['feature_id'])
        chunks = positive_values.get(feature_id, [])
        values = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
        if values.numel() == 0:
            selected_scale = None
            status = 'non_firing'
            q50 = q90 = q95 = q99 = None
        else:
            selected_scale = float(torch.quantile(values, quantile).item())
            q50 = float(torch.quantile(values, 0.50).item())
            q90 = float(torch.quantile(values, 0.90).item())
            q95 = float(torch.quantile(values, 0.95).item())
            q99 = float(torch.quantile(values, 0.99).item())
            if selected_scale <= 0:
                status = 'invalid_scale'
            elif values.numel() < 20:
                status = 'very_sparse'
            elif values.numel() < 200:
                status = 'rare'
            else:
                status = 'ok'
        rows.append({
            'model_key': model_key,
            'layer': int(layer),
            'feature_id': feature_id,
            'vocab_membership': str(feature_row['vocab_membership']),
            'num_tokens': int(total_tokens),
            'num_positive': int(values.numel()),
            'firing_rate': float(values.numel() / total_tokens) if total_tokens else 0.0,
            'positive_q50': q50,
            'positive_q90': q90,
            'positive_q95': q95,
            'positive_q99': q99,
            'selected_scale': selected_scale,
            'scale_method': method_name,
            'status': status,
            'dataset_path': dataset_metadata['dataset_path'],
            'split_manifest': dataset_metadata['split_manifest'],
            'split_name': dataset_metadata['split_name'],
            'instruction_variant': dataset_metadata['instruction_variant'],
            'num_prompts': len(prompts),
            'dataset_sha256': sha256_file(dataset_metadata['dataset_path']),
        })

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    new_frame = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        keys = ['model_key', 'layer', 'feature_id']
        existing = existing.merge(new_frame[keys], on=keys, how='left', indicator=True)
        existing = existing.loc[existing['_merge'] == 'left_only'].drop(columns=['_merge'])
        new_frame = pd.concat([existing, new_frame], ignore_index=True)
    new_frame = new_frame.sort_values(['model_key', 'layer', 'feature_id']).reset_index(drop=True)
    new_frame.to_csv(output_path, index=False)

    try:
        del steerer.sae
        del backend.model
        del backend.tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return new_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect robust per-feature activation scales.')
    parser.add_argument('--experiment-config', required=True)
    parser.add_argument('--model-key', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--output', default='artifacts/feature_scales/ambik_v1_feature_scales.csv')
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--max-prompts', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = collect_feature_scales(
        experiment_config=args.experiment_config,
        model_key=args.model_key,
        layer=args.layer,
        output_path=args.output,
        quantile=args.quantile,
        max_prompts=args.max_prompts,
    )
    subset = frame.loc[
        (frame['model_key'].astype(str) == args.model_key)
        & (frame['layer'].astype(int) == args.layer)
    ]
    print(subset.to_string(index=False))


if __name__ == '__main__':
    main()
