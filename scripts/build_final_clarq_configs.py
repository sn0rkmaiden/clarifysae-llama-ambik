from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from clarifysae_llama.config import dump_yaml, load_yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'configs' / 'final' / 'clarq'

FINAL_CONFIGS: dict[str, dict[str, Any]] = {
    'gemma2b': {
        'base': 'configs/clarq/gemma_2b_clarq_sweep_base.yaml',
        'feature_indices': [538],
        'strength': -5.0,
        'hookpoint': 'blocks.12.hook_resid_post',
        'module_path': 'model.layers.12',
        'sae_id': 'blocks.12.hook_resid_post',
    },
    'gemma9b': {
        'base': 'configs/clarq/gemma2_9b_clarq_sweep_base.yaml',
        'feature_indices': [344],
        'strength': 3.0,
        'hookpoint': 'blocks.20.hook_resid_post',
        'module_path': 'model.layers.20',
        'sae_id': 'layer_20/width_16k/average_l0_91',
    },
    'llama1b': {
        'base': 'configs/clarq/1b_instruct_clarq_sweep_base.yaml',
        'feature_indices': [6230],
        'strength': -5.0,
        'hookpoint': 'model.layers.12',
        'module_path': 'model.layers.12',
    },
    'llama8b': {
        'base': 'configs/clarq/8b_instruct_clarq_sweep_base.yaml',
        'feature_indices': [62124],
        'strength': 10.0,
        'hookpoint': 'model.layers.27',
        'module_path': 'model.layers.27',
        'sae_file': 'resid_post_layer_27/trainer_1',
    },
}

MATCHED_SECTIONS = (
    'model',
    'generation',
    'prompting',
    'provider_model',
    'provider_generation',
    'provider_prompting',
    'judge_model',
    'judge_generation',
    'judge_prompting',
)


def build_pair(model_key: str, specification: dict[str, Any]) -> tuple[dict, dict]:
    base = load_yaml(ROOT / specification['base'])
    baseline = copy.deepcopy(base)
    steered = copy.deepcopy(base)

    baseline['experiment_name'] = f'clarq_eval0to5_{model_key}_baseline_v1'
    steered['experiment_name'] = f'clarq_eval0to5_{model_key}_clarifysae_v1'
    for config in (baseline, steered):
        config['clarq']['evaluation_set'] = '0-5'
        config['clarq']['exclude_task_types_from_macro'] = [0]
        config['clarq']['write_html_report'] = True
        config['clarq']['unload_models_before_judge'] = True
        config['output']['root_dir'] = 'outputs/clarq_reviewer_eval_v1'

    baseline['steering'] = {'enabled': False}
    steered['steering']['enabled'] = True
    for key in (
        'feature_indices',
        'strength',
        'hookpoint',
        'module_path',
        'sae_id',
        'sae_file',
    ):
        if key in specification:
            steered['steering'][key] = specification[key]

    for section in MATCHED_SECTIONS:
        if baseline[section] != steered[section]:
            raise AssertionError(f'{model_key}: baseline and steering differ in {section}')
    return baseline, steered


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_key, specification in FINAL_CONFIGS.items():
        baseline, steered = build_pair(model_key, specification)
        dump_yaml(OUTPUT_DIR / f'{model_key}_baseline_eval0to5.yaml', baseline)
        dump_yaml(OUTPUT_DIR / f'{model_key}_clarifysae_eval0to5.yaml', steered)
    print(f'Wrote {len(FINAL_CONFIGS) * 2} configs to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
