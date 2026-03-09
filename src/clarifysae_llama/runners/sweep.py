from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

from clarifysae_llama.config import dump_yaml, load_yaml, set_by_dotted_path
from clarifysae_llama.runners.run_eval import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to sweep YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sweep_cfg = load_yaml(args.config)
    base_cfg = load_yaml(sweep_cfg['base_config'])
    parameter = sweep_cfg['sweep']['parameter']
    values = sweep_cfg['sweep']['values']

    for value in values:
        run_cfg = copy.deepcopy(base_cfg)
        set_by_dotted_path(run_cfg, parameter, value)
        suffix = str(value).replace(' ', '').replace('[', '').replace(']', '').replace(',', '-')
        run_cfg['experiment_name'] = f"{base_cfg['experiment_name']}__{parameter.replace('.', '_')}__{suffix}"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / 'run_config.yaml'
            dump_yaml(tmp_path, run_cfg)
            run_eval(run_cfg)
