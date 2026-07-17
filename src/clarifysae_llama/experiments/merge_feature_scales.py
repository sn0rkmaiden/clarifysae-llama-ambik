from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from clarifysae_llama.utils.io import ensure_dir


def merge_feature_scales(inputs: list[str], output: str | Path) -> pd.DataFrame:
    if not inputs:
        raise ValueError('At least one scale part is required.')
    frames: list[pd.DataFrame] = []
    for pattern in inputs:
        matches = [Path(value) for value in sorted(glob.glob(pattern))]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        if not matches:
            raise FileNotFoundError(f'No feature-scale files matched {pattern!r}.')
        for path in matches:
            frame = pd.read_csv(path)
            frame['source_part'] = str(path)
            frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    keys = ['model_key', 'layer', 'feature_id']
    if merged.duplicated(keys).any():
        duplicates = merged.loc[merged.duplicated(keys, keep=False), keys + ['source_part']]
        raise ValueError(f'Duplicate feature scale keys across parts:\n{duplicates.to_string(index=False)}')
    if (pd.to_numeric(merged['selected_scale'], errors='coerce') <= 0).any() or merged['selected_scale'].isna().any():
        bad = merged.loc[
            pd.to_numeric(merged['selected_scale'], errors='coerce').isna()
            | (pd.to_numeric(merged['selected_scale'], errors='coerce') <= 0)
        ]
        raise ValueError(
            'All enabled features need a positive selected_scale before manifest creation:\n'
            + bad[['model_key', 'layer', 'feature_id', 'status', 'source_part']].to_string(index=False)
        )
    merged = merged.drop(columns=['source_part']).sort_values(keys).reset_index(drop=True)
    output = Path(output)
    ensure_dir(output.parent)
    merged.to_csv(output, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge independently collected feature-scale parts.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = merge_feature_scales(args.inputs, args.output)
    print(frame.groupby(['model_key', 'layer']).size())
    print(f'Wrote {len(frame)} feature scales to {args.output}')


if __name__ == '__main__':
    main()
