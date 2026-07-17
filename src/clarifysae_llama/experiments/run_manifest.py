from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.runners.run_eval import run_eval
from clarifysae_llama.utils.io import append_jsonl, ensure_dir


def _selected_indices(
    n_rows: int,
    *,
    row_index: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> list[int]:
    if row_index is not None:
        if row_index < 0 or row_index >= n_rows:
            raise IndexError(f'row_index={row_index} is outside 0..{n_rows - 1}.')
        return [row_index]
    if shard_index is None and num_shards is None:
        return list(range(n_rows))
    if shard_index is None or num_shards is None:
        raise ValueError('--shard-index and --num-shards must be provided together.')
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError('Invalid shard specification.')
    return [index for index in range(n_rows) if index % num_shards == shard_index]


def run_manifest(
    manifest_path: str | Path,
    *,
    row_index: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    if 'config_json' not in manifest.columns or 'run_id' not in manifest.columns:
        raise ValueError('Manifest must contain run_id and config_json columns.')
    indices = _selected_indices(
        len(manifest),
        row_index=row_index,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    status_path = manifest_path.with_suffix('.execution.jsonl')
    ensure_dir(status_path.parent)
    results: list[dict[str, Any]] = []
    for index in indices:
        row = manifest.iloc[index]
        run_id = str(row['run_id'])
        config = json.loads(str(row['config_json']))
        try:
            result = run_eval(config)
            payload = {
                'row_index': index,
                'run_id': run_id,
                'status': 'complete',
                **result,
            }
            append_jsonl(status_path, payload)
            results.append(payload)
        except Exception as exc:
            payload = {
                'row_index': index,
                'run_id': run_id,
                'status': 'failed',
                'error_type': type(exc).__name__,
                'error': str(exc),
            }
            append_jsonl(status_path, payload)
            results.append(payload)
            if not continue_on_error:
                raise
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one row or a shard of a generated manifest.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--row-index', type=int, default=None)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--num-shards', type=int, default=None)
    parser.add_argument('--continue-on-error', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_manifest(
        args.manifest,
        row_index=args.row_index,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
