from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clarifysae_llama.experiments.fingerprint import fingerprint_payload
from clarifysae_llama.utils.io import append_jsonl, ensure_dir


def assert_run_config_compatible(
    run_dir: str | Path,
    config: dict[str, Any],
) -> None:
    run_dir = Path(run_dir)
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return

    fingerprint_path = run_dir / 'run_fingerprint.json'
    if not fingerprint_path.exists():
        raise RuntimeError(
            f'Refusing to reuse non-empty ClarQ run directory without provenance: {run_dir}'
        )

    existing = json.loads(fingerprint_path.read_text(encoding='utf-8'))
    existing_fingerprint = existing.get('config_fingerprint')
    requested_fingerprint = fingerprint_payload(config)
    if existing_fingerprint != requested_fingerprint:
        raise RuntimeError(
            'Refusing to reuse ClarQ run directory with a different resolved configuration: '
            f'{run_dir}'
        )


def append_dialogue_checkpoint(
    path: str | Path,
    *,
    task_type_index: int,
    task_type_name: str,
    dialogue_index: int,
    conversation: list[str],
) -> None:
    append_jsonl(
        path,
        {
            'task_type_index': int(task_type_index),
            'task_type_name': str(task_type_name),
            'dialogue_index': int(dialogue_index),
            'conversation': list(conversation),
        },
    )


def load_dialogue_checkpoints(
    path: str | Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}

    checkpoints: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (int(row['task_type_index']), int(row['dialogue_index']))
                if not isinstance(row.get('conversation'), list):
                    raise ValueError('conversation is not a list')
            except Exception as error:
                raise ValueError(
                    f'Invalid ClarQ dialogue checkpoint at {path}:{line_number}'
                ) from error
            checkpoints[key] = row
    return checkpoints


def checkpoint_path_for_run(run_dir: str | Path) -> Path:
    return ensure_dir(Path(run_dir) / 'checkpoints') / 'dialogues.jsonl'
