from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    'id',
    'environment_full',
    'ambiguity_type',
    'ambiguous_task',
    'question',
    'answer',
}

OPTIONAL_COLUMNS = {
    'plan_for_clear_task',
}


def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in df.columns:
        return df
    if 'Unnamed: 0' in df.columns:
        return df.rename(columns={'Unnamed: 0': 'id'})
    df = df.copy()
    df.insert(0, 'id', range(len(df)))
    return df


def load_ambik_clarification_dataset(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_id_column(df)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in dataset: {sorted(missing)}')

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ''

    keep_cols = [
        'id',
        'environment_full',
        'ambiguity_type',
        'ambiguous_task',
        'question',
        'answer',
        'plan_for_clear_task',
    ]
    result = df[keep_cols].copy().reset_index(drop=True)

    if limit is not None:
        result = result.head(limit).copy()

    return result


# Backward-compatible alias used by existing configs / imports.
load_ambik_no_help_dataset = load_ambik_clarification_dataset


def load_ambik_selective_dataset(
    path: str | Path,
    limit_pairs: int | None = None,
    *,
    include_unambiguous_pairs: bool = True,
) -> pd.DataFrame:
    """Load AmbiK as a mixed ambiguous/clear selective-clarification set.

    Each source row contributes its ``ambiguous_task``. When
    ``include_unambiguous_pairs`` is true, the paired ``unambiguous_direct``
    instruction is also included. This is required to evaluate CLAM's
    selectivity and over-asking behavior rather than only question quality on
    already-known ambiguous inputs.
    """
    df = pd.read_csv(path)
    df = _ensure_id_column(df)

    required = set(REQUIRED_COLUMNS)
    if include_unambiguous_pairs:
        required.add('unambiguous_direct')
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            'Missing columns required for CLAM selective evaluation: '
            f'{sorted(missing)}. Use the paired AmbiK CSV, or set '
            'dataset.include_unambiguous_pairs=false for an ambiguous-only '
            'diagnostic run.'
        )

    if limit_pairs is not None:
        df = df.head(limit_pairs).copy()

    rows: list[dict] = []
    for _, row in df.iterrows():
        source_id = str(row['id'])
        common = {
            'source_id': source_id,
            'environment_full': str(row['environment_full']),
            'gold_answer': str(row.get('answer', '') or ''),
            'gold_plan_for_clear': str(row.get('plan_for_clear_task', '') or ''),
        }
        rows.append({
            **common,
            'id': f'{source_id}:ambiguous',
            'variant': 'ambiguous',
            'task': str(row['ambiguous_task']),
            'ambiguity_type': str(row['ambiguity_type']),
            'gold_ambiguous': True,
            'gold_question': str(row.get('question', '') or ''),
        })
        if include_unambiguous_pairs:
            rows.append({
                **common,
                'id': f'{source_id}:clear',
                'variant': 'clear',
                'task': str(row['unambiguous_direct']),
                'ambiguity_type': 'unambiguous_direct',
                'gold_ambiguous': False,
                'gold_question': '',
            })

    return pd.DataFrame(rows)
