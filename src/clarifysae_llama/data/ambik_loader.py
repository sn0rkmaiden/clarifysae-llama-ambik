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
