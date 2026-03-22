from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    'id', 'environment_full', 'ambiguity_type', 'ambiguous_task', 'question', 'answer',
    'plan_for_amb_task', 'end_of_ambiguity', 'user_intent', 'plan_for_clear_task',
    'unambiguous_direct'
}


def load_ambik_no_help_dataset(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Be tolerant to common CSV export variants:
    # 1) pandas-saved index column called "Unnamed: 0"
    # 2) no explicit id column at all
    if 'id' not in df.columns:
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'id'})
        else:
            df.insert(0, 'id', range(len(df)))

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in dataset: {sorted(missing)}')

    amb = df[[
        'id', 'environment_full', 'ambiguity_type', 'ambiguous_task', 'question', 'answer',
        'plan_for_amb_task', 'end_of_ambiguity', 'user_intent'
    ]].copy()

    clear = df.copy()
    clear['ambiguity_type'] = 'unambiguous_direct'
    clear['ambiguous_task'] = clear['unambiguous_direct']
    clear['plan_for_amb_task'] = clear['plan_for_clear_task']

    merged = pd.concat([clear, amb], ignore_index=True)
    merged['plan'] = merged['plan_for_amb_task']
    merged['task'] = merged['ambiguous_task']

    keep_cols = [
        'id', 'environment_full', 'ambiguity_type', 'task', 'question', 'answer',
        'plan', 'end_of_ambiguity', 'user_intent'
    ]
    merged = merged[keep_cols].reset_index(drop=True)

    if limit is not None:
        merged = merged.head(limit).copy()

    return merged