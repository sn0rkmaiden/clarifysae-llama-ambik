from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def _normalize_task_text(value: Any) -> str:
    return re.sub(r'\s+', ' ', _clean_text(value)).casefold()


def _filter_by_split_manifest(
    df: pd.DataFrame,
    *,
    split_manifest: str | Path | None,
    split_name: str | None,
) -> pd.DataFrame:
    if split_manifest is None:
        if split_name is not None:
            raise ValueError("dataset.split_name requires dataset.split_manifest.")
        return df

    manifest = pd.read_csv(split_manifest, dtype={"example_id": str})
    required = {"example_id", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if split_name is None:
        raise ValueError("dataset.split_name is required when a split manifest is provided.")

    split_name = str(split_name)
    if split_name == "smoke20":
        if "is_smoke20" not in manifest.columns:
            raise ValueError(f"Split manifest {split_manifest} has no is_smoke20 column.")
        selected = manifest.loc[manifest["is_smoke20"].astype(bool)].copy()
    elif split_name == "pilot100":
        if "is_pilot100" not in manifest.columns:
            raise ValueError(f"Split manifest {split_manifest} has no is_pilot100 column.")
        selected = manifest.loc[manifest["is_pilot100"].astype(bool)].copy()
    else:
        selected = manifest.loc[manifest["split"].astype(str) == split_name].copy()
    if selected.empty:
        raise ValueError(f"Split {split_name!r} contains no rows in {split_manifest}.")
    selected_ids = selected["example_id"].astype(str)
    if selected_ids.duplicated().any():
        raise ValueError(f"Split {split_name!r} contains duplicate example IDs.")

    source = df.copy()
    source["__id_str"] = source["id"].astype(str)
    selected = selected.rename(columns={"example_id": "__id_str"})
    merged = selected.merge(source, on="__id_str", how="left", validate="one_to_one")
    missing_ids = merged.loc[merged["id"].isna(), "__id_str"].tolist()
    if missing_ids:
        raise ValueError(
            f"Split manifest references IDs absent from the dataset: {missing_ids[:10]}"
        )
    merged = merged.drop(columns=["__id_str"])
    return merged


def load_ambik_clarification_dataset(
    path: str | Path,
    limit: int | None = None,
    *,
    split_manifest: str | Path | None = None,
    split_name: str | None = None,
    instruction_variant: str = "ambiguous",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_id_column(df)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    variant = str(instruction_variant).strip().lower()
    if variant not in {"ambiguous", "clear", "paired"}:
        raise ValueError(
            "dataset.instruction_variant must be one of: ambiguous, clear, paired."
        )
    if variant in {"clear", "paired"} and "unambiguous_direct" not in df.columns:
        raise ValueError(
            "Clear/paired AmbiK evaluation requires the unambiguous_direct column."
        )

    df = _filter_by_split_manifest(
        df, split_manifest=split_manifest, split_name=split_name
    )

    common_cols = [
        "id",
        "environment_full",
        "ambiguity_type",
        "ambiguous_task",
        "question",
        "answer",
        "plan_for_clear_task",
    ]
    if "unambiguous_direct" in df.columns:
        common_cols.append("unambiguous_direct")
    result = df[common_cols].copy().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        source_id = _clean_text(row["id"])
        base = {
            "source_id": source_id,
            "environment_full": _clean_text(row["environment_full"]),
            "source_ambiguity_type": _clean_text(row["ambiguity_type"]),
            "answer": _clean_text(row.get("answer", "")),
            "plan_for_clear_task": _clean_text(row.get("plan_for_clear_task", "")),
        }
        if variant in {"ambiguous", "paired"}:
            rows.append({
                **base,
                "id": f"{source_id}:ambiguous" if variant == "paired" else source_id,
                "variant": "ambiguous",
                "ambiguity_type": _clean_text(row["ambiguity_type"]),
                "task": _clean_text(row["ambiguous_task"]),
                "ambiguous_task": _clean_text(row["ambiguous_task"]),
                "question": _clean_text(row.get("question", "")),
            })
        if variant in {"clear", "paired"}:
            clear_task = _clean_text(row.get("unambiguous_direct", ""))
            rows.append({
                **base,
                "id": f"{source_id}:clear" if variant == "paired" else source_id,
                "variant": "clear",
                "ambiguity_type": "unambiguous_direct",
                "task": clear_task,
                "ambiguous_task": clear_task,
                "question": "",
            })

    output = pd.DataFrame(rows)
    if limit is not None:
        output = output.head(limit).copy()
    return output.reset_index(drop=True)


# Backward-compatible alias used by existing configs / imports.
load_ambik_no_help_dataset = load_ambik_clarification_dataset


def load_ambik_selective_dataset(
    path: str | Path,
    limit_pairs: int | None = None,
    *,
    include_unambiguous_pairs: bool = True,
    split_manifest: str | Path | None = None,
    split_name: str | None = None,
) -> pd.DataFrame:
    """Load AmbiK as a mixed ambiguous/clear selective-clarification set.

    Each source row contributes its ``ambiguous_task``. When
    ``include_unambiguous_pairs`` is true, the paired ``unambiguous_direct``
    instruction is also included.

    Some AmbiK rows contain textually identical ambiguous and clear variants.
    Both variants are retained for oracle-gated question-generation analysis,
    but they are marked ``classification_eligible=False`` so that a
    deterministic classifier is not evaluated against contradictory labels for
    the same input.
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

    df = _filter_by_split_manifest(
        df, split_manifest=split_manifest, split_name=split_name
    )

    if limit_pairs is not None:
        df = df.head(limit_pairs).copy()

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        source_id = _clean_text(row['id'])
        ambiguous_task = _clean_text(row['ambiguous_task'])
        clear_task = _clean_text(row.get('unambiguous_direct', ''))
        pair_texts_identical = bool(
            include_unambiguous_pairs
            and _normalize_task_text(ambiguous_task) == _normalize_task_text(clear_task)
        )
        classification_eligible = bool(
            include_unambiguous_pairs and not pair_texts_identical
        )
        source_ambiguity_type = _clean_text(row['ambiguity_type'])

        common = {
            'source_id': source_id,
            'environment_full': _clean_text(row['environment_full']),
            'source_ambiguity_type': source_ambiguity_type,
            'gold_answer': _clean_text(row.get('answer', '')),
            'gold_plan_for_clear': _clean_text(row.get('plan_for_clear_task', '')),
            'pair_texts_identical': pair_texts_identical,
            'classification_eligible': classification_eligible,
        }
        rows.append({
            **common,
            'id': f'{source_id}:ambiguous',
            'variant': 'ambiguous',
            'task': ambiguous_task,
            'ambiguity_type': source_ambiguity_type,
            'gold_ambiguous': True,
            'gold_question': _clean_text(row.get('question', '')),
        })
        if include_unambiguous_pairs:
            rows.append({
                **common,
                'id': f'{source_id}:clear',
                'variant': 'clear',
                'task': clear_task,
                'ambiguity_type': 'unambiguous_direct',
                'gold_ambiguous': False,
                'gold_question': '',
            })

    return pd.DataFrame(rows)
