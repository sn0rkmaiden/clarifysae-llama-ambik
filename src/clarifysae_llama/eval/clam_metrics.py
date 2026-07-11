from __future__ import annotations

from typing import Any

import pandas as pd


def binary_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute AUROC from ranks without adding a scikit-learn dependency."""
    if len(labels) != len(scores):
        raise ValueError('labels and scores must have the same length')
    n_pos = sum(bool(label) for label in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = avg_rank
        start = end

    sum_pos_ranks = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def add_clam_aggregate_metrics(
    aggregate_df: pd.DataFrame,
    example_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if aggregate_df.empty or example_metrics.empty:
        return aggregate_df

    rows = example_metrics.to_dict(orient='records')
    ambiguous_rows = [row for row in rows if bool(row['gold_ambiguous'])]
    clear_rows = [row for row in rows if not bool(row['gold_ambiguous'])]

    selective_success_values: list[float] = []
    for row in rows:
        if bool(row['gold_ambiguous']):
            selective_success_values.append(1.0 if bool(row['resolved_proxy_any']) else 0.0)
        else:
            selective_success_values.append(1.0 if not bool(row['asked_question']) else 0.0)

    result = aggregate_df.copy()
    result.loc[0, 'clam_selective_success'] = sum(selective_success_values) / len(selective_success_values)
    result.loc[0, 'resolved_proxy_rate_ambiguous'] = (
        sum(1 for row in ambiguous_rows if bool(row['resolved_proxy_any'])) / len(ambiguous_rows)
        if ambiguous_rows else None
    )
    result.loc[0, 'question_similarity_asked_ambiguous'] = (
        sum(float(row['model_question_best_similarity']) for row in ambiguous_rows if bool(row['asked_question']))
        / sum(1 for row in ambiguous_rows if bool(row['asked_question']))
        if any(bool(row['asked_question']) for row in ambiguous_rows) else None
    )
    result.loc[0, 'oracle_gate_resolved_proxy_rate_ambiguous'] = (
        sum(1 for row in ambiguous_rows if bool(row.get('oracle_gate_resolved_proxy_any'))) / len(ambiguous_rows)
        if ambiguous_rows else None
    )
    result.loc[0, 'oracle_gate_question_similarity_ambiguous'] = (
        sum(float(row.get('oracle_gate_question_similarity', 0.0)) for row in ambiguous_rows) / len(ambiguous_rows)
        if ambiguous_rows else None
    )
    result.loc[0, 'classification_auroc'] = binary_auroc(
        [bool(row['gold_ambiguous']) for row in rows],
        [float(row['ambiguity_probability']) for row in rows],
    )
    result.loc[0, 'n_ambiguous'] = len(ambiguous_rows)
    result.loc[0, 'n_clear'] = len(clear_rows)
    return result


def clean_single_question(raw_output: Any) -> str:
    text = str(raw_output or '').strip()
    if not text:
        return ''

    for prefix in ('```text', '```', 'Question:', 'Clarification question:', 'Answer:'):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), '')
    first_line = first_line.lstrip('-*• ').strip()
    while first_line and first_line[0].isdigit():
        first_line = first_line[1:].lstrip('.): ').strip()
    first_line = first_line.strip('`"\' ')

    if '?' in first_line:
        first_line = first_line[: first_line.index('?') + 1]
    return first_line.strip()
