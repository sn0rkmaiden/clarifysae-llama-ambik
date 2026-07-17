from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.utils.io import ensure_dir, write_json


def _run_metrics(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config['output']['root_dir']) / config['experiment_name'] / 'metrics' / 'example_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing example metrics for completed exploration run: {path}')
    return pd.read_csv(path)


def _summary(metrics: pd.DataFrame) -> dict[str, float]:
    if 'variant' not in metrics.columns:
        metrics = metrics.copy()
        metrics['variant'] = metrics['ambiguity_type'].map(
            lambda value: 'clear' if str(value) == 'unambiguous_direct' else 'ambiguous'
        )
    ambiguous = metrics.loc[metrics['variant'].astype(str) == 'ambiguous'].copy()
    clear = metrics.loc[metrics['variant'].astype(str) == 'clear'].copy()
    if ambiguous.empty or clear.empty:
        raise ValueError('Alpha selection requires paired ambiguous and clear examples in every run.')

    category_scores = ambiguous.groupby('ambiguity_type')['resolved_proxy_first'].mean()
    return {
        'macro_first_resolution': float(category_scores.mean()),
        'mean_first_similarity': float(ambiguous['model_question_first_similarity'].mean()),
        'avg_questions_ambiguous': float(ambiguous['num_questions'].mean()),
        'clear_overasking': float(clear['asked_question'].astype(float).mean()),
    }


def select_alpha(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    clear_tolerance: float = 0.05,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    summaries: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        config = json.loads(str(row['config_json']))
        stats = _summary(_run_metrics(config))
        summaries.append({
            'run_id': str(row['run_id']),
            'model_key': str(row['model_key']),
            'layer': None if pd.isna(row['layer']) else int(row['layer']),
            'feature_id': None if pd.isna(row['feature_id']) else int(row['feature_id']),
            'vocab_membership': None if pd.isna(row.get('vocab_membership')) else str(row['vocab_membership']),
            'alpha': None if pd.isna(row['alpha']) else float(row['alpha']),
            **stats,
        })
    summary = pd.DataFrame(summaries)

    baseline = summary.loc[summary['feature_id'].isna(), ['model_key', 'clear_overasking']].rename(
        columns={'clear_overasking': 'baseline_clear_overasking'}
    )
    if baseline['model_key'].duplicated().any():
        raise ValueError('Expected exactly one baseline per model in exploration manifest.')

    candidates = summary.loc[summary['feature_id'].notna()].merge(
        baseline, on='model_key', how='left', validate='many_to_one'
    )
    if candidates['baseline_clear_overasking'].isna().any():
        raise ValueError('Missing baseline for one or more models.')
    candidates['clear_constraint'] = (
        candidates['clear_overasking']
        <= candidates['baseline_clear_overasking'] + float(clear_tolerance)
    )
    candidates['clear_excess'] = (
        candidates['clear_overasking'] - candidates['baseline_clear_overasking']
    ).clip(lower=0)
    candidates['abs_alpha'] = candidates['alpha'].abs()

    selected_rows: list[pd.Series] = []
    for _key, group in candidates.groupby(['model_key', 'layer', 'feature_id'], sort=True):
        feasible = group.loc[group['clear_constraint']].copy()
        if not feasible.empty:
            ranked = feasible.sort_values(
                [
                    'macro_first_resolution',
                    'mean_first_similarity',
                    'avg_questions_ambiguous',
                    'abs_alpha',
                ],
                ascending=[False, False, True, True],
            )
            selection_reason = 'satisfies_clear_constraint'
        else:
            ranked = group.sort_values(
                [
                    'clear_excess',
                    'macro_first_resolution',
                    'mean_first_similarity',
                    'avg_questions_ambiguous',
                    'abs_alpha',
                ],
                ascending=[True, False, False, True, True],
            )
            selection_reason = 'no_feasible_alpha_minimized_clear_excess'
        chosen = ranked.iloc[0].copy()
        chosen['selection_reason'] = selection_reason
        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows)[[
        'model_key', 'layer', 'feature_id', 'vocab_membership', 'alpha',
        'macro_first_resolution', 'mean_first_similarity', 'avg_questions_ambiguous',
        'clear_overasking', 'baseline_clear_overasking', 'clear_excess',
        'clear_constraint', 'selection_reason', 'run_id',
    ]].sort_values(['model_key', 'layer', 'feature_id']).reset_index(drop=True)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    selected.to_csv(output_path, index=False)
    write_json(output_path.with_suffix('.metadata.json'), {
        'source_manifest': str(manifest_path),
        'clear_tolerance': clear_tolerance,
        'num_selected_features': len(selected),
        'selection_rule': [
            'maximize macro first-question proxy resolution subject to clear overasking <= baseline + tolerance',
            'tie-break: higher mean first similarity',
            'tie-break: fewer ambiguous questions',
            'tie-break: smaller absolute alpha',
            'when infeasible: minimize clear excess before the same quality tie-breaks',
        ],
    })
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Freeze one normalized alpha per feature from explore results.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='artifacts/selection/ambik_explore_v1_selected_alpha.csv')
    parser.add_argument('--clear-tolerance', type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_alpha(args.manifest, args.output, clear_tolerance=args.clear_tolerance)
    print(selected.to_string(index=False))


if __name__ == '__main__':
    main()
