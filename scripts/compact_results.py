from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP_BASE = [
    'id',
    'ambiguity_type',
    'ambiguous_instruction',
    'gold_question',
    'gold_ambiguous',
    'predicted_ambiguous',
    'ambiguity_decision_correct',
    'model_questions',
    'num_questions',
    'asked_question',
    'model_question_first_similarity',
    'model_question_best_similarity',
    'resolved_proxy_first',
    'resolved_proxy_any',
    'json_parsed_output',
    'json_exact_valid',
    'json_schema_valid',
    'json_recoverable_parse',
    'raw_model_output',
    'raw_label_output',
    'raw_question_output',
    'raw_json_output',
]
KEEP_NLI = [
    'model_question_first_nli_similarity',
    'model_question_best_nli_similarity',
    'resolved_nli_first',
    'resolved_nli_any',
]


def compact_row(row: dict, *, include_nli: bool) -> dict:
    keys = list(KEEP_BASE)
    if include_nli:
        keys.extend(KEEP_NLI)
    compact = {}
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        compact[key] = value
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    parser.add_argument('--include-nli', action='store_true')
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text())
    examples = payload.get('examples', [])
    compact_examples = [compact_row(row, include_nli=args.include_nli) for row in examples]
    out = {'run_info': payload.get('run_info', {}), 'examples': compact_examples}
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
