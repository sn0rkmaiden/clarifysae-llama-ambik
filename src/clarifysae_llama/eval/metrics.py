from __future__ import annotations

import numpy as np
import pandas as pd


AMBIGUITY_TYPES = ['unambiguous_direct', 'preferences', 'common_sense_knowledge', 'safety']


def safe_mean(arr) -> float:
    if len(arr) == 0:
        return -1.0
    return float(np.mean(arr))


def parse_intents(y_amb_intents):
    if isinstance(y_amb_intents, str):
        return [item.strip() for item in y_amb_intents.split(',') if item.strip()]
    if isinstance(y_amb_intents, list):
        return y_amb_intents
    return []


def split_answers(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # keep the original behavior close to the old repo: more than one line/question counts as asking for clarification
    lines = [line.strip('-• 	') for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [text]


def success_rate(llm_answers: list[str], y_amb_intents: list[str], y_amb_type: str) -> float:
    if not y_amb_intents or not llm_answers:
        return -1.0

    values = []
    total = len(y_amb_intents)
    for answer in llm_answers:
        answer = answer.lower()
        success_counter = 0
        for el in y_amb_intents:
            if el.startswith('-'):
                variants = el.replace('-', '').split('|')
                flag = any(var in answer for var in variants)
                if not flag:
                    success_counter += 1
            else:
                variants = el.split('|')
                flag = any(var in answer for var in variants)
                if flag:
                    success_counter += 1
        values.append(success_counter / total)
    return max(values) if values else 0.0


def help_rate(llm_answers: list[str]) -> int:
    return int(len(llm_answers) > 1)


def correct_help_rate(llm_answers: list[str], amb_type: str) -> int:
    if 'unambiguous' in amb_type:
        return int(len(llm_answers) == 1)
    return int(len(llm_answers) > 1)


def compute_example_metrics(prediction_text: str, ambiguity_type: str, user_intent) -> dict:
    answers = split_answers(prediction_text)
    intents = parse_intents(user_intent)
    return {
        'prediction_text': prediction_text,
        'y_amb_type': ambiguity_type,
        'y_amb_intents': intents,
        'SR': success_rate(answers, intents, ambiguity_type),
        'help_rate': help_rate(answers),
        'correct_help_rate': correct_help_rate(answers, ambiguity_type),
    }


def aggregate_metrics(example_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ambiguity_type in AMBIGUITY_TYPES:
        subset = example_metrics.loc[example_metrics['y_amb_type'] == ambiguity_type]
        sr_rates = np.asarray(subset['SR'])
        sr = safe_mean(sr_rates[sr_rates >= 0])
        chr_rates = np.asarray(subset['correct_help_rate'])
        chr_value = safe_mean(chr_rates[chr_rates >= 0])
        help_rates = np.asarray(subset['help_rate'])
        help_value = float(np.sum(help_rates[help_rates >= 0]) / len(help_rates)) if len(help_rates) else -1.0
        rows.append({
            'ambiguity_type': ambiguity_type,
            'sr_agg': sr,
            'correct_help_rate_agg': chr_value,
            'help_rate_agg': help_value,
            'n_examples': int(len(subset)),
        })
    return pd.DataFrame(rows)
