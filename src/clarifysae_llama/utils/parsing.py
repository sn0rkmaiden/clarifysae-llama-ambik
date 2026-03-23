from __future__ import annotations

import json
import re
from typing import Any


def _strip_fences_and_eos(text: str) -> str:
    text = (text or '').replace('<eos>', '').strip()

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.I)
    if fenced:
        return fenced.group(1).strip()

    text = re.sub(r"```(?:json)?", '', text, flags=re.I)
    text = text.replace('```', '')
    return text.strip()


def _first_curly_to_end(text: str) -> str | None:
    start = text.find('{')
    return text[start:] if start != -1 else None


def _extract_first_balanced_json_object(text: str) -> str | None:
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaping = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escaping:
                escaping = False
            elif ch == '\\':
                escaping = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


def _balance_closers(text: str) -> str:
    in_string = False
    escaping = False
    braces = 0
    brackets = 0

    for ch in text:
        if ch == '"' and not escaping:
            in_string = not in_string
        escaping = (ch == '\\' and not escaping) if in_string else False
        if in_string:
            continue
        if ch == '{':
            braces += 1
        elif ch == '}':
            braces = max(0, braces - 1)
        elif ch == '[':
            brackets += 1
        elif ch == ']':
            brackets = max(0, brackets - 1)

    text = text.rstrip()
    if brackets > 0:
        text += ']' * brackets
    if braces > 0:
        text += '}' * braces
    return text


def _scan_string_list(text: str, start_idx: int) -> tuple[list[str], int]:
    items: list[str] = []
    current: list[str] = []
    in_string = False
    escaping = False
    idx = start_idx

    while idx < len(text):
        ch = text[idx]

        if in_string:
            if escaping:
                current.append(ch)
                escaping = False
            elif ch == '\\':
                escaping = True
            elif ch == '"':
                in_string = False
                items.append(''.join(current))
                current = []
            else:
                current.append(ch)
            idx += 1
            continue

        if ch == '"':
            in_string = True
            idx += 1
            continue
        if ch == ',':
            if current:
                items.append(''.join(current))
                current = []
            idx += 1
            continue
        if ch == ']':
            if current:
                items.append(''.join(current))
            idx += 1
            break
        if ch in '{}':
            if current:
                items.append(''.join(current))
            break
        idx += 1

    return items, idx


def _schema_parse_fallback(text: str) -> dict[str, Any]:
    ambiguous_match = re.search(r'"ambiguous"\s*:\s*(true|false)', text, flags=re.I)
    ambiguous = ambiguous_match is not None and ambiguous_match.group(1).lower() == 'true'

    questions: list[str] = []
    question_start = re.search(r'"question"\s*:\s*\[', text, flags=re.I)
    if question_start:
        questions, _ = _scan_string_list(text, question_start.end())

    return {
        'ambiguous': ambiguous,
        'question': questions,
    }


def parse_model_json(raw_output: str) -> dict[str, Any] | None:
    body = _strip_fences_and_eos(raw_output)
    candidate = _extract_first_balanced_json_object(body)
    if candidate is None:
        candidate = _first_curly_to_end(body) or body

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    repaired = re.sub(r',(\s*[}\]])', r'\1', candidate)
    repaired = re.sub(r'\bTrue\b', 'true', repaired)
    repaired = re.sub(r'\bFalse\b', 'false', repaired)
    repaired = re.sub(r'\bNone\b', 'null', repaired)
    repaired = _balance_closers(repaired)

    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    return _schema_parse_fallback(repaired)
