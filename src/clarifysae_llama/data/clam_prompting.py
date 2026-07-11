from __future__ import annotations

from typing import Any


CLASSIFICATION_HEADER = """You are evaluating instructions for a kitchen robot.
Decide whether the instruction is ambiguous in the described environment.

AMBIGUOUS means that multiple plausible task executions remain and the missing choice should be requested from the user before acting.
CLEAR means that the intended execution is sufficiently specified by the instruction, environment, and ordinary task knowledge.

Return only one label: AMBIGUOUS or CLEAR.
"""


QUESTION_HEADER = """You are a kitchen robot. The instruction below has been classified as ambiguous.
Ask exactly one concise clarification question that identifies the missing choice needed to execute the task correctly.
Ground the question in the described environment and mention concrete alternatives when useful.
Return only the question. Do not add an answer, explanation, list, numbering, or JSON.
"""


def _format_case(environment: str, task: str) -> str:
    return f"Environment: {environment}\nInstruction: {task}"


def build_clam_classification_prompt(
    *,
    environment: str,
    task: str,
    demonstrations: list[dict[str, Any]],
) -> str:
    parts = [CLASSIFICATION_HEADER]
    for demo in demonstrations:
        label = str(demo['label']).strip().upper()
        if label not in {'AMBIGUOUS', 'CLEAR'}:
            raise ValueError(f"Unsupported CLAM demonstration label: {label!r}")
        parts.append(
            _format_case(str(demo['environment']), str(demo['task']))
            + f"\nLabel: {label}\n"
        )
    parts.append(_format_case(environment, task) + "\nLabel:")
    return "\n".join(parts)


def build_clam_question_prompt(
    *,
    environment: str,
    task: str,
    demonstrations: list[dict[str, Any]],
) -> str:
    parts = [QUESTION_HEADER]
    for demo in demonstrations:
        if str(demo.get('label', '')).strip().upper() != 'AMBIGUOUS':
            continue
        question = str(demo.get('question', '')).strip()
        if not question:
            continue
        parts.append(
            _format_case(str(demo['environment']), str(demo['task']))
            + f"\nClarification question: {question}\n"
        )
    parts.append(_format_case(environment, task) + "\nClarification question:")
    return "\n".join(parts)
