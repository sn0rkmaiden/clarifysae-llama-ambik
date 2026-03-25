from __future__ import annotations


def _base_context(description: str, task: str) -> str:
    return (
        'You are a robot operating in an office kitchen. '
        'You are in front of a counter with two closed drawers, a top one and a bottom one. '
        'There is also a landfill bin, a recycling bin, and a compost bin.\n'
        f'Apart from that, in the kitchen there is {description}.\n'
        f'You have received the task "{task}".\n'
    )


LABEL_TEMPLATE = (
    '{context}'
    'Decide whether this task is ambiguous.\n\n'
    'Answer with exactly one word:\n'
    '- AMBIGUOUS\n'
    '- CLEAR\n'
)


QUESTION_TEMPLATE = (
    '{context}'
    'If the task is ambiguous, ask up to {max_questions} short clarifying questions that would help you perform it correctly.\n'
    'If the task is not ambiguous, output NONE.\n\n'
    'Formatting rules:\n'
    '- Output only questions or NONE.\n'
    '- Put each question on its own line.\n'
    '- Do not output JSON.\n'
    '- Do not output explanations.\n'
    '- Ask at most {max_questions} questions.\n'
)


JSON_TEMPLATE = (
    '{context}'
    'Decide whether this task is ambiguous.\n'
    'If it is ambiguous, generate up to {max_questions} clarifying questions that would help you perform it correctly.\n\n'
    'Return your final answer only as a JSON object in the following format:\n\n'
    '{{\n'
    '  "ambiguous": true or false,\n'
    '  "question": ["question 1", "question 2", ...]\n'
    '}}\n\n'
    'Formatting rules:\n'
    '- Always output exactly one JSON object.\n'
    '- If the task is not ambiguous, use "ambiguous": false and "question": [].\n'
    '- If the task is ambiguous, use "ambiguous": true and provide between 1 and {max_questions} questions.\n'
    '- Never output explanations or text outside the JSON.\n'
    '- If unsure, make your best judgment and still output a valid JSON.\n\n'
    'Example:\n'
    'Task: "Put the red cup away."\n'
    'Output:\n'
    '{{\n'
    '  "ambiguous": true,\n'
    '  "question": ["Where should I put the red cup?", "In which drawer or on the counter?"]\n'
    '}}\n'
)


# Backward-compatible alias for the old single-prompt path.
CLARIFICATION_TEMPLATE = JSON_TEMPLATE



def build_ambiguity_prompt(description: str, task: str) -> str:
    return LABEL_TEMPLATE.format(context=_base_context(description, task))



def build_question_prompt(description: str, task: str, max_questions: int = 3) -> str:
    return QUESTION_TEMPLATE.format(
        context=_base_context(description, task),
        max_questions=max_questions,
    )



def build_json_compliance_prompt(description: str, task: str, max_questions: int = 3) -> str:
    return JSON_TEMPLATE.format(
        context=_base_context(description, task),
        max_questions=max_questions,
    )



def build_clarification_prompt(description: str, task: str, max_questions: int = 3) -> str:
    return build_json_compliance_prompt(description=description, task=task, max_questions=max_questions)
