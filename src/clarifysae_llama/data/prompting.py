from __future__ import annotations

CLARIFICATION_TEMPLATE = """We: You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
We: Apart from that, in the kitchen there is <DESCRIPTION>.
We: You have received the task “<TASK>“.
Decide whether this task is ambiguous.
If it is ambiguous, generate one or more clarifying questions that would help you perform it correctly.

Return your final answer only as a JSON object in the following format:

{
  "ambiguous": true or false,
  "question": ["question 1", "question 2", ...]
}

Formatting rules:
- Always output exactly one JSON object.
- If the task is not ambiguous, use "ambiguous": false and "question": [].
- If the task is ambiguous, use "ambiguous": true and provide at least one question.
- Never output explanations or text outside the JSON.
- If unsure, make your best judgment and still output a valid JSON.

Example:
Task: "Put the red cup away."
Output:
{
  "ambiguous": true,
  "question": ["Where should I put the red cup?", "In which drawer or on the counter?"]
}

Now analyze the following task:
"<TASK>"
"""


def build_clarification_prompt(description: str, task: str) -> str:
    prompt = CLARIFICATION_TEMPLATE.replace('<DESCRIPTION>', description)
    prompt = prompt.replace('<TASK>', task)
    return prompt
