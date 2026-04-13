from __future__ import annotations

import ast
import json
import re
from typing import Any


class BackendLLMAdapter:
    """Wraps a clarifysae backend and exposes the old `.request(...)` interface."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def _format_from_messages(self, messages: list[dict[str, str]], prompt: str) -> str:
        if hasattr(self.backend, "tokenizer") and getattr(self.backend.tokenizer, "chat_template", None):
            chat_messages = list(messages) + [{"role": "user", "content": prompt}]
            return self.backend.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").strip().lower()
            content = str(msg.get("content", ""))
            if role == "system":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append(f"User: {prompt}\nAssistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_json_substring(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    @classmethod
    def _coerce_json_text(cls, text: str) -> str:
        candidate = cls._extract_json_substring(text) or text.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                payload = loader(candidate)
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                pass

        cleaned = candidate.replace("'", '"')
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = re.sub(r"\bNone\b", "null", cleaned)
        try:
            payload = json.loads(cleaned)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return candidate.strip()

    def request(self, prompt: str, stop=None, **kwargs: Any):
        previous_message = kwargs.get("previous_message")
        json_format = bool(kwargs.get("json_format", False))

        if previous_message:
            prompt_text = self._format_from_messages(previous_message, prompt)
        else:
            prompt_text = prompt

        response = self.backend.generate(prompt_text)
        if json_format:
            response = self._coerce_json_text(response)
        return response, None
