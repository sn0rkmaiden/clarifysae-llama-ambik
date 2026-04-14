from __future__ import annotations

from typing import Any, Iterable


class BackendLLMAdapter:
    """
    Thin adapter that makes the current HF backend look like the old ClarQ code's
    expected LLM interface.

    Important behavior:
    - For normal dialogue turns, truncate generation to a *single seeker turn*.
    - For json_format=True, keep only the JSON-ish object text.
    - Respect stop sequences when possible.
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def _extract_first_braced_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        quote_char = ""
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        return None

    def _coerce_json_text(self, text: str) -> str:
        """
        Keep only the most likely JSON/object span.
        This is for provider/judge prompts that expect a structured response.
        """
        text = (text or "").strip()
        if not text:
            return "{}"

        block = self._extract_first_braced_object(text)
        if block is not None:
            return block.strip()

        return text

    def _apply_stop(self, text: str, stop: str | Iterable[str] | None) -> str:
        if not text or stop is None:
            return text

        if isinstance(stop, str):
            stops = [stop]
        else:
            stops = [s for s in stop if s]

        cut = len(text)
        for marker in stops:
            idx = text.find(marker)
            if idx != -1:
                cut = min(cut, idx)

        return text[:cut]

    def _truncate_to_single_turn(self, text: str) -> str:
        """
        The seeker prompt ends with `You:` and local Llama often continues the whole
        transcript (`Jax: ...`, `You: ...`, notes, markdown, etc.).
        We keep only the first seeker utterance.
        """
        text = (text or "").strip()

        markers = [
            "\nJax:",
            "\nYou:",
            "\nUser:",
            "\nAssistant:",
            "\nHuman:",
            "\nPlease wait",
            "\n(Note:",
            "\n## ",
            "```",
        ]

        cut = len(text)
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                cut = min(cut, idx)

        text = text[:cut].strip()

        # Clean common prefixes
        prefixes = ["You:", "User:", "Assistant:", "Human:"]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    changed = True

        # Strip wrapping quotes once
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            text = text[1:-1].strip()

        return text

    def request(
        self,
        prompt_text: str,
        stop: str | Iterable[str] | None = None,
        json_format: bool = False,
    ):
        """
        Returns (response_text, metadata) to match the legacy ClarQ interface.
        """
        try:
            response = self.backend.generate(prompt_text, stop=stop)
        except TypeError:
            # Current backend.generate usually does not accept stop.
            response = self.backend.generate(prompt_text)

        response = "" if response is None else str(response)

        # First apply explicit stop strings, if any were provided.
        response = self._apply_stop(response, stop)

        if json_format:
            response = self._coerce_json_text(response)
        else:
            response = self._truncate_to_single_turn(response)

        return response, None