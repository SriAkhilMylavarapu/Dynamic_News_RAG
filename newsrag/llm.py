"""LLM client. Gemini by default; anything with `generate`/`stream` will do."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]: ...


class GeminiLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _config(self, system: str):
        return self._types.GenerateContentConfig(system_instruction=system, temperature=self.temperature)

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=self._config(system))
        return resp.text or ""

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        for part in self.client.models.generate_content_stream(model=self.model, contents=prompt, config=self._config(system)):
            if part.text:
                yield part.text
