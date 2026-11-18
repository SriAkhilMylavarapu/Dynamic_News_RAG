"""LLM client: Claude via the Anthropic SDK. Anything with `generate`/`stream` will do."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

log = logging.getLogger(__name__)

# On a safety-classifier decline, the API re-runs the request on Anthropic's
# recommended fallback model for that refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSED = "(The model declined to answer this one. Try rephrasing the question.)"


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]: ...


class ClaudeLLM:
    def __init__(self, model: str = "claude-sonnet-5-5", effort: str = "low", api_key: str | None = None):
        import anthropic

        # With api_key=None the SDK reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort

    def _params(self, system: str, prompt: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            # Answering from supplied passages doesn't need deep reasoning; `low`
            # keeps answers fast and cheap. Raise NEWSRAG_LLM_EFFORT if quality suffers.
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def generate(self, system: str, prompt: str) -> str:
        msg = self.client.beta.messages.create(**self._params(system, prompt, 16000))
        if msg.stop_reason == "refusal":
            log.warning("refusal: %s", getattr(msg.stop_details, "category", None))
            return REFUSED
        return "".join(b.text for b in msg.content if b.type == "text")

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        with self.client.beta.messages.stream(**self._params(system, prompt, 64000)) as s:
            yield from s.text_stream
            msg = s.get_final_message()
        if msg.stop_reason == "refusal":
            log.warning("refusal: %s", getattr(msg.stop_details, "category", None))
            yield "\n\n" + REFUSED
