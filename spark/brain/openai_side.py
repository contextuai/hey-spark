"""Tool-less helper requests on an OpenAI-compatible provider (P5.7e).

* :class:`OpenAIClassifier`: the router's optional fast-model classifier
  (``routing.classifier: fast_model``) on ``brain.openai_compat``: one tool-less request to the
  fast model, the same prompt as on Claude, one word back.
* :class:`OpenAISideModel`: the same interface as :class:`spark.brain.side.SideModel`
  (``warmup`` / ``ask`` / ``aclose``) for side questions that must never enter the user's
  conversation ("was that meant for Spark?", P5.4). Each question is a fresh request, so there
  is no side context to grow or reset.

:func:`spark.brain.side.make_side_model` picks the right one for the configured backend.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from spark.brain.router import CLASSIFIER_PROMPT

if TYPE_CHECKING:
    from spark.brain.openai_chat import ChatClient
    from spark.config import Settings

__all__ = ["OpenAIClassifier", "OpenAISideModel"]

log = logging.getLogger("spark.brain.openai")

_LABEL = re.compile(r"\b(fast|deep)\b")
#: Room for a reasoning model's hidden tokens before its one-word answer.
CLASSIFIER_MAX_TOKENS = 64


class OpenAIClassifier:
    """``RouteClassifier`` on the fast model of an OpenAI-compatible endpoint."""

    def __init__(self, chat: ChatClient, model: Callable[[], str]) -> None:
        self.chat = chat
        self.model = model
        self.calls = 0

    async def classify(self, text: str) -> str | None:
        self.calls += 1
        reply = await self.chat.complete(
            self.model(),
            [
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Request: {text}"},
            ],
            max_tokens=CLASSIFIER_MAX_TOKENS,
        )
        match = _LABEL.search(reply.lower())
        return match.group(1) if match else None

    async def warmup(self) -> None:
        return None  # nothing to connect

    async def aclose(self) -> None:
        return None  # the brain owns (and closes) the chat client


class OpenAISideModel:
    """One-off questions to ``brain.openai_compat``'s fast model with a fixed system prompt."""

    def __init__(
        self,
        settings: Settings,
        system_prompt: str,
        *,
        chat: ChatClient | None = None,
        model: str | None = None,
        name: str = "side",
        transport: Any | None = None,
        max_tokens: int = 256,
    ) -> None:
        from spark.brain.openai_presets import resolve_provider

        provider = resolve_provider(settings.brain.openai_compat)
        self.settings = settings
        self.system_prompt = system_prompt
        self.model = model or provider.fast_model or ""
        self.name = name
        self.max_tokens = max_tokens
        self.calls = 0
        self._owns_chat = chat is None
        if chat is None:
            from spark.brain.openai_chat import ChatClient

            chat = ChatClient(provider, settings.brain.openai_compat, transport=transport)
        self.chat = chat

    async def warmup(self) -> None:
        return None  # an HTTP endpoint: nothing to spawn

    async def ask(self, prompt: str) -> str:
        """The model's whole text reply to ``prompt`` ('' for an empty prompt)."""
        if not prompt:
            return ""
        self.calls += 1
        return await self.chat.complete(
            self.model,
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
        )

    async def aclose(self) -> None:
        if self._owns_chat:
            await self.chat.aclose()
