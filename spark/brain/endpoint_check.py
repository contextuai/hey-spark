"""Checking an OpenAI-compatible endpoint: shared by ``spark doctor`` and the Brain window.

* :func:`fetch_models`: ``GET /models`` under a hard timeout (reachable, key accepted, which
  models it offers).
* :func:`probe_tools`: one tiny streamed request on a model with a single function tool and
  ``tool_choice: auto``, to see whether the model calls tools at all. Spark's connectors,
  timers and the researcher only work on models that do.
* :func:`explain`: a :class:`~spark.brain.openai_chat.ProviderError` as one plain sentence for
  the window ("Ollama isn't running. Start it from the Start menu.").

Every call goes through :class:`~spark.brain.openai_chat.ChatClient`, so the key handling and
error mapping are the brain's own: the key only ever travels in a request header.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from spark.brain.openai_chat import ProviderError

if TYPE_CHECKING:
    from spark.brain.openai_chat import ChatClient
    from spark.brain.openai_presets import Provider

__all__ = ["MODELS_TIMEOUT_S", "PROBE_TIMEOUT_S", "explain", "fetch_models", "probe_tools"]

MODELS_TIMEOUT_S = 8.0
PROBE_TIMEOUT_S = 25.0
PROBE_MAX_TOKENS = 256  # room for a reasoning model to think briefly before the call

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local time. Always use this to answer questions "
        "about the time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
PROBE_MESSAGES = [
    {"role": "system", "content": "You are a voice assistant. Use the tools you are given."},
    {"role": "user", "content": "What time is it right now? Use the tool."},
]


def _timeout_error(label: str, seconds: float) -> ProviderError:
    return ProviderError(
        "timeout",
        f"{label} didn't answer within {seconds:.0f} s",
        f"{label} took too long to answer.",
    )


async def fetch_models(chat: ChatClient, *, timeout_s: float = MODELS_TIMEOUT_S) -> list[str]:
    """The endpoint's model ids (``[]`` when it lists none). Raises :class:`ProviderError`,
    also for a timeout."""
    try:
        async with asyncio.timeout(timeout_s + 2):
            return await chat.list_models(timeout_s=timeout_s)
    except TimeoutError as exc:
        raise _timeout_error(chat.provider.label, timeout_s) from exc


async def probe_tools(chat: ChatClient, model: str, *, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """True when ``model`` called the probe tool, False when it answered in text instead.
    Raises :class:`ProviderError` (a refused key, an unknown model, a model that rejects
    ``tools`` outright: ``kind == "bad_request"``, a timeout)."""
    try:
        async with asyncio.timeout(timeout_s):
            result = await chat.stream(
                model,
                [dict(m) for m in PROBE_MESSAGES],
                [json.loads(json.dumps(PROBE_TOOL))],
                tool_choice="auto",
                max_tokens=PROBE_MAX_TOKENS,
            )
    except TimeoutError as exc:
        raise _timeout_error(chat.provider.label, timeout_s) from exc
    return any(call.name == PROBE_TOOL["function"]["name"] for call in result.tool_calls)


def explain(exc: ProviderError, provider: Provider, *, model: str | None = None) -> str:
    """One sentence for a person, with what to do about it. Never contains the key."""
    label = provider.label
    status = f" (HTTP {exc.status})" if exc.status else ""
    kind = exc.kind
    if kind == "auth" and exc.status == 403:
        return (
            f"{label} recognised the key but refused the request{status}. Check that the key's "
            "account or project has access to this model and that billing is set up."
        )
    if kind == "auth":
        if not provider.key_env or not provider.key_required:
            return f"{label} wants an API key{status}. Enter the key the server expects."
        return (
            f"{label} didn't accept the API key{status}. Check that you copied the whole key, "
            "or make a new one."
        )
    if kind == "model":
        if model:
            return f"{label} doesn't have a model called “{model}”{status}. Pick one from the list."
        return (
            f"Nothing answered at that address{status}. Check the address; it usually ends in /v1."
        )
    if kind == "rate_limit":
        return (
            f"{label} says too many requests{status}. Wait a minute and try again, and check "
            "that the account has credit."
        )
    if kind == "server":
        return f"{label} had a problem on its side{status}. Try again in a moment."
    if kind == "timeout":
        return f"{label} didn't answer in time. Check the address and your connection."
    if kind == "connect":
        if provider.name == "ollama":
            return "Ollama isn't running. Start it from the Start menu, then test again."
        if provider.name == "hermes":
            return (
                f"Can't reach Hermes Agent's API server at {provider.base_url}. Check that "
                "hermes gateway is running there with API_SERVER_ENABLED and API_SERVER_KEY "
                "set in Hermes's .env, and API_SERVER_HOST=0.0.0.0 if it's another machine."
            )
        if provider.name == "lmstudio":
            return (
                "LM Studio's server isn't running. Open LM Studio, load a model and start the "
                "server (Developer tab), then test again."
            )
        if provider.local:
            return f"Nothing is running at {provider.base_url}. Start the server, then test again."
        return f"Can't reach {label}. Check your internet connection and the address."
    if kind == "bad_request" and "tool" in str(exc).lower():
        return "This model can't use tools. Pick another model."
    return f"{label} rejected the request{status}."
