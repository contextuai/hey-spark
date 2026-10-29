"""Streaming Chat Completions over ``httpx`` for the OpenAI-compatible brain (P5.7e).

``POST {base_url}/chat/completions`` with ``stream: true``. The response is server-sent events:
one ``data: {json}`` line per chunk, ``data: [DONE]`` at the end, and ``:`` comment lines some
providers use as keep-alives (OpenRouter). Each chunk's ``choices[0].delta`` carries either
``content`` (reply text, passed on as it arrives so the app's sentence splitter can start
speaking) or ``tool_calls`` fragments: ``index`` says which call, the first fragment has the
``id`` and ``function.name``, and ``function.arguments`` arrives as JSON split across chunks.
:class:`StreamAccumulator` stitches them together. ``finish_reason`` is ``stop``,
``tool_calls`` or ``length``. A last chunk may carry ``usage`` (with
``stream_options.include_usage``).

Errors become :class:`ProviderError` with a ``spoken`` sentence the app says instead of its
generic failure line: a refused key (401/403), an unknown model (404), rate limits (429, retried
twice with backoff first), server errors (5xx, retried once), and connection failures ("is
Ollama running?" for local servers). The key is only ever put in a request header, never in a
log line or an error message.

``httpx`` is imported inside :class:`ChatClient`, so importing this module stays cheap.
Cancelling the task that awaits :meth:`ChatClient.stream` closes the HTTP response at once
(barge-in).
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spark.brain.core import BrainError

if TYPE_CHECKING:
    from spark.brain.openai_presets import Provider
    from spark.config import OpenAICompatConfig
    from spark.util.clock import Clock

__all__ = [
    "ChatClient",
    "ChatResult",
    "ProviderError",
    "StreamAccumulator",
    "ToolCall",
    "parse_sse_line",
]

log = logging.getLogger("spark.brain.openai")

BACKEND = "openai_compat"
RETRY_429_DELAYS_S = (1.0, 2.5)  # then give up and say "rate limited"
RETRY_5XX_DELAYS_S = (1.0,)
MAX_RETRY_AFTER_S = 8.0
OLLAMA_KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded between questions


class ProviderError(BrainError):
    """A request to the provider failed. ``kind``: auth, model, rate_limit, server, connect,
    timeout, bad_request, stream. ``spoken`` is safe to say (no key, URL or raw error)."""

    def __init__(self, kind: str, message: str, spoken: str, *, status: int | None = None) -> None:
        super().__init__(message, spoken=spoken)
        self.kind = kind
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""


@dataclass
class ChatResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None  # the model the provider says answered


def parse_sse_line(line: str) -> dict[str, Any] | str | None:
    """One SSE line -> the chunk dict, ``"[DONE]"``, or None (blank, comment, other field)."""
    line = line.strip()
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data:
        return None
    if data == "[DONE]":
        return "[DONE]"
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        log.debug("openai: skipping a stream line that is not JSON (%d chars)", len(data))
        return None
    return chunk if isinstance(chunk, dict) else None


class StreamAccumulator:
    """Folds stream chunks into text and complete tool calls.

    ``feed(chunk)`` returns the chunk's text delta ('' if none). Tool-call fragments are joined
    by ``index``. Some servers leave ``index`` out (one whole call per chunk); then a new ``id``
    starts a new call and a fragment without one continues the last call."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.calls: dict[int, ToolCall] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.model: str | None = None
        self.new_calls: list[int] = []  # indexes first seen in the last feed()

    def feed(self, chunk: Mapping[str, Any]) -> str:
        self.new_calls = []
        if isinstance(chunk.get("model"), str) and chunk["model"] and self.model is None:
            self.model = chunk["model"]
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = usage
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason"):
            self.finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            return ""
        for frag in delta.get("tool_calls") or []:
            if isinstance(frag, dict):
                self._tool_fragment(frag)
        text = delta.get("content")
        if isinstance(text, str) and text:
            self.text_parts.append(text)
            return text
        return ""

    def _tool_fragment(self, frag: Mapping[str, Any]) -> None:
        index = frag.get("index")
        frag_id = frag.get("id") if isinstance(frag.get("id"), str) else None
        if not isinstance(index, int):
            match = next((i for i, c in self.calls.items() if frag_id and c.id == frag_id), None)
            if match is not None:
                index = match
            elif frag_id or not self.calls:
                index = max(self.calls, default=-1) + 1
            else:
                index = max(self.calls)
        call = self.calls.get(index)
        if call is None:
            call = self.calls[index] = ToolCall(id="", name="")
            self.new_calls.append(index)
        if frag_id and not call.id:
            call.id = frag_id
        fn = frag.get("function")
        if isinstance(fn, dict):
            if isinstance(fn.get("name"), str) and fn["name"]:
                # Names arrive whole in practice; a few servers repeat them per fragment.
                if not call.name:
                    call.name = fn["name"]
                elif not call.name.endswith(fn["name"]):
                    call.name += fn["name"]
            args = fn.get("arguments")
            if isinstance(args, str):
                call.arguments += args
            elif isinstance(args, dict):  # some local servers send an object, not a string
                call.arguments = json.dumps(args)

    def result(self) -> ChatResult:
        calls = [self.calls[i] for i in sorted(self.calls)]
        for n, call in enumerate(calls):
            if not call.id:
                call.id = f"call_{n}"
        return ChatResult(
            text="".join(self.text_parts),
            tool_calls=[c for c in calls if c.name],
            finish_reason=self.finish_reason,
            usage=self.usage,
            model=self.model,
        )


class ChatClient:
    """One provider endpoint: streaming chat, a model list, error mapping, retries.

    Args:
        provider: the resolved preset (:func:`spark.brain.openai_presets.resolve_provider`).
        cfg: ``brain.openai_compat`` (timeouts, max_tokens, temperature, extra_body, ...).
        transport: an ``httpx`` transport (tests pass ``httpx.MockTransport``).
        clock: for retry backoff (tests pass a fake clock).
        environ: where the key is read from (default ``os.environ``; ``.env`` is merged there).
    """

    def __init__(
        self,
        provider: Provider,
        cfg: OpenAICompatConfig,
        *,
        transport: Any | None = None,
        clock: Clock | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a core dependency
            from spark.interfaces import BackendUnavailable

            reason = f"httpx is not importable ({exc})"
            raise BackendUnavailable(BACKEND, reason, "uv sync --all-extras") from exc
        from spark.util.clock import SystemClock

        self._httpx = httpx
        self.provider = provider
        self.cfg = cfg
        self._transport = transport
        self._clock: Clock = clock or SystemClock()
        self._environ = environ
        self._client: Any | None = None
        self.requests = 0

    # -- plumbing ---------------------------------------------------------------------------------

    def _http(self) -> Any:
        if self._client is None:
            httpx = self._httpx
            timeout = httpx.Timeout(self.cfg.timeout_s, connect=min(10.0, self.cfg.timeout_s))
            self._client = httpx.AsyncClient(timeout=timeout, transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception as exc:
                log.debug("openai: closing the HTTP client failed: %s", exc)

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            **self.provider.request_headers(self._environ),
        }

    def body(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: Any | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
            if cfg.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = cfg.parallel_tool_calls
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        limit = max_tokens if max_tokens is not None else cfg.max_tokens
        if limit:
            body[self.provider.max_tokens_field] = limit
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if self.provider.stream_usage:
            body["stream_options"] = {"include_usage": True}
        if self.provider.name == "ollama":
            # Ollama unloads an idle model after 5 minutes by default, so the next question
            # waits for a reload and for the long prompt to be read again.
            body["keep_alive"] = OLLAMA_KEEP_ALIVE
        body.update(cfg.extra_body)  # wins, e.g. {keep_alive: 2h}
        return body

    # -- requests ---------------------------------------------------------------------------------

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[int], None] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Run one streamed request. ``on_text(delta)`` gets reply text as it arrives and
        ``on_tool_start(index)`` fires when a tool call first appears. Retries 429 and 5xx only
        before anything was streamed."""
        body = self.body(model, messages, tools, tool_choice=tool_choice, max_tokens=max_tokens)
        url = self.provider.url("chat/completions", model)
        attempt_429 = attempt_5xx = 0
        while True:
            try:
                return await self._stream_once(url, body, model, on_text, on_tool_start)
            except ProviderError as exc:
                delay = None
                if exc.kind == "rate_limit" and attempt_429 < len(RETRY_429_DELAYS_S):
                    delay = getattr(exc, "retry_after", None) or RETRY_429_DELAYS_S[attempt_429]
                    attempt_429 += 1
                elif exc.kind == "server" and attempt_5xx < len(RETRY_5XX_DELAYS_S):
                    delay = RETRY_5XX_DELAYS_S[attempt_5xx]
                    attempt_5xx += 1
                if delay is None or getattr(exc, "streamed", False):
                    raise
                log.info("openai: %s (HTTP %s); retrying in %.1f s", exc.kind, exc.status, delay)
                await self._clock.sleep(min(delay, MAX_RETRY_AFTER_S))

    async def _stream_once(
        self,
        url: str,
        body: dict[str, Any],
        model: str,
        on_text: Callable[[str], None] | None,
        on_tool_start: Callable[[int], None] | None,
    ) -> ChatResult:
        httpx = self._httpx
        acc = StreamAccumulator()
        streamed = False
        self.requests += 1
        try:
            async with self._http().stream(
                "POST", url, json=body, headers=self._headers(), params=self.provider.params()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise self._http_error(response.status_code, raw, model, response.headers)
                async for line in response.aiter_lines():
                    item = parse_sse_line(line)
                    if item is None:
                        continue
                    if item == "[DONE]":
                        break
                    assert isinstance(item, dict)
                    if isinstance(item.get("error"), dict | str):
                        raise self._stream_error(item["error"], model)
                    delta = acc.feed(item)
                    if acc.new_calls and on_tool_start is not None:
                        streamed = True
                        for index in acc.new_calls:
                            on_tool_start(index)
                    if delta:
                        streamed = True
                        if on_text is not None:
                            on_text(delta)
        except ProviderError as exc:
            exc.streamed = streamed  # type: ignore[attr-defined]
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "timeout",
                f"{self.provider.label} timed out ({type(exc).__name__})",
                f"{self.provider.label} took too long to answer. Try again in a moment.",
            ) from exc
        except httpx.TransportError as exc:
            raise self._connect_error(exc) from exc
        return acc.result()

    async def complete(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int | None = None
    ) -> str:
        """The whole text reply of a tool-less request (router classifier, side questions)."""
        result = await self.stream(model, messages, max_tokens=max_tokens)
        return result.text.strip()

    async def list_models(self, *, timeout_s: float = 8.0) -> list[str]:
        """``GET {base_url}/models`` -> model ids. Raises :class:`ProviderError` on failure."""
        httpx = self._httpx
        headers = self.provider.request_headers(self._environ)
        try:
            response = await self._http().get(
                self.provider.url("models"),
                headers=headers,
                params=self.provider.params(),
                timeout=timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "timeout",
                f"{self.provider.label}: GET /models timed out",
                f"{self.provider.label} took too long to answer.",
            ) from exc
        except httpx.TransportError as exc:
            raise self._connect_error(exc) from exc
        if response.status_code >= 400:
            raise self._http_error(response.status_code, response.content, "", response.headers)
        try:
            data = response.json()
        except ValueError:
            return []
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = data.get("models", []) if isinstance(data, dict) else []
        ids = []
        for item in items:
            mid = item.get("id") or item.get("name") if isinstance(item, dict) else item
            if isinstance(mid, str) and mid:
                ids.append(mid)
        return ids

    # -- errors -----------------------------------------------------------------------------------

    def _detail(self, raw: bytes | str) -> str:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        try:
            data = json.loads(text)
        except ValueError:
            return " ".join(text.split())[:300]
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, dict):
            text = str(err.get("message") or err.get("code") or err)
        else:
            text = str(err)
        return " ".join(text.split())[:300]

    def _http_error(
        self, status: int, raw: bytes | str, model: str, headers: Mapping[str, str]
    ) -> ProviderError:
        label = self.provider.label
        detail = self._detail(raw)
        message = f"{label} answered HTTP {status}: {detail}"
        low = detail.lower()
        if status in (401, 403):
            env = f" (the key comes from {self.provider.key_env})" if self.provider.key_env else ""
            return ProviderError(
                "auth",
                message + env,
                f"{label} refused the API key. Check the key in the .env file.",
                status=status,
            )
        if status == 404 or "model_not_found" in low or "does not exist" in low:
            return ProviderError(
                "model",
                f"{message} (model {model!r}; check brain.openai_compat model names and base_url)",
                f"{label} doesn't have the model I asked for. Check the model name in the "
                "settings.",
                status=status,
            )
        if status == 429:
            exc = ProviderError(
                "rate_limit",
                message,
                f"{label} is rate limiting me right now. Try again in a minute.",
                status=status,
            )
            with contextlib.suppress(ValueError):
                exc.retry_after = float(headers.get("retry-after", ""))  # type: ignore[attr-defined]
            return exc
        if status >= 500:
            return ProviderError(
                "server",
                message,
                f"{label} had a problem on its side. Try again in a moment.",
                status=status,
            )
        if "tool" in low and ("support" in low or "not allowed" in low):
            return ProviderError(
                "bad_request",
                message,
                "The selected model can't use tools. Pick a model that supports tool calling.",
                status=status,
            )
        return ProviderError(
            "bad_request", message, f"{label} rejected the request.", status=status
        )

    def _stream_error(self, err: Any, model: str) -> ProviderError:
        detail = self._detail(json.dumps(err) if isinstance(err, dict) else str(err))
        code = err.get("code") if isinstance(err, dict) else None
        if isinstance(code, int) and code >= 400:
            return self._http_error(code, json.dumps({"error": err}), model, {})
        return ProviderError(
            "stream",
            f"{self.provider.label} stream error: {detail}",
            f"{self.provider.label} stopped in the middle of the answer. Try again.",
        )

    def _connect_error(self, exc: BaseException) -> ProviderError:
        label = self.provider.label
        if self.provider.local:
            spoken = f"I can't reach {label} on this computer. Is it running?"
        else:
            spoken = f"I can't reach {label} right now. Check the internet connection."
        return ProviderError("connect", f"can't reach {label}: {type(exc).__name__}", spoken)
