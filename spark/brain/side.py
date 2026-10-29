"""A small, tool-less side session on ``brain.fast_model`` (Phase 5).

Some jobs need a quick model answer that must never enter the user's conversation: "is this
utterance meant for Spark?" (P5.4), a conversation's title and summary (P5.8). Each
:class:`SideModel` keeps its own ``ClaudeSDKClient`` with a fixed system prompt, owned by its own
worker task (the SDK client must be used from the task that connected it, docs/sdk_notes.md §1),
and reconnects every ``reset_every`` calls so its context stays small. Same pattern as the
router's ``FastModelClassifier``.

``claude_agent_sdk`` is imported lazily, on first use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["SideModel", "make_side_model"]


@dataclass
class _Ask:
    prompt: str
    future: asyncio.Future[str]


class SideModel:
    """Ask ``brain.fast_model`` (or ``model``) one-off questions with a fixed system prompt.

    Args:
        settings: Spark settings (model names, CLI path, connect timeout, env).
        system_prompt: the instructions for every question.
        sdk: the ``claude_agent_sdk`` module (tests inject a fake; default: lazy import).
        model: defaults to ``settings.brain.fast_model``.
        reset_every: reconnect after this many questions (keeps the side context tiny).
        name: for logs and the worker task's name.
    """

    def __init__(
        self,
        settings: Settings,
        system_prompt: str,
        *,
        sdk: ModuleType | None = None,
        model: str | None = None,
        reset_every: int = 20,
        name: str = "side",
    ) -> None:
        self.settings = settings
        self.system_prompt = system_prompt
        self.model = model or settings.brain.fast_model
        self.reset_every = reset_every
        self.name = name
        self.calls = 0
        self._sdk = sdk
        self._client: Any | None = None
        self._since_connect = 0
        self._requests: asyncio.Queue[_Ask | None] | None = None
        self._worker: asyncio.Task[None] | None = None

    def _load_sdk(self) -> ModuleType:
        if self._sdk is None:
            import claude_agent_sdk

            self._sdk = claude_agent_sdk
        return self._sdk

    def _options(self) -> Any:
        from spark.brain.mcp_config import env_for_sdk

        kwargs: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "tools": [],
            "setting_sources": [],
            "strict_mcp_config": True,
            "max_turns": 1,
            "env": env_for_sdk(self.settings, claudeai=False),
        }
        if self.settings.brain.cli_path:
            kwargs["cli_path"] = self.settings.brain.cli_path
        return self._load_sdk().ClaudeAgentOptions(**kwargs)

    def _ensure_worker(self) -> asyncio.Queue[_Ask | None]:
        if self._worker is None or self._worker.done():
            self._requests = asyncio.Queue()
            self._worker = asyncio.create_task(self._run(), name=f"spark-{self.name}")
        assert self._requests is not None
        return self._requests

    async def warmup(self) -> None:
        """Connect ahead of the first question (the CLI spawn takes seconds)."""
        await self.ask("")

    async def ask(self, prompt: str) -> str:
        """The model's whole text reply to ``prompt`` ('' for an empty prompt: connect only)."""
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._ensure_worker().put_nowait(_Ask(prompt, fut))
        return await fut

    async def aclose(self) -> None:
        worker = self._worker
        if worker is None or worker.done():
            return
        assert self._requests is not None
        self._requests.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=10.0)
        except TimeoutError:
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker

    async def _run(self) -> None:
        assert self._requests is not None
        requests = self._requests
        try:
            while True:
                ask = await requests.get()
                if ask is None:
                    return
                try:
                    text = await self._handle(ask.prompt)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("%s: call failed: %s", self.name, exc)
                    await self._disconnect()
                    if not ask.future.done():
                        ask.future.set_exception(exc)
                    continue
                if not ask.future.done():  # the caller may have timed out already
                    ask.future.set_result(text)
        finally:
            await self._disconnect()
            while not requests.empty():
                left = requests.get_nowait()
                if left is not None and not left.future.done():
                    left.future.cancel()

    async def _handle(self, prompt: str) -> str:
        if self._client is not None and self._since_connect >= self.reset_every:
            await self._disconnect()
        if self._client is None:
            client = self._load_sdk().ClaudeSDKClient(options=self._options())
            async with asyncio.timeout(self.settings.brain.connect_timeout_s):
                await client.connect()
            self._client = client
            self._since_connect = 0
        if not prompt:
            return ""
        sdk = self._load_sdk()
        self._since_connect += 1
        self.calls += 1
        await self._client.query(prompt)
        chunks: list[str] = []
        async for msg in self._client.receive_response():
            if isinstance(msg, sdk.AssistantMessage):
                chunks.extend(b.text for b in msg.content if isinstance(b, sdk.TextBlock))
        return "".join(chunks).strip()

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as exc:
                log.debug("%s: disconnect failed: %s", self.name, exc)


def make_side_model(settings: Settings, system_prompt: str, *, name: str = "side") -> Any:
    """A side session on the configured backend's fast model: :class:`SideModel` on Claude,
    :class:`spark.brain.openai_side.OpenAISideModel` on ``openai_compat`` (P5.7e)."""
    if settings.brain.backend == "openai_compat":
        from spark.brain.openai_side import OpenAISideModel

        return OpenAISideModel(settings, system_prompt, name=name)
    return SideModel(settings, system_prompt, name=name)
