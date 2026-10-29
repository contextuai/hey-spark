"""MCP sessions run by Spark itself, for brains that are not the Claude CLI (P5.7e).

On the Claude brain the ``claude`` CLI connects to the MCP servers. The OpenAI-compatible brain
has no CLI, so :class:`McpManager` does it in-process with the ``mcp`` client library:

* One **owner task** per server connects (stdio, streamable HTTP or SSE, built by
  :func:`spark.brain.mcp_probe._make_client`, the same code the Connectors window's "Test
  connection" uses, with ``${VAR}`` placeholders expanded here and never logged), lists its
  tools and then keeps the session open across turns. The ``mcp`` client holds an anyio task
  group, which must be entered and exited in the same task, hence one owner per server. Tool
  calls can come from any task.
* A server that fails to connect, or whose session breaks during a call, is marked ``failed``
  with a short reason (:func:`spark.brain.mcp_probe.friendly_probe_error`). The brain asks
  :meth:`McpManager.ensure` before each turn, which retries failed servers at most every
  ``retry_after_s``.
* The per-server status goes to ``on_status`` in the CLI's shape (``{"WAREHOUSE": {"status":
  "connected"}}``, ``"failed"`` with ``"error"``), so connector health (P5.7i), the Connectors
  window and the "X is unreachable" line work as on Claude.
* Spark's own tools (spark/tools: time, timers, status, start_project, connector notes) are
  ``claude_agent_sdk`` ``@tool`` functions. :class:`InProcessServer` calls their ``handler``
  coroutines directly, so there is one implementation of each tool for both brains.

``mcp`` is imported only when a server connects.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spark.brain.mcp_config import split_tool_name, tool_name
from spark.brain.openai_tools import result_text

__all__ = ["ClientFactory", "InProcessServer", "McpManager", "ToolOutcome", "ToolSpec"]

log = logging.getLogger("spark.brain.mcp")

StatusMap = dict[str, dict[str, str]]


@dataclass(frozen=True)
class ToolSpec:
    name: str  # full name: mcp__<server>__<tool>
    server: str
    tool: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolOutcome:
    text: str
    is_error: bool = False


class InProcessServer:
    """Spark's in-process tools (``SdkMcpTool`` objects: ``name``, ``description``,
    ``input_schema``, ``handler``), called without an MCP transport."""

    def __init__(self, tools: Iterable[Any]) -> None:
        self.tools: dict[str, Any] = {t.name: t for t in tools}

    def specs(self, server: str) -> list[ToolSpec]:
        out = []
        for name, t in self.tools.items():
            schema = t.input_schema if isinstance(t.input_schema, dict) else {"type": "object"}
            out.append(ToolSpec(tool_name(server, name), server, name, t.description, schema))
        return out

    async def call(self, tool: str, args: Mapping[str, Any]) -> ToolOutcome:
        t = self.tools.get(tool)
        if t is None:
            return ToolOutcome(f"There is no tool called {tool}.", True)
        result = await t.handler(dict(args))
        result = result if isinstance(result, Mapping) else {}
        is_error = bool(result.get("is_error") or result.get("isError"))
        return ToolOutcome(result_text(result.get("content")), is_error)


@dataclass
class _Server:
    name: str
    config: Mapping[str, Any]
    client: Any | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    status: str = "pending"
    error: str | None = None
    first: asyncio.Event = field(default_factory=asyncio.Event)  # first connect attempt is over
    wake: asyncio.Event = field(default_factory=asyncio.Event)  # reconnect, or stop
    task: asyncio.Task[None] | None = None
    last_attempt: float = 0.0


ClientFactory = Callable[[str, Mapping[str, Any], Any], "tuple[Any, Any]"]


class McpManager:
    """Keeps one MCP session per external server, plus Spark's in-process tools.

    Args:
        servers: ``McpSetup.servers`` (stdio / http / sse dicts with ``${VAR}`` placeholders,
            and :class:`InProcessServer` values for in-process servers).
        base_dir: where relative stdio commands resolve (the config folder).
        environ: where ``${VAR}`` values come from (default ``os.environ``).
        client_factory: ``factory(name, config, errlog) -> (client, http_to_close)``; tests pass
            fakes. The client is an async context manager with ``list_tools(cursor=)`` and
            ``call_tool(name, arguments)``.
        on_status: called with the external servers' status whenever it changes.
    """

    def __init__(
        self,
        servers: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
        connect_timeout_s: float = 30.0,
        call_timeout_s: float = 120.0,
        retry_after_s: float = 30.0,
        on_status: Callable[[StatusMap], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.environ = environ
        self._factory = client_factory or self._default_factory
        self.connect_timeout_s = connect_timeout_s
        self.call_timeout_s = call_timeout_s
        self.retry_after_s = retry_after_s
        self.on_status = on_status
        self._now = clock or (lambda: asyncio.get_running_loop().time())
        self.inproc: dict[str, InProcessServer] = {}
        self._servers: dict[str, _Server] = {}
        for name, cfg in servers.items():
            if isinstance(cfg, InProcessServer):
                self.inproc[name] = cfg
            elif isinstance(cfg, Mapping) and cfg.get("type", "stdio") in ("stdio", "http", "sse"):
                self._servers[name] = _Server(name, cfg)
            else:
                log.warning("MCP %s: unsupported server entry for this brain; skipped", name)
        self._started = False
        self._closing = False
        self._reported: StatusMap | None = None

    # -- lifecycle --------------------------------------------------------------------------------

    def _default_factory(self, name: str, config: Mapping[str, Any], errlog: Any) -> Any:
        from spark.brain.mcp_probe import _make_client

        env = os.environ if self.environ is None else self.environ
        return _make_client(config, self.base_dir, env, errlog)

    async def start(self, wait_s: float | None = None) -> None:
        """Start every owner task (once) and wait up to ``wait_s`` for their first attempt."""
        if not self._started:
            self._started = True
            for st in self._servers.values():
                st.task = asyncio.create_task(self._own(st), name=f"spark-mcp-{st.name}")
            self._report()
        waits = [asyncio.ensure_future(st.first.wait()) for st in self._servers.values()]
        if not waits:
            return
        try:
            await asyncio.wait(waits, timeout=wait_s)
        finally:
            for w in waits:
                w.cancel()

    def ensure(self) -> None:
        """Retry servers that failed more than ``retry_after_s`` ago (called before a turn)."""
        if not self._started or self._closing:
            return
        now = self._now()
        for st in self._servers.values():
            if st.status == "failed" and now - st.last_attempt >= self.retry_after_s:
                st.wake.set()

    async def aclose(self, timeout_s: float = 10.0) -> None:
        self._closing = True
        tasks = [st.task for st in self._servers.values() if st.task is not None]
        for st in self._servers.values():
            st.wake.set()
        if not tasks:
            return
        _done, pending = await asyncio.wait(tasks, timeout=timeout_s)
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(BaseException):
                await task

    async def _own(self, st: _Server) -> None:
        """Connect, list tools, hold the session until woken; repeat until closing."""
        from spark.brain.mcp_probe import _list_tools, _stderr_tail, friendly_probe_error

        while not self._closing:
            st.wake.clear()
            st.last_attempt = self._now()
            errlog = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")  # noqa: SIM115
            http = None
            try:
                async with contextlib.AsyncExitStack() as stack:
                    async with asyncio.timeout(self.connect_timeout_s):
                        client_cm, http = self._factory(st.name, st.config, errlog)
                        client = await stack.enter_async_context(client_cm)
                        listed = await _list_tools(client)
                    st.tools = [
                        ToolSpec(
                            tool_name(st.name, t.name),
                            st.name,
                            t.name,
                            getattr(t, "description", None) or "",
                            dict(getattr(t, "input_schema", None) or {"type": "object"}),
                        )
                        for t in listed
                    ]
                    st.client, st.status, st.error = client, "connected", None
                    log.info("MCP %s: connected, %d tools", st.name, len(st.tools))
                    st.first.set()
                    self._report()
                    await st.wake.wait()
                    st.client = None
            except asyncio.CancelledError:
                st.client = None
                if self._closing:
                    raise
                # anyio cancelled the owner because the session died: report and wait.
                task = asyncio.current_task()
                if task is not None:
                    while task.uncancel():
                        pass
                self._fail(st, "the connection closed")
            except Exception as exc:
                st.client = None
                self._fail(st, friendly_probe_error(exc, stderr_tail=_stderr_tail(errlog)))
            finally:
                if http is not None:
                    with contextlib.suppress(Exception):
                        await http.aclose()
                errlog.close()
            if self._closing:
                break
            if st.status == "failed":
                await st.wake.wait()  # ensure() wakes it after retry_after_s

    def _fail(self, st: _Server, error: str) -> None:
        st.status, st.error, st.tools = "failed", error, []
        st.first.set()
        self._report()

    # -- status -----------------------------------------------------------------------------------

    @property
    def status(self) -> StatusMap:
        out: StatusMap = {}
        for st in self._servers.values():
            info = {"status": st.status}
            if st.error:
                info["error"] = st.error[:200]
            out[st.name] = info
        return out

    def _report(self) -> None:
        current = self.status
        if current == self._reported:
            return
        self._reported = current
        if self.on_status is not None:
            try:
                self.on_status({k: dict(v) for k, v in current.items()})
            except Exception as exc:
                log.debug("MCP: on_status failed: %s", exc)

    # -- tools ------------------------------------------------------------------------------------

    def tools(self) -> list[ToolSpec]:
        """Every tool of the in-process servers and the connected external servers."""
        out: list[ToolSpec] = []
        for name, server in self.inproc.items():
            out.extend(server.specs(name))
        for st in self._servers.values():
            if st.status == "connected":
                out.extend(st.tools)
        return out

    async def call(self, full_name: str, args: Mapping[str, Any]) -> ToolOutcome:
        """Run one tool. Never raises for a tool or connection failure: the outcome says it."""
        parts = split_tool_name(full_name)
        if parts is None:
            return ToolOutcome(f"There is no tool called {full_name}.", True)
        server, tool = parts
        if server in self.inproc:
            return await self.inproc[server].call(tool, args)
        st = self._servers.get(server)
        if st is None:
            return ToolOutcome(f"There is no connector called {server}.", True)
        client = st.client
        if client is None or st.status != "connected":
            return ToolOutcome(f"The {server} connector isn't connected right now.", True)
        try:
            async with asyncio.timeout(self.call_timeout_s):
                result = await client.call_tool(tool, dict(args))
        except TimeoutError:
            return ToolOutcome(f"The {server} connector didn't answer in time.", True)
        except Exception as exc:
            from spark.brain.mcp_probe import friendly_probe_error

            reason = friendly_probe_error(exc)
            log.warning("MCP %s: call to %s failed (%s); reconnecting", server, tool, reason)
            self._fail(st, reason)
            st.last_attempt = 0.0
            st.wake.set()  # leave the broken session and try again
            return ToolOutcome(f"The {server} connector failed: {reason}", True)
        content = getattr(result, "content", None)
        if content is None and isinstance(result, Mapping):
            content = result.get("content")
        structured = getattr(result, "structured_content", None)
        is_error = bool(getattr(result, "is_error", False))
        return ToolOutcome(result_text(content, structured), is_error)
