"""Connect to one MCP server and list its tools (the Connectors window's "Test connection").

This runs in Spark's own process with the ``mcp`` client library (installed with
``claude-agent-sdk``), not through the ``claude`` CLI, so the ``${VAR}`` placeholders are
expanded here from the environment. Nothing goes on a command line except a stdio server's own
``command`` and ``args``, which the brain passes the same way.

:func:`probe` never raises: it returns a :class:`ProbeResult` with either the tools or a short
error the window can show. Tool hints come from the server's MCP annotations
(``readOnlyHint`` / ``destructiveHint``); servers may leave them out.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spark.brain.mcp_config import _resolve_command, expand_vars, referenced_vars

log = logging.getLogger("spark.brain.mcp")

DEFAULT_TIMEOUT_S = 25.0
_MAX_DESCRIPTION = 400


@dataclass
class ProbeResult:
    ok: bool
    tools: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    server_name: str | None = None
    server_version: str | None = None


def _expand(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return expand_vars(value, environ)
    if isinstance(value, dict):
        return {k: _expand(v, environ) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, environ) for v in value]
    return value


def tool_summary(tool: Any) -> dict[str, Any]:
    """The fields the window shows for one ``mcp.types.Tool``."""
    ann = getattr(tool, "annotations", None)
    read_only = getattr(ann, "read_only_hint", None)
    destructive = getattr(ann, "destructive_hint", None)
    hint = "read" if read_only else "destructive" if destructive else None
    description = " ".join((getattr(tool, "description", None) or "").split())
    if len(description) > _MAX_DESCRIPTION:
        description = description[: _MAX_DESCRIPTION - 1].rstrip() + "…"
    title = getattr(tool, "title", None) or getattr(ann, "title", None)
    return {"name": tool.name, "title": title, "description": description, "hint": hint}


def _leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def friendly_probe_error(exc: BaseException, *, stderr_tail: str = "") -> str:
    exc = _leaf(exc)
    text = str(exc).strip()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return f"The server refused the credentials (HTTP {status}). Check the token in .env."
    if status == 404:
        return "Nothing at that URL (HTTP 404). Check the server address."
    if status:
        return f"The server answered HTTP {status}."
    if isinstance(exc, TimeoutError):
        return "No answer in time. Is the server running and reachable from this computer?"
    name = type(exc).__name__
    lowered = text.lower()
    if name in ("ConnectError", "ConnectTimeout") or any(
        m in lowered for m in ("connection refused", "getaddrinfo", "name or service not known")
    ):
        return "Can't reach the server. Check the address and that you're on the right network."
    if isinstance(exc, FileNotFoundError | PermissionError):
        return f"Can't start the server: {text or name}"
    detail = text or name
    if stderr_tail:
        detail = f"{detail}. Server said: {stderr_tail}"
    return detail.splitlines()[0][:400] if "\n" in detail else detail[:400]


def _stderr_tail(fh: Any) -> str:
    try:
        fh.flush()
        fh.seek(0)
        lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return ""
    return lines[-1][:300] if lines else ""


async def _list_tools(client: Any) -> list[Any]:
    tools: list[Any] = []
    cursor: str | None = None
    for _ in range(50):  # paginate, but never forever
        result = await client.list_tools(cursor=cursor)
        tools.extend(result.tools)
        cursor = getattr(result, "next_cursor", None)
        if not cursor:
            break
    return tools


def _make_client(
    config: Mapping[str, Any], base_dir: Path, environ: Mapping[str, str], errlog: Any
) -> tuple[Any, Any]:
    """``(mcp.Client, http client to close or None)`` for one server definition."""
    from mcp import Client, StdioServerParameters

    kind = config.get("type", "stdio" if "command" in config else None)
    if kind == "stdio":
        resolved = _resolve_command(str(config.get("command", "")), base_dir, environ)
        if isinstance(resolved, str):
            raise FileNotFoundError(resolved)
        command, prefix = resolved
        env = {k: str(v) for k, v in _expand(config.get("env") or {}, environ).items()}
        params = StdioServerParameters(
            command=command,
            args=[*prefix, *_expand(list(config.get("args") or []), environ)],
            env=env or None,
            cwd=str(base_dir),
        )
        from mcp.client.stdio import stdio_client

        return Client(stdio_client(params, errlog=errlog)), None
    url = expand_vars(str(config.get("url", "")), environ).strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("The URL must start with http:// or https://")
    headers = {k: str(v) for k, v in _expand(config.get("headers") or {}, environ).items()}
    if kind == "sse":
        from mcp.client.sse import sse_client

        return Client(sse_client(url, headers=headers or None, timeout=15.0)), None
    if kind == "http":
        import httpx2
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        http = create_mcp_http_client(
            headers=headers or None, timeout=httpx2.Timeout(15.0, read=60.0)
        )
        return Client(streamable_http_client(url, http_client=http)), http
    raise ValueError(f"Unsupported server type {kind!r}")


async def probe_async(
    config: Mapping[str, Any],
    *,
    base_dir: Path,
    environ: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ProbeResult:
    env = os.environ if environ is None else environ
    missing = sorted(v for v in referenced_vars(dict(config))[0] if not env.get(v))
    if missing:
        return ProbeResult(False, error=f"Missing in .env: {', '.join(missing)}")
    errlog = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")  # noqa: SIM115
    http = None
    try:
        async with asyncio.timeout(timeout_s):
            client, http = _make_client(config, base_dir, env, errlog)
            async with client:
                tools = await _list_tools(client)
                server_info = getattr(client, "server_info", None)
        return ProbeResult(
            True,
            tools=[tool_summary(t) for t in tools],
            server_name=getattr(server_info, "name", None),
            server_version=getattr(server_info, "version", None),
        )
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # anyio raises groups; report them, never raise
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        log.info("MCP test failed: %s", type(_leaf(exc)).__name__, exc_info=True)
        return ProbeResult(False, error=friendly_probe_error(exc, stderr_tail=_stderr_tail(errlog)))
    finally:
        if http is not None:
            try:
                await http.aclose()
            except Exception:
                log.debug("closing the MCP http client failed", exc_info=True)
        errlog.close()


def probe(
    config: Mapping[str, Any],
    *,
    base_dir: Path,
    environ: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ProbeResult:
    """Blocking wrapper for threads without an event loop (the window's API threads)."""
    return asyncio.run(probe_async(config, base_dir=base_dir, environ=environ, timeout_s=timeout_s))
