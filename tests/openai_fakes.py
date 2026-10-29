"""Fakes for the OpenAI-compatible brain: scripted SSE responses over ``httpx.MockTransport`` and
a fake MCP client (no real servers, no network)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx

from spark.activity import NullActivityLog

KEY_ENV = {"OPENAI_API_KEY": "sk-test-not-real"}


def sse(*chunks: dict[str, Any] | str, done: bool = True) -> bytes:
    """``data: ...`` lines for each chunk (a str is sent raw, e.g. a comment line)."""
    lines = []
    for chunk in chunks:
        lines.append(chunk if isinstance(chunk, str) else "data: " + json.dumps(chunk))
        lines.append("")
    if done:
        lines += ["data: [DONE]", ""]
    return ("\n".join(lines) + "\n").encode()


def text_chunk(text: str, *, finish: str | None = None, model: str = "served") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }


def finish_chunk(reason: str = "stop", usage: dict[str, int] | None = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
    if usage:
        chunk["usage"] = usage
    return chunk


def tool_chunk(
    index: int, *, id: str | None = None, name: str | None = None, args: str | None = None
) -> dict[str, Any]:
    frag: dict[str, Any] = {"index": index, "type": "function", "function": {}}
    if id:
        frag["id"] = id
    if name:
        frag["function"]["name"] = name
    if args is not None:
        frag["function"]["arguments"] = args
    return {"choices": [{"index": 0, "delta": {"tool_calls": [frag]}, "finish_reason": None}]}


def reply(*parts: str) -> bytes:
    """A plain text reply streamed in ``parts``."""
    return sse(*(text_chunk(p) for p in parts), finish_chunk("stop"))


def call(name: str, args: dict[str, Any] | str, *, id: str = "call_1", split: int = 2) -> bytes:
    """One tool call whose arguments arrive in ``split`` pieces."""
    raw = args if isinstance(args, str) else json.dumps(args)
    step = max(1, len(raw) // split)
    pieces = [raw[i : i + step] for i in range(0, len(raw), step)] or [""]
    chunks = [tool_chunk(0, id=id, name=name, args=pieces[0])]
    chunks += [tool_chunk(0, args=p) for p in pieces[1:]]
    return sse(*chunks, finish_chunk("tool_calls"))


class Script:
    """A MockTransport handler: answers chat requests from a list, ``GET /models`` from
    ``models``. Records every chat request body and its headers."""

    def __init__(self, *responses: bytes | httpx.Response | Exception, models: Iterable[str] = ()):
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.models = list(models)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        self.bodies.append(json.loads(request.content))
        self.headers.append(request.headers)
        if not self.responses:
            raise AssertionError("unexpected chat request (script exhausted)")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, content=item, headers={"content-type": "text/event-stream"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class HangingStream(httpx.AsyncByteStream):
    """Sends ``first`` and then waits forever (until the client closes it)."""

    def __init__(self, first: bytes) -> None:
        self.first = first
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield self.first
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed.set()


class RecActivity(NullActivityLog):
    """Captures activity events instead of writing them."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, Any]]] = []

    def record(self, kind: str, **fields: Any) -> dict[str, Any] | None:
        self.events.append((kind, fields))
        return None

    def kinds(self, kind: str) -> list[dict[str, Any]]:
        return [f for k, f in self.events if k == kind]


# --- fake MCP client --------------------------------------------------------------------------


@dataclass
class FakeMcpClient:
    tools: list[tuple[str, str, dict[str, Any]]]
    results: dict[str, str] = field(default_factory=dict)
    fail_connect: Exception | None = None
    fail_call: Exception | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    entered: int = 0
    exited: int = 0

    async def __aenter__(self) -> FakeMcpClient:
        if self.fail_connect is not None:
            raise self.fail_connect
        self.entered += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited += 1

    async def list_tools(self, cursor: str | None = None) -> Any:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(name=n, description=d, input_schema=s) for n, d, s in self.tools
            ],
            next_cursor=None,
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.fail_call is not None:
            exc, self.fail_call = self.fail_call, None
            raise exc
        text = self.results.get(name, f"{name} done")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            structured_content=None,
            is_error=False,
        )


def factory(clients: dict[str, FakeMcpClient]):
    """A ``client_factory`` for :class:`spark.brain.mcp_client.McpManager`."""

    def make(name: str, config: Any, errlog: Any) -> tuple[Any, Any]:
        return clients[name], None

    return make


OBJ = {"type": "object", "properties": {"state": {"type": "string"}}}
GITHUB_TOOLS = [
    ("list_issues", "List issues", OBJ),
    (
        "create_issue",
        "Create an issue",
        {"type": "object", "properties": {"title": {"type": "string"}}},
    ),
    ("delete_repository", "Delete a repo", OBJ),
    ("not_listed", "On the server, not in any list", OBJ),
]


def github_setup(*, deep: bool = False) -> Any:
    from spark.brain.mcp_config import McpSetup

    setup = McpSetup()
    setup.servers["github"] = {"type": "stdio", "command": "fake", "args": []}
    setup._register(
        "github",
        allowed=["list_issues"],
        side_effecting=["create_issue"],
        destructive=["delete_repository"],
        allow_destructive=False,
        known=(),
    )
    if deep:
        setup.deep_route["github"] = ("github",)
    return setup
