"""The streaming Chat Completions client, SSE parsing, tool-call stitching, error mapping,
function-name mapping and schema conversion (P5.7e). Offline: httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from spark.app import failure_line
from spark.brain.openai_chat import ChatClient, ProviderError, StreamAccumulator, parse_sse_line
from spark.brain.openai_presets import resolve_provider
from spark.brain.openai_tools import (
    FUNCTION_NAME_RE,
    ToolNameMap,
    clip_result,
    convert_schema,
    function_tool,
    result_text,
)
from spark.config import OpenAICompatConfig
from tests.fakes import FakeClock
from tests.openai_fakes import (
    KEY_ENV,
    Script,
    call,
    finish_chunk,
    reply,
    sse,
    text_chunk,
    tool_chunk,
)


def client(script: Script, provider: str = "openai", clock=None, **cfg) -> ChatClient:
    oc = OpenAICompatConfig(provider=provider, fast_model="m", **cfg)
    return ChatClient(
        resolve_provider(oc), oc, transport=script.transport(), clock=clock, environ=KEY_ENV
    )


# --- SSE and accumulation ------------------------------------------------------------------


def test_parse_sse_line_handles_done_comments_and_junk():
    assert parse_sse_line("data: [DONE]") == "[DONE]"
    assert parse_sse_line(": OPENROUTER PROCESSING") is None
    assert parse_sse_line("event: ping") is None
    assert parse_sse_line("") is None
    assert parse_sse_line("data: {not json") is None
    assert parse_sse_line('data:{"a": 1}') == {"a": 1}


def test_accumulator_joins_split_and_parallel_tool_calls():
    acc = StreamAccumulator()
    acc.feed(text_chunk("Checking. "))
    acc.feed(tool_chunk(0, id="a", name="mcp__github__list_issues", args='{"sta'))
    acc.feed(tool_chunk(1, id="b", name="mcp__spark__current_time", args=""))
    acc.feed(tool_chunk(0, args='te": "open"}'))
    acc.feed(tool_chunk(1, args="{}"))
    acc.feed(finish_chunk("tool_calls", usage={"prompt_tokens": 7, "completion_tokens": 3}))
    result = acc.result()
    assert result.text == "Checking. "
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("a", "mcp__github__list_issues", '{"state": "open"}'),
        ("b", "mcp__spark__current_time", "{}"),
    ]
    assert result.finish_reason == "tool_calls"
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 3}


def test_accumulator_without_index_or_id():
    acc = StreamAccumulator()
    whole = {"function": {"name": "x", "arguments": {"a": 1}}}
    acc.feed({"choices": [{"delta": {"tool_calls": [whole]}}]})
    acc.feed({"choices": [{"delta": {"tool_calls": [{"id": "z", "function": {"name": "y"}}]}}]})
    calls = acc.result().tool_calls
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_0", "x", '{"a": 1}'),
        ("z", "y", ""),
    ]


# --- requests --------------------------------------------------------------------------------


async def test_stream_text_and_request_shape():
    script = Script(sse(": keep-alive", text_chunk("Hel"), text_chunk("lo."), finish_chunk()))
    chat = client(script, max_tokens=50)
    seen: list[str] = []
    result = await chat.stream("m", [{"role": "user", "content": "hi"}], on_text=seen.append)
    assert seen == ["Hel", "lo."]
    assert result.text == "Hello." and result.model == "served"
    body = script.bodies[0]
    assert body["stream"] is True and body["model"] == "m"
    assert body["max_completion_tokens"] == 50  # OpenAI's preset field
    assert body["stream_options"] == {"include_usage": True}
    assert "tools" not in body and "temperature" not in body
    assert script.headers[0]["authorization"] == "Bearer sk-test-not-real"
    await chat.aclose()


async def test_body_options_for_other_presets():
    script = Script(reply("ok"))
    chat = client(
        script,
        provider="groq",
        temperature=0.3,
        parallel_tool_calls=False,
        extra_body={"seed": 1},
    )
    tools = [function_tool("t", "d", {"type": "object"})]
    await chat.stream("m", [], tools, tool_choice="auto")
    body = script.bodies[0]
    assert body["max_tokens"] == 1024 and "max_completion_tokens" not in body
    assert body["temperature"] == 0.3 and body["seed"] == 1
    assert body["parallel_tool_calls"] is False and body["tool_choice"] == "auto"
    assert "stream_options" not in body


async def test_ollama_keeps_the_model_loaded():
    from spark.brain.openai_chat import OLLAMA_KEEP_ALIVE

    script = Script(reply("ok"), reply("ok"), reply("ok"))
    await client(script, provider="ollama").stream("m", [])
    await client(script, provider="ollama", extra_body={"keep_alive": "2h"}).stream("m", [])
    await client(script, provider="openai").stream("m", [])
    assert script.bodies[0]["keep_alive"] == OLLAMA_KEEP_ALIVE
    assert script.bodies[1]["keep_alive"] == "2h"  # extra_body wins
    assert "keep_alive" not in script.bodies[2]  # other servers may reject unknown fields


async def test_tool_call_split_across_chunks():
    chat = client(Script(call("mcp__github__list_issues", {"state": "open"}, split=4)))
    started: list[int] = []
    result = await chat.stream("m", [], on_tool_start=started.append)
    assert started == [0]
    assert result.tool_calls[0].arguments == '{"state": "open"}'


@pytest.mark.parametrize(
    ("status", "kind", "words"),
    [
        (401, "auth", "API key"),
        (403, "auth", "API key"),
        (404, "model", "model name"),
        (400, "bad_request", "rejected"),
    ],
)
async def test_http_errors_become_spoken_provider_errors(status, kind, words):
    chat = client(Script(httpx.Response(status, json={"error": {"message": "nope sk-123"}})))
    with pytest.raises(ProviderError) as info:
        await chat.stream("m", [])
    assert info.value.kind == kind
    assert words in info.value.spoken
    assert failure_line(info.value) == info.value.spoken
    assert "sk-test-not-real" not in str(info.value)


async def test_model_without_tool_support_is_explained():
    body = {"error": {"message": "registry.ollama.ai/library/x does not support tools"}}
    chat = client(Script(httpx.Response(400, json=body)), provider="ollama")
    with pytest.raises(ProviderError) as info:
        await chat.stream("m", [])
    assert "tool calling" in info.value.spoken


async def test_429_is_retried_with_backoff_then_succeeds():
    clock = FakeClock()
    limited = httpx.Response(429, json={"error": {"message": "slow down"}})
    script = Script(limited, httpx.Response(429, headers={"retry-after": "3"}), reply("ok"))
    result = await client(script, clock=clock).stream("m", [])
    assert result.text == "ok"
    assert clock.sleeps == [1.0, 3.0]


async def test_429_gives_up_with_a_spoken_line():
    script = Script(*(httpx.Response(429) for _ in range(3)))
    with pytest.raises(ProviderError) as info:
        await client(script, clock=FakeClock()).stream("m", [])
    assert info.value.kind == "rate_limit"
    assert "rate limiting" in info.value.spoken
    assert len(script.bodies) == 3


async def test_server_error_retried_once():
    script = Script(httpx.Response(503), reply("fine"))
    assert (await client(script, clock=FakeClock()).stream("m", [])).text == "fine"


async def test_connection_refused_local_asks_if_running():
    script = Script(httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as info:
        await client(script, provider="ollama").stream("m", [])
    assert info.value.kind == "connect"
    assert info.value.spoken == "I can't reach Ollama on this computer. Is it running?"
    remote = Script(httpx.ConnectError("dns"))
    with pytest.raises(ProviderError) as info:
        await client(remote).stream("m", [])
    assert "internet" in info.value.spoken


async def test_error_inside_the_stream():
    script = Script(sse(text_chunk("Hi"), {"error": {"message": "overloaded"}}))
    with pytest.raises(ProviderError) as info:
        await client(script).stream("m", [])
    assert info.value.kind == "stream"


async def test_list_models_and_azure_auth():
    script = Script(models=["a", "b"])
    assert await client(script).list_models() == ["a", "b"]
    oc = OpenAICompatConfig(
        provider="azure", base_url="https://r.example/openai/v1", api_version="v", fast_model="d"
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=reply("ok"))

    chat = ChatClient(
        resolve_provider(oc),
        oc,
        transport=httpx.MockTransport(handler),
        environ={"AZURE_OPENAI_API_KEY": "az"},
    )
    await chat.stream("d", [])
    assert seen[0].headers["api-key"] == "az"
    assert "authorization" not in seen[0].headers
    assert seen[0].url.params["api-version"] == "v"


# --- names, schemas, results -------------------------------------------------------------------


def test_name_map_round_trip_and_long_names():
    long = "mcp__claude_ai_Google_Calendar__" + "x" * 80
    names = ["mcp__github__list_issues", "mcp__srv__has.dot", long, "research"]
    nm = ToolNameMap(names)
    for name in names:
        fn = nm.function(name)
        assert FUNCTION_NAME_RE.match(fn), fn
        assert nm.tool(fn) == name
    assert nm.function("mcp__github__list_issues") == "mcp__github__list_issues"
    assert ToolNameMap(names).function(long) == nm.function(long)  # stable across builds
    assert nm.tool("never_offered") is None


def test_name_map_collisions_stay_distinct():
    nm = ToolNameMap(["a.b", "a_b"])
    assert nm.function("a.b") != nm.function("a_b")
    assert {nm.tool(nm.function("a.b")), nm.tool(nm.function("a_b"))} == {"a.b", "a_b"}


def test_convert_schema():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"q": {"type": "string", "$comment": "x"}},
        "required": ["q"],
    }
    assert convert_schema(schema) == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    assert convert_schema({}) == {"type": "object", "properties": {}}
    assert convert_schema(None) == {"type": "object", "properties": {}}
    assert convert_schema({"type": "object", "required": "q"}) == {
        "type": "object",
        "properties": {},
    }
    tool = function_tool("f", "word " * 400, {"type": "object"})
    assert len(tool["function"]["description"]) <= 1000


def test_result_text_and_clip():
    blocks = [{"type": "text", "text": "a"}, {"type": "image", "data": "..."}]
    assert result_text(blocks) == "a\n[image]"
    assert result_text([], {"rows": 2}) == '{"rows": 2}'
    clipped = clip_result("x" * 1000, 600)
    assert clipped.startswith("x" * 600) and "400 more characters" in clipped
    assert clip_result("short", 600) == "short"
