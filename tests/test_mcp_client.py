"""McpManager: MCP sessions run by Spark for the OpenAI-compatible brain (P5.7e).

Fake clients for the state machine; one real stdio session against tests/mcp/echo_server.py to
prove the owner-task design works with the installed ``mcp`` client (calls from another task,
``${VAR}`` expanded, clean shutdown)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from spark.brain.mcp_client import InProcessServer, McpManager
from spark.tools.server import build_spark_tools
from spark.tools.timers import TimerService
from tests.openai_fakes import GITHUB_TOOLS, FakeMcpClient, factory

ECHO = Path(__file__).parent / "mcp" / "echo_server.py"


async def test_connects_lists_tools_calls_and_reports_status():
    gh = FakeMcpClient(GITHUB_TOOLS, results={"list_issues": "3 open issues"})
    statuses: list[dict] = []
    mgr = McpManager(
        {"github": {"type": "stdio", "command": "x"}},
        client_factory=factory({"github": gh}),
        on_status=statuses.append,
    )
    await mgr.start(wait_s=2)
    assert mgr.status == {"github": {"status": "connected"}}
    assert statuses[0] == {"github": {"status": "pending"}}
    assert statuses[-1] == {"github": {"status": "connected"}}
    names = [t.name for t in mgr.tools()]
    assert "mcp__github__list_issues" in names
    out = await mgr.call("mcp__github__list_issues", {"state": "open"})
    assert (out.text, out.is_error) == ("3 open issues", False)
    assert gh.calls == [("list_issues", {"state": "open"})]
    await mgr.aclose()
    assert gh.exited == 1


async def test_failed_connect_is_reported_and_retried_by_ensure():
    gh = FakeMcpClient(GITHUB_TOOLS, fail_connect=ConnectionRefusedError("connection refused"))
    now = [100.0]
    mgr = McpManager(
        {"github": {"type": "http", "url": "https://x"}},
        client_factory=factory({"github": gh}),
        retry_after_s=30,
        clock=lambda: now[0],
    )
    await mgr.start(wait_s=2)
    info = mgr.status["github"]
    assert info["status"] == "failed" and "reach" in info["error"]
    assert mgr.tools() == []
    out = await mgr.call("mcp__github__list_issues", {})
    assert out.is_error and "isn't connected" in out.text
    gh.fail_connect = None
    mgr.ensure()  # too soon: no retry
    await asyncio.sleep(0.01)
    assert mgr.status["github"]["status"] == "failed"
    now[0] += 31
    mgr.ensure()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.status["github"]["status"] == "connected":
            break
    assert mgr.status["github"]["status"] == "connected"
    await mgr.aclose()


async def test_broken_session_during_a_call_reconnects():
    gh = FakeMcpClient(GITHUB_TOOLS, fail_call=BrokenPipeError("pipe closed"))
    mgr = McpManager(
        {"github": {"type": "stdio", "command": "x"}}, client_factory=factory({"github": gh})
    )
    await mgr.start(wait_s=2)
    out = await mgr.call("mcp__github__list_issues", {})
    assert out.is_error and "failed" in out.text
    for _ in range(50):
        await asyncio.sleep(0.01)
        if gh.entered == 2 and mgr.status["github"]["status"] == "connected":
            break
    assert gh.entered == 2 and mgr.status["github"]["status"] == "connected"
    assert (await mgr.call("mcp__github__list_issues", {})).text == "list_issues done"
    await mgr.aclose()


async def test_in_process_tools_run_their_handlers():
    timers = TimerService(on_fire=lambda t: None)
    mgr = McpManager({"spark": InProcessServer(build_spark_tools(timers))})
    await mgr.start()
    names = {t.name for t in mgr.tools()}
    assert {"mcp__spark__set_timer", "mcp__spark__current_time"} <= names
    out = await mgr.call("mcp__spark__set_timer", {"duration": "5 minutes", "label": "tea"})
    assert not out.is_error and "5 minutes" in out.text
    err = await mgr.call("mcp__spark__set_timer", {})
    assert err.is_error
    assert mgr.status == {}  # Spark's own server is not a connector
    timers.cancel_all()
    await mgr.aclose()


async def test_real_stdio_session_with_expanded_env(tmp_path):
    server = {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(ECHO)],
        "env": {"ECHO_TOKEN": "${ECHO_TOKEN}"},
    }
    mgr = McpManager({"echo": server}, base_dir=tmp_path, environ={"ECHO_TOKEN": "abc12345"})
    try:
        await mgr.start(wait_s=25)
        assert mgr.status["echo"]["status"] == "connected", mgr.status
        # Called from a different task than the owner that opened the session.
        out = await asyncio.create_task(mgr.call("mcp__echo__echo", {"text": "ping"}))
        assert out.text == "ping"
        assert (await mgr.call("mcp__echo__token_seen", {})).text == "yes"
    finally:
        await mgr.aclose()
