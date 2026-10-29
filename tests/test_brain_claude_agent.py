"""ClaudeAgentBrain with a fake ``claude_agent_sdk`` (offline) plus live tests (``-m live``).

The fake mirrors the verified shapes in docs/sdk_notes.md §1: ``StreamEvent.event`` is the raw
Anthropic stream event, ``AssistantMessage.content`` holds ``TextBlock``s, and
``receive_response()`` stops after the ``ResultMessage``. The fake client also asserts that it
is only ever used from the task that connected it (the SDK's anyio task-group constraint).
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from spark.brain.claude_agent import BrainClosed, BrainError, ClaudeAgentBrain
from spark.brain.confirm import DENY_REASONS
from spark.brain.mcp_config import McpSetup
from spark.brain.prompts import (
    FRESH_START_ACK,
    MARKDOWN_CHARS,
    NO_TOOLS_NOTE,
    VOICE_SYSTEM_PROMPT,
    voice_system_prompt,
)
from spark.config import Settings
from spark.interfaces import BackendUnavailable
from spark.tools.server import spark_tool_names
from spark.tools.timers import TimerService
from tests.fakes import FakeClock

# --- fake SDK ---------------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AssistantMessage:
    content: list[Any]
    model: str
    parent_tool_use_id: str | None = None
    error: str | None = None


@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]
    parent_tool_use_id: str | None = None


@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    errors: list[str] | None = None
    api_error_status: int | None = None
    terminal_reason: str | None = None


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any] = field(default_factory=dict)


class ClaudeSDKError(Exception):
    pass


class CLIConnectionError(ClaudeSDKError):
    pass


class CLINotFoundError(CLIConnectionError):
    pass


class ClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def text_start(index: int = 0) -> StreamEvent:
    return StreamEvent(
        "u",
        "s",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    )


def text_delta(text: str, index: int = 0) -> StreamEvent:
    return StreamEvent(
        "u",
        "s",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def result(**kw: Any) -> ResultMessage:
    base: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 900,
        "duration_api_ms": 700,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sess-1",
        "total_cost_usd": 0.001,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    base.update(kw)
    return ResultMessage(**base)


def streamed(*chunks: str) -> list[Any]:
    """A normal partial-streaming turn: stream events, then the whole message, then result."""
    msgs: list[Any] = [
        SystemMessage("init"),
        StreamEvent("u", "s", {"type": "message_start", "message": {}}),
        text_start(),
        *[text_delta(c) for c in chunks],
        StreamEvent("u", "s", {"type": "content_block_stop", "index": 0}),
        AssistantMessage([TextBlock("".join(chunks))], "claude-haiku"),
        result(),
    ]
    return msgs


@dataclass
class HookCall:
    """A script item: the CLI calls the ``PreToolUse`` hook from the options for one tool call
    while the turn is streaming (as the real CLI does before running a tool)."""

    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = "toolu_h"
    output: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. agent_id for a worker's call

    async def run(self, options: Any) -> None:
        (matcher,) = options.hooks["PreToolUse"]
        (hook,) = matcher.hooks
        data = {
            "session_id": "s",
            "transcript_path": "t",
            "cwd": ".",
            "hook_event_name": "PreToolUse",
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_use_id": self.tool_use_id,
            **self.extra,
        }
        self.output = await hook(data, self.tool_use_id, {"signal": None})


Script = Callable[[str, "FakeClient"], list[Any]]


class FakeClient:
    instances: ClassVar[list[FakeClient]] = []
    script: Script = staticmethod(lambda text, client: streamed("You said ", text, "."))
    connect_error: BaseException | None = None
    connect_delay: float = 0.0
    step_delay: float = 0.0
    set_model_error: BaseException | None = None
    mcp_status: dict[str, Any] | None = None  # what get_mcp_status() returns (P5.7i)

    def __init__(self, options: ClaudeAgentOptions | None = None) -> None:
        self.options = options
        self.connected = False
        self.disconnected = False
        self.queries: list[str] = []
        self.interrupts = 0
        self.models: list[str | None] = []  # set_model() calls, in order
        self.owner: asyncio.Task[Any] | None = None
        self._pending: list[Any] = []
        self._interrupted = asyncio.Event()
        FakeClient.instances.append(self)

    def _check_task(self) -> None:
        assert asyncio.current_task() is self.owner, "client used outside its connecting task"

    async def connect(self, prompt: Any = None) -> None:
        self.owner = asyncio.current_task()
        if FakeClient.connect_delay:
            await asyncio.sleep(FakeClient.connect_delay)
        if FakeClient.connect_error is not None:
            raise FakeClient.connect_error
        self.connected = True

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self._check_task()
        assert self.connected and not self.disconnected
        self.queries.append(prompt)
        self._interrupted = asyncio.Event()
        self._pending = list(FakeClient.script(prompt, self))

    async def set_model(self, model: str | None = None) -> None:
        self._check_task()
        assert self.connected and not self.disconnected
        if FakeClient.set_model_error is not None:
            raise FakeClient.set_model_error
        self.models.append(model)

    async def receive_response(self):
        self._check_task()
        for msg in self._pending:
            if self._interrupted.is_set():
                break
            if isinstance(msg, asyncio.Event):  # a script can hold the stream until released
                await msg.wait()
                continue
            if isinstance(msg, HookCall):  # the CLI asks the PreToolUse hook mid-turn
                await msg.run(self.options)
                continue
            if FakeClient.step_delay:
                try:
                    await asyncio.wait_for(self._interrupted.wait(), FakeClient.step_delay)
                    break
                except TimeoutError:
                    pass
            else:
                await asyncio.sleep(0)
            yield msg
            if isinstance(msg, ResultMessage):
                return
        else:
            return
        yield result(
            subtype="error_during_execution", is_error=True, terminal_reason="aborted_streaming"
        )

    async def get_mcp_status(self) -> dict[str, Any]:
        self._check_task()
        return FakeClient.mcp_status or {"mcpServers": []}

    async def interrupt(self) -> None:
        self.interrupts += 1
        self._interrupted.set()

    async def disconnect(self) -> None:
        self._check_task()
        self.disconnected = True
        self.connected = False


def make_fake_sdk() -> types.ModuleType:
    # The pure-Python helpers come from the real SDK (no CLI involved): the brain passes a real
    # HookMatcher, and the spark server is built with the real @tool / create_sdk_mcp_server.
    import claude_agent_sdk as real

    mod = types.ModuleType("claude_agent_sdk")
    for name in (
        "HookMatcher",
        "ToolAnnotations",
        "tool",
        "create_sdk_mcp_server",
        "AgentDefinition",
    ):
        setattr(mod, name, getattr(real, name))
    for obj in (
        TextBlock,
        ToolUseBlock,
        AssistantMessage,
        StreamEvent,
        ResultMessage,
        SystemMessage,
        ClaudeSDKError,
        CLIConnectionError,
        CLINotFoundError,
        ClaudeAgentOptions,
    ):
        setattr(mod, obj.__name__, obj)
    mod.ClaudeSDKClient = FakeClient
    return mod


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeClient.instances = []
    FakeClient.script = staticmethod(lambda text, client: streamed("You said ", text, "."))
    FakeClient.connect_error = None
    FakeClient.connect_delay = 0.0
    FakeClient.step_delay = 0.0
    FakeClient.set_model_error = None
    FakeClient.mcp_status = None
    mod = make_fake_sdk()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


# No connectors unless a test asks: the repo's mcp_servers.json must not leak into unit tests.
NO_CONNECTORS = {
    "mcp_config": "no-such-mcp-servers.json",
    "ui_file": "no-such-connectors.json",
    "servers": {},
}


def settings(connectors: dict[str, Any] | None = None, **brain: Any) -> Settings:
    brain.setdefault("prime_on_connect", False)  # priming has its own tests below
    brain.setdefault("mode", "auto")  # these tests are about routing; deep mode has its own
    brain.setdefault("workers", False)  # the researcher worker has its own tests
    return Settings(
        brain={"backend": "claude_agent", **brain},
        connectors=connectors if connectors is not None else NO_CONNECTORS,
    )


@pytest.fixture
async def brain(fake_sdk):
    clock = FakeClock(start=1000.0)
    b = ClaudeAgentBrain(settings(), clock=clock)
    b.test_clock = clock  # type: ignore[attr-defined]
    yield b
    await b.close()


async def collect(brain: ClaudeAgentBrain, text: str) -> list[str]:
    return [d async for d in brain.respond(text)]


# --- construction and options ------------------------------------------------------------------


def test_import_does_not_pull_in_the_sdk():
    code = (
        "import sys, spark.brain.claude_agent; "
        "assert 'claude_agent_sdk' not in sys.modules, 'SDK imported at module level'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert proc.returncode == 0, proc.stderr


def test_missing_sdk_raises_backend_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # makes import raise
    with pytest.raises(BackendUnavailable) as info:
        ClaudeAgentBrain(settings())
    assert info.value.backend == "claude_agent"
    assert "uv sync" in info.value.fix


def test_build_options(fake_sdk):
    b = ClaudeAgentBrain(
        settings(fast_model="claude-haiku-4-5-20251001", max_turns=3, deep_max_turns=10)
    )
    opts = b._build_options()
    assert isinstance(opts, ClaudeAgentOptions)
    assert opts.tools == []
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert opts.include_partial_messages is True
    assert opts.model == "claude-haiku-4-5-20251001"
    # no servers at all (no connectors, no timers dep): the no-tools variant of the prompt
    assert opts.system_prompt == voice_system_prompt(tools_available=False)
    assert NO_TOOLS_NOTE in opts.system_prompt
    assert opts.mcp_servers == {} and opts.allowed_tools == []
    assert opts.env == {"ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
    assert opts.max_turns == 12  # backstop above the step budgets (the gate answers first)
    assert "cli_path" not in opts.kwargs  # bundled CLI is used unless configured
    assert callable(opts.stderr)


def test_build_options_cli_path_only_when_configured(fake_sdk):
    b = ClaudeAgentBrain(settings(cli_path="C:/tools/claude.exe"))
    assert b._build_options().cli_path == "C:/tools/claude.exe"


def test_registry_builds_the_brain(fake_sdk):
    from spark import registry

    b = registry.build("brain", settings())
    assert isinstance(b, ClaudeAgentBrain)


# --- MCP wiring (P2.4) ---------------------------------------------------------------------------

GITHUB_STDIO = {"type": "stdio", "command": "C:/bin/github-mcp-server.exe", "args": ["stdio"]}


def github_setup() -> McpSetup:
    """An McpSetup as load_mcp_servers would build it for a github server."""
    setup = McpSetup()
    setup.servers["github"] = dict(GITHUB_STDIO)
    setup._register(
        "github",
        allowed=["list_pull_requests", "issue_read"],
        side_effecting=["create_repository"],
        destructive=["merge_pull_request", "delete_*"],
        allow_destructive=False,
        known=(),
    )
    return setup


async def test_options_carry_mcp_servers_spark_server_allowlist_and_env(fake_sdk):
    timers = TimerService(clock=FakeClock())
    b = ClaudeAgentBrain(settings(), mcp=github_setup(), timers=timers)
    try:
        opts = b._build_options()
        assert set(opts.mcp_servers) == {"github", "spark"}
        assert opts.mcp_servers["github"] == GITHUB_STDIO
        spark = opts.mcp_servers["spark"]
        assert spark["type"] == "sdk" and spark["name"] == "spark" and "instance" in spark
        # exact names only: the external read-only tools plus every spark tool
        assert opts.allowed_tools == [
            "mcp__github__issue_read",
            "mcp__github__list_pull_requests",
            *sorted(spark_tool_names()),
        ]
        assert not [t for t in opts.allowed_tools if any(c in t for c in "*?[")]
        assert "mcp__github__create_repository" not in opts.allowed_tools  # side-effecting
        assert opts.disallowed_tools == [
            "mcp__github__merge_pull_request",
            "mcp__github__delete_*",
        ]
        assert opts.env == {"ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
        # still no built-in Claude Code tools and no inherited settings
        assert opts.tools == [] and opts.setting_sources == [] and opts.strict_mcp_config
        (matcher,) = opts.hooks["PreToolUse"]
        assert matcher.matcher is None and matcher.hooks == [b.tool_gate]
        # the CLI's hook budget (default 60 s) must outlast a spoken question plus the answer
        assert matcher.timeout == b.settings.confirm.hook_timeout_s == 120.0
        assert opts.system_prompt == VOICE_SYSTEM_PROMPT  # tools exist: Phase 2 prompt
        assert b.connectors == ["github", "spark"]
    finally:
        await b.close()
        await timers.close()


async def test_spark_server_tools_are_the_real_handlers(fake_sdk):
    timers = TimerService(clock=FakeClock())
    b = ClaudeAgentBrain(settings(), timers=timers, status_provider=lambda: {"state": "idle"})
    try:
        server = b._build_options().mcp_servers["spark"]["instance"]
        assert server.name == "spark"
        assert b.mcp.policy.classify("mcp__spark__set_timer") == "read"
    finally:
        await b.close()
        await timers.close()


def _hook_input(name: str) -> dict[str, Any]:
    return {
        "session_id": "s",
        "transcript_path": "t",
        "cwd": ".",
        "hook_event_name": "PreToolUse",
        "tool_name": name,
        "tool_input": {"secret_arg": "jane@example.com"},
        "tool_use_id": "toolu_1",
    }


@pytest.mark.parametrize(
    ("tool", "kind"),
    [
        ("mcp__github__create_repository", "side_effect"),
        ("mcp__github__merge_pull_request", "destructive"),
        ("mcp__github__delete_file", "destructive"),
        ("mcp__github__some_new_tool", "unknown"),
        ("mcp__other__list_things", "unknown"),
        ("Bash", "unknown"),
        ("", "unknown"),
    ],
)
async def test_tool_gate_denies_everything_not_pre_approved(fake_sdk, caplog, tool, kind):
    b = ClaudeAgentBrain(settings(), mcp=github_setup())
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    out = await b.tool_gate(_hook_input(tool), "toolu_1", {"signal": None})
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": DENY_REASONS[kind],
        }
    }
    assert f"tool denied: {tool or '<unnamed>'} ({kind})" in caplog.text
    assert "jane@example.com" not in caplog.text  # never log arguments


async def test_tool_gate_passes_read_only_tools_to_the_allowlist(fake_sdk):
    timers = TimerService(clock=FakeClock())
    b = ClaudeAgentBrain(settings(), mcp=github_setup(), timers=timers)
    try:
        for tool in ("mcp__github__list_pull_requests", "mcp__spark__current_time"):
            assert await b.tool_gate(_hook_input(tool), "toolu_1", {"signal": None}) == {}
    finally:
        await timers.close()


def _tool_start(tool_id: str, index: int = 1) -> StreamEvent:
    return StreamEvent(
        "u",
        "s",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": tool_id, "name": "mcp__x__y", "input": {}},
        },
    )


async def test_tool_call_in_the_stream_flushes_text_and_releases_the_gate(fake_sdk):
    """P3.1 ordering: text before a tool call gets a sentence break (so the app's splitter
    releases it), and the gate's before_ask wakes once the call has been seen in the stream."""
    FakeClient.script = staticmethod(
        lambda text, client: [
            SystemMessage("init"),
            text_start(),
            text_delta("Sure, I'll set that up"),
            _tool_start("toolu_9"),
            AssistantMessage(
                [TextBlock("Sure, I'll set that up"), ToolUseBlock("toolu_9", "mcp__x__y", {})],
                "m",
            ),
            text_start(2),
            text_delta("Done."),
            result(num_turns=2),
        ]
    )
    b = ClaudeAgentBrain(settings(), mcp=McpSetup())
    b.tool_announce_wait_s = 5.0
    try:
        waiter = asyncio.create_task(b._await_tool_announced("toolu_9"))
        deltas = await collect(b, "set it up")
        await asyncio.wait_for(waiter, 1.0)  # released by the stream, not by the 5 s timeout
        assert deltas == ["Sure, I'll set that up", "\n", "Done."]
        assert b._tool_seen == {}  # cleared at the end of the turn
    finally:
        await b.close()


async def test_before_ask_gives_up_quickly_when_the_call_is_never_seen(fake_sdk):
    b = ClaudeAgentBrain(settings(), mcp=McpSetup())
    b.tool_announce_wait_s = 0.01
    await asyncio.wait_for(b._await_tool_announced("toolu_missing"), 1.0)
    await asyncio.wait_for(b._await_tool_announced(None), 1.0)


async def test_no_sentence_break_for_a_tool_call_before_any_text(fake_sdk):
    FakeClient.script = staticmethod(
        lambda text, client: [
            SystemMessage("init"),
            _tool_start("toolu_1", index=0),
            text_start(1),
            text_delta("It is noon."),
            result(),
        ]
    )
    b = ClaudeAgentBrain(settings(), mcp=McpSetup())
    try:
        assert await collect(b, "time?") == ["It is noon."]
    finally:
        await b.close()


async def test_brain_builds_the_confirmation_gate_with_the_confirmer(fake_sdk):
    from spark.brain.confirm import ConfirmationGate
    from tests.fakes import FakeConfirmer

    confirmer = FakeConfirmer([False])
    timers = TimerService(clock=FakeClock())
    b = ClaudeAgentBrain(settings(), mcp=github_setup(), timers=timers, confirmer=confirmer)
    b.tool_announce_wait_s = 0.01
    try:
        assert isinstance(b.tool_gate, ConfirmationGate)
        assert b.tool_gate.confirmer is confirmer
        assert b.tool_gate.ask_timeout_s == b.settings.confirm.ask_budget_s
        assert "mcp__spark__start_project" in b.tool_gate.summarizers
        out = await b.tool_gate(
            {**_hook_input("mcp__github__create_repository"), "tool_input": {"name": "wb"}},
            "toolu_1",
            {"signal": None},
        )
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert confirmer.asked == ["I'll create a GitHub repo called wb. Should I go ahead?"]
    finally:
        await b.close()
        await timers.close()


def test_mcp_loads_from_settings_and_disables_servers_missing_env(
    fake_sdk, tmp_path, monkeypatch, caplog
):
    (tmp_path / "mcp_servers.json").write_text(
        json.dumps(
            {
                "calendar": {
                    "type": "http",
                    "url": "https://mcp.example.test/cal",
                    "headers": {"x-api-key": "${SPARK_TEST_CAL_KEY}"},
                },
                "mail": {
                    "type": "http",
                    "url": "https://mcp.example.test/mail",
                    "headers": {"x-api-key": "${SPARK_TEST_MAIL_KEY}"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SPARK_TEST_CAL_KEY", "super-secret-value")
    monkeypatch.delenv("SPARK_TEST_MAIL_KEY", raising=False)
    connectors = {
        "servers": {
            "calendar": {"enabled": True, "allowed_tools": ["EVENTS_LIST"]},
            "mail": {"enabled": True, "allowed_tools": ["FETCH_EMAILS"]},
        }
    }
    caplog.set_level(logging.INFO)
    s = Settings(
        brain={"backend": "claude_agent", "prime_on_connect": False, "workers": False},
        connectors=connectors,
        config_dir=tmp_path,
    )
    opts = ClaudeAgentBrain(s)._build_options()
    assert set(opts.mcp_servers) == {"calendar"}
    # the placeholder goes to the CLI unexpanded; the secret never enters argv or logs
    assert opts.mcp_servers["calendar"]["headers"] == {"x-api-key": "${SPARK_TEST_CAL_KEY}"}
    assert opts.allowed_tools == ["mcp__calendar__EVENTS_LIST"]
    assert "MCP servers enabled: calendar" in caplog.text
    assert "MCP servers disabled: mail" in caplog.text
    assert "SPARK_TEST_MAIL_KEY" in caplog.text  # the reason names the variable
    assert "super-secret-value" not in caplog.text


def test_broken_mcp_json_runs_without_connectors(fake_sdk, tmp_path, caplog):
    (tmp_path / "mcp_servers.json").write_text("{not json", encoding="utf-8")
    s = Settings(
        brain={"backend": "claude_agent", "prime_on_connect": False},
        connectors={"servers": {}},
        config_dir=tmp_path,
    )
    b = ClaudeAgentBrain(s)
    assert b._build_options().mcp_servers == {}
    assert "running without external connectors" in caplog.text


async def test_tool_calls_are_logged_by_name_only(brain, caplog):
    FakeClient.script = staticmethod(
        lambda t, c: [
            AssistantMessage(
                [ToolUseBlock("toolu_1", "mcp__github__list_pull_requests", {"repo": "acme/x"})],
                "m",
            ),
            text_start(),
            text_delta("You have two open pull requests."),
            AssistantMessage([TextBlock("You have two open pull requests.")], "m"),
            result(num_turns=2),
        ]
    )
    caplog.set_level(logging.INFO, logger="spark.brain.claude_agent")
    assert "".join(await collect(brain, "open PRs?")) == "You have two open pull requests."
    assert "tool call: mcp__github__list_pull_requests" in caplog.text
    assert "acme/x" not in caplog.text


# --- streaming ----------------------------------------------------------------------------------


async def test_streams_text_deltas_in_order(brain):
    FakeClient.script = staticmethod(lambda t, c: streamed("Tokyo ", "is the ", "capital."))
    deltas = await collect(brain, "What's the capital of Japan?")
    assert deltas == ["Tokyo ", "is the ", "capital."]
    (client,) = FakeClient.instances
    assert client.queries == ["What's the capital of Japan?"]
    stats = brain.last_turn
    assert stats is not None and stats.partial and not stats.is_error
    assert stats.ttft_s is not None and stats.duration_ms == 900
    assert stats.usage == {"input_tokens": 10, "output_tokens": 5}
    assert stats.as_meta()["brain_output_tokens"] == 5


async def test_first_delta_arrives_before_the_turn_ends(brain):
    FakeClient.step_delay = 0.01
    FakeClient.script = staticmethod(lambda t, c: streamed("First. ", "Second."))
    gen = brain.respond("hi")
    first = await gen.__anext__()
    assert first == "First. "
    assert brain.last_turn is None  # the ResultMessage hasn't arrived yet
    rest = [d async for d in gen]
    assert rest == ["Second."]


async def test_one_client_is_reused_across_turns(brain):
    await brain.warmup()
    await collect(brain, "one")
    await collect(brain, "two")
    assert len(FakeClient.instances) == 1
    assert FakeClient.instances[0].queries == ["one", "two"]


async def test_falls_back_to_assistant_text_blocks_without_partials(brain):
    FakeClient.script = staticmethod(
        lambda t, c: [
            SystemMessage("init"),
            AssistantMessage([TextBlock("Let me think.")], "m"),
            AssistantMessage([ToolUseBlock("t1", "x", {}), TextBlock("It is Tokyo.")], "m"),
            result(num_turns=2),
        ]
    )
    deltas = await collect(brain, "q")
    # the tool call ends the sentence before it, so a confirmation question comes after it
    assert "".join(deltas) == "Let me think.\nIt is Tokyo."
    assert brain.last_turn is not None and not brain.last_turn.partial


async def test_partials_are_not_duplicated_by_the_assistant_message(brain):
    deltas = await collect(brain, "abc")
    assert "".join(deltas) == "You said abc."


async def test_separate_text_blocks_get_a_space(brain):
    FakeClient.script = staticmethod(
        lambda t, c: [
            text_start(0),
            text_delta("Checking.", 0),
            text_start(1),
            text_delta("Done.", 1),
            result(),
        ]
    )
    assert "".join(await collect(brain, "q")) == "Checking. Done."


async def test_subagent_events_are_ignored(brain):
    FakeClient.script = staticmethod(
        lambda t, c: [
            StreamEvent("u", "s", text_delta("secret").event, parent_tool_use_id="tool-1"),
            text_delta("Visible."),
            result(),
        ]
    )
    assert await collect(brain, "q") == ["Visible."]


async def test_is_error_result_raises_brain_error(brain):
    FakeClient.script = staticmethod(
        lambda t, c: [
            AssistantMessage([TextBlock("Invalid API key")], "m", error="authentication_failed"),
            result(
                subtype="success",
                is_error=True,
                result="Invalid API key · Please run /login",
                api_error_status=401,
            ),
        ]
    )
    with pytest.raises(BrainError) as info:
        await collect(brain, "q")
    assert "Invalid API key" in str(info.value) and "401" in str(info.value)
    assert brain.last_turn is not None and brain.last_turn.is_error
    # The brain stays usable afterwards.
    FakeClient.script = staticmethod(lambda t, c: streamed("ok"))
    assert await collect(brain, "again") == ["ok"]


async def test_stream_failure_raises_and_reconnects_next_turn(brain):
    def boom(t, c):
        raise RuntimeError("pipe closed")

    FakeClient.script = staticmethod(boom)
    with pytest.raises(BrainError, match="pipe closed"):
        await collect(brain, "q")
    FakeClient.script = staticmethod(lambda t, c: streamed("back"))
    assert await collect(brain, "q2") == ["back"]
    assert len(FakeClient.instances) == 2 and FakeClient.instances[0].disconnected


async def test_missing_result_message_is_an_error(brain):
    FakeClient.script = staticmethod(lambda t, c: [text_delta("half")])
    with pytest.raises(BrainError, match="without a result"):
        await collect(brain, "q")


# --- connect errors -----------------------------------------------------------------------------


async def test_cli_not_found_becomes_backend_unavailable(brain):
    FakeClient.connect_error = CLINotFoundError("claude not found")
    with pytest.raises(BackendUnavailable) as info:
        await brain.warmup()
    assert "cli_path" in info.value.fix


async def test_connect_error_during_respond_is_raised(brain):
    FakeClient.connect_error = CLIConnectionError("spawn failed")
    with pytest.raises(BackendUnavailable):
        await collect(brain, "q")
    FakeClient.connect_error = None
    assert await collect(brain, "q") == ["You said ", "q", "."]


async def test_connect_timeout(fake_sdk):
    FakeClient.connect_delay = 5
    b = ClaudeAgentBrain(settings(connect_timeout_s=0.05))
    try:
        with pytest.raises(BackendUnavailable, match="timed out"):
            await b.warmup()
    finally:
        await b.close()


# --- idle reset and new topic -------------------------------------------------------------------


async def test_idle_reset_starts_a_fresh_client(brain):
    clock: FakeClock = brain.test_clock
    await collect(brain, "one")
    clock.advance(599)
    await collect(brain, "two")
    assert len(FakeClient.instances) == 1
    clock.advance(601)
    await collect(brain, "three")
    assert len(FakeClient.instances) == 2
    old, new = FakeClient.instances
    assert old.disconnected and old.queries == ["one", "two"]
    assert new.queries == ["three"]
    assert brain.resets == 1


async def test_new_topic_resets_and_acknowledges_without_the_model(brain):
    await collect(brain, "My favourite colour is teal.")
    deltas = await collect(brain, "Spark, new topic")
    assert deltas == [FRESH_START_ACK]
    assert brain.resets == 1
    assert FakeClient.instances[0].disconnected
    assert len(FakeClient.instances) == 2  # reconnected eagerly, kept warm
    assert FakeClient.instances[1].queries == []  # the model was not called


async def test_new_topic_with_a_question_answers_the_rest(brain):
    await collect(brain, "earlier")
    deltas = await collect(brain, "New topic. What's two plus two?")
    assert "".join(deltas) == "You said What's two plus two?."
    assert FakeClient.instances[-1].queries == ["What's two plus two?"]
    assert brain.resets == 1


async def test_new_topic_before_any_connection(brain):
    assert await collect(brain, "new topic please") == [FRESH_START_ACK]
    assert FakeClient.instances == []


async def test_explicit_reset_reconnects(brain):
    await brain.warmup()
    await brain.reset()
    assert len(FakeClient.instances) == 2
    assert FakeClient.instances[0].disconnected and FakeClient.instances[1].connected


# --- interrupt and close ------------------------------------------------------------------------


async def test_interrupt_stops_the_turn_and_drains(brain):
    FakeClient.step_delay = 0.02
    FakeClient.script = staticmethod(lambda t, c: streamed(*[f"w{i} " for i in range(200)]))
    gen = brain.respond("long story")
    first = await gen.__anext__()
    assert first == "w0 "
    await brain.interrupt()
    client = FakeClient.instances[0]
    assert client.interrupts == 1
    assert brain.last_turn is not None and brain.last_turn.interrupted
    assert brain.last_turn.terminal_reason == "aborted_streaming"
    # The consumer sees the stream end quietly (no error for an aborted turn).
    rest = [d async for d in gen]
    assert len(rest) < 199
    # The next turn works on the same client.
    FakeClient.step_delay = 0.0
    FakeClient.script = staticmethod(lambda t, c: streamed("next"))
    assert await collect(brain, "next") == ["next"]
    assert len(FakeClient.instances) == 1


async def test_abandoning_the_stream_interrupts_generation(brain):
    FakeClient.step_delay = 0.02
    FakeClient.script = staticmethod(lambda t, c: streamed(*[f"w{i} " for i in range(200)]))
    gen = brain.respond("long")
    await gen.__anext__()
    await gen.aclose()
    await brain.interrupt()  # waits for the drain
    assert FakeClient.instances[0].interrupts == 1


async def test_interrupt_when_idle_is_a_no_op(brain):
    await brain.interrupt()
    await collect(brain, "q")
    await brain.interrupt()
    assert FakeClient.instances[0].interrupts == 0


async def test_close_disconnects_and_rejects_new_requests(brain):
    await collect(brain, "q")
    await brain.close()
    assert FakeClient.instances[0].disconnected
    assert brain._worker is not None and brain._worker.done()
    with pytest.raises(BrainClosed):
        await collect(brain, "after close")
    await brain.close()  # idempotent


async def test_close_mid_turn_is_clean(brain):
    FakeClient.step_delay = 0.02
    FakeClient.script = staticmethod(lambda t, c: streamed(*[f"w{i} " for i in range(200)]))
    gen = brain.respond("long")
    await gen.__anext__()
    await brain.close()
    assert FakeClient.instances[0].disconnected
    rest = [d async for d in gen]
    assert len(rest) < 199


async def test_close_without_warnings(fake_sdk, recwarn):
    b = ClaudeAgentBrain(settings())
    await b.warmup()
    await collect(b, "q")
    await b.close()
    gc.collect()
    await asyncio.sleep(0)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert not [t for t in tasks if t.get_name() == "spark-brain-session"]
    assert not [w for w in recwarn if issubclass(w.category, (RuntimeWarning, ResourceWarning))]


def test_close_leaves_no_pending_tasks_when_the_loop_closes(fake_sdk):
    async def main() -> None:
        b = ClaudeAgentBrain(settings())
        await collect(b, "q")
        await b.close()

    loop = asyncio.new_event_loop()
    errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            loop.run_until_complete(main())
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            assert pending == []
        finally:
            loop.close()
    gc.collect()
    assert errors == []


async def test_turns_are_serialized(brain):
    FakeClient.step_delay = 0.005
    FakeClient.script = staticmethod(lambda t, c: streamed(f"{t}-a ", f"{t}-b"))
    a, b = await asyncio.gather(collect(brain, "x"), collect(brain, "y"))
    assert a == ["x-a ", "x-b"] and b == ["y-a ", "y-b"]


# --- live (real Claude) --------------------------------------------------------------------------


def _live_settings() -> Settings:
    from spark.config import load_settings

    return load_settings()


def _assert_no_markdown(text: str) -> None:
    leaked = sorted(set(text) & MARKDOWN_CHARS)
    assert not leaked, f"markdown characters {leaked} in {text!r}"
    assert "http" not in text.lower()


@pytest.mark.live
async def test_live_capital_of_japan():
    if not (shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY")):
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    brain = ClaudeAgentBrain(_live_settings())
    try:
        t0 = asyncio.get_running_loop().time()
        await brain.warmup()
        connect_s = asyncio.get_running_loop().time() - t0
        text = "".join([d async for d in brain.respond("What's the capital of Japan?")])
        stats = brain.last_turn
        assert stats is not None and stats.ttft_s is not None
        print(
            f"\n[live] connect={connect_s:.2f}s ttft={stats.ttft_s * 1000:.0f}ms "
            f"total={stats.total_s * 1000:.0f}ms api={stats.duration_api_ms}ms "
            f"partial={stats.partial} model={stats.model} reply={text!r}"
        )
        assert "Tokyo" in text
        _assert_no_markdown(text)
        assert stats.partial, "expected token-level StreamEvent deltas"
    finally:
        await brain.close()


@pytest.mark.live
async def test_live_two_turn_context():
    if not (shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY")):
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    brain = ClaudeAgentBrain(_live_settings())
    try:
        await brain.warmup()
        first = "".join([d async for d in brain.respond("My favourite colour is teal.")])
        second = "".join([d async for d in brain.respond("What's my favourite colour?")])
        print(f"\n[live] turn1={first!r} turn2={second!r}")
        assert "teal" in second.lower()
        _assert_no_markdown(first + second)
    finally:
        await brain.close()


# --- priming -----------------------------------------------------------------------------------


async def test_prime_on_connect_sends_throwaway_turn(fake_sdk):
    from spark.brain.claude_agent import PRIME_TEXT

    b = ClaudeAgentBrain(settings(prime_on_connect=True), clock=FakeClock(start=1000.0))
    try:
        await b.warmup()
        client = FakeClient.instances[0]
        assert client.queries == [PRIME_TEXT]
        assert b.last_turn is None  # priming doesn't count as a user turn
        assert await collect(b, "hello") == ["You said ", "hello", "."]
        assert client.queries == [PRIME_TEXT, "hello"]
    finally:
        await b.close()


async def test_prime_failure_is_not_fatal(fake_sdk):
    calls = {"n": 0}

    def script(text, client):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("prime blew up")
        return streamed("ok")

    FakeClient.script = staticmethod(script)
    b = ClaudeAgentBrain(settings(prime_on_connect=True), clock=FakeClock(start=1000.0))
    try:
        await b.warmup()  # must not raise
        assert "".join(await collect(b, "hi")) == "ok"
    finally:
        await b.close()


async def test_prime_never_runs_inside_a_user_turn(fake_sdk):
    """A reconnect during a turn (e.g. after an error) must not prime or fail the turn."""

    def script(text, client):
        if "Connection check" in text:
            raise RuntimeError("priming always fails")
        return streamed("ok")

    FakeClient.script = staticmethod(script)
    b = ClaudeAgentBrain(settings(prime_on_connect=True), clock=FakeClock(start=1000.0))
    try:
        # no warmup(): the first turn connects inside the turn and must answer normally
        assert "".join(await collect(b, "hi")) == "ok"
        assert all("Connection check" not in q for q in FakeClient.instances[-1].queries)
    finally:
        await b.close()


# --- live: MCP tool results are spoken without raw IDs or URLs (spec §10 Phase 2) ---------------

RAW_PRS = {
    "total_count": 3,
    "items": [
        {
            "number": 418,
            "id": 2187654321,
            "node_id": "PR_kwDOK3x9Ys5nQ2Jb",
            "title": "Add dark mode toggle",
            "user": {"login": "sam-dev", "email": "sam.dev@example.com"},
            "created_at": "2026-09-23T16:42:10Z",
            "html_url": "https://github.com/acme/weather-bot/pull/418",
            "head": {"ref": "feat/dark-mode", "sha": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b"},
            "trace_id": "01J8ZK4Q6V2W3X4Y5Z6A7B8C9D",
            "request_id": "123e4567-e89b-12d3-a456-426614174000",
        },
        {
            "number": 415,
            "id": 2187650007,
            "node_id": "PR_kwDOK3x9Ys5nQ1Zz",
            "title": "Fix login timeout",
            "user": {"login": "priya", "email": "priya@example.com"},
            "created_at": "2026-09-21T09:05:00Z",
            "html_url": "https://github.com/acme/weather-bot/pull/415",
            "head": {"ref": "fix/login", "sha": "3f2a9c1e5b7d9f0a1c3e5b7d9f0a1c3e5b7d9f0a"},
        },
        {
            "number": 409,
            "id": 2187640123,
            "node_id": "PR_kwDOK3x9Ys5nP9Aa",
            "title": "Bump dependencies",
            "user": {"login": "renovate[bot]", "email": "bot@renovateapp.com"},
            "created_at": "2026-09-18T02:00:00Z",
            "html_url": "https://github.com/acme/weather-bot/pull/409",
            "head": {"ref": "renovate/all", "sha": "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"},
        },
    ],
}

SPOKEN_LEAKS = re.compile(
    r"https?://|www\.|github\.com|@|PR_kw|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}\b|"
    r"\b01J8ZK|2187\d{6}",
    re.IGNORECASE,
)
COUNT_RE = re.compile(r"\b(?:3|three)\b", re.IGNORECASE)


def _live_available() -> bool:
    return bool(shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY"))


def _fake_pr_server(calls: list[str]) -> McpSetup:
    """An in-process SDK server standing in for a GitHub connector: one read-only tool that
    returns raw API JSON, and one side-effecting tool that must never run in Phase 2."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("lookup_prs", "List open pull requests on the user's repository (raw GitHub JSON).", {})
    async def lookup_prs(args: dict[str, Any]) -> dict[str, Any]:
        calls.append("lookup_prs")
        return {"content": [{"type": "text", "text": json.dumps(RAW_PRS)}]}

    @tool("create_pr", "Open a new pull request.", {"title": str})
    async def create_pr(args: dict[str, Any]) -> dict[str, Any]:
        calls.append("create_pr")
        return {"content": [{"type": "text", "text": "created PR #999"}]}

    setup = McpSetup()
    server = create_sdk_mcp_server(name="gh", version="1.0.0", tools=[lookup_prs, create_pr])
    setup.add_sdk_server("gh", server, read_only=["lookup_prs"], side_effecting=["create_pr"])
    return setup


async def _spoken_reply(brain: ClaudeAgentBrain, question: str) -> tuple[str, list[str]]:
    """Run one turn through the app's pipeline pieces: sentence splitter, then SpeechFilter."""
    from spark.brain.prompts import SpeechFilter
    from spark.util.sentences import split_sentences

    raw: list[str] = []

    async def deltas():
        async for d in brain.respond(question):
            raw.append(d)
            yield d

    speech = SpeechFilter()
    spoken = [out async for s in split_sentences(deltas()) if (out := speech.feed(s))]
    return "".join(raw), spoken


@pytest.mark.live
async def test_live_tool_results_spoken_without_ids_urls(caplog):
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    caplog.set_level(logging.INFO, logger="spark.brain")
    calls: list[str] = []
    brain = ClaudeAgentBrain(_live_settings(), mcp=_fake_pr_server(calls))
    try:
        await brain.warmup()
        raw, spoken = await _spoken_reply(
            brain, "How many open PRs are there, and what's the newest one?"
        )
        text = " ".join(spoken)
        print(f"\n[live] tool calls={calls} raw={raw!r}\n[live] spoken={text!r}")
        assert "lookup_prs" in calls, "the model did not call the tool"
        assert "tool call: mcp__gh__lookup_prs" in caplog.text
        assert not SPOKEN_LEAKS.search(text), f"raw data spoken: {text!r}"
        _assert_no_markdown(text)
        assert COUNT_RE.search(text), f"no count in {text!r}"
        assert "dark mode" in text.lower(), f"newest PR not named in {text!r}"
    finally:
        await brain.close()


@pytest.mark.live
async def test_live_side_effecting_tool_is_denied_in_phase_2(caplog):
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    caplog.set_level(logging.INFO, logger="spark.brain")
    calls: list[str] = []
    brain = ClaudeAgentBrain(_live_settings(), mcp=_fake_pr_server(calls))
    try:
        await brain.warmup()
        _raw, spoken = await _spoken_reply(
            brain, "Open a pull request titled Update the readme. Yes, go ahead, I confirm."
        )
        print(f"\n[live] tool calls={calls} spoken={' '.join(spoken)!r}")
        assert "create_pr" not in calls, "a side-effecting tool ran without confirmation"
        if "tool call: mcp__gh__create_pr" in caplog.text:
            assert "tool denied: mcp__gh__create_pr (side_effect)" in caplog.text
        assert spoken, "Spark said nothing"
    finally:
        await brain.close()


# --- routing (P3.2): model switching, acknowledgement, context --------------------------------


class ManualClock:
    """Virtual time that only moves when the test calls :meth:`advance`. Unlike ``FakeClock``
    (whose ``sleep`` jumps ahead at once), a sleeper here waits for the test, so "no text for
    5 s" is exactly what the test says happened."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        target = self.now + max(0.0, seconds)
        if target <= self.now:
            await asyncio.sleep(0)
            return
        entry = (target, asyncio.get_running_loop().create_future())
        self._sleepers.append(entry)
        try:
            await entry[1]
        finally:
            self._sleepers.remove(entry)

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        for target, fut in list(self._sleepers):
            if target <= self.now and not fut.done():
                fut.set_result(None)
        await settle()


async def settle(rounds: int = 20) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


ACK = "On it, give me a moment. "
DEEP_ASK = "Draft a plan for the product launch"


def held_reply(hold: asyncio.Event, *chunks: str) -> Script:
    """A turn that streams nothing until ``hold`` is set, then ``chunks``."""

    def script(text: str, client: FakeClient) -> list[Any]:
        msgs = streamed(*chunks)
        return [msgs[0], hold, *msgs[1:]]

    return script


async def test_set_model_only_when_the_route_changes_and_context_is_kept(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    clock = ManualClock()
    b = ClaudeAgentBrain(settings(), clock=clock)
    cfg = b.config
    try:
        assert await collect(b, "what time is it") == ["You said ", "what time is it", "."]
        client = FakeClient.instances[0]
        assert client.options.model == cfg.fast_model  # connected on the fast model
        assert client.models == []
        assert b.last_turn.route == "fast" and not b.last_turn.model_switched

        await collect(b, DEEP_ASK)
        assert client.models == [cfg.deep_model]
        assert b.model == cfg.deep_model
        assert b.last_turn.model == cfg.deep_model and b.last_turn.model_switched
        assert b.last_turn.route == "deep"

        await collect(b, "Draft the first milestone too")  # deep again: no second switch
        await collect(b, "thanks")  # a follow-up of a deep conversation stays deep (P5.7d)
        assert client.models == [cfg.deep_model]
        assert b.last_turn.route_reason.startswith("the conversation already went deep")
        # One client, one conversation: every turn went to the same session.
        assert FakeClient.instances == [client]
        assert client.queries == [
            "what time is it",
            DEEP_ASK,
            "Draft the first milestone too",
            "thanks",
        ]
        await b.reset()  # a new conversation starts fast again
        await collect(b, "what's on my calendar tomorrow")
        meta = b.last_turn.as_meta()
        assert meta["brain_route"] == "fast"
        assert meta["brain_route_reason"] == "default"
        assert meta["brain_model"] == cfg.fast_model
        assert meta["brain_served_model"] == "claude-haiku"  # AssistantMessage.model
        assert "route: deep (phrase 'draft a plan')" in caplog.text
        assert "route: fast (keyword 'what time')" in caplog.text
        assert f"model {cfg.fast_model} -> {cfg.deep_model}" in caplog.text
    finally:
        await b.close()


async def test_route_deep_example_new_project(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    b = ClaudeAgentBrain(settings(), clock=ManualClock())
    try:
        await collect(b, "Hey Spark, let's start a new project called weather-bot")
        assert FakeClient.instances[0].models == [b.config.deep_model]
        assert b.last_turn.as_meta()["brain_route"] == "deep"
        assert "route: deep (phrase 'new project')" in caplog.text
    finally:
        await b.close()


async def test_failed_set_model_answers_on_the_current_model(fake_sdk, caplog):
    FakeClient.set_model_error = RuntimeError("model not available")
    b = ClaudeAgentBrain(settings(), clock=ManualClock())
    try:
        assert await collect(b, DEEP_ASK) == ["You said ", DEEP_ASK, "."]
        assert b.model == b.config.fast_model
        assert b.last_turn.model == b.config.fast_model
        assert not b.last_turn.model_switched
        assert f"switching to {b.config.deep_model} failed" in caplog.text
    finally:
        await b.close()


async def test_routing_disabled_never_switches(fake_sdk):
    s = settings()
    s.routing.enabled = False
    b = ClaudeAgentBrain(s, clock=ManualClock())
    try:
        await collect(b, DEEP_ASK)
        assert FakeClient.instances[0].models == []
        assert b.last_turn.route_reason == "routing disabled"
    finally:
        await b.close()


async def _consume(brain: ClaudeAgentBrain, text: str, out: list[str]) -> None:
    async for delta in brain.respond(text):
        out.append(delta)


async def test_ack_after_5s_of_virtual_time_without_text(fake_sdk):
    hold = asyncio.Event()
    FakeClient.script = staticmethod(held_reply(hold, "Here's the plan. ", "Done, in short."))
    clock = ManualClock()
    b = ClaudeAgentBrain(settings(), clock=clock)
    out: list[str] = []
    try:
        task = asyncio.create_task(_consume(b, DEEP_ASK, out))
        await settle()
        await clock.advance(4.9)
        assert out == []
        await clock.advance(0.2)
        assert out == [ACK]  # its own sentence, so the app speaks it right away
        await clock.advance(30)
        assert out == [ACK]  # once only
        hold.set()
        await asyncio.wait_for(task, 2)
        assert out == [ACK, "Here's the plan. ", "Done, in short."]
        assert b.last_turn.acked and b.last_turn.as_meta()["brain_acked"] is True
    finally:
        await b.close()


async def test_no_ack_when_the_first_text_arrives_in_time(fake_sdk):
    hold = asyncio.Event()
    FakeClient.script = staticmethod(held_reply(hold, "Sure. ", "Plan ready."))
    clock = ManualClock()
    b = ClaudeAgentBrain(settings(), clock=clock)
    out: list[str] = []
    try:
        task = asyncio.create_task(_consume(b, DEEP_ASK, out))
        await settle()
        await clock.advance(4.0)
        hold.set()
        await settle()
        await clock.advance(10)
        await asyncio.wait_for(task, 2)
        assert out == ["Sure. ", "Plan ready."]
        assert not b.last_turn.acked
    finally:
        await b.close()


async def test_no_ack_on_the_fast_route(fake_sdk):
    hold = asyncio.Event()
    FakeClient.script = staticmethod(held_reply(hold, "It's noon."))
    clock = ManualClock()
    b = ClaudeAgentBrain(settings(), clock=clock)
    out: list[str] = []
    try:
        task = asyncio.create_task(_consume(b, "what time is it", out))
        await settle()
        await clock.advance(30)
        assert out == []
        hold.set()
        await asyncio.wait_for(task, 2)
        assert out == ["It's noon."]
    finally:
        await b.close()


async def test_no_ack_for_the_new_topic_fast_ack(fake_sdk):
    b = ClaudeAgentBrain(settings(), clock=ManualClock())
    try:
        assert await collect(b, "Spark, new topic") == [FRESH_START_ACK]
        assert FakeClient.instances == []  # nothing was sent to Claude
    finally:
        await b.close()


class HeldConfirmer:
    """Answers yes once the test releases it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.asked: list[str] = []

    async def ask(self, summary: str) -> bool | None:
        self.asked.append(summary)
        await self.release.wait()
        return True


async def test_ack_timer_pauses_while_a_confirmation_is_asking(fake_sdk):
    hold = asyncio.Event()
    FakeClient.script = staticmethod(held_reply(hold, "Created it."))
    clock = ManualClock()
    confirmer = HeldConfirmer()
    b = ClaudeAgentBrain(settings(), clock=clock, mcp=github_setup(), confirmer=confirmer)
    b.tool_announce_wait_s = 0.01
    out: list[str] = []
    try:
        task = asyncio.create_task(_consume(b, "Create a repo called wb", out))
        await settle()
        await clock.advance(3.0)
        # The CLI calls the PreToolUse hook: the gate asks the user and waits.
        gate = asyncio.create_task(
            b.tool_gate(
                {**_hook_input("mcp__github__create_repository"), "tool_input": {"name": "wb"}},
                "toolu_1",
                {"signal": None},
            )
        )
        await settle()
        await clock.advance(20.0)  # the user takes their time: no ack over the question
        assert out == []
        confirmer.release.set()
        decision = await asyncio.wait_for(gate, 2)
        assert decision["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert confirmer.asked == ["I'll create a GitHub repo called wb. Should I go ahead?"]
        await settle()
        await clock.advance(1.9)  # 3.0 + 1.9 < 5 s of unpaused time
        assert out == []
        await clock.advance(0.2)
        assert out == [ACK]
        hold.set()
        await asyncio.wait_for(task, 2)
        assert out == [ACK, "Created it."]
    finally:
        await b.close()


def test_voice_prompt_asks_multi_step_answers_for_a_closing_summary():
    p = VOICE_SYSTEM_PROMPT.lower()
    assert "sums up what you did or found" in p
    assert "on it, give me a moment" in p


async def test_warmup_visits_the_deep_model_once_and_returns_to_fast(fake_sdk):
    b = ClaudeAgentBrain(settings(prime_on_connect=True), clock=ManualClock())
    try:
        await b.warmup()
        client = FakeClient.instances[0]
        assert client.models == [b.config.deep_model, b.config.fast_model]
        assert b.model == b.config.fast_model
        client.models.clear()
        await collect(b, DEEP_ASK)  # the real switch still happens per turn
        assert client.models == [b.config.deep_model]
    finally:
        await b.close()


async def test_warmup_reports_a_rejected_deep_model_and_keeps_going(fake_sdk, caplog):
    FakeClient.set_model_error = Exception("Model 'claude-sonnet-5' not found")
    b = ClaudeAgentBrain(settings(prime_on_connect=True), clock=ManualClock())
    try:
        await b.warmup()
        assert f"model {b.config.deep_model} was not accepted at warm-up" in caplog.text
        assert await collect(b, "what time is it") == ["You said ", "what time is it", "."]
    finally:
        await b.close()


async def test_warmup_without_routing_makes_no_model_calls(fake_sdk):
    s = settings(prime_on_connect=True)
    s.routing.enabled = False
    b = ClaudeAgentBrain(s, clock=ManualClock())
    try:
        await b.warmup()
        assert FakeClient.instances[0].models == []
    finally:
        await b.close()


@pytest.mark.live
async def test_live_route_deep_switches_model_and_answers(caplog):
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    caplog.set_level(logging.INFO, logger="spark.brain")
    s = _live_settings()
    brain = ClaudeAgentBrain(s, mcp=McpSetup())
    try:
        await brain.warmup()
        fast = "".join([d async for d in brain.respond("My favourite colour is teal. Say ok.")])
        fast_stats = brain.last_turn
        text = "".join(
            [
                d
                async for d in brain.respond(
                    "Draft a two sentence plan to paint my room in the colour I told you a "
                    "moment ago."
                )
            ]
        )
        stats = brain.last_turn
        print(
            f"\n[live] fast: route={fast_stats.route} served={fast_stats.served_model!r} "
            f"reply={fast!r}"
            f"\n[live] deep: configured={s.brain.deep_model!r} served={stats.served_model!r} "
            f"switched={stats.model_switched} route={stats.route} ({stats.route_reason}) "
            f"ttft_s={stats.ttft_s} acked={stats.acked} reply={text!r}"
        )
        assert stats.route == "deep"
        assert "route: deep" in caplog.text
        assert stats.model_switched, "set_model was not accepted (see the log warning)"
        assert brain.model == s.brain.deep_model
        assert text.strip(), "no answer on the deep model"
        # One conversation: the CLI session is the same before and after set_model.
        assert stats.session_id and stats.session_id == fast_stats.session_id
        assert "teal" in text.lower(), "context was lost across the model switch"
        _assert_no_markdown(text)
    finally:
        await brain.close()


# --- fast route: limited tool set, enforced per call in the gate (spec §8.4) ------------------

START_PROJECT = "mcp__spark__start_project"
CURRENT_TIME = "mcp__spark__current_time"
LIST_PRS = "mcp__github__list_pull_requests"


def hook_turns(*tools: str) -> tuple[Script, list[list[HookCall]]]:
    """A script whose every turn calls the hook once per tool (in order), then answers.
    Returns the script and, per turn, the hook calls with their outputs."""
    turns: list[list[HookCall]] = []

    def script(text: str, client: FakeClient) -> list[Any]:
        calls = [HookCall(t, {"name": "weather-bot"}, f"toolu_{i}") for i, t in enumerate(tools)]
        turns.append(calls)
        return [
            SystemMessage("init"),
            text_start(),
            text_delta("Okay."),
            *[item for i, c in enumerate(calls) for item in (_tool_start(f"toolu_{i}"), c)],
            text_start(2),
            text_delta("Done."),
            result(num_turns=2),
        ]

    return script, turns


def _decision(out: dict[str, Any] | None) -> str | None:
    if out == {}:
        return "pass"
    return (out or {}).get("hookSpecificOutput", {}).get("permissionDecision")


def _reason(out: dict[str, Any] | None) -> str | None:
    return (out or {}).get("hookSpecificOutput", {}).get("permissionDecisionReason")


def _routed_brain(s: Settings, confirmer: Any) -> tuple[ClaudeAgentBrain, TimerService]:
    timers = TimerService(clock=FakeClock())
    b = ClaudeAgentBrain(
        s, clock=ManualClock(), mcp=github_setup(), timers=timers, confirmer=confirmer
    )
    b.tool_announce_wait_s = 0.5
    return b, timers


WEATHER = "how's the weather looking"  # fast (default rule)


async def test_fast_turn_denies_start_project_without_asking_then_retry_goes_deep(fake_sdk, caplog):
    from spark.brain.confirm import REASON_DECLINED, REASON_FAST_ROUTE
    from tests.fakes import FakeConfirmer

    caplog.set_level(logging.INFO, logger="spark")
    script, turns = hook_turns(START_PROJECT)
    FakeClient.script = staticmethod(script)
    confirmer = FakeConfirmer([False])
    b, timers = _routed_brain(settings(), confirmer)
    try:
        assert b.current_route is None  # no turn yet
        await collect(b, WEATHER)
        assert b.last_turn.route == "fast"
        (call,) = turns[0]
        assert call.output == {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": REASON_FAST_ROUTE,
            }
        }
        assert confirmer.asked == []  # denied WITHOUT asking
        assert f"tool denied: {START_PROJECT} (fast route)" in caplog.text
        assert b.current_route is None  # cleared when the turn finished

        # "Ask again": the next turn is routed deep, where the tool is confirmed as usual.
        await collect(b, "okay do it")
        assert b.last_turn.route == "deep"
        assert b.last_turn.route_reason.startswith("retry after a fast-route tool refusal")
        assert "route: deep (retry after a fast-route tool refusal" in caplog.text
        assert FakeClient.instances[0].models == [b.config.deep_model]
        (call,) = turns[1]
        assert _reason(call.output) == REASON_DECLINED
        assert len(confirmer.asked) == 1

        await collect(b, "thanks")  # the conversation went deep, so it stays deep (P5.7d)
        assert b.last_turn.route == "deep"
        assert b.last_turn.route_reason.startswith("the conversation already went deep")
        assert len(FakeClient.instances) == 1  # one session throughout
    finally:
        await b.close()
        await timers.close()


async def test_deep_turn_asks_before_start_project(fake_sdk, caplog):
    from tests.fakes import FakeConfirmer

    caplog.set_level(logging.INFO, logger="spark")
    script, turns = hook_turns(START_PROJECT)
    FakeClient.script = staticmethod(script)
    confirmer = FakeConfirmer([True])
    b, timers = _routed_brain(settings(), confirmer)
    try:
        await collect(b, "Hey Spark, let's start a new project called weather-bot")
        assert b.last_turn.route == "deep"
        (call,) = turns[0]
        assert _decision(call.output) == "allow"
        assert len(confirmer.asked) == 1 and "weather-bot" in confirmer.asked[0]
        assert "(fast route)" not in caplog.text
    finally:
        await b.close()
        await timers.close()


@pytest.mark.parametrize("text", ["what time is it", DEEP_ASK])
async def test_read_tools_pass_on_both_routes(fake_sdk, text):
    from tests.fakes import FakeConfirmer

    script, turns = hook_turns(CURRENT_TIME, LIST_PRS)
    FakeClient.script = staticmethod(script)
    confirmer = FakeConfirmer([True])
    b, timers = _routed_brain(settings(), confirmer)
    try:
        await collect(b, text)
        assert [_decision(c.output) for c in turns[0]] == ["pass", "pass"]
        assert confirmer.asked == []
    finally:
        await b.close()
        await timers.close()


async def test_explicit_fast_tools_list_is_honoured(fake_sdk, caplog):
    from spark.brain.confirm import REASON_FAST_ROUTE
    from tests.fakes import FakeConfirmer

    caplog.set_level(logging.INFO, logger="spark")
    script, turns = hook_turns(CURRENT_TIME, LIST_PRS)
    FakeClient.script = staticmethod(script)
    s = settings()
    s.routing.fast_tools = [CURRENT_TIME, "mcp__nope__thing", "mcp__spark__*"]
    b, timers = _routed_brain(s, FakeConfirmer([True]))
    try:
        # entries that can never match are reported at startup
        assert "routing.fast_tools: mcp__nope__thing is not a known tool" in caplog.text
        assert "'mcp__spark__*' is a pattern" in caplog.text

        await collect(b, "what time is it")  # fast: only the listed tool
        assert b.last_turn.route == "fast"
        assert _decision(turns[0][0].output) == "pass"
        assert _reason(turns[0][1].output) == REASON_FAST_ROUTE
        assert f"tool denied: {LIST_PRS} (fast route)" in caplog.text

        await collect(b, DEEP_ASK)  # deep: the full set
        assert [_decision(c.output) for c in turns[1]] == ["pass", "pass"]
    finally:
        await b.close()
        await timers.close()


async def test_routing_disabled_means_no_fast_tool_limit(fake_sdk, caplog):
    from spark.brain.confirm import REASON_DECLINED
    from tests.fakes import FakeConfirmer

    caplog.set_level(logging.INFO, logger="spark")
    script, turns = hook_turns(START_PROJECT)
    FakeClient.script = staticmethod(script)
    s = settings()
    s.routing.enabled = False
    confirmer = FakeConfirmer([False])
    b, timers = _routed_brain(s, confirmer)
    try:
        await collect(b, "what time is it")
        assert b.last_turn.route_reason == "routing disabled"
        (call,) = turns[0]
        assert _reason(call.output) == REASON_DECLINED  # asked, as before routing existed
        assert len(confirmer.asked) == 1
        assert "(fast route)" not in caplog.text
    finally:
        await b.close()
        await timers.close()


async def test_injected_confirmation_gate_gets_the_route(fake_sdk):
    from spark.brain.confirm import REASON_FAST_ROUTE, ConfirmationGate
    from tests.fakes import FakeConfirmer

    script, turns = hook_turns(START_PROJECT)
    FakeClient.script = staticmethod(script)
    timers = TimerService(clock=FakeClock())
    setup = github_setup()
    gate = ConfirmationGate(setup.policy, FakeConfirmer([True]))
    b = ClaudeAgentBrain(settings(), clock=ManualClock(), mcp=setup, timers=timers, tool_gate=gate)
    b.tool_announce_wait_s = 0.5
    try:
        assert gate.route_provider is not None
        await collect(b, WEATHER)
        assert _reason(turns[0][0].output) == REASON_FAST_ROUTE
    finally:
        await b.close()
        await timers.close()


async def test_new_topic_clears_a_pending_escalation(fake_sdk):
    from tests.fakes import FakeConfirmer

    script, _turns = hook_turns(START_PROJECT)
    FakeClient.script = staticmethod(script)
    b, timers = _routed_brain(settings(), FakeConfirmer([False]))
    try:
        await collect(b, WEATHER)  # fast refusal: the next turn would go deep...
        await collect(b, f"Spark, new topic. {WEATHER}")  # ...but a fresh conversation doesn't
        assert b.last_turn.route == "fast"
    finally:
        await b.close()
        await timers.close()


# --- P5.1: step budget, deep connectors, ack once a fast turn calls a tool ---------------------


def _budget_brain(s: Settings, setup: McpSetup | None = None) -> ClaudeAgentBrain:
    timers = TimerService(clock=FakeClock())
    b = ClaudeAgentBrain(s, clock=ManualClock(), mcp=setup or github_setup(), timers=timers)
    b.tool_announce_wait_s = 0.5
    return b


async def test_step_budget_refuses_tools_past_the_fast_budget(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    script, turns = hook_turns(CURRENT_TIME, CURRENT_TIME, CURRENT_TIME, LIST_PRS)
    FakeClient.script = staticmethod(script)
    b = _budget_brain(settings(max_turns=3, deep_max_turns=10))
    try:
        await collect(b, WEATHER)
        assert b.last_turn.route == "fast"
        decisions = [_decision(c.output) for c in turns[0]]
        assert decisions == ["pass", "pass", "deny", "deny"]  # 3 steps: 2 tools + the answer
        from spark.brain.confirm import REASON_STEP_BUDGET

        assert _reason(turns[0][2].output) == REASON_STEP_BUDGET
        assert "tool denied: mcp__spark__current_time (step budget)" in caplog.text
    finally:
        await b.close()


async def test_deep_turns_get_the_larger_step_budget(fake_sdk):
    script, turns = hook_turns(CURRENT_TIME, CURRENT_TIME, CURRENT_TIME, LIST_PRS)
    FakeClient.script = staticmethod(script)
    b = _budget_brain(settings(max_turns=3, deep_max_turns=10))
    try:
        await collect(b, DEEP_ASK)
        assert b.last_turn.route == "deep"
        assert [_decision(c.output) for c in turns[0]] == ["pass"] * 4
        assert b.step_budget() == 3  # between turns: the fast budget
    finally:
        await b.close()


async def test_step_budget_counts_per_request(fake_sdk):
    script, turns = hook_turns(CURRENT_TIME, CURRENT_TIME)
    FakeClient.script = staticmethod(script)
    b = _budget_brain(settings(max_turns=3))
    try:
        await collect(b, WEATHER)
        await collect(b, WEATHER)
        assert [[_decision(c.output) for c in t] for t in turns] == [["pass", "pass"]] * 2
    finally:
        await b.close()


def _deep_github() -> McpSetup:
    setup = github_setup()
    setup.deep_route["github"] = ("github", "pull request")
    return setup


async def test_a_request_naming_a_deep_connector_routes_deep(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    b = _budget_brain(settings(), _deep_github())
    try:
        await collect(b, "so can you connect to GitHub?")
        assert b.last_turn.route == "deep"
        assert "names deep connector github" in caplog.text
        await collect(b, "how's the weather looking")
        assert b.last_turn.route == "fast"  # named only; none of its tools ran
    finally:
        await b.close()


async def test_using_a_deep_connector_makes_the_conversation_deep_until_reset(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    script, _ = hook_turns(LIST_PRS)
    FakeClient.script = staticmethod(script)
    b = _budget_brain(settings(), _deep_github())
    try:
        await collect(b, "anything new for me?")  # fast, reaches a github tool: re-run deep
        assert b.last_turn.route == "deep"
        await collect(b, "how many came in this month")
        assert b.last_turn.route == "deep"
        assert "conversation is using deep connector github" in caplog.text
        await b.reset()
        FakeClient.script = staticmethod(lambda text, client: streamed("Fine."))
        await collect(b, "how many came in this month")
        assert b.last_turn.route == "fast"
    finally:
        await b.close()


async def test_a_fast_turn_that_reaches_a_deep_connector_is_rerun_deep(fake_sdk, caplog):
    """The 2026-10-02 16:13 case: 'how many orders are shipping today?' names no server, routes
    fast, and the fast model starts exploring WAREHOUSE. Hand it to the deep model at once."""
    from spark.brain.claude_agent import REASON_ESCALATING

    caplog.set_level(logging.INFO, logger="spark.brain")
    turns: list[list[HookCall]] = []

    def script(text: str, client: FakeClient) -> list[Any]:
        call = HookCall(LIST_PRS, {}, f"toolu_{len(turns)}")
        turns.append([call])
        return [
            SystemMessage("init"),
            _tool_start(call.tool_use_id),
            call,
            text_start(2),
            text_delta("Three came in." if len(turns) == 2 else "Let me try another way."),
            result(num_turns=2),
        ]

    FakeClient.script = staticmethod(script)
    b = _budget_brain(settings(), _deep_github())
    try:
        out = await collect(b, "how many orders are shipping today?")
        assert "".join(out) == "Three came in."  # only the deep turn is heard
        client = FakeClient.instances[0]
        assert client.queries == ["how many orders are shipping today?"] * 2
        assert client.models == [b.config.deep_model]
        assert _decision(turns[0][0].output) == "deny"
        assert _reason(turns[0][0].output) == REASON_ESCALATING
        assert _decision(turns[1][0].output) == "pass"  # the deep turn runs the tool
        assert b.last_turn.route == "deep"
        assert "fast turn reached deep connector github; re-running it deep" in caplog.text
        assert "route: deep (fast turn reached deep connector github" in caplog.text
    finally:
        await b.close()


def test_deep_route_keywords():
    from spark.brain.mcp_config import deep_route_keywords

    assert deep_route_keywords("CATALOG_API_DOCS", ["sales report"]) == (
        "catalog_api_docs",
        "catalog api docs",
        "sales report",
    )


async def test_fast_turn_acks_once_a_tool_is_running(fake_sdk):
    hold = asyncio.Event()
    call = HookCall(CURRENT_TIME, {}, "toolu_0")

    def script(text: str, client: FakeClient) -> list[Any]:
        return [SystemMessage("init"), _tool_start("toolu_0"), call, hold, *streamed("Done.")[1:]]

    FakeClient.script = staticmethod(script)
    clock = ManualClock()
    b = ClaudeAgentBrain(
        settings(), clock=clock, mcp=github_setup(), timers=TimerService(clock=FakeClock())
    )
    b.tool_announce_wait_s = 0.5
    out: list[str] = []
    try:
        task = asyncio.create_task(_consume(b, WEATHER, out))
        await settle()
        assert b.last_turn is None or b.last_turn.route != "deep"
        await clock.advance(4.9)
        assert out == []
        await clock.advance(0.2)
        assert out == [ACK]
        hold.set()
        await asyncio.wait_for(task, 2)
        assert out == [ACK, "Done."]
    finally:
        await b.close()


# --- P5.6: each session starts with a fresh briefing --------------------------------------------


async def test_briefing_is_appended_at_every_connect(fake_sdk):
    briefings = iter(["Earlier: forty two open orders.", ""])
    b = ClaudeAgentBrain(settings(), context_provider=lambda: next(briefings))
    try:
        await b.warmup()
        first = FakeClient.instances[0].options.system_prompt
        assert first == b.system_prompt + "\n\nEarlier: forty two open orders."
        await b.reset()
        assert FakeClient.instances[1].options.system_prompt == b.system_prompt
        assert b.last_context == ""
    finally:
        await b.close()


async def test_a_failing_briefing_never_blocks_the_session(fake_sdk, caplog):
    def broken() -> str:
        raise OSError("history unreadable")

    b = ClaudeAgentBrain(settings(), context_provider=broken)
    try:
        assert await collect(b, "hi") == ["You said ", "hi", "."]
        assert FakeClient.instances[0].options.system_prompt == b.system_prompt
        assert "briefing failed" in caplog.text
    finally:
        await b.close()


async def test_next_route_forces_one_turn(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    b = ClaudeAgentBrain(settings(), clock=ManualClock())
    try:
        b.next_route, b.next_route_reason = "deep", "shortcut 'Morning briefing'"
        await collect(b, "good morning")
        assert b.last_turn.route == "deep"
        assert "route: deep (shortcut 'Morning briefing')" in caplog.text
        await collect(b, "good morning")
        assert b.last_turn.route == "deep"  # the conversation went deep and stays deep
        assert b.last_turn.route_reason.startswith("the conversation already went deep")
        await b.reset()
        await collect(b, "good morning")
        assert b.last_turn.route == "fast"  # the forced route itself was one turn only
    finally:
        await b.close()


# --- P5.7d: the researcher worker -----------------------------------------------------------------


def _worker_brain(tmp_path=None, **brain: Any) -> ClaudeAgentBrain:
    from spark.memory import MemoryStore

    memory = MemoryStore(tmp_path / "memory") if tmp_path is not None else None
    b = ClaudeAgentBrain(
        settings(mode="deep", workers=True, **brain),
        clock=ManualClock(),
        mcp=github_setup(),
        timers=TimerService(clock=FakeClock()),
        memory=memory,
    )
    b.tool_announce_wait_s = 0.1
    return b


def test_worker_options(fake_sdk, tmp_path):
    from spark.brain.prompts import WORKER_RULES

    b = _worker_brain(tmp_path, worker_max_turns=12)
    opts = b._build_options()
    assert opts.tools == ["Agent"]
    assert "Agent" in opts.allowed_tools
    (name, agent), *_ = opts.agents.items()
    assert name == "researcher" and type(agent).__name__ == "AgentDefinition"
    assert agent.model == "claude-haiku-5-5" and agent.maxTurns == 12
    assert sorted(agent.tools) == [
        "mcp__github__issue_read",
        "mcp__github__list_pull_requests",
        "mcp__spark__rewrite_connector_notes",
        "mcp__spark__save_connector_note",
    ]
    assert not any("create_repository" in t or "start_project" in t for t in agent.tools)
    assert WORKER_RULES in opts.system_prompt
    assert opts.model == "claude-opus-5-5"  # deep mode connects on the voice model


def test_no_worker_without_connectors_or_when_off(fake_sdk):
    from spark.brain.prompts import WORKER_RULES

    alone = ClaudeAgentBrain(
        settings(mode="deep", workers=True), timers=TimerService(clock=FakeClock())
    )
    assert not alone.workers_active and alone._build_options().tools == []
    off = ClaudeAgentBrain(settings(mode="deep", workers=False), mcp=github_setup())
    opts = off._build_options()
    assert opts.tools == [] and "agents" not in opts.kwargs
    assert WORKER_RULES not in opts.system_prompt


def test_worker_model_can_be_set(fake_sdk):
    b = _worker_brain(worker_model="claude-sonnet-5")
    assert b._build_options().agents["researcher"].model == "claude-sonnet-5"


async def test_worker_calls_pass_the_gate_without_using_the_voice_budget(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    worker = {"agent_id": "a1", "agent_type": "researcher"}
    calls = [
        HookCall("Agent", {"subagent_type": "researcher"}, "toolu_0"),
        *[HookCall(LIST_PRS, {}, f"toolu_w{i}", extra=worker) for i in range(5)],
        HookCall(LIST_PRS, {}, "toolu_1"),
        HookCall(LIST_PRS, {}, "toolu_2"),
    ]

    def script(text: str, client: FakeClient) -> list[Any]:
        return [SystemMessage("init"), *calls, text_start(), text_delta("Done."), result()]

    FakeClient.script = staticmethod(script)
    b = _worker_brain(max_turns=3, deep_max_turns=3)
    try:
        await collect(b, "how many pull requests are open")
        decisions = [_decision(c.output) for c in calls]
        # The voice model's own calls count (Agent is its 1st step, the next query its 2nd); a
        # budget of 3 leaves room for 2 tools plus the answer. The worker's 5 calls don't count.
        assert decisions == ["pass", *["pass"] * 5, "pass", "deny"]
        assert "model claude-haiku" not in caplog.text  # deep mode: no switch, connected deep
    finally:
        await b.close()


async def test_the_researchers_messages_are_logged_but_never_spoken(fake_sdk, tmp_path):
    from spark.activity import ActivityLog

    def script(text: str, client: FakeClient) -> list[Any]:
        worker_msg = AssistantMessage(
            [TextBlock("Working on it..."), ToolUseBlock("toolu_w", LIST_PRS, {"state": "open"})],
            "claude-haiku-4-5-20251001",
            parent_tool_use_id="toolu_agent",
        )
        return [
            SystemMessage("init"),
            StreamEvent("u", "s", text_delta("worker text").event, parent_tool_use_id="toolu_a"),
            worker_msg,
            text_start(),
            text_delta("Three are open."),
            result(),
        ]

    FakeClient.script = staticmethod(script)
    activity = ActivityLog(tmp_path / "history")
    b = ClaudeAgentBrain(
        settings(mode="deep", workers=True),
        clock=ManualClock(),
        mcp=github_setup(),
        timers=TimerService(clock=FakeClock()),
        activity=activity,
    )
    try:
        assert "".join(await collect(b, "how many PRs are open")) == "Three are open."
    finally:
        await b.close()
    (call,) = [e for e in activity.iter_events() if e["kind"] == "tool_call"]
    assert call["agent"] == "researcher" and call["tool"] == LIST_PRS
    assert call["model"] == "claude-haiku-4-5-20251001"


# --- P5.7i: connector health --------------------------------------------------------------------


def _status(**servers: str) -> dict[str, Any]:
    entries = []
    for name, status in servers.items():
        entry: dict[str, Any] = {
            "name": name,
            "status": status,
            "config": {"type": "http", "url": "http://10.1.2.3/secret-path/mcp"},
        }
        if status == "failed":
            entry["error"] = "Connection timed out after 30000ms"
        entries.append(entry)
    entries.append({"name": "spark", "status": "connected"})
    return {"mcpServers": entries}


async def test_connector_status_is_read_at_connect_and_reported(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    FakeClient.mcp_status = _status(WAREHOUSE="failed", TICKETS="connected")
    seen: list[dict[str, Any]] = []
    b = ClaudeAgentBrain(settings())
    b.on_mcp_status = seen.append
    try:
        await b.warmup()
        assert b.mcp_status == {
            "TICKETS": {"status": "connected"},
            "WAREHOUSE": {"status": "failed", "error": "Connection timed out after 30000ms"},
        }
        assert b.unreachable_connectors() == ["WAREHOUSE"]
        assert seen == [b.mcp_status]
        assert "MCP WAREHOUSE: failed (Connection timed out after 30000ms)" in caplog.text
        assert "MCP TICKETS: connected" in caplog.text
        assert "secret-path" not in caplog.text  # the server config is never logged
    finally:
        await b.close()


async def test_the_model_hears_about_an_unreachable_connector_once_per_session(fake_sdk):
    FakeClient.mcp_status = _status(WAREHOUSE="failed")
    b = ClaudeAgentBrain(settings())
    try:
        await b.warmup()  # the app always warms up (and resets) before the user speaks
        await collect(b, "how many orders ship today")
        await collect(b, "and tomorrow?")
        first, second = FakeClient.instances[0].queries
        assert first.startswith("how many orders ship today\n\n(Status from Spark")
        assert "WAREHOUSE is unreachable right now" in first
        assert second == "and tomorrow?"
        await b.reset()
        await collect(b, "try again")
        assert "WAREHOUSE is unreachable" in FakeClient.instances[-1].queries[-1]
    finally:
        await b.close()


async def test_status_is_rechecked_after_each_turn(fake_sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    FakeClient.mcp_status = _status(WAREHOUSE="pending")
    seen: list[dict[str, Any]] = []
    b = ClaudeAgentBrain(settings())
    b.on_mcp_status = seen.append
    try:
        await collect(b, "hello")
        FakeClient.mcp_status = _status(WAREHOUSE="connected")
        await collect(b, "hello again")
        assert b.mcp_status == {"WAREHOUSE": {"status": "connected"}}
        assert [s["WAREHOUSE"]["status"] for s in seen] == ["pending", "connected"]
        assert "WAREHOUSE is unreachable" not in " ".join(FakeClient.instances[0].queries)
    finally:
        await b.close()
