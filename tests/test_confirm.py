"""Spoken confirmation of side-effecting tools (P3.1, spec §8.3, Phase 3 acceptance).

The gate is the SDK's ``PreToolUse`` hook. Its outputs are asserted as exact dicts (shape from
``PreToolUseHookSpecificOutput``, claude_agent_sdk/types.py:437). The allow shape is required,
because ``{}`` leaves a non-allowlisted tool to the CLI, which refuses it (docs/sdk_notes.md P3.1).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from spark.app import SparkApp, State
from spark.brain.confirm import (
    DENY_REASONS,
    REASON_ALLOWED,
    REASON_DECLINED,
    REASON_FAST_ROUTE,
    REASON_NO_ANSWER,
    ConfirmationGate,
    LineReader,
    TextConfirmer,
    VoiceConfirmer,
    classify_yes_no,
    generic_summary,
    humanize_tool_name,
    summarize_tool_call,
)
from spark.brain.mcp_config import McpSetup
from spark.brain.prompts import MARKDOWN_CHARS
from spark.config import ConfirmConfig
from tests.fakes import (
    FakeConfirmer,
    FakeMic,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWakeWord,
    ToolCallingBrain,
    concat,
    make_settings,
    silence,
    tone,
)

NOTE = "mcp__notes__create_note"
LIST = "mcp__notes__list_notes"
WIPE = "mcp__notes__delete_all_notes"

ALLOW = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": REASON_ALLOWED,
    }
}


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def notes_setup(*, allow_destructive: bool = False) -> McpSetup:
    setup = McpSetup()
    setup.add_sdk_server(
        "notes",
        object(),
        read_only=["list_notes"],
        side_effecting=["create_note"],
        destructive=["delete_all_notes"],
        allow_destructive=allow_destructive,
    )
    return setup


def hook_input(name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "session_id": "s",
        "transcript_path": "t",
        "cwd": ".",
        "hook_event_name": "PreToolUse",
        "tool_name": name,
        "tool_input": tool_input if tool_input is not None else {"title": "groceries"},
        "tool_use_id": "toolu_1",
    }


async def run_gate(gate: ConfirmationGate, name: str, tool_input=None) -> dict[str, Any]:
    return await gate(hook_input(name, tool_input), "toolu_1", {"signal": None})


# --- classify_yes_no ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "Yes.",
        "yeah",
        "Yep!",
        "sure",
        "Sure, go ahead.",
        "go ahead",
        "do it",
        "please do",
        "okay",
        "Spark, yes please",
        "yes, do it",
        "no problem, go ahead",
        "sure, why not",
        "YES",
        "  yeah  sure ",
    ],
)
def test_classify_yes(text):
    assert classify_yes_no(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "no",
        "No.",
        "nope",
        "nah",
        "don't",
        "Don’t do it",
        "do not do that",
        "stop",
        "cancel",
        "cancel that",
        "wait",
        "not now",
        "no, don't",
        "yes... actually no",
        "yes, wait",
        "sure, but not yet",
        "never mind",
        "hold on",
    ],
)
def test_classify_no(text):
    assert classify_yes_no(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hmm",
        "what was the question",
        "I'm not sure",
        "maybe",
        "I don't know",
        "yes, I guess",
        "I know",  # "know" is not "no"
        "the weather is nice",
    ],
)
def test_classify_unclear(text):
    assert classify_yes_no(text) is None


def test_classify_extra_phrases():
    assert classify_yes_no("make it so") is None
    assert classify_yes_no("make it so", extra_yes=["make it so"]) is True
    assert classify_yes_no("belay that", extra_no=["belay that"]) is False


# --- summaries ---------------------------------------------------------------------------------

_LEAK = re.compile(r"https?://|www\.|@|[0-9a-f]{7,}|ghp_|\b\d{5,}\b", re.I)


def test_builtin_summary_for_github_repo():
    line = summarize_tool_call("mcp__github__create_repository", {"name": "weather-bot"})
    assert line == "I'll create a GitHub repo called weather-bot. Should I go ahead?"


def test_github_issue_write_is_said_in_plain_words():
    # GitHub's server has one issue_write tool; the generic line read out "issue write ... method
    # create". The title is what the user says yes to, so a full sentence of one is kept.
    args = {"method": "create", "owner": "me", "repo": "demo-repo", "body": "Steps: ..."}
    title = "The login button does not work on mobile"
    line = summarize_tool_call("mcp__github__issue_write", {**args, "title": title})
    assert line == (
        "I'll create an issue in demo-repo titled The login button does not work on mobile."
        " Should I go ahead?"
    )
    no_url = summarize_tool_call("mcp__github__issue_write", {**args, "title": "see https://x.y/z"})
    assert no_url == "I'll create an issue in demo-repo. Should I go ahead?"
    update = {"method": "update", "repo": "demo-repo", "issue_number": 4}
    assert summarize_tool_call("mcp__github__issue_write", update).startswith(
        "I'll update issue 4 in demo-repo."
    )
    closing = {**update, "state": "closed"}
    assert summarize_tool_call("mcp__github__issue_write", closing).startswith(
        "I'll close issue 4 in demo-repo."
    )


def test_calendar_summary_speaks_the_time_not_the_iso_string():
    line = summarize_tool_call(
        "mcp__composio__GOOGLECALENDAR_CREATE_EVENT",
        {"summary": "Dentist", "start_datetime": "2026-09-24T15:30:00", "calendar_id": "primary"},
    )
    assert line == "I'll add Dentist to your calendar on Thursday at 3:30 PM. Should I go ahead?"


def test_generic_summary_humanizes_the_name_and_keeps_two_short_args():
    assert humanize_tool_name("mcp__composio__GMAIL_CREATE_EMAIL_DRAFT") == (
        "Gmail create email draft"
    )
    assert humanize_tool_name("mcp__notes__createNote") == "create note"
    line = summarize_tool_call(NOTE, {"title": "groceries", "folder": "home", "tag": "weekly"})
    assert line == "I'll create note called groceries, folder home. Should I go ahead?"
    assert generic_summary("mcp__x__frobnicate", {}) == "I'll use frobnicate."


def test_summary_never_contains_urls_ids_emails_or_secrets():
    args = {
        "url": "https://github.com/acme/weather-bot",
        "repo_url": "www.example.com/x",
        "owner": "acme",
        "issue_id": "I_kwDOAbc123",
        "sha": "9fceb02d0ae598e95dc970b74767f19372d61af8",
        "email": "jane@example.com",
        "to": "jane@example.com",
        "api_key": "sk-live-abc",
        "token": "ghp_abcdefghijklmnop1234567890",
        "body": "see https://evil.example.com/a?b=c for details",
        "ref": "a1b2c3d4e5f6a7b8c9d0",
        "number": 12345,
        "title": "Fix the login bug",
    }
    line = summarize_tool_call("mcp__github__update_pull_request", args)
    assert not _LEAK.search(line), line
    assert "sk-live" not in line and "acme" not in line
    assert "Fix the login bug" in line
    assert not set(line) & MARKDOWN_CHARS
    assert line.endswith("Should I go ahead?")


def test_summary_uses_a_registered_summarizer_and_falls_back_on_errors():
    custom = {NOTE: lambda args: f"I'll save a note about {args['title']}. Should I go ahead?"}
    assert summarize_tool_call(NOTE, {"title": "milk"}, custom) == (
        "I'll save a note about milk. Should I go ahead?"
    )
    bare = {"create_note": lambda args: "I'll jot that down."}
    assert summarize_tool_call(NOTE, {}, bare) == "I'll jot that down. Should I go ahead?"

    def boom(args):
        raise KeyError("title")

    assert summarize_tool_call(NOTE, {"title": "milk"}, {NOTE: boom}) == (
        "I'll create note called milk. Should I go ahead?"
    )


def test_summary_markdown_from_a_summarizer_is_sanitized():
    line = summarize_tool_call(NOTE, {}, {NOTE: lambda a: "I'll **create** it, see https://x.io"})
    assert "*" not in line and "http" not in line


# --- the gate ----------------------------------------------------------------------------------


async def test_gate_allows_on_yes_with_the_explicit_allow_shape(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    confirmer = FakeConfirmer([True])
    gate = ConfirmationGate(notes_setup().policy, confirmer)
    assert await run_gate(gate, NOTE) == ALLOW
    assert confirmer.asked == ["I'll create note called groceries. Should I go ahead?"]
    assert f"tool confirm: {NOTE} -> yes" in caplog.text


async def test_gate_denies_on_no_with_the_relayable_reason(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    gate = ConfirmationGate(notes_setup().policy, FakeConfirmer([False]))
    out = await run_gate(gate, NOTE, {"title": "groceries", "owner_email": "jane@example.com"})
    assert out == _deny("The user said no. Don't do it; acknowledge briefly.")
    assert out == _deny(REASON_DECLINED)
    assert f"tool confirm: {NOTE} -> no" in caplog.text
    assert "jane@example.com" not in caplog.text and "groceries" not in caplog.text
    assert gate.decisions == [(NOTE, "no")]


async def test_gate_denies_on_no_answer():
    gate = ConfirmationGate(notes_setup().policy, FakeConfirmer([None]))
    assert await run_gate(gate, NOTE) == _deny("No confirmation was heard, so I didn't do it.")
    assert REASON_NO_ANSWER == "No confirmation was heard, so I didn't do it."


async def test_gate_times_out_a_slow_confirmer_as_no_answer(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    gate = ConfirmationGate(
        notes_setup().policy, FakeConfirmer([True], delay_s=5.0), ask_timeout_s=0.05
    )
    assert await run_gate(gate, NOTE) == _deny(REASON_NO_ANSWER)
    assert f"tool confirm: {NOTE} -> timeout" in caplog.text


async def test_gate_treats_a_confirmer_error_as_no_answer():
    gate = ConfirmationGate(notes_setup().policy, FakeConfirmer(fail_with=OSError("mic gone")))
    assert await run_gate(gate, NOTE) == _deny(REASON_NO_ANSWER)


async def test_gate_passes_read_only_tools_without_asking():
    confirmer = FakeConfirmer([False])
    gate = ConfirmationGate(notes_setup().policy, confirmer)
    assert await run_gate(gate, LIST) == {}
    assert confirmer.asked == []


@pytest.mark.parametrize(
    ("tool", "kind"),
    [(WIPE, "destructive"), ("mcp__notes__export_all", "unknown"), ("Bash", "unknown")],
)
async def test_gate_denies_destructive_and_unknown_without_asking(tool, kind, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    confirmer = FakeConfirmer([True])
    gate = ConfirmationGate(notes_setup().policy, confirmer)
    assert await run_gate(gate, tool) == _deny(DENY_REASONS[kind])
    assert confirmer.asked == []
    assert f"tool denied: {tool} ({kind})" in caplog.text


async def test_gate_confirms_destructive_tools_when_the_server_allows_them():
    confirmer = FakeConfirmer([True])
    gate = ConfirmationGate(notes_setup(allow_destructive=True).policy, confirmer)
    assert await run_gate(gate, WIPE, {}) == ALLOW
    assert confirmer.asked[0].startswith("This can't be undone. I'll delete all notes.")


async def test_gate_without_a_confirmer_denies_side_effects():
    gate = ConfirmationGate(notes_setup().policy, None)
    assert await run_gate(gate, NOTE) == _deny(DENY_REASONS["side_effect"])


async def test_gate_asks_about_parallel_tool_calls_one_at_a_time():
    active = 0
    peak = 0

    class Slow(FakeConfirmer):
        async def ask(self, summary: str) -> bool | None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return await super().ask(summary)

    gate = ConfirmationGate(notes_setup().policy, Slow([True, False]))
    first, second = await asyncio.gather(
        run_gate(gate, NOTE, {"title": "a"}), run_gate(gate, NOTE, {"title": "b"})
    )
    assert peak == 1
    assert [first, second] == [ALLOW, _deny(REASON_DECLINED)]


async def test_gate_runs_before_ask_with_the_tool_use_id():
    seen: list[str | None] = []

    async def before(tool_use_id):
        seen.append(tool_use_id)

    gate = ConfirmationGate(notes_setup().policy, FakeConfirmer([True]), before_ask=before)
    await run_gate(gate, NOTE)
    assert seen == ["toolu_1"]


async def test_gate_propagates_cancellation_from_the_cli(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    gate = ConfirmationGate(notes_setup().policy, FakeConfirmer([True], delay_s=10))
    task = asyncio.create_task(run_gate(gate, NOTE))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert f"tool confirm: {NOTE} -> cancelled" in caplog.text


# --- fast route: limited tool set (spec §8.4) ---------------------------------------------------


def _routed_gate(route: str | None, confirmer=None, **kw: Any) -> ConfirmationGate:
    return ConfirmationGate(
        notes_setup().policy,
        confirmer if confirmer is not None else FakeConfirmer([True]),
        route_provider=lambda: route,
        **kw,
    )


async def test_fast_route_denies_side_effects_without_asking(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    confirmer = FakeConfirmer([True])
    denied: list[str] = []
    gate = _routed_gate("fast", confirmer, on_fast_denied=denied.append)
    assert gate.current_route == "fast"
    assert await run_gate(gate, NOTE) == _deny(REASON_FAST_ROUTE)
    assert confirmer.asked == []  # never asked
    assert denied == [NOTE]
    assert gate.decisions == [(NOTE, "fast-denied")]
    assert f"tool denied: {NOTE} (fast route)" in caplog.text
    assert "groceries" not in caplog.text  # names only, never arguments
    # the reason is relayable speech: it asks for a retry and names the task as multi-step
    assert "ask again" in REASON_FAST_ROUTE and "multi-step" in REASON_FAST_ROUTE


async def test_deep_route_still_asks():
    confirmer = FakeConfirmer([True])
    gate = _routed_gate("deep", confirmer)
    assert await run_gate(gate, NOTE) == ALLOW
    assert confirmer.asked == ["I'll create note called groceries. Should I go ahead?"]


@pytest.mark.parametrize("route", ["fast", "deep", None])
async def test_read_tools_pass_on_every_route(route):
    confirmer = FakeConfirmer([False])
    gate = _routed_gate(route, confirmer)
    assert await run_gate(gate, LIST) == {}
    assert confirmer.asked == []


@pytest.mark.parametrize("route", ["fast", "deep"])
async def test_blocked_tools_keep_their_own_reason_on_every_route(route):
    # "ask again" would be wrong: these are refused on the deep route too
    gate = _routed_gate(route)
    assert await run_gate(gate, WIPE) == _deny(DENY_REASONS["destructive"])
    assert await run_gate(gate, "Bash") == _deny(DENY_REASONS["unknown"])


async def test_no_route_means_no_fast_limit():
    # No route_provider (Phase 2 gate), or one that says None (no routed turn / routing off).
    for gate in (ConfirmationGate(notes_setup().policy, FakeConfirmer([True])), _routed_gate(None)):
        assert gate.current_route is None
        assert await run_gate(gate, NOTE) == ALLOW


async def test_explicit_fast_tools_list_is_exact(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.tools")
    confirmer = FakeConfirmer([True])
    fast = _routed_gate("fast", confirmer, fast_tools=[NOTE])
    # A listed side-effecting tool may run on the fast route, but it is still confirmed.
    assert await run_gate(fast, NOTE) == ALLOW
    assert len(confirmer.asked) == 1
    # A read-only tool that isn't listed is refused on the fast route...
    assert await run_gate(fast, LIST) == _deny(REASON_FAST_ROUTE)
    assert f"tool denied: {LIST} (fast route)" in caplog.text
    # ...and allowed on the deep route, which has the full set.
    assert await run_gate(_routed_gate("deep", fast_tools=[NOTE]), LIST) == {}
    assert fast.in_fast_set(NOTE) and not fast.in_fast_set(LIST)


async def test_a_failing_route_provider_means_no_fast_limit(caplog):
    def broken() -> str:
        raise RuntimeError("boom")

    confirmer = FakeConfirmer([False])
    gate = ConfirmationGate(notes_setup().policy, confirmer, route_provider=broken)
    # deep behaviour: side effects are still confirmed, so failing open here is safe
    assert await run_gate(gate, NOTE) == _deny(REASON_DECLINED)
    assert len(confirmer.asked) == 1
    assert "route_provider failed" in caplog.text


def test_confirm_config_requires_the_hook_timeout_to_outlast_the_question():
    assert ConfirmConfig().ask_budget_s == 110.0
    with pytest.raises(ValueError, match="hook_timeout_s"):
        ConfirmConfig(timeout_s=8, text_timeout_s=60, hook_timeout_s=60)


# --- start_project through the gate (P3.4 wiring) ----------------------------------------------


class _Host:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_repo(self, name, *, private, description, source=None):
        from spark.tools.project import RepoInfo

        self.calls.append(name)
        return RepoInfo(full_name=f"me/{name}", url=f"https://github.com/me/{name}")


class _Tracker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create_task(self, title, body, *, repo=None, cwd=None):
        from spark.tools.project import TaskInfo

        self.calls.append(title)
        return TaskInfo(ref="#1", url="https://github.com/me/x/issues/1")


class _Launcher:
    def __init__(self) -> None:
        self.opened: list[Path] = []

    def open(self, path: Path) -> None:
        self.opened.append(path)


def _project_brain(tmp_path: Path, confirmer: FakeConfirmer):
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.config import ProjectsConfig
    from spark.tools.project import ProjectService
    from spark.tools.timers import TimerService

    host, tracker, launcher = _Host(), _Tracker(), _Launcher()
    service = ProjectService(
        ProjectsConfig(), root=tmp_path, host=host, tracker=tracker, launcher=launcher
    )
    timers = TimerService()
    brain = ClaudeAgentBrain(
        make_settings(brain={"backend": "claude_agent"}),
        mcp=McpSetup(),
        timers=timers,
        projects=service,
        confirmer=confirmer,
    )
    brain.tool_announce_wait_s = 0.01  # no stream in this test: don't wait for the tool block
    from spark.tools.server import build_spark_tools

    tools = build_spark_tools(timers, None, projects=service)
    handler = next(t.handler for t in tools if t.name == "start_project")
    return brain, timers, host, tracker, handler


@pytest.fixture
def project_brain(tmp_path):
    return lambda confirmer: _project_brain(tmp_path, confirmer)


async def _cli_like_call(brain, handler, args) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """What the CLI does: ask the hook, and run the tool only on an explicit allow."""
    name = "mcp__spark__start_project"
    decision = await brain.tool_gate(hook_input(name, args), "toolu_1", {"signal": None})
    ran = None
    if decision.get("hookSpecificOutput", {}).get("permissionDecision") == "allow":
        ran = await handler(args)
    return decision, ran


async def test_start_project_is_side_effecting_and_no_creates_nothing(project_brain, tmp_path):
    confirmer = FakeConfirmer([False])
    brain, timers, host, tracker, handler = project_brain(confirmer)
    try:
        assert brain.mcp.policy.classify("mcp__spark__start_project") == "side_effect"
        assert "mcp__spark__start_project" not in brain.mcp.allowed_tools
        decision, ran = await _cli_like_call(brain, handler, {"name": "weather-bot"})
        assert decision == _deny(REASON_DECLINED)
        assert ran is None
        assert host.calls == [] and tracker.calls == []
        assert not (tmp_path / "weather-bot").exists()
        # the project tool's own summarizer produced the question
        assert len(confirmer.asked) == 1
        assert "weather-bot" in confirmer.asked[0]
        assert confirmer.asked[0].endswith("?")
        assert not _LEAK.search(confirmer.asked[0])
    finally:
        await brain.close()
        await timers.close()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
async def test_start_project_yes_runs_it(project_brain, tmp_path, monkeypatch):
    # the real `git commit` needs an identity; CI runners have none configured
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Spark Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "spark-test@example.com")
    brain, timers, host, _tracker, handler = project_brain(FakeConfirmer([True]))
    try:
        decision, ran = await _cli_like_call(brain, handler, {"name": "weather-bot"})
        assert decision == ALLOW
        assert ran is not None and not ran.get("is_error")
        assert (tmp_path / "weather-bot").is_dir()
        assert host.calls == ["weather-bot"]
    finally:
        await brain.close()
        await timers.close()


# --- TextConfirmer and LineReader --------------------------------------------------------------


class _Reader:
    def __init__(self, *lines: str | None) -> None:
        self.lines = list(lines)
        self.timeouts: list[float | None] = []

    async def readline(self, timeout=None):  # noqa: ASYNC109 - mirrors LineReader
        self.timeouts.append(timeout)
        return self.lines.pop(0)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("y\n", True),
        ("yes\n", True),
        ("Sure\n", True),
        ("go ahead\n", True),
        ("n\n", False),
        ("no\n", False),
        ("maybe\n", False),  # anything else is no
        ("\n", False),
        (None, None),  # timeout
        ("", None),  # end of input
    ],
)
async def test_text_confirmer(line, expected):
    out = io.StringIO()
    reader = _Reader(line)
    confirmer = TextConfirmer(reader=reader, out=out, timeout_s=7.0)
    assert await confirmer.ask("I'll create note called groceries. Go ahead?") is expected
    assert "I'll create note called groceries. Go ahead? (yes/no)> " in out.getvalue()
    assert reader.timeouts == [7.0]


async def test_text_confirmer_with_say_prints_only_the_choices():
    said: list[str] = []

    async def say(text):
        said.append(text)

    out = io.StringIO()
    confirmer = TextConfirmer(reader=_Reader("yes\n"), out=out, say=say)
    assert await confirmer.ask("Go ahead?") is True
    assert said == ["Go ahead?"]
    assert out.getvalue() == "(yes/no)> "


async def test_line_reader_reads_lines_then_sticky_eof():
    reader = LineReader(io.StringIO("hello\nno\n"))
    assert await reader.readline(1.0) == "hello\n"
    assert await reader.readline(1.0) == "no\n"
    assert await reader.readline(1.0) == ""
    assert await reader.readline(1.0) == ""


async def test_line_reader_times_out_without_losing_the_next_line():
    import threading

    gate = threading.Event()

    class SlowStream:
        def __init__(self) -> None:
            self.lines = ["late answer\n", ""]

        def readline(self):
            gate.wait(5)
            return self.lines.pop(0)

    reader = LineReader(SlowStream())
    assert await reader.readline(0.05) is None  # timed out
    gate.set()
    assert await reader.readline(2.0) == "late answer\n"  # the same line is not swallowed


# --- VoiceConfirmer ----------------------------------------------------------------------------


def _voice(answers: list[str | None], **kw):
    tts = FakeTTS()
    heard_timeouts: list[float] = []

    async def listen(timeout_s: float) -> str:
        heard_timeouts.append(timeout_s)
        answer = answers.pop(0)
        if answer is None:  # never answers: the confirmer's own timeout must fire
            await asyncio.sleep(3600)
        return answer

    confirmer = VoiceConfirmer(listen=listen, tts=tts, **kw)
    return confirmer, tts, heard_timeouts


async def test_voice_confirmer_yes():
    confirmer, tts, timeouts = _voice(["yeah go ahead"], timeout_s=8.0)
    assert await confirmer.ask("I'll create note called groceries. Go ahead?") is True
    assert tts.spoken == ["I'll create note called groceries. Go ahead?"]
    assert timeouts == [8.0]
    assert confirmer.heard == ["yeah go ahead"]


async def test_voice_confirmer_no():
    confirmer, _tts, _ = _voice(["no, don't"])
    assert await confirmer.ask("Go ahead?") is False


async def test_voice_confirmer_silence_is_no_answer():
    confirmer, tts, _ = _voice([""])
    assert await confirmer.ask("Go ahead?") is None
    assert tts.spoken == ["Go ahead?"]  # no reprompt after silence


async def test_voice_confirmer_hung_listen_times_out():
    confirmer, _, _ = _voice([None], timeout_s=0.02, answer_grace_s=0.02)
    assert await confirmer.ask("Go ahead?") is None


async def test_voice_confirmer_reprompts_once_after_an_unclear_answer():
    confirmer, tts, _ = _voice(["what was that about", "yes"])
    assert await confirmer.ask("Go ahead?") is True
    assert tts.spoken == ["Go ahead?", VoiceConfirmer.REPROMPT]


async def test_voice_confirmer_unclear_twice_is_no_answer():
    confirmer, _, _ = _voice(["hmm", "I'm not sure"])
    assert await confirmer.ask("Go ahead?") is None


async def test_voice_confirmer_without_reprompt():
    confirmer, tts, _ = _voice(["maybe"], reprompt=False)
    assert await confirmer.ask("Go ahead?") is None
    assert tts.spoken == ["Go ahead?"]


async def test_voice_confirmer_plays_the_cue_before_listening():
    order: list[str] = []

    async def say(text):
        order.append("say")

    async def cue():
        order.append("cue")

    async def listen(timeout_s):
        order.append("listen")
        return "yes"

    assert await VoiceConfirmer(listen=listen, say=say, cue=cue).ask("Go ahead?") is True
    assert order == ["say", "cue", "listen"]


# --- the app: a tool call mid-turn, confirmed through the mic path -----------------------------

BLOCK = 512


async def _feed(app: SparkApp, pcm) -> None:
    for i in range(0, len(pcm), BLOCK):
        await app.feed(pcm[i : i + BLOCK])


async def _until(pred, limit_s: float = 5.0) -> None:
    async with asyncio.timeout(limit_s):
        while not pred():  # noqa: ASYNC110 - polling app state that has no event
            await asyncio.sleep(0.001)


def _voice_app(tmp_path, answer: str):
    settings = make_settings(
        logging={"dir": str(tmp_path / "logs")},
        debug_audio={"enabled": False, "dir": str(tmp_path / "dbg")},
    )
    brain = ToolCallingBrain(settings)
    app = SparkApp(
        settings,
        mic=FakeMic(),
        player=FakePlayer(),
        wakeword=FakeWakeWord(),
        vad=FakeVAD(),
        stt=FakeSTT(transcripts=["make a note called groceries", answer]),
        tts=FakeTTS(),
        brain=brain,
    )
    confirmer = VoiceConfirmer(
        listen=app.listen_for_answer,
        say=app.say_for_confirmation,
        cue=app.confirmation_cue,
        timeout_s=settings.confirm.timeout_s,
    )
    app.confirmer = confirmer
    brain.gate = ConfirmationGate(notes_setup().policy, confirmer)
    return app, brain


async def _ask_by_voice(app: SparkApp) -> None:
    app.wakeword.trigger()
    await _feed(app, silence(BLOCK / 16000))
    await _feed(app, concat(tone(0.6), silence(0.8)))  # "make a note called groceries"
    await _until(lambda: app.state is State.CONFIRMING)


@pytest.mark.parametrize("answer", ["no", "nope, don't"])
async def test_app_voice_no_denies_the_tool_and_returns_to_normal(tmp_path, answer):
    app, brain = _voice_app(tmp_path, answer)
    await _ask_by_voice(app)
    question = "I'll create note called groceries. Should I go ahead?"
    # The reply so far and then the question were spoken before listening started.
    assert app.tts.spoken == ["Sure, I can do that.", question]
    assert app.confirm_questions == [question]
    assert app.player.earcons[-1] == "wake"  # the "your turn" cue
    await _feed(app, concat(tone(0.5), silence(0.8)))  # the spoken answer
    await asyncio.wait_for(app.turn_task, 5)
    assert brain.executed == []
    assert brain.decisions == [_deny(REASON_DECLINED)]
    assert app.tts.spoken[-1] == "Okay, I won't do that."
    assert app.state is State.FOLLOW_UP
    history = app.state_history
    i = history.index(State.CONFIRMING)
    assert history[i - 1] is State.SPEAKING and history[i + 1] is State.THINKING
    assert history[-2:] == [State.SPEAKING, State.FOLLOW_UP]
    # wake-word and follow-up logic never saw the answer frames
    assert app.wakeword.detections == [0]
    assert app.stt.results == ["make a note called groceries", answer]
    assert app.last_timer is not None and app.last_timer.meta.get("confirmations") == 1


async def test_app_voice_yes_runs_the_tool(tmp_path):
    app, brain = _voice_app(tmp_path, "yes, go ahead")
    await _ask_by_voice(app)
    await _feed(app, concat(tone(0.5), silence(0.8)))
    await asyncio.wait_for(app.turn_task, 5)
    assert brain.executed == [{"title": "groceries"}]
    assert brain.decisions == [ALLOW]
    assert app.tts.spoken[-1] == "Done, the note is saved."
    assert app.state is State.FOLLOW_UP


async def test_app_voice_silence_is_no_answer(tmp_path):
    app, brain = _voice_app(tmp_path, "unused")
    await _ask_by_voice(app)
    await _feed(app, silence(app.settings.confirm.timeout_s + 0.2))
    await asyncio.wait_for(app.turn_task, 5)
    assert brain.executed == []
    assert brain.decisions == [_deny(REASON_NO_ANSWER)]
    assert app.stt.results == ["make a note called groceries"]  # silence never transcribed
    assert app.state is State.FOLLOW_UP


async def test_app_cancel_during_confirmation_denies(tmp_path):
    app, brain = _voice_app(tmp_path, "yes")
    await _ask_by_voice(app)
    turn = app.turn_task
    await app.cancel_turn()
    assert turn is not None and turn.done()
    assert brain.executed == []


async def test_app_text_mode_question_follows_the_reply(tmp_path):
    settings = make_settings(logging={"dir": str(tmp_path / "logs")})
    brain = ToolCallingBrain(settings)
    tts = FakeTTS()
    app = SparkApp(
        settings, mic=None, player=None, wakeword=None, vad=None, stt=None, tts=tts, brain=brain
    )
    out = io.StringIO()
    confirmer = TextConfirmer(reader=_Reader("n\n"), out=out, say=app.say_for_confirmation)
    brain.gate = ConfirmationGate(notes_setup().policy, confirmer)
    reply = await app.text_turn("make a note called groceries")
    assert tts.spoken == [
        "Sure, I can do that.",
        "I'll create note called groceries. Should I go ahead?",
        "Okay, I won't do that.",
    ]
    assert reply == "Sure, I can do that. Okay, I won't do that."
    assert brain.executed == []
    assert out.getvalue() == "(yes/no)> "
    assert app.state is State.IDLE


def test_from_settings_builds_text_and_voice_confirmers(tmp_path, monkeypatch):
    from spark import registry

    settings = make_settings(logging={"dir": str(tmp_path / "logs")})
    built: dict[str, Any] = {}

    def fake_build(kind, s, **deps):
        if kind == "brain":
            built["brain_deps"] = deps
            return ToolCallingBrain(s)
        return {
            "tts": FakeTTS,
            "audio_out": FakePlayer,
            "audio_in": FakeMic,
            "wakeword": FakeWakeWord,
            "vad": FakeVAD,
            "stt": FakeSTT,
        }[kind](s)

    monkeypatch.setattr(registry, "build", fake_build)
    text_app = SparkApp.from_settings(settings, audio=False)
    assert isinstance(text_app.confirmer, TextConfirmer)
    assert built["brain_deps"]["confirmer"] is text_app.confirmer
    assert text_app.line_reader is text_app.confirmer.reader
    assert text_app.confirmer.timeout_s == settings.confirm.text_timeout_s

    voice_app = SparkApp.from_settings(settings, audio=True)
    assert isinstance(voice_app.confirmer, VoiceConfirmer)
    assert built["brain_deps"]["confirmer"] is voice_app.confirmer
    assert voice_app.confirmer.timeout_s == settings.confirm.timeout_s
    assert voice_app.line_reader is None


# --- live --------------------------------------------------------------------------------------


def _live_available() -> bool:
    return bool(shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY"))


def _notes_server(ran: list[dict[str, Any]]) -> McpSetup:
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        "create_note", "Create a note with the given title in the user's notes app.", {"title": str}
    )
    async def create_note(args: dict[str, Any]) -> dict[str, Any]:
        ran.append(dict(args))
        return {"content": [{"type": "text", "text": "Note created."}]}

    setup = McpSetup()
    server = create_sdk_mcp_server(name="notes", version="1.0.0", tools=[create_note])
    setup.add_sdk_server("notes", server, side_effecting=["create_note"])
    return setup


async def _live_text_turn(answer: str) -> tuple[list[dict[str, Any]], SparkApp, str]:
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.config import load_settings

    settings = load_settings()
    ran: list[dict[str, Any]] = []
    app_ref: list[SparkApp] = []

    async def say(text: str) -> None:
        await app_ref[0].say_for_confirmation(text)

    confirmer = TextConfirmer(
        reader=LineReader(io.StringIO(answer + "\n")), out=io.StringIO(), say=say, timeout_s=30
    )
    brain = ClaudeAgentBrain(settings, mcp=_notes_server(ran), confirmer=confirmer)
    app = SparkApp(
        settings,
        mic=None,
        player=None,
        wakeword=None,
        vad=None,
        stt=None,
        tts=FakeTTS(),
        brain=brain,
        confirmer=confirmer,
    )
    app_ref.append(app)
    try:
        await brain.warmup()
        reply = await app.text_turn("Please create a note titled groceries.")
    finally:
        await app.close()
    return ran, app, reply


_ACK = re.compile(
    r"\b(won't|will not|didn't|did not|not|cancel\w*|okay|ok|alright|no problem|understood|"
    r"got it|sure thing|no worries|leave it)\b",
    re.I,
)


@pytest.mark.live
async def test_live_confirm_no(caplog):
    """Phase 3: saying no prevents the side-effecting action, and Spark acknowledges it."""
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    caplog.set_level(logging.INFO, logger="spark")
    ran, app, reply = await _live_text_turn("no")
    spoken = app.tts.spoken
    print(f"\n[live] asked={app.confirm_questions} spoken={spoken} ran={ran}")
    assert "tool confirm: mcp__notes__create_note -> no" in caplog.text
    assert ran == [], "the tool ran although the user said no"
    assert app.confirm_questions and "groceries" in app.confirm_questions[0]
    assert reply, "Spark said nothing after the refusal"
    assert _ACK.search(reply), f"no acknowledgement in {reply!r}"
    assert not set(reply) & MARKDOWN_CHARS


@pytest.mark.live
async def test_live_confirm_yes(caplog):
    """The allow direction: an explicit hook allow runs a tool that is not in allowed_tools."""
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    caplog.set_level(logging.INFO, logger="spark")
    ran, app, _reply = await _live_text_turn("yes")
    print(f"\n[live] asked={app.confirm_questions} spoken={app.tts.spoken} ran={ran}")
    assert "tool confirm: mcp__notes__create_note -> yes" in caplog.text
    assert ran and ran[0].get("title", "").lower() == "groceries"
    # the question was spoken after the text that preceded the call, before the result
    spoken = app.tts.spoken
    assert app.confirm_questions[0] in spoken


class _AlwaysFast:
    """A route classifier that labels every request fast, so a live turn that wants a side
    effect is forced onto the fast route (the rules alone would send it deep)."""

    async def classify(self, text: str) -> str | None:
        return "fast"

    async def warmup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _notes_server_with_list(ran: list[str]) -> McpSetup:
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        "create_note", "Create a note with the given title in the user's notes app.", {"title": str}
    )
    async def create_note(args: dict[str, Any]) -> dict[str, Any]:
        ran.append(f"create_note:{args.get('title', '')}")
        return {"content": [{"type": "text", "text": "Note created."}]}

    @tool("list_notes", "List the titles of the user's notes.", {})
    async def list_notes(args: dict[str, Any]) -> dict[str, Any]:
        ran.append("list_notes")
        return {"content": [{"type": "text", "text": "Notes: groceries, packing."}]}

    setup = McpSetup()
    server = create_sdk_mcp_server(name="notes", version="1.0.0", tools=[create_note, list_notes])
    setup.add_sdk_server("notes", server, read_only=["list_notes"], side_effecting=["create_note"])
    return setup


@pytest.mark.live
async def test_live_confirm_fast_route_refuses_without_asking_then_deep_retry_asks(caplog):
    """Spec §8.4 fast path has a limited tool set: a fast turn's side-effecting call is refused
    by the gate without a question, and the retry (routed deep) is confirmed and runs."""
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.brain.router import Router
    from spark.config import load_settings

    caplog.set_level(logging.INFO, logger="spark")
    settings = load_settings()
    ran: list[str] = []
    confirmer = FakeConfirmer([True])
    router = Router(
        settings.routing,
        fast_model=settings.brain.fast_model,
        deep_model=settings.brain.deep_model,
        classifier=_AlwaysFast(),
    )
    brain = ClaudeAgentBrain(
        settings, mcp=_notes_server_with_list(ran), confirmer=confirmer, router=router
    )
    try:
        await brain.warmup()
        first = "".join([d async for d in brain.respond("Please create a note titled groceries.")])
        first_stats = brain.last_turn
        print(f"\n[live] fast: route={first_stats.route} asked={confirmer.asked} ran={ran}")
        print(f"[live] fast reply: {first!r}")
        assert first_stats.route == "fast"
        assert "tool denied: mcp__notes__create_note (fast route)" in caplog.text, (
            "the model did not try the tool on the fast turn"
        )
        assert confirmer.asked == [], "asked on the fast route"
        assert ran == [], "the tool ran on the fast route"
        assert first.strip() and not set(first) & MARKDOWN_CHARS

        second = "".join([d async for d in brain.respond("Okay, please do it now.")])
        stats = brain.last_turn
        print(f"[live] retry: route={stats.route} ({stats.route_reason}) asked={confirmer.asked}")
        print(f"[live] retry reply: {second!r} ran={ran}")
        assert stats.route == "deep"
        assert stats.route_reason.startswith("retry after a fast-route tool refusal")
        assert confirmer.asked, "the deep retry did not ask"
        assert ran == ["create_note:groceries"]
    finally:
        await brain.close()


@pytest.mark.live
async def test_live_confirm_fast_route_deny_beats_allowed_tools(caplog):
    """An explicit routing.fast_tools list: a read-only tool that IS in allowed_tools but not
    in the fast set is refused on a fast turn, i.e. the hook's deny wins over allowed_tools."""
    if not _live_available():
        pytest.skip("no claude CLI login or ANTHROPIC_API_KEY")
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.config import load_settings

    caplog.set_level(logging.INFO, logger="spark")
    settings = load_settings()
    settings.routing.fast_tools = ["mcp__notes__create_note"]
    ran: list[str] = []
    brain = ClaudeAgentBrain(
        settings, mcp=_notes_server_with_list(ran), confirmer=FakeConfirmer([False])
    )
    try:
        assert "mcp__notes__list_notes" in brain._build_options().allowed_tools
        await brain.warmup()
        reply = "".join([d async for d in brain.respond("List my notes.")])
        print(f"\n[live] route={brain.last_turn.route} ran={ran} reply={reply!r}")
        assert brain.last_turn.route == "fast"
        assert "tool denied: mcp__notes__list_notes (fast route)" in caplog.text, (
            "the model did not try list_notes"
        )
        assert "list_notes" not in ran, "allowed_tools overrode the hook's deny"
    finally:
        await brain.close()
