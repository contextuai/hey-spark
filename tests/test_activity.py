"""The activity log (Phase 5, P5.5): chain, redaction, tool results, app and brain wiring."""

from __future__ import annotations

import datetime as dt
import itertools
import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from spark.activity import GENESIS, ActivityLog, event_hash, verify_file
from tests.test_brain_claude_agent import (  # noqa: F401
    AssistantMessage,
    FakeClient,
    SystemMessage,
    ToolUseBlock,
    _tool_start,
    fake_sdk,
    github_setup,
    result,
    text_delta,
    text_start,
)


class Clock:
    def __init__(self, start: dt.datetime) -> None:
        self.t = start

    def __call__(self) -> dt.datetime:
        return self.t

    def advance(self, **kw: float) -> None:
        self.t += dt.timedelta(**kw)


TZ = dt.timezone(dt.timedelta(hours=-4))


def _log(tmp_path, **kw) -> tuple[ActivityLog, Clock]:
    clock = Clock(dt.datetime(2026, 10, 2, 12, 0, tzinfo=TZ))
    return ActivityLog(tmp_path / "history", now=clock, **kw), clock


def _lines(path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def test_events_are_chained_and_verify(tmp_path):
    log, _ = _log(tmp_path)
    conv = log.start_conversation("voice")
    log.user("how many orders shipped this month", "voice")
    log.assistant("Forty two.")
    log.end_conversation("awake window over")
    events = _lines(log.day_path(dt.date(2026, 10, 2)))
    assert [e["kind"] for e in events] == [
        "conversation_start",
        "user",
        "assistant",
        "conversation_end",
    ]
    assert [e["seq"] for e in events] == [1, 2, 3, 4]
    assert all(e["conv"] == conv for e in events)
    assert events[0]["prev"] == GENESIS
    assert all(b["prev"] == a["hash"] for a, b in itertools.pairwise(events))
    assert all(e["hash"] == event_hash(e["prev"], e) for e in events)
    assert log.verify() == (True, None)
    assert log.conversation is None


def test_tampering_is_detected(tmp_path):
    log, _ = _log(tmp_path)
    log.user("send the report to Sam", "voice")
    log.assistant("Sent.")
    log.user("thanks", "follow_up")
    path = log.day_path(dt.date(2026, 10, 2))
    lines = path.read_text(encoding="utf-8").splitlines()
    edited = json.loads(lines[1])
    edited["text"] = "Not sent."
    lines[1] = json.dumps(edited)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify() == (False, "2026-10-02.jsonl line 2")
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")  # a line deleted
    assert log.verify()[0] is False


def test_the_chain_continues_across_days_and_restarts(tmp_path):
    log, clock = _log(tmp_path)
    log.user("first", "voice")
    clock.advance(days=1)
    log.user("second", "voice")
    again = ActivityLog(log.dir, now=clock)  # a restart picks up the tail
    again.user("third", "voice")
    day2 = _lines(log.day_path(dt.date(2026, 10, 3)))
    day1 = _lines(log.day_path(dt.date(2026, 10, 2)))
    assert day2[0]["prev"] == day1[-1]["hash"]
    assert [e["seq"] for e in day2] == [2, 3]
    assert again.verify() == (True, None)
    ok, bad, _ = verify_file(log.day_path(dt.date(2026, 10, 3)), prev=GENESIS)
    assert not ok and bad == 1  # the day's first link points at yesterday, not genesis


def test_secrets_are_redacted_everywhere(tmp_path, monkeypatch):
    secret = "warehouse-token-value-0123456789"
    monkeypatch.setenv("SPARK_MCP_WAREHOUSE_TOKEN", secret)
    log, _ = _log(tmp_path)
    log.tool_call("mcp__WAREHOUSE__warehouse_query", "t1", {"sql": "select 1", "auth": secret})
    log.tool_result("mcp__WAREHOUSE__warehouse_query", "t1", f"ok {secret}", False)
    log.user(f"my token is {secret}", "text")
    text = log.day_path(dt.date(2026, 10, 2)).read_text(encoding="utf-8")
    assert secret not in text
    assert "select 1" in text
    assert log.verify() == (True, None)


@pytest.mark.parametrize(
    ("mode", "content", "truncated"),
    [("full", "x" * 3000, None), ("excerpt", "x" * 200, True), ("meta", None, None)],
)
def test_tool_result_modes(tmp_path, mode, content, truncated):
    log, _ = _log(tmp_path, tool_results=mode, excerpt_chars=200)
    blocks = [{"type": "text", "text": "x" * 3000}]
    log.tool_result("mcp__WAREHOUSE__warehouse_query", "t1", blocks, None)
    (event,) = _lines(log.day_path(dt.date(2026, 10, 2)))
    assert event["chars"] == 3000 and event["is_error"] is False
    assert event.get("content") == content
    assert event.get("truncated") == truncated


def test_purge_keeps_everything_by_default(tmp_path):
    log, clock = _log(tmp_path)
    log.user("old", "voice")
    clock.advance(days=400)
    log.user("new", "voice")
    assert log.purge() == []
    assert len(log.days()) == 2
    short, _ = _log(tmp_path, retention_days=30)
    short._now = clock
    assert short.purge() == [dt.date(2026, 10, 2)]
    assert short.verify() == (True, None)  # the oldest kept day may follow purged ones


def test_disabled_writes_nothing(tmp_path):
    log, _ = _log(tmp_path, enabled=False)
    log.start_conversation("voice")
    log.user("hi", "voice")
    assert not log.dir.exists()


def test_a_write_failure_never_raises(tmp_path, caplog):
    blocker = tmp_path / "history"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    log = ActivityLog(blocker)
    assert log.record("user", text="hi") is None
    assert log.record("user", text="again") is None
    assert caplog.text.count("activity: could not write") == 1


def test_listeners_see_written_events(tmp_path):
    log, _ = _log(tmp_path)
    seen: list[str] = []
    log.listeners.append(lambda e: seen.append(e["kind"]))
    log.user("hi", "voice")
    assert seen == ["user"]


# --- app wiring ----------------------------------------------------------------------------------


async def test_a_voice_conversation_is_recorded_and_closed_on_sleep(tmp_path):
    from tests.test_app import _app, _settings, _utterance, _wake

    activity, _ = _log(tmp_path)
    app = _app(
        tmp_path,
        settings=_settings(tmp_path, attention={"awake_s": 2.0}),
        activity=activity,
    )
    from tests.fakes import silence
    from tests.test_app import _feed, _finish_turn

    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    await _feed(app, silence(3.0), block=16000)
    kinds = [e["kind"] for e in activity.iter_events()]
    assert kinds == ["conversation_start", "user", "assistant", "conversation_end"]
    events = list(activity.iter_events())
    assert events[1]["text"] == "what is two plus two" and events[1]["mode"] == "voice"
    assert events[2]["text"] == "Two plus two is four."
    assert events[3]["reason"] == "awake window over"
    assert activity.verify() == (True, None)
    await app.close()


async def test_commands_ignored_speech_and_failures_are_recorded(tmp_path):
    from spark.brain.mock import MockBrain
    from tests.fakes import FakeSTT
    from tests.test_app import _app, _finish_turn, _settings, _utterance, _wake

    activity, _ = _log(tmp_path)
    stt = FakeSTT(transcripts=["what is two plus two", "lunch was good", "go to sleep"])
    app = _app(tmp_path, stt=stt, activity=activity, settings=_settings(tmp_path))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    events = list(activity.iter_events())
    kinds = [e["kind"] for e in events]
    assert kinds == [
        "conversation_start",
        "user",
        "assistant",
        "ignored",
        "command",
        "assistant",
        "conversation_end",
    ]
    assert events[3]["text"] == "lunch was good" and events[3]["reason"] == "no request"
    assert events[4]["command"] == "sleep"
    assert events[5]["text"] == "Okay."
    assert len({e["conv"] for e in events}) == 1

    failing = _app(
        tmp_path,
        brain=MockBrain(fail_with=ConnectionError("offline")),
        activity=activity,
    )
    await _wake(failing)
    await _utterance(failing)
    await _finish_turn(failing)
    from tests.test_app import _settle

    await _settle(lambda: list(activity.iter_events())[-1]["kind"] == "announce")
    error, ended, said = list(activity.iter_events())[-3:]
    assert error["kind"] == "error" and error["error"] == "ConnectionError"
    assert ended["kind"] == "conversation_end" and ended["reason"] == "failed"
    assert said["text"].startswith("Sorry")  # the spoken failure line (P5.1)
    await app.close()
    await failing.close()


# --- brain wiring --------------------------------------------------------------------------------


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


@dataclass
class UserMessage:
    content: Any
    parent_tool_use_id: str | None = None


async def test_the_brain_records_route_tools_results_and_the_session(fake_sdk, tmp_path):  # noqa: F811
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.tools.timers import TimerService
    from tests.fakes import FakeClock
    from tests.test_brain_claude_agent import settings as brain_settings

    fake_sdk.UserMessage = UserMessage
    fake_sdk.ToolResultBlock = ToolResultBlock
    query = "mcp__github__list_pull_requests"

    def script(text: str, client: FakeClient) -> list[Any]:
        return [
            SystemMessage("init"),
            _tool_start("toolu_1"),
            AssistantMessage([ToolUseBlock("toolu_1", query, {"repo": "acme/api"})], "haiku"),
            UserMessage([ToolResultBlock("toolu_1", [{"type": "text", "text": "3 open PRs"}])]),
            text_start(2),
            text_delta("You have three."),
            result(num_turns=2, session_id="sess-42"),
        ]

    FakeClient.script = staticmethod(script)
    activity, _ = _log(tmp_path)
    b = ClaudeAgentBrain(
        brain_settings(),
        mcp=github_setup(),
        timers=TimerService(clock=FakeClock()),
        activity=activity,
    )
    try:
        await b.warmup()
        assert list(activity.iter_events()) == []  # warm-up never shows up
        out = [d async for d in b.respond("any open pull requests?")]
        assert "".join(out) == "You have three."
    finally:
        await b.close()
    events = list(activity.iter_events())
    assert [e["kind"] for e in events] == ["route", "tool_call", "tool_result", "turn_end"]
    route, call, res, end = events
    assert route["route"] == "fast"
    assert call["tool"] == query and call["input"] == {"repo": "acme/api"}
    assert res["tool"] == query and res["content"] == "3 open PRs" and res["chars"] == 10
    assert end["session_id"] == "sess-42" and end["num_turns"] == 2
    assert activity.verify() == (True, None)


def test_tool_details_stay_out_of_the_app_log(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    log, _ = _log(tmp_path)
    log.tool_call("mcp__x__y", "t", {"ssn": "123-45-6789"})
    assert "123-45-6789" not in caplog.text
