"""In-process tools: timers, current_time, spark_status, and the ``spark`` SDK MCP server."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from spark.tools.clock_tools import current_time_text
from spark.tools.status import describe_status, spark_status_text
from spark.tools.timers import (
    Timer,
    TimerError,
    TimerService,
    normalize_label,
    parse_duration,
    speak_duration,
)
from tests.fakes import FakeClock


class ManualClock:
    """A clock whose ``sleep`` blocks until the test calls ``advance`` past the wake time."""

    def __init__(self) -> None:
        self.now = 0.0
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (self.now + seconds, fut)
        self._waiters.append(entry)
        try:
            await fut
        finally:
            self._waiters.remove(entry)

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        for due, fut in list(self._waiters):
            if due <= self.now and not fut.done():
                fut.set_result(None)
        for _ in range(10):
            await asyncio.sleep(0)


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------- parse_duration


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        (300, 300.0),
        (2.5, 2.5),
        ("300", 300.0),
        ("5 minutes", 300.0),
        ("5 min", 300.0),
        ("1h30m", 5400.0),
        ("1h 30m 15s", 5415.0),
        ("90 seconds", 90.0),
        ("an hour and a half", 5400.0),
        ("half an hour", 1800.0),
        ("a minute and a half", 90.0),
        ("one and a half hours", 5400.0),
        ("twenty five minutes", 1500.0),
        ("twenty-five minutes", 1500.0),
        ("ten seconds", 10.0),
        ("2 hours 15", 8100.0),
        ("1:30", 90.0),
        ("1:00:00", 3600.0),
        ("a couple of minutes", 120.0),
        ("  5 Minutes ", 300.0),
    ],
)
def test_parse_duration_accepts(value: Any, seconds: float) -> None:
    assert parse_duration(value) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "value", ["", "soon", "five bananas", "minutes", True, None, "-5 minutes", "1 2 minutes"]
)
def test_parse_duration_rejects_with_speakable_error(value: Any) -> None:
    with pytest.raises(TimerError) as info:
        parse_duration(value)
    message = str(info.value)
    assert message.endswith(".") and "{" not in message


def test_speak_duration() -> None:
    assert speak_duration(300) == "5 minutes"
    assert speak_duration(330) == "5 minutes and 30 seconds"
    assert speak_duration(5400) == "1 hour and 30 minutes"
    assert speak_duration(1) == "1 second"
    assert speak_duration(0.2) == "1 second"
    assert speak_duration(0) == "less than a second"
    assert speak_duration(3725) == "1 hour and 2 minutes"


def test_normalize_label() -> None:
    assert normalize_label("The Pasta Timer") == "pasta"
    assert normalize_label("my tea") == "tea"
    assert normalize_label("  ") is None
    assert normalize_label(None) is None


# --------------------------------------------------------------------------- TimerService


async def test_set_and_list_remaining() -> None:
    clock = ManualClock()
    svc = TimerService(clock=clock)
    pasta = svc.set("5 minutes", label="pasta")
    tea = svc.set(60)
    assert (pasta.id, pasta.label, pasta.seconds) == ("1", "pasta", 300.0)
    assert (tea.id, tea.label, tea.spoken_name) == ("2", None, "timer 2")
    assert pasta.spoken_name == "pasta timer"

    await clock.advance(20)
    listed = svc.list()
    assert [t.id for t in listed] == ["2", "1"]  # soonest first
    assert listed[0].remaining == pytest.approx(40)
    assert listed[1].remaining == pytest.approx(280)
    await svc.close()


async def test_fire_calls_on_fire_once_and_removes_timer() -> None:
    clock = ManualClock()
    fired: list[Timer] = []
    svc = TimerService(clock=clock)

    async def on_fire(timer: Timer) -> None:
        fired.append(timer)

    svc.on_fire = on_fire  # settable after construction
    svc.set(10, label="eggs")
    await clock.advance(9)
    assert fired == []
    await clock.advance(1)
    assert [t.label for t in fired] == ["eggs"]
    assert fired[0].remaining == 0.0
    assert svc.list() == []
    await clock.advance(100)
    assert len(fired) == 1
    await svc.close()


async def test_fire_with_fake_clock_and_sync_callback() -> None:
    fired: list[str] = []
    svc = TimerService(clock=FakeClock(), on_fire=lambda t: fired.append(t.spoken_name))
    svc.set("90 seconds", label="tea")
    await settle()
    assert fired == ["tea timer"]
    assert len(svc) == 0
    await svc.close()


async def test_on_fire_exception_is_contained() -> None:
    def boom(timer: Timer) -> None:
        raise RuntimeError("speaker unplugged")

    svc = TimerService(clock=FakeClock(), on_fire=boom)
    svc.set(5)
    await settle()
    assert len(svc) == 0
    svc.set(5)  # service still usable
    await svc.close()


async def test_cancel_by_label_id_and_spoken_forms() -> None:
    clock = ManualClock()
    fired: list[Timer] = []
    svc = TimerService(clock=clock, on_fire=fired.append)
    svc.set(300, label="pasta")
    svc.set(600, label="laundry")
    svc.set(900)

    cancelled = svc.cancel("the pasta timer")
    assert cancelled is not None and cancelled.label == "pasta"
    assert svc.cancel("pasta") is None  # already gone
    three = svc.cancel("timer three")
    assert three is not None and three.id == "3"
    laundry = svc.cancel("2")
    assert laundry is not None and laundry.label == "laundry"
    assert svc.list() == []

    await clock.advance(1000)
    assert fired == []  # nothing cancelled ever fires
    await svc.close()


async def test_cancel_ambiguous_returns_none_and_cancel_all() -> None:
    clock = ManualClock()
    svc = TimerService(clock=clock)
    svc.set(60, label="red tea")
    svc.set(60, label="green tea")
    assert len(svc.find("tea")) == 2
    assert svc.cancel("tea") is None
    assert len(svc) == 2
    assert {t.label for t in svc.cancel_all()} == {"red tea", "green tea"}
    assert len(svc) == 0
    await svc.close()


async def test_ids_restart_when_no_timers_left() -> None:
    svc = TimerService(clock=ManualClock())
    svc.set(60)
    svc.cancel_all()
    assert svc.set(60).id == "1"
    await svc.close()


async def test_max_timers_and_duplicate_label() -> None:
    svc = TimerService(clock=ManualClock(), max_timers=2)
    svc.set(60, label="pasta")
    with pytest.raises(TimerError, match="already have a pasta timer"):
        svc.set(60, label="Pasta timer")
    svc.set(60)
    with pytest.raises(TimerError, match="2 timers running"):
        svc.set(60)
    await svc.close()


@pytest.mark.parametrize(
    ("duration", "fragment"),
    [
        (0, "at least one second"),
        (0.5, "at least one second"),
        (86401, "24 hours"),
        ("25 hours", "24 hours"),
        (float("nan"), "understand"),
    ],
)
async def test_bad_durations(duration: Any, fragment: str) -> None:
    svc = TimerService(clock=ManualClock())
    with pytest.raises(TimerError, match=fragment):
        svc.set(duration)
    assert len(svc) == 0
    await svc.close()


async def test_close_cancels_tasks_and_blocks_new_timers() -> None:
    clock = ManualClock()
    fired: list[Timer] = []
    svc = TimerService(clock=clock, on_fire=fired.append)
    svc.set(60)
    svc.set(120)
    tasks = list(svc._tasks.values())
    assert len(tasks) == 2
    await svc.close()
    assert all(t.done() for t in tasks)
    assert svc._tasks == {} and svc.list() == []
    await clock.advance(1000)
    assert fired == []
    with pytest.raises(TimerError):
        svc.set(60)
    await svc.close()  # idempotent


# --------------------------------------------------------------------------- current_time


def test_current_time_with_fixed_now() -> None:
    pdt = timezone(timedelta(hours=-7), "Pacific Daylight Time")
    fixed = datetime(2026, 9, 24, 15, 7, tzinfo=pdt)
    assert current_time_text(lambda: fixed) == (
        "It's 3:07 PM on Thursday, September 24, 2026, Pacific Daylight Time."
    )


def test_current_time_on_the_hour_and_numeric_zone() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))  # tzname() is "UTC+05:30"
    assert current_time_text(lambda: datetime(2026, 1, 1, 9, 0, tzinfo=ist)) == (
        "It's 9 AM on Thursday, January 1, 2026, UTC plus 5 30."
    )
    assert "12 noon" in current_time_text(lambda: datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    assert current_time_text(lambda: datetime(2026, 1, 1, 0, 5)).startswith("It's 12:05 AM")


# --------------------------------------------------------------------------- spark_status


def test_describe_status_is_speakable_and_drops_secrets() -> None:
    text = describe_status(
        {
            "state": "IDLE",
            "backends": {"wakeword": "porcupine", "stt": "faster_whisper", "tts": "piper"},
            "model": "claude-haiku-4-5-20251001",
            "uptime_s": 3725,
            "connectors": ["github", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"],
            "api_key": "sk-ant-secret-value",
            "session_token": "abc",
        }
    )
    assert "idle right now" in text
    assert "porcupine for wake word" in text
    assert "faster whisper for speech recognition" in text
    assert "claude haiku 4 5." in text
    assert "1 hour and 2 minutes" in text
    assert "github" in text
    for leaked in ("sk-ant", "ghp_", "abc", "api", "token", "_", "{", "20251001"):
        assert leaked not in text


async def test_spark_status_text_with_async_provider_and_timers() -> None:
    svc = TimerService(clock=ManualClock())
    svc.set(240, label="pasta")

    async def provider() -> dict[str, Any]:
        return {"working_on": "your weekly summary"}

    text = await spark_status_text(provider, svc)
    assert text == (
        "I'm working on your weekly summary. You have one timer: pasta timer with 4 minutes left."
    )
    await svc.close()


async def test_spark_status_text_survives_broken_provider() -> None:
    def provider() -> dict[str, Any]:
        raise RuntimeError("nope")

    text = await spark_status_text(provider, None)
    assert text == "I couldn't read my own status just now. No timers running."
    assert await spark_status_text(None, None) == "I'm up and listening. No timers running."


# --------------------------------------------------------------------------- SDK server


def _result_text(result: dict[str, Any]) -> str:
    assert set(result) <= {"content", "is_error"}
    content = result["content"]
    assert isinstance(content, list) and len(content) == 1
    assert content[0]["type"] == "text"
    text = content[0]["text"]
    assert isinstance(text, str) and text
    assert "{" not in text and "mcp__" not in text and "http" not in text
    return text


FIXED_NOW = datetime(2026, 9, 24, 15, 7, tzinfo=timezone(timedelta(hours=-7), "PDT"))


@pytest.fixture
async def spark_tools() -> Any:
    from spark.tools.server import build_spark_tools

    clock = ManualClock()
    svc = TimerService(clock=clock)
    tools = build_spark_tools(
        svc, status_provider=lambda: {"state": "thinking"}, now=lambda: FIXED_NOW
    )
    yield {t.name: t for t in tools}, svc, clock
    await svc.close()


def test_build_spark_server_config_and_names() -> None:
    from spark.tools.server import (
        SERVER_NAME,
        build_spark_server,
        full_tool_name,
        spark_side_effecting_names,
        spark_tool_names,
    )

    svc = TimerService(clock=ManualClock())
    config, read_only, side_effecting = build_spark_server(svc, status_provider=dict)
    assert config["type"] == "sdk"
    assert config["name"] == SERVER_NAME == "spark"
    assert config["instance"] is not None
    expected = [
        "mcp__spark__current_time",
        "mcp__spark__set_timer",
        "mcp__spark__list_timers",
        "mcp__spark__cancel_timer",
        "mcp__spark__spark_status",
    ]
    assert read_only == expected == spark_tool_names()
    # start_project (P3.4) is the one side-effecting spark tool: confirmed, never pre-approved
    assert side_effecting == ["mcp__spark__start_project"] == spark_side_effecting_names()
    assert full_tool_name("set_timer") == "mcp__spark__set_timer"


async def test_tool_objects_and_schemas(spark_tools: Any) -> None:
    import jsonschema
    from claude_agent_sdk import SdkMcpTool

    tools, _, _ = spark_tools
    assert list(tools) == [
        "current_time",
        "set_timer",
        "list_timers",
        "cancel_timer",
        "spark_status",
        "start_project",
    ]
    for t in tools.values():
        assert isinstance(t, SdkMcpTool)
        assert asyncio.iscoroutinefunction(t.handler)
    schema = tools["set_timer"].input_schema
    # The SDK validates arguments with jsonschema before calling the handler, so both the
    # number and the text form of a duration must pass, and a missing duration must not.
    jsonschema.validate({"duration": 300}, schema)
    jsonschema.validate({"duration": "5 minutes", "label": "pasta"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"label": "pasta"}, schema)

    def wire(name: str) -> dict[str, Any]:
        return tools[name].annotations.model_dump(by_alias=True, exclude_none=True)

    assert wire("current_time")["readOnlyHint"] is True
    assert wire("set_timer")["destructiveHint"] is False


async def test_current_time_tool(spark_tools: Any) -> None:
    tools, _, _ = spark_tools
    result = await tools["current_time"].handler({})
    assert "is_error" not in result
    assert _result_text(result) == "It's 3:07 PM on Thursday, September 24, 2026, PDT."


async def test_timer_tools_round_trip(spark_tools: Any) -> None:
    tools, svc, clock = spark_tools
    result = await tools["set_timer"].handler({"duration": "5 minutes", "label": "pasta"})
    assert _result_text(result) == "Started the pasta timer for 5 minutes."
    result = await tools["set_timer"].handler({"duration": 90})
    assert _result_text(result) == "Started timer 2 for 1 minute and 30 seconds."

    await clock.advance(30)
    text = _result_text(await tools["list_timers"].handler({}))
    assert text == (
        "You have 2 timers: timer 2 with 1 minute left and pasta timer with 4 minutes and "
        "30 seconds left."
    )

    result = await tools["cancel_timer"].handler({"timer": "pasta"})
    assert _result_text(result) == "Cancelled the pasta timer."
    result = await tools["cancel_timer"].handler({})  # only one left, no name needed
    assert _result_text(result) == "Cancelled timer 2."
    assert _result_text(await tools["list_timers"].handler({})) == "No timers running."
    assert len(svc) == 0


async def test_cancel_all_tool(spark_tools: Any) -> None:
    tools, svc, _ = spark_tools
    await tools["set_timer"].handler({"duration": 60, "label": "a1"})
    await tools["set_timer"].handler({"duration": 60, "label": "b2"})
    assert _result_text(await tools["cancel_timer"].handler({"timer": "all"})) == (
        "Cancelled all 2 timers."
    )
    assert len(svc) == 0


@pytest.mark.parametrize(
    ("tool_name", "args", "fragment"),
    [
        ("set_timer", {"duration": "soon"}, "couldn't understand"),
        ("set_timer", {"duration": 0}, "at least one second"),
        ("set_timer", {"duration": 999999}, "24 hours"),
        ("set_timer", {}, "How long"),
        ("cancel_timer", {"timer": "pasta"}, "no timers running"),
    ],
)
async def test_tools_return_speakable_errors(
    spark_tools: Any, tool_name: str, args: dict[str, Any], fragment: str
) -> None:
    tools, _, _ = spark_tools
    result = await tools[tool_name].handler(args)
    assert result.get("is_error") is True
    assert fragment in _result_text(result)


async def test_cancel_unknown_and_ambiguous(spark_tools: Any) -> None:
    tools, _, _ = spark_tools
    await tools["set_timer"].handler({"duration": 60, "label": "red tea"})
    await tools["set_timer"].handler({"duration": 60, "label": "green tea"})
    unknown = await tools["cancel_timer"].handler({"timer": "pizza"})
    assert unknown["is_error"] is True
    assert _result_text(unknown).startswith("I don't have a timer called pizza.")
    ambiguous = await tools["cancel_timer"].handler({"timer": "tea"})
    assert ambiguous["is_error"] is True
    assert _result_text(ambiguous) == "Which one, the red tea timer or the green tea timer?"


async def test_spark_status_tool(spark_tools: Any) -> None:
    tools, _, _ = spark_tools
    text = _result_text(await tools["spark_status"].handler({}))
    assert text == "I'm thinking right now. No timers running."


async def test_unexpected_exception_becomes_generic_error() -> None:
    from spark.tools.server import build_spark_tools

    class BrokenTimers(TimerService):
        def list(self) -> list[Timer]:
            raise RuntimeError("internal detail /tmp/x 0xdeadbeef")

    tools = {t.name: t for t in build_spark_tools(BrokenTimers(clock=ManualClock()))}
    result = await tools["list_timers"].handler({})
    assert result["is_error"] is True
    assert _result_text(result) == "Sorry, something went wrong with that."


def test_importing_tools_does_not_import_sdk() -> None:
    code = (
        "import sys, spark.tools.server, spark.tools.timers, spark.tools.status, "
        "spark.tools.clock_tools; assert 'claude_agent_sdk' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
