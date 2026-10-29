"""A fresh Claude session per wake, with a carry-over briefing (Phase 5, P5.6)."""

from __future__ import annotations

import datetime as dt

from spark.activity import ActivityLog
from spark.briefing import build_briefing, conversations, last_conversation
from tests.fakes import make_settings
from tests.test_activity import TZ, Clock


def _history(tmp_path) -> tuple[ActivityLog, Clock]:
    clock = Clock(dt.datetime(2026, 10, 2, 12, 0, tzinfo=TZ))
    log = ActivityLog(tmp_path / "history", now=clock)
    log.start_conversation("voice")
    log.user("how many orders shipped this month", "voice")
    log.record("tool_call", tool="mcp__WAREHOUSE__warehouse_query", input={"sql": "select"})
    log.assistant("Forty two orders shipped in October.")
    log.user("and in Phoenix?", "follow_up")
    log.assistant("Twelve of them are in Phoenix.")
    clock.advance(minutes=3)
    log.end_conversation("awake window over")
    return log, clock


def test_conversations_are_reassembled(tmp_path):
    log, clock = _history(tmp_path)
    (conv,) = conversations(log.iter_events())
    assert conv.closed and conv.end_reason == "awake window over"
    assert conv.exchanges[0] == ("user", "how many orders shipped this month")
    assert conv.exchanges[-1] == ("assistant", "Twelve of them are in Phoenix.")
    assert last_conversation(log, clock()).id == conv.id


def test_a_recent_conversation_is_carried_over(tmp_path):
    log, clock = _history(tmp_path)
    clock.advance(minutes=10)
    s = make_settings(app={"name": "Spark"})
    text = build_briefing(s, log, now=clock())
    assert "it ended about 10 minutes ago" in text
    assert "User: how many orders shipped this month" in text
    assert "Spark: Twelve of them are in Phoenix." in text
    assert "warehouse_query" not in text  # exchanges only, no tool noise


def test_an_old_conversation_is_not_carried_over(tmp_path):
    log, clock = _history(tmp_path)
    clock.advance(minutes=31)
    assert build_briefing(make_settings(), log, now=clock()) == ""


def test_carry_over_keeps_only_the_last_exchanges(tmp_path):
    log, clock = _history(tmp_path)
    s = make_settings(attention={"carry_over_exchanges": 1})
    text = build_briefing(s, log, now=clock())
    assert "and in Phoenix?" in text
    assert "how many tenants" not in text


def test_an_open_conversation_is_not_carried_over(tmp_path):
    log, clock = _history(tmp_path)
    log.start_conversation("voice")
    log.user("unfinished", "voice")
    text = build_briefing(make_settings(), log, now=clock())
    assert "unfinished" not in text and "Phoenix" in text


def test_no_history_no_briefing(tmp_path):
    log = ActivityLog(tmp_path / "none")
    assert build_briefing(make_settings(), log) == ""
