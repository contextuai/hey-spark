"""Spark's memory, part 1: connector notes (Phase 5, P5.9)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from spark.memory import MemoryStore, MemoryWriteError
from tests.fakes import make_settings

CONV = "c-20261002-161303-ab12"


def _store(tmp_path, **kw) -> MemoryStore:
    return MemoryStore(
        tmp_path / "memory",
        today=lambda: dt.date(2026, 10, 2),
        conversation=lambda: CONV,
        **kw,
    )


def test_notes_are_markdown_with_their_source(tmp_path):
    m = _store(tmp_path)
    first = "Orders: SALES.dbo.WAREHOUSE_Orders, OrderDate"
    assert m.add_connector_note("WAREHOUSE", first) == "Noted for WAREHOUSE."
    m.add_connector_note("WAREHOUSE", "Filter active customers with Status = 'Active'")
    text = m.connector_path("WAREHOUSE").read_text(encoding="utf-8")
    assert text.startswith("# WAREHOUSE notes")
    assert f"- {first} (2026-10-02, {CONV})" in text
    assert m.connector_bullets("WAREHOUSE") == [
        first,
        "Filter active customers with Status = 'Active'",
    ]


def test_duplicates_long_notes_and_full_files(tmp_path):
    m = _store(tmp_path, note_chars=60, file_chars=300)
    m.add_connector_note("WAREHOUSE", "Products live in dbo.Products")
    assert (
        m.add_connector_note("WAREHOUSE", "products live in dbo.products")
        == "Already noted for WAREHOUSE."
    )
    with pytest.raises(MemoryWriteError, match="under 60"):
        m.add_connector_note("WAREHOUSE", "x" * 61)
    with pytest.raises(MemoryWriteError, match="empty"):
        m.add_connector_note("WAREHOUSE", "   ")
    for i in range(10):
        try:
            m.add_connector_note("WAREHOUSE", f"note number {i} about some table")
        except MemoryWriteError as exc:
            assert "rewrite_connector_notes" in str(exc)
            break
    else:
        pytest.fail("the file never filled up")


def test_rewrite_replaces_and_keeps_sources(tmp_path):
    m = _store(tmp_path)
    m.add_connector_note("WAREHOUSE", "Wrong table: dbo.Old")
    m.rewrite_connector_notes("WAREHOUSE", ["Orders are in dbo.Orders", "# heading ignored", ""])
    assert m.connector_bullets("WAREHOUSE") == ["Orders are in dbo.Orders"]
    m.rewrite_connector_notes("WAREHOUSE", "- A (2026-09-01, c-old)\n- B")
    text = m.connector_notes("WAREHOUSE")
    assert "- A (2026-09-01, c-old)" in text  # an existing source is kept
    assert f"- B (2026-10-02, {CONV})" in text


def test_secrets_never_reach_the_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_MCP_WAREHOUSE_TOKEN", "tok-very-secret-123456")
    m = _store(tmp_path)
    m.add_connector_note("WAREHOUSE", "auth header is Bearer tok-very-secret-123456")
    assert "tok-very-secret-123456" not in m.connector_notes("WAREHOUSE")


def test_a_hand_edited_file_is_read_back(tmp_path):
    m = _store(tmp_path)
    path = m.connector_path("WAREHOUSE")
    path.parent.mkdir(parents=True)
    path.write_text(
        "# mine\n\n* Invoices: dbo.Invoices\nsome prose\n- Products: dbo.Products (2026-09-30)\n",
        encoding="utf-8",
    )
    assert m.connector_bullets("WAREHOUSE") == ["Invoices: dbo.Invoices", "Products: dbo.Products"]


def test_briefing_includes_only_enabled_connectors_within_budget(tmp_path):
    m = _store(tmp_path)
    m.add_connector_note("WAREHOUSE", "Orders are in dbo.Orders")
    m.add_connector_note("CATALOG", "Sales report is report 12")
    text = m.connector_briefing(["WAREHOUSE"], max_chars=8000)
    assert "What you already know about the connected systems" in text
    assert "WAREHOUSE:\n- Orders are in dbo.Orders" in text
    assert "CATALOG" not in text
    assert m.connector_briefing(["TICKETS"], max_chars=8000) == ""
    for i in range(40):
        m.add_connector_note("WAREHOUSE", f"table number {i} holds thing {i} with several columns")
    assert len(m.connector_briefing(["WAREHOUSE"], max_chars=600)) < 900


def test_build_briefing_puts_notes_first(tmp_path):
    from spark.activity import ActivityLog
    from spark.briefing import build_briefing

    m = _store(tmp_path)
    m.add_connector_note("WAREHOUSE", "Orders are in dbo.Orders")
    text = build_briefing(
        make_settings(), ActivityLog(tmp_path / "h"), memory=m, connectors=["WAREHOUSE"]
    )
    assert text.startswith("What you already know")


# --- the tools ----------------------------------------------------------------------------------


def _tools(tmp_path, connectors=("WAREHOUSE", "CATALOG"), activity=None) -> dict[str, Any]:
    from spark.tools.server import build_spark_tools
    from spark.tools.timers import TimerService
    from tests.fakes import FakeClock

    tools = build_spark_tools(
        TimerService(clock=FakeClock()),
        memory=_store(tmp_path),
        connectors=lambda: list(connectors),
        activity=activity,
    )
    return {t.name: t for t in tools}


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


async def test_save_note_tool_resolves_the_server_and_records_it(tmp_path):
    from spark.activity import ActivityLog

    activity = ActivityLog(tmp_path / "history")
    tools = _tools(tmp_path, activity=activity)
    res = await tools["save_connector_note"].handler(
        {"server": "mcp__warehouse__warehouse_query", "note": "Orders are in dbo.Orders"}
    )
    assert _text(res) == "Noted for WAREHOUSE." and not res.get("is_error")
    bad = await tools["save_connector_note"].handler({"server": "WIKI", "note": "x"})
    assert bad["is_error"] and "WAREHOUSE, CATALOG" in _text(bad)
    (event,) = [e for e in activity.iter_events() if e["kind"] == "memory"]
    assert event["server"] == "WAREHOUSE" and event["action"] == "note"


async def test_rewrite_tool(tmp_path):
    tools = _tools(tmp_path)
    res = await tools["rewrite_connector_notes"].handler(
        {"server": "warehouse", "notes": ["A is in dbo.A", "B is in dbo.B"]}
    )
    assert _text(res) == "Rewrote the notes for WAREHOUSE (2 notes)."


def test_memory_tools_exist_only_with_a_store_and_are_pre_approved(tmp_path):
    from spark.tools.server import build_spark_server, spark_tool_names
    from spark.tools.timers import TimerService
    from tests.fakes import FakeClock

    _, plain, _ = build_spark_server(TimerService(clock=FakeClock()))
    assert plain == spark_tool_names()
    _, with_memory, side = build_spark_server(
        TimerService(clock=FakeClock()), memory=_store(tmp_path)
    )
    assert set(with_memory) - set(plain) == {
        "mcp__spark__save_connector_note",
        "mcp__spark__rewrite_connector_notes",
    }
    assert side == ["mcp__spark__start_project"]


def test_connector_note_rules_only_with_memory_and_an_external_connector(tmp_path, monkeypatch):
    import sys

    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.brain.prompts import CONNECTOR_NOTES_RULES
    from spark.tools.timers import TimerService
    from tests.fakes import FakeClock
    from tests.test_brain_claude_agent import github_setup, make_fake_sdk, settings

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", make_fake_sdk())
    memory = _store(tmp_path)

    def brain(**kw):
        return ClaudeAgentBrain(settings(), timers=TimerService(clock=FakeClock()), **kw)

    both = brain(mcp=github_setup(), memory=memory)
    assert CONNECTOR_NOTES_RULES in both.system_prompt
    assert "mcp__spark__save_connector_note" in both.mcp.allowed_tools
    assert both.external_connectors == ["github"]
    assert CONNECTOR_NOTES_RULES not in brain(memory=memory).system_prompt
    no_memory = brain(mcp=github_setup())
    assert CONNECTOR_NOTES_RULES not in no_memory.system_prompt
    assert "mcp__spark__save_connector_note" not in no_memory.mcp.allowed_tools
