"""claude.ai account connectors (Gmail, Google Calendar, ...): config, policy, hiding, status.

Offline: the SDK is the fake from test_brain_claude_agent. The live test at the bottom opens a
real session and checks the init tool list (``-m live``)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from spark import doctor
from spark.brain import claudeai
from spark.brain import connector_store as cs
from spark.brain.claude_agent import ClaudeAgentBrain
from spark.brain.mcp_config import claudeai_enabled, env_for_sdk, load_mcp_servers
from spark.config import (
    CLAUDEAI_CALENDAR_ASK,
    CLAUDEAI_CALENDAR_OFF,
    CLAUDEAI_CALENDAR_READ,
    CLAUDEAI_GMAIL_ASK,
    CLAUDEAI_GMAIL_OFF,
    CLAUDEAI_GMAIL_READ,
    Settings,
    load_settings,
)
from tests.test_brain_claude_agent import (  # noqa: F401
    FakeClient,
    collect,
    fake_sdk,
)
from tests.test_brain_claude_agent import settings as brain_settings

ROOT = Path(__file__).resolve().parents[1]
ON = {"ENABLE_CLAUDEAI_MCP_SERVERS": "true"}
OFF = {"ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
#: Every Gmail tool the connector offered on 2026-10-05.
GMAIL_TOOLS = (  # noqa: SIM905
    "create_draft update_draft delete_draft send_message reply forward list_drafts get_draft "
    "get_thread get_message search_threads label_thread unlabel_thread "
    "apply_sensitive_thread_label trash_thread untrash_thread mark_thread_spam "
    "unmark_thread_spam list_labels label_message update_message_labels unlabel_message "
    "apply_sensitive_message_label trash_message untrash_message mark_message_spam "
    "unmark_message_spam create_label update_label delete_label"
).split()
CALENDAR_TOOLS = (  # noqa: SIM905
    "list_events get_event list_calendars suggest_time create_event update_event delete_event "
    "respond_to_event search_events"
).split()


def _servers_with_claudeai_on() -> dict[str, Any]:
    """The default servers with the claude_ai ones switched on (they ship disabled)."""
    servers = {k: v.model_dump() for k, v in Settings().connectors.servers.items()}
    for name, cfg in servers.items():
        if name.startswith("claude_ai_"):
            cfg["enabled"] = True
    return servers


def _settings(tmp_path: Path, **connectors: Any) -> Settings:
    """Defaults with the claude_ai servers on, no mcp_servers.json, files in tmp_path."""
    connectors.setdefault("servers", _servers_with_claudeai_on())
    return Settings(config_dir=tmp_path, connectors=connectors)


def _status_entry(name: str, status: str = "connected", tools: list[str] | None = None):
    return {
        "name": name,
        "status": status,
        "source": "claudeai",
        "scope": "claudeai",
        "serverInfo": {"name": name},
        "config": {"type": "claudeai-proxy", "url": "https://mcp-proxy.example/secret-id"},
        "tools": [{"name": t} for t in tools or []],
    }


# --- config ------------------------------------------------------------------------------------


def test_defaults_cover_every_gmail_and_calendar_tool_once():
    for read, ask, off, every in (
        (CLAUDEAI_GMAIL_READ, CLAUDEAI_GMAIL_ASK, CLAUDEAI_GMAIL_OFF, GMAIL_TOOLS),
        (CLAUDEAI_CALENDAR_READ, CLAUDEAI_CALENDAR_ASK, CLAUDEAI_CALENDAR_OFF, CALENDAR_TOOLS),
    ):
        listed = [*read, *ask, *off]
        assert sorted(listed) == sorted(every)
        assert len(set(listed)) == len(listed)
    assert "send_message" in CLAUDEAI_GMAIL_OFF and "create_draft" in CLAUDEAI_GMAIL_ASK
    assert CLAUDEAI_CALENDAR_OFF == ("delete_event",)


def test_config_defaults_and_repo_yaml_have_the_claudeai_servers(monkeypatch):
    s = Settings()
    assert s.connectors.block_claudeai_connectors is False
    for name in ("claude_ai_Gmail", "claude_ai_Google_Calendar"):
        cfg = s.connectors.servers[name]
        # off by default: they need a claude.ai login, which the public app doesn't offer
        assert not cfg.enabled and cfg.route == "deep" and not cfg.allow_destructive
    loaded = load_settings(ROOT / "config.yaml", load_env_file=False).connectors
    assert loaded.block_claudeai_connectors is False
    assert loaded.servers["claude_ai_Gmail"] == s.connectors.servers["claude_ai_Gmail"]
    assert (
        loaded.servers["claude_ai_Google_Calendar"]
        == s.connectors.servers["claude_ai_Google_Calendar"]
    )


def test_server_key_mapping():
    assert claudeai.server_key("claude.ai Gmail") == "claude_ai_Gmail"
    assert claudeai.server_key("claude.ai Google Calendar") == "claude_ai_Google_Calendar"
    assert claudeai.server_key("claude.ai Microsoft 365") == "claude_ai_Microsoft_365"
    assert claudeai.server_key("claude_ai_Slack") == "claude_ai_Slack"
    assert claudeai.server_key("github") is None
    assert claudeai.server_key("claude.ai ") is None
    assert claudeai.display_name("claude_ai_Google_Calendar") == "Google Calendar"


# --- loading and policy ------------------------------------------------------------------------


def test_load_registers_policy_without_a_json_entry(tmp_path):
    setup = load_mcp_servers(_settings(tmp_path, servers=_only_claudeai()), environ={})
    assert setup.claudeai == ["claude_ai_Gmail", "claude_ai_Google_Calendar"]
    assert "claude_ai_Gmail" not in setup.servers  # the CLI injects it
    assert "claude_ai_Gmail" not in setup.disabled
    p = setup.policy
    assert p.classify("mcp__claude_ai_Gmail__search_threads") == "read"
    assert p.classify("mcp__claude_ai_Gmail__create_draft") == "side_effect"
    assert p.classify("mcp__claude_ai_Gmail__send_message") == "destructive"
    assert p.is_blocked("mcp__claude_ai_Gmail__send_message")
    assert p.classify("mcp__claude_ai_Gmail__authenticate") == "unknown"
    assert p.classify("mcp__claude_ai_Slack__slack_send_message") == "unknown"
    assert p.classify("mcp__claude_ai_Google_Calendar__create_event") == "side_effect"
    assert "mcp__claude_ai_Gmail__get_thread" in setup.allowed_tools
    assert "mcp__claude_ai_Google_Calendar__list_events" in setup.allowed_tools
    assert "mcp__claude_ai_Gmail__create_draft" not in setup.allowed_tools
    # destructive tools of claude.ai servers are deny rules, like any server's
    assert "mcp__claude_ai_Gmail__send_message" in setup.disallowed_tools
    assert "mcp__claude_ai_Google_Calendar__delete_event" in setup.disallowed_tools
    assert setup.deep_route["claude_ai_Gmail"][:2] == ("claude_ai_gmail", "claude ai gmail")
    assert "inbox" in setup.deep_route["claude_ai_Gmail"]


def _only_claudeai() -> dict[str, Any]:
    return {k: v for k, v in _servers_with_claudeai_on().items() if k.startswith("claude_")}


def test_unconfigured_claudeai_servers_are_hidden_server_wide(tmp_path):
    settings = _settings(tmp_path, servers=_only_claudeai())
    # the first session: only the static common list is hidden
    setup = load_mcp_servers(settings, environ={})
    assert "mcp__claude_ai_Slack" in setup.disallowed_tools
    assert "mcp__claude_ai_Gmail" not in setup.disallowed_tools
    # after the CLI reported the account's connectors, those are hidden too
    claudeai.save_state(
        settings,
        claudeai.parse_status(
            [
                _status_entry("claude.ai Gmail", tools=["search_threads"]),
                _status_entry("claude.ai Notion", tools=["notion_search"]),
            ]
        ),
    )
    setup = load_mcp_servers(settings, environ={})
    assert "mcp__claude_ai_Notion" in setup.disallowed_tools
    assert "claude_ai_Notion" in setup.claudeai_hidden
    assert "claude_ai_Gmail" not in setup.claudeai_hidden


def test_disabled_claudeai_server_is_hidden_too(tmp_path):
    servers = _only_claudeai()
    servers["claude_ai_Google_Calendar"]["enabled"] = False
    setup = load_mcp_servers(_settings(tmp_path, servers=servers), environ={})
    assert setup.claudeai == ["claude_ai_Gmail"]
    assert "mcp__claude_ai_Google_Calendar" in setup.disallowed_tools


def test_block_forces_them_off(tmp_path):
    settings = _settings(tmp_path, servers=_only_claudeai(), block_claudeai_connectors=True)
    setup = load_mcp_servers(settings, environ={})
    assert setup.claudeai == [] and setup.claudeai_hidden == []
    assert "block_claudeai_connectors" in setup.disabled["claude_ai_Gmail"]
    assert setup.policy.classify("mcp__claude_ai_Gmail__search_threads") == "unknown"
    assert env_for_sdk(settings) == OFF and claudeai_enabled(settings) == []


def test_no_claudeai_server_means_no_hiding_and_env_off(tmp_path):
    settings = _settings(tmp_path, servers={})
    setup = load_mcp_servers(settings, environ={})
    assert setup.claudeai == [] and not any("claude_ai" in t for t in setup.disallowed_tools)
    assert env_for_sdk(settings) == OFF


# --- window choices (connectors.json) override config ------------------------------------------


def test_window_choices_override_config_defaults(tmp_path):
    settings = _settings(tmp_path, servers=_only_claudeai())
    cs.set_claudeai(settings, "claude_ai_Gmail", tools={"send_message": "ask", "get_thread": "off"})
    cs.set_claudeai(settings, "claude_ai_Google_Calendar", enabled=False)
    cs.set_claudeai(
        settings, "claude_ai_Slack", enabled=True, tools={"slack_search_public": "allow"}
    )
    setup = load_mcp_servers(settings, environ={})
    p = setup.policy
    assert p.classify("mcp__claude_ai_Gmail__send_message") == "side_effect"
    assert p.classify("mcp__claude_ai_Gmail__get_thread") == "destructive"
    assert "mcp__claude_ai_Gmail__get_thread" in setup.disallowed_tools
    assert "mcp__claude_ai_Gmail__send_message" not in setup.disallowed_tools
    assert p.classify("mcp__claude_ai_Gmail__search_threads") == "read"  # untouched default
    assert setup.claudeai == ["claude_ai_Gmail", "claude_ai_Slack"]
    assert p.classify("mcp__claude_ai_Slack__slack_search_public") == "read"
    assert p.classify("mcp__claude_ai_Slack__slack_send_message") == "unknown"
    assert "mcp__claude_ai_Slack" not in setup.disallowed_tools
    assert "mcp__claude_ai_Google_Calendar" in setup.disallowed_tools
    with pytest.raises(cs.ConnectorError):
        cs.set_claudeai(settings, "github", enabled=True)
    with pytest.raises(cs.ConnectorError):
        cs.set_claudeai(settings, "claude_ai_Gmail", tools={"x": "maybe"})


def test_pasted_servers_cannot_take_a_claudeai_name(tmp_path):
    settings = _settings(tmp_path, servers={})
    r = cs.add_servers(settings, '{"claude_ai_Gmail": {"url": "https://x.test/mcp"}}')
    assert r.added == [] and "reserved" in r.skipped["claude_ai_Gmail"]


# --- brain options -----------------------------------------------------------------------------


def _brain_connectors(servers: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "mcp_config": "no-such-mcp-servers.json",
        "ui_file": "no-such-connectors.json",
        "servers": _only_claudeai() if servers is None else servers,
        **extra,
    }


async def test_brain_session_is_non_strict_only_with_claudeai(fake_sdk):  # noqa: F811
    b = ClaudeAgentBrain(brain_settings(_brain_connectors()))
    try:
        opts = b._build_options()
        assert opts.strict_mcp_config is False
        assert opts.env == ON
        assert "mcp__claude_ai_Slack" in opts.disallowed_tools
        assert "mcp__claude_ai_Gmail__send_message" in opts.disallowed_tools
        assert "mcp__claude_ai_Gmail__search_threads" in opts.allowed_tools
        assert "claude_ai_Gmail" not in opts.mcp_servers
        assert b.external_connectors == ["claude_ai_Gmail", "claude_ai_Google_Calendar"]
    finally:
        await b.close()
    b = ClaudeAgentBrain(brain_settings(_brain_connectors(block_claudeai_connectors=True)))
    try:
        opts = b._build_options()
        assert opts.strict_mcp_config is True and opts.env == OFF
        assert not any("claude_ai" in t for t in opts.allowed_tools)
    finally:
        await b.close()


async def test_researcher_gets_the_claudeai_read_tools(fake_sdk):  # noqa: F811
    b = ClaudeAgentBrain(brain_settings(_brain_connectors(), workers=True))
    try:
        assert b.workers_active
        tools = b.worker_tools()
        assert "mcp__claude_ai_Gmail__search_threads" in tools
        assert "mcp__claude_ai_Gmail__create_draft" not in tools
    finally:
        await b.close()


def test_router_and_side_sessions_stay_strict_and_blocked(fake_sdk, tmp_path):  # noqa: F811
    from spark.brain.router import FastModelClassifier
    from spark.brain.side import SideModel

    s = _settings(tmp_path)  # defaults: claude.ai servers enabled
    side = SideModel(s, "Answer briefly.", sdk=fake_sdk)
    for opts in (FastModelClassifier(s, sdk=fake_sdk)._options(), side._options()):
        assert opts.strict_mcp_config is True and opts.env == OFF


# --- live status -------------------------------------------------------------------------------


async def test_status_maps_names_saves_seen_servers_and_hides_others(fake_sdk, caplog):  # noqa: F811
    caplog.set_level("INFO", logger="spark.brain")
    settings = brain_settings(_brain_connectors())
    FakeClient.mcp_status = {
        "mcpServers": [
            _status_entry("claude.ai Gmail", "needs-auth", ["authenticate"]),
            _status_entry("claude.ai Google Calendar", "connected", CALENDAR_TOOLS),
            _status_entry("claude.ai Slack", "connected", ["slack_send_message"]),
            _status_entry("claude.ai Microsoft 365", "needs-auth", ["authenticate"]),
            {"name": "spark", "status": "connected"},
        ]
    }
    seen: list[dict[str, Any]] = []
    b = ClaudeAgentBrain(settings)
    b.on_mcp_status = seen.append
    try:
        await b.warmup()
        assert b.mcp_status == {
            "claude_ai_Gmail": {"status": "needs-auth", "error": claudeai.SIGN_IN_HINT},
            "claude_ai_Google_Calendar": {"status": "connected"},
        }
        assert b.unreachable_connectors() == ["claude_ai_Gmail"]
        state = claudeai.load_state(settings)
        assert set(state) == {
            "claude_ai_Gmail",
            "claude_ai_Google_Calendar",
            "claude_ai_Slack",
            "claude_ai_Microsoft_365",
        }
        assert state["claude_ai_Google_Calendar"]["tools"] == sorted(CALENDAR_TOOLS)
        assert "secret-id" not in claudeai.state_path(settings).read_text(encoding="utf-8")
        assert "secret-id" not in caplog.text
        await collect(b, "how many unread emails do I have")
        query = FakeClient.instances[-1].queries[-1]
        assert "claude_ai_Gmail is unreachable right now" in query
        assert "sign in on claude.ai (Settings -> Connectors) or run /mcp" in query
        assert "Microsoft" not in query
    finally:
        await b.close()
    # the next session hides what was seen
    setup = load_mcp_servers(settings, environ={})
    assert "mcp__claude_ai_Slack" in setup.disallowed_tools
    assert "mcp__claude_ai_Microsoft_365" in setup.disallowed_tools


def test_state_file_is_isolated_and_robust(tmp_path):
    settings = _settings(tmp_path)
    assert claudeai.state_path(settings).parent != ROOT  # conftest keeps it in tmp
    assert claudeai.load_state(settings) == {}
    claudeai.state_path(settings).write_text("{broken", encoding="utf-8")
    assert claudeai.load_state(settings) == {}
    entry = {"claude_ai_X": {"name": "claude.ai X", "status": "connected", "tools": ["b", "a"]}}
    assert claudeai.save_state(settings, entry) is True
    assert claudeai.save_state(settings, entry) is False  # unchanged: not rewritten
    assert claudeai.load_state(settings)["claude_ai_X"]["tools"] == ["a", "b"]


# --- doctor ------------------------------------------------------------------------------------


def test_doctor_reports_claudeai(tmp_path):
    assert doctor.check_claudeai in doctor.default_checks()
    blocked = doctor.check_claudeai(_settings(tmp_path, block_claudeai_connectors=True))
    assert blocked.ok and "block_claudeai_connectors" in blocked.detail
    none = doctor.check_claudeai(_settings(tmp_path, servers={}))
    assert none.ok and "no claude_ai_* server" in none.detail
    settings = _settings(tmp_path)
    fresh = doctor.check_claudeai(settings)
    assert fresh.ok and "claude_ai_Gmail (not seen yet)" in fresh.detail
    claudeai.save_state(
        settings,
        claudeai.parse_status(
            [
                _status_entry("claude.ai Gmail", "needs-auth"),
                _status_entry("claude.ai Google Calendar"),
                _status_entry("claude.ai Slack"),
            ]
        ),
    )
    c = doctor.check_claudeai(settings)
    assert not c.ok and not c.required
    assert "claude_ai_Gmail (needs-auth)" in c.detail and "hidden: claude_ai_Slack" in c.detail
    assert "claude.ai/settings/connectors" in c.fix


# --- live --------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_claudeai_gmail_present_and_slack_hidden(tmp_path):
    """A real session with Spark's options: Gmail's tools are offered to the model, Slack's are
    not. Skips when the CLI isn't logged in or the account has no Gmail connector."""
    if shutil.which("claude") is None and not os.environ.get("SPARK_BRAIN__CLI_PATH"):
        pytest.skip("claude CLI not on PATH")
    import claude_agent_sdk as sdk

    settings = Settings(
        config_dir=tmp_path,
        connectors={"mcp_config": "none.json", "ui_file": "none.json"},
    )
    claudeai.save_state(settings, {"claude_ai_Slack": {"name": "claude.ai Slack"}})
    setup = load_mcp_servers(settings, environ={})
    assert "mcp__claude_ai_Slack" in setup.disallowed_tools

    async def run() -> tuple[list[str], dict[str, Any]]:
        options = sdk.ClaudeAgentOptions(
            model=settings.brain.fast_model,
            tools=[],
            setting_sources=[],
            strict_mcp_config=False,
            allowed_tools=setup.allowed_tools,
            disallowed_tools=setup.disallowed_tools,
            env=env_for_sdk(settings, claudeai=bool(setup.claudeai)),
            max_turns=1,
        )
        tools: list[str] = []
        async with sdk.ClaudeSDKClient(options=options) as client:
            await client.query("Reply with just: OK")
            async for msg in client.receive_response():
                if isinstance(msg, sdk.SystemMessage) and msg.subtype == "init":
                    tools = list(msg.data.get("tools") or [])
            status = await client.get_mcp_status()  # after the turn: no longer "pending"
        return tools, status

    tools, status = asyncio.run(run())
    names = {e.get("name"): e.get("status") for e in (status or {}).get("mcpServers", [])}
    if "claude.ai Gmail" not in names:
        pytest.skip(f"no claude.ai Gmail connector on this login (servers: {sorted(names)})")
    if names["claude.ai Gmail"] == "needs-auth":
        pytest.skip(f"claude.ai Gmail is {names['claude.ai Gmail']}: sign in on claude.ai")
    gmail = [t for t in tools if t.startswith("mcp__claude_ai_Gmail__")]
    assert "mcp__claude_ai_Gmail__search_threads" in gmail, json.dumps(tools)
    assert "mcp__claude_ai_Gmail__send_message" not in tools  # destructive: off
    assert not [t for t in tools if t.startswith("mcp__claude_ai_Slack__")]


# --- Connectors window -------------------------------------------------------------------------


def _window(tmp_path: Path, live: dict[str, Any] | None = None):
    from spark.desktop.connectors import ConnectorsApi

    (tmp_path / "mcp_servers.json").write_text("{}", encoding="utf-8")
    settings = _settings(tmp_path, servers=_only_claudeai())
    claudeai.save_state(
        settings,
        claudeai.parse_status(
            [
                _status_entry("claude.ai Gmail", "connected", GMAIL_TOOLS),
                _status_entry("claude.ai Google Calendar", "needs-auth", ["authenticate"]),
                _status_entry("claude.ai Slack", "connected", ["slack_search_public", "slack_x"]),
            ]
        ),
    )
    api = ConnectorsApi(settings, probe_fn=lambda *a, **k: None, live_status=lambda: live or {})
    return settings, api


def test_window_lists_claudeai_group(tmp_path):
    _, api = _window(tmp_path, live={"claude_ai_Gmail": {"status": "connected"}})
    servers = {s["name"]: s for s in api.list()["servers"]}
    assert set(servers) == {"claude_ai_Gmail", "claude_ai_Google_Calendar", "claude_ai_Slack"}
    gmail = servers["claude_ai_Gmail"]
    assert gmail["source"] == "claudeai" and gmail["label"] == "Gmail"
    assert gmail["enabled"] and gmail["status"] == "ok" and gmail["error"] is None
    assert gmail["live"] == {"status": "connected"}
    assert gmail["link"] == "https://claude.ai/settings/connectors"
    modes = {t["name"]: t["mode"] for t in gmail["tools"]}
    assert len(modes) == 30
    assert modes["search_threads"] == "allow" and modes["create_draft"] == "ask"
    assert modes["send_message"] == "off"
    cal = servers["claude_ai_Google_Calendar"]
    assert cal["status"] == "needs-auth" and "sign in on claude.ai" in cal["error"]
    assert {t["name"] for t in cal["tools"]} == set(CALENDAR_TOOLS)  # the configured ones
    slack = servers["claude_ai_Slack"]
    assert not slack["enabled"] and {t["mode"] for t in slack["tools"]} == {"off"}


def test_window_toggles_persist_and_reach_the_policy(tmp_path):
    settings, api = _window(tmp_path)
    r = api.set_enabled("claude_ai_Slack", True)
    assert r["ok"] and r["server"]["enabled"] and api.list()["dirty"]
    assert api.set_tool("claude_ai_Slack", "slack_search_public", "allow")["ok"]
    assert api.set_tool("claude_ai_Gmail", "send_message", "ask")["server"]["tools"]
    assert not api.set_tool("claude_ai_Gmail", "nope", "ask")["ok"]
    assert not api.set_tool("claude_ai_Gmail", None, "maybe")["ok"]
    assert api.set_route("claude_ai_Slack", "auto")["server"]["route"] == "auto"
    assert api.set_enabled("claude_ai_Google_Calendar", False)["ok"]
    entry = cs.load_store(cs.store_path(settings))["claude_ai_Gmail"]
    assert entry["kind"] == "claude_ai" and "config" not in entry
    setup = load_mcp_servers(settings, environ={})
    assert setup.claudeai == ["claude_ai_Gmail", "claude_ai_Slack"]
    assert setup.policy.classify("mcp__claude_ai_Gmail__send_message") == "side_effect"
    assert setup.policy.classify("mcp__claude_ai_Slack__slack_search_public") == "read"
    assert "mcp__claude_ai_Google_Calendar" in setup.disallowed_tools
    assert "claude_ai_Slack" not in setup.deep_route  # route auto
    # every tool at once, and the app group doesn't show claude.ai entries
    assert api.set_tool("claude_ai_Slack", None, "off")["ok"]
    listing = api.list()["servers"]
    assert not [s for s in listing if s["source"] == "app"]
    slack = next(s for s in listing if s["name"] == "claude_ai_Slack")
    assert {t["mode"] for t in slack["tools"]} == {"off"}
    assert not api.remove("claude_ai_Slack")["ok"]
    assert not api.test("claude_ai_Slack")["ok"]
    assert not api.open_link("https://evil.example")["ok"]
