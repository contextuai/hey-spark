"""Connectors added in the desktop window: parsing, secrets to .env, policy, test, window API."""

import json
import sys
from pathlib import Path

import pytest

from spark.brain import connector_store as cs
from spark.brain.connector_store import ConnectorError, SecretSink
from spark.brain.mcp_config import load_mcp_servers
from spark.brain.mcp_probe import ProbeResult, probe_async, tool_summary
from spark.config import Settings
from spark.desktop.connectors import ConnectorsApi

ECHO = Path(__file__).parent / "mcp" / "echo_server.py"
TOKEN = "fake-0123456789abcdef0123456789abcdef"

DESKTOP = {
    "mcpServers": {
        "CATALOG_DOCS": {
            "command": "node",
            "args": ["C:/x/tools/mcp-bridge.js", "http://10.0.0.5/catalog-docs/mcp", TOKEN],
        },
        "WIKI": {
            "command": "node",
            "args": ["C:/x/tools/mcp-bridge.js", "http://10.0.0.5/wiki/mcp", TOKEN],
        },
    },
    "preferences": {"sidebarMode": "chat"},
}


def _settings(tmp_path: Path, servers_json: dict | None = None, policy: dict | None = None):
    (tmp_path / "mcp_servers.json").write_text(json.dumps(servers_json or {}), encoding="utf-8")
    return Settings(config_dir=tmp_path, connectors={"servers": policy or {}})


def _sink(existing: dict | None = None) -> SecretSink:
    return SecretSink(existing or {}, environ={})


# --- parsing ---------------------------------------------------------------------------------


def test_claude_desktop_bridge_becomes_direct_http_with_token_in_env():
    sink = _sink()
    parsed = cs.parse_pasted(json.dumps(DESKTOP), sink)
    assert [p.name for p in parsed] == ["CATALOG_DOCS", "WIKI"]
    catalog = parsed[0].config
    assert catalog == {
        "type": "http",
        "url": "http://10.0.0.5/catalog-docs/mcp",
        "headers": {"Authorization": "Bearer ${SPARK_MCP_CATALOG_DOCS_TOKEN}"},
    }
    # one shared token -> one variable for both servers
    assert parsed[1].config["headers"]["Authorization"] == "Bearer ${SPARK_MCP_CATALOG_DOCS_TOKEN}"
    assert sink.new == {"SPARK_MCP_CATALOG_DOCS_TOKEN": TOKEN}
    assert TOKEN not in json.dumps([p.config for p in parsed])
    assert any("bridge" in n for n in parsed[0].notes)


def test_existing_env_value_is_reused():
    sink = _sink({"MY_TOKEN": TOKEN})
    parsed = cs.parse_pasted(json.dumps(DESKTOP), sink)
    assert parsed[0].config["headers"]["Authorization"] == "Bearer ${MY_TOKEN}"
    assert sink.new == {}


def test_mcp_remote_wrapper_unwrapped():
    text = json.dumps(
        {
            "notion": {
                "command": "npx",
                "args": [
                    *("-y", "mcp-remote", "https://mcp.notion.test/sse"),
                    *("--header", "X-Key: abc123"),
                ],
            }
        }
    )
    [p] = cs.parse_pasted(text, _sink())
    assert p.config["type"] == "sse"
    assert p.config["url"] == "https://mcp.notion.test/sse"
    assert p.config["headers"] == {"X-Key": "${SPARK_MCP_NOTION_X_KEY}"}


def test_single_unnamed_server_and_url_variants():
    [p] = cs.parse_pasted('{"serverUrl": "https://api.example.test/github/mcp"}', _sink())
    assert p.name == "github"
    assert p.config == {"type": "http", "url": "https://api.example.test/github/mcp"}
    [p] = cs.parse_pasted('"files": {"url": "https://h.test/sse"}', _sink())  # bare fragment
    assert (p.name, p.config["type"]) == ("files", "sse")


def test_stdio_env_only_secretish_keys_move():
    text = json.dumps(
        {"s": {"command": "uvx", "args": ["srv"], "env": {"API_KEY": "k-1", "LOG_LEVEL": "info"}}}
    )
    sink = _sink()
    [p] = cs.parse_pasted(text, sink)
    assert p.config["env"] == {"API_KEY": "${SPARK_MCP_S_API_KEY}", "LOG_LEVEL": "info"}
    assert sink.new == {"SPARK_MCP_S_API_KEY": "k-1"}


def test_url_with_key_in_query_moves_whole_url():
    sink = _sink()
    [p] = cs.parse_pasted('{"c": {"url": "https://x.test/mcp?api_key=zzz"}}', sink)
    assert p.config["url"] == "${SPARK_MCP_C_URL}"
    assert sink.new["SPARK_MCP_C_URL"].endswith("api_key=zzz")


def test_placeholders_are_kept_and_names_cleaned():
    text = json.dumps({"my server!": {"url": "https://x.test/mcp", "headers": {"A": "${KEEP}"}}})
    [p] = cs.parse_pasted(text, _sink())
    assert p.name == "my_server"
    assert p.config["headers"] == {"A": "${KEEP}"}


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[1, 2]", '{"a": 1}', '{"x": {"type": "ws", "url": "ws://x"}}'],
)
def test_bad_input_gives_plain_error(text):
    with pytest.raises(ConnectorError):
        cs.parse_pasted(text, _sink())


# --- store and policy ------------------------------------------------------------------------


def test_add_writes_store_and_env_and_sets_process_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARK_MCP_CATALOG_DOCS_TOKEN", raising=False)
    s = _settings(tmp_path)
    r = cs.add_servers(s, json.dumps(DESKTOP))
    assert r.added == ["CATALOG_DOCS", "WIKI"]
    assert r.env_vars == ["SPARK_MCP_CATALOG_DOCS_TOKEN"]
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert f"SPARK_MCP_CATALOG_DOCS_TOKEN={TOKEN}" in env_text
    store_text = (tmp_path / "connectors.json").read_text(encoding="utf-8")
    assert TOKEN not in store_text
    entry = cs.load_store(tmp_path / "connectors.json")["WIKI"]
    assert entry["enabled"] is False and entry["tools"] == []
    import os

    assert os.environ["SPARK_MCP_CATALOG_DOCS_TOKEN"] == TOKEN
    # a second paste of the same servers is skipped, and no second variable appears
    r2 = cs.add_servers(s, json.dumps(DESKTOP))
    assert r2.added == [] and set(r2.skipped) == {"CATALOG_DOCS", "WIKI"}
    assert (tmp_path / ".env").read_text(encoding="utf-8") == env_text


def test_reserved_names_are_refused(tmp_path):
    s = _settings(tmp_path)
    r = cs.add_servers(s, json.dumps(DESKTOP), reserved={"WIKI"})
    assert r.added == ["CATALOG_DOCS"]
    assert "config.yaml" in r.skipped["WIKI"]


def test_merge_tools_keeps_choices_and_defaults_by_hint():
    previous = [{"name": "a", "mode": "off"}, {"name": "gone", "mode": "allow"}]
    found = [
        {"name": "a", "hint": "read"},
        {"name": "r", "hint": "read"},
        {"name": "d", "hint": "destructive"},
        {"name": "x", "hint": None},
    ]
    modes = {t["name"]: t["mode"] for t in cs.merge_tools(previous, found)}
    assert modes == {"a": "off", "r": "allow", "d": "off", "x": "ask"}


def test_enabled_app_server_reaches_the_brain_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_MCP_CATALOG_DOCS_TOKEN", TOKEN)
    s = _settings(tmp_path)
    cs.add_servers(s, json.dumps(DESKTOP))
    tools = [{"name": n, "hint": None} for n in ("search", "update", "delete")]
    cs.record_test(s, "CATALOG_DOCS", tools, None)
    cs.set_tool_mode(s, "CATALOG_DOCS", "search", "allow")
    cs.set_tool_mode(s, "CATALOG_DOCS", "delete", "off")
    cs.update(s, "CATALOG_DOCS", enabled=True)

    setup = load_mcp_servers(s)
    assert set(setup.servers) == {"CATALOG_DOCS"}  # WIKI is still off
    assert setup.servers["CATALOG_DOCS"]["headers"] == {
        "Authorization": "Bearer ${SPARK_MCP_CATALOG_DOCS_TOKEN}"
    }
    assert setup.allowed_tools == ["mcp__CATALOG_DOCS__search"]
    assert "mcp__CATALOG_DOCS__delete" in setup.disallowed_tools
    assert setup.policy.needs_confirmation("mcp__CATALOG_DOCS__update")
    assert setup.policy.is_blocked("mcp__CATALOG_DOCS__brand_new_tool")
    assert TOKEN not in json.dumps(setup.servers)


def test_config_yaml_server_wins_over_app_server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_MCP_CATALOG_DOCS_TOKEN", TOKEN)
    s = _settings(
        tmp_path,
        {"WIKI": {"type": "http", "url": "https://z.test/mcp"}},
        {"WIKI": {"allowed_tools": ["q"]}},
    )
    # written directly, as if the name had been free when it was added
    cs.save_store(
        tmp_path / "connectors.json",
        {"WIKI": {"config": {"type": "http", "url": "https://other.test"}, "enabled": True}},
    )
    setup = load_mcp_servers(s)
    assert setup.servers["WIKI"]["url"] == "https://z.test/mcp"
    assert any("skipped" in w for w in setup.warnings)


# --- probe -----------------------------------------------------------------------------------


def _echo(env: dict | None = None) -> dict:
    return {"type": "stdio", "command": sys.executable, "args": [str(ECHO)], "env": env or {}}


async def test_probe_lists_tools_of_a_stdio_server(tmp_path):
    r = await probe_async(
        _echo({"ECHO_TOKEN": "${T}"}), base_dir=tmp_path, environ={"T": "v", "PATH": ""}
    )
    assert r.ok, r.error
    assert {t["name"] for t in r.tools} == {"echo", "token_seen"}
    assert r.server_name == "echo"


async def test_probe_missing_variable_and_bad_command(tmp_path):
    r = await probe_async(_echo({"ECHO_TOKEN": "${NOPE}"}), base_dir=tmp_path, environ={})
    assert not r.ok and "NOPE" in r.error
    r = await probe_async({"command": "definitely-not-a-cmd-xyz"}, base_dir=tmp_path, environ={})
    assert not r.ok and "not found" in r.error


async def test_probe_unreachable_http_fails_fast(tmp_path):
    r = await probe_async(
        {"type": "http", "url": "http://127.0.0.1:9/mcp"}, base_dir=tmp_path, timeout_s=10
    )
    assert not r.ok and r.error


def test_tool_summary_reads_annotations():
    class Ann:
        read_only_hint = None
        destructive_hint = True
        title = "Delete it"

    class T:
        name = "rm"
        description = "  Removes\n a thing  "
        annotations = Ann()
        title = None

    assert tool_summary(T()) == {
        "name": "rm",
        "title": "Delete it",
        "description": "Removes a thing",
        "hint": "destructive",
    }


# --- window API ------------------------------------------------------------------------------


def _fake_probe(config, *, base_dir):
    if "wiki" in config.get("url", ""):
        return ProbeResult(False, error="The server refused the credentials (HTTP 401).")
    return ProbeResult(True, tools=[{"name": "yd_search", "description": "", "hint": "read"}])


def test_window_api_flow(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARK_MCP_CATALOG_DOCS_TOKEN", raising=False)
    applied = []
    api = ConnectorsApi(
        _settings(tmp_path), on_apply=lambda: applied.append(1), probe_fn=_fake_probe
    )
    added = api.add(json.dumps(DESKTOP))
    assert added["ok"] and added["added"] == ["CATALOG_DOCS", "WIKI"]

    ok = api.test("CATALOG_DOCS")
    assert ok["ok"] and ok["server"]["status"] == "ok"
    assert ok["server"]["tools"] == [
        {"name": "yd_search", "title": None, "description": "", "hint": "read", "mode": "allow"}
    ]
    bad = api.test("WIKI")
    assert not bad["ok"] and "401" in bad["server"]["error"]

    assert api.set_enabled("CATALOG_DOCS", True)["server"]["enabled"] is True
    assert api.set_tool("CATALOG_DOCS", "yd_search", "ask")["server"]["tools"][0]["mode"] == "ask"
    assert api.set_tool("CATALOG_DOCS", None, "off")["ok"]
    assert not api.set_tool("CATALOG_DOCS", "nope", "off")["ok"]

    listing = api.list()
    assert listing["dirty"] is True
    assert [s["name"] for s in listing["servers"]] == ["CATALOG_DOCS", "WIKI"]
    assert TOKEN not in json.dumps(listing)

    config = api.get_config("CATALOG_DOCS")
    assert TOKEN not in config["text"] and "${SPARK_MCP_CATALOG_DOCS_TOKEN}" in config["text"]

    assert api.apply() == {"ok": True} and applied == [1]
    assert api.list()["dirty"] is False
    assert api.remove("WIKI")["ok"]
    assert [s["name"] for s in api.list()["servers"]] == ["CATALOG_DOCS"]


def test_window_api_edit_replaces_config_and_keeps_modes(tmp_path):
    api = ConnectorsApi(_settings(tmp_path), probe_fn=_fake_probe)
    api.add('{"docs": {"url": "https://a.test/mcp"}}')
    api.test("docs")
    api.set_tool("docs", "yd_search", "off")
    r = api.update_config("docs", '{"docs": {"url": "https://b.test/mcp"}}')
    assert r["ok"], r
    entry = cs.load_store(tmp_path / "connectors.json")["docs"]
    assert entry["config"]["url"] == "https://b.test/mcp"
    assert entry["tools"][0]["mode"] == "off"
    assert not api.update_config("docs", "{")["ok"]


def test_window_api_lists_config_file_servers_read_only(tmp_path):
    api = ConnectorsApi(
        _settings(
            tmp_path,
            {"github": {"type": "http", "url": "https://g.test/mcp", "headers": {"A": "${GH}"}}},
            {"github": {"allowed_tools": ["get_me"], "side_effecting": ["issue_write"]}},
        ),
        probe_fn=_fake_probe,
    )
    [gh] = api.list()["servers"]
    assert gh["source"] == "config" and gh["enabled"] is False
    assert "GH" in gh["error"]
    assert {t["name"]: t["mode"] for t in gh["tools"]} == {"get_me": "allow", "issue_write": "ask"}
    assert not api.add('{"github": {"url": "https://x.test"}}')["added"]


# --- P5.7j: route, live status, notes in the window ---------------------------------------------


def _api_with_warehouse(tmp_path, live=None):
    from spark.brain.connector_store import save_store, store_path

    settings = _settings(tmp_path)
    save_store(
        store_path(settings),
        {
            "WAREHOUSE": {
                "config": {"type": "http", "url": "http://warehouse.example/mcp"},
                "enabled": True,
                "tools": [{"name": "warehouse_query", "mode": "allow"}],
                "tested_at": "2026-10-05T10:00:00+00:00",
            }
        },
    )
    return ConnectorsApi(settings, probe_fn=_fake_probe, live_status=lambda: live or {})


def test_route_defaults_to_deep_and_can_be_switched(tmp_path):
    api = _api_with_warehouse(tmp_path)
    (server,) = [s for s in api.list()["servers"] if s["name"] == "WAREHOUSE"]
    assert server["route"] == "deep"
    res = api.set_route("WAREHOUSE", "auto")
    assert res["ok"] and res["server"]["route"] == "auto"
    assert api.list()["dirty"] is True  # the brain must reload to route differently
    assert api.set_route("WAREHOUSE", "warp")["ok"] is False
    assert api.set_route("NOPE", "deep")["ok"] is False


def test_live_status_and_notes_in_the_view(tmp_path):
    live = {"WAREHOUSE": {"status": "failed", "error": "timed out"}}
    api = _api_with_warehouse(tmp_path, live)
    (server,) = [s for s in api.list()["servers"] if s["name"] == "WAREHOUSE"]
    assert server["live"] == {"status": "failed", "error": "timed out"} and server["notes"] == 0
    got = api.get_notes("WAREHOUSE")
    assert got["ok"] and got["text"] == "" and got["path"].endswith("WAREHOUSE.md")
    saved = api.save_notes("WAREHOUSE", "# WAREHOUSE notes\n\n- Orders are in dbo.Orders\n")
    assert saved["ok"] and saved["notes"] == 1
    assert api.get_notes("WAREHOUSE")["text"].endswith("- Orders are in dbo.Orders")
    assert api.save_notes("WAREHOUSE", "  ")["notes"] == 0  # cleared: the file is removed
    assert not Path(got["path"]).exists()
