import json
import logging
import re
from pathlib import Path

import pytest

from spark.brain import mcp_config
from spark.brain.mcp_config import (
    McpConfigError,
    env_for_sdk,
    load_mcp_servers,
    read_mcp_json,
    referenced_vars,
    split_tool_name,
)
from spark.config import ConnectorServerConfig, Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]

SECRET_PAT = "fake-pat-value-do-not-leak-0001"
SECRET_KEY = "fake-composio-key-do-not-leak-0002"


def _settings(tmp_path: Path, servers_json: dict, connectors: dict | None = None) -> Settings:
    (tmp_path / "mcp_servers.json").write_text(json.dumps(servers_json), encoding="utf-8")
    return Settings(config_dir=tmp_path, connectors=connectors or {"servers": {}})


def _srv(**kw) -> dict:
    return ConnectorServerConfig(**kw).model_dump()


def _exe(tmp_path: Path, name: str = "server.exe") -> Path:
    p = tmp_path / name
    p.write_bytes(b"")
    return p


HTTP = {"type": "http", "url": "${SVC_URL}", "headers": {"x-api-key": "${SVC_KEY}"}}
HTTP_ENV = {"SVC_URL": "https://example.test/mcp?user_id=u1", "SVC_KEY": SECRET_KEY}


# --- ${VAR} handling ---------------------------------------------------------------------------


def test_referenced_vars_required_and_optional():
    req, opt = referenced_vars(
        {"a": "${ONE}", "b": ["x ${TWO} y", "${THREE:-dflt}"], "c": {"d": "Bearer ${ONE}"}}
    )
    assert req == {"ONE", "TWO"}
    assert opt == {"THREE"}


def test_missing_var_disables_server_and_never_leaks_values(tmp_path, caplog):
    s = _settings(tmp_path, {"svc": HTTP}, {"servers": {"svc": _srv(allowed_tools=["read_a"])}})
    env = {"SVC_URL": "https://example.test/mcp", "SVC_KEY": ""}  # empty counts as missing
    with caplog.at_level(logging.WARNING, logger="spark.brain.mcp"):
        setup = load_mcp_servers(s, environ=env)
    assert "svc" not in setup.servers
    assert "SVC_KEY" in setup.disabled["svc"]
    assert "SVC_URL" not in setup.disabled["svc"]
    assert any("SVC_KEY" in w for w in setup.warnings)
    assert "SVC_KEY" in caplog.text
    assert "example.test" not in caplog.text
    assert "example.test" not in " ".join(setup.warnings)
    assert setup.allowed_tools == []
    assert setup.policy.classify("mcp__svc__read_a") == "unknown"


def test_placeholders_preserved_and_no_secret_in_sdk_config(tmp_path):
    s = _settings(tmp_path, {"svc": HTTP}, {"servers": {"svc": _srv(allowed_tools=["read_a"])}})
    setup = load_mcp_servers(s, environ=HTTP_ENV)
    assert setup.servers["svc"] == HTTP
    dumped = json.dumps(setup.servers)
    assert SECRET_KEY not in dumped
    assert "example.test" not in dumped
    assert setup.required_env["svc"] == ["SVC_KEY", "SVC_URL"]


def test_optional_var_with_default_does_not_disable(tmp_path):
    entry = {"type": "http", "url": "${SVC_URL:-https://fallback.test/mcp}"}
    s = _settings(tmp_path, {"svc": entry}, {"servers": {"svc": _srv()}})
    setup = load_mcp_servers(s, environ={})
    assert setup.servers["svc"]["url"] == "${SVC_URL:-https://fallback.test/mcp}"


def test_non_http_url_disabled_without_value(tmp_path):
    s = _settings(tmp_path, {"svc": HTTP}, {"servers": {"svc": _srv()}})
    setup = load_mcp_servers(s, environ={"SVC_URL": "not-a-url-secretish", "SVC_KEY": "k"})
    assert "SVC_URL" in setup.disabled["svc"]
    assert "secretish" not in setup.disabled["svc"]


def test_stdio_env_placeholders_preserved(tmp_path):
    exe = _exe(tmp_path)
    entry = {"command": "${BIN}", "args": ["stdio"], "env": {"TOKEN": "${PAT}"}}
    s = _settings(tmp_path, {"gh": entry}, {"servers": {"gh": _srv()}})
    setup = load_mcp_servers(s, environ={"BIN": str(exe), "PAT": SECRET_PAT})
    cfg = setup.servers["gh"]
    assert cfg == {
        "type": "stdio",
        "command": str(exe),
        "args": ["stdio"],
        "env": {"TOKEN": "${PAT}"},
    }
    assert SECRET_PAT not in json.dumps(setup.servers)


# --- enabled / membership ----------------------------------------------------------------------


def test_enabled_false_skips_quietly(tmp_path):
    s = _settings(
        tmp_path,
        {"svc": HTTP},
        {"servers": {"svc": _srv(enabled=False, allowed_tools=["read_a"])}},
    )
    setup = load_mcp_servers(s, environ=HTTP_ENV)
    assert setup.servers == {}
    assert "enabled: false" in setup.disabled["svc"]
    assert setup.warnings == []
    assert setup.allowed_tools == []


def test_config_server_missing_from_json_warns(tmp_path):
    s = _settings(tmp_path, {}, {"servers": {"ghost": _srv(allowed_tools=["x"])}})
    setup = load_mcp_servers(s, environ={})
    assert "no entry" in setup.disabled["ghost"]
    assert setup.warnings


def test_json_server_without_config_entry_not_loaded(tmp_path):
    s = _settings(tmp_path, {"svc": HTTP}, {"servers": {}})
    setup = load_mcp_servers(s, environ=HTTP_ENV)
    assert setup.servers == {}
    assert "connectors.servers.svc" in setup.disabled["svc"]


def test_comment_keys_and_mcpservers_wrapper_ignored(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(
        json.dumps({"mcpServers": {"_x": {}, "svc": {**HTTP, "_comment": "hi"}}}),
        encoding="utf-8",
    )
    assert read_mcp_json(p) == {"svc": {**HTTP, "_comment": "hi"}}
    s = _settings(
        tmp_path,
        {"_comment": ["x"], "svc": {**HTTP, "_comment": "c"}},
        {"servers": {"svc": _srv()}},
    )
    setup = load_mcp_servers(s, environ=HTTP_ENV)
    assert setup.servers["svc"] == HTTP
    assert setup.warnings == []


def test_missing_file_and_bad_json(tmp_path):
    s = Settings(config_dir=tmp_path, connectors={"servers": {}})
    setup = load_mcp_servers(s, environ={})
    assert setup.servers == {} and setup.warnings
    (tmp_path / "mcp_servers.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(McpConfigError):
        load_mcp_servers(s, environ={})


def test_unsupported_type_disabled(tmp_path):
    s = _settings(tmp_path, {"svc": {"type": "sdk", "name": "x"}}, {"servers": {"svc": _srv()}})
    setup = load_mcp_servers(s, environ={})
    assert "unsupported type" in setup.disabled["svc"]


# --- command resolution ------------------------------------------------------------------------


def test_npx_wrapped_with_cmd_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_config, "_is_windows", lambda: True)
    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: r"C:\Program Files\nodejs\npx.CMD")
    entry = {"command": "npx", "args": ["-y", "@scope/server", "${ROOT_DIR}"]}
    s = _settings(tmp_path, {"fs": entry}, {"servers": {"fs": _srv()}})
    setup = load_mcp_servers(s, environ={"ROOT_DIR": "C:/data"})
    assert setup.servers["fs"] == {
        "type": "stdio",
        "command": "cmd",
        "args": ["/c", "npx", "-y", "@scope/server", "${ROOT_DIR}"],
    }


def test_npx_not_wrapped_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_config, "_is_windows", lambda: False)
    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: "/usr/bin/npx")
    s = _settings(
        tmp_path, {"fs": {"command": "npx", "args": ["-y", "pkg"]}}, {"servers": {"fs": _srv()}}
    )
    setup = load_mcp_servers(s, environ={})
    assert setup.servers["fs"]["command"] == "/usr/bin/npx"
    assert setup.servers["fs"]["args"] == ["-y", "pkg"]


def test_npx_missing_from_path_disabled_with_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_config.shutil, "which", lambda name: None)
    s = _settings(tmp_path, {"fs": {"command": "npx"}}, {"servers": {"fs": _srv()}})
    setup = load_mcp_servers(s, environ={})
    assert "not found on PATH" in setup.disabled["fs"]
    assert "Node.js" in setup.disabled["fs"]


def test_missing_binary_from_var_disabled_with_fix(tmp_path):
    entry = {"command": "${SPARK_GITHUB_MCP_BIN}", "args": ["stdio"]}
    s = _settings(tmp_path, {"github": entry}, {"servers": {"github": _srv()}})
    setup = load_mcp_servers(s, environ={"SPARK_GITHUB_MCP_BIN": "bin/nope.exe"})
    reason = setup.disabled["github"]
    assert "executable not found" in reason
    assert "SPARK_GITHUB_MCP_BIN" in reason and "fix:" in reason


def test_relative_binary_resolves_against_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_config, "_is_windows", lambda: True)
    (tmp_path / "bin").mkdir()
    exe = _exe(tmp_path / "bin", "github-mcp-server.exe")
    entry = {"command": "${BIN}", "args": ["stdio"]}
    s = _settings(tmp_path, {"github": entry}, {"servers": {"github": _srv()}})
    # No suffix given: .exe is tried on Windows.
    setup = load_mcp_servers(s, environ={"BIN": "bin/github-mcp-server"})
    assert Path(setup.servers["github"]["command"]) == exe


# --- allowlists and classification -------------------------------------------------------------


def _policy_setup(tmp_path, cfg: dict, known=None, env=HTTP_ENV):
    s = _settings(tmp_path, {"svc": HTTP}, {"servers": {"svc": cfg}})
    return load_mcp_servers(s, environ=env, known_tools=known)


def test_allowlist_exact_names_and_glob_expansion(tmp_path):
    cfg = _srv(allowed_tools=["GMAIL_FETCH_*", "mcp__svc__LIST_X", "mcp__other__Y", "NOPE_*"])
    setup = _policy_setup(
        tmp_path,
        cfg,
        known={"svc": ["GMAIL_FETCH_EMAILS", "mcp__svc__GMAIL_FETCH_THREAD", "GMAIL_SEND_EMAIL"]},
    )
    assert setup.allowed_tools == [
        "mcp__svc__GMAIL_FETCH_EMAILS",
        "mcp__svc__GMAIL_FETCH_THREAD",
        "mcp__svc__LIST_X",
    ]
    assert all("*" not in t for t in setup.allowed_tools)
    assert any("NOPE_*" in w for w in setup.warnings)
    assert any("mcp__other__Y" in w for w in setup.warnings)


def test_server_wide_wildcard_rejected_by_config():
    for bad in ["*", "mcp__svc__*", "mcp__svc", "**"]:
        with pytest.raises(ValueError):
            ConnectorServerConfig(allowed_tools=[bad])


def test_side_effecting_and_destructive_classification(tmp_path):
    cfg = _srv(
        allowed_tools=["READ_A", "WRITE_B"],
        side_effecting=["WRITE_B", "CREATE_*"],
        destructive=["DELETE_*", "SEND"],
    )
    setup = _policy_setup(tmp_path, cfg, known={"svc": ["CREATE_EVENT", "DELETE_MSG"]})
    p = setup.policy
    assert p.classify("mcp__svc__READ_A") == "read"
    assert p.classify("mcp__svc__WRITE_B") == "side_effect"
    assert p.classify("mcp__svc__CREATE_DRAFT") == "side_effect"  # glob, not in known list
    assert p.classify("mcp__svc__DELETE_ANYTHING") == "destructive"
    assert p.classify("mcp__svc__SEND") == "destructive"
    # side-effecting tools are never pre-approved
    assert setup.allowed_tools == ["mcp__svc__READ_A"]
    assert setup.side_effecting == {"mcp__svc__WRITE_B", "mcp__svc__CREATE_EVENT"}
    assert p.is_allowed("mcp__svc__READ_A") and not p.is_allowed("mcp__svc__WRITE_B")
    assert p.is_side_effecting("mcp__svc__WRITE_B") and p.needs_confirmation("mcp__svc__WRITE_B")
    assert p.is_destructive("mcp__svc__SEND") and p.is_blocked("mcp__svc__SEND")


def test_destructive_blocked_unless_allowed(tmp_path):
    blocked = _policy_setup(tmp_path, _srv(destructive=["DELETE_*", "SEND"]))
    assert blocked.disallowed_tools == ["mcp__svc__DELETE_*", "mcp__svc__SEND"]
    assert blocked.policy.is_blocked("mcp__svc__SEND")
    assert not blocked.policy.needs_confirmation("mcp__svc__SEND")

    allowed = _policy_setup(tmp_path, _srv(destructive=["SEND"], allow_destructive=True))
    assert allowed.disallowed_tools == []
    assert allowed.policy.classify("mcp__svc__SEND") == "destructive"
    assert not allowed.policy.is_blocked("mcp__svc__SEND")
    assert allowed.policy.needs_confirmation("mcp__svc__SEND")
    assert "mcp__svc__SEND" in allowed.side_effecting
    assert "mcp__svc__SEND" not in allowed.allowed_tools
    assert any("allow_destructive" in w for w in allowed.warnings)


def test_unknown_tools_classified_unknown(tmp_path):
    setup = _policy_setup(tmp_path, _srv(allowed_tools=["READ_A"]))
    p = setup.policy
    for name in ["mcp__svc__SOMETHING_ELSE", "mcp__ghost__READ_A", "Bash", "mcp__svc", "mcp__"]:
        assert p.classify(name) == "unknown", name
        assert p.is_blocked(name)
        assert not p.is_allowed(name)


def test_split_tool_name():
    assert split_tool_name("mcp__spark__current_time") == ("spark", "current_time")
    assert split_tool_name("mcp__srv__a__b") == ("srv", "a__b")
    assert split_tool_name("Read") is None


# --- in-process servers ------------------------------------------------------------------------


def test_add_sdk_server_merges(tmp_path):
    setup = _policy_setup(tmp_path, _srv(allowed_tools=["READ_A"]))
    sdk = {"type": "sdk", "name": "spark", "instance": object()}
    setup.add_sdk_server(
        "spark",
        sdk,
        read_only=["current_time", "mcp__spark__list_timers"],
        side_effecting=["start_project"],
    )
    assert setup.servers["spark"] is sdk
    assert "mcp__spark__current_time" in setup.allowed_tools
    assert "mcp__spark__list_timers" in setup.allowed_tools
    assert "mcp__svc__READ_A" in setup.allowed_tools
    assert "mcp__spark__start_project" in setup.side_effecting
    assert setup.policy.classify("mcp__spark__start_project") == "side_effect"
    assert setup.policy.classify("mcp__spark__current_time") == "read"
    with pytest.raises(ValueError):
        setup.add_sdk_server("spark", sdk)
    with pytest.raises(ValueError):
        setup.add_sdk_server("other", sdk, read_only=["x_*"])
    with pytest.raises(ValueError):
        setup.add_sdk_server("bad__name", sdk)


# --- env ---------------------------------------------------------------------------------------


def test_env_for_sdk(tmp_path):
    off = {"ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
    on = {"ENABLE_CLAUDEAI_MCP_SERVERS": "true"}
    # defaults: claude_ai_Gmail and claude_ai_Google_Calendar ship disabled -> don't load them
    assert env_for_sdk(Settings(config_dir=tmp_path)) == off
    gmail = Settings().connectors.servers["claude_ai_Gmail"].model_dump() | {"enabled": True}
    enabled = Settings(config_dir=tmp_path, connectors={"servers": {"claude_ai_Gmail": gmail}})
    assert env_for_sdk(enabled) == on
    assert env_for_sdk(enabled, claudeai=False) == off
    blocked = Settings(config_dir=tmp_path, connectors={"block_claudeai_connectors": True})
    assert env_for_sdk(blocked) == off
    assert env_for_sdk(blocked, claudeai=True) == off  # the block always wins
    none = Settings(config_dir=tmp_path, connectors={"servers": {}})
    assert env_for_sdk(none) == off
    assert env_for_sdk(none, claudeai=True) == on


# --- committed files ---------------------------------------------------------------------------

_TOKEN_SHAPES = [
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"sk-[A-Za-z0-9_-]{16,}",
    r"ak_[A-Za-z0-9_-]{16,}",
    r"xox[abpr]-[A-Za-z0-9-]{10,}",
    r"eyJ[A-Za-z0-9_-]{20,}",
    r"\b[A-Fa-f0-9]{32,}\b",
]


@pytest.mark.parametrize("name", ["mcp_servers.json", "mcp_servers.example.json"])
def test_committed_mcp_json_has_no_secrets(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    json.loads(text)
    for pattern in _TOKEN_SHAPES:
        assert not re.search(pattern, text), pattern
    for server in read_mcp_json(ROOT / name).values():
        for key in ("env", "headers"):
            for value in server.get(key, {}).values():
                assert "${" in value, f"{name}: literal {key} value"


def test_repo_config_and_json_load(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("SPARK_"):
            monkeypatch.delenv(key)
    s = load_settings(ROOT / "config.yaml", load_env_file=False)
    s.connectors.ui_file = "no-such-connectors.json"  # the user's own Connectors-window servers
    assert set(s.connectors.servers) == {"github", "claude_ai_Gmail", "claude_ai_Google_Calendar"}
    assert s.connectors.servers["github"].allow_destructive is False
    # The one sample that needs the user's own credentials ships off (safe public default).
    assert s.connectors.servers["github"].enabled is False
    assert set(read_mcp_json(ROOT / "mcp_servers.json")) == {"github"}
    assert load_mcp_servers(s, environ={}).servers == {}
    # A user who sets the token turns it on in config.yaml:
    s.connectors.servers["github"].enabled = True
    setup = load_mcp_servers(s, environ={"SPARK_GITHUB_PAT": SECRET_PAT})
    assert set(setup.servers) == {"github"}
    assert setup.servers["github"]["url"] == "https://api.githubcopilot.com/mcp/"  # not /readonly
    assert set(setup.disabled) == {"claude_ai_Gmail", "claude_ai_Google_Calendar"}
    assert "mcp__github__list_pull_requests" in setup.allowed_tools
    assert "mcp__github__merge_pull_request" in setup.disallowed_tools
    assert setup.policy.classify("mcp__github__create_repository") == "side_effect"
    assert setup.policy.classify("mcp__github__issue_write") == "side_effect"
    assert not any(t.endswith("__*") or "__" not in t[5:] for t in setup.allowed_tools)
    assert SECRET_PAT not in json.dumps(setup.servers)

    empty = load_mcp_servers(s, environ={})
    assert empty.servers == {}
    assert "SPARK_GITHUB_PAT" in empty.disabled["github"]
