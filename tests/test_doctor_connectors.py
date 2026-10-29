"""Doctor connector checks (P2.5): env vars, spawn + list tools, http reachability, timeouts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from spark import doctor
from spark.config import Settings

ECHO = Path(__file__).parent / "mcp" / "echo_server.py"
ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path, servers: dict, policy: dict) -> Settings:
    (tmp_path / "mcp_servers.json").write_text(json.dumps(servers), encoding="utf-8")
    return Settings(config_dir=tmp_path, connectors={"servers": policy})


def _echo_server(env: dict | None = None) -> dict:
    return {"type": "stdio", "command": sys.executable, "args": [str(ECHO)], "env": env or {}}


async def test_stdio_server_spawns_and_lists_tools(tmp_path):
    s = _settings(
        tmp_path,
        {"echo": _echo_server({"ECHO_TOKEN": "${ECHO_TOKEN}"})},
        {"echo": {"allowed_tools": ["echo"]}},
    )
    checks = await doctor.check_connectors(s, environ={"ECHO_TOKEN": "s3cr3t-value-123"})
    [c] = checks
    assert c.ok, c.detail
    assert c.name == "connector: echo"
    assert "2 tools" in c.detail


async def test_probe_passes_expanded_env_to_the_server(tmp_path):
    tools = await doctor.probe_mcp_server(
        _echo_server({"ECHO_TOKEN": "${ECHO_TOKEN}"}), {"ECHO_TOKEN": "abc123456"}
    )
    assert sorted(tools) == ["echo", "token_seen"]


async def test_missing_env_var_is_a_warning_with_fix(tmp_path):
    s = _settings(
        tmp_path,
        {"echo": _echo_server({"ECHO_TOKEN": "${ECHO_TOKEN}"})},
        {"echo": {"allowed_tools": ["echo"]}},
    )
    [c] = await doctor.check_connectors(s, environ={})
    assert not c.ok and not c.required
    assert "ECHO_TOKEN" in c.detail
    assert ".env" in c.fix


async def test_disabled_in_config_is_informational(tmp_path):
    s = _settings(tmp_path, {"echo": _echo_server()}, {"echo": {"enabled": False}})
    [c] = await doctor.check_connectors(s, environ={})
    assert c.ok and "disabled in config.yaml" in c.detail


async def test_unreachable_http_server_fails_softly_without_leaking_key(tmp_path):
    secret = "zz-very-secret-key-987654"
    s = _settings(
        tmp_path,
        {
            "web": {
                "type": "http",
                "url": "http://127.0.0.1:9/mcp",
                "headers": {"x-api-key": "${WEB_KEY}"},
            }
        },
        {"web": {"allowed_tools": ["search"]}},
    )
    [c] = await doctor.check_connectors(s, environ={"WEB_KEY": secret})
    assert not c.ok and not c.required
    assert secret not in c.detail and secret not in c.fix


async def test_probe_timeout_reported(tmp_path, monkeypatch):
    async def hang(cfg, environ):
        await asyncio.sleep(10)

    monkeypatch.setattr(doctor, "probe_mcp_server", hang)
    monkeypatch.setattr(doctor, "MCP_PROBE_TIMEOUT_S", 0.05)
    s = _settings(tmp_path, {"echo": _echo_server()}, {"echo": {}})
    [c] = await doctor.check_connectors(s, environ={})
    assert not c.ok and "no answer" in c.detail


async def test_bad_json_is_reported(tmp_path):
    (tmp_path / "mcp_servers.json").write_text("{not json", encoding="utf-8")
    checks = await doctor.check_connectors(Settings(config_dir=tmp_path), environ={})
    assert not checks[0].ok


async def test_repo_config_without_credentials_only_warns():
    checks = await doctor.check_connectors(Settings(config_dir=ROOT), environ={})
    assert checks
    assert all(c.ok or not c.required for c in checks)


def test_connectors_in_default_checks():
    assert doctor.check_connectors in doctor.default_checks()


@pytest.mark.parametrize("kind", ["websocket"])
async def test_unsupported_type(kind):
    with pytest.raises(ValueError):
        await doctor.probe_mcp_server({"type": kind, "url": "x"}, {})
