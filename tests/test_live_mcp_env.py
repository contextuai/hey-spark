"""Live proof that ``${VAR}`` placeholders in mcp_servers.json reach a real MCP server expanded.

Spark keeps placeholders unexpanded in the options it hands the SDK (secrets never pass through our
code); the Claude CLI expands them. This runs the real loader + ClaudeAgentBrain + CLI end to end.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from spark.config import Settings

ECHO = Path(__file__).parent / "mcp" / "echo_server.py"


@pytest.mark.live
async def test_live_mcp_env_var_placeholder_is_expanded(tmp_path, monkeypatch, caplog):
    from spark.brain.claude_agent import ClaudeAgentBrain

    servers = {
        "echo": {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(ECHO)],
            "env": {"ECHO_TOKEN": "${SPARK_TEST_ECHO_TOKEN}"},
        }
    }
    (tmp_path / "mcp_servers.json").write_text(json.dumps(servers), encoding="utf-8")
    monkeypatch.setenv("SPARK_TEST_ECHO_TOKEN", "live-test-token-8472")
    settings = Settings(
        config_dir=tmp_path,
        connectors={"servers": {"echo": {"allowed_tools": ["token_seen"]}}},
        logging={"dir": str(tmp_path / "logs")},
    )
    brain = ClaudeAgentBrain(settings)
    caplog.set_level("INFO", logger="spark")
    try:
        await brain.warmup()
        reply = "".join(
            [
                d
                async for d in brain.respond(
                    "Call the echo server's token_seen tool and tell me exactly the word it "
                    "returned, yes or no, nothing else."
                )
            ]
        )
    finally:
        await brain.close()
    print(f"[live] reply={reply!r}")
    assert "tool call: mcp__echo__token_seen" in caplog.text, "the model never called the tool"
    words = set(re.findall(r"[a-z]+", reply.lower()))
    assert "yes" in words and "no" not in words, reply
