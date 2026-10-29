"""A tiny stdio MCP server used by the doctor's connector tests (no network, no secrets)."""

import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


@server.tool()
def token_seen() -> str:
    """Report whether an *expanded* token reached the server (never its value).

    An unexpanded literal like ``${SPARK_TEST_ECHO_TOKEN}`` counts as "no", so this proves
    that the ``${VAR}`` placeholder was expanded on the way to the server.
    """
    value = os.environ.get("ECHO_TOKEN", "")
    return "yes" if value and "${" not in value else "no"


if __name__ == "__main__":
    server.run("stdio")
