"""claude.ai account connectors (Gmail, Google Calendar, ...) for Spark (docs/connectors.md §4.7).

The user connects them once on claude.ai (Settings -> Connectors) or with ``/mcp`` in Claude
Code. The logged-in CLI then injects them into an SDK session itself, so they need no
``mcp_servers.json`` entry. Verified live 2026-10-05 (SDK 0.2.159, docs/sdk_notes.md):

* They load only when ``ENABLE_CLAUDEAI_MCP_SERVERS`` is truthy **and**
  ``strict_mcp_config=False``. With ``strict_mcp_config=True`` they never load.
* ``client.get_mcp_status()`` lists each as ``{"name": "claude.ai Gmail", "status": ...,
  "tools": [{"name": ...}, ...]}``. The model sees ``mcp__claude_ai_Gmail__<tool>``: the server
  key is the display name with ``"claude.ai "`` -> ``"claude_ai_"`` and spaces -> ``"_"``.
* A server-level ``disallowed_tools`` entry (``mcp__claude_ai_Slack``) removes all of that
  server's tools from the model's list.

Spark configures them as ordinary ``connectors.servers`` entries keyed ``claude_ai_<Name>``.
Which claude.ai servers exist is only known at runtime, so the brain saves the last reported
list here (:func:`save_state`), and the next session hides every one that isn't enabled
(:func:`hidden_servers`). A small static list covers the very first session, and the PreToolUse
gate denies any tool that isn't in a policy regardless.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger("spark.brain.mcp")

PREFIX = "claude_ai_"
DISPLAY_PREFIX = "claude.ai "
SETTINGS_URL = "https://claude.ai/settings/connectors"
SIGN_IN_HINT = "sign in on claude.ai (Settings -> Connectors) or run /mcp in Claude Code"
#: Common claude.ai connectors, hidden when not configured even before the CLI has reported
#: the account's real list once (the first session after install).
KNOWN_SERVERS = (
    "claude_ai_Slack",
    "claude_ai_Asana",
    "claude_ai_Atlassian_Rovo",
    "claude_ai_Claude_Docs",
    "claude_ai_Microsoft_365",
    "claude_ai_Google_Drive",
)
_KEY_CLEAN = re.compile(r"[^A-Za-z0-9_-]")


def is_claudeai(name: str) -> bool:
    """A ``connectors.servers`` key (or MCP server key) of a claude.ai account connector."""
    return name.startswith(PREFIX)


def server_key(display_name: str) -> str | None:
    """``"claude.ai Google Calendar"`` -> ``"claude_ai_Google_Calendar"``; None for a server
    that isn't a claude.ai connector. A name that already is a key is returned as is."""
    name = display_name.strip()
    if is_claudeai(name):
        return name
    if not name.startswith(DISPLAY_PREFIX):
        return None
    rest = name[len(DISPLAY_PREFIX) :].strip()
    return PREFIX + _KEY_CLEAN.sub("_", rest) if rest else None


def display_name(key: str) -> str:
    """``"claude_ai_Google_Calendar"`` -> ``"Google Calendar"`` (for people)."""
    return key.removeprefix(PREFIX).replace("_", " ")


def state_path(settings: Settings) -> Path:
    return settings.path(settings.connectors.claudeai_state_file)


def load_state(settings: Settings) -> dict[str, dict[str, Any]]:
    """The claude.ai servers the CLI reported last: key -> ``{"name", "status", "tools"}``.
    A missing or broken file yields {}."""
    path = state_path(settings)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("claude.ai connectors: can't read %s (%s); ignoring it", path, exc)
        return {}
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in servers.items():
        if isinstance(key, str) and is_claudeai(key) and isinstance(entry, dict):
            tools = [t for t in entry.get("tools") or [] if isinstance(t, str) and t]
            out[key] = {
                "name": str(entry.get("name") or display_name(key)),
                "status": str(entry.get("status") or "unknown"),
                "tools": tools,
            }
    return out


def save_state(settings: Settings, servers: Mapping[str, Mapping[str, Any]]) -> bool:
    """Write the reported servers if they changed. Returns True when the file was written."""
    clean = {
        key: {
            "name": str(entry.get("name") or display_name(key)),
            "status": str(entry.get("status") or "unknown"),
            "tools": sorted({str(t) for t in entry.get("tools") or [] if t}),
        }
        for key, entry in sorted(servers.items())
        if is_claudeai(key)
    }
    if clean == load_state(settings):
        return False
    from spark.brain.connector_store import utcnow
    from spark.util.files import write_text_atomic

    data = {
        "_comment": "Managed by Spark: the claude.ai connectors the Claude CLI reported last. "
        "Unconfigured ones are hidden from Spark's model.",
        "updated_at": utcnow(),
        "servers": clean,
    }
    try:
        write_text_atomic(state_path(settings), json.dumps(data, indent=2) + "\n")
    except OSError as exc:
        log.warning("claude.ai connectors: can't save %s (%s)", state_path(settings), exc)
        return False
    return True


def parse_status(entries: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The claude.ai servers in a ``get_mcp_status()`` server list: key -> name/status/tools."""
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = server_key(str(entry.get("name") or ""))
        if key is None:
            continue
        tools = []
        for tool in entry.get("tools") or []:
            name = tool.get("name") if isinstance(tool, Mapping) else tool
            if isinstance(name, str) and name:
                # the CLI may report bare names or full mcp__<server>__<tool> names
                tools.append(name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name)
        out[key] = {
            "name": str(entry.get("name")),
            "status": str(entry.get("status") or "unknown"),
            "tools": tools,
        }
    return out


def hidden_servers(
    settings: Settings, enabled: Iterable[str], configured: Iterable[str] = ()
) -> list[str]:
    """claude.ai server keys to hide (server-level deny rule): every one reported last time,
    commonly present, or configured but off, minus the enabled ones."""
    keep = set(enabled)
    seen = set(load_state(settings)) | set(KNOWN_SERVERS)
    seen |= {k for k in configured if is_claudeai(k)}
    return sorted(k for k in seen if k not in keep)


def status_hint(status: str) -> str:
    """What the user should do about a claude.ai connector in this status ('' when fine)."""
    return SIGN_IN_HINT if status == "needs-auth" else ""
