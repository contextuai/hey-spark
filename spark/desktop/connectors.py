"""The Connectors window's API: list, add (paste or import from Claude Desktop), test, and set
each server and tool on or off.

pywebview calls these methods from its own worker threads, so each one is plain blocking code
that returns JSON-able dicts. A connection test opens its own event loop (:func:`probe`). The
store does its own locking (spark/brain/connector_store.py).

Two kinds of servers are listed:

* ``app``: added here and kept in ``connectors.json``; everything is editable.
* ``config``: set up by hand in ``config.yaml`` + ``mcp_servers.json``. They're shown (with why
  they're off, if they are) and can be tested, but are edited in those files.
* ``claudeai``: the user's claude.ai account connectors (:mod:`spark.brain.claudeai`): every one
  the CLI reported last, plus the ``claude_ai_*`` servers in ``config.yaml``. The on/off switch,
  route and per-tool modes are editable; the choices go to ``connectors.json`` and override the
  ``config.yaml`` defaults. They can't be tested or removed here (the CLI connects them; the
  user signs in or disconnects on claude.ai).

Changes reach the voice brain when it is rebuilt: :meth:`ConnectorsApi.apply` asks the
controller to do that (a restart if it is running).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from spark.brain import claudeai
from spark.brain import connector_store as store
from spark.brain.connector_store import ConnectorError
from spark.brain.mcp_config import McpConfigError, load_mcp_servers, read_mcp_json
from spark.brain.mcp_probe import ProbeResult, probe
from spark.config import Settings

log = logging.getLogger(__name__)

Probe = Callable[..., ProbeResult]


def _target(config: dict[str, Any]) -> str:
    if config.get("type", "stdio") == "stdio":
        parts = [str(config.get("command", "")), *map(str, config.get("args") or [])]
        return " ".join(parts)
    return str(config.get("url", ""))


def _transport(config: dict[str, Any]) -> str:
    return {"stdio": "Local (stdio)", "http": "Remote (HTTP)", "sse": "Remote (SSE)"}.get(
        str(config.get("type", "stdio")), str(config.get("type"))
    )


class ConnectorsApi:
    """Methods the Connectors page calls as ``pywebview.api.<name>()``."""

    def __init__(
        self,
        settings: Settings,
        *,
        on_apply: Callable[[], Any] | None = None,
        is_running: Callable[[], bool] | None = None,
        probe_fn: Probe = probe,
        live_status: Callable[[], dict[str, dict[str, str]]] | None = None,
    ) -> None:
        self._settings = settings
        self._on_apply = on_apply
        self._is_running = is_running or (lambda: False)
        self._live_status = live_status or (lambda: {})
        self._probe = probe_fn
        self._dirty = False
        self._config_tests: dict[str, dict[str, Any]] = {}

    # --- reading -------------------------------------------------------------------------

    def _memory(self) -> Any:
        from spark.memory import MemoryStore

        return MemoryStore.from_settings(self._settings)

    def _extras(self, name: str) -> dict[str, Any]:
        """Live status in Spark's session and the number of notes Spark keeps (P5.7j)."""
        try:
            live = self._live_status().get(name)
        except Exception:
            live = None
        notes = len(self._memory().connector_bullets(name)) if self._settings.memory.enabled else 0
        return {"live": live, "notes": notes}

    def _config_servers(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Hand-written servers: (definitions, why-disabled)."""
        try:
            raw = read_mcp_json(self._settings.path(self._settings.connectors.mcp_config))
        except McpConfigError as exc:
            return {}, {"mcp_servers.json": str(exc)}
        try:
            setup = load_mcp_servers(self._settings)
        except McpConfigError as exc:
            return raw, {name: str(exc) for name in raw}
        return raw, setup.disabled

    def _app_view(self, name: str, entry: dict[str, Any]) -> dict[str, Any]:
        config = entry.get("config") or {}
        tools = entry.get("tools") or []
        if entry.get("last_error"):
            status = "error"
        elif entry.get("tested_at"):
            status = "ok"
        else:
            status = "untested"
        return {
            "name": name,
            "source": "app",
            "transport": _transport(config),
            "target": _target(config),
            "enabled": bool(entry.get("enabled")),
            "status": status,
            "error": entry.get("last_error"),
            "tested_at": entry.get("tested_at"),
            "route": entry.get("route") or "deep",
            "tools": [
                {k: t.get(k) for k in ("name", "title", "description", "hint", "mode")}
                for t in tools
            ],
            **self._extras(name),
        }

    def _config_view(
        self, name: str, config: dict[str, Any], why_off: str | None
    ) -> dict[str, Any]:
        cfg = self._settings.connectors.servers.get(name)
        test = self._config_tests.get(name, {})
        modes: dict[str, str] = {}
        if cfg is not None:
            for t in cfg.allowed_tools:
                modes[t.split("__")[-1]] = "allow"
            for t in cfg.side_effecting:
                modes[t.split("__")[-1]] = "ask"
            for t in cfg.destructive:
                modes[t.split("__")[-1]] = "off"
        tools = test.get("tools")
        if tools is None:
            tools = [{"name": n, "title": None, "description": "", "hint": None} for n in modes]
        return {
            "name": name,
            "source": "config",
            "transport": _transport(config),
            "target": _target(config),
            "enabled": why_off is None,
            "status": "error" if test.get("error") else "ok" if test.get("tools") else "untested",
            "error": test.get("error") or why_off,
            "tested_at": test.get("tested_at"),
            "route": cfg.route if cfg is not None else "auto",
            "tools": [{**t, "mode": modes.get(t["name"], "off")} for t in tools],
            **self._extras(name),
        }

    # --- claude.ai account connectors ------------------------------------------------------

    def _claudeai_policy(self, name: str, entries: dict[str, Any] | None = None) -> Any:
        from spark.config import ConnectorServerConfig

        if entries is None:
            entries = store.load_store(store.store_path(self._settings))
        base = self._settings.connectors.servers.get(name)
        entry = entries.get(name)
        if entry is not None:
            return store.claudeai_policy(base, entry)
        return base or ConnectorServerConfig(enabled=False, route="deep")

    def _claudeai_names(self, entries: dict[str, Any]) -> list[str]:
        names = set(claudeai.load_state(self._settings))
        names |= {n for n in self._settings.connectors.servers if claudeai.is_claudeai(n)}
        names |= {n for n in entries if claudeai.is_claudeai(n)}
        return sorted(names)

    def _claudeai_view(self, name: str, entries: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = self._claudeai_policy(name, entries)
        seen = claudeai.load_state(self._settings).get(name)
        blocked = self._settings.connectors.block_claudeai_connectors

        def bare(e: str) -> str:
            return e.rsplit("__", 1)[-1]

        modes: dict[str, str] = {}
        for t in cfg.destructive:
            modes[bare(t)] = "off"
        for t in cfg.side_effecting:
            modes[bare(t)] = "ask"
        for t in cfg.allowed_tools:
            modes[bare(t)] = "allow"
        names = list(seen["tools"]) if seen and seen["tools"] else sorted(modes)
        if seen and seen.get("status") == "needs-auth":
            names = sorted(modes) or names  # only authenticate tools until signed in
        tools = [
            {"name": n, "title": None, "description": "", "hint": None, "mode": modes.get(n, "off")}
            for n in names
            if not any(c in n for c in "*?[")
        ]
        status = seen.get("status") if seen else None
        if status == "connected":
            view_status, error = "ok", None
        elif status == "needs-auth":
            view_status, error = "needs-auth", f"Needs sign-in: {claudeai.SIGN_IN_HINT}."
        elif status is None:
            view_status = "untested"
            error = (
                "Not seen in your claude.ai account yet. Connect it on claude.ai, then start "
                "Spark once so it can check."
            )
        else:
            view_status, error = "error", f"claude.ai reports it as {status}."
        if blocked:
            error = "All claude.ai connectors are off: block_claudeai_connectors is true."
        return {
            "name": name,
            "label": claudeai.display_name(name),
            "source": "claudeai",
            "transport": "claude.ai account",
            "target": seen.get("name", "") if seen else f"claude.ai {claudeai.display_name(name)}",
            "enabled": bool(cfg.enabled) and not blocked,
            "status": view_status,
            "error": error,
            "tested_at": None,
            "route": cfg.route,
            "link": claudeai.SETTINGS_URL,
            "tools": tools,
            **self._extras(name),
        }

    def _set_claudeai(self, name: str, **changes: Any) -> dict[str, Any]:
        try:
            store.set_claudeai(self._settings, name, **changes)
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        self._dirty = True
        return {"ok": True, "server": self._claudeai_view(name)}

    def open_link(self, url: str) -> dict[str, Any]:
        """Open claude.ai's connector settings in the browser (the only link allowed)."""
        if url != claudeai.SETTINGS_URL:
            return {"ok": False, "error": "That link can't be opened from here."}
        import webbrowser

        webbrowser.open(url)
        return {"ok": True}

    def list(self) -> dict[str, Any]:
        """Every server, app ones first, plus whether the brain needs a restart."""
        entries = store.load_store(store.store_path(self._settings))
        servers = [
            self._app_view(name, entry)
            for name, entry in sorted(entries.items())
            if not claudeai.is_claudeai(name)
        ]
        raw, disabled = self._config_servers()
        for name in sorted(set(raw) | set(self._settings.connectors.servers)):
            config = raw.get(name)
            if not isinstance(config, dict) or claudeai.is_claudeai(name):
                continue
            servers.append(self._config_view(name, config, disabled.get(name)))
        servers += [self._claudeai_view(name, entries) for name in self._claudeai_names(entries)]
        return {
            "servers": servers,
            "dirty": self._dirty,
            "running": bool(self._is_running()),
            "claude_desktop": str(p) if (p := store.claude_desktop_config_path()) else None,
            "env_file": str(store.env_path(self._settings)),
        }

    def get_config(self, name: str) -> dict[str, Any]:
        """The server definition as editable JSON (placeholders, never secret values)."""
        entry = store.load_store(store.store_path(self._settings)).get(name)
        if entry is None:
            return {"ok": False, "error": f"No connector named {name!r}."}
        return {"ok": True, "text": json.dumps({name: entry.get("config", {})}, indent=2)}

    # --- changing ------------------------------------------------------------------------

    def _reserved(self) -> set[str]:
        raw, _ = self._config_servers()
        return set(raw) | set(self._settings.connectors.servers)

    def add(self, text: str) -> dict[str, Any]:
        """Paste -> stored servers (off, untested). The page then calls :meth:`test` on each."""
        try:
            result = store.add_servers(self._settings, text, reserved=self._reserved())
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        log.info(
            "connectors: added %s (new .env vars: %s)",
            ", ".join(result.added) or "nothing",
            ", ".join(result.env_vars) or "none",
        )
        return {
            "ok": True,
            "added": result.added,
            "notes": result.notes,
            "env_vars": result.env_vars,
            "skipped": result.skipped,
        }

    def import_claude_desktop(self) -> dict[str, Any]:
        path = store.claude_desktop_config_path()
        if path is None:
            return {"ok": False, "error": "Claude Desktop's config file wasn't found."}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"Can't read {path.name}: {exc.strerror}"}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"{path.name} isn't valid JSON ({exc.msg})."}
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict) or not servers:
            return {"ok": False, "error": "Claude Desktop has no MCP servers set up."}
        return self.add(json.dumps({"mcpServers": servers}))

    def update_config(self, name: str, text: str) -> dict[str, Any]:
        """Replace a server's definition (from the editor). The name may not change."""
        try:
            parsed = store.split_pasted(text)
            if len(parsed) != 1:
                raise ConnectorError("Edit one server at a time.")
            _, raw = parsed[0]
            result = store.add_servers(
                self._settings,
                json.dumps({name: raw}),
                reserved=self._reserved(),
                replace=True,
            )
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        self._dirty = True
        return {"ok": True, "notes": result.notes.get(name, []), "env_vars": result.env_vars}

    def test(self, name: str) -> dict[str, Any]:
        """Connect, list the tools, store them (keeping chosen modes), return the server."""
        if claudeai.is_claudeai(name):
            return {
                "ok": False,
                "error": "claude.ai connectors are checked each time Spark connects.",
                "server": self._claudeai_view(name),
            }
        entries = store.load_store(store.store_path(self._settings))
        if name in entries:
            config = entries[name].get("config") or {}
            result = self._probe(config, base_dir=self._settings.config_dir)
            entry = store.record_test(
                self._settings, name, result.tools if result.ok else None, result.error
            )
            log.info(
                "connectors: test %s -> %s",
                name,
                f"{len(result.tools)} tools" if result.ok else "failed",
            )
            if entry.get("enabled"):
                self._dirty = True
            return {"ok": result.ok, "server": self._app_view(name, entry)}
        raw, disabled = self._config_servers()
        if name not in raw:
            return {"ok": False, "error": f"No connector named {name!r}."}
        result = self._probe(raw[name], base_dir=self._settings.config_dir)
        self._config_tests[name] = {
            "tools": result.tools if result.ok else None,
            "error": result.error,
            "tested_at": store.utcnow(),
        }
        return {"ok": result.ok, "server": self._config_view(name, raw[name], disabled.get(name))}

    def set_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        if claudeai.is_claudeai(name):
            return self._set_claudeai(name, enabled=bool(enabled))
        try:
            entry = store.update(self._settings, name, enabled=bool(enabled))
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        self._dirty = True
        return {"ok": True, "server": self._app_view(name, entry)}

    def set_tool(self, name: str, tool: str | None, mode: str) -> dict[str, Any]:
        """One tool's mode (``tool``), or every tool's (``tool`` None)."""
        if claudeai.is_claudeai(name):
            if mode not in store.TOOL_MODES:
                return {"ok": False, "error": f"Unknown mode {mode!r}."}
            known = [t["name"] for t in self._claudeai_view(name)["tools"]]
            if tool is not None and tool not in known:
                return {"ok": False, "error": f"{name} has no tool {tool!r}."}
            return self._set_claudeai(
                name, tools=dict.fromkeys(known if tool is None else [tool], mode)
            )
        try:
            entry = store.set_tool_mode(self._settings, name, tool, mode)
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        if entry.get("enabled"):
            self._dirty = True
        return {"ok": True, "server": self._app_view(name, entry)}

    def set_route(self, name: str, route: str) -> dict[str, Any]:
        """Deep: requests about this connector go to the deep model (P5.1). Window-added only."""
        if route not in ("deep", "auto"):
            return {"ok": False, "error": "Route must be deep or auto."}
        if claudeai.is_claudeai(name):
            return self._set_claudeai(name, route=route)
        try:
            entry = store.update(self._settings, name, route=route)
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        if entry.get("enabled"):
            self._dirty = True
        return {"ok": True, "server": self._app_view(name, entry)}

    def get_notes(self, name: str) -> dict[str, Any]:
        """What Spark learned about this connector (memory/connectors/<name>.md)."""
        memory = self._memory()
        return {
            "ok": True,
            "text": memory.connector_notes(name),
            "path": str(memory.connector_path(name)),
            "enabled": self._settings.memory.enabled,
        }

    def save_notes(self, name: str, text: str) -> dict[str, Any]:
        """Replace the notes with what the user wrote. Used from Spark's next conversation."""
        try:
            self._memory().write_connector_notes(name, str(text or ""))
        except OSError as exc:
            return {"ok": False, "error": f"Couldn't save: {exc}"}
        return {**self.get_notes(name), "notes": len(self._memory().connector_bullets(name))}

    def remove(self, name: str) -> dict[str, Any]:
        if claudeai.is_claudeai(name):
            return {"ok": False, "error": "Disconnect claude.ai connectors on claude.ai."}
        try:
            entry = store.load_store(store.store_path(self._settings)).get(name, {})
            store.remove(self._settings, name)
        except ConnectorError as exc:
            return {"ok": False, "error": str(exc)}
        if entry.get("enabled"):
            self._dirty = True
        return {"ok": True}

    def apply(self) -> dict[str, Any]:
        """Rebuild the voice brain so it loads the current connectors."""
        if self._on_apply is not None:
            self._on_apply()
        self._dirty = False
        return {"ok": True}
