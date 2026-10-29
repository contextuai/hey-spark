"""Load ``mcp_servers.json`` and build Spark's MCP tool policy (spec §7, §12).

The output of :func:`load_mcp_servers` is an :class:`McpSetup`. Its fields map onto the SDK
options: ``servers`` goes to ``ClaudeAgentOptions.mcp_servers``, ``allowed_tools`` to
``allowed_tools``, and ``disallowed_tools`` to ``disallowed_tools``. :func:`env_for_sdk` builds
``ClaudeAgentOptions.env``. ``policy`` is what the PreToolUse confirmation hook consults.

Decisions (the evidence is in docs/connectors.md §3-4 and docs/sdk_notes.md, P2.1 section):

* **``${VAR}`` placeholders stay unexpanded** in ``args``, ``env``, ``url`` and ``headers``.
  The SDK puts a dict ``mcp_servers`` on the ``claude`` command line as JSON
  (``_internal/transport/subprocess_cli.py:663-684``), where other processes and EDR tools
  can read it. The CLI itself expands ``${VAR}`` and ``${VAR:-default}`` in an inline
  ``--mcp-config`` from its own environment, which is ``os.environ`` plus
  ``ClaudeAgentOptions.env``. Python only *checks* that every referenced variable is set
  and non-empty. If one is missing, the server is disabled with a warning that names the
  variable and never shows a value. ``${VAR:-default}`` does not count as required.
* **``command`` is the one field expanded in Python.** It is a path, not a secret, and it
  has to be resolved to check that the binary exists and to handle Windows shims.
  - A bare name is looked up with ``shutil.which``.
  - A relative path is resolved against the config directory, with ``.exe`` tried on
    Windows.
  - On Windows, a ``.cmd``/``.bat`` shim such as ``npx`` becomes
    ``cmd /c <name> <args...>``, because Node's ``spawn`` cannot run a ``.cmd`` without a
    shell (connectors.md §4.6).
  - A missing binary disables the server, and the reason includes a fix.
* **Allowlists are exact.**
  - ``allowed_tools`` entries become exact ``mcp__<server>__<tool>`` names. Globs expand
    only against the known tool names passed in (for example, a live ``tools/list``). A
    glob with no match is dropped with a warning and is never passed through. A
    server-wide wildcard is rejected, both by config validation and here.
  - Side-effecting tools are never pre-approved. They fall through to the confirmation
    hook.
  - ``destructive`` entries become ``disallowed_tools`` deny rules unless the server sets
    ``allow_destructive: true``. Deny rules accept wildcards anywhere (CLI rule parser), so
    a glob such as ``GMAIL_DELETE_*`` is passed as-is. When destructive tools are allowed,
    they still need confirmation.
* **Precedence:** destructive, then side_effect, then read. An ``mcp__`` tool that isn't on
  any list, or that belongs to a server Spark didn't load, is classified ``"unknown"``. The
  hook must deny it by default.
* **Which servers load:**
  - Every server needs an entry in ``config.yaml`` under ``connectors.servers``. A server
    that is only in the JSON is skipped with a warning, because it has no allowlist.
  - A server that is in the config but missing from the JSON is skipped with a warning.
  - ``enabled: false`` skips a server quietly.
  - Servers added in the desktop Connectors window (``connectors.json``, see
    :mod:`spark.brain.connector_store`) are merged in with their own per-tool lists. A name
    already used in ``config.yaml`` / ``mcp_servers.json`` wins over the app's entry.
* **claude.ai account connectors** (:mod:`spark.brain.claudeai`): a ``connectors.servers`` key
  starting with ``claude_ai_`` needs no JSON entry, because the logged-in CLI injects the server
  itself. Its policy is registered as usual, but it is not put in ``servers``; its key goes to
  ``claudeai``. When at least one is enabled, every other claude.ai server (the ones the CLI
  reported last, plus a few common ones) gets a server-level ``mcp__<key>`` deny rule, so the
  model never sees it. ``block_claudeai_connectors: true`` disables them all.
* **File format:** keys starting with ``_`` are ignored at the top level and inside server
  entries. They hold comments. A Claude-Desktop-style ``{"mcpServers": {...}}`` wrapper is
  accepted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from spark.config import ConnectorServerConfig, Settings

log = logging.getLogger("spark.brain.mcp")

ToolClass = Literal["read", "side_effect", "destructive", "unknown"]

MCP_PREFIX = "mcp__"
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}")
_SERVER_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_GLOB_CHARS = frozenset("*?[")
_WILDCARD_ONLY = frozenset("*?")
_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "stdio": frozenset({"type", "command", "args", "env"}),
    "http": frozenset({"type", "url", "headers"}),
    "sse": frozenset({"type", "url", "headers"}),
}
_WINDOWS_SHIM_SUFFIXES = frozenset({".cmd", ".bat"})
_INSTALL_HINTS = {
    "npx": "install Node.js (it provides npx)",
    "uvx": "install uv (it provides uvx)",
}


class McpConfigError(ValueError):
    """``mcp_servers.json`` exists but is not valid JSON or not a JSON object."""


def _is_windows() -> bool:
    return os.name == "nt"


def _is_glob(entry: str) -> bool:
    return any(c in _GLOB_CHARS for c in entry)


def tool_name(server: str, tool: str) -> str:
    """Full tool name as the model sees it: ``mcp__<server>__<tool>``."""
    return f"{MCP_PREFIX}{server}__{tool}"


def split_tool_name(name: str) -> tuple[str, str] | None:
    """``mcp__srv__tool`` -> ``("srv", "tool")``; None for anything else."""
    if not name.startswith(MCP_PREFIX):
        return None
    server, sep, tool = name[len(MCP_PREFIX) :].partition("__")
    if not sep or not server or not tool:
        return None
    return server, tool


def referenced_vars(value: Any) -> tuple[set[str], set[str]]:
    """Return ``(required, optional)`` env var names referenced as ``${VAR}`` in ``value``.

    ``${VAR:-default}`` is optional, since the CLI substitutes the default."""
    required: set[str] = set()
    optional: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, str):
            for m in _VAR_RE.finditer(v):
                (optional if m.group(2) is not None else required).add(m.group(1))
        elif isinstance(v, Mapping):
            for item in v.values():
                walk(item)
        elif isinstance(v, list | tuple):
            for item in v:
                walk(item)

    walk(value)
    return required, optional - required


def expand_vars(text: str, environ: Mapping[str, str]) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` like the CLI does. Only for non-secret fields."""

    def sub(m: re.Match[str]) -> str:
        val = environ.get(m.group(1), "")
        if not val and m.group(2) is not None:
            return m.group(2)[2:]
        return val

    return _VAR_RE.sub(sub, text)


# --- policy ------------------------------------------------------------------------------------


@dataclass
class _ServerRules:
    read: set[str] = field(default_factory=set)
    side_exact: set[str] = field(default_factory=set)
    side_globs: list[str] = field(default_factory=list)
    destructive_exact: set[str] = field(default_factory=set)
    destructive_globs: list[str] = field(default_factory=list)
    allow_destructive: bool = False

    def is_destructive(self, tool: str) -> bool:
        return tool in self.destructive_exact or any(
            fnmatchcase(tool, g) for g in self.destructive_globs
        )

    def is_side_effecting(self, tool: str) -> bool:
        return tool in self.side_exact or any(fnmatchcase(tool, g) for g in self.side_globs)


class ToolPolicy:
    """Classifies full tool names for the confirmation hook. Default for anything unlisted:
    ``"unknown"`` (the hook denies it)."""

    def __init__(self) -> None:
        self._rules: dict[str, _ServerRules] = {}

    @property
    def servers(self) -> frozenset[str]:
        return frozenset(self._rules)

    def _set_rules(self, server: str, rules: _ServerRules) -> None:
        self._rules[server] = rules

    def classify(self, tool_name: str) -> ToolClass:
        parts = split_tool_name(tool_name)
        if parts is None:
            return "unknown"
        rules = self._rules.get(parts[0])
        if rules is None:
            return "unknown"
        tool = parts[1]
        if rules.is_destructive(tool):
            return "destructive"
        if rules.is_side_effecting(tool):
            return "side_effect"
        if tool in rules.read:
            return "read"
        return "unknown"

    def is_side_effecting(self, tool_name: str) -> bool:
        return self.classify(tool_name) == "side_effect"

    def is_destructive(self, tool_name: str) -> bool:
        return self.classify(tool_name) == "destructive"

    def is_allowed(self, tool_name: str) -> bool:
        """True only for pre-approved read-only tools, which run without confirmation."""
        return self.classify(tool_name) == "read"

    def destructive_permitted(self, tool_name: str) -> bool:
        parts = split_tool_name(tool_name)
        rules = self._rules.get(parts[0]) if parts else None
        return bool(rules and rules.allow_destructive and rules.is_destructive(parts[1]))

    def needs_confirmation(self, tool_name: str) -> bool:
        """Side-effecting tools, plus destructive tools on a server with allow_destructive."""
        cls = self.classify(tool_name)
        return cls == "side_effect" or (
            cls == "destructive" and self.destructive_permitted(tool_name)
        )

    def is_blocked(self, tool_name: str) -> bool:
        """Unknown tools, and destructive tools that the server does not allow."""
        cls = self.classify(tool_name)
        return cls == "unknown" or (
            cls == "destructive" and not self.destructive_permitted(tool_name)
        )


# --- setup -------------------------------------------------------------------------------------


@dataclass
class McpSetup:
    """Everything the brain needs to wire MCP into ``ClaudeAgentOptions``."""

    servers: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    side_effecting: set[str] = field(default_factory=set)
    disabled: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    required_env: dict[str, list[str]] = field(default_factory=dict)
    policy: ToolPolicy = field(default_factory=ToolPolicy)
    #: Servers with ``route: deep``: name -> lowercase keywords that send a request deep.
    deep_route: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Enabled claude.ai account connectors (``claude_ai_*``). The CLI injects them, so they are
    #: not in ``servers``; non-empty means the session must allow them (non-strict MCP config).
    claudeai: list[str] = field(default_factory=list)
    #: claude.ai servers hidden with a server-level ``disallowed_tools`` entry.
    claudeai_hidden: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        log.warning(message)

    def disable(self, server: str, reason: str, *, quiet: bool = False) -> None:
        self.disabled[server] = reason
        if quiet:
            log.info("MCP server %r disabled: %s", server, reason)
        else:
            self.warn(f"MCP server {server!r} disabled: {reason}")

    def add_sdk_server(
        self,
        name: str,
        server: Any,
        *,
        read_only: Iterable[str] = (),
        side_effecting: Iterable[str] = (),
        destructive: Iterable[str] = (),
        allow_destructive: bool = False,
    ) -> None:
        """Merge an in-process server (``create_sdk_mcp_server(...)``) and its tool lists.

        Tool names must be exact, either bare (``current_time``) or full
        (``mcp__spark__current_time``). Globs are rejected here, because in-process tool names are
        always known."""
        _check_server_key(name)
        if name in self.servers or name in self.policy.servers:
            raise ValueError(f"MCP server {name!r} is already configured")
        lists = {
            "read_only": list(read_only),
            "side_effecting": list(side_effecting),
            "destructive": list(destructive),
        }
        for label, entries in lists.items():
            for entry in entries:
                bare = _normalize_entry(name, entry)
                if bare is None or _is_glob(bare):
                    raise ValueError(
                        f"{label} entry {entry!r} for SDK server {name!r} must be an exact "
                        f"tool name (bare or mcp__{name}__<tool>)"
                    )
        self.servers[name] = server
        self._register(
            name,
            allowed=lists["read_only"],
            side_effecting=lists["side_effecting"],
            destructive=lists["destructive"],
            allow_destructive=allow_destructive,
            known=(),
        )

    def _register(
        self,
        server: str,
        *,
        allowed: Iterable[str],
        side_effecting: Iterable[str],
        destructive: Iterable[str],
        allow_destructive: bool,
        known: Iterable[str],
    ) -> None:
        known_bare = {b for k in known if (b := _normalize_entry(server, k)) and not _is_glob(b)}
        rules = _ServerRules(allow_destructive=allow_destructive)

        for entry in destructive:
            bare = self._bare_or_warn(server, entry, "destructive")
            if bare is None:
                continue
            if _is_glob(bare):
                rules.destructive_globs.append(bare)
                rules.destructive_exact.update(k for k in known_bare if fnmatchcase(k, bare))
            else:
                rules.destructive_exact.add(bare)
            if not allow_destructive:
                self.disallowed_tools.append(tool_name(server, bare))

        for entry in side_effecting:
            bare = self._bare_or_warn(server, entry, "side_effecting")
            if bare is None:
                continue
            if _is_glob(bare):
                rules.side_globs.append(bare)
                rules.side_exact.update(k for k in known_bare if fnmatchcase(k, bare))
            else:
                rules.side_exact.add(bare)

        for entry in allowed:
            bare = self._bare_or_warn(server, entry, "allowed_tools")
            if bare is None:
                continue
            if _is_glob(bare):
                if set(bare) <= _WILDCARD_ONLY:
                    self.warn(
                        f"MCP server {server!r}: allowed_tools entry {entry!r} would auto-approve "
                        "every tool on the server; ignored"
                    )
                    continue
                matches = sorted(k for k in known_bare if fnmatchcase(k, bare))
                if not matches:
                    self.warn(
                        f"MCP server {server!r}: allowed_tools glob {entry!r} matched no known "
                        "tool names; ignored (globs are never passed through as auto-approval)"
                    )
                    continue
                candidates = matches
            else:
                candidates = [bare]
            for tool in candidates:
                if rules.is_destructive(tool) or rules.is_side_effecting(tool):
                    self.warn(
                        f"MCP server {server!r}: {tool!r} is listed as read-only but is also "
                        "side-effecting or destructive; it will not be pre-approved"
                    )
                    continue
                rules.read.add(tool)

        self.policy._set_rules(server, rules)
        for tool in sorted(rules.read):
            full = tool_name(server, tool)
            if full not in self.allowed_tools:
                self.allowed_tools.append(full)
        self.side_effecting.update(tool_name(server, t) for t in rules.side_exact)
        if allow_destructive:
            self.side_effecting.update(tool_name(server, t) for t in rules.destructive_exact)
            if rules.destructive_exact or rules.destructive_globs:
                self.warn(
                    f"MCP server {server!r}: allow_destructive is on; destructive tools are "
                    "available (each still needs a spoken confirmation)"
                )
        self.disallowed_tools[:] = list(dict.fromkeys(self.disallowed_tools))

    def _bare_or_warn(self, server: str, entry: str, label: str) -> str | None:
        bare = _normalize_entry(server, entry)
        if bare is None:
            self.warn(
                f"MCP server {server!r}: {label} entry {entry!r} does not name a tool of this "
                "server; ignored"
            )
        return bare


def _normalize_entry(server: str, entry: str) -> str | None:
    """Return the bare tool part of ``entry``, or None if it names another server or no tool."""
    entry = entry.strip()
    if not entry:
        return None
    if entry.startswith(MCP_PREFIX):
        parts = split_tool_name(entry)
        if parts is None or parts[0] != server:
            return None
        return parts[1]
    return entry


def _check_server_key(name: str) -> None:
    if not _SERVER_KEY_RE.match(name) or "__" in name:
        raise ValueError(
            f"MCP server name {name!r} must use only letters, digits, '_' or '-' and must not "
            "contain '__'"
        )


# --- loading -----------------------------------------------------------------------------------


def _strip_comments(obj: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}


def read_mcp_json(path: Path) -> dict[str, Any]:
    """Read the server definitions, dropping ``_comment``-style keys. A missing file yields {}."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise McpConfigError(f"{path} must contain a JSON object of servers")
    data = _strip_comments(data)
    if set(data) == {"mcpServers"} and isinstance(data["mcpServers"], dict):
        data = _strip_comments(data["mcpServers"])
    return data


def _resolve_command(
    command: str, base_dir: Path, environ: Mapping[str, str]
) -> tuple[str, list[str]] | str:
    """Return ``(command, prefix_args)`` or an error message with a fix."""
    source_vars = sorted(referenced_vars(command)[0])
    expanded = expand_vars(command, environ).strip().strip('"')
    via = f" (from ${{{source_vars[0]}}})" if source_vars else ""
    if not expanded:
        return f"stdio command is empty{via}"

    if not any(sep in expanded for sep in ("/", "\\")) and not expanded.startswith("~"):
        found = shutil.which(expanded)
        if found is None:
            hint = _INSTALL_HINTS.get(Path(expanded).stem.lower(), "install it or use a full path")
            return f"command {expanded!r}{via} not found on PATH; fix: {hint}"
        if _is_windows() and Path(found).suffix.lower() in _WINDOWS_SHIM_SUFFIXES:
            return "cmd", ["/c", expanded]
        return found, []

    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if _is_windows() and not path.exists() and not path.suffix:
        exe = path.with_suffix(".exe")
        if exe.exists():
            path = exe
    if not path.is_file():
        fix = (
            f"set {source_vars[0]} in .env to the full path of the server executable"
            if source_vars
            else "fix the command path in mcp_servers.json"
        )
        return f"executable not found at {path}{via}; fix: {fix}"
    if _is_windows() and path.suffix.lower() in _WINDOWS_SHIM_SUFFIXES:
        return "cmd", ["/c", str(path)]
    return str(path), []


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _is_str_map(v: Any) -> bool:
    return isinstance(v, dict) and all(
        isinstance(k, str) and isinstance(x, str) for k, x in v.items()
    )


def _prepare_server(
    name: str,
    raw: Any,
    *,
    base_dir: Path,
    environ: Mapping[str, str],
    setup: McpSetup,
) -> dict[str, Any] | str:
    """Validate one JSON entry. Returns the SDK config dict or a reason for disabling it."""
    try:
        _check_server_key(name)
    except ValueError as exc:
        return str(exc)
    if not isinstance(raw, Mapping):
        return "entry in mcp_servers.json must be a JSON object"
    entry = _strip_comments(raw)
    kind = entry.get("type", "stdio" if "command" in entry else None)
    if kind not in _ALLOWED_KEYS:
        return f"unsupported type {kind!r} (use stdio, http or sse; in-process servers are code)"
    unknown = sorted(set(entry) - _ALLOWED_KEYS[kind])
    if unknown:
        setup.warn(f"MCP server {name!r}: ignoring unknown keys {unknown} for type {kind!r}")
    entry = {k: v for k, v in entry.items() if k in _ALLOWED_KEYS[kind]}

    if kind == "stdio":
        if not isinstance(entry.get("command"), str):
            return "stdio server needs a string 'command'"
        if "args" in entry and not _is_str_list(entry["args"]):
            return "'args' must be a list of strings"
        if "env" in entry and not _is_str_map(entry["env"]):
            return "'env' must be an object of string values"
    else:
        if not isinstance(entry.get("url"), str):
            return f"{kind} server needs a string 'url'"
        if "headers" in entry and not _is_str_map(entry["headers"]):
            return "'headers' must be an object of string values"

    required, _optional = referenced_vars(entry)
    setup.required_env[name] = sorted(required)
    missing = sorted(v for v in required if not environ.get(v))
    if missing:
        return f"missing environment variable(s) {', '.join(missing)}; set them in .env"

    if kind == "stdio":
        resolved = _resolve_command(entry["command"], base_dir, environ)
        if isinstance(resolved, str):
            return resolved
        command, prefix = resolved
        out: dict[str, Any] = {
            "type": "stdio",
            "command": command,
            "args": [*prefix, *entry.get("args", [])],
        }
        if "env" in entry:
            out["env"] = dict(entry["env"])
        return out

    url = expand_vars(entry["url"], environ).strip()
    if not url.lower().startswith(("http://", "https://")):
        url_vars = sorted(referenced_vars(entry["url"])[0])
        where = f" (check {', '.join(url_vars)})" if url_vars else ""
        return f"'url' is not an http(s) URL{where}"
    out = {"type": kind, "url": entry["url"]}
    if "headers" in entry:
        out["headers"] = dict(entry["headers"])
    return out


def load_mcp_servers(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
    known_tools: Mapping[str, Iterable[str]] | None = None,
) -> McpSetup:
    """Build the MCP setup from ``connectors`` in config and ``mcp_servers.json``.

    ``environ`` defaults to ``os.environ`` (``load_settings`` has already merged ``.env``).
    ``known_tools`` maps a server name to its tool names (bare or full), for example from a
    live ``tools/list``. Globs in the config lists expand against these names."""
    env: Mapping[str, str] = os.environ if environ is None else environ
    known = known_tools or {}
    conn = settings.connectors
    setup = McpSetup()
    path = settings.path(conn.mcp_config)
    raw_servers = read_mcp_json(path)
    if not path.exists():
        setup.warn(f"MCP config {path} not found; no external connectors loaded")
    configured: dict[str, Any] = dict(conn.servers)
    from spark.brain.connector_store import merge_into

    merge_into(settings, raw_servers, configured, setup.warn)

    for name in raw_servers:
        if name not in configured:
            setup.disable(
                name,
                f"no connectors.servers.{name} entry in config.yaml (a server needs an "
                "explicit tool allowlist before it is loaded)",
            )

    from spark.brain import claudeai

    for name, server_cfg in configured.items():
        if not server_cfg.enabled:
            setup.disable(name, "enabled: false in config.yaml", quiet=True)
            continue
        if claudeai.is_claudeai(name):
            if conn.block_claudeai_connectors:
                setup.disable(name, "block_claudeai_connectors is true in config.yaml", quiet=True)
                continue
            try:
                _check_server_key(name)
            except ValueError as exc:
                setup.disable(name, str(exc))
                continue
            # The CLI injects it: no mcp_servers.json entry, not in setup.servers.
            setup.claudeai.append(name)
            _register_configured(setup, name, server_cfg, known.get(name, ()))
            if server_cfg.route == "deep":
                setup.deep_route[name] = deep_route_keywords(name, server_cfg.route_keywords)
            continue
        if name not in raw_servers:
            setup.disable(name, f"enabled in config.yaml but has no entry in {path.name}")
            continue
        prepared = _prepare_server(
            name, raw_servers[name], base_dir=settings.config_dir, environ=env, setup=setup
        )
        if isinstance(prepared, str):
            setup.disable(name, prepared)
            continue
        setup.servers[name] = prepared
        _register_configured(setup, name, server_cfg, known.get(name, ()))
        if server_cfg.route == "deep":
            setup.deep_route[name] = deep_route_keywords(name, server_cfg.route_keywords)
    if setup.claudeai:
        setup.claudeai_hidden = claudeai.hidden_servers(settings, setup.claudeai, configured)
        for key in setup.claudeai_hidden:
            setup.disallowed_tools.append(f"{MCP_PREFIX}{key}")
        setup.disallowed_tools[:] = list(dict.fromkeys(setup.disallowed_tools))
        log.info(
            "claude.ai connectors: using %s; hiding %s",
            ", ".join(setup.claudeai),
            ", ".join(setup.claudeai_hidden) or "none",
        )
    return setup


def claudeai_enabled(settings: Settings) -> list[str]:
    """The claude.ai connectors Spark would load (config plus Connectors-window choices); []
    when ``block_claudeai_connectors`` is on."""
    from spark.brain import claudeai
    from spark.brain.connector_store import merge_into

    if settings.connectors.block_claudeai_connectors:
        return []
    configured: dict[str, Any] = dict(settings.connectors.servers)
    merge_into(settings, {}, configured, log.debug)
    return sorted(n for n, c in configured.items() if claudeai.is_claudeai(n) and c.enabled)


def deep_route_keywords(name: str, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Lowercase words that name a ``route: deep`` server in a request: the server name (with
    ``_``/``-`` also as spaces) plus its ``route_keywords``."""
    words = [name, name.replace("_", " ").replace("-", " "), *extra]
    return tuple(dict.fromkeys(w.strip().lower() for w in words if w.strip()))


def _register_configured(
    setup: McpSetup, name: str, cfg: ConnectorServerConfig, known: Iterable[str]
) -> None:
    setup._register(
        name,
        allowed=cfg.allowed_tools,
        side_effecting=cfg.side_effecting,
        destructive=cfg.destructive,
        allow_destructive=cfg.allow_destructive,
        known=known,
    )


def env_for_sdk(settings: Settings, *, claudeai: bool | None = None) -> dict[str, str]:
    """Extra environment for ``ClaudeAgentOptions.env``.

    ``ENABLE_CLAUDEAI_MCP_SERVERS`` decides whether a logged-in CLI injects the user's claude.ai
    account connectors (connectors.md §4.7). It is ``true`` only when the session should use
    them: ``claudeai`` True (the brain passes ``bool(setup.claudeai)``), or, when ``claudeai`` is
    None, when some ``claude_ai_*`` server is enabled and not blocked. Helper sessions (router,
    side questions) pass False. They also need ``strict_mcp_config=False`` to load at all."""
    use = bool(claudeai_enabled(settings)) if claudeai is None else claudeai
    use = use and not settings.connectors.block_claudeai_connectors
    return {"ENABLE_CLAUDEAI_MCP_SERVERS": "true" if use else "false"}
