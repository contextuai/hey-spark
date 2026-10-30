"""Connectors added from the desktop window: ``connectors.json`` (next to ``config.yaml``).

The Connectors window owns this file; people don't need to edit it by hand. It holds, per server:

* ``config``: the server definition in the same shape as an ``mcp_servers.json`` entry, with every
  secret replaced by a ``${VAR}`` placeholder. The values live in ``.env`` (spec §12).
* ``enabled``: whether the voice brain loads the server. New servers start off.
* ``tools``: the tools found by the last connection test, each with a mode:

  - ``allow``: runs without asking (becomes ``allowed_tools``)
  - ``ask``: needs a spoken "yes" first (``side_effecting``)
  - ``off``: never available to the model (``destructive`` with ``allow_destructive: false``,
    so it is passed as a ``disallowed_tools`` deny rule)

  A tool the server offers that isn't in this list (it appeared after the last test) is
  "unknown" to the policy, and the confirmation hook denies it.

claude.ai account connectors (keys starting ``claude_ai_``, see :mod:`spark.brain.claudeai`)
are stored here too, as ``{"kind": "claude_ai", "enabled": ..., "tools": [...]}`` with no
``config`` (the CLI injects the server). Such an entry overrides the ``config.yaml`` defaults:
its ``enabled`` wins and each listed tool's mode replaces the configured one.

:func:`merge_into` adds these servers to what :func:`spark.brain.mcp_config.load_mcp_servers`
reads from ``mcp_servers.json`` and ``config.yaml``. A server switched on in ``config.yaml``
keeps its name: an app server with the same name is skipped with a warning (the window refuses
to create one). One that is off there gives the name up, and the app server replaces it.

Pasted text (:func:`parse_pasted`) may be a Claude Desktop ``claude_desktop_config.json``
(``{"mcpServers": {...}}``), a map of named servers, or one server object. Two common stdio
wrappers around a remote server are turned back into a direct ``http`` connection, so no Node
process is needed and the token leaves the command line:

* ``node .../mcp-bridge.js <url> [token]`` (a stdio-to-Streamable-HTTP bridge)
* ``npx mcp-remote <url> [--header "Name: value"]...``
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlsplit

from spark.brain.mcp_config import referenced_vars

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger("spark.brain.mcp")

ToolMode = Literal["allow", "ask", "off"]
TOOL_MODES: tuple[ToolMode, ...] = ("allow", "ask", "off")

_LOCK = threading.RLock()
_NAME_CLEAN = re.compile(r"[^A-Za-z0-9_-]+")
_ENV_NAME_CLEAN = re.compile(r"[^A-Z0-9]+")
_AUTH_SCHEME = re.compile(r"^(Bearer|Basic|Token|Bot)\s+(\S.*)$", re.IGNORECASE)
_SECRETISH = re.compile(r"key|token|secret|pass|pat\b|auth|credential|cookie|session", re.I)
_TOKENISH = re.compile(r"^[A-Za-z0-9_\-.~+/=]{20,}$")
_VSCODE_INPUT = re.compile(r"\$\{input:[^}]*\}")
_VSCODE_ENV = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


class ConnectorError(ValueError):
    """Something in the pasted text or a requested change can't be used. The message is shown
    in the window as is, so it never contains a secret value."""


def utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --- file ----------------------------------------------------------------------------------------


def store_path(settings: Settings) -> Path:
    return settings.path(settings.connectors.ui_file)


def env_path(settings: Settings) -> Path:
    return settings.config_dir / ".env"


def load_store(path: Path) -> dict[str, Any]:
    """The servers in ``connectors.json`` (name -> entry). A missing or broken file yields {}."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("connectors: can't read %s (%s); ignoring it", path, exc)
        return {}
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        return {}
    return {k: v for k, v in servers.items() if isinstance(v, dict) and isinstance(k, str)}


def save_store(path: Path, servers: Mapping[str, Any]) -> None:
    data = {
        "_comment": "Managed by the Connectors window. Secrets are ${VAR} placeholders; "
        "the values are in .env next to this file.",
        "servers": dict(servers),
    }
    from spark.util.files import write_text_atomic

    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- .env ----------------------------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        return {}
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _quote_env(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_\-.:/@+=~]*", value):
        return value
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


@dataclass
class SecretSink:
    """Collects secrets pulled out of pasted configs and writes them to ``.env`` once.

    The same value pasted for several servers (one shared token) is stored once and every
    server refers to that one variable."""

    existing: Mapping[str, str]
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    new: dict[str, str] = field(default_factory=dict)
    source: str = "the Connectors window"  # named in the comment line above the values

    def var_for(self, value: str, suggested: str) -> str:
        for name, known in (*self.new.items(), *self.existing.items()):
            if known == value:
                return name
        base = _ENV_NAME_CLEAN.sub("_", suggested.upper()).strip("_") or "SPARK_MCP_SECRET"
        name, n = base, 2
        while name in self.new or name in self.existing or name in self.environ:
            if self.environ.get(name) == value and name not in self.new:
                return name  # already set in the environment to this very value
            name, n = f"{base}_{n}", n + 1
        self.new[name] = value
        return name

    def write(self, path: Path, environ: MutableMapping[str, str] | None = None) -> list[str]:
        """Append the new variables to ``.env`` and set them in this process. Returns the names."""
        if not self.new:
            return []
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = path.is_file() and not path.read_bytes().endswith(b"\n")
        with path.open("a", encoding="utf-8") as fh:
            if needs_newline:
                fh.write("\n")
            fh.write(f"# Added by {self.source}, {utcnow()}\n")
            for name, value in self.new.items():
                fh.write(f"{name}={_quote_env(value)}\n")
        target = os.environ if environ is None else environ
        target.update(self.new)
        names = list(self.new)
        self.new = {}
        return names


def _assigns(line: str, names: set[str]) -> bool:
    m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
    return bool(m) and m.group(1) in names


def remove_env_vars(
    path: Path, names: Iterable[str], environ: MutableMapping[str, str] | None = None
) -> bool:
    """Delete every ``NAME=...`` line for ``names`` from ``.env`` (other lines and comments are
    kept) and unset them in this process. True if the file had any of them."""
    wanted = set(names)
    target = os.environ if environ is None else environ
    for name in wanted:
        target.pop(name, None)
    with _LOCK:
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except FileNotFoundError:
            return False
        kept = [line for line in lines if not _assigns(line, wanted)]
        if len(kept) == len(lines):
            return False
        from spark.util.files import write_text_atomic

        write_text_atomic(path, "".join(kept))
        return True


def set_env_secret(
    path: Path,
    name: str,
    value: str,
    *,
    environ: MutableMapping[str, str] | None = None,
    source: str = "the Brain window",
) -> None:
    """Store ``value`` in ``.env`` as exactly ``name`` (an earlier value is replaced) and set it
    in this process. The value is never logged."""
    with _LOCK:
        remove_env_vars(path, [name], environ)
        sink = SecretSink({}, environ={}, source=source)
        sink.new[name] = value  # this exact name, even if the same value is stored elsewhere
        sink.write(path, environ)


# --- parsing pasted text ------------------------------------------------------------------------


@dataclass
class ParsedServer:
    name: str
    config: dict[str, Any]
    notes: list[str] = field(default_factory=list)


def clean_name(name: str) -> str:
    out = _NAME_CLEAN.sub("_", name.strip()).strip("_-")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:64]


def _derive_name(entry: Mapping[str, Any]) -> str:
    url = entry.get("url") or entry.get("serverUrl") or entry.get("httpUrl")
    if isinstance(url, str) and url:
        parts = urlsplit(url)
        segs = [s for s in parts.path.split("/") if s and s.lower() not in ("mcp", "sse", "v1")]
        return clean_name(segs[-1] if segs else (parts.hostname or "server"))
    args = [a for a in entry.get("args") or [] if isinstance(a, str) and not a.startswith("-")]
    for arg in args:
        stem = Path(arg.split("@")[0] if not arg.startswith("@") else arg).name
        if stem and stem.lower() not in ("/c", "-y", "npx", "uvx"):
            return clean_name(stem.removesuffix(".js").removesuffix(".py"))
    command = entry.get("command")
    return clean_name(Path(command).stem) if isinstance(command, str) else "server"


def _looks_like_server(obj: Mapping[str, Any]) -> bool:
    return any(k in obj for k in ("command", "url", "serverUrl", "httpUrl"))


def split_pasted(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Named raw entries from pasted JSON. Raises :class:`ConnectorError` with a plain message."""
    text = text.strip()
    if not text:
        raise ConnectorError("Paste a server configuration first.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:  # a bare `"name": {...}` fragment copied out of a bigger file
            data = json.loads("{" + text.strip().strip(",") + "}")
        except json.JSONDecodeError as exc:
            raise ConnectorError(f"That isn't valid JSON ({exc.msg}, line {exc.lineno}).") from exc
    if not isinstance(data, dict):
        raise ConnectorError("Expected a JSON object with one or more servers.")
    for wrapper in ("mcpServers", "servers", "mcp_servers"):
        if isinstance(data.get(wrapper), dict):
            data = data[wrapper]
            break
    if _looks_like_server(data):
        return [(_derive_name(data), dict(data))]
    out = [
        (str(k), dict(v))
        for k, v in data.items()
        if not str(k).startswith("_") and isinstance(v, dict)
    ]
    if not out:
        raise ConnectorError(
            "No servers found. Expected something like "
            '{"mcpServers": {"name": {"command": "...", "args": [...]}}}.'
        )
    return out


def _is_node(command: str) -> bool:
    return Path(command.strip('"')).stem.lower() == "node"


def _is_npx(command: str, args: list[str]) -> tuple[bool, list[str]]:
    stem = Path(command.strip('"')).stem.lower()
    if stem == "cmd" and args[:1] and args[0].lower() == "/c":
        return _is_npx(args[1], args[2:]) if len(args) > 1 else (False, args)
    return (stem in ("npx", "bunx", "pnpx"), args)


def _unwrap_remote(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Turn known stdio-to-HTTP wrappers into a direct http/sse entry."""
    command = entry.get("command")
    args = [a for a in entry.get("args") or [] if isinstance(a, str)]
    if not isinstance(command, str):
        return entry, None

    if _is_node(command) and len(args) >= 2 and args[0].lower().endswith("bridge.js"):
        url = args[1]
        if url.lower().startswith(("http://", "https://")):
            out: dict[str, Any] = {"type": "http", "url": url}
            if len(args) >= 3 and args[2]:
                out["headers"] = {"Authorization": f"Bearer {args[2]}"}
            return out, f"Connects straight to {url} (no Node bridge needed)."

    npx, rest = _is_npx(command, args)
    if npx:
        rest = [a for a in rest if a not in ("-y", "--yes")]
        if rest and rest[0].split("@")[0] == "mcp-remote":
            url = next((a for a in rest[1:] if a.lower().startswith(("http://", "https://"))), None)
            if url:
                headers: dict[str, str] = {}
                for i, a in enumerate(rest):
                    if a == "--header" and i + 1 < len(rest) and ":" in rest[i + 1]:
                        k, v = rest[i + 1].split(":", 1)
                        headers[k.strip()] = v.strip()
                kind = "sse" if urlsplit(url).path.rstrip("/").endswith("/sse") else "http"
                out = {"type": kind, "url": url}
                if headers:
                    out["headers"] = headers
                return out, f"Connects straight to {url} (no mcp-remote process needed)."
    return entry, None


def normalize_entry(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """One pasted entry -> an ``mcp_servers.json``-shaped entry (secrets still literal)."""
    notes: list[str] = []
    entry = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    for alt in ("serverUrl", "httpUrl"):
        if alt in entry and "url" not in entry:
            entry["url"] = entry.pop(alt)
            entry.setdefault("type", "http")
    if entry.get("type") in ("streamable-http", "streamable_http", "streamableHttp"):
        entry["type"] = "http"
    if "url" in entry and "command" not in entry and "type" not in entry:
        path = urlsplit(str(entry["url"])).path.rstrip("/")
        entry["type"] = "sse" if path.endswith("/sse") else "http"
    if "command" in entry:
        entry, note = _unwrap_remote(entry)
        if note:
            notes.append(note)
    kind = entry.get("type", "stdio" if "command" in entry else None)
    if kind == "stdio" or (kind is None and "command" in entry):
        out: dict[str, Any] = {"type": "stdio", "command": entry.get("command")}
        if not isinstance(out["command"], str) or not out["command"].strip():
            raise ConnectorError("a stdio server needs a 'command'")
        args = entry.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ConnectorError("'args' must be a list of strings")
        out["args"] = list(args)
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            raise ConnectorError("'env' must be an object")
        if env:
            out["env"] = {str(k): str(v) for k, v in env.items()}
        dropped = sorted(set(entry) - {"type", "command", "args", "env"})
    elif kind in ("http", "sse"):
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ConnectorError(f"an {kind} server needs a 'url'")
        out = {"type": kind, "url": url.strip()}
        headers = entry.get("headers") or {}
        if not isinstance(headers, dict):
            raise ConnectorError("'headers' must be an object")
        if headers:
            out["headers"] = {str(k): str(v) for k, v in headers.items()}
        dropped = sorted(set(entry) - {"type", "url", "headers"})
    else:
        raise ConnectorError(f"unsupported server type {kind!r} (use stdio, http or sse)")
    if dropped:
        notes.append(f"Ignored settings Spark doesn't use: {', '.join(dropped)}.")
    notes += _vscode_placeholders(out)
    return out, notes


def _vscode_placeholders(config: dict[str, Any]) -> list[str]:
    """VS Code's own variables in a config copied from its docs (GitHub's README shows one):
    ``${env:NAME}`` becomes Spark's ``${NAME}``; ``${input:id}`` makes VS Code prompt for the
    value, which Spark can't do, so it's refused rather than stored as if it were the token."""
    converted = False

    def fix(value: str) -> str:
        nonlocal converted
        m = _VSCODE_INPUT.search(value)
        if m:
            raise ConnectorError(
                f"{m.group(0)} is a VS Code prompt for a value, and Spark can't ask for it. "
                "Put the real value (your token or key) in its place; Spark moves it to .env."
            )
        new = _VSCODE_ENV.sub(r"${\1}", value)
        converted |= new != value
        return new

    for key in ("command", "url"):
        if isinstance(config.get(key), str):
            config[key] = fix(config[key])
    if "args" in config:
        config["args"] = [fix(a) for a in config["args"]]
    for key in ("env", "headers"):
        if key in config:
            config[key] = {k: fix(v) for k, v in config[key].items()}
    return ["Changed VS Code's ${env:NAME} to Spark's ${NAME}."] if converted else []


def extract_secrets(name: str, config: dict[str, Any], sink: SecretSink) -> list[str]:
    """Replace literal secrets in ``config`` with ``${VAR}`` and hand the values to ``sink``."""
    notes: list[str] = []
    prefix = f"SPARK_MCP_{name}"
    moved: list[str] = []

    headers = config.get("headers")
    if isinstance(headers, dict):
        for key, value in list(headers.items()):
            if not value or referenced_vars(value)[0]:
                continue
            m = _AUTH_SCHEME.match(value)
            if m:
                var = sink.var_for(m.group(2).strip(), f"{prefix}_TOKEN")
                headers[key] = f"{m.group(1)} ${{{var}}}"
            else:
                var = sink.var_for(value, f"{prefix}_{key}")
                headers[key] = f"${{{var}}}"
            moved.append(f"header {key}")

    env = config.get("env")
    if isinstance(env, dict):
        for key, value in list(env.items()):
            if not value or referenced_vars(value)[0] or not _SECRETISH.search(key):
                continue
            var = sink.var_for(value, f"{prefix}_{key}")
            env[key] = f"${{{var}}}"
            moved.append(f"env {key}")

    url = config.get("url")
    if isinstance(url, str) and not referenced_vars(url)[0]:
        query = parse_qsl(urlsplit(url).query, keep_blank_values=True)
        if any(_SECRETISH.search(k) for k, _ in query):
            var = sink.var_for(url, f"{prefix}_URL")
            config["url"] = f"${{{var}}}"
            moved.append("the URL (it carries a key)")

    if moved:
        notes.append(f"Moved {', '.join(moved)} to .env; the config only keeps a placeholder.")
    suspicious = [
        a
        for a in config.get("args", [])
        if _TOKENISH.match(a)
        and re.search(r"\d", a)
        and re.search(r"[A-Za-z]", a)
        and "/" not in a
        and not referenced_vars(a)[0]
    ]
    if suspicious:
        notes.append(
            "An argument looks like a secret and stays on the command line, where other "
            "programs can read it. Prefer an env variable if the server supports one."
        )
    return notes


def parse_pasted(text: str, sink: SecretSink) -> list[ParsedServer]:
    """Pasted JSON -> servers ready to store (secrets already moved to ``sink``)."""
    out: list[ParsedServer] = []
    seen: set[str] = set()
    for raw_name, raw in split_pasted(text):
        name = clean_name(raw_name) or _derive_name(raw)
        if not name:
            raise ConnectorError(f"Can't use {raw_name!r} as a server name.")
        base, n = name, 2
        while name in seen:
            name, n = f"{base}_{n}", n + 1
        seen.add(name)
        try:
            config, notes = normalize_entry(raw)
        except ConnectorError as exc:
            raise ConnectorError(f"{raw_name}: {exc}") from exc
        if name != raw_name:
            notes.insert(0, f"Named {name!r} (from {raw_name!r}).")
        notes += extract_secrets(name, config, sink)
        out.append(ParsedServer(name, config, notes))
    return out


# --- tools ---------------------------------------------------------------------------------------


def default_mode(hint: str | None) -> ToolMode:
    """A new tool's mode from the server's own hints: read-only runs, destructive is off, the
    rest asks first."""
    if hint == "read":
        return "allow"
    if hint == "destructive":
        return "off"
    return "ask"


def merge_tools(
    previous: Iterable[Mapping[str, Any]], found: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The tool list after a test: modes the user chose are kept; new tools get the default;
    tools the server no longer offers are dropped."""
    modes = {t.get("name"): t.get("mode") for t in previous}
    out = []
    for tool in found:
        mode = modes.get(tool["name"])
        out.append({**tool, "mode": mode if mode in TOOL_MODES else default_mode(tool.get("hint"))})
    return out


def policy_lists(entry: Mapping[str, Any]) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {"allow": [], "ask": [], "off": []}
    for tool in entry.get("tools") or []:
        name, mode = tool.get("name"), tool.get("mode")
        if isinstance(name, str) and name and mode in lists:
            lists[mode].append(name)
    return lists


def configured_on(configured: Mapping[str, Any], name: str) -> bool:
    """Whether ``config.yaml`` switches server ``name`` on. Only such a server keeps its name
    from the Connectors window; one that is off (like the shipped ``github`` example) or has
    no ``config.yaml`` entry gives the name to a window server, which then replaces it."""
    cfg = configured.get(name)
    return cfg is not None and bool(getattr(cfg, "enabled", False))


def merge_into(
    settings: Settings,
    raw_servers: dict[str, Any],
    configured: dict[str, Any],
    warn: Any,
) -> None:
    """Add the app's servers to the JSON definitions and the config policies (in place)."""
    from spark.brain.claudeai import is_claudeai
    from spark.config import ConnectorServerConfig

    for name, entry in load_store(store_path(settings)).items():
        if is_claudeai(name):
            try:
                configured[name] = claudeai_policy(configured.get(name), entry)
            except ValueError as exc:
                warn(f"claude.ai connector {name!r} from {settings.connectors.ui_file}: {exc}")
            continue
        if name in configured and configured_on(configured, name):
            warn(
                f"MCP server {name!r} from {settings.connectors.ui_file} skipped: a server "
                "with this name is switched on in config.yaml"
            )
            continue
        config = entry.get("config")
        if not isinstance(config, dict):
            continue
        lists = policy_lists(entry)
        try:
            policy = ConnectorServerConfig(
                enabled=bool(entry.get("enabled")),
                allowed_tools=lists["allow"],
                side_effecting=lists["ask"],
                destructive=lists["off"],
                allow_destructive=False,
                # Servers added in the window are mostly business data systems: answering
                # needs schema discovery plus queries, which the fast route can't fit.
                route=entry.get("route") or "deep",
                route_keywords=list(entry.get("route_keywords") or []),
            )
        except ValueError as exc:
            warn(f"MCP server {name!r} from {settings.connectors.ui_file} skipped: {exc}")
            continue
        raw_servers[name] = config
        configured[name] = policy


def claudeai_policy(base: Any, entry: Mapping[str, Any]) -> Any:
    """A claude.ai connector's policy: the ``config.yaml`` one (``base``, may be None) with the
    window's choices in ``entry`` on top. "Off" means off: ``allow_destructive`` is dropped once
    the user has chosen modes here."""
    from spark.config import ConnectorServerConfig

    if base is None:
        base = ConnectorServerConfig(enabled=False, route="deep")
    modes = {
        t["name"]: t["mode"]
        for t in entry.get("tools") or []
        if isinstance(t, Mapping) and isinstance(t.get("name"), str) and t.get("mode") in TOOL_MODES
    }

    def bare(e: str) -> str:
        return e.rsplit("__", 1)[-1] if e.startswith("mcp__") else e

    lists = {
        "allow": [e for e in base.allowed_tools if bare(e) not in modes],
        "ask": [e for e in base.side_effecting if bare(e) not in modes],
        "off": [e for e in base.destructive if bare(e) not in modes],
    }
    for tool, mode in modes.items():
        lists[mode].append(tool)
    return base.model_copy(
        update={
            "enabled": bool(entry.get("enabled", base.enabled)),
            "allowed_tools": lists["allow"],
            "side_effecting": lists["ask"],
            "destructive": lists["off"],
            "allow_destructive": base.allow_destructive and not modes,
            "route": entry.get("route") or base.route,
        }
    )


def set_claudeai(
    settings: Settings,
    name: str,
    *,
    enabled: bool | None = None,
    tools: Mapping[str, str] | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """Store the window's choices for a claude.ai connector (created on first change)."""
    from spark.brain.claudeai import is_claudeai

    if not is_claudeai(name):
        raise ConnectorError(f"{name!r} is not a claude.ai connector.")
    for mode in (tools or {}).values():
        if mode not in TOOL_MODES:
            raise ConnectorError(f"Unknown mode {mode!r}.")
    with _LOCK:
        path = store_path(settings)
        servers = load_store(path)
        entry = servers.setdefault(name, {"kind": "claude_ai", "tools": []})
        entry["kind"] = "claude_ai"
        if enabled is not None:
            entry["enabled"] = bool(enabled)
        if route is not None:
            if route not in ("deep", "auto"):
                raise ConnectorError("Route must be deep or auto.")
            entry["route"] = route
        if tools:
            current = {t.get("name"): t for t in entry.get("tools") or [] if isinstance(t, dict)}
            for tool, mode in tools.items():
                current[tool] = {"name": tool, "mode": mode}
            entry["tools"] = sorted(current.values(), key=lambda t: str(t.get("name")))
        save_store(path, servers)
        return entry


# --- operations the window uses ------------------------------------------------------------------


@dataclass
class AddResult:
    added: list[str]
    notes: dict[str, list[str]]
    env_vars: list[str]
    skipped: dict[str, str]


def add_servers(
    settings: Settings,
    text: str,
    *,
    reserved: Iterable[str] = (),
    replace: bool = False,
) -> AddResult:
    """Parse pasted text and store the servers (disabled, untested). ``reserved`` are names
    taken by servers switched on in config.yaml (:func:`configured_on`). An existing app
    server is skipped unless ``replace`` (its tool modes are kept)."""
    with _LOCK:
        path = store_path(settings)
        servers = load_store(path)
        sink = SecretSink(read_env_file(env_path(settings)))
        parsed = parse_pasted(text, sink)
        taken = set(reserved)
        result = AddResult([], {}, [], {})
        for server in parsed:
            if server.name in taken:
                result.skipped[server.name] = (
                    "a server with this name is switched on in config.yaml; use another name"
                )
                continue
            if server.name.startswith("claude_ai_"):
                result.skipped[server.name] = "names starting with claude_ai_ are reserved"
                continue
            if server.name in servers and not replace:
                result.skipped[server.name] = "already added"
                continue
            old = servers.get(server.name, {})
            servers[server.name] = {
                "config": server.config,
                "enabled": bool(old.get("enabled", False)),
                "tools": old.get("tools", []),
                "added_at": old.get("added_at", utcnow()),
                "tested_at": None,
                "last_error": None,
            }
            result.added.append(server.name)
            result.notes[server.name] = server.notes
        if result.added:
            result.env_vars = sink.write(env_path(settings))
            save_store(path, servers)
        return result


def update(settings: Settings, name: str, **changes: Any) -> dict[str, Any]:
    with _LOCK:
        path = store_path(settings)
        servers = load_store(path)
        if name not in servers:
            raise ConnectorError(f"No connector named {name!r}.")
        servers[name].update(changes)
        save_store(path, servers)
        return servers[name]


def set_tool_mode(settings: Settings, name: str, tool: str | None, mode: str) -> dict[str, Any]:
    """Set one tool's mode, or every tool's when ``tool`` is None."""
    if mode not in TOOL_MODES:
        raise ConnectorError(f"Unknown mode {mode!r}.")
    with _LOCK:
        path = store_path(settings)
        servers = load_store(path)
        entry = servers.get(name)
        if entry is None:
            raise ConnectorError(f"No connector named {name!r}.")
        hit = False
        for t in entry.get("tools") or []:
            if tool is None or t.get("name") == tool:
                t["mode"] = mode
                hit = True
        if tool is not None and not hit:
            raise ConnectorError(f"{name} has no tool {tool!r}; test the connection again.")
        save_store(path, servers)
        return entry


def remove(settings: Settings, name: str) -> list[str]:
    """Delete a window server, and from ``.env`` the secrets this window stored for it
    (``SPARK_MCP_*``) that nothing else refers to. Returns the variables removed."""
    with _LOCK:
        path = store_path(settings)
        servers = load_store(path)
        entry = servers.pop(name, None)
        if entry is None:
            raise ConnectorError(f"No connector named {name!r}.")
        save_store(path, servers)
        mine = _all_vars(entry.get("config"))
        if not mine:
            return []
        from spark.brain.mcp_config import McpConfigError, read_mcp_json

        try:
            hand_written = read_mcp_json(settings.path(settings.connectors.mcp_config))
        except McpConfigError:
            return []  # can't tell what the file refers to: keep every value
        in_use = _all_vars(hand_written) | _all_vars([e.get("config") for e in servers.values()])
        unused = sorted(v for v in mine - in_use if v.startswith("SPARK_MCP_"))
        if unused:
            remove_env_vars(env_path(settings), unused)
        return unused


def _all_vars(value: Any) -> set[str]:
    required, optional = referenced_vars(value)
    return required | optional


def record_test(
    settings: Settings, name: str, tools: list[dict[str, Any]] | None, error: str | None
) -> dict[str, Any]:
    """Store a test result: the merged tool list on success, the error otherwise."""
    with _LOCK:
        path = store_path(settings)
        servers = load_store(path)
        entry = servers.get(name)
        if entry is None:
            raise ConnectorError(f"No connector named {name!r}.")
        entry["tested_at"] = utcnow()
        entry["last_error"] = error
        if tools is not None:
            entry["tools"] = merge_tools(entry.get("tools") or [], tools)
        save_store(path, servers)
        return entry


def claude_desktop_config_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """Claude Desktop's config file on this machine, if there is one."""
    env = os.environ if environ is None else environ
    candidates: list[Path] = []
    if appdata := env.get("APPDATA"):
        candidates.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
    if local := env.get("LOCALAPPDATA"):  # the Microsoft Store build keeps it in its package
        packages = Path(local) / "Packages"
        if packages.is_dir():
            candidates += sorted(packages.glob("Claude_*/LocalCache/Roaming/Claude/*.json"))
    home = Path.home()
    candidates.append(home / "Library/Application Support/Claude/claude_desktop_config.json")
    candidates.append(home / ".config/Claude/claude_desktop_config.json")
    for path in candidates:
        if path.name == "claude_desktop_config.json" and path.is_file():
            return path
    return None
