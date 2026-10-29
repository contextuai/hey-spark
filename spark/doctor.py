"""``python -m spark doctor``: environment checks with copy-pasteable fixes (spec §2).

Every check returns a :class:`Check` and never raises; the doctor prints them all and exits
non-zero when a *required* check fails. Optional checks print a warning (``!``) only.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from spark.config import Settings

MIN_FREE_GB = 2.0
BRAIN_TIMEOUT_S = 45.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    required: bool = True

    @property
    def symbol(self) -> str:
        if self.ok:
            return "✔"
        return "✖" if self.required else "!"


def _fail(name: str, detail: str, fix: str, *, required: bool = True) -> Check:
    return Check(name, False, detail, fix, required)


def _from_unavailable(name: str, exc: Exception, *, required: bool = True) -> Check:
    from spark.interfaces import BackendUnavailable

    if isinstance(exc, BackendUnavailable):
        return _fail(name, exc.reason, exc.fix, required=required)
    return _fail(name, f"{type(exc).__name__}: {exc}", "see the error above", required=required)


# --- individual checks -------------------------------------------------------------------------


def check_python(settings: Settings) -> Check:
    v = sys.version_info
    detail = f"{platform.python_implementation()} {v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return Check("python", True, detail)
    return _fail("python", detail + " (need 3.11+)", "uv python install 3.11 && uv sync")


def check_uv_env(settings: Settings) -> Check:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    detail = sys.prefix
    if in_venv and shutil.which("uv"):
        return Check("uv environment", True, detail)
    if not shutil.which("uv"):
        return _fail(
            "uv environment",
            "uv not found on PATH",
            "install uv: https://docs.astral.sh/uv/getting-started/installation/",
            required=False,
        )
    return _fail(
        "uv environment",
        f"not running in the project venv ({detail})",
        "uv sync && uv run python -m spark doctor",
        required=False,
    )


def check_dotenv(settings: Settings) -> Check:
    path = settings.config_dir / ".env"
    if path.exists():
        return Check(".env", True, str(path))
    return _fail(".env", "missing", "copy .env.example .env  (then fill in keys)", required=False)


def check_mic(settings: Settings) -> Check:
    from spark.audio.stream import mic_fix, probe_input

    try:
        info = probe_input(settings, seconds=0.3)
    except Exception as exc:
        return _from_unavailable("microphone", exc)
    rate = f"{info['samplerate']} Hz" + (" (resampled to 16 kHz)" if info["resampled"] else "")
    detail = f"{info['name']} via {info['hostapi']}, {rate}"
    if info["samples"] == 0:
        return _fail(
            "microphone",
            detail + ", but no audio was captured",
            mic_fix(),
        )
    return Check("microphone", True, detail)


def check_speaker(settings: Settings) -> Check:
    try:
        import sounddevice as sd
    except Exception as exc:
        return _fail("speaker", f"sounddevice unavailable ({exc})", "uv sync")
    try:
        from spark.audio.stream import resolve_device

        device = resolve_device(sd, "output", settings.audio.output_device, settings.audio.host_api)
        index = device if device is not None else sd.default.device[1]
        info = sd.query_devices(index)
        if int(info["max_output_channels"]) < 1:
            raise RuntimeError(f"{info['name']} has no output channels")
        sd.check_output_settings(
            device=index, channels=1, dtype="int16", samplerate=info["default_samplerate"]
        )
    except Exception as exc:
        return _fail(
            "speaker",
            str(exc),
            getattr(exc, "fix", "")
            or "plug in / enable an output device, or set audio.output_device in config.yaml",
        )
    return Check("speaker", True, f"{info['name']} @ {int(info['default_samplerate'])} Hz")


def check_disk(settings: Settings) -> Check:
    path = settings.models_path
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    detail = f"{free_gb:.1f} GB free on the drive holding {path}"
    if free_gb >= MIN_FREE_GB:
        return Check("disk space", True, detail)
    return _fail(
        "disk space",
        detail,
        f"free up at least {MIN_FREE_GB:g} GB or set models_dir to a bigger drive",
    )


def check_models(settings: Settings) -> list[Check]:
    from spark.models import model_status

    try:
        status = model_status(settings)
    except Exception as exc:
        return [_fail("models", f"could not inspect models_dir ({exc})", "check models_dir")]
    checks = []
    fix = "uv run python -m spark models download"
    for name, info in status.items():
        needed = info.get("required", True)
        if info.get("present"):
            checks.append(Check(f"model: {name}", True, str(info.get("path", ""))))
        else:
            model_fix = f"{fix} --kokoro" if name == "kokoro" else fix
            checks.append(
                _fail(
                    f"model: {name}",
                    f"missing ({info.get('path', '')})",
                    model_fix,
                    required=needed,
                )
            )
    return checks


def check_wakeword(settings: Settings) -> Check:
    from spark import registry

    name = settings.wakeword.backend
    try:
        detector = registry.build("wakeword", settings)
    except Exception as exc:
        return _from_unavailable(f"wake word ({name})", exc)
    try:
        detail = f"ready, {detector.frame_length}-sample frames"
    finally:
        detector.close()
    return Check(f"wake word ({name})", True, detail)


def _bundled_cli() -> Path | None:
    """The CLI inside the claude-agent-sdk wheel. Source installs have it; the frozen app
    leaves it out (packaging/hey_spark.spec)."""
    spec = importlib.util.find_spec("claude_agent_sdk")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    for candidate in (root / "_bundled").glob("claude*"):
        if candidate.is_file() and candidate.name in ("claude", "claude.exe"):
            return candidate
    return None


def _native_exe(path: str) -> bool:
    return path.lower().endswith((".exe", ".com"))


def find_claude_cli(settings: Settings) -> str | None:
    """The ``claude`` CLI the Claude brain will run, or None.

    Same order as claude-agent-sdk 0.2.174 (``SubprocessCLITransport._find_cli``):
    ``brain.cli_path``, the SDK's bundled binary, ``claude`` on PATH (on Windows only a native
    ``.exe``; the SDK refuses npm's ``claude.cmd`` shim), then the native installer's folder.
    """
    if settings.brain.cli_path:
        return settings.brain.cli_path
    bundled = _bundled_cli()
    if bundled:
        return str(bundled)
    if sys.platform == "win32":
        for name in ("claude", "claude.exe"):
            found = shutil.which(name)
            if found and _native_exe(found):
                return found
        locations = [Path.home() / ".local" / "bin" / "claude.exe"]
    else:
        found = shutil.which("claude")
        if found:
            return found
        locations = [
            Path.home() / ".npm-global" / "bin" / "claude",
            Path("/opt/homebrew/bin/claude"),  # Homebrew on Apple Silicon (not on a GUI app's PATH)
            Path("/usr/local/bin/claude"),
            Path.home() / ".local" / "bin" / "claude",
            Path.home() / "node_modules" / ".bin" / "claude",
            Path.home() / ".yarn" / "bin" / "claude",
            Path.home() / ".claude" / "local" / "claude",
        ]
    for path in locations:
        if path.is_file():
            return str(path)
    return None


def check_claude_cli(settings: Settings) -> Check:
    # Only the Claude brain needs it; on openai_compat a missing CLI is just a note.
    from spark.brain.claude_agent import INSTALL_CLAUDE_CODE

    required = settings.brain.backend == "claude_agent"
    path = find_claude_cli(settings)
    if path is None:
        return _fail(
            "claude CLI", "Claude Code isn't installed", INSTALL_CLAUDE_CODE, required=required
        )
    bundled = _bundled_cli()
    if bundled and not settings.brain.cli_path:
        return Check(
            "claude CLI", True, f"bundled with claude-agent-sdk: {bundled}", required=required
        )
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        version = (out.stdout or out.stderr).strip().splitlines()[0] if out else ""
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        version = f"could not run ({exc})"
    return Check("claude CLI", True, f"{path}: {version}", required=required)


async def check_brain(settings: Settings) -> Check:
    timeout = BRAIN_TIMEOUT_S
    from spark import registry
    from spark.brain.openai_presets import resolved_models

    name = f"brain ({settings.brain.backend}, {resolved_models(settings)[0]})"
    try:
        brain = registry.build("brain", settings)
    except Exception as exc:
        return _from_unavailable(name, exc)
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        async with asyncio.timeout(timeout):
            await brain.warmup()
            connected = loop.time() - start
            reply = ""
            async for delta in brain.respond("Reply with just the word OK."):
                reply += delta
    except TimeoutError:
        return _fail(
            name,
            f"no answer within {timeout:.0f} s",
            "check network/proxy access to api.anthropic.com and that Claude Code is signed in "
            "(or ANTHROPIC_API_KEY is set)"
            if settings.brain.backend == "claude_agent"
            else "check that the model server is running and reachable",
        )
    except Exception as exc:
        check = _from_unavailable(name, exc)
        if settings.brain.backend == "claude_agent" and (
            not check.fix or check.fix == "see the error above"
        ):
            check.fix = "run claude and /login, or set ANTHROPIC_API_KEY in .env"
        return check
    finally:
        with contextlib.suppress(Exception):
            await brain.close()
    total = loop.time() - start
    return Check(name, bool(reply.strip()), f"connect {connected:.1f} s, reply {total:.1f} s")


OPENAI_MODELS_TIMEOUT_S = 8.0


async def check_openai_compat(
    settings: Settings, *, transport: Any | None = None, environ: Mapping[str, str] | None = None
) -> list[Check]:
    """``brain.backend: openai_compat``: the preset resolves, the key's env var is set (never
    printed), ``GET /models`` answers, and the configured models are listed when the endpoint
    lists any. Nothing (an empty list) for other backends."""
    if settings.brain.backend != "openai_compat":
        return []
    from spark.brain.openai_presets import resolve_provider

    prov = resolve_provider(settings.brain.openai_compat)
    name = f"openai_compat ({prov.name})"
    if prov.problems:
        return [
            _fail(
                name,
                "; ".join(prov.problems),
                "fix it in the Brain window, or edit brain.openai_compat in config.yaml",
            )
        ]
    checks = [Check(name, True, f"{prov.label} at {prov.base_url}")]
    env = os.environ if environ is None else environ
    missing = prov.missing_env(env)
    if missing:
        checks.append(
            _fail(f"{name}: key", f"{', '.join(missing)} not set", f"put {missing[0]}=... in .env")
        )
        return checks
    if prov.key_env:
        state = "set" if prov.api_key(env) else "not set (optional here)"
        checks.append(Check(f"{name}: key", True, f"{prov.key_env} {state}"))
    from spark.brain.endpoint_check import fetch_models
    from spark.brain.openai_chat import ChatClient, ProviderError

    chat = ChatClient(prov, settings.brain.openai_compat, transport=transport, environ=env)
    try:
        models = await fetch_models(chat, timeout_s=OPENAI_MODELS_TIMEOUT_S)
    except ProviderError as exc:
        detail = str(exc)
        fix = (
            f"start {prov.label} (e.g. `ollama serve`) or fix base_url"
            if prov.local
            else "check base_url, the key and the network"
        )
        checks.append(_fail(f"{name}: endpoint", detail, fix))
        return checks
    finally:
        await chat.aclose()
    checks.append(Check(f"{name}: endpoint", True, f"GET /models listed {len(models)} models"))
    if prov.name == "ollama" and not models:
        checks.append(
            _fail(f"{name}: models", "Ollama has no models", f"ollama pull {prov.fast_model}")
        )
        return checks
    for role, model in (("fast", prov.fast_model), ("deep", prov.deep_model)):
        if not models or model in models:
            checks.append(Check(f"{name}: {role} model", True, str(model)))
        else:
            fix = (
                f"ollama pull {model}"
                if prov.name == "ollama"
                else f"set brain.openai_compat.{role}_model to one the endpoint lists"
            )
            checks.append(_fail(f"{name}: {role} model", f"{model} is not listed", fix))
    from spark.brain.mcp_config import claudeai_enabled

    skipped = claudeai_enabled(settings)
    if skipped:
        checks.append(
            _fail(
                f"{name}: claude.ai connectors",
                f"{', '.join(skipped)} need the Claude brain; skipped on openai_compat",
                "use brain.backend: claude_agent for them, or disable them in config.yaml",
                required=False,
            )
        )
    return checks


OPTIONAL_BACKENDS = {
    "openwakeword": ("openwakeword", "uv sync --extra oww"),
    "deepgram": ("deepgram", "uv sync --extra deepgram"),
    "kokoro": ("kokoro_onnx", "uv sync --extra kokoro"),
    "elevenlabs": ("elevenlabs", "uv sync --extra elevenlabs"),
}


def check_optional_backends(settings: Settings) -> list[Check]:
    selected = {settings.wakeword.backend, settings.stt.backend, settings.tts.backend}
    checks = []
    for backend, (module, fix) in OPTIONAL_BACKENDS.items():
        present = importlib.util.find_spec(module) is not None
        required = backend in selected
        if present:
            checks.append(Check(f"optional: {backend}", True, "installed", required=required))
        else:
            detail = "selected in config but not installed" if required else "not installed"
            checks.append(_fail(f"optional: {backend}", detail, fix, required=required))
    return checks


def check_npx(settings: Settings) -> Check:
    path = shutil.which("npx") or shutil.which("npx.cmd")
    if path:
        return Check("npx (Phase 2 MCP servers)", True, path, required=False)
    return _fail(
        "npx (Phase 2 MCP servers)",
        "not on PATH",
        "install Node.js LTS (https://nodejs.org) so stdio MCP servers can run",
        required=False,
    )


# --- connectors (Phase 2) ----------------------------------------------------------------------


def check_claudeai(settings: Settings) -> Check:
    """claude.ai account connectors: on or off, which claude_ai_* servers are configured, and
    what the CLI reported for them last time Spark ran (no live call here)."""
    from spark.brain import claudeai
    from spark.brain.mcp_config import claudeai_enabled

    name = "claude.ai connectors"
    if settings.connectors.block_claudeai_connectors:
        return Check(name, True, "off (block_claudeai_connectors: true)", required=False)
    enabled = claudeai_enabled(settings)
    if not enabled:
        return Check(name, True, "off (no claude_ai_* server enabled)", required=False)
    seen = claudeai.load_state(settings)
    parts, fixes = [], []
    for key in enabled:
        status = seen.get(key, {}).get("status")
        if status is None:
            parts.append(f"{key} (not seen yet)")
        else:
            parts.append(f"{key} ({status})")
            if status == "needs-auth":
                fixes.append(key)
    hidden = sorted(set(seen) - set(enabled))
    detail = "on: " + ", ".join(parts)
    if hidden:
        detail += f"; hidden: {', '.join(hidden)}"
    if fixes:
        return _fail(
            name,
            detail,
            f"{claudeai.SIGN_IN_HINT} ({claudeai.SETTINGS_URL}) for {', '.join(fixes)}",
            required=False,
        )
    missing = [k for k in enabled if k not in seen]
    if seen and missing:
        return _fail(
            name,
            detail,
            f"connect {', '.join(claudeai.display_name(k) for k in missing)} on "
            f"{claudeai.SETTINGS_URL}, or turn it off in config.yaml",
            required=False,
        )
    return Check(name, True, detail, required=False)


MCP_PROBE_TIMEOUT_S = 25.0
CONNECTOR_FIX = "set the variables in .env; see docs/connectors.md (what a human must do)"


def _expand_all(value: Any, environ: Mapping[str, str]) -> Any:
    """Expand ``${VAR}`` for the doctor's own probe. Values are used in-process, never printed."""
    from spark.brain.mcp_config import expand_vars

    if isinstance(value, str):
        return expand_vars(value, environ)
    if isinstance(value, list):
        return [_expand_all(v, environ) for v in value]
    if isinstance(value, dict):
        return {k: _expand_all(v, environ) for k, v in value.items()}
    return value


async def _list_tools(read: Any, write: Any) -> list[str]:
    from mcp.client.session import ClientSession

    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.list_tools()
        return [tool.name for tool in result.tools]


async def probe_mcp_server(cfg: Mapping[str, Any], environ: Mapping[str, str]) -> list[str]:
    """Connect to one MCP server (stdio spawn, http, or sse) and return its tool names."""
    cfg = _expand_all(dict(cfg), environ)
    kind = cfg.get("type", "stdio")
    if kind == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=cfg["command"], args=list(cfg.get("args", [])), env=cfg.get("env") or None
        )
        # the server stderr needs a real file handle; opening devnull never blocks
        with open(os.devnull, "w", encoding="utf-8") as errlog:  # noqa: ASYNC230
            async with stdio_client(params, errlog=errlog) as streams:
                return await _list_tools(streams[0], streams[1])
    if kind == "http":
        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        headers = dict(cfg.get("headers") or {})
        async with (
            httpx2.AsyncClient(headers=headers, timeout=MCP_PROBE_TIMEOUT_S) as client,
            streamable_http_client(cfg["url"], http_client=client) as streams,
        ):
            return await _list_tools(streams[0], streams[1])
    if kind == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(cfg["url"], headers=dict(cfg.get("headers") or {})) as streams:
            return await _list_tools(streams[0], streams[1])
    raise ValueError(f"unsupported MCP server type {kind!r}")


def _probe_error(exc: BaseException) -> str:
    """A short reason that can't leak secrets (the redaction filter also masks env values)."""
    from spark.util.logging import redact

    if isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return redact(f"{type(exc).__name__}: {exc}")[:200]


async def check_connectors(
    settings: Settings, *, environ: Mapping[str, str] | None = None
) -> list[Check]:
    """Each configured MCP server: env vars set, binary/npx resolvable, and it lists its tools."""
    from spark.brain.mcp_config import load_mcp_servers

    env = os.environ if environ is None else environ
    # httpx logs full request URLs at INFO; MCP URLs can carry server/user ids
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        setup = load_mcp_servers(settings, environ=env)
    except Exception as exc:
        return [_fail("connectors", f"mcp_servers.json unreadable ({exc})", "fix the JSON syntax")]
    checks: list[Check] = []
    configured = settings.connectors.servers
    for name, reason in sorted(setup.disabled.items()):
        if name.startswith("claude_ai_"):
            continue  # check_claudeai reports these
        if name in configured and not configured[name].enabled:
            checks.append(
                Check(f"connector: {name}", True, "disabled in config.yaml", required=False)
            )
        else:
            checks.append(_fail(f"connector: {name}", reason, CONNECTOR_FIX, required=False))
    for name, cfg in sorted(setup.servers.items()):
        if not isinstance(cfg, Mapping) or cfg.get("type") == "sdk":
            continue
        try:
            async with asyncio.timeout(MCP_PROBE_TIMEOUT_S):
                tools = await probe_mcp_server(cfg, env)
        except TimeoutError:
            checks.append(
                _fail(
                    f"connector: {name}",
                    f"no answer within {MCP_PROBE_TIMEOUT_S:.0f} s",
                    "check the URL/binary and network; run the server by hand to see its errors",
                    required=False,
                )
            )
            continue
        except Exception as exc:
            checks.append(
                _fail(f"connector: {name}", _probe_error(exc), CONNECTOR_FIX, required=False)
            )
            continue
        checks.append(
            Check(f"connector: {name}", True, f"{cfg.get('type', 'stdio')}, {len(tools)} tools")
        )
    if not checks:
        checks.append(Check("connectors", True, "none configured", required=False))
    return checks


# --- runner ------------------------------------------------------------------------------------

SyncCheck = Callable[["Settings"], "Check | list[Check]"]
AsyncCheck = Callable[["Settings"], Awaitable["Check | list[Check]"]]


def default_checks(*, audio: bool = True, brain: bool = True) -> list[SyncCheck | AsyncCheck]:
    checks: list[SyncCheck | AsyncCheck] = [check_python, check_uv_env, check_dotenv]
    if audio:
        checks += [check_mic, check_speaker]
    checks += [check_disk, check_models]
    if audio:
        checks.append(check_wakeword)
    checks += [check_claude_cli, check_openai_compat]
    if brain:
        checks.append(check_brain)
    checks += [check_optional_backends, check_npx, check_claudeai, check_connectors]
    return checks


async def run_checks(
    settings: Settings, checks: list[SyncCheck | AsyncCheck] | None = None
) -> list[Check]:
    """Run every check; an unexpected exception becomes a failed check, never a crash."""
    results: list[Check] = []
    for fn in checks if checks is not None else default_checks():
        name = getattr(fn, "__name__", "check").removeprefix("check_").replace("_", " ")
        try:
            out: Any = fn(settings)
            if asyncio.iscoroutine(out):
                out = await out
        except Exception as exc:
            out = _fail(name, f"check crashed: {type(exc).__name__}: {exc}", "report this bug")
        results.extend(out if isinstance(out, list) else [out])
    return results


def render(results: list[Check], console: Console) -> None:
    from rich.markup import escape

    from spark.util.logging import redact

    for c in results:
        color = "green" if c.ok else ("red" if c.required else "yellow")
        line = f"[{color}]{c.symbol}[/] [bold]{escape(c.name)}[/]"
        if c.detail:
            line += f"  {escape(redact(c.detail))}"
        console.print(line)
        if not c.ok and c.fix:
            console.print(f"    fix: {escape(redact(c.fix))}")


def exit_code(results: list[Check]) -> int:
    return 1 if any(c.required and not c.ok for c in results) else 0


def main(
    settings: Settings,
    *,
    console: Console | None = None,
    checks: list[SyncCheck | AsyncCheck] | None = None,
) -> int:
    from rich.console import Console

    console = console or Console(highlight=False)
    console.print(f"[bold]Spark doctor[/]  ({platform.system()} {platform.release()}, {os.name})")
    # Check the brain Spark will really use: the app's choices (the Brain window, the model
    # pickers) in prefs.json override config.yaml.
    from spark.prefs import Prefs, apply_prefs

    apply_prefs(settings, Prefs.from_settings(settings))
    results = asyncio.run(run_checks(settings, checks))
    render(results, console)
    code = exit_code(results)
    failed = sum(1 for c in results if c.required and not c.ok)
    warned = sum(1 for c in results if not c.required and not c.ok)
    if code == 0:
        console.print(f"[green]All required checks passed[/] ({warned} warning(s)).")
    else:
        console.print(f"[red]{failed} required check(s) failed[/], {warned} warning(s).")
    return code
