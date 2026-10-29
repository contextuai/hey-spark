"""macOS glue for the desktop app: theme, error alerts and the PATH of an app opened from Finder.

An app started from Finder or the Dock gets launchd's bare ``PATH`` (``/usr/bin:/bin:...``), not the
user's shell ``PATH``, so ``claude``, ``npx``/``node`` (stdio MCP servers), ``git`` and ``gh``
installed with Homebrew, npm or the Claude installer would all look missing. :func:`fix_gui_path`
asks the user's login shell for its ``PATH`` once at startup and adds the usual install folders.

Everything here shells out to stock macOS tools (``defaults``, ``osascript``), so no extra
dependencies. Each function is best effort and never raises.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

SHELL_TIMEOUT_S = 5.0
_MARKER = "__SPARK_PATH__="

# Where Homebrew, npm, the Claude installer and friends put their commands.
COMMON_BIN_DIRS = (
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "~/.local/bin",
    "~/.claude/local",
    "~/.npm-global/bin",
    "~/.bun/bin",
    "~/.cargo/bin",
)

Run = Callable[..., subprocess.CompletedProcess[str]]


def prefers_dark(run: Run = subprocess.run) -> bool:
    """True when the system appearance is Dark (``AppleInterfaceStyle`` is unset in Light)."""
    try:
        out = run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip().lower() == "dark"


def applescript_string(text: str) -> str:
    """``text`` as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def show_alert(title: str, message: str, run: Run = subprocess.run) -> None:
    """A modal error alert, for failures before the window exists."""
    script = (
        f"display alert {applescript_string(title)} message {applescript_string(message)} "
        "as critical"
    )
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        run(["osascript", "-e", script], capture_output=True, timeout=120, check=False)


def login_shell_path(
    environ: Mapping[str, str] | None = None, run: Run = subprocess.run
) -> str | None:
    """The ``PATH`` the user's interactive login shell sets up, or None if it can't be read."""
    env = os.environ if environ is None else environ
    shell = env.get("SHELL") or "/bin/zsh"
    try:
        out = run(
            [shell, "-ilc", f'printf "%s%s" "{_MARKER}" "$PATH"'],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SHELL_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("couldn't read the login shell PATH from %s: %s", shell, exc)
        return None
    # rc files may print banners; the marker finds our line among them
    _, found, value = out.stdout.rpartition(_MARKER)
    value = value.strip()
    return value if found and value else None


def merged_path(
    current: str, shell_path: str | None, *, home: Path, extra: Sequence[str] = COMMON_BIN_DIRS
) -> str:
    """``shell_path``, then ``current``, then the ``extra`` folders that exist; no repeats."""
    kept = [p for p in current.split(os.pathsep) if p]
    shell = [p for p in (shell_path or "").split(os.pathsep) if p]
    extras = [str(home / d[2:]) if d.startswith("~/") else d for d in extra]
    parts: list[str] = []
    for part in [*shell, *kept, *extras]:
        if part not in parts and (part in kept or Path(part).is_dir()):
            parts.append(part)
    return os.pathsep.join(parts)


def fix_gui_path(environ: dict[str, str] | None = None, run: Run = subprocess.run) -> str:
    """Give an app opened from Finder the user's shell ``PATH``. Returns the new ``PATH``."""
    env = os.environ if environ is None else environ
    new = merged_path(env.get("PATH", ""), login_shell_path(env, run), home=Path.home())
    env["PATH"] = new
    log.info("PATH for child processes: %s", new)
    return new
