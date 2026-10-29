"""Where the desktop app keeps its files (spec §10 Phase 4, "User data").

Run from source, the app uses the repo's ``config.yaml`` like the CLI does. Installed (frozen by
PyInstaller), it uses a per-user folder seeded from the defaults bundled with the app:

- Windows: ``%LOCALAPPDATA%\\HeySpark`` (Local, not Roaming: the models are ~200 MB)
- macOS: ``~/Library/Application Support/HeySpark``
- other: ``$XDG_CONFIG_HOME/heyspark`` (``~/.config/heyspark``)

Seeding never overwrites what the user owns (``config.yaml``, ``mcp_servers.json``, ``.env``).
Files under ``assets/`` belong to the app: they're refreshed when the bundled copy differs, and
files the user added there are left alone.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

APP_NAME = "HeySpark"
# macOS bundle identifier (packaging/hey_spark.spec puts it in Info.plist)
BUNDLE_ID = "com.contextuai.heyspark"

# Copied once; the user edits them afterwards.
USER_FILES = ("config.yaml", "mcp_servers.json", ".env.example")
# Refreshed from the bundle when they differ (wake-word model, earcons).
APP_DIRS = ("assets",)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_root() -> Path:
    """The repo root when running from source."""
    return Path(__file__).resolve().parents[2]


def bundled_defaults() -> Path:
    """The folder holding the default config and assets shipped with the app."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "defaults"
    return source_root()


def user_data_dir(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    platform = platform or sys.platform
    env = os.environ if environ is None else environ
    home = home or Path.home()
    if platform == "win32":
        base = env.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / APP_NAME
    if platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    base = env.get("XDG_CONFIG_HOME")
    return (Path(base) if base else home / ".config") / APP_NAME.lower()


def seed(target: Path, defaults: Path) -> list[Path]:
    """Copy the bundled defaults into ``target``. Returns the files written."""
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in USER_FILES:
        src, dst = defaults / name, target / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            written.append(dst)
    for name in APP_DIRS:
        root = defaults / name
        if not root.is_dir():
            continue
        for src in root.rglob("*"):
            if not src.is_file():
                continue
            dst = target / src.relative_to(defaults)
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            written.append(dst)
    return written


def resolve_config(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The ``config.yaml`` the desktop app should load, seeding the user folder if frozen."""
    if explicit is not None:
        return Path(explicit)
    if env := os.environ.get("SPARK_CONFIG") or os.environ.get("CHIP_CONFIG"):
        return Path(env)
    if not is_frozen():
        return source_root() / "config.yaml"
    target = user_data_dir()
    seed(target, bundled_defaults())
    return target / "config.yaml"
