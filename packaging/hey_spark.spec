# PyInstaller spec for the Hey Spark desktop app (spec §10 Phase 4, "Installer").
#
#   uv run --group build pyinstaller packaging/hey_spark.spec --noconfirm --distpath build/dist --workpath build/work
#
# One-folder, windowed build: build/dist/HeySpark/ on Windows, build/dist/HeySpark.app on macOS
# (Apple Silicon, macOS 14+, the oldest onnxruntime supports). Ships only the local backends
# (openWakeWord, Silero via faster-whisper, faster-whisper, Piper, Kokoro). The proprietary Claude CLI
# that claude-agent-sdk bundles is left out: the app uses the Claude Code the user installs
# (PATH or brain.cli_path).
# Speech models are NOT bundled: the first Start downloads them into the user's data folder.
# ruff: noqa
import os
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).parent  # noqa: F821 - defined by PyInstaller
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

# Defaults seeded into the per-user folder on first run (spark/desktop/paths.py).
defaults = [
    (str(ROOT / "config.yaml"), "defaults"),
    (str(ROOT / "mcp_servers.json"), "defaults"),
    (str(ROOT / ".env.example"), "defaults"),
    (str(ROOT / "assets" / "sounds"), "defaults/assets/sounds"),
    (str(ROOT / "assets" / "wakewords" / "hey_spark.onnx"), "defaults/assets/wakewords"),
    (str(ROOT / "assets" / "wakewords" / "README.txt"), "defaults/assets/wakewords"),
]

# License texts at the bundle root (the _internal folder). The wake-word model's own license
# travels with the model, so it is seeded into the user's assets/wakewords folder too.
notices = [
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(ROOT / "LICENSES" / "GPL-3.0.txt"), "LICENSES"),  # piper-tts, espeak-ng, FFmpeg parts
    (str(ROOT / "LICENSES" / "LGPL-2.1.txt"), "LICENSES"),  # FFmpeg (through PyAV)
    (str(ROOT / "assets" / "wakewords" / "LICENSE"), "defaults/assets/wakewords"),
]

datas = [
    *defaults,
    *notices,
    (str(ROOT / "spark" / "desktop" / "ui"), "spark/desktop/ui"),
    *collect_data_files("faster_whisper"),  # Silero VAD ONNX
    *collect_data_files("piper"),  # espeak-ng data
    *collect_data_files("openwakeword"),
    *collect_data_files("claude_agent_sdk", excludes=["_bundled"]),  # not the bundled CLI
    *collect_data_files("webview"),  # js bridge + platform libs
    # Kokoro voices (the butler's): its vocab config, espeak-ng's data, phonemizer's tables, and
    # the package metadata both read at import time.
    *collect_data_files("kokoro_onnx"),
    *collect_data_files("espeakng_loader"),
    *collect_data_files("phonemizer"),
    *copy_metadata("kokoro-onnx"),
    *copy_metadata("phonemizer"),
]

binaries = collect_dynamic_libs("espeakng_loader")  # espeak-ng.dll / libespeak-ng.dylib

# The registry imports backends from "module:Class" strings, so PyInstaller can't see them.
hiddenimports = [
    *collect_submodules("spark", filter=lambda m: not m.startswith("spark.tests")),
    *collect_submodules("webview"),
    *(["clr_loader", "pythonnet"] if IS_WIN else []),  # WinForms/WebView2 backend
    *(["objc"] if IS_MAC else []),  # spark.audio.mic_access loads AVFoundation through it
    "kokoro_onnx",
]

excludes = [
    # optional/cloud backends are source-only (spec §10 Phase 4, "Bundled backends")
    "deepgram",
    "elevenlabs",
    "pvporcupine",
    "torch",
    # never needed at runtime
    "tkinter",
    "matplotlib",
    "IPython",
    "pytest",
    "PyInstaller",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "spark" / "desktop" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

if IS_WIN:
    icon = str(ROOT / "packaging" / "windows" / "hey_spark.ico")
elif IS_MAC:
    icon = str(ROOT / "packaging" / "macos" / "hey_spark.icns")
else:
    icon = None
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HeySpark",
    console=os.environ.get("SPARK_CONSOLE") == "1",  # set to debug a build
    icon=icon,
    upx=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    name="HeySpark",
    upx=False,
)

if IS_MAC:
    app = BUNDLE(  # noqa: F821
        coll,
        name="HeySpark.app",
        icon=icon,
        bundle_identifier="com.contextuai.heyspark",  # spark/desktop/paths.py BUNDLE_ID
        version=VERSION,
        info_plist={
            "CFBundleName": "HeySpark",
            "CFBundleDisplayName": "Hey Spark",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "14.0",
            "LSApplicationCategoryType": "public.app-category.productivity",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Copyright (c) 2026 Nagendra Rao Jogi. MIT License.",
            # Without these macOS silently feeds the app zeros (mic) or refuses (Terminal).
            "NSMicrophoneUsageDescription": (
                "Hey Spark listens for its wake word and your request. Audio stays on this Mac "
                "unless you pick a cloud speech service."
            ),
            "NSAppleEventsUsageDescription": (
                "Hey Spark opens Terminal to start Claude Code in a new project folder."
            ),
        },
    )
