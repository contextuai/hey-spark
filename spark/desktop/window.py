"""The desktop window: pywebview around ``ui/index.html`` (spec §10 Phase 4).

``python -m spark desktop`` (or the installed app) lands in :func:`main`. pywebview owns the main
thread; :class:`~spark.desktop.controller.DesktopController` runs the pipeline on its own loop.
Events from the controller never touch the web view directly: they go through
:class:`EventPump`, whose thread batches them and collapses mic-level updates, so a slow
``run_js`` can't stall the audio loop.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import sys
import threading
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from spark.config import Settings

log = logging.getLogger(__name__)

UI_FILE = Path(__file__).parent / "ui" / "index.html"
CONNECTORS_FILE = UI_FILE.parent / "connectors.html"
SHORTCUTS_FILE = UI_FILE.parent / "shortcuts.html"
BRAIN_FILE = UI_FILE.parent / "brain.html"
VOICE_FILE = UI_FILE.parent / "voice.html"
WIDTH, HEIGHT = 360, 560
CONNECTORS_WIDTH, CONNECTORS_HEIGHT = 940, 660
SHORTCUTS_WIDTH, SHORTCUTS_HEIGHT = 900, 680
BRAIN_WIDTH, BRAIN_HEIGHT = 900, 720
VOICE_WIDTH, VOICE_HEIGHT = 720, 760


class EventPump:
    """Delivers controller events to the page in batches, on its own thread.

    Only the newest ``level`` event of a batch is kept. Events that arrive before the page is
    ready are held (levels dropped) and flushed by :meth:`ready`.
    """

    def __init__(self, send: Callable[[list[dict[str, Any]]], None]) -> None:
        self._send = send
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="spark-ui-events", daemon=True)
        self._thread.start()

    def put(self, event: dict[str, Any]) -> None:
        self._q.put(event)

    def ready(self) -> None:
        self._ready.set()
        self._q.put({"type": "noop"})  # wake the thread to flush what was held

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(2)

    @staticmethod
    def coalesce(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = [e for e in events if e["type"] not in ("level", "noop")]
        levels = [e for e in events if e["type"] == "level"]
        return [*out, levels[-1]] if levels else out

    def _loop(self) -> None:
        held: list[dict[str, Any]] = []
        while True:
            first = self._q.get()
            if first is None:
                return
            batch = [first]
            while True:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._flush(held + batch)
                    return
                batch.append(nxt)
            if not self._ready.is_set():
                held += [e for e in batch if e["type"] not in ("level", "noop")]
                continue
            self._flush(held + batch)
            held = []

    def _flush(self, events: list[dict[str, Any]]) -> None:
        events = self.coalesce(events)
        if not events:
            return
        try:
            self._send(events)
        except Exception:  # the window may be closing
            log.debug("could not deliver UI events", exc_info=True)


class Bridge:
    """Methods the page calls as ``pywebview.api.<name>()``.

    pywebview exposes every public attribute, so all state stays in underscore names.
    """

    def __init__(
        self,
        controller: Any,
        pump: EventPump,
        open_connectors: Callable[[], None] | None = None,
        open_shortcuts: Callable[[], None] | None = None,
        open_brain: Callable[[], None] | None = None,
        open_voice: Callable[[], None] | None = None,
    ) -> None:
        self._controller = controller
        self._pump = pump
        self._open_connectors = open_connectors
        self._open_shortcuts = open_shortcuts
        self._open_brain = open_brain
        self._open_voice = open_voice

    def voice(self) -> None:
        """Open (or bring back) the Personality & Voice window."""
        log.info("ui: voice settings opened")
        if self._open_voice is not None:
            self._open_voice()

    def brain(self) -> None:
        """Open (or bring back) the Brain window (provider, key, models)."""
        log.info("ui: brain settings opened")
        if self._open_brain is not None:
            self._open_brain()

    def connectors(self) -> None:
        """Open (or bring back) the Connectors window."""
        log.info("ui: connectors opened")
        if self._open_connectors is not None:
            self._open_connectors()

    def shortcuts(self) -> None:
        """Open (or bring back) the Shortcuts window."""
        log.info("ui: shortcuts opened")
        if self._open_shortcuts is not None:
            self._open_shortcuts()

    def start(self) -> None:
        log.info("ui: start pressed")
        self._controller.start()

    def stop(self) -> None:
        log.info("ui: stop pressed")
        self._controller.stop()

    def set_muted(self, muted: bool) -> None:
        """Stop or resume listening (the mic button, Ctrl+M)."""
        log.info("ui: %s", "mute" if muted else "unmute")
        self._controller.set_muted(bool(muted))

    def set_mode(self, mode: str) -> None:
        """Auto / always fast / always deep."""
        log.info("ui: model mode %s", mode)
        self._controller.set_mode(str(mode))

    def set_models(self, fast: str, deep: str, worker: str = "") -> None:
        """The model behind each route; ``worker`` '' = the researcher uses the fast model."""
        log.info("ui: models fast=%s deep=%s researcher=%s", fast, deep, worker or "(fast)")
        self._controller.set_models(str(fast or ""), str(deep or ""), str(worker or ""))

    def snapshot(self) -> dict[str, Any]:
        """Called once the page has loaded: the current state, then live events."""
        snap = self._controller.snapshot()
        log.info("ui: page connected (%s)", snap["phase"])
        self._pump.ready()
        return snap


# --- process setup ---------------------------------------------------------------------------


def _ensure_std_streams() -> None:
    """A windowed (no-console) build has ``sys.stdout``/``stderr`` set to None."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def setup_file_logging(settings: Settings) -> Path:
    from spark.desktop.paths import is_frozen
    from spark.util.logging import _SPARK_HANDLER_ATTR, SecretRedactingFilter, setup_logging

    setup_logging(settings, console=not is_frozen())
    path = settings.log_path / settings.desktop.log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=settings.logging.max_bytes,
        backupCount=settings.logging.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactingFilter())
    setattr(handler, _SPARK_HANDLER_ATTR, True)
    logging.getLogger().addHandler(handler)
    # Model downloads log every HTTP request with long signed URLs; keep the file readable.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Native crashes (WebView2, PortAudio, onnxruntime) bypass logging: dump their stacks too.
    import faulthandler

    global _fault_file
    _fault_file = open(path.with_name("crash.log"), "a", encoding="utf-8")  # noqa: SIM115
    faulthandler.enable(_fault_file)
    return path


_fault_file: Any = None


def prefers_dark() -> bool:
    """The system theme, so the window doesn't flash white before the page paints."""
    if sys.platform == "darwin":
        from spark.desktop import macos

        return macos.prefers_dark()
    if sys.platform != "win32":
        return False
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        with key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hey-spark", description="Spark desktop window")
    p.add_argument("--config", help="path to config.yaml (default: per-user folder when installed)")
    p.add_argument("--debug", action="store_true", help="enable the web inspector")
    # for the build scripts: check the frozen bundle without opening a window
    p.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return p


# Native-heavy modules the frozen app needs; a missing library shows up here, not on a user's Mac.
SELF_TEST_MODULES = (
    "numpy",
    "onnxruntime",
    "ctranslate2",
    "faster_whisper",
    "av",
    "piper",
    "openwakeword",
    "sounddevice",
    "webview",
    "claude_agent_sdk",
    "kokoro_onnx",
)


def self_test() -> int:
    """Import every native dependency and check the bundled files. Prints a report; 0 = pass."""
    import importlib

    from spark.desktop.paths import bundled_defaults

    failures: list[str] = []
    for name in SELF_TEST_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # any import-time failure (ImportError, OSError from dlopen)
            failures.append(f"import {name}: {type(exc).__name__}: {exc}")
    if sys.platform == "darwin":
        from spark.audio import mic_access

        # Reads the permission without asking; "unknown" means AVFoundation didn't load.
        access = mic_access.status()
        print(f"microphone access: {access}")
        if access == "unknown":
            failures.append("microphone permission check: AVFoundation unavailable")
    defaults = bundled_defaults()
    for path in (
        UI_FILE,
        defaults / "config.yaml",
        defaults / "assets" / "wakewords" / "hey_spark.onnx",
    ):
        if not path.is_file():
            failures.append(f"missing {path}")
    try:  # Kokoro reads text through the bundled espeak-ng library and its data
        from kokoro_onnx.tokenizer import Tokenizer

        if not Tokenizer().phonemize("Good evening.", "en-gb"):
            failures.append("kokoro: espeak-ng returned no phonemes")
    except Exception as exc:
        failures.append(f"kokoro: {type(exc).__name__}: {exc}")
    for line in failures:
        print(f"FAIL {line}")
    print("self-test " + ("failed" if failures else f"ok ({len(SELF_TEST_MODULES)} modules)"))
    return 1 if failures else 0


def show_fatal(title: str, message: str) -> None:
    """A windowed build has no console: tell the user in a message box (Windows, macOS)."""
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)  # MB_ICONERROR
    elif sys.platform == "darwin":
        from spark.desktop import macos

        macos.show_alert(title, message)


def main(argv: list[str] | None = None) -> int:
    _ensure_std_streams()
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        return _run(args)
    except Exception as exc:
        log.exception("desktop app crashed")
        show_fatal(
            "Hey Spark", f"Hey Spark couldn't start:\n\n{exc}\n\nDetails are in the log folder."
        )
        return 1


def _run(args: argparse.Namespace) -> int:

    from spark.config import load_settings
    from spark.desktop.controller import DesktopController
    from spark.desktop.paths import APP_NAME, is_frozen, resolve_config
    from spark.desktop.single import SingleInstance

    config = resolve_config(args.config)
    settings = load_settings(config)
    title = settings.app.name
    # fixed lock id (the installer waits on this mutex); the window is found by its title
    instance = SingleInstance(APP_NAME, window_title=title)
    if not instance.acquire():
        instance.focus_existing()
        return 0
    log_path = setup_file_logging(settings)
    log.info("desktop app starting; config %s; log %s", config, log_path)
    if sys.platform == "darwin" and is_frozen():
        from spark.desktop import macos

        macos.fix_gui_path()  # opened from Finder: find claude, npx, git, gh

    try:
        import webview
    except ImportError:
        log.exception("pywebview missing")
        show_fatal("Hey Spark", "The desktop window needs pywebview: uv sync --extra desktop")
        return 1

    window: Any = None

    def send(events: list[dict[str, Any]]) -> None:
        if window is not None:
            window.run_js(f"window.spark && window.spark.apply({json.dumps(events)})")

    pump = EventPump(send)
    controller = DesktopController(settings, pump.put)
    controller.launch()
    connectors_window: Any = None

    def open_connectors() -> None:
        nonlocal connectors_window
        if connectors_window is not None:
            connectors_window.restore()
            connectors_window.show()
            return
        from spark.desktop.connectors import ConnectorsApi

        api = ConnectorsApi(
            settings,
            on_apply=controller.reload,
            is_running=lambda: controller.phase == "running",
            live_status=lambda: controller.connector_status,
        )
        connectors_window = webview.create_window(
            f"{title} · Connectors",
            url=CONNECTORS_FILE.as_uri(),
            js_api=api,
            width=CONNECTORS_WIDTH,
            height=CONNECTORS_HEIGHT,
            min_size=(720, 480),
            background_color="#111318" if prefers_dark() else "#F6F7F9",
        )

        def closed() -> None:
            nonlocal connectors_window
            connectors_window = None

        connectors_window.events.closed += closed

    shortcuts_window: Any = None

    def open_shortcuts() -> None:
        nonlocal shortcuts_window
        if shortcuts_window is not None:
            shortcuts_window.restore()
            shortcuts_window.show()
            return
        from spark.desktop.shortcuts import ShortcutsApi

        shortcuts_window = webview.create_window(
            f"{title} · Shortcuts",
            url=SHORTCUTS_FILE.as_uri(),
            js_api=ShortcutsApi(settings),
            width=SHORTCUTS_WIDTH,
            height=SHORTCUTS_HEIGHT,
            min_size=(700, 480),
            background_color="#111318" if prefers_dark() else "#F6F7F9",
        )

        def closed() -> None:
            nonlocal shortcuts_window
            shortcuts_window = None

        shortcuts_window.events.closed += closed

    brain_window: Any = None

    def open_brain() -> None:
        nonlocal brain_window
        if brain_window is not None:
            brain_window.restore()
            brain_window.show()
            return
        from spark.desktop.brain_settings import APPLY_TIMEOUT_S, BrainSettingsApi

        def apply() -> str | None:
            import concurrent.futures

            try:
                return controller.apply_brain().result(APPLY_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                return "Spark is taking a long time to start. Check the main window."

        brain_window = webview.create_window(
            f"{title} · Brain",
            url=BRAIN_FILE.as_uri(),
            js_api=BrainSettingsApi(settings, baseline=controller.config_brain, on_apply=apply),
            width=BRAIN_WIDTH,
            height=BRAIN_HEIGHT,
            min_size=(640, 520),
            background_color="#111318" if prefers_dark() else "#F6F7F9",
        )

        def closed() -> None:
            nonlocal brain_window
            brain_window = None

        brain_window.events.closed += closed

    voice_window: Any = None

    def open_voice() -> None:
        nonlocal voice_window
        if voice_window is not None:
            voice_window.restore()
            voice_window.show()
            return
        from spark.desktop.brain_settings import APPLY_TIMEOUT_S
        from spark.desktop.voice_settings import VoiceSettingsApi

        def apply() -> str | None:
            import concurrent.futures

            try:
                controller.reload().result(APPLY_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                return "Spark is taking a long time to start. Check the main window."
            if controller.phase == "error":
                return controller.detail or "Spark couldn't start with the new voice."
            return None

        voice_window = webview.create_window(
            f"{title} · Personality & Voice",
            url=VOICE_FILE.as_uri(),
            js_api=VoiceSettingsApi(settings, on_apply=apply),
            width=VOICE_WIDTH,
            height=VOICE_HEIGHT,
            min_size=(520, 520),
            background_color="#111318" if prefers_dark() else "#F6F7F9",
        )

        def closed() -> None:
            nonlocal voice_window
            voice_window = None

        voice_window.events.closed += closed

    window = webview.create_window(
        title,
        url=UI_FILE.as_uri(),
        js_api=Bridge(controller, pump, open_connectors, open_shortcuts, open_brain, open_voice),
        width=WIDTH,
        height=HEIGHT,
        min_size=(320, 480),
        background_color="#111318" if prefers_dark() else "#F6F7F9",
    )

    def main_closed() -> None:
        for extra in (connectors_window, shortcuts_window, brain_window, voice_window):
            if extra is not None:
                with contextlib.suppress(Exception):
                    extra.destroy()

    window.events.closed += main_closed

    def on_started() -> None:
        if settings.desktop.autostart_listening:
            controller.start()

    icon = UI_FILE.parent / ("icon.ico" if sys.platform == "win32" else "icon.png")
    try:
        webview.start(on_started, debug=args.debug, icon=str(icon) if icon.exists() else None)
    finally:
        log.info("window closed; shutting down")
        controller.shutdown()
        pump.close()
        instance.release()
    return 0
