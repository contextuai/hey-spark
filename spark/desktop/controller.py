"""Drives :class:`spark.app.SparkApp` for the desktop window (spec §10 Phase 4, "Wiring").

The window talks to one :class:`DesktopController`. It owns a private asyncio loop on a
background thread: every PortAudio stream is opened on that thread (the Windows WASAPI gotcha),
and pywebview keeps the main thread. The pipeline doesn't know the UI exists; the controller
only subscribes to ``on_state``, taps the transcript log and wraps the mic to measure its level.

Everything the UI needs arrives through ``emit(event)`` as a JSON-able dict:

- ``{"type": "phase", "phase": "stopped" | "starting" | "running" | "error", "detail": str}``
- ``{"type": "state", "state": "idle" | "listening" | ...}`` (a :class:`State` value). A
  ``follow_up`` (awake) state also carries ``awake_s``: seconds left before Spark sleeps.
- ``{"type": "ignored", "reason": str}`` (speech heard while awake wasn't for Spark)
- ``{"type": "muted", "muted": bool}`` (the mic button or "mute" by voice, P5.7k)
- ``{"type": "connectors", "status": {name: {"status": str, "error"?: str}}}`` (live MCP status
  changed; the window says when a connector became unreachable)
- ``{"type": "model", "mode": "auto" | "fast" | "deep", "fast_model", "deep_model",
  "fast_label", "deep_label"}`` (the model choice changed, in the window or by voice)
- ``{"type": "transcript", "role": "user" | "assistant", "text": str}``
- ``{"type": "level", "level": float}`` (0..1, about 15 per second while running)
- ``{"type": "error", "message": str}`` (a turn failed; the app keeps running)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import Future
from typing import Any

import numpy as np

from spark.app import SparkApp, State
from spark.config import Settings
from spark.interfaces import AudioInput, BackendUnavailable
from spark.prefs import (
    MODES,
    Prefs,
    apply_prefs,
    model_choices,
    model_info,
    pref_updates,
    set_route_models,
)

log = logging.getLogger(__name__)

Event = dict[str, Any]
Emit = Callable[[Event], None]
AppFactory = Callable[[Settings], SparkApp]
ModelCheck = Callable[[Settings, Callable[[str], None]], None]

LEVEL_INTERVAL_S = 1 / 15
# int16 RMS that shows as a full ring; normal speech at arm's length sits around 1000-3000.
LEVEL_FULL_SCALE = 4000.0


def ensure_models(settings: Settings, say: Callable[[str], None]) -> None:
    """Download the configured speech models if any are missing (first run)."""
    from spark.models import download_models, model_status
    from spark.persona import DEFAULT_VOICE, voice_by_id, voice_id

    status = model_status(settings)
    missing = [k for k, s in status.items() if s["required"] and not s["present"]]
    if not missing:
        return
    sizes = {"whisper": 150, "piper": 60, "kokoro": 340}
    total = sum(sizes.get(k, 0) for k in missing)
    say(f"Downloading speech models (about {total} MB)…" if total else "Downloading speech models…")
    vid = voice_id(settings)
    voice = voice_by_id(vid)
    if vid == DEFAULT_VOICE:
        kokoro_line = "Downloading the default voice, George (about 340 MB)…"
    elif voice is not None:
        kokoro_line = f"Downloading the voice, {voice.label.split(' (')[0]} (about 340 MB)…"
    else:
        kokoro_line = "Downloading the Kokoro voice (about 340 MB)…"

    class _Say:
        """download_models narrates in CLI terms; the window shows one plain line per model."""

        def print(self, msg: object, *args: Any, **kwargs: Any) -> None:
            text = str(msg).lower()
            if "whisper" in text and "download" in text:
                say("Downloading speech recognition (about 150 MB)…")
            elif "piper" in text and "download" in text:
                say("Downloading the voice (about 60 MB)…")
            elif "kokoro" in text and "download" in text:
                say(kokoro_line)

    download_models(
        settings,
        whisper="whisper" in missing,
        piper="piper" in missing,
        kokoro="kokoro" in missing,
        console=_Say(),
    )


def init_audio() -> None:
    """Initialise PortAudio on the calling thread (the controller's loop thread).

    ``import sounddevice`` runs ``Pa_Initialize``, and on Windows WASAPI keeps per-thread COM
    state from that call. Left to happen lazily, the first import lands on the short-lived
    ``asyncio.to_thread`` worker that builds the backends, and the speaker then fails to open
    ("WdmSyncIoctl ... GLE = 0x490") on the loop thread, sometimes or every time.
    """
    try:
        import sounddevice  # noqa: F401
    except Exception:  # not installed, or no PortAudio: the backends report it properly later
        log.debug("sounddevice unavailable at launch", exc_info=True)


def friendly_error(exc: BaseException) -> str:
    """One line the window can show, with the fix when we know it."""
    if isinstance(exc, BackendUnavailable):
        return f"{exc.reason}. Fix: {exc.fix}"
    spoken = getattr(exc, "spoken", None)  # the OpenAI-compatible brain's errors (P5.7e)
    if isinstance(spoken, str) and spoken and type(exc).__name__ == "ProviderError":
        return spoken
    text = str(exc) or type(exc).__name__
    low = text.lower()
    if any(k in low for k in ("login", "authenticat", "unauthorized", "401", "api key")):
        return (
            "Claude didn't accept the sign-in. Run claude in a terminal and type /login, or add an "
            "Anthropic API key in the Brain window."
        )
    if "mic" in low or "portaudio" in low or "input device" in low:
        return f"Can't open the microphone ({text}). Check it's connected and allowed."
    return text.splitlines()[0][:300]


def level_of(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(block.astype(np.float32)))))
    return min(1.0, rms / LEVEL_FULL_SCALE)


class LevelMic(AudioInput):
    """Passes the mic through unchanged and reports its level, throttled."""

    def __init__(
        self,
        inner: AudioInput,
        on_level: Callable[[float], None],
        *,
        interval_s: float = LEVEL_INTERVAL_S,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        self.sample_rate = inner.sample_rate
        self.on_level = on_level
        self.interval_s = interval_s
        self._now = now
        self._next = 0.0
        self._peak = 0.0

    async def start(self) -> None:
        await self.inner.start()

    async def stop(self) -> None:
        await self.inner.stop()
        self.on_level(0.0)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        async for block in self.inner.frames():
            self._peak = max(self._peak, level_of(block))
            t = self._now()
            if t >= self._next:
                self._next = t + self.interval_s
                self.on_level(self._peak)
                self._peak = 0.0
            yield block

    def __getattr__(self, name: str) -> Any:  # recent(), is_open, ... on the real mic
        return getattr(self.inner, name)


class TranscriptTap:
    """Forwards to the real transcript log and tells the window."""

    def __init__(self, inner: Any, on_line: Callable[[str, str], None]) -> None:
        self.inner = inner
        self.on_line = on_line

    def user(self, text: str) -> None:
        self.inner.user(text)
        self.on_line("user", text)

    def assistant(self, text: str) -> None:
        self.inner.assistant(text)
        self.on_line("assistant", text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class DesktopController:
    def __init__(
        self,
        settings: Settings,
        emit: Emit,
        *,
        app_factory: AppFactory | None = None,
        model_check: ModelCheck | None = ensure_models,
        level_interval_s: float = LEVEL_INTERVAL_S,
    ) -> None:
        self.settings = settings
        self.emit = emit
        self.app_factory: AppFactory = app_factory or SparkApp.from_settings
        self.model_check = model_check
        self.level_interval_s = level_interval_s
        self.app: SparkApp | None = None
        self.prefs = Prefs.from_settings(settings)
        self.connector_status: dict[str, dict[str, str]] = {}  # live MCP status (P5.7i)
        self.muted = False  # P5.7k: kept across Stop/Start, not across restarts
        # config.yaml's brain section before prefs: the Brain window (P5.7f) resets to it and
        # applies prefs again, so a removed pref falls back to config.yaml.
        self.config_brain = settings.brain.model_copy(deep=True)
        apply_prefs(settings, self.prefs)  # so the window shows the saved model choice at once
        self.phase = "stopped"
        self.detail = ""
        self._run_task: asyncio.Task[None] | None = None
        self._errors_seen = 0
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # --- thread API (called from the window) ------------------------------------------------

    def launch(self) -> None:
        """Start the controller's event loop thread."""
        if self._thread is not None:
            return
        ready = threading.Event()

        def main() -> None:
            init_audio()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()
            loop.close()

        self._thread = threading.Thread(target=main, name="spark-loop", daemon=True)
        self._thread.start()
        ready.wait()

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        if self._loop is None:
            self.launch()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start(self) -> Future[None]:
        return self._submit(self.astart())

    def stop(self) -> Future[None]:
        return self._submit(self.astop())

    def set_muted(self, muted: bool) -> Future[None]:
        return self._submit(self.aset_muted(muted))

    def set_mode(self, mode: str) -> Future[None]:
        return self._submit(self.aset_mode(mode))

    def set_models(
        self, fast: str | None, deep: str | None, worker: str | None = None
    ) -> Future[None]:
        return self._submit(self.aset_models(fast, deep, worker))

    def reload(self) -> Future[None]:
        """Rebuild the pipeline so config changes (connectors) take effect."""
        return self._submit(self.areload())

    def apply_brain(self) -> Future[str | None]:
        """Rebuild with the brain now in ``settings`` (the Brain window). The future's result is
        None, or what went wrong in words."""
        return self._submit(self.aapply_brain())

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop, release everything and end the loop thread. Blocks up to ``timeout``."""
        if self._loop is None:
            return
        with contextlib.suppress(Exception):
            self._submit(self.aclose()).result(timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout)
        self._loop = self._thread = None

    def snapshot(self) -> Event:
        running = self.app is not None and self.phase == "running"
        state = self.app.state.value if running and self.app is not None else None
        app = self.settings.app
        snap: Event = {
            "phase": self.phase,
            "state": state,
            "name": app.name,
            "wake": app.wake_prompt,
        }
        if running and self.app is not None and (left := self.app.awake_remaining()) is not None:
            snap["awake_s"] = round(left, 1)
        snap["muted"] = self.muted
        snap["model"] = self.model_info()
        snap["model_choices"] = self.model_choices()
        return snap

    def _brain_label(self) -> str:
        if self.settings.brain.backend == "openai_compat":
            from spark.brain.openai_presets import resolve_provider

            return resolve_provider(self.settings.brain.openai_compat).label
        return "Claude"

    def model_info(self) -> Event:
        if self.app is not None:
            return self.app.model_info()
        return model_info(self.settings)

    def model_choices(self) -> list[list[str]]:
        """``[[id, label], ...]`` for the window, including the configured ids."""
        brain = self.app.brain if self.app is not None else None
        return model_choices(self.settings, getattr(brain, "available_models", ()) or ())

    def _state_event(self, state: State) -> Event:
        event: Event = {"type": "state", "state": state.value}
        if state is State.FOLLOW_UP and self.app is not None:
            left = self.app.awake_remaining()
            if left is not None:
                event["awake_s"] = round(left, 1)
        return event

    # --- coroutines (run on the controller loop; tests await them directly) ----------------

    def _guard(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _phase(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = detail
        self.emit({"type": "phase", "phase": phase, "detail": detail})

    async def astart(self) -> None:
        async with self._guard():
            if self.phase == "running":
                return
            try:
                if self.app is None:
                    self.app = await self._build()
                self._begin_run()
            except Exception as exc:
                log.exception("start failed")
                await self._discard_app()
                self._phase("error", friendly_error(exc))

    async def _build(self) -> SparkApp:
        def say(msg: str) -> None:
            self._phase("starting", msg)

        say("Checking speech models…")
        if self.model_check is not None:
            await asyncio.to_thread(self.model_check, self.settings, say)
        say("Loading models…")
        app = await asyncio.to_thread(self.app_factory, self.settings)
        app.on_state = self._on_state
        app.on_ignored = self._on_ignored
        app.on_model = self._on_model
        app.on_mute = self._on_mute
        if self.muted:
            app.set_muted(True, source="window")
        if hasattr(app.brain, "on_mcp_status"):
            app.brain.on_mcp_status = self._on_mcp_status
        app.transcripts = TranscriptTap(app.transcripts, self._on_line)  # type: ignore[assignment]
        if app.mic is not None:
            app.mic = LevelMic(app.mic, self._on_level, interval_s=self.level_interval_s)
        try:
            say(f"Connecting to {self._brain_label()}…")
            await app.warmup()
        except BaseException:
            with contextlib.suppress(Exception):
                await app.close()
            raise
        self._errors_seen = len(app.errors)
        return app

    def _begin_run(self) -> None:
        assert self.app is not None
        if self.app.state is not State.IDLE:
            self.app.set_state(State.IDLE)
        self._run_task = asyncio.create_task(self.app.run())
        self._run_task.add_done_callback(self._run_ended)
        self._phase("running")
        self.emit(self._state_event(self.app.state))

    def _run_ended(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or self._run_task is not task:
            return
        self._run_task = None
        if (exc := task.exception()) is not None:
            log.error("mic loop failed: %s", exc, exc_info=exc)
            self._phase("error", friendly_error(exc))
        elif self.phase == "running":
            self._phase("stopped", "The microphone stream ended.")

    async def astop(self) -> None:
        async with self._guard():
            task, self._run_task = self._run_task, None
            if self.app is not None:
                await self.app.cancel_turn()
                with contextlib.suppress(Exception):
                    await self.app.stop()
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(task, 5.0)
            if self.app is not None and self.app.state is not State.IDLE:
                self.app.set_state(State.IDLE)
            if self.phase != "error" or task is not None:
                self._phase("stopped")

    async def aset_muted(self, muted: bool) -> None:
        muted = bool(muted)
        if self.app is not None:
            self.app.set_muted(muted, source="window")  # reports back through _on_mute
        if self.muted != muted:
            self._on_mute(muted)

    async def aset_mode(self, mode: str) -> None:
        if mode not in MODES:
            return
        if self.app is not None:
            self.app.set_model_mode(mode)  # saves and notifies through _on_model
            return
        self.settings.brain.mode = mode
        self.prefs.update(model_mode=mode)
        self._on_model(self.model_info())

    async def aset_models(
        self, fast: str | None, deep: str | None, worker: str | None = None
    ) -> None:
        fast = (fast or "").strip() or None
        deep = (deep or "").strip() or None
        worker = None if worker is None else worker.strip()  # '' = same as fast
        if self.app is not None:
            self.app.set_models(fast, deep, worker)
            return
        set_route_models(self.settings, fast, deep, worker)
        self.prefs.update(**pref_updates(self.settings, fast, deep, worker))
        self._on_model(self.model_info())

    async def areload(self) -> None:
        """Drop the built app (it holds the brain's MCP setup); start again if it was running."""
        was_running = self.phase == "running"
        await self.aclose()
        if was_running:
            await self.astart()

    async def aapply_brain(self) -> str | None:
        """Like :meth:`areload`, but report whether the new brain works: if Spark was running,
        whether it started again; if not, whether the brain can be built at all (a missing key,
        an unknown provider). Tells the window about the new models either way."""
        was_running = self.phase == "running"
        await self.aclose()
        error: str | None = None
        if was_running:
            await self.astart()
            if self.phase != "running":
                error = self.detail or "Spark couldn't start with the new brain."
        else:
            from spark import registry

            try:
                brain = await asyncio.to_thread(registry.build, "brain", self.settings)
            except Exception as exc:
                log.info("brain settings: the new brain can't be built: %s", exc)
                error = friendly_error(exc)
            else:
                with contextlib.suppress(Exception):
                    await brain.close()
        self._on_model(self.model_info(), choices=True)
        return error

    async def aclose(self) -> None:
        await self.astop()
        async with self._guard():
            await self._discard_app()

    async def _discard_app(self) -> None:
        app, self.app = self.app, None
        if app is not None:
            with contextlib.suppress(Exception):
                await app.close()

    # --- SparkApp hooks (called on the controller loop) --------------------------------------

    def _on_state(self, _old: State, new: State) -> None:
        if self.phase == "running":
            self.emit(self._state_event(new))
        app = self.app
        if app is not None and len(app.errors) > self._errors_seen:
            self._errors_seen = len(app.errors)
            self.emit({"type": "error", "message": friendly_error(app.errors[-1])})

    def _on_mute(self, muted: bool) -> None:
        self.muted = muted
        self.emit({"type": "muted", "muted": muted})
        if muted:
            self.emit({"type": "level", "level": 0.0})

    def _on_mcp_status(self, status: dict[str, dict[str, str]]) -> None:
        self.connector_status = status
        self.emit({"type": "connectors", "status": status})

    def _on_model(self, info: Event, *, choices: bool = False) -> None:
        event: Event = {"type": "model", **info}
        if choices:  # the brain changed: the pickers need the new provider's models
            event["choices"] = self.model_choices()
        self.emit(event)

    def _on_ignored(self, _text: str, verdict: Any) -> None:
        self.emit({"type": "ignored", "reason": getattr(verdict, "reason", "")})

    def _on_line(self, role: str, text: str) -> None:
        self.emit({"type": "transcript", "role": role, "text": text})

    def _on_level(self, level: float) -> None:
        self.emit({"type": "level", "level": 0.0 if self.muted else round(level, 3)})
