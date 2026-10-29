"""Desktop app without a window: user-data paths and the controller (spec §10 Phase 4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from spark.app import SparkApp, State
from spark.desktop import controller as ctl
from spark.desktop import paths
from spark.desktop.controller import DesktopController, LevelMic, friendly_error, level_of
from spark.interfaces import BackendUnavailable
from spark.shortcuts import DEFAULT_SHORTCUTS
from tests.fakes import (
    FakeMic,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWakeWord,
    MockBrain,
    concat,
    make_settings,
    silence,
    tone,
)

# --- paths -----------------------------------------------------------------------------------


def test_user_data_dir_per_platform(tmp_path):
    env = {"LOCALAPPDATA": str(tmp_path / "Local")}
    assert paths.user_data_dir(platform="win32", environ=env) == tmp_path / "Local" / "HeySpark"
    assert (
        paths.user_data_dir(platform="darwin", environ={}, home=tmp_path)
        == tmp_path / "Library" / "Application Support" / "HeySpark"
    )
    assert (
        paths.user_data_dir(platform="linux", environ={}, home=tmp_path)
        == tmp_path / ".config" / "heyspark"
    )


def _defaults(root: Path) -> Path:
    (root / "assets" / "sounds").mkdir(parents=True)
    (root / "config.yaml").write_text("app:\n  name: Hey Spark\n")
    (root / "mcp_servers.json").write_text("{}")
    (root / "assets" / "sounds" / "wake.wav").write_bytes(b"v1")
    return root


def test_seed_copies_defaults_once_and_keeps_user_edits(tmp_path):
    defaults, target = _defaults(tmp_path / "bundle"), tmp_path / "user"
    written = paths.seed(target, defaults)
    assert {p.name for p in written} == {"config.yaml", "mcp_servers.json", "wake.wav"}
    (target / "config.yaml").write_text("app:\n  name: Mine\n")
    (target / "assets" / "sounds" / "custom.wav").write_bytes(b"mine")

    assert paths.seed(target, defaults) == []
    assert (target / "config.yaml").read_text() == "app:\n  name: Mine\n"
    assert (target / "assets" / "sounds" / "custom.wav").read_bytes() == b"mine"


def test_seed_refreshes_app_assets_when_the_bundle_changes(tmp_path):
    defaults, target = _defaults(tmp_path / "bundle"), tmp_path / "user"
    paths.seed(target, defaults)
    (defaults / "assets" / "sounds" / "wake.wav").write_bytes(b"v2")
    assert [p.name for p in paths.seed(target, defaults)] == ["wake.wav"]
    assert (target / "assets" / "sounds" / "wake.wav").read_bytes() == b"v2"


def test_resolve_config_from_source_uses_the_repo(monkeypatch):
    monkeypatch.delenv("SPARK_CONFIG", raising=False)
    assert paths.resolve_config() == paths.source_root() / "config.yaml"
    assert paths.resolve_config("x.yaml") == Path("x.yaml")


def test_resolve_config_frozen_seeds_the_user_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARK_CONFIG", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "bundled_defaults", lambda: _defaults(tmp_path / "bundle"))
    monkeypatch.setattr(paths, "user_data_dir", lambda: tmp_path / "user")
    assert paths.resolve_config() == tmp_path / "user" / "config.yaml"
    assert (tmp_path / "user" / "config.yaml").is_file()


def test_the_repo_ships_every_default_the_installer_bundles():
    root = paths.source_root()
    for name in paths.USER_FILES:
        assert (root / name).is_file(), name
    assert (root / "assets" / "wakewords" / "hey_spark.onnx").is_file()


# --- helpers ---------------------------------------------------------------------------------


def test_level_of_and_friendly_errors():
    assert level_of(np.zeros(512, dtype=np.int16)) == 0.0
    assert level_of(np.full(512, 20000, dtype=np.int16)) == 1.0
    assert "Fix: uv sync" in friendly_error(BackendUnavailable("oww", "not installed", "uv sync"))
    assert "Anthropic API key" in friendly_error(
        RuntimeError("Invalid API key · Please run /login")
    )
    assert friendly_error(RuntimeError("boom\ntrace")) == "boom"


async def test_level_mic_throttles_and_passes_audio_through():
    t = [0.0]
    levels: list[float] = []
    inner = FakeMic([tone(0.032), silence(0.032), tone(0.032)])
    mic = LevelMic(inner, levels.append, interval_s=0.05, now=lambda: t[0])
    await mic.start()
    blocks = []
    async for b in mic.frames():
        blocks.append(b)
        t[0] += 0.032
    assert len(blocks) == 3
    assert len(levels) == 2 and levels[0] > 0  # block 1 reported, block 2 throttled
    await mic.stop()
    assert levels[-1] == 0.0 and inner.stopped


# --- controller ------------------------------------------------------------------------------


class _Built:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.apps: list[SparkApp] = []

    def __call__(self, settings) -> SparkApp:
        app = SparkApp(
            settings,
            mic=FakeMic(hold_open=True),
            player=FakePlayer(),
            wakeword=FakeWakeWord(),
            vad=FakeVAD(),
            stt=FakeSTT(transcripts=["what is two plus two"]),
            tts=FakeTTS(),
            brain=MockBrain(),
        )
        self.apps.append(app)
        return app


def _settings(tmp_path):
    return make_settings(
        logging={"dir": str(tmp_path / "logs")},
        debug_audio={"enabled": False, "dir": str(tmp_path / "dbg")},
    )


def _controller(tmp_path, **kw):
    events: list[dict] = []
    factory = kw.pop("app_factory", None) or _Built(tmp_path)
    c = DesktopController(
        _settings(tmp_path),
        events.append,
        app_factory=factory,
        model_check=None,
        level_interval_s=0.0,
        **kw,
    )
    return c, events, factory


async def _until(events, pred):
    async with asyncio.timeout(5.0):
        while not any(pred(e) for e in events):  # noqa: ASYNC110 - events is a plain list
            await asyncio.sleep(0.01)


async def test_start_runs_the_app_and_reports_a_full_turn(tmp_path):
    c, events, built = _controller(tmp_path)
    await c.astart()
    assert c.phase == "running"
    assert {"type": "state", "state": "idle"} in events
    app = built.apps[0]
    assert isinstance(app.mic, LevelMic)

    app.wakeword.trigger()
    app.mic.inner.push(concat(silence(0.032), tone(0.6), silence(0.8)))
    await _until(events, lambda e: e.get("state") == "follow_up")
    states = [e["state"] for e in events if e["type"] == "state"]
    assert states[:5] == ["idle", "listening", "thinking", "speaking", "follow_up"]
    lines = [(e["role"], e["text"]) for e in events if e["type"] == "transcript"]
    assert lines == [("user", "what is two plus two"), ("assistant", "Two plus two is four.")]
    assert any(e["type"] == "level" and e["level"] > 0 for e in events)
    assert c.snapshot()["state"] == "follow_up"
    await c.aclose()


async def test_stop_releases_the_mic_and_start_reuses_the_loaded_app(tmp_path):
    c, events, built = _controller(tmp_path)
    await c.astart()
    app = built.apps[0]
    await c.astop()
    assert c.phase == "stopped" and app.mic.inner.stopped
    snap = c.snapshot()
    model, choices = snap.pop("model"), snap.pop("model_choices")
    assert snap.pop("muted") is False
    assert snap == {"phase": "stopped", "state": None, "name": "Spark", "wake": "Spark"}
    assert model["mode"] == "auto" and model["deep_label"] == "Opus 5.5"
    assert ["claude-opus-5-5", "Opus 5.5 (most capable)"] in choices

    await c.astart()
    assert c.phase == "running" and len(built.apps) == 1
    await asyncio.sleep(0.01)  # run() opens the mic in its own task
    assert app.mic.inner.started and not app.mic.inner.stopped
    app.wakeword.trigger()
    app.mic.inner.push(silence(0.032))
    await _until(events, lambda e: e.get("state") == "listening")
    await c.aclose()
    assert c.app is None and c.phase == "stopped"


async def test_stop_mid_turn_cancels_it_and_returns_to_idle(tmp_path):
    c, _events, built = _controller(tmp_path)
    await c.astart()
    app = built.apps[0]
    app.wakeword.trigger()
    app.mic.inner.push(silence(0.032))
    await asyncio.sleep(0.05)
    assert app.state is State.LISTENING
    await c.astop()
    assert app.state is State.IDLE
    await c.aclose()


async def test_build_failure_shows_the_fix_and_can_retry(tmp_path):
    calls = []

    def broken(settings):
        calls.append(1)
        raise BackendUnavailable("openwakeword", "model file missing", "put hey_spark.onnx back")

    c, events, _ = _controller(tmp_path, app_factory=broken)
    await c.astart()
    assert c.phase == "error"
    assert events[-1]["detail"] == "model file missing. Fix: put hey_spark.onnx back"
    await c.astart()
    assert len(calls) == 2


async def test_first_start_checks_models_and_shows_progress(tmp_path):
    seen = []

    def check(settings, say):
        seen.append(settings)
        say("Downloading speech models (about 200 MB)…")

    c, events, _ = _controller(tmp_path)
    c.model_check = check
    await c.astart()
    assert len(seen) == 1
    details = [e["detail"] for e in events if e["type"] == "phase" and e["phase"] == "starting"]
    assert "Downloading speech models (about 200 MB)…" in details
    await c.aclose()


async def test_a_failed_turn_is_reported_but_keeps_running(tmp_path):
    class Boom(FakeSTT):
        async def transcribe(self, pcm):
            raise RuntimeError("stt exploded")

    def factory(settings):
        app = _Built(tmp_path)(settings)
        app.stt = Boom()
        return app

    c, events, _ = _controller(tmp_path, app_factory=factory)
    await c.astart()
    c.app.wakeword.trigger()
    c.app.mic.inner.push(concat(silence(0.032), tone(0.6), silence(0.8)))
    await _until(events, lambda e: e["type"] == "error")
    assert {"type": "error", "message": "stt exploded"} in events
    assert c.phase == "running"
    await c.aclose()


def test_thread_api_starts_stops_and_shuts_down(tmp_path):
    c, _events, built = _controller(tmp_path)
    c.launch()
    c.start().result(5)
    assert c.phase == "running"
    c.stop().result(5)
    assert c.phase == "stopped"
    c.shutdown()
    assert c._thread is None and built.apps[0].mic.inner.stopped


def test_ensure_models_downloads_only_what_is_missing(tmp_path, monkeypatch):
    import spark.models as models

    got = {}
    monkeypatch.setattr(
        models,
        "model_status",
        lambda s: {
            "whisper": {"required": True, "present": False},
            "piper": {"required": True, "present": True},
            "kokoro": {"required": False, "present": False},
        },
    )
    monkeypatch.setattr(models, "download_models", lambda s, **kw: got.update(kw))
    said: list[str] = []
    ctl.ensure_models(_settings(tmp_path), said.append)
    assert got["whisper"] is True and got["piper"] is False and got["kokoro"] is False
    assert said == ["Downloading speech models (about 150 MB)…"]


def test_ensure_models_is_silent_when_everything_is_present(tmp_path, monkeypatch):
    import spark.models as models

    monkeypatch.setattr(
        models, "model_status", lambda s: {"whisper": {"required": True, "present": True}}
    )
    monkeypatch.setattr(models, "download_models", lambda *a, **k: pytest.fail("downloaded"))
    said: list[str] = []
    ctl.ensure_models(_settings(tmp_path), said.append)
    assert said == []


async def test_stop_right_after_start_keeps_the_mic_closed(tmp_path):
    c, _events, built = _controller(tmp_path)
    await c.astart()
    await asyncio.wait_for(c.astop(), 1.0)  # before run() even began
    await asyncio.sleep(0.02)
    mic = built.apps[0].mic.inner
    assert c.phase == "stopped" and (mic.stopped or not mic.started)
    await c.aclose()


# --- window plumbing (no real window) --------------------------------------------------------


def test_event_pump_keeps_only_the_latest_level():
    from spark.desktop.window import EventPump

    events = [
        {"type": "level", "level": 0.1},
        {"type": "state", "state": "idle"},
        {"type": "level", "level": 0.4},
        {"type": "noop"},
    ]
    assert EventPump.coalesce(events) == [
        {"type": "state", "state": "idle"},
        {"type": "level", "level": 0.4},
    ]


def test_event_pump_holds_events_until_the_page_is_ready():
    import threading

    from spark.desktop.window import EventPump

    got: list[list[dict]] = []
    done = threading.Event()

    def send(batch):
        got.append(batch)
        done.set()

    pump = EventPump(send)
    pump.put({"type": "phase", "phase": "starting", "detail": ""})
    pump.put({"type": "level", "level": 0.3})  # dropped: nothing to animate yet
    assert not done.wait(0.1)
    pump.ready()
    assert done.wait(2)
    pump.close()
    assert got[0] == [{"type": "phase", "phase": "starting", "detail": ""}]


def test_bridge_exposes_only_its_methods():
    from spark.desktop.window import Bridge

    public = {n for n in dir(Bridge(object(), object())) if not n.startswith("_")}
    assert public == {
        "start",
        "stop",
        "snapshot",
        "connectors",
        "shortcuts",
        "brain",
        "voice",
        "set_mode",
        "set_models",
        "set_muted",
    }


def test_single_instance_blocks_a_second_owner(tmp_path):
    from spark.desktop.single import SingleInstance

    name = f"SparkTest{tmp_path.name}"
    first, second = SingleInstance(name, lock_dir=tmp_path), SingleInstance(name, lock_dir=tmp_path)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_cli_has_a_desktop_command():
    from spark.__main__ import build_parser

    assert build_parser().parse_args(["desktop"]).cmd == "desktop"


def test_desktop_modules_import_without_pywebview():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['webview'] = None\n"
        "import spark.desktop.window, spark.desktop.controller, spark.desktop.single\n"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_ensure_models_shows_plain_progress(tmp_path, monkeypatch):
    import spark.models as models

    monkeypatch.setattr(
        models,
        "model_status",
        lambda s: {
            "whisper": {"required": True, "present": False},
            "kokoro": {"required": True, "present": False},
        },
    )

    def fake_download(settings, *, console, **kw):
        for line in (
            "models dir: X (free disk: 30 GB)",
            "whisper base.en: downloading Systran/faster-whisper-base.en ...",
            "whisper base.en: done (141 MB) at X",
            "piper en_US-lessac-medium: downloading rhasspy/piper-voices/x.onnx ...",
            "kokoro: downloading kokoro-v1.0.onnx ...",
        ):
            console.print(line)

    monkeypatch.setattr(models, "download_models", fake_download)
    said: list[str] = []
    ctl.ensure_models(_settings(tmp_path), said.append)
    assert said == [
        "Downloading speech models (about 490 MB)…",
        "Downloading speech recognition (about 150 MB)…",
        "Downloading the voice (about 60 MB)…",
        "Downloading the default voice, George (about 340 MB)…",
    ]


def test_connectors_api_exposes_only_its_methods(tmp_path):
    from spark.config import Settings
    from spark.desktop.connectors import ConnectorsApi

    api = ConnectorsApi(Settings(config_dir=tmp_path))
    public = {n for n in dir(api) if not n.startswith("_")}
    assert public == {
        "list",
        "get_config",
        "add",
        "import_claude_desktop",
        "update_config",
        "test",
        "set_enabled",
        "set_tool",
        "remove",
        "apply",
        "set_route",
        "get_notes",
        "save_notes",
        "open_link",  # only ever opens claude.ai connector settings
    }


async def test_reload_rebuilds_the_app_and_restarts_if_running(tmp_path):
    c, _events, built = _controller(tmp_path)
    await c.astart()
    assert c.phase == "running" and len(built.apps) == 1
    await c.areload()
    assert c.phase == "running" and len(built.apps) == 2
    assert built.apps[1] is c.app
    await c.astop()
    await c.areload()
    assert c.phase == "stopped" and c.app is None and len(built.apps) == 2
    await c.aclose()


# --- P5.7b: Shortcuts window API ------------------------------------------------------------------


def test_shortcuts_api_exposes_only_its_methods(tmp_path):
    from spark.desktop.shortcuts import ShortcutsApi

    api = ShortcutsApi(make_settings(shortcuts={"file": str(tmp_path / "s.json")}))
    public = {n for n in dir(api) if not n.startswith("_")}
    assert public == {"list", "save", "remove", "set_enabled", "test"}


def test_shortcuts_api_round_trip(tmp_path):
    from spark.desktop.shortcuts import ShortcutsApi

    api = ShortcutsApi(make_settings(shortcuts={"file": str(tmp_path / "s.json")}))
    listed = api.list()
    assert listed["ok"] and listed["shortcuts"][0]["name"] == "Morning briefing"
    bad = api.save({"name": "x", "triggers": ["hi"], "prompt": "p"})
    assert bad == {"ok": False, "error": bad["error"]} and "two words" in bad["error"]
    wrap = {"name": "Wrap up", "triggers": ["close the books", ""], "prompt": "Sum up."}
    res = api.save({**wrap, "route": "fast"})
    assert res["ok"] and res["saved"]
    hit = api.test("Spark, close the books please")["match"]
    assert hit["name"] == "Wrap up" and hit["route"] == "fast"
    assert hit["prompt"].startswith("Sum up.")
    assert api.test("what's the weather")["match"] is None
    assert api.test("  ")["match"] is None
    assert api.set_enabled(res["saved"], False)["ok"]
    assert api.test("close the books")["match"] is None
    remaining = [s["name"] for s in api.remove(res["saved"])["shortcuts"]]
    assert remaining == [d["name"] for d in DEFAULT_SHORTCUTS]


async def test_live_connector_status_reaches_the_window(tmp_path):
    from spark.brain.mock import MockBrain

    class StatusBrain(MockBrain):
        on_mcp_status = None

    class Built(_Built):
        def __call__(self, settings):
            app = super().__call__(settings)
            app.brain = StatusBrain()
            return app

    c, events, built = _controller(tmp_path, app_factory=Built(tmp_path))
    await c.astart()
    brain = built.apps[0].brain
    assert brain.on_mcp_status is not None
    brain.on_mcp_status({"WAREHOUSE": {"status": "failed", "error": "timed out"}})
    assert events[-1] == {
        "type": "connectors",
        "status": {"WAREHOUSE": {"status": "failed", "error": "timed out"}},
    }
    assert c.connector_status["WAREHOUSE"]["status"] == "failed"
    await c.aclose()


# --- P5.7f: the Brain window's apply -------------------------------------------------------------


async def test_apply_brain_restarts_a_running_app_and_sends_the_new_models(tmp_path):
    c, events, built = _controller(tmp_path)
    await c.astart()
    error = await c.aapply_brain()
    assert error is None
    assert c.phase == "running" and len(built.apps) == 2
    model = [e for e in events if e["type"] == "model"][-1]
    assert model["choices"] and model["fast_model"]
    await c.aclose()


async def test_apply_brain_reports_a_start_that_fails(tmp_path):
    class Failing(_Built):
        def __call__(self, settings):
            if self.apps:
                raise BackendUnavailable("openai_compat", "OPENAI_API_KEY is not set", "add it")
            return super().__call__(settings)

    c, _events, _built = _controller(tmp_path, app_factory=Failing(tmp_path))
    await c.astart()
    error = await c.aapply_brain()
    assert error == "OPENAI_API_KEY is not set. Fix: add it"
    assert c.phase == "error"
    await c.aclose()


async def test_apply_brain_when_stopped_only_builds_the_brain(tmp_path, monkeypatch):
    c, events, built = _controller(tmp_path)
    assert await c.aapply_brain() is None  # the mock brain builds fine
    assert built.apps == [] and c.phase == "stopped"
    assert any(e["type"] == "model" and "choices" in e for e in events)

    from spark import registry

    def broken(kind, settings, **kw):
        raise BackendUnavailable("openai_compat", "no model set", "pick one")

    monkeypatch.setattr(registry, "build", broken)
    assert await c.aapply_brain() == "no model set. Fix: pick one"
    await c.aclose()


def test_config_brain_is_config_yaml_before_prefs(tmp_path):
    settings = _settings(tmp_path)
    from spark.prefs import Prefs

    Prefs.from_settings(settings).update(deep_model="claude-sonnet-5")
    c = DesktopController(settings, lambda e: None, model_check=None)
    assert settings.brain.deep_model == "claude-sonnet-5"
    assert c.config_brain.deep_model != "claude-sonnet-5"


def test_friendly_error_uses_the_providers_spoken_line():
    from spark.brain.openai_chat import ProviderError

    exc = ProviderError("auth", "HTTP 401 ...", "OpenAI refused the API key. Check the key.")
    assert friendly_error(exc) == "OpenAI refused the API key. Check the key."


def test_single_instance_lock_id_and_window_title_are_separate():
    from spark.desktop.single import SingleInstance

    assert SingleInstance("HeySpark", window_title="Spark").window_title == "Spark"
    assert SingleInstance("HeySpark").window_title == "HeySpark"


def test_self_test_passes_from_source(monkeypatch, capsys) -> None:
    from spark.desktop import window

    # the real list needs PortAudio etc. (absent on CI's Linux); the bundled-file checks still run
    monkeypatch.setattr(window, "SELF_TEST_MODULES", ("json", "numpy"))
    assert window.main(["--self-test"]) == 0
    assert "self-test ok" in capsys.readouterr().out


def test_self_test_reports_a_missing_module(monkeypatch, capsys) -> None:
    from spark.desktop import window

    monkeypatch.setattr(window, "SELF_TEST_MODULES", ("json", "no_such_module_xyz"))
    assert window.self_test() == 1
    out = capsys.readouterr().out
    assert "FAIL import no_such_module_xyz" in out and "self-test failed" in out
