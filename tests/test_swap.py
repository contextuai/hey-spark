"""Phase 1 acceptance: STT and TTS backends swap by editing config.yaml alone (spec §10)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

from spark import registry
from spark.app import SparkApp
from spark.config import load_settings
from spark.interfaces import STT, TTS
from tests.fakes import FakeMic, FakePlayer, FakeSTT, FakeTTS, FakeVAD, FakeWakeWord

ROOT = Path(__file__).resolve().parents[1]


class AltSTT(FakeSTT):
    """A second STT backend, distinct from FakeSTT."""


class AltTTS(FakeTTS):
    """A second TTS backend, distinct from FakeTTS."""


@pytest.fixture
def fake_backends():
    entries = {
        "audio_in": {"fake": FakeMic},
        "audio_out": {"fake": FakePlayer},
        "wakeword": {"fake": FakeWakeWord},
        "vad": {"fake": FakeVAD},
        "stt": {"stt_a": FakeSTT, "stt_b": AltSTT},
        "tts": {"tts_a": FakeTTS, "tts_b": AltTTS},
    }
    for kind, names in entries.items():
        for name, cls in names.items():
            registry.register(kind, name, cls)
    yield
    for kind, names in entries.items():
        for name in names:
            registry.unregister(kind, name)


def _write_config(path: Path, stt: str, tts: str) -> Path:
    """The repo's config.yaml with only the backends (and log dir) changed."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["audio"]["input_backend"] = "fake"
    cfg["audio"]["output_backend"] = "fake"
    cfg["wakeword"]["backend"] = "fake"
    cfg["vad"]["backend"] = "fake"
    cfg["brain"]["backend"] = "mock"
    cfg["logging"]["dir"] = str(path.parent / "logs")
    cfg["stt"]["backend"] = stt
    cfg["tts"]["backend"] = tts
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.usefixtures("fake_backends")
async def test_swap_stt_and_tts_via_config_only(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith("SPARK_")]:
        monkeypatch.delenv(key)
    a = load_settings(_write_config(tmp_path / "a.yaml", "stt_a", "tts_a"), load_env_file=False)
    b = load_settings(_write_config(tmp_path / "b.yaml", "stt_b", "tts_b"), load_env_file=False)

    app_a = SparkApp.from_settings(a)
    app_b = SparkApp.from_settings(b)
    assert type(app_a.stt) is FakeSTT and type(app_a.tts) is FakeTTS
    assert type(app_b.stt) is AltSTT and type(app_b.tts) is AltTTS

    # both configurations run a full text turn through the same app code
    for app in (app_a, app_b):
        assert await app.text_turn("what is two plus two") == "Two plus two is four."
        assert app.tts.spoken == ["Two plus two is four."]
        await app.close()


@pytest.mark.usefixtures("fake_backends")
def test_swap_via_env_override(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path / "c.yaml", "stt_a", "tts_a")
    monkeypatch.setenv("SPARK_STT__BACKEND", "stt_b")
    monkeypatch.setenv("SPARK_TTS__BACKEND", "tts_b")
    app = SparkApp.from_settings(load_settings(cfg, load_env_file=False))
    assert type(app.stt) is AltSTT and type(app.tts) is AltTTS


@pytest.mark.parametrize(
    ("kind", "names"),
    [("stt", ["faster_whisper", "deepgram"]), ("tts", ["piper", "kokoro", "elevenlabs", "print"])],
)
def test_real_backend_names_resolve(kind, names):
    """Every real backend selectable in config.yaml maps to a class of the right type."""
    base = {"stt": STT, "tts": TTS}[kind]
    for name in names:
        module = str(registry.REGISTRY[kind][name]).partition(":")[0]
        if importlib.util.find_spec(module) is None:
            pytest.fail(f"{kind} backend {name!r} points at missing module {module}")
        assert issubclass(registry.resolve(kind, name), base)
