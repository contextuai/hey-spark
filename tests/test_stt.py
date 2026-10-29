"""STT backends: faster-whisper plumbing (fake module), real model (skipped if absent), Deepgram."""

from __future__ import annotations

import io
import os
import sys
import threading
import types
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from spark.config import Settings
from spark.interfaces import STT, BackendUnavailable
from spark.models import whisper_model_dir, whisper_present

REPO = Path(__file__).resolve().parents[1]
REAL_SETTINGS = Settings(config_dir=REPO)
REAL_WHISPER = whisper_present(whisper_model_dir(REAL_SETTINGS))


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(config_dir=tmp_path, models_dir=str(tmp_path / "models"), **kw)


def _fake_whisper_files(settings: Settings) -> Path:
    d = whisper_model_dir(settings)
    d.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        (d / name).write_bytes(b"x")
    return d


class _FakeWhisperModel:
    instances: ClassVar[list[_FakeWhisperModel]] = []

    def __init__(self, model_size_or_path, **kwargs):
        self.path = model_size_or_path
        self.kwargs = kwargs
        self.calls: list[tuple[np.ndarray, dict]] = []
        self.iter_threads: list[int] = []
        self.text = [" Hello", " world. "]
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))

        def gen():
            self.iter_threads.append(threading.get_ident())
            for t in self.text:
                yield SimpleNamespace(text=t)

        return gen(), SimpleNamespace(language="en")


@pytest.fixture
def fake_fw(monkeypatch):
    _FakeWhisperModel.instances = []
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return mod


# --- faster-whisper plumbing ------------------------------------------------------------------


async def test_transcribe_converts_int16_and_runs_in_thread(tmp_path, fake_fw):
    from spark.stt.faster_whisper import FasterWhisperSTT

    s = _settings(tmp_path)
    model_dir = _fake_whisper_files(s)
    stt = FasterWhisperSTT(s)
    assert isinstance(stt, STT)
    pcm = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    text = await stt.transcribe(pcm)
    assert text == "Hello world."

    (model,) = _FakeWhisperModel.instances
    assert model.path == str(model_dir)
    cfg = s.stt.options.faster_whisper
    assert model.kwargs["device"] == cfg.device == "cpu"
    assert model.kwargs["compute_type"] == cfg.compute_type == "int8"
    assert model.kwargs["cpu_threads"] == cfg.cpu_threads
    assert model.kwargs["download_root"] == str(s.models_path / "whisper")

    audio, kw = model.calls[0]
    assert audio.dtype == np.float32 and audio.ndim == 1
    np.testing.assert_allclose(audio, [0.0, 0.5, -1.0, 32767 / 32768], rtol=1e-6)
    assert kw["beam_size"] == cfg.beam_size
    assert kw["language"] == "en"
    assert kw["without_timestamps"] is True
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is False
    # The lazy segment generator is consumed off the event loop thread.
    assert model.iter_threads and model.iter_threads[0] != threading.get_ident()


async def test_empty_audio_and_no_speech_give_empty_string(tmp_path, fake_fw):
    from spark.stt.faster_whisper import FasterWhisperSTT

    s = _settings(tmp_path)
    _fake_whisper_files(s)
    stt = FasterWhisperSTT(s)
    assert await stt.transcribe(np.zeros(0, dtype=np.int16)) == ""
    assert _FakeWhisperModel.instances == []  # nothing loaded for empty input
    await stt.warmup()
    (model,) = _FakeWhisperModel.instances
    model.text = []
    assert await stt.transcribe(np.zeros(1600, dtype=np.int16)) == ""


async def test_warmup_loads_once_and_transcribes_one_second_of_silence(tmp_path, fake_fw):
    from spark.stt.faster_whisper import FasterWhisperSTT

    s = _settings(tmp_path)
    _fake_whisper_files(s)
    stt = FasterWhisperSTT(s)
    await stt.warmup()
    await stt.transcribe(np.ones(800, dtype=np.int16))
    assert len(_FakeWhisperModel.instances) == 1
    audio, _ = _FakeWhisperModel.instances[0].calls[0]
    assert audio.shape == (16_000,) and not audio.any()


async def test_float_and_stereo_input_are_accepted(tmp_path, fake_fw):
    from spark.stt.faster_whisper import FasterWhisperSTT

    s = _settings(tmp_path)
    _fake_whisper_files(s)
    stt = FasterWhisperSTT(s)
    stereo = np.array([[1000, 3000], [-2000, -2000]], dtype=np.int16)
    await stt.transcribe(stereo)
    audio, _ = _FakeWhisperModel.instances[0].calls[0]
    np.testing.assert_allclose(audio, [2000 / 32768, -2000 / 32768], rtol=1e-4)


def test_missing_model_raises_backend_unavailable_with_download_fix(tmp_path, fake_fw):
    from spark.stt.faster_whisper import FasterWhisperSTT

    with pytest.raises(BackendUnavailable) as info:
        FasterWhisperSTT(_settings(tmp_path))
    assert "models download" in info.value.fix
    assert "base.en" in info.value.reason


def test_partial_model_dir_counts_as_missing(tmp_path, fake_fw):
    from spark.stt.faster_whisper import FasterWhisperSTT

    s = _settings(tmp_path)
    d = _fake_whisper_files(s)
    (d / "model.bin").unlink()
    with pytest.raises(BackendUnavailable):
        FasterWhisperSTT(s)


def test_missing_package_raises_backend_unavailable(tmp_path, monkeypatch):
    from spark.stt.faster_whisper import FasterWhisperSTT

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(BackendUnavailable) as info:
        FasterWhisperSTT(_settings(tmp_path))
    assert info.value.fix == "uv sync"


def test_registry_builds_faster_whisper(tmp_path, fake_fw):
    from spark import registry
    from spark.stt.faster_whisper import FasterWhisperSTT

    s = _settings(tmp_path)
    _fake_whisper_files(s)
    assert isinstance(registry.build("stt", s), FasterWhisperSTT)


# --- faster-whisper, real model -----------------------------------------------------------------


@pytest.fixture(scope="module")
def real_whisper(shared_whisper):
    return shared_whisper  # loaded once per session (tests/conftest.py)


@pytest.mark.skipif(
    not REAL_WHISPER, reason="whisper model absent: uv run python -m spark models download"
)
async def test_real_whisper_warmup_and_silence(real_whisper):
    await real_whisper.warmup()
    assert await real_whisper.transcribe(np.zeros(16_000, dtype=np.int16)) == ""


@pytest.mark.skipif(
    not REAL_WHISPER, reason="whisper model absent: uv run python -m spark models download"
)
async def test_real_whisper_transcribes_speech(real_whisper):
    from spark.audio.stream import resample_to_16k
    from spark.models import piper_voice_paths

    onnx, _ = piper_voice_paths(REAL_SETTINGS)
    if not onnx.is_file():
        pytest.skip("Piper voice absent (used to make test speech)")
    from spark.tts.piper import PiperTTS
    from tests.fakes import FakePlayer

    pcm, rate = PiperTTS(REAL_SETTINGS, player=FakePlayer()).synthesize("Turn on the lights.")
    text = await real_whisper.transcribe(resample_to_16k(pcm, rate))
    assert "light" in text.lower()


# --- Deepgram ---------------------------------------------------------------------------------


class _FakeMedia:
    def __init__(self, transcript="  hello there ", exc: Exception | None = None):
        self.transcript = transcript
        self.exc = exc
        self.calls: list[dict] = []

    async def transcribe_file(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        alt = SimpleNamespace(transcript=self.transcript)
        return SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
        )


def _fake_dg_client(media: _FakeMedia):
    return SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(media=media)))


def test_deepgram_missing_package(tmp_path, monkeypatch):
    from spark.stt.deepgram import DeepgramSTT

    monkeypatch.setitem(sys.modules, "deepgram", None)
    with pytest.raises(BackendUnavailable) as info:
        DeepgramSTT(_settings(tmp_path))
    assert info.value.fix == "uv sync --extra deepgram"


def test_deepgram_missing_key(tmp_path, monkeypatch):
    from spark.stt.deepgram import DeepgramSTT

    mod = types.ModuleType("deepgram")
    mod.AsyncDeepgramClient = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "deepgram", mod)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable) as info:
        DeepgramSTT(_settings(tmp_path))
    assert "DEEPGRAM_API_KEY" in info.value.reason
    assert "DEEPGRAM_API_KEY" in info.value.fix


def test_deepgram_builds_client_from_env_key(tmp_path, monkeypatch):
    from spark.stt.deepgram import DeepgramSTT

    made: list[dict] = []
    mod = types.ModuleType("deepgram")
    mod.AsyncDeepgramClient = lambda **kw: made.append(kw) or SimpleNamespace()
    monkeypatch.setitem(sys.modules, "deepgram", mod)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "sekrit-123")
    DeepgramSTT(_settings(tmp_path))
    assert made == [{"api_key": "sekrit-123", "timeout": 10.0}]


async def test_deepgram_sends_wav_and_parses_transcript(tmp_path):
    from spark.stt.deepgram import DeepgramSTT

    media = _FakeMedia()
    stt = DeepgramSTT(_settings(tmp_path), client=_fake_dg_client(media))
    pcm = (np.sin(np.arange(1600) / 5) * 8000).astype(np.int16)
    assert await stt.transcribe(pcm) == "hello there"
    (call,) = media.calls
    assert call["model"] == "nova-3" and call["language"] == "en"
    assert call["request_options"]["timeout"] == 10.0
    with wave.open(io.BytesIO(call["request"]), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (16_000, 1, 2)
        got = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(got, pcm)
    assert await stt.transcribe(np.zeros(0, dtype=np.int16)) == ""
    assert len(media.calls) == 1


async def test_deepgram_errors_never_contain_the_key(tmp_path, monkeypatch):
    from spark.stt.deepgram import DeepgramError, DeepgramSTT

    key = "dg-super-secret"
    monkeypatch.setenv("DEEPGRAM_API_KEY", key)
    exc = RuntimeError(f"401 Unauthorized, Authorization: Token {key}")
    exc.status_code = 401
    stt = DeepgramSTT(_settings(tmp_path), client=_fake_dg_client(_FakeMedia(exc=exc)))
    with pytest.raises(DeepgramError) as info:
        await stt.transcribe(np.ones(160, dtype=np.int16))
    assert key not in str(info.value) and "401" in str(info.value)
    assert info.value.__cause__ is None


async def test_deepgram_empty_results(tmp_path):
    from spark.stt.deepgram import DeepgramSTT

    class Empty(_FakeMedia):
        async def transcribe_file(self, **kwargs):
            return SimpleNamespace(results=SimpleNamespace(channels=[]))

    stt = DeepgramSTT(_settings(tmp_path), client=_fake_dg_client(Empty()))
    assert await stt.transcribe(np.ones(160, dtype=np.int16)) == ""


@pytest.mark.live
async def test_deepgram_live():
    pytest.importorskip("deepgram")
    if not os.environ.get("DEEPGRAM_API_KEY"):
        pytest.skip("DEEPGRAM_API_KEY not set")
    from spark.audio.stream import resample_to_16k
    from spark.models import piper_voice_paths

    if not piper_voice_paths(REAL_SETTINGS)[0].is_file():
        pytest.skip("Piper voice absent (used to make test speech)")
    from spark.stt.deepgram import DeepgramSTT
    from spark.tts.piper import PiperTTS
    from tests.fakes import FakePlayer

    pcm, rate = PiperTTS(REAL_SETTINGS, player=FakePlayer()).synthesize(
        "What is the capital of Japan?"
    )
    text = await DeepgramSTT(REAL_SETTINGS).transcribe(resample_to_16k(pcm, rate))
    assert "japan" in text.lower()
