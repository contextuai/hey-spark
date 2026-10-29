"""TTS: the sentence pipeline (fake synth + FakePlayer), Piper, round trip, Kokoro, ElevenLabs."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import types
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spark.config import Settings
from spark.interfaces import TTS, BackendUnavailable
from spark.models import piper_voice_paths, whisper_model_dir, whisper_present
from spark.tts.base import SentenceTTS
from tests.fakes import FakePlayer

REPO = Path(__file__).resolve().parents[1]
REAL_SETTINGS = Settings(config_dir=REPO)
PIPER_PRESENT = all(p.is_file() for p in piper_voice_paths(REAL_SETTINGS))
WHISPER_PRESENT = whisper_present(whisper_model_dir(REAL_SETTINGS))
NO_PIPER = "Piper voice absent: uv run python -m spark models download"


async def chunks(items, delay: float = 0.0) -> AsyncIterator[str]:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


class FakeSynthTTS(SentenceTTS):
    """Audio length encodes the sentence: ``100 * len(text)`` samples at 16 kHz."""

    backend = "tts/fake"

    def __init__(self, settings, *, synth_s: float = 0.0, rate: int = 16_000, **deps):
        super().__init__(settings, **deps)
        self.synth_s = synth_s
        self.rate = rate
        self.synthesized: list[str] = []
        self.synth_starts: list[float] = []
        self.threads: set[int] = set()
        self.loaded = 0
        self.fail_on: str | None = None

    def load(self):
        self.loaded += 1

    def synthesize(self, text):
        self.synth_starts.append(time.perf_counter())
        self.threads.add(threading.get_ident())
        if self.synth_s:
            time.sleep(self.synth_s)
        if text == self.fail_on:
            raise RuntimeError("synth exploded")
        self.synthesized.append(text)
        return np.full(100 * len(text), 1000, dtype=np.int16), self.rate


class StreamPlayer(FakePlayer):
    """FakePlayer plus a gapless ``play_stream`` like SoundDevicePlayer's."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.streams: list[tuple[int, list[int]]] = []

    async def play_stream(self, source, sample_rate):
        got: list[int] = []
        self.streams.append((sample_rate, got))
        async for chunk in source:
            got.append(len(chunk))


# --- pipeline ---------------------------------------------------------------------------------


async def test_order_preserved_and_empty_chunks_skipped():
    player = FakePlayer()
    tts = FakeSynthTTS(Settings(), player=player)
    assert isinstance(tts, TTS)
    await tts.speak(chunks(["One.", "", "   ", "Three three.", "\n", "Four!"]))
    assert tts.synthesized == ["One.", "Three three.", "Four!"]
    assert player.played == [(400, 16_000), (1200, 16_000), (500, 16_000)]
    assert tts.threads and threading.get_ident() not in tts.threads
    assert not tts.speaking


async def test_whitespace_is_normalized_before_synthesis():
    tts = FakeSynthTTS(Settings(), player=FakePlayer())
    await tts.speak(chunks(["  Hello \n  there.  "]))
    assert tts.synthesized == ["Hello there."]


async def test_first_audio_after_first_sentence_and_synthesis_is_pipelined():
    player = FakePlayer(speed=1.0)  # real-time playback
    marks: list[float] = []
    tts = FakeSynthTTS(
        Settings(), player=player, synth_s=0.2, on_first_audio=lambda: marks.append(1)
    )
    # 9600 samples = 0.6 s of audio each, three times the 0.2 s synthesis, so pipelined and
    # sequential runs are far apart: CI runners (macOS especially) oversleep by 100 ms or more.
    sentences = ["A" * 96, "B" * 96, "C" * 96]
    t0 = time.perf_counter()
    await tts.speak(chunks(sentences))
    assert marks == [1]  # exactly once per speak
    first = tts.first_audio_at - t0
    assert 0.19 <= first < 0.38, first  # one synthesis, not two
    # Each synthesis starts while the previous sentence is still playing. Sequential would start
    # them at 0.8 s and 1.6 s; pipelined at about 0.2 s and no later than 0.8 s. (No wall-clock
    # total: shared CI runners oversleep too much for that to be reliable.)
    starts = [s - t0 for s in tts.synth_starts]
    assert starts[1] < 0.6, starts
    assert starts[2] < 1.2, starts


async def test_stop_returns_promptly_and_drops_later_chunks():
    player = FakePlayer(speed=1.0)
    tts = FakeSynthTTS(Settings(), player=player, synth_s=0.01)
    long = "x" * 320  # 2 s of audio
    produced: list[str] = []

    async def brain():
        for i in range(10):
            produced.append(f"{long}{i}")
            yield f"{long}{i}"
            await asyncio.sleep(0.05)

    started = asyncio.Event()
    tts.on_first_audio = started.set
    task = asyncio.create_task(tts.speak(brain()))
    await asyncio.wait_for(started.wait(), 1.0)
    await asyncio.sleep(0.05)
    t_stop = time.perf_counter()
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert time.perf_counter() - t_stop < 0.1
    assert player.stops >= 1
    assert len(player.played) == 1
    # A synthesis already running in its worker thread may still finish (to_thread can't be
    # interrupted), but none may start after stop.
    n = len(tts.synth_starts)
    await asyncio.sleep(0.2)
    assert len(tts.synth_starts) == n
    assert len(produced) < 10  # the text stream was not drained
    assert not tts.speaking


async def test_stop_while_waiting_for_text():
    tts = FakeSynthTTS(Settings(), player=FakePlayer())
    gate = asyncio.Event()

    async def slow_brain():
        await gate.wait()
        yield "never spoken."

    task = asyncio.create_task(tts.speak(slow_brain()))
    await asyncio.sleep(0.02)
    t0 = time.perf_counter()
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert time.perf_counter() - t0 < 0.1
    assert tts.synthesized == []


async def test_speak_after_stop_works_again():
    player = FakePlayer()
    tts = FakeSynthTTS(Settings(), player=player)
    await tts.stop()  # stale stop before speaking must not affect the next turn
    await tts.speak(chunks(["Hi."]))
    await tts.speak(chunks(["Again."]))
    assert tts.synthesized == ["Hi.", "Again."]


async def test_synthesis_error_propagates():
    tts = FakeSynthTTS(Settings(), player=FakePlayer())
    tts.fail_on = "bad."
    with pytest.raises(RuntimeError, match="synth exploded"):
        await tts.speak(chunks(["ok.", "bad.", "never."]))
    assert tts.synthesized == ["ok."]
    assert not tts.speaking


async def test_gapless_play_stream_is_used_and_rates_are_matched():
    player = StreamPlayer()
    tts = FakeSynthTTS(Settings(), player=player, rate=22_050)
    marks: list[int] = []
    tts.on_first_audio = lambda: marks.append(1)
    await tts.speak(chunks(["One.", "Two two."]))
    assert player.streams == [(22_050, [400, 800])]
    assert player.played == []
    assert marks == [1]


async def test_play_stream_resamples_a_mismatched_rate():
    player = StreamPlayer()
    tts = FakeSynthTTS(Settings(), player=player)
    rates = iter([16_000, 8_000])
    original = tts.synthesize

    def synth(text):
        pcm, _ = original(text)
        return pcm, next(rates)

    tts.synthesize = synth
    await tts.speak(chunks(["abcd", "abcd"]))
    assert player.streams == [(16_000, [400, 800])]


async def test_warmup_loads_and_synthesizes_off_loop():
    tts = FakeSynthTTS(Settings(), player=FakePlayer())
    await tts.warmup()
    assert tts.loaded == 1 and tts.synthesized == [SentenceTTS.warmup_text]
    assert threading.get_ident() not in tts.threads


async def test_first_audio_callback_errors_do_not_break_speech():
    player = FakePlayer()

    def boom():
        raise ValueError("timer broke")

    tts = FakeSynthTTS(Settings(), player=player, on_first_audio=boom)
    await tts.speak(chunks(["One.", "Two."]))
    assert len(player.played) == 2


# --- Piper ------------------------------------------------------------------------------------


def test_piper_missing_voice(tmp_path):
    from spark.tts.piper import PiperTTS

    s = Settings(config_dir=tmp_path, models_dir="./nothing")
    with pytest.raises(BackendUnavailable) as info:
        PiperTTS(s, player=FakePlayer())
    assert "models download" in info.value.fix
    assert "en_US-ljspeech-medium" in info.value.reason


def test_piper_missing_package(tmp_path, monkeypatch):
    from spark.tts.piper import PiperTTS

    monkeypatch.setitem(sys.modules, "piper", None)
    with pytest.raises(BackendUnavailable) as info:
        PiperTTS(Settings(config_dir=tmp_path), player=FakePlayer())
    assert info.value.fix == "uv sync"


@pytest.fixture(scope="module")
def piper_tts():
    from spark.tts.piper import PiperTTS

    return PiperTTS(REAL_SETTINGS, player=FakePlayer())


@pytest.mark.skipif(not PIPER_PRESENT, reason=NO_PIPER)
def test_piper_real_synthesis(piper_tts):
    pcm, rate = piper_tts.synthesize("Hello from Spark. This is a second sentence.")
    assert pcm.dtype == np.int16 and pcm.ndim == 1
    assert rate == 22_050
    assert 1.0 < len(pcm) / rate < 8.0
    assert np.abs(pcm).max() > 1000


@pytest.mark.skipif(not PIPER_PRESENT, reason=NO_PIPER)
async def test_piper_speaks_each_sentence_through_the_player(piper_tts):
    player = FakePlayer()
    piper_tts.player = player
    await piper_tts.speak(chunks(["Hello there.", "How are you?"]))
    assert [r for _, r in player.played] == [22_050, 22_050]
    assert all(n > 5000 for n, _ in player.played)


@pytest.mark.skipif(not (PIPER_PRESENT and WHISPER_PRESENT), reason="Piper voice or whisper absent")
async def test_round_trip_piper_to_whisper(piper_tts, shared_whisper):
    from spark.audio.stream import resample_to_16k

    pcm, rate = await asyncio.to_thread(piper_tts.synthesize, "what is the capital of Japan")
    audio16 = resample_to_16k(pcm, rate)
    assert audio16.dtype == np.int16
    text = await shared_whisper.transcribe(audio16)
    assert "japan" in text.lower(), text


# --- Kokoro -----------------------------------------------------------------------------------


def _kokoro_settings(tmp_path: Path) -> Settings:
    return Settings(config_dir=tmp_path, models_dir="./m")


def test_kokoro_missing_package(tmp_path, monkeypatch):
    from spark.tts.kokoro_onnx import KokoroTTS

    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
    with pytest.raises(BackendUnavailable) as info:
        KokoroTTS(_kokoro_settings(tmp_path), player=FakePlayer())
    assert info.value.fix == "uv sync --extra kokoro"


def _fake_kokoro_module(created: list):
    class Kokoro:
        def __init__(self, model_path, voices_path):
            created.append((model_path, voices_path))

        def create(self, text, voice, speed=1.0, lang="en-us"):
            created.append((text, voice, speed, lang))
            return np.array([0.0, 0.5, -1.0], dtype=np.float32), 24_000

    mod = types.ModuleType("kokoro_onnx")
    mod.Kokoro = Kokoro
    return mod


def test_kokoro_missing_files(tmp_path, monkeypatch):
    from spark.tts.kokoro_onnx import KokoroTTS

    monkeypatch.setitem(sys.modules, "kokoro_onnx", _fake_kokoro_module([]))
    with pytest.raises(BackendUnavailable) as info:
        KokoroTTS(_kokoro_settings(tmp_path), player=FakePlayer())
    assert "--kokoro" in info.value.fix


def test_kokoro_synthesize_with_fake_engine(tmp_path, monkeypatch):
    from spark.models import kokoro_paths
    from spark.tts.kokoro_onnx import KokoroTTS

    created: list = []
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _fake_kokoro_module(created))
    s = _kokoro_settings(tmp_path)
    for p in kokoro_paths(s):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    tts = KokoroTTS(s, player=FakePlayer())
    pcm, rate = tts.synthesize("Hi.")
    assert rate == 24_000 and pcm.dtype == np.int16
    np.testing.assert_array_equal(pcm, [0, 16384, -32767])
    assert created[1] == ("Hi.", "bm_george", 1.0, "en-gb")
    tts.synthesize("Again.")
    assert sum(1 for c in created if len(c) == 2) == 1  # engine loaded once


# --- ElevenLabs -------------------------------------------------------------------------------


def test_elevenlabs_missing_package(tmp_path, monkeypatch):
    from spark.tts.elevenlabs import ElevenLabsTTS

    monkeypatch.setitem(sys.modules, "elevenlabs", None)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", None)
    with pytest.raises(BackendUnavailable) as info:
        ElevenLabsTTS(Settings(config_dir=tmp_path), player=FakePlayer())
    assert info.value.fix == "uv sync --extra elevenlabs"


def test_elevenlabs_missing_key(tmp_path, monkeypatch):
    from spark.tts.elevenlabs import ElevenLabsTTS

    pkg = types.ModuleType("elevenlabs")
    client_mod = types.ModuleType("elevenlabs.client")
    client_mod.ElevenLabs = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "elevenlabs", pkg)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", client_mod)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable) as info:
        ElevenLabsTTS(Settings(config_dir=tmp_path), player=FakePlayer())
    assert "ELEVENLABS_API_KEY" in info.value.reason


class _FakeTTSApi:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc
        self.calls: list[tuple] = []

    def convert(self, voice_id, **kwargs):
        self.calls.append((voice_id, kwargs))
        if self.exc is not None:
            raise self.exc
        pcm = np.array([1, -2, 300, -32768], dtype="<i2").tobytes()
        return iter([pcm[:3], pcm[3:], b""])


def test_elevenlabs_requests_raw_pcm_and_decodes_it(tmp_path):
    from spark.tts.elevenlabs import ElevenLabsTTS

    api = _FakeTTSApi()
    tts = ElevenLabsTTS(
        Settings(config_dir=tmp_path),
        player=FakePlayer(),
        client=SimpleNamespace(text_to_speech=api),
    )
    pcm, rate = tts.synthesize("Hello.")
    assert rate == 22_050
    np.testing.assert_array_equal(pcm, [1, -2, 300, -32768])
    voice_id, kw = api.calls[0]
    assert voice_id == "21m00Tcm4TlvDq8ikWAM"
    assert kw["output_format"] == "pcm_22050"
    assert kw["model_id"] == "eleven_flash_v2_5" and kw["text"] == "Hello."


async def test_elevenlabs_warmup_makes_no_request(tmp_path):
    from spark.tts.elevenlabs import ElevenLabsTTS

    api = _FakeTTSApi()
    tts = ElevenLabsTTS(
        Settings(config_dir=tmp_path),
        player=FakePlayer(),
        client=SimpleNamespace(text_to_speech=api),
    )
    await tts.warmup()
    assert api.calls == []


def test_elevenlabs_errors_never_contain_the_key(tmp_path):
    from spark.tts.elevenlabs import ElevenLabsError, ElevenLabsTTS

    key = "xi-very-secret"
    exc = RuntimeError(f"401 bad key {key}")
    exc.status_code = 401
    tts = ElevenLabsTTS(
        Settings(config_dir=tmp_path),
        player=FakePlayer(),
        client=SimpleNamespace(text_to_speech=_FakeTTSApi(exc)),
    )
    with pytest.raises(ElevenLabsError) as info:
        tts.synthesize("Hi.")
    assert key not in str(info.value) and "401" in str(info.value)


# --- live cloud -------------------------------------------------------------------------------


@pytest.mark.live
def test_elevenlabs_live():
    pytest.importorskip("elevenlabs")
    if not os.environ.get("ELEVENLABS_API_KEY"):
        pytest.skip("ELEVENLABS_API_KEY not set")
    from spark.tts.elevenlabs import ElevenLabsTTS

    pcm, rate = ElevenLabsTTS(REAL_SETTINGS, player=FakePlayer()).synthesize("Hello from Spark.")
    assert rate == 22_050 and len(pcm) > rate * 0.3


@pytest.mark.live
def test_kokoro_real_if_installed():
    pytest.importorskip("kokoro_onnx")
    from spark.models import kokoro_paths
    from spark.tts.kokoro_onnx import KokoroTTS

    if not all(p.is_file() for p in kokoro_paths(REAL_SETTINGS)):
        pytest.skip("Kokoro model files absent")
    pcm, rate = KokoroTTS(REAL_SETTINGS, player=FakePlayer()).synthesize("Hello from Spark.")
    assert rate == 24_000 and len(pcm) > rate * 0.3


async def test_synthesis_only_without_player_then_speak_requires_one():
    """``spark bench tts`` builds a TTS with no player; only speak() needs one."""

    class _Synth(SentenceTTS):
        backend = "synth"

        def synthesize(self, text):
            return np.zeros(160, dtype=np.int16), 16000

        def _warm_blocking(self):
            return None

    tts = _Synth(Settings(), player=None)
    pcm, rate = tts.synthesize("hi")
    assert rate == 16000 and len(pcm) == 160
    await tts.stop()  # no player: still safe

    async def one():
        yield "Hello."

    with pytest.raises(BackendUnavailable):
        await tts.speak(one())
