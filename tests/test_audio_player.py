"""PlaybackBuffer mixing and stop latency, SoundDevicePlayer with a fake output stream."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from spark.audio.earcons import error_tone, generate, thinking_tick, wake_chime
from spark.audio.player import BLOCK_MS, PlaybackBuffer, SoundDevicePlayer
from spark.audio.wavio import read_wav
from spark.config import Settings
from spark.interfaces import BackendUnavailable

ROOT = Path(__file__).resolve().parents[1]
RATE = 48_000
BLOCK = RATE * BLOCK_MS // 1000  # 960 samples = 20 ms


def _settings(tmp_path: Path | None = None, **app) -> Settings:
    kw: dict = {"config_dir": ROOT}
    if tmp_path is not None:
        app.setdefault("sounds_dir", str(tmp_path))
    if app:
        kw["app"] = app
    return Settings(**kw)


class FakeOutputStream:
    """Stands in for sounddevice.OutputStream. ``tick()`` runs one device callback."""

    def __init__(self, *, samplerate, blocksize, channels, callback):
        self.samplerate, self.blocksize, self.channels = samplerate, blocksize, channels
        self.callback = callback
        self.started = self.closed = False
        self.played: list[np.ndarray] = []

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def tick(self) -> np.ndarray:
        out = np.full((self.blocksize, self.channels), 9.0, dtype=np.float32)  # garbage
        self.callback(out, self.blocksize, None, 0)
        self.played.append(out.copy())
        return out


class RealtimeFakeStream(FakeOutputStream):
    """Runs the callback on its own thread every block period, like PortAudio does."""

    def start(self):
        super().start()
        self._halt = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        period = self.blocksize / self.samplerate
        next_t = time.perf_counter()
        while not self._halt.is_set():
            self.tick()
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def stop(self):
        self._halt.set()
        self.thread.join(timeout=1)
        super().stop()


class Factory:
    def __init__(self, cls=FakeOutputStream):
        self.cls = cls
        self.streams: list[FakeOutputStream] = []

    def __call__(self, **kw):
        s = self.cls(**kw)
        self.streams.append(s)
        return s


def _player(settings: Settings, cls=FakeOutputStream) -> tuple[SoundDevicePlayer, Factory]:
    f = Factory(cls)
    return SoundDevicePlayer(settings, stream_factory=f, device_rate=RATE), f


# --- PlaybackBuffer (pure numpy) -------------------------------------------------------------


def test_fill_plays_samples_then_silence_and_finishes():
    buf = PlaybackBuffer()
    done = []
    buf.add(np.full(1500, 0.25, np.float32), on_done=lambda: done.append(1))
    out = np.zeros((BLOCK, 2), np.float32)
    buf.fill(out)
    assert np.all(out == 0.25) and not done and buf.active
    buf.fill(out)
    assert np.all(out[: 1500 - BLOCK] == 0.25) and np.all(out[1500 - BLOCK :] == 0)
    assert done == [1] and not buf.active


def test_fill_mixes_and_clips():
    buf = PlaybackBuffer()
    buf.add(np.full(100, 0.7, np.float32))
    buf.add(np.full(50, 0.7, np.float32))
    out = np.zeros(100, np.float32)
    buf.fill(out)
    assert np.all(out[:50] == 1.0) and np.allclose(out[50:], 0.7)


def test_stop_silences_the_very_next_block():
    """Stop latency in samples: nothing but zeros after stop() returns."""
    buf = PlaybackBuffer()
    done = []
    buf.add(np.full(RATE * 5, 0.5, np.float32), on_done=lambda: done.append(1))
    out = np.zeros(BLOCK, np.float32)
    for _ in range(10):
        buf.fill(out)
        assert np.all(out == 0.5)
    t0 = time.perf_counter()
    buf.stop()
    stop_call_s = time.perf_counter() - t0
    samples_after_stop = 0
    for _ in range(20):
        buf.fill(out)
        samples_after_stop += int(np.count_nonzero(out))
    assert samples_after_stop == 0
    assert done == [1] and not buf.active
    assert stop_call_s < 0.01
    # Worst case audible tail = the block already handed to the device (BLOCK_MS) + its buffer.
    assert BLOCK_MS < 300


def test_streamed_source_append_close_and_cancel():
    buf = PlaybackBuffer()
    done = []
    src = buf.add(None, closed=False, on_done=lambda: done.append(1))
    out = np.zeros(BLOCK, np.float32)
    buf.fill(out)
    assert not done and np.all(out == 0)  # open source with no audio yet: silent, not finished
    assert buf.append(src, np.full(BLOCK, 0.1, np.float32))
    buf.fill(out)
    assert np.allclose(out, 0.1)
    buf.close(src)
    assert done == [1]
    assert not buf.append(src, np.ones(10, np.float32))
    src2 = buf.add(np.ones(RATE, np.float32))
    buf.cancel(src2)
    buf.fill(out)
    assert np.all(out == 0) and src2.done


# --- SoundDevicePlayer with a fake stream ----------------------------------------------------


async def test_play_awaits_completion_and_resamples():
    player, f = _player(_settings())
    pcm = (np.ones(16_000) * 8000).astype(np.int16)  # 1 s at 16 kHz → 48000 samples at 48k
    task = asyncio.create_task(player.play(pcm, 16_000))
    for _ in range(100):  # play() opens the stream on a worker thread first
        await asyncio.sleep(0.005)
        if player.is_playing:
            break
    stream = f.streams[0]
    assert stream.started and stream.blocksize == BLOCK and stream.channels == 1
    assert player.is_playing
    for _ in range(RATE // BLOCK):
        stream.tick()
    await asyncio.wait_for(task, 1)
    assert not player.is_playing
    played = np.concatenate([b[:, 0] for b in stream.played])
    assert np.count_nonzero(played) == pytest.approx(RATE, abs=2)


async def test_stop_returns_play_within_300ms_realtime():
    """Real-time fake device: stop() → silence and play() returns well under 300 ms."""
    player, f = _player(_settings(), cls=RealtimeFakeStream)
    tone = (0.3 * np.sin(np.arange(22_050 * 5) * 2 * np.pi * 440 / 22_050)).astype(np.float32)
    task = asyncio.create_task(player.play(tone, 22_050))
    await asyncio.sleep(0.2)
    stream = f.streams[0]
    assert player.is_playing
    n_before = len(stream.played)
    t0 = time.perf_counter()
    player.stop()
    stop_return_s = time.perf_counter() - t0
    await asyncio.wait_for(task, 0.3)
    play_return_s = time.perf_counter() - t0
    await asyncio.sleep(0.1)
    blocks_after = stream.played[n_before:]
    # The block being filled at the instant of stop() may be the last audible one.
    audible_after = sum(1 for b in blocks_after if np.any(b != 0))
    await player.close()
    assert stop_return_s < 0.01
    assert play_return_s < 0.3
    assert audible_after <= 1
    assert audible_after * BLOCK_MS < 300


async def test_play_stream_is_gapless_and_stoppable():
    player, f = _player(_settings())

    async def chunks():
        for _ in range(3):
            yield np.full(1600, 1000, np.int16)  # 100 ms at 16k → 4800 at 48k
            await asyncio.sleep(0)

    await player.warmup()
    task = asyncio.create_task(player.play_stream(chunks(), 16_000))
    for _ in range(5):
        await asyncio.sleep(0)
    stream = f.streams[0]
    for _ in range(3 * 4800 // BLOCK):
        stream.tick()
    await asyncio.wait_for(task, 1)
    played = np.concatenate([b[:, 0] for b in stream.played])
    assert np.count_nonzero(played) == pytest.approx(3 * 4800, abs=3)

    async def endless():
        while True:
            yield np.full(1600, 1000, np.int16)
            await asyncio.sleep(0.005)

    task = asyncio.create_task(player.play_stream(endless(), 16_000))
    await asyncio.sleep(0.02)
    player.stop()
    await asyncio.wait_for(task, 0.3)
    assert np.all(stream.tick() == 0)


async def test_cancelling_play_stops_its_audio():
    player, f = _player(_settings())
    await player.warmup()
    task = asyncio.create_task(player.play(np.ones(16_000, np.int16) * 5000, 16_000))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert np.all(f.streams[0].tick() == 0)


# --- earcons ---------------------------------------------------------------------------------


def _join_open(player: SoundDevicePlayer) -> None:
    if player._open_thread is not None:
        player._open_thread.join(timeout=2)


def test_earcon_plays_with_volume(tmp_path):
    generate(tmp_path)
    player, f = _player(_settings(tmp_path))
    player.settings.audio.earcon_volume = 0.5
    player.play_earcon("wake")
    assert player.is_playing
    _join_open(player)
    stream = f.streams[0]
    peak = max(float(np.max(np.abs(stream.tick()))) for _ in range(15))
    assert 0.25 < peak <= 0.31  # chime peaks at 0.6, halved by earcon_volume
    player.play_earcon("wake")
    assert "wake" in player._earcons  # cached after the first load


def test_first_earcon_without_warmup_opens_inline_and_plays(tmp_path):
    """Without warmup(), the first cue opens the device on the calling thread, then plays.

    (It blocks for the device-open time once; the app always warms the player at startup.)
    """
    import threading

    generate(tmp_path)
    opened_on = []

    class RecordingStream(FakeOutputStream):
        def start(self):
            opened_on.append(threading.get_ident())
            super().start()

    player, f = _player(_settings(tmp_path), cls=RecordingStream)
    player.play_earcon("wake")
    assert opened_on == [threading.get_ident()]
    assert player.is_open and np.any(f.streams[0].tick() != 0)  # the queued cue plays


async def test_warmup_opens_on_the_calling_thread():
    import threading

    opened_on = []

    class RecordingStream(FakeOutputStream):
        def start(self):
            opened_on.append(threading.get_ident())
            super().start()

    player, f = _player(_settings(), cls=RecordingStream)
    assert not player.is_open
    await player.warmup()
    # same thread as the event loop (the mic's thread): WASAPI output opened from another
    # thread fails on Windows once the mic stream exists (reproduced on the target laptop)
    assert opened_on == [threading.get_ident()]
    assert player.is_open and f.streams[0].started
    await player.close()
    assert f.streams[0].closed


def test_earcons_disabled_does_nothing(tmp_path):
    generate(tmp_path)
    player, f = _player(_settings(tmp_path, earcons=False))
    player.play_earcon("wake")
    assert not player.is_playing
    _join_open(player)
    assert f.streams == []  # device never even opened


def test_missing_earcon_warns_once(tmp_path, caplog):
    player, _f = _player(_settings(tmp_path))  # empty dir
    with caplog.at_level(logging.WARNING, logger="spark.audio.player"):
        player.play_earcon("thinking")
        player.play_earcon("thinking")
    assert not player.is_playing
    assert sum("thinking" in r.getMessage() for r in caplog.records) == 1


def test_earcon_device_failure_is_logged_not_raised(tmp_path, caplog):
    generate(tmp_path)

    def broken(**_kw):
        raise RuntimeError("no speaker")

    player = SoundDevicePlayer(_settings(tmp_path), stream_factory=broken, device_rate=RATE)
    with caplog.at_level(logging.WARNING, logger="spark.audio.player"):
        player.play_earcon("error")
        _join_open(player)
    assert any("no speaker" in r.getMessage() for r in caplog.records)
    assert not player.is_playing  # the stranded cue was dropped


async def test_play_raises_backend_unavailable_without_device():
    def broken(**_kw):
        raise RuntimeError("no speaker")

    player = SoundDevicePlayer(_settings(), stream_factory=broken, device_rate=RATE)
    with pytest.raises(BackendUnavailable, match="no speaker"):
        await player.play(np.ones(100, np.int16), 16_000)


# --- committed earcon assets -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "lo_ms", "hi_ms"), [("wake", 150, 250), ("thinking", 40, 80), ("error", 250, 350)]
)
def test_committed_earcons(name, lo_ms, hi_ms):
    path = ROOT / "assets" / "sounds" / f"{name}.wav"
    assert path.is_file(), "run: uv run python -m spark.audio.earcons"
    assert path.stat().st_size < 20_000
    pcm, rate = read_wav(path)
    assert rate == 16_000
    assert lo_ms <= 1000 * pcm.size / rate <= hi_ms
    # Fades: first and last samples near zero, so no click.
    assert abs(int(pcm[0])) < 200 and abs(int(pcm[-1])) < 200


def test_generators_are_deterministic():
    for fn in (wake_chime, thinking_tick, error_tone):
        a, b = fn(), fn()
        assert a.dtype == np.int16 and np.array_equal(a, b)
