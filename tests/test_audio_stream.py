"""Ring buffer, re-chunking, resampling and the mic callback → queue path (no real device)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from spark.audio.stream import (
    MicUnavailable,
    Rechunker,
    RingBuffer,
    SoundDeviceMic,
    StreamResampler,
    resample,
    resample_to_16k,
    to_float32,
    to_int16,
    to_mono,
)
from spark.audio.wavio import read_wav, write_wav
from spark.config import Settings
from spark.interfaces import BackendUnavailable

ROOT = Path(__file__).resolve().parents[1]


def _settings(**kw) -> Settings:
    return Settings(config_dir=ROOT, **kw)


def _peak_hz(x: np.ndarray, rate: int) -> float:
    spec = np.abs(np.fft.rfft(to_float32(x) * np.hanning(x.size)))
    return float(np.fft.rfftfreq(x.size, 1 / rate)[np.argmax(spec)])


def _tone(freq: float, rate: int, seconds: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- helpers ---------------------------------------------------------------------------------


def test_float_int16_round_trip_and_clipping():
    x = np.array([-1.5, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    y = to_int16(x)
    assert y.dtype == np.int16
    assert y[0] == -32768 and y[-1] == 32767 and y[2] == 0
    back = to_float32(y)
    assert back.dtype == np.float32
    assert np.allclose(back[2:5], [0.0, 0.5, 1.0], atol=1e-4)


def test_stereo_downmix():
    stereo = np.stack([np.full(4, 1000, np.int16), np.full(4, 3000, np.int16)], axis=1)
    mono = to_mono(stereo)
    assert mono.shape == (4,) and mono.dtype == np.int16
    assert np.all(mono == 2000)
    assert to_mono(np.zeros((5, 1), np.int16)).shape == (5,)


# --- ring buffer -----------------------------------------------------------------------------


def test_ring_buffer_wraps_and_keeps_newest():
    rb = RingBuffer(10)
    rb.write(np.arange(7, dtype=np.int16))
    assert list(rb.read_latest(3)) == [4, 5, 6]
    rb.write(np.arange(7, 13, dtype=np.int16))  # wraps
    assert len(rb) == 10
    assert list(rb.read_latest(10)) == list(range(3, 13))
    assert list(rb.read_latest(4)) == [9, 10, 11, 12]
    assert list(rb.read_latest(50)) == list(range(3, 13))  # clamps to what's held
    rb.write(np.arange(100, 125, dtype=np.int16))  # bigger than capacity
    assert list(rb.read_latest(10)) == list(range(115, 125))
    assert rb.total_written == 38
    rb.clear()
    assert rb.read_latest(5).size == 0


def test_ring_buffer_many_small_writes_match_reference():
    rng = np.random.default_rng(0)
    rb = RingBuffer(1000)
    ref = []
    for _ in range(200):
        block = rng.integers(-3000, 3000, size=rng.integers(1, 97)).astype(np.int16)
        rb.write(block)
        ref.extend(block.tolist())
        n = int(rng.integers(0, 1200))
        expected = ref[-min(n, 1000, len(ref)) :] if n else []
        assert rb.read_latest(n).tolist() == expected


# --- rechunker -------------------------------------------------------------------------------


@pytest.mark.parametrize("frame_length", [512, 1280])
@pytest.mark.parametrize("block_sizes", [[512], [1280], [441], [100, 700, 13], [2048], [1, 3, 997]])
def test_rechunker_is_exact(frame_length, block_sizes):
    total = 16_000 * 2 + 77
    stream = (np.arange(total) % 32000).astype(np.int16)
    rc = Rechunker(frame_length)
    out: list[np.ndarray] = []
    pos, i = 0, 0
    while pos < total:
        n = block_sizes[i % len(block_sizes)]
        out.extend(rc.feed(stream[pos : pos + n]))
        pos += n
        i += 1
    assert all(f.size == frame_length and f.dtype == np.int16 for f in out)
    assert len(out) == total // frame_length
    assert rc.pending == total % frame_length
    joined = np.concatenate(out)
    assert np.array_equal(joined, stream[: joined.size])  # no reordering, no gaps


def test_two_consumers_from_one_stream():
    porcupine, oww = Rechunker(512), Rechunker(1280)
    counts = [0, 0]
    for _ in range(40):  # 40 * 480 = 19200 samples
        block = np.ones(480, np.int16)
        counts[0] += len(porcupine.feed(block))
        counts[1] += len(oww.feed(block))
    assert counts == [19200 // 512, 19200 // 1280]
    assert (porcupine.pending, oww.pending) == (19200 % 512, 19200 % 1280)


def test_rechunker_frames_are_independent_copies():
    rc = Rechunker(4)
    block = np.arange(8, dtype=np.int16)
    frames = rc.feed(block)
    block[:] = 0
    assert frames[0].tolist() == [0, 1, 2, 3]


# --- resampling ------------------------------------------------------------------------------


@pytest.mark.parametrize("src", [48_000, 44_100, 22_050, 8_000])
def test_resample_length_and_frequency(src):
    x = _tone(1000.0, src, seconds=1.0)
    y = resample_to_16k(x, src)
    assert y.dtype == np.int16
    assert abs(y.size - 16_000) <= 1
    assert abs(_peak_hz(y, 16_000) - 1000.0) < 5.0


def test_resample_attenuates_content_above_new_nyquist():
    # 12 kHz at 48 kHz would alias to 4 kHz at 16 kHz without the low-pass filter.
    x = _tone(12_000.0, 48_000, seconds=0.5)
    y = to_float32(resample_to_16k(x, 48_000))
    rms_in = float(np.sqrt(np.mean(x**2)))
    rms_out = float(np.sqrt(np.mean(y[200:-200] ** 2)))
    assert rms_out < rms_in * 0.01  # > 40 dB down


def test_resample_identity_and_upsample():
    x = to_int16(_tone(440.0, 16_000))
    assert np.array_equal(resample_to_16k(x, 16_000), x)
    up = resample(x, 16_000, 48_000)
    assert up.dtype == np.float32 and abs(up.size - 3 * x.size) <= 1
    assert abs(_peak_hz(up, 48_000) - 440.0) < 5.0


@pytest.mark.parametrize("src", [48_000, 44_100])
def test_stream_resampler_blockwise_matches_rate_and_has_no_seams(src):
    x = _tone(1000.0, src, seconds=1.0)
    rs = StreamResampler(src)
    rng = np.random.default_rng(1)
    parts, pos = [], 0
    while pos < x.size:
        n = int(rng.integers(100, 2000))
        parts.append(rs.process(to_int16(x[pos : pos + n])))
        pos += n
    y = np.concatenate(parts)
    assert abs(y.size - 16_000) <= 2
    assert abs(_peak_hz(y, 16_000) - 1000.0) < 5.0
    # A clean sine has small sample-to-sample steps; a seam would show up as a spike.
    body = to_float32(y[100:-100])
    max_step = 0.5 * 2 * np.pi * 1000 / 16_000
    assert np.max(np.abs(np.diff(body))) < max_step * 1.1


def test_wav_round_trip(tmp_path):
    x = to_int16(_tone(300.0, 16_000, seconds=0.1))
    path = write_wav(tmp_path / "sub" / "a.wav", x, 16_000)
    y, rate = read_wav(path)
    assert rate == 16_000 and np.array_equal(x, y)


# --- mic with a fake stream ------------------------------------------------------------------


class FakeInputStream:
    """Stands in for sounddevice.InputStream; tests call ``push`` from a worker thread."""

    def __init__(self, *, samplerate, blocksize, channels, callback):
        self.samplerate, self.blocksize, self.channels = samplerate, blocksize, channels
        self.callback = callback
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def push(self, block: np.ndarray, status: int = 0):
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        self.callback(block, block.shape[0], None, status)


class FakeFactory:
    def __init__(self, reject_16k: bool = False):
        self.reject_16k = reject_16k
        self.streams: list[FakeInputStream] = []
        self.attempts: list[tuple[int, int]] = []

    def __call__(self, *, samplerate, blocksize, channels, callback):
        self.attempts.append((samplerate, channels))
        if self.reject_16k and samplerate == 16_000:
            raise RuntimeError("Invalid sample rate")
        s = FakeInputStream(
            samplerate=samplerate, blocksize=blocksize, channels=channels, callback=callback
        )
        self.streams.append(s)
        return s


async def _collect(mic: SoundDeviceMic, n: int) -> list[np.ndarray]:
    out = []
    async for block in mic.frames():
        out.append(block)
        if len(out) == n:
            break
    return out


async def test_mic_callback_enqueues_from_another_thread():
    factory = FakeFactory()
    mic = SoundDeviceMic(_settings(), stream_factory=factory)
    await mic.start()
    stream = factory.streams[0]
    assert stream.started and stream.samplerate == 16_000 and stream.blocksize == 512
    blocks = [np.full(512, i, np.int16) for i in range(5)]
    t = threading.Thread(target=lambda: [stream.push(b.copy()) for b in blocks])
    t.start()
    got = await asyncio.wait_for(_collect(mic, 5), timeout=2)
    t.join()
    assert [int(b[0]) for b in got] == [0, 1, 2, 3, 4]
    assert all(b.dtype == np.int16 and b.ndim == 1 for b in got)
    assert list(mic.recent(1000)[-512:]) == [4] * 512  # ring buffer tracks the stream
    await mic.stop()
    assert stream.closed and not mic.running


async def test_mic_callback_copies_block():
    factory = FakeFactory()
    mic = SoundDeviceMic(_settings(), stream_factory=factory)
    await mic.start()
    buf = np.full((512, 1), 7, np.int16)
    factory.streams[0].push(buf)
    buf[:] = 0  # PortAudio reuses its buffer after the callback returns
    got = await asyncio.wait_for(_collect(mic, 1), timeout=1)
    assert int(got[0][0]) == 7
    await mic.stop()


async def test_mic_queue_overflow_drops_oldest_and_counts():
    factory = FakeFactory()
    mic = SoundDeviceMic(_settings(), stream_factory=factory, max_queue_blocks=4)
    await mic.start()
    stream = factory.streams[0]
    for i in range(10):
        stream.push(np.full(512, i, np.int16), status=1 if i == 0 else 0)
    await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run
    await asyncio.sleep(0)
    assert mic.dropped_blocks == 6
    assert mic.status_flags == 1
    got = await asyncio.wait_for(_collect(mic, 4), timeout=1)
    assert [int(b[0]) for b in got] == [6, 7, 8, 9]  # newest survive
    await mic.stop()


async def test_mic_falls_back_to_native_rate_and_resamples():
    factory = FakeFactory(reject_16k=True)
    mic = SoundDeviceMic(_settings(), stream_factory=factory, native_rate=48_000, max_channels=2)
    await mic.start()
    assert factory.attempts[:2] == [(16_000, 1), (48_000, 1)]
    assert mic.device_rate == 48_000
    stream = factory.streams[0]
    assert stream.blocksize == 1536  # same 32 ms block at the native rate
    x = to_int16(_tone(1000.0, 48_000, seconds=1.0))
    for i in range(0, x.size, 1536):
        stream.push(x[i : i + 1536])
    got = await asyncio.wait_for(_collect(mic, -(-x.size // 1536)), timeout=2)
    y = np.concatenate(got)
    assert abs(y.size - 16_000) <= 2
    assert abs(_peak_hz(y, 16_000) - 1000.0) < 5.0
    await mic.stop()


async def test_mic_stop_ends_frames_iterator():
    mic = SoundDeviceMic(_settings(), stream_factory=FakeFactory())
    await mic.start()

    async def drain():
        return [b async for b in mic.frames()]

    task = asyncio.create_task(drain())
    await asyncio.sleep(0)
    await mic.stop()
    assert await asyncio.wait_for(task, timeout=1) == []


async def test_mic_unavailable_when_nothing_opens():
    def broken(**_kw):
        raise RuntimeError("device busy")

    mic = SoundDeviceMic(_settings(), stream_factory=broken, native_rate=44_100)
    with pytest.raises(MicUnavailable) as exc:
        await mic.start()
    assert isinstance(exc.value, BackendUnavailable)
    assert "device busy" in str(exc.value) and exc.value.fix


def test_import_does_not_pull_in_sounddevice():
    import subprocess
    import sys

    code = (
        "import sys, spark.audio.stream, spark.audio.player, spark.audio.recorder;"
        "sys.exit(1 if 'sounddevice' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


@pytest.mark.hardware
async def test_real_default_mic_half_second():
    mic = SoundDeviceMic(_settings())
    await mic.start()
    try:
        total = 0

        async def read():
            nonlocal total
            async for block in mic.frames():
                total += block.size
                if total >= 8000:
                    return

        await asyncio.wait_for(read(), timeout=3)
    finally:
        await mic.stop()
    assert total >= 8000
