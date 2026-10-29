"""Microphone input, ring buffer, re-chunking and resampling (spec §3, §4).

Internal audio is always 16 kHz mono int16. The mic yields blocks of whatever size the device
delivers; every consumer (Porcupine 512, openWakeWord 1280, Silero 512) owns a
:class:`Rechunker` that turns those blocks into exact frames.

``sounddevice`` is imported lazily inside :class:`SoundDeviceMic` and the probe helpers, so this
module imports cleanly without PortAudio (``--text`` mode, CI).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import sys
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.interfaces import SAMPLE_RATE, AudioInput, BackendUnavailable

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = [
    "MicUnavailable",
    "Rechunker",
    "RingBuffer",
    "SoundDeviceMic",
    "StreamResampler",
    "list_devices",
    "probe_input",
    "resample",
    "resample_to_16k",
    "resolve_device",
    "to_float32",
    "to_int16",
    "to_mono",
]

# --- sample format helpers -------------------------------------------------------------------


def to_float32(pcm: np.ndarray) -> np.ndarray:
    """int16 (or any int/float) samples → float32 in [-1, 1]."""
    arr = np.asarray(pcm)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float64) / float(-info.min)).astype(np.float32)
    return arr.astype(np.float32, copy=False)


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """float in [-1, 1] (clipped) or int samples → int16."""
    arr = np.asarray(pcm)
    if arr.dtype == np.int16:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, -32768, 32767).astype(np.int16)
    return np.clip(np.rint(arr * 32767.0), -32768, 32767).astype(np.int16)


def to_mono(pcm: np.ndarray) -> np.ndarray:
    """Downmix ``(frames, channels)`` to 1-D by averaging channels. 1-D input is returned as is."""
    arr = np.asarray(pcm)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or (frames, channels) audio, got shape {arr.shape}")
    if arr.shape[1] == 1:
        return arr[:, 0]
    mixed = arr.astype(np.float32).mean(axis=1)
    return mixed.astype(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else mixed


# --- resampling ------------------------------------------------------------------------------

_FIR_TAPS = 63


def _lowpass(cutoff: float, taps: int = _FIR_TAPS) -> np.ndarray:
    """Windowed-sinc low-pass FIR. ``cutoff`` is in cycles/sample (0..0.5). Unity DC gain."""
    m = (taps - 1) / 2
    n = np.arange(taps) - m
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(taps)
    return (h / h.sum()).astype(np.float32)


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a whole 1-D signal. Returns float32 (int16 input is scaled to [-1, 1]).

    Downsampling low-pass filters first (cutoff 0.45 * dst) so content above the new Nyquist
    doesn't alias; then linear interpolation places the output samples.
    """
    x = to_float32(to_mono(pcm))
    if src_rate == dst_rate or x.size == 0:
        return x.copy()
    if dst_rate < src_rate:
        x = np.convolve(x, _lowpass(0.45 * dst_rate / src_rate), mode="same").astype(np.float32)
    n_out = round(x.size * dst_rate / src_rate)
    positions = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    return np.interp(positions, np.arange(x.size), x).astype(np.float32)


def resample_to_16k(pcm: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample any-rate audio to 16 kHz mono int16 (for devices that won't open at 16k)."""
    if src_rate == SAMPLE_RATE and np.asarray(pcm).dtype == np.int16:
        return to_mono(pcm)
    return to_int16(resample(pcm, src_rate, SAMPLE_RATE))


class StreamResampler:
    """Block-by-block resampler with carried filter and phase state (no seams between blocks).

    Adds a fixed delay of ``taps // 2`` source samples (~0.7 ms at 48 kHz) when downsampling.
    """

    def __init__(self, src_rate: int, dst_rate: int = SAMPLE_RATE) -> None:
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self._step = self.src_rate / self.dst_rate
        self._fir = (
            _lowpass(0.45 * self.dst_rate / self.src_rate)
            if self.dst_rate < self.src_rate
            else None
        )
        self._history = np.zeros(_FIR_TAPS - 1, dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)  # filtered samples not yet fully consumed
        self._buf_start = 0  # absolute source index of _buf[0]
        self._pos = 0.0  # absolute source position of the next output sample

    def process(self, block: np.ndarray) -> np.ndarray:
        """Feed source samples (any dtype, mono or multi-channel); returns 16-bit output samples."""
        x = to_float32(to_mono(block))
        if self.src_rate == self.dst_rate:
            return to_int16(x)
        if self._fir is not None:
            padded = np.concatenate([self._history, x])
            self._history = padded[-(_FIR_TAPS - 1) :]
            x = np.convolve(padded, self._fir, mode="valid").astype(np.float32)
        self._buf = np.concatenate([self._buf, x])
        last = self._buf_start + self._buf.size - 1  # need idx and idx+1 inside the buffer
        if self._buf.size < 2 or self._pos >= last:
            return np.zeros(0, dtype=np.int16)
        n_out = math.ceil((last - self._pos) / self._step)
        positions = self._pos + np.arange(n_out) * self._step
        rel = positions - self._buf_start
        out = np.interp(rel, np.arange(self._buf.size), self._buf)
        self._pos = float(positions[-1] + self._step)
        drop = max(0, int(self._pos) - self._buf_start)
        drop = min(drop, self._buf.size - 1)
        self._buf = self._buf[drop:]
        self._buf_start += drop
        return to_int16(out)


# --- ring buffer and rechunker ---------------------------------------------------------------


class RingBuffer:
    """Fixed-capacity int16 ring buffer that always holds the most recent ``capacity`` samples."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data = np.zeros(self.capacity, dtype=np.int16)
        self._write = 0  # next write index
        self._count = 0  # valid samples (<= capacity)
        self.total_written = 0

    def __len__(self) -> int:
        return self._count

    def write(self, samples: np.ndarray) -> None:
        x = to_int16(to_mono(samples))
        n = x.size
        self.total_written += n
        if n == 0:
            return
        if n >= self.capacity:
            self._data[:] = x[-self.capacity :]
            self._write = 0
            self._count = self.capacity
            return
        end = self._write + n
        if end <= self.capacity:
            self._data[self._write : end] = x
        else:
            first = self.capacity - self._write
            self._data[self._write :] = x[:first]
            self._data[: n - first] = x[first:]
        self._write = end % self.capacity
        self._count = min(self.capacity, self._count + n)

    def read_latest(self, n: int) -> np.ndarray:
        """Return a copy of the newest ``min(n, len(self))`` samples, oldest first."""
        n = max(0, min(int(n), self._count))
        if n == 0:
            return np.zeros(0, dtype=np.int16)
        start = (self._write - n) % self.capacity
        if start + n <= self.capacity:
            return self._data[start : start + n].copy()
        return np.concatenate([self._data[start:], self._data[: (start + n) - self.capacity]])

    def clear(self) -> None:
        self._write = 0
        self._count = 0


class Rechunker:
    """Accumulates arbitrary-size blocks and emits frames of exactly ``frame_length`` samples."""

    def __init__(self, frame_length: int) -> None:
        if frame_length <= 0:
            raise ValueError("frame_length must be positive")
        self.frame_length = int(frame_length)
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def pending(self) -> int:
        """Samples buffered toward the next frame."""
        return self._pending.size

    def feed(self, block: np.ndarray) -> list[np.ndarray]:
        x = to_int16(to_mono(block))
        buf = np.concatenate([self._pending, x]) if self._pending.size else x
        n_frames = buf.size // self.frame_length
        cut = n_frames * self.frame_length
        frames = [buf[i : i + self.frame_length].copy() for i in range(0, cut, self.frame_length)]
        self._pending = buf[cut:].copy()
        return frames

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.int16)


# --- device selection ------------------------------------------------------------------------


def mic_fix(platform: str | None = None) -> str:
    """What to try when the mic can't be opened, for this OS."""
    platform = platform or sys.platform
    if platform == "darwin":
        return (
            "check System Settings > Privacy & Security > Microphone (turn on Hey Spark, or your "
            "terminal app when running from source) and System Settings > Sound > Input (a Mac "
            "mini or Mac Studio needs an external mic or headset), then quit and reopen"
        )
    if platform == "win32":
        return (
            "check Windows Settings > Privacy & security > Microphone (allow desktop apps), "
            "then run `uv run python -m spark doctor` to list devices and set audio.input_device"
        )
    return (
        "check that a microphone is connected and not muted, then run "
        "`uv run python -m spark doctor` to list devices and set audio.input_device"
    )


class MicUnavailable(BackendUnavailable):
    """The microphone can't be opened. ``fix`` says what to try."""

    def __init__(self, reason: str, fix: str | None = None) -> None:
        super().__init__("audio_in", reason, fix or mic_fix())


def _import_sounddevice(backend: str) -> Any:
    if "sounddevice" not in sys.modules:
        # The macOS app asks for mic access before PortAudio starts and lists the devices.
        from spark.audio import mic_access

        mic_access.ensure()
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
        raise BackendUnavailable(
            backend,
            f"sounddevice/PortAudio not importable ({exc})",
            "uv sync  (sounddevice is a core dependency)",
        ) from exc
    return sd


def _hostapi_index(sd: Any, host_api: str | None) -> int | None:
    if not host_api:
        return None
    for i, api in enumerate(sd.query_hostapis()):
        if host_api.lower() in str(api["name"]).lower():
            return i
    return None


def resolve_device(
    sd: Any, kind: str, device: str | int | None, host_api: str | None
) -> int | None:
    """Pick a device index for ``kind`` ("input" or "output").

    * int → used as is.
    * str → first device whose name contains it (case-insensitive) with channels for ``kind``,
      preferring one on ``host_api``.
    * None → ``host_api``'s default device for ``kind`` if that host API exists, else None
      (PortAudio's system default).
    """
    if isinstance(device, int):
        return device
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    api_idx = _hostapi_index(sd, host_api)
    if isinstance(device, str) and device.strip():
        if device.strip().lstrip("-").isdigit():
            return int(device)
        matches = [
            d
            for d in sd.query_devices()
            if device.lower() in str(d["name"]).lower() and d[channels_key] > 0
        ]
        if not matches:
            raise BackendUnavailable(
                "audio_in" if kind == "input" else "audio_out",
                f"no {kind} device matches {device!r}",
                "run `uv run python -m spark doctor` to list devices, then fix "
                f"audio.{kind}_device in config.yaml",
            )
        preferred = [d for d in matches if api_idx is not None and d["hostapi"] == api_idx]
        return int((preferred or matches)[0]["index"])
    if api_idx is not None:
        default = sd.query_hostapis(api_idx)[f"default_{kind}_device"]
        if default is not None and default >= 0:
            return int(default)
    return None


def wasapi_extra(sd: Any, device: int | None) -> Any:
    """WASAPI auto-convert settings (lets shared mode open at 16 kHz / mono) when applicable."""
    try:
        info = sd.query_devices(device if device is not None else sd.default.device[0])
        api = sd.query_hostapis(info["hostapi"])["name"]
    except Exception:
        return None
    if "wasapi" in str(api).lower() and hasattr(sd, "WasapiSettings"):
        return sd.WasapiSettings(auto_convert=True)
    return None


def list_devices() -> list[dict[str, Any]]:
    """Every PortAudio device with its host API name (for ``spark doctor``)."""
    sd = _import_sounddevice("audio_in")
    apis = sd.query_hostapis()
    out = []
    for d in sd.query_devices():
        out.append(
            {
                "index": int(d["index"]),
                "name": str(d["name"]),
                "hostapi": str(apis[d["hostapi"]]["name"]),
                "inputs": int(d["max_input_channels"]),
                "outputs": int(d["max_output_channels"]),
                "default_samplerate": float(d["default_samplerate"]),
            }
        )
    return out


StreamFactory = Callable[..., Any]
"""``factory(samplerate=, blocksize=, channels=, callback=) -> stream`` with start/stop/close."""


def _open_input(
    factory: StreamFactory,
    native_rate: int,
    max_channels: int,
    block_ms: float,
    callback: Callable[..., None] | None,
) -> tuple[Any, int, int]:
    """Try 16 kHz mono, then native-rate mono, then native-rate multi-channel."""
    candidates = [(SAMPLE_RATE, 1), (native_rate, 1)]
    if max_channels > 1:
        candidates.append((native_rate, max_channels))
    errors: list[str] = []
    seen = set()
    for rate, ch in candidates:
        if (rate, ch) in seen:
            continue
        seen.add((rate, ch))
        blocksize = max(1, round(rate * block_ms / 1000))
        try:
            stream = factory(samplerate=rate, blocksize=blocksize, channels=ch, callback=callback)
        except Exception as exc:
            errors.append(f"{rate} Hz x{ch}: {exc}")
            continue
        if rate != SAMPLE_RATE:
            log.info("mic opened at %d Hz x%d; resampling to 16 kHz", rate, ch)
        return stream, rate, ch
    raise MicUnavailable("could not open the microphone (" + "; ".join(errors) + ")")


def _refresh_portaudio(sd: Any) -> None:
    """Restart PortAudio so it lists the devices again (it reads them once, at start)."""
    with contextlib.suppress(Exception):
        sd._terminate()
        sd._initialize()


def _device_factory(sd: Any, device: int | None) -> tuple[StreamFactory, int, int]:
    from spark.audio import mic_access

    if mic_access.ensure() in ("denied", "restricted"):
        raise MicUnavailable(
            "macOS hasn't given Hey Spark access to the microphone", mic_access.SETTINGS_FIX
        )
    try:
        info = sd.query_devices(device if device is not None else sd.default.device[0])
    except Exception as exc:
        if sys.platform != "darwin" or device is not None:
            raise MicUnavailable(f"no usable input device ({exc})") from exc
        # macOS: the mic may have been allowed or plugged in after PortAudio listed devices.
        _refresh_portaudio(sd)
        try:
            info = sd.query_devices(sd.default.device[0])
        except Exception as exc2:
            raise MicUnavailable(f"no usable input device ({exc2})") from exc2
    if int(info["max_input_channels"]) < 1:
        raise MicUnavailable(f"device {info['name']!r} has no input channels")
    extra = wasapi_extra(sd, device)

    def factory(*, samplerate: int, blocksize: int, channels: int, callback: Any) -> Any:
        return sd.InputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            channels=channels,
            dtype="int16",
            latency="low",
            extra_settings=extra,
            callback=callback,
        )

    return factory, int(info["default_samplerate"]), int(info["max_input_channels"])


def probe_input(settings: Settings, seconds: float = 0.5) -> dict[str, Any]:
    """Open the configured mic, record ``seconds``, close it. Returns device info and level.

    Raises :class:`MicUnavailable` (a :class:`BackendUnavailable`) with a fix on failure.
    """
    sd = _import_sounddevice("audio_in")
    audio = settings.audio
    device = resolve_device(sd, "input", audio.input_device, audio.host_api)
    factory, native, max_ch = _device_factory(sd, device)
    stream, rate, ch = _open_input(factory, native, max_ch, 32.0, None)
    try:
        stream.start()
        data, overflowed = stream.read(max(1, int(rate * seconds)))
        stream.stop()
    except Exception as exc:
        raise MicUnavailable(f"mic opened but reading failed ({exc})") from exc
    finally:
        stream.close()
    pcm = resample_to_16k(np.asarray(data), rate)
    info = sd.query_devices(device if device is not None else sd.default.device[0])
    rms = float(np.sqrt(np.mean(to_float32(pcm) ** 2))) if pcm.size else 0.0
    return {
        "device": int(info["index"]),
        "name": str(info["name"]),
        "hostapi": str(sd.query_hostapis(info["hostapi"])["name"]),
        "samplerate": rate,
        "channels": ch,
        "resampled": rate != SAMPLE_RATE,
        "samples": int(pcm.size),
        "rms": rms,
        "overflowed": bool(overflowed),
    }


# --- microphone ------------------------------------------------------------------------------


class SoundDeviceMic(AudioInput):
    """PortAudio microphone via ``sounddevice``, yielding 16 kHz mono int16 blocks.

    The PortAudio callback only copies the block and hands it to the event loop with
    ``call_soon_threadsafe``. The loop-side queue is bounded; on overflow the oldest block is
    dropped and counted in :attr:`dropped_blocks`. Downmixing and resampling happen in
    :meth:`frames`, on the event loop, never in the callback.

    ``stream_factory`` / ``native_rate`` / ``max_channels`` let tests inject a fake stream.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        stream_factory: StreamFactory | None = None,
        native_rate: int | None = None,
        max_channels: int = 1,
        max_queue_blocks: int = 64,
    ) -> None:
        self.settings = settings
        self._block_ms = 1000.0 * settings.audio.block_size / SAMPLE_RATE
        self._factory = stream_factory
        self._native_rate = native_rate or SAMPLE_RATE
        self._max_channels = max_channels
        self._max_queue = max_queue_blocks
        self._stream: Any = None
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resampler: StreamResampler | None = None
        self.device_rate = SAMPLE_RATE
        self.channels = 1
        self.dropped_blocks = 0
        self.status_flags = 0  # callbacks that reported an input overflow/underflow
        self._reported_drops = 0
        self._reported_status = 0
        self.ring = RingBuffer(max(1, int(settings.audio.ring_seconds * SAMPLE_RATE)))
        if self._factory is None:
            sd = _import_sounddevice("audio_in")
            audio = settings.audio
            device = resolve_device(sd, "input", audio.input_device, audio.host_api)
            self._factory, self._native_rate, self._max_channels = _device_factory(sd, device)

    @property
    def running(self) -> bool:
        return self._stream is not None

    # PortAudio thread: copy + hand off, nothing else.
    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.status_flags += 1
        loop = self._loop
        if loop is None:
            return
        block = indata.copy()
        try:
            loop.call_soon_threadsafe(self._enqueue, block)
        except RuntimeError:  # loop closed during shutdown
            return

    def _enqueue(self, block: np.ndarray | None) -> None:
        q = self._queue
        if q is None:
            return
        if q.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()
                self.dropped_blocks += 1
        q.put_nowait(block)

    async def start(self) -> None:
        if self._stream is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        assert self._factory is not None
        stream, rate, ch = _open_input(
            self._factory, self._native_rate, self._max_channels, self._block_ms, self._callback
        )
        self.device_rate, self.channels = rate, ch
        self._resampler = StreamResampler(rate) if rate != SAMPLE_RATE else None
        try:
            stream.start()
        except Exception as exc:
            stream.close()
            raise MicUnavailable(f"mic opened but failed to start ({exc})") from exc
        self._stream = stream

    async def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        if self._queue is not None:
            self._enqueue(None)  # wake frames() so it can finish

    def _report(self) -> None:
        if self.dropped_blocks != self._reported_drops:
            log.warning("mic queue overflow: %d blocks dropped so far", self.dropped_blocks)
            self._reported_drops = self.dropped_blocks
        if self.status_flags != self._reported_status:
            log.warning("mic stream reported %d overflow/underflow callbacks", self.status_flags)
            self._reported_status = self.status_flags

    def convert(self, block: np.ndarray) -> np.ndarray:
        """Device block → 16 kHz mono int16 (downmix + resample if needed)."""
        mono = to_mono(block)
        if self._resampler is not None:
            return self._resampler.process(mono)
        return to_int16(mono)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        if self._queue is None:
            raise RuntimeError("SoundDeviceMic.frames() called before start()")
        queue = self._queue
        last_report = time.monotonic()
        while True:
            block = await queue.get()
            if block is None:
                if self._stream is None:
                    return
                continue
            pcm = self.convert(block)
            if pcm.size == 0:
                continue
            self.ring.write(pcm)
            now = time.monotonic()
            if now - last_report > 5.0:
                self._report()
                last_report = now
            yield pcm

    def recent(self, ms: float) -> np.ndarray:
        """The newest ``ms`` milliseconds of mic audio (for wake-word pre-roll)."""
        return self.ring.read_latest(int(SAMPLE_RATE * ms / 1000))
