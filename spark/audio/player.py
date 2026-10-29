"""Interruptible speaker output and earcons (spec §3, §6).

One long-lived PortAudio output stream runs at the device's native rate with small blocks
(20 ms). Its callback pulls from a :class:`PlaybackBuffer`, which mixes any active sources
(speech plus an earcon, say). :meth:`PlaybackBuffer.stop` clears every source under a lock, so
the very next callback block is silent: stop latency is one output block plus the device's
own buffer, far under the 300 ms budget.

``sounddevice`` is imported lazily in :class:`SoundDevicePlayer` when the stream is opened.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import AsyncIterable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.audio.stream import (
    _import_sounddevice,
    resample,
    resolve_device,
    to_float32,
    to_mono,
    wasapi_extra,
)
from spark.interfaces import AudioOutput, BackendUnavailable

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["EARCONS", "PlaybackBuffer", "PlaybackSource", "SoundDevicePlayer"]

EARCONS = ("wake", "thinking", "error")
BLOCK_MS = 20  # output callback block; bounds stop latency


class PlaybackSource:
    """One queued sound. Chunks can be appended while it plays (streamed TTS)."""

    def __init__(self, on_done: Callable[[], None] | None = None) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0  # read position inside _chunks[0]
        self.closed = False  # no more chunks will be appended
        self.cancelled = False
        self.done = False
        self.samples_played = 0
        self._on_done = on_done

    def pull(self, n: int) -> np.ndarray:
        """Up to ``n`` float32 samples (fewer when the queued audio runs out)."""
        parts: list[np.ndarray] = []
        need = n
        while need > 0 and self._chunks:
            head = self._chunks[0]
            take = min(need, head.size - self._offset)
            parts.append(head[self._offset : self._offset + take])
            self._offset += take
            need -= take
            if self._offset >= head.size:
                self._chunks.popleft()
                self._offset = 0
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self.samples_played += out.size
        return out

    @property
    def exhausted(self) -> bool:
        return self.closed and not self._chunks

    def _finish(self) -> None:
        if self.done:
            return
        self.done = True
        if self._on_done is not None:
            self._on_done()


class PlaybackBuffer:
    """Thread-safe mixer the output callback reads from. Pure numpy, no device needed."""

    def __init__(self, channels: int = 1) -> None:
        self.channels = channels
        self._lock = threading.Lock()
        self._sources: list[PlaybackSource] = []
        self.stop_count = 0

    @property
    def active(self) -> bool:
        with self._lock:
            return any(not s.done for s in self._sources)

    def add(
        self,
        samples: np.ndarray | None = None,
        *,
        closed: bool = True,
        on_done: Callable[[], None] | None = None,
    ) -> PlaybackSource:
        """Queue float32 mono samples (already at the output rate). Returns the source."""
        src = PlaybackSource(on_done)
        if samples is not None and samples.size:
            src._chunks.append(np.asarray(samples, dtype=np.float32))
        src.closed = closed
        with self._lock:
            self._sources.append(src)
        return src

    def append(self, src: PlaybackSource, samples: np.ndarray) -> bool:
        """Add more audio to an open source. False if it was stopped or already closed."""
        with self._lock:
            if src.cancelled or src.closed:
                return False
            if samples.size:
                src._chunks.append(np.asarray(samples, dtype=np.float32))
            return True

    def close(self, src: PlaybackSource) -> None:
        """Mark an open source complete; it finishes once its queued audio has played."""
        finished = False
        with self._lock:
            src.closed = True
            if not src._chunks and src in self._sources:
                self._sources.remove(src)
                finished = True
        if finished:
            src._finish()

    def cancel(self, src: PlaybackSource) -> None:
        """Stop one source immediately."""
        with self._lock:
            src.cancelled = True
            src.closed = True
            src._chunks.clear()
            if src in self._sources:
                self._sources.remove(src)
        src._finish()

    def stop(self) -> None:
        """Silence everything. The next :meth:`fill` outputs zeros. Never blocks for long."""
        with self._lock:
            stopped, self._sources = self._sources, []
            for s in stopped:
                s.cancelled = True
                s.closed = True
                s._chunks.clear()
            self.stop_count += 1
        for s in stopped:
            s._finish()

    def fill(self, outdata: np.ndarray) -> None:
        """Write the next block into ``outdata`` (``(frames,)`` or ``(frames, channels)``)."""
        frames = outdata.shape[0]
        mix = np.zeros(frames, dtype=np.float32)
        finished: list[PlaybackSource] = []
        with self._lock:
            keep: list[PlaybackSource] = []
            for src in self._sources:
                chunk = src.pull(frames)
                mix[: chunk.size] += chunk
                if src.exhausted:
                    finished.append(src)
                else:
                    keep.append(src)
            self._sources = keep
        np.clip(mix, -1.0, 1.0, out=mix)
        if outdata.ndim == 1:
            outdata[:] = mix
        else:
            outdata[:] = mix[:, None]
        for src in finished:
            src._finish()


def _loop_future(
    loop: asyncio.AbstractEventLoop,
) -> tuple[asyncio.Future[None], Callable[[], None]]:
    fut: asyncio.Future[None] = loop.create_future()

    def resolve() -> None:
        if not fut.done():
            fut.set_result(None)

    def on_done() -> None:  # may run on the PortAudio thread
        try:
            loop.call_soon_threadsafe(resolve)
        except RuntimeError:  # loop closed
            return

    return fut, on_done


class SoundDevicePlayer(AudioOutput):
    """Interruptible playback through ``sounddevice``.

    ``stream_factory(samplerate=, blocksize=, channels=, callback=)`` and ``device_rate`` let
    tests replace the device with a fake stream.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        stream_factory: Callable[..., Any] | None = None,
        device_rate: int | None = None,
        channels: int | None = None,
    ) -> None:
        self.settings = settings
        self._factory = stream_factory
        self._device_rate = device_rate
        self._channels = channels
        self.buffer = PlaybackBuffer(channels or 1)
        self._stream: Any = None
        self._open_lock = threading.Lock()
        self._earcons: dict[str, np.ndarray] = {}
        self._warned: set[str] = set()
        self._max_out = 2  # updated from the device when the real device factory is used
        self._open_thread: threading.Thread | None = None

    # --- device ---------------------------------------------------------------------------

    @property
    def device_rate(self) -> int:
        """Output sample rate. Resolving it only queries the device; it doesn't open it."""
        self._resolve()
        assert self._device_rate is not None
        return self._device_rate

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        self.buffer.fill(outdata)

    def _resolve(self) -> None:
        """Pick the device, rate and stream factory (fast; no stream is opened)."""
        if self._factory is not None and self._device_rate is not None:
            return
        with self._open_lock:
            if self._factory is None:
                self._factory = self._device_factory()
            if self._device_rate is None:
                self._device_rate = 48_000

    def _device_factory(self) -> Callable[..., Any]:
        sd = _import_sounddevice("audio_out")
        audio = self.settings.audio
        device = resolve_device(sd, "output", audio.output_device, audio.host_api)
        try:
            info = sd.query_devices(device if device is not None else sd.default.device[1])
        except Exception as exc:
            raise BackendUnavailable(
                "audio_out",
                f"no usable output device ({exc})",
                "plug in or enable a speaker, then run `uv run python -m spark doctor`",
            ) from exc
        if self._device_rate is None:
            self._device_rate = int(info["default_samplerate"])
        self._max_out = int(info["max_output_channels"])
        extra = wasapi_extra(sd, device)

        def factory(*, samplerate: int, blocksize: int, channels: int, callback: Any) -> Any:
            return sd.OutputStream(
                samplerate=samplerate,
                blocksize=blocksize,
                device=device,
                channels=channels,
                dtype="float32",
                latency="low",
                extra_settings=extra,
                callback=callback,
            )

        return factory

    def ensure_open(self) -> None:
        """Open and start the output stream if needed (blocking; ~0.1-0.7 s on WASAPI).

        Raises :class:`BackendUnavailable` on failure. Call :meth:`warmup` at startup.
        """
        if self._stream is not None:
            return
        self._resolve()
        with self._open_lock:
            if self._stream is not None:
                return
            assert self._factory is not None and self._device_rate is not None
            rate = self._device_rate
            blocksize = max(1, rate * BLOCK_MS // 1000)
            attempts = [self._channels] if self._channels else [1, max(2, self._max_out)]
            errors: list[str] = []
            for ch in attempts:
                try:
                    stream = self._factory(
                        samplerate=rate, blocksize=blocksize, channels=ch, callback=self._callback
                    )
                    stream.start()
                except Exception as exc:
                    errors.append(f"x{ch}: {exc}")
                    continue
                self.buffer.channels = ch
                self._stream = stream
                return
            raise BackendUnavailable(
                "audio_out",
                "could not open the speaker (" + "; ".join(errors) + ")",
                "check the output device in Windows sound settings, set audio.output_device, "
                "then run `uv run python -m spark doctor`",
            )

    async def warmup(self) -> None:
        """Open the output stream now, on the calling (event-loop) thread, so the first sound
        isn't delayed.

        Deliberately *not* in a worker thread: on Windows, once a PortAudio stream (the mic)
        has been opened on the event-loop thread, opening the WASAPI output from another
        thread fails ("Unanticipated host error ... WdmSyncIoctl"). Reproduced on the target
        laptop. Keeping every stream open on one thread avoids it; it costs ~0.6 s once, at
        startup.
        """
        self.ensure_open()

    def _open_in_background(self) -> None:
        """Open the stream for a cue queued before :meth:`warmup` (same thread; see warmup)."""
        if self._stream is not None:
            return
        self._background_open()

    def _background_open(self) -> None:
        try:
            self.ensure_open()
        except BackendUnavailable as exc:
            self.buffer.stop()  # nothing will ever drain the queued earcon
            self._warn_once("device", "speaker unavailable, earcon dropped: %s", exc)

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _prepare(self, pcm: np.ndarray, sample_rate: int, gain: float = 1.0) -> np.ndarray:
        x = resample(to_float32(to_mono(pcm)), int(sample_rate), self.device_rate)
        if gain != 1.0:
            x = x * np.float32(gain)
        return x

    # --- AudioOutput ----------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self.buffer.active

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        samples = self._prepare(pcm, sample_rate)
        if samples.size == 0:
            return
        if self._stream is None:
            await self.warmup()
        fut, on_done = _loop_future(asyncio.get_running_loop())
        src = self.buffer.add(samples, on_done=on_done)
        try:
            await fut
        except asyncio.CancelledError:
            self.buffer.cancel(src)
            raise

    async def play_stream(self, chunks: AsyncIterable[np.ndarray], sample_rate: int) -> None:
        """Play chunks gaplessly as they arrive. Returns when done or after :meth:`stop`."""
        if self._stream is None:
            await self.warmup()
        fut, on_done = _loop_future(asyncio.get_running_loop())
        src = self.buffer.add(None, closed=False, on_done=on_done)
        try:
            async for chunk in chunks:
                if not self.buffer.append(src, self._prepare(chunk, sample_rate)):
                    break
            self.buffer.close(src)
            await fut
        except asyncio.CancelledError:
            self.buffer.cancel(src)
            raise

    def _load_earcon(self, name: str) -> np.ndarray | None:
        if name in self._earcons:
            return self._earcons[name]
        from spark.audio.wavio import read_wav

        path = Path(self.settings.path(self.settings.app.sounds_dir)) / f"{name}.wav"
        if not path.is_file():
            self._warn_once(
                name,
                "earcon %r missing at %s (regenerate: uv run python -m spark.audio.earcons)",
                name,
                path,
            )
            return None
        pcm, rate = read_wav(path)
        samples = self._prepare(pcm, rate, gain=float(self.settings.audio.earcon_volume))
        self._earcons[name] = samples
        return samples

    def play_earcon(self, name: str) -> None:
        """Queue a cue and return at once.

        If the stream isn't open yet it is opened first, on this thread (~0.6 s once; the app
        calls :meth:`warmup` at startup so that never happens mid-conversation).
        """
        if not self.settings.app.earcons:
            return
        try:
            samples = self._load_earcon(name)
        except BackendUnavailable as exc:
            self._warn_once("device", "earcon %r not played: %s", name, exc)
            return
        if samples is None or samples.size == 0:
            return
        self.buffer.add(samples)
        if self._stream is None:
            self._open_in_background()

    def play_clip(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Queue a short prepared sound (the spoken wake reply) and return at once, like
        :meth:`play_earcon` but at speech volume and whatever ``app.earcons`` says."""
        samples = self._prepare(pcm, sample_rate)
        if samples.size == 0:
            return
        self.buffer.add(samples)
        if self._stream is None:
            self._open_in_background()

    def stop(self) -> None:
        self.buffer.stop()

    async def close(self) -> None:
        self.buffer.stop()
        thread = self._open_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 5.0)
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
