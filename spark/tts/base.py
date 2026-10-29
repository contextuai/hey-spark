"""Shared sentence-pipelined TTS (spec §4, §9).

:class:`SentenceTTS` turns a stream of text chunks (normally one sentence each, from
:func:`spark.util.sentences.split_sentences`) into speech:

* a *producer* task synthesizes each chunk in ``asyncio.to_thread`` and queues the audio;
* a *consumer* task hands queued audio to the :class:`~spark.interfaces.AudioOutput`.

So sentence N+1 is synthesized while sentence N plays, and the first audio starts after only
the first sentence's synthesis. When the player offers ``play_stream(chunks, sample_rate)``
(``SoundDevicePlayer`` does) sentences are played gaplessly through one stream; otherwise each
is passed to ``play(pcm, sample_rate)``. The player resamples to the device rate.

:meth:`SentenceTTS.stop` stops the player, cancels pending synthesis and makes :meth:`speak`
return promptly; chunks that arrive afterwards are never synthesized. A synthesis already
running in a worker thread can't be interrupted, but its result is discarded.

``on_first_audio`` (a dep, or set the attribute per turn) is called once per :meth:`speak`
when the first audio is handed to the player, so the app's ``TurnTimer`` can mark
``first_audio``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.audio.stream import resample, to_int16, to_mono
from spark.interfaces import TTS, AudioOutput, BackendUnavailable

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["SentenceTTS"]


class _Failed:
    """Queue marker carrying a synthesis error from the producer to the consumer."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_END = object()


class SentenceTTS(TTS):
    """Base for backends that synthesize one chunk of text at a time.

    Subclasses implement :meth:`synthesize` (blocking; runs in a worker thread) and may
    override :meth:`load` (blocking voice/model load, called by :meth:`warmup`).
    """

    backend: str = "tts"
    lookahead: int = 2  # synthesized sentences allowed to wait for the player
    warmup_text: str = "Ready."
    cleanup_timeout_s: float = 0.05

    def __init__(
        self,
        settings: Settings,
        *,
        player: AudioOutput | None = None,
        on_first_audio: Callable[[], None] | None = None,
        **deps: Any,
    ) -> None:
        # player may be None for synthesis-only use (``spark bench tts``); speak() needs one.
        self.settings = settings
        self.player = player
        self.on_first_audio = on_first_audio
        self.deps = deps
        self.first_audio_at: float | None = None
        self.speaking = False
        self._stop_event: asyncio.Event | None = None
        self._synth_lock = threading.Lock()

    # --- subclass API -------------------------------------------------------------------------

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Blocking: speech for ``text`` as ``(int16 mono samples, sample_rate)``."""

    def load(self) -> Any:
        """Blocking: load the voice/model (called from a worker thread by :meth:`warmup`).

        Returns the loaded engine; backends with nothing to load return None.
        """
        return None

    # --- helpers --------------------------------------------------------------------------------

    def _synthesize_locked(self, text: str) -> tuple[np.ndarray, int]:
        with self._synth_lock:
            pcm, rate = self.synthesize(text)
        return to_int16(to_mono(np.asarray(pcm))), int(rate)

    def render(self, text: str) -> tuple[np.ndarray, int]:
        """Blocking: ``text`` as int16 mono audio, not played (a clip to play later)."""
        return self._synthesize_locked(text)

    def _warm_blocking(self) -> None:
        self.load()
        self._synthesize_locked(self.warmup_text)

    def _mark_first_audio(self) -> None:
        if self.first_audio_at is not None:
            return
        self.first_audio_at = time.perf_counter()
        if self.on_first_audio is not None:
            try:
                self.on_first_audio()
            except Exception:
                log.exception("on_first_audio callback failed")

    # --- pipeline -------------------------------------------------------------------------------

    async def _produce(self, text_chunks: AsyncIterator[str], queue: asyncio.Queue[Any]) -> None:
        try:
            async for chunk in text_chunks:
                text = " ".join(str(chunk).split())
                if not text:
                    continue
                pcm, rate = await asyncio.to_thread(self._synthesize_locked, text)
                if pcm.size:
                    await queue.put((pcm, rate))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(_Failed(exc))
            return
        await queue.put(_END)

    @staticmethod
    async def _next(
        queue: asyncio.Queue[Any], stop: asyncio.Event
    ) -> tuple[np.ndarray, int] | None:
        """Next synthesized sentence; None at the end of the stream or once ``stop`` is set.

        Watching ``stop`` directly (not only relying on task cancellation) matters: on Python
        3.11 ``asyncio.wait_for`` can swallow a cancel that races with the inner wait finishing,
        which is exactly what happens when ``player.stop()`` and our cancel land together.
        """
        if stop.is_set():
            return None
        get = asyncio.ensure_future(queue.get())
        stopped = asyncio.ensure_future(stop.wait())
        try:
            await asyncio.wait({get, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopped.cancel()
            if not get.done():
                get.cancel()
        if stop.is_set() or get.cancelled():
            return None
        item = get.result()
        if item is _END:
            return None
        if isinstance(item, _Failed):
            raise item.exc
        return item

    async def _consume(self, queue: asyncio.Queue[Any], stop: asyncio.Event) -> None:
        play_stream = getattr(self.player, "play_stream", None)
        if play_stream is None:
            while (item := await self._next(queue, stop)) is not None:
                pcm, rate = item
                self._mark_first_audio()
                await self.player.play(pcm, rate)
            return

        first = await self._next(queue, stop)
        if first is None:
            return
        stream_rate = first[1]

        async def chunks() -> AsyncIterator[np.ndarray]:
            item: tuple[np.ndarray, int] | None = first
            while item is not None:
                pcm, rate = item
                if rate != stream_rate:
                    pcm = to_int16(resample(pcm, rate, stream_rate))
                self._mark_first_audio()
                yield pcm
                item = await self._next(queue, stop)

        await play_stream(chunks(), stream_rate)

    # --- TTS ------------------------------------------------------------------------------------

    async def warmup(self) -> None:
        """Load the voice and synthesize a short phrase off the event loop."""
        await asyncio.to_thread(self._warm_blocking)

    async def speak(self, text_chunks: AsyncIterator[str]) -> None:
        if self.player is None:
            raise BackendUnavailable(
                self.backend,
                "no audio output was provided",
                "build TTS through spark.registry.build('tts', settings, player=<AudioOutput>)",
            )
        stop = asyncio.Event()
        self._stop_event = stop
        self.first_audio_at = None
        self.speaking = True
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.lookahead)
        producer = asyncio.create_task(self._produce(text_chunks, queue), name="tts-synth")
        consumer = asyncio.create_task(self._consume(queue, stop), name="tts-play")
        stop_wait = asyncio.create_task(stop.wait(), name="tts-stop-wait")
        try:
            await asyncio.wait({consumer, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
            if consumer.done() and not stop.is_set():
                consumer.result()  # re-raise a synthesis or playback error
        finally:
            tasks = (producer, consumer, stop_wait)
            for task in tasks:
                task.cancel()
            # Bounded: a task that swallowed its cancel must not keep speak() from returning.
            _, pending = await asyncio.wait(tasks, timeout=self.cleanup_timeout_s)
            for task in pending:
                log.warning(
                    "TTS task %s did not stop within %.2f s",
                    task.get_name(),
                    self.cleanup_timeout_s,
                )
            for task in tasks:
                if task.done() and not task.cancelled():
                    task.exception()  # mark retrieved; errors were re-raised above
            self.speaking = False
            if self._stop_event is stop:
                self._stop_event = None

    async def stop(self) -> None:
        """Stop playback now, cancel pending synthesis, and make :meth:`speak` return."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self.player is not None:
            self.player.stop()

    async def close(self) -> None:
        await self.stop()
