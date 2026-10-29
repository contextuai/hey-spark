"""Test doubles for every Spark component (CLAUDE.md "Testing rules", spec §11).

Every fake subclasses the matching ABC in :mod:`spark.interfaces`, is deterministic, and
never touches real hardware, models, or the network. Timing-sensitive fakes accept an
injectable clock (:class:`FakeClock`) so tests don't need real sleeps.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

import numpy as np

from spark.brain.confirm import Confirmer
from spark.brain.mock import MockBrain
from spark.config import Settings
from spark.interfaces import (
    SAMPLE_RATE,
    STT,
    TTS,
    VAD,
    AudioInput,
    AudioOutput,
    Brain,
    WakeWordDetector,
)
from spark.util.clock import Clock

__all__ = [
    "SAMPLE_RATE",
    "FakeClock",
    "FakeConfirmer",
    "FakeMic",
    "FakePlayer",
    "FakeSTT",
    "FakeTTS",
    "FakeVAD",
    "FakeWakeWord",
    "MockBrain",
    "ToolCallingBrain",
    "concat",
    "make_settings",
    "rms",
    "silence",
    "speech_like",
    "tone",
]


# --- audio helpers ---------------------------------------------------------------------------


def _n(seconds: float) -> int:
    return max(0, round(seconds * SAMPLE_RATE))


def silence(seconds: float) -> np.ndarray:
    """Digital silence, 16 kHz int16."""
    return np.zeros(_n(seconds), dtype=np.int16)


def tone(seconds: float, freq: float = 440.0, amplitude: int = 8000) -> np.ndarray:
    """A pure sine, 16 kHz int16 (RMS is ``amplitude / sqrt(2)``)."""
    t = np.arange(_n(seconds), dtype=np.float64) / SAMPLE_RATE
    return np.round(amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def speech_like(seconds: float, amplitude: int = 5000, seed: int = 0) -> np.ndarray:
    """Seeded noise with a ~4 Hz syllable envelope: loud enough that every 32 ms frame is
    "speech" to :class:`FakeVAD` (frame RMS stays well above 500)."""
    n = _n(seconds)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.4 + 0.6 * np.abs(np.sin(np.pi * 4.0 * t))
    noise = rng.normal(0.0, amplitude, n) * envelope
    return np.clip(np.round(noise), -32768, 32767).astype(np.int16)


def concat(*arrays: np.ndarray) -> np.ndarray:
    """Join audio clips into one 16 kHz int16 array."""
    if not arrays:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate([np.asarray(a, dtype=np.int16).ravel() for a in arrays])


def rms(frame: np.ndarray) -> float:
    """Root-mean-square level of an int16 frame (0 for an empty frame)."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


def make_settings(**overrides: Any) -> Settings:
    """``Settings`` with the mock brain selected, plus ``overrides`` (nested dicts merge
    into the ``brain`` section)."""
    brain = {"backend": "mock", **overrides.pop("brain", {})}
    return Settings(brain=brain, **overrides)


# --- clock -----------------------------------------------------------------------------------


class FakeClock:
    """Virtual time implementing :class:`spark.util.clock.Clock`.

    ``sleep(s)`` advances time by ``s`` instantly and yields control once, so loops that
    wait on the clock still let other tasks run.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


async def _interruptible_sleep(seconds: float, event: asyncio.Event, clock: Clock | None) -> bool:
    """Sleep ``seconds`` (virtual when ``clock`` is given) unless ``event`` fires first.

    Returns True if interrupted.
    """
    if event.is_set():
        return True
    if seconds <= 0:
        await asyncio.sleep(0)
        return event.is_set()
    if clock is not None:
        await clock.sleep(seconds)
        return event.is_set()
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


# --- microphone ------------------------------------------------------------------------------


class FakeMic(AudioInput):
    """Yields pre-loaded (and live-pushed) audio as blocks.

    Args:
        audio: a 1-D int16 array (cut into ``block_size`` blocks, the last one zero-padded)
            or a list of blocks (yielded exactly as given).
        block_size: samples per block for array input.
        realtime: pace blocks at their real duration (via ``clock`` if given, else asyncio).
            Otherwise each block still yields control once so other tasks progress.
        hold_open: when the audio runs out, wait for :meth:`push` or :meth:`stop` instead
            of ending the stream.
    """

    def __init__(
        self,
        audio: np.ndarray | Sequence[np.ndarray] | Settings | None = None,
        *,
        block_size: int = 512,
        realtime: bool = False,
        clock: Clock | None = None,
        hold_open: bool = False,
        settings: Settings | None = None,
        **deps: Any,
    ) -> None:
        if isinstance(audio, Settings):  # built through the registry: FakeMic(settings)
            settings, audio = audio, None
        self.settings = settings
        self.deps = deps
        self.block_size = block_size
        self.realtime = realtime
        self.clock = clock
        self.hold_open = hold_open
        self.started = False
        self.stopped = False
        self.blocks_yielded = 0
        self.samples_yielded = 0
        self._queue: deque[np.ndarray] = deque()
        self._wakeup = asyncio.Event()
        if audio is not None:
            self.push(audio)

    def _blocks(self, audio: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            pcm = audio.astype(np.int16, copy=False)
            out = []
            for i in range(0, len(pcm), self.block_size):
                block = pcm[i : i + self.block_size]
                if len(block) < self.block_size:
                    block = np.concatenate(
                        [block, np.zeros(self.block_size - len(block), dtype=np.int16)]
                    )
                out.append(block.copy())
            return out
        return [np.asarray(b, dtype=np.int16).ravel().copy() for b in audio]

    def push(self, audio: np.ndarray | Sequence[np.ndarray]) -> None:
        """Append more audio to the stream (wakes a ``hold_open`` consumer)."""
        self._queue.extend(self._blocks(audio))
        self._wakeup.set()

    @property
    def pending_blocks(self) -> int:
        return len(self._queue)

    async def start(self) -> None:
        self.started = True
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True
        self._wakeup.set()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while not self.stopped:
            if self._queue:
                block = self._queue.popleft()
                self.blocks_yielded += 1
                self.samples_yielded += len(block)
                yield block
                if self.realtime:
                    seconds = len(block) / self.sample_rate
                    if self.clock is not None:
                        await self.clock.sleep(seconds)
                    else:
                        await asyncio.sleep(seconds)
                else:
                    await asyncio.sleep(0)
            elif self.hold_open:
                self._wakeup.clear()
                await self._wakeup.wait()
            else:
                return


# --- wake word and VAD -----------------------------------------------------------------------


def _check_length(name: str, frame: np.ndarray, expected: int) -> None:
    if len(frame) != expected:
        raise ValueError(f"{name} expects {expected}-sample frames, got {len(frame)}")


class FakeWakeWord(WakeWordDetector):
    """Fires on chosen frame indices, on demand, or on a marker tone.

    Frame indices count every :meth:`process` call over the detector's lifetime (they are
    not reset by :meth:`reset`). Frames must be exactly ``frame_length`` samples, which
    catches re-chunking bugs.

    Args:
        fire_at: zero-based call indices that return True.
        frame_length: samples per frame.
        marker_hz: if set, also fire when a frame's RMS exceeds ``rms_threshold`` and its
            dominant frequency is within ``marker_tolerance_hz`` of ``marker_hz``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        fire_at: Iterable[int] = (),
        frame_length: int = 512,
        marker_hz: float | None = None,
        rms_threshold: float = 2000.0,
        marker_tolerance_hz: float = 50.0,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        self.fire_at = set(fire_at)
        self.frame_length = frame_length
        self.marker_hz = marker_hz
        self.rms_threshold = rms_threshold
        self.marker_tolerance_hz = marker_tolerance_hz
        self.calls = 0
        self.resets = 0
        self.closed = False
        self.detections: list[int] = []
        self._armed = False

    def trigger(self) -> None:
        """Make the next :meth:`process` call return True."""
        self._armed = True

    def _is_marker(self, frame: np.ndarray) -> bool:
        if self.marker_hz is None or rms(frame) <= self.rms_threshold:
            return False
        spectrum = np.abs(np.fft.rfft(frame.astype(np.float64)))
        spectrum[0] = 0.0
        peak_hz = float(np.argmax(spectrum)) * SAMPLE_RATE / len(frame)
        return abs(peak_hz - self.marker_hz) <= self.marker_tolerance_hz

    def process(self, frame: np.ndarray) -> bool:
        _check_length("FakeWakeWord", frame, self.frame_length)
        index = self.calls
        self.calls += 1
        fired = index in self.fire_at or self._armed or self._is_marker(frame)
        self._armed = False
        if fired:
            self.detections.append(index)
        return fired

    def reset(self) -> None:
        self.resets += 1
        self._armed = False

    def close(self) -> None:
        self.closed = True


class FakeVAD(VAD):
    """Speech means frame RMS > ``threshold``, or follows an explicit per-frame ``mask``."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        threshold: float = 500.0,
        frame_length: int = 512,
        mask: Sequence[bool] | None = None,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        self.threshold = threshold
        self.frame_length = frame_length
        self.mask = list(mask) if mask is not None else None
        self.calls = 0
        self.speech_frames = 0
        self.resets = 0
        self.decisions: list[bool] = []

    def is_speech(self, frame: np.ndarray) -> bool:
        _check_length("FakeVAD", frame, self.frame_length)
        index = self.calls
        self.calls += 1
        if self.mask is not None:
            speech = bool(self.mask[index]) if index < len(self.mask) else False
        else:
            speech = rms(frame) > self.threshold
        self.decisions.append(speech)
        self.speech_frames += int(speech)
        return speech

    def reset(self) -> None:
        self.resets += 1


# --- STT -------------------------------------------------------------------------------------


class FakeSTT(STT):
    """Returns scripted transcripts.

    Args:
        transcripts: a list consumed in order (then ``default``), a single string returned
            every time, or a callable ``pcm -> text``.
        delay_s: simulated transcription time (virtual when ``clock`` is given).
        fail_with: exception (instance or class) raised by :meth:`transcribe`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transcripts: Sequence[str] | str | Callable[[np.ndarray], str] | None = None,
        default: str = "",
        delay_s: float = 0.0,
        clock: Clock | None = None,
        fail_with: BaseException | type[BaseException] | None = None,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        self._fn: Callable[[np.ndarray], str] | None = None
        self._fixed: str | None = None
        self._queue: list[str] = []
        if callable(transcripts):
            self._fn = transcripts
        elif isinstance(transcripts, str):
            self._fixed = transcripts
        elif transcripts is not None:
            self._queue = list(transcripts)
        self.default = default
        self.delay_s = delay_s
        self.clock = clock
        self.fail_with = fail_with
        self.received: list[int] = []
        self.results: list[str] = []
        self.warmups = 0
        self.closed = False

    async def warmup(self) -> None:
        self.warmups += 1

    async def transcribe(self, pcm: np.ndarray) -> str:
        self.received.append(len(pcm))
        if self.delay_s > 0:
            if self.clock is not None:
                await self.clock.sleep(self.delay_s)
            else:
                await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            exc = self.fail_with
            raise exc() if isinstance(exc, type) else exc
        if self._fn is not None:
            text = self._fn(pcm)
        elif self._fixed is not None:
            text = self._fixed
        elif self._queue:
            text = self._queue.pop(0)
        else:
            text = self.default
        self.results.append(text)
        return text

    async def close(self) -> None:
        self.closed = True


# --- TTS and player --------------------------------------------------------------------------


class FakeTTS(TTS):
    """Records what would be spoken, with timestamps.

    Each non-blank chunk is appended to ``spoken`` and ``chunk_times`` (``(text, t)``, with
    ``t`` from ``clock`` or ``time.perf_counter``). If ``player`` is given each chunk is
    "played" through it as ``per_chunk_s`` of silence; otherwise ``per_chunk_s`` is an
    interruptible wait. :meth:`stop` makes :meth:`speak` return promptly even while it is
    waiting for the next chunk from the brain.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        per_chunk_s: float = 0.0,
        player: AudioOutput | None = None,
        clock: Clock | None = None,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        self.per_chunk_s = per_chunk_s
        self.player = player
        self.clock = clock
        self.spoken: list[str] = []
        self.chunk_times: list[tuple[str, float]] = []
        self.speak_calls = 0
        self.speak_started_at: float | None = None
        self.first_chunk_at: float | None = None
        self.stop_requested_at: float | None = None
        self.returned_at: float | None = None
        self.speaking = False
        self.stopped = False
        self.stops = 0
        self.warmups = 0
        self.closed = False
        self.rendered: list[str] = []
        self.render_s = 0.5
        self._stop = asyncio.Event()

    def _now(self) -> float:
        return self.clock.monotonic() if self.clock is not None else time.perf_counter()

    @property
    def stop_latency(self) -> float | None:
        """Seconds from :meth:`stop` to :meth:`speak` returning (wall or virtual time)."""
        if self.stop_requested_at is None or self.returned_at is None:
            return None
        return self.returned_at - self.stop_requested_at

    async def warmup(self) -> None:
        self.warmups += 1

    def render(self, text: str) -> tuple[np.ndarray, int]:
        """A clip for ``text`` (the wake reply): ``render_s`` of silence."""
        self.rendered.append(text)
        return silence(self.render_s), SAMPLE_RATE

    async def _play_chunk(self) -> bool:
        """Simulate playback of one chunk; True if stopped meanwhile."""
        if self.player is not None:
            await self.player.play(silence(self.per_chunk_s), SAMPLE_RATE)
            return self._stop.is_set()
        return await _interruptible_sleep(self.per_chunk_s, self._stop, self.clock)

    async def speak(self, text_chunks: AsyncIterator[str]) -> None:
        self._stop = asyncio.Event()
        self.stopped = False
        self.speaking = True
        self.speak_calls += 1
        self.speak_started_at = self._now()
        self.first_chunk_at = None
        iterator = aiter(text_chunks)
        stop_wait = asyncio.ensure_future(self._stop.wait())

        async def next_chunk() -> str:
            return await anext(iterator)

        try:
            while not self._stop.is_set():
                pending = asyncio.ensure_future(next_chunk())
                done, _ = await asyncio.wait(
                    {pending, stop_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if pending not in done:
                    pending.cancel()
                    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                        await pending
                    return
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    return
                if not chunk.strip():
                    continue
                now = self._now()
                if self.first_chunk_at is None:
                    self.first_chunk_at = now
                self.spoken.append(chunk)
                self.chunk_times.append((chunk, now))
                if await self._play_chunk():
                    return
        finally:
            stop_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_wait
            self.speaking = False
            self.returned_at = self._now()

    async def stop(self) -> None:
        self.stops += 1
        self.stopped = True
        self.stop_requested_at = self._now()
        self._stop.set()
        if self.player is not None:
            self.player.stop()

    async def close(self) -> None:
        await self.stop()
        self.closed = True


class FakePlayer(AudioOutput):
    """Records playback and earcons.

    ``play`` takes ``duration * speed`` seconds (``speed=0`` returns immediately, 1 is real
    time; virtual when ``clock`` is given) and returns early on :meth:`stop`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        speed: float = 0.0,
        clock: Clock | None = None,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        self.speed = speed
        self.clock = clock
        self.played: list[tuple[int, int]] = []
        self.earcons: list[str] = []
        self.clips: list[tuple[int, int]] = []  # play_clip(): (samples, rate)
        self.stops = 0
        self.interrupted_plays = 0
        self.stop_requested_at: float | None = None
        self.returned_at: float | None = None
        self.closed = False
        self._playing = False
        self._stop = asyncio.Event()

    def _now(self) -> float:
        return self.clock.monotonic() if self.clock is not None else time.perf_counter()

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def stop_latency(self) -> float | None:
        """Seconds from :meth:`stop` to the interrupted :meth:`play` returning."""
        if self.stop_requested_at is None or self.returned_at is None:
            return None
        return self.returned_at - self.stop_requested_at

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        self._stop = asyncio.Event()
        self.played.append((len(pcm), sample_rate))
        self._playing = True
        try:
            seconds = (len(pcm) / sample_rate) * self.speed if sample_rate else 0.0
            if await _interruptible_sleep(seconds, self._stop, self.clock):
                self.interrupted_plays += 1
        finally:
            self._playing = False
            self.returned_at = self._now()

    def play_earcon(self, name: str) -> None:
        self.earcons.append(name)

    def play_clip(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.clips.append((len(pcm), sample_rate))

    def stop(self) -> None:
        self.stops += 1
        self.stop_requested_at = self._now()
        self._stop.set()

    async def close(self) -> None:
        self.stop()
        self.closed = True


# --- confirmation ----------------------------------------------------------------------------


class FakeConfirmer(Confirmer):
    """Scripted answers for :class:`spark.brain.confirm.ConfirmationGate`.

    ``answers`` are consumed in order: True = yes, False = no, None = no answer. Once they run
    out, ``default`` is returned, which is None (no answer, so the action is denied). With
    ``delay_s`` the answer takes that long (to exercise the gate's timeout).
    """

    def __init__(
        self,
        answers: Iterable[bool | None] = (),
        default: bool | None = None,
        *,
        delay_s: float = 0.0,
        fail_with: BaseException | None = None,
    ) -> None:
        self._answers = list(answers)
        self.default = default
        self.delay_s = delay_s
        self.fail_with = fail_with
        self.asked: list[str] = []

    async def ask(self, summary: str) -> bool | None:
        self.asked.append(summary)
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with
        return self._answers.pop(0) if self._answers else self.default


class ToolCallingBrain(Brain):
    """A brain that makes one tool call mid-reply, through a real ``PreToolUse`` gate.

    It streams ``preamble``, then awaits ``gate(hook_input, tool_use_id, context)`` the way the
    SDK does (``_internal/query.py:565``). If the decision is allow, it "runs" the tool (records
    it in ``executed``) and streams ``done_reply``. Otherwise it streams ``denied_reply``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        gate: Callable[..., Any] | None = None,
        tool_name: str = "mcp__notes__create_note",
        tool_input: dict[str, Any] | None = None,
        preamble: str = "Sure, I can do that.\n",
        done_reply: str = "Done, the note is saved.",
        denied_reply: str = "Okay, I won't do that.",
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        self.gate = gate
        self.tool_name = tool_name
        self.tool_input = tool_input if tool_input is not None else {"title": "groceries"}
        self.preamble = preamble
        self.done_reply = done_reply
        self.denied_reply = denied_reply
        self.decisions: list[dict[str, Any]] = []
        self.executed: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.interrupts = 0

    async def respond(self, text: str) -> AsyncIterator[str]:
        self.calls.append(text)
        if self.preamble:
            yield self.preamble
        assert self.gate is not None, "ToolCallingBrain needs gate="
        hook_input = {
            "session_id": "s",
            "transcript_path": "t",
            "cwd": ".",
            "hook_event_name": "PreToolUse",
            "tool_name": self.tool_name,
            "tool_input": dict(self.tool_input),
            "tool_use_id": f"toolu_{len(self.calls)}",
        }
        decision = await self.gate(hook_input, hook_input["tool_use_id"], {"signal": None})
        self.decisions.append(decision)
        verdict = (decision.get("hookSpecificOutput") or {}).get("permissionDecision")
        if verdict == "allow":
            self.executed.append(dict(self.tool_input))
            yield self.done_reply
        else:
            yield self.denied_reply

    async def interrupt(self) -> None:
        self.interrupts += 1
