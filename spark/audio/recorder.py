"""Utterance recorder with VAD endpointing (spec §3, §6, §9).

All timing is measured in **audio samples consumed**, not wall-clock time, so endpointing is
deterministic in tests and immune to event-loop stalls. The clock only stamps ``started_at``
for latency logging.

Lifecycle::

    rec.start(pre_roll=mic.recent(300))        # after the wake word
    async for block in mic.frames():
        rec.add(block)
        if rec.end_of_speech: ...               # transcribe rec.pcm()
        elif rec.timed_out: ...                 # rec.reason says why
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from spark.audio.stream import Rechunker, to_int16, to_mono
from spark.interfaces import SAMPLE_RATE
from spark.util.clock import Clock, SystemClock

if TYPE_CHECKING:
    from spark.config import Settings
    from spark.interfaces import VAD

__all__ = ["Recorder", "StopReason"]

StopReason = Literal["end_of_speech", "no_speech", "max_length"]


class Recorder:
    """Collects one utterance and decides when it has ended.

    * ``end_of_speech``: at least ``min_speech_ms`` of speech was heard, followed by
      ``end_silence_ms`` of consecutive non-speech.
    * ``timed_out``: no speech within the no-speech timeout (``no_speech_timeout_s``, or the
      value passed to :meth:`start`, e.g. the rest of the awake window), or ``max_utterance_s``
      reached. On ``max_length`` with speech heard, the audio is still worth transcribing.

    ``max_utterance_s`` counts from the first speech frame, not from :meth:`start`, so a long
    wait for speech (the 3-minute awake window) is not an utterance. While no speech has been
    heard, only the last ``pre_roll_ms`` of the waiting audio is kept, so STT never gets minutes
    of silence in front of a follow-up.

    ``ignore_s`` (passed to :meth:`start`): audio right after the start that is neither judged
    nor kept, while Spark itself is talking (the spoken wake reply). It still counts toward the
    no-speech timeout.

    Once either fires the recorder is finished and ignores further :meth:`add` calls.
    """

    def __init__(
        self,
        settings: Settings,
        vad: VAD,
        clock: Clock | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        cfg = settings.recorder
        self.vad = vad
        self.clock: Clock = clock or SystemClock()
        self.sample_rate = sample_rate
        self.end_silence_samples = int(cfg.end_silence_ms * sample_rate / 1000)
        self.min_speech_samples = int(cfg.min_speech_ms * sample_rate / 1000)
        self.max_samples = int(cfg.max_utterance_s * sample_rate)
        self.pre_roll_ms = cfg.pre_roll_ms
        self.default_no_speech_s = cfg.no_speech_timeout_s
        self._rechunker = Rechunker(vad.frame_length)
        self._reset(self.default_no_speech_s)
        self.active = False

    def _reset(self, no_speech_s: float) -> None:
        self._chunks: list[np.ndarray] = []
        self._pre_roll_samples = 0
        self.no_speech_samples = int(no_speech_s * self.sample_rate)
        self.samples_consumed = 0  # post-start samples the VAD has judged
        self.speech_samples = 0
        self.silence_run = 0  # consecutive non-speech samples since the last speech frame
        self.heard_speech = False
        self.reason: StopReason | None = None
        self.end_sample: int | None = None  # samples_consumed when the recorder finished
        self.started_at: float | None = None
        self.first_speech_sample: int | None = None  # samples_consumed at the first speech frame
        self.ignore_samples = 0  # samples after start() not judged or kept (Spark is talking)
        self._received = 0  # samples given to add() since start()

    def start(
        self,
        pre_roll: np.ndarray | None = None,
        no_speech_timeout_s: float | None = None,
        *,
        ignore_s: float = 0.0,
    ) -> None:
        """Begin a new utterance. ``pre_roll`` is prepended to :meth:`pcm` (not VAD-judged), and
        the first ``ignore_s`` seconds are recorded but never count as speech."""
        self._reset(
            self.default_no_speech_s if no_speech_timeout_s is None else no_speech_timeout_s
        )
        self.ignore_samples = max(0, int(ignore_s * self.sample_rate))
        self._rechunker = Rechunker(self.vad.frame_length)
        self.vad.reset()
        if pre_roll is not None and np.asarray(pre_roll).size:
            pr = to_int16(to_mono(pre_roll)).copy()
            self._chunks.append(pr)
            self._pre_roll_samples = pr.size
        self.started_at = self.clock.monotonic()
        self.active = True

    def add(self, block: np.ndarray) -> None:
        """Append mic audio. Re-chunks to ``vad.frame_length`` and updates endpointing."""
        if not self.active or self.done:
            return
        x = to_int16(to_mono(block))
        if x.size == 0:
            return
        self._received += x.size
        if self._received > self.ignore_samples:  # a block that ends after Spark stopped talking
            self._chunks.append(x.copy())
        for frame in self._rechunker.feed(x):
            self._judge(frame)
            if self.done:
                break
        if self.first_speech_sample is None and not self.done:
            self._trim_waiting_audio()

    def _trim_waiting_audio(self) -> None:
        """No speech yet: keep the explicit pre-roll plus the last ``pre_roll_ms`` of audio."""
        keep = int(self.pre_roll_ms * self.sample_rate / 1000) + self.vad.frame_length
        head = 1 if self._pre_roll_samples else 0  # chunk 0 is the pre-roll passed to start()
        waiting = self._chunks[head:]
        total = sum(c.size for c in waiting)
        if total <= 2 * keep:
            return
        tail = np.concatenate(waiting)[-keep:]
        self._chunks = [*self._chunks[:head], tail]

    def _judge(self, frame: np.ndarray) -> None:
        n = frame.size
        self.samples_consumed += n
        if self.samples_consumed <= self.ignore_samples:
            if self.samples_consumed >= self.no_speech_samples:
                self._finish("no_speech")
            return  # Spark's own voice: don't feed it to the VAD
        if self.vad.is_speech(frame):
            if self.first_speech_sample is None:
                self.first_speech_sample = self.samples_consumed - n
            self.speech_samples += n
            self.silence_run = 0
            if self.speech_samples >= self.min_speech_samples:
                self.heard_speech = True
        else:
            self.silence_run += n
        if self.heard_speech and self.silence_run >= self.end_silence_samples:
            self._finish("end_of_speech")
        elif not self.heard_speech and self.samples_consumed >= self.no_speech_samples:
            self._finish("no_speech")
        elif (
            self.first_speech_sample is not None
            and self.samples_consumed - self.first_speech_sample >= self.max_samples
        ):
            self._finish("max_length")

    def _finish(self, reason: StopReason) -> None:
        self.reason = reason
        self.end_sample = self.samples_consumed

    @property
    def done(self) -> bool:
        return self.reason is not None

    @property
    def end_of_speech(self) -> bool:
        return self.reason == "end_of_speech"

    @property
    def timed_out(self) -> bool:
        return self.reason in ("no_speech", "max_length")

    def pcm(self) -> np.ndarray:
        """The utterance so far, pre-roll included, as 16 kHz mono int16."""
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._chunks)

    @property
    def duration_s(self) -> float:
        return sum(c.size for c in self._chunks) / self.sample_rate

    @property
    def pre_roll_samples(self) -> int:
        return self._pre_roll_samples

    def save_wav(self, path: str | Path) -> Path:
        """Write the utterance to ``path`` (for ``--debug-audio``)."""
        from spark.audio.wavio import write_wav

        return write_wav(path, self.pcm(), self.sample_rate)
