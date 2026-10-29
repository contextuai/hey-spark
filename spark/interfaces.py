"""Abstract interfaces for every swappable Spark component (spec §4).

Audio inside Spark is always 16 kHz mono int16 numpy arrays. Backends are chosen through
``config.yaml`` and built by :mod:`spark.registry`; app code depends only on these ABCs.
Nothing heavy is imported here so ``--text`` mode and tests import cleanly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import numpy as np

SAMPLE_RATE = 16_000

__all__ = [
    "SAMPLE_RATE",
    "STT",
    "TTS",
    "VAD",
    "AudioInput",
    "AudioOutput",
    "BackendUnavailable",
    "Brain",
    "WakeWordDetector",
]


class BackendUnavailable(RuntimeError):
    """A backend can't be built here (missing package, model file, or key).

    ``fix`` is a copy-pasteable command or instruction that resolves it.
    """

    def __init__(self, backend: str, reason: str, fix: str) -> None:
        self.backend = backend
        self.reason = reason
        self.fix = fix
        super().__init__(f"{backend} unavailable: {reason}. Fix: {fix}")


class WakeWordDetector(ABC):
    """Consumes fixed-size frames and reports whether the wake word fired."""

    frame_length: int = 512  # samples at 16 kHz this detector consumes per call

    @abstractmethod
    def process(self, frame: np.ndarray) -> bool:
        """Feed exactly ``frame_length`` int16 samples. True when the wake word is detected."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear any internal state (called after a detection or a state change)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release native resources."""


class VAD(ABC):
    """Voice activity detection on fixed-size frames."""

    frame_length: int = 512

    @abstractmethod
    def is_speech(self, frame: np.ndarray) -> bool:
        """Feed exactly ``frame_length`` int16 samples. True if the frame contains speech."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear recurrent state between utterances."""


class STT(ABC):
    """Speech-to-text on a complete utterance."""

    async def warmup(self) -> None:  # noqa: B027 - optional hook
        """Load models ahead of the first turn."""

    @abstractmethod
    async def transcribe(self, pcm: np.ndarray) -> str:
        """Transcribe 16 kHz mono int16 audio. Returns '' when nothing was said."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


class TTS(ABC):
    """Text-to-speech that speaks a stream of sentences through an :class:`AudioOutput`."""

    async def warmup(self) -> None:  # noqa: B027 - optional hook
        """Load voices ahead of the first turn."""

    @abstractmethod
    async def speak(self, text_chunks: AsyncIterator[str]) -> None:
        """Speak each chunk (normally one sentence) as soon as it arrives.

        Returns when everything has been played or :meth:`stop` was called.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop speaking immediately and drop anything queued."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


class Brain(ABC):
    """The conversational brain (Claude Agent SDK by default)."""

    async def warmup(self) -> None:  # noqa: B027 - optional hook
        """Connect ahead of the first turn."""

    @abstractmethod
    def respond(self, text: str) -> AsyncIterator[str]:
        """Stream the reply to ``text`` as text deltas (not necessarily whole sentences)."""

    async def reset(self) -> None:  # noqa: B027 - optional hook
        """Start a fresh conversation (idle timeout or "new topic")."""

    async def interrupt(self) -> None:  # noqa: B027 - optional hook
        """Abort the in-flight response (barge-in)."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Disconnect."""


class AudioInput(ABC):
    """Microphone source producing 16 kHz mono int16 blocks of arbitrary size."""

    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    async def start(self) -> None:
        """Open the device and begin capturing."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop capturing and close the device."""

    @abstractmethod
    def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield captured blocks until stopped. Consumers must re-chunk (see audio.stream)."""


class AudioOutput(ABC):
    """Interruptible speaker output."""

    async def warmup(self) -> None:  # noqa: B027 - optional hook
        """Open the device ahead of time so the first earcon isn't late."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """True while audio is being played."""

    @abstractmethod
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Play int16 mono audio and return when done, or early if :meth:`stop` is called."""

    @abstractmethod
    def play_earcon(self, name: str) -> None:
        """Start a short named cue ("wake", "thinking", "error") without blocking."""

    @abstractmethod
    def stop(self) -> None:
        """Stop playback immediately. Safe to call from any task; never blocks."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release the device."""
