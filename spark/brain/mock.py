"""MockBrain: a deterministic, offline :class:`~spark.interfaces.Brain` for tests and demos.

Selected with ``brain: {backend: mock}``. It never touches the network and imports nothing
heavy. Replies are streamed as several small deltas so sentence splitting and early TTS
start are exercised exactly as with the real brain.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Iterable
from typing import TYPE_CHECKING, Any

from spark.interfaces import Brain

if TYPE_CHECKING:
    from spark.config import Settings
    from spark.util.clock import Clock

__all__ = ["MockBrain", "default_reply"]

Replies = Iterable[str] | Callable[[str], str] | None

_CANNED: dict[str, str] = {
    "what is two plus two": "Two plus two is four.",
    "whats two plus two": "Two plus two is four.",
    "what is 2 plus 2": "Two plus two is four.",
    "what is 2 + 2": "Two plus two is four.",
    "whats 2 + 2": "Two plus two is four.",
    "hello": "Hi, I'm Spark. How can I help?",
    "hi": "Hi, I'm Spark. How can I help?",
    "hey spark": "Hi, I'm Spark. How can I help?",
    "who are you": "I'm Spark, your voice assistant. I'm running in offline mock mode.",
}


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^\w\s+]", " ", text)
    return " ".join(text.split())


def default_reply(text: str) -> str:
    """The canned reply for ``text``: a few known questions, otherwise an echo."""
    stripped = text.strip()
    if not stripped:
        return "Sorry, I didn't catch that."
    canned = _CANNED.get(_normalise(stripped))
    if canned is not None:
        return canned
    return f"You said: {stripped.rstrip('.!?')}."


def _split_deltas(reply: str, chunk_size: int | None) -> list[str]:
    """Cut ``reply`` into small deltas whose concatenation is exactly ``reply``."""
    if not reply:
        return []
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        return [reply[i : i + chunk_size] for i in range(0, len(reply), chunk_size)]
    # One word per delta, keeping the whitespace that follows it.
    return re.findall(r"\S+\s*|\s+", reply)


class MockBrain(Brain):
    """Scripted brain.

    Args:
        settings: accepted for registry compatibility (``MockBrain(settings, **deps)``).
        replies: a sequence of replies consumed in order (falls back to :func:`default_reply`
            once exhausted), or a callable ``text -> reply``. ``None`` uses the canned replies.
        delay_s: wait this long before each delta (interruptible).
        chunk_size: characters per delta; ``None`` streams one word per delta.
        fail_with: an exception (instance or class) raised by :meth:`respond`.
        fail_after: how many deltas to yield before raising ``fail_with`` (0 = before any).
        clock: optional :class:`~spark.util.clock.Clock` used for ``delay_s``.
        **deps: any extra registry dependencies are kept in ``self.deps``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        replies: Replies = None,
        delay_s: float = 0.0,
        chunk_size: int | None = None,
        fail_with: BaseException | type[BaseException] | None = None,
        fail_after: int = 0,
        clock: Clock | None = None,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.deps = deps
        if replies is None or callable(replies):
            self._reply_fn: Callable[[str], str] | None = replies
            self._queue: list[str] = []
        else:
            self._reply_fn = None
            self._queue = list(replies)
        self.delay_s = delay_s
        self.chunk_size = chunk_size
        self.fail_with = fail_with
        self.fail_after = fail_after
        self.clock = clock

        self.calls: list[str] = []
        self.replies_given: list[str] = []
        self.deltas_yielded = 0
        self.resets = 0
        self.interrupts = 0
        self.warmups = 0
        self.closed = False
        self.in_flight = False
        self._interrupted = asyncio.Event()

    def add_replies(self, *replies: str) -> None:
        """Queue more scripted replies (only valid when not using a callable)."""
        if self._reply_fn is not None:
            raise TypeError("MockBrain was built with a callable; it has no reply queue")
        self._queue.extend(replies)

    def _next_reply(self, text: str) -> str:
        if self._reply_fn is not None:
            return self._reply_fn(text)
        if self._queue:
            return self._queue.pop(0)
        return default_reply(text)

    def _raise(self) -> None:
        exc = self.fail_with
        if exc is None:
            return
        raise exc() if isinstance(exc, type) else exc

    async def _pause(self) -> bool:
        """Wait ``delay_s``; return True if interrupted meanwhile."""
        if self._interrupted.is_set():
            return True
        if self.delay_s <= 0:
            await asyncio.sleep(0)
            return self._interrupted.is_set()
        if self.clock is not None:
            await self.clock.sleep(self.delay_s)
            return self._interrupted.is_set()
        try:
            await asyncio.wait_for(self._interrupted.wait(), timeout=self.delay_s)
        except TimeoutError:
            return False
        return True

    async def warmup(self) -> None:
        self.warmups += 1

    async def respond(self, text: str) -> AsyncIterator[str]:
        self.calls.append(text)
        self._interrupted = asyncio.Event()
        self.in_flight = True
        try:
            if self.fail_with is not None and self.fail_after <= 0:
                self._raise()
            reply = self._next_reply(text)
            self.replies_given.append(reply)
            for i, delta in enumerate(_split_deltas(reply, self.chunk_size)):
                if self.fail_with is not None and i >= self.fail_after:
                    self._raise()
                if await self._pause():
                    return
                self.deltas_yielded += 1
                yield delta
            # A reply shorter than ``fail_after`` still fails, at the end of the stream.
            self._raise()
        finally:
            self.in_flight = False

    async def reset(self) -> None:
        self.resets += 1

    async def interrupt(self) -> None:
        self.interrupts += 1
        self._interrupted.set()

    async def close(self) -> None:
        self._interrupted.set()
        self.closed = True
