"""``--no-tts``: a TTS backend that prints each sentence instead of speaking it."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, TextIO

from spark.interfaces import TTS

if TYPE_CHECKING:
    from spark.config import Settings


class PrintTTS(TTS):
    """Writes sentences to ``stream`` (stdout) as they arrive, one reply per line."""

    def __init__(
        self, settings: Settings | None = None, *, stream: TextIO | None = None, **deps: Any
    ) -> None:
        self.settings = settings
        self.stream = stream
        self.on_first_audio: Callable[[], Any] | None = None
        self._stop = asyncio.Event()

    async def speak(self, text_chunks: AsyncIterator[str]) -> None:
        self._stop.clear()
        out = self.stream or sys.stdout
        wrote = False
        async for chunk in text_chunks:
            if self._stop.is_set():
                break
            text = chunk.strip()
            if not text:
                continue
            if not wrote and self.on_first_audio is not None:
                self.on_first_audio()
            out.write((" " if wrote else "") + text)
            out.flush()
            wrote = True
        if wrote:
            out.write("\n")
            out.flush()

    async def stop(self) -> None:
        self._stop.set()
