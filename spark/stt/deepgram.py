"""Cloud speech-to-text with Deepgram's pre-recorded (REST) API.

Spark endpoints the utterance locally, so the finished utterance is sent once as a 16 kHz mono
16-bit WAV. Audio leaves the machine only after the wake word and only when this backend is
selected (spec §12). Verified against ``deepgram-sdk`` 7.10.0:
``AsyncDeepgramClient(api_key=, timeout=)`` and
``client.listen.v1.media.transcribe_file(request=<bytes>, model=, language=, smart_format=,
punctuate=, request_options={"timeout": ..., "max_retries": ...})`` returning
``ListenV1Response.results.channels[0].alternatives[0].transcript``.

The API key is read from the env var named in ``stt.options.deepgram.api_key_env`` and is never
logged or included in error messages.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.audio.stream import to_int16, to_mono
from spark.interfaces import SAMPLE_RATE, STT, BackendUnavailable

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["DeepgramError", "DeepgramSTT", "wav_bytes"]

BACKEND = "stt/deepgram"


class DeepgramError(RuntimeError):
    """A Deepgram request failed. The message never contains the API key."""


def wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono audio as an in-memory 16-bit PCM WAV."""
    data = to_int16(to_mono(pcm))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data.astype("<i2").tobytes())
    return buf.getvalue()


def _transcript(response: Any) -> str:
    results = getattr(response, "results", None)
    channels = getattr(results, "channels", None) or []
    if not channels:
        return ""
    alternatives = getattr(channels[0], "alternatives", None) or []
    if not alternatives:
        return ""
    return (getattr(alternatives[0], "transcript", None) or "").strip()


class DeepgramSTT(STT):
    """Pre-recorded transcription of one endpointed utterance."""

    def __init__(self, settings: Settings, *, client: Any = None, **deps: Any) -> None:
        self.settings = settings
        self.cfg = settings.stt.options.deepgram
        if client is not None:  # injected (tests)
            self._client = client
            return
        try:
            from deepgram import AsyncDeepgramClient
        except ImportError as exc:
            raise BackendUnavailable(
                BACKEND,
                f"deepgram-sdk is not installed ({exc.__class__.__name__})",
                "uv sync --extra deepgram",
            ) from exc
        key = os.environ.get(self.cfg.api_key_env, "").strip()
        if not key:
            raise BackendUnavailable(
                BACKEND,
                f"{self.cfg.api_key_env} is not set",
                f"add {self.cfg.api_key_env}=<your key> to .env (https://console.deepgram.com)",
            )
        self._client = AsyncDeepgramClient(api_key=key, timeout=self.cfg.timeout_s)

    async def transcribe(self, pcm: np.ndarray) -> str:
        if np.asarray(pcm).size == 0:
            return ""
        body = wav_bytes(pcm)
        timeout = float(self.cfg.timeout_s)
        try:
            response = await asyncio.wait_for(
                self._client.listen.v1.media.transcribe_file(
                    request=body,
                    model=self.cfg.model,
                    language=self.cfg.language,
                    smart_format=True,
                    punctuate=True,
                    request_options={"timeout": timeout, "max_retries": 1},
                ),
                timeout=timeout + 1.0,
            )
        except TimeoutError:
            raise DeepgramError(f"deepgram request timed out after {timeout:.1f} s") from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status is not None else ""
            raise DeepgramError(
                f"deepgram request failed: {exc.__class__.__name__}{detail}"
            ) from None
        return _transcript(response)
