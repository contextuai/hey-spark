"""Cloud TTS with ElevenLabs (optional extra).

Requests raw 16-bit PCM (``pcm_22050``), so no MP3 decoder is needed. Verified against
``elevenlabs`` 2.69.0: ``ElevenLabs(api_key=, timeout=)`` and
``client.text_to_speech.convert(voice_id, text=, model_id=, output_format=) -> Iterator[bytes]``.
The key comes from the env var named in ``tts.options.elevenlabs.api_key_env`` and is never
logged.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.interfaces import BackendUnavailable
from spark.tts.base import SentenceTTS

if TYPE_CHECKING:
    from spark.config import Settings

__all__ = ["ElevenLabsError", "ElevenLabsTTS"]

OUTPUT_FORMAT = "pcm_22050"
OUTPUT_RATE = 22_050
REQUEST_TIMEOUT_S = 20.0


class ElevenLabsError(RuntimeError):
    """An ElevenLabs request failed. The message never contains the API key."""


class ElevenLabsTTS(SentenceTTS):
    backend = "tts/elevenlabs"
    warmup_text = ""  # don't spend characters (money) on a warm-up request

    def __init__(self, settings: Settings, *, client: Any = None, **deps: Any) -> None:
        super().__init__(settings, **deps)
        self.cfg = settings.tts.options.elevenlabs
        if client is not None:  # injected (tests)
            self._client = client
            return
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise BackendUnavailable(
                self.backend,
                f"elevenlabs is not installed ({exc.__class__.__name__})",
                "uv sync --extra elevenlabs",
            ) from exc
        key = os.environ.get(self.cfg.api_key_env, "").strip()
        if not key:
            raise BackendUnavailable(
                self.backend,
                f"{self.cfg.api_key_env} is not set",
                f"add {self.cfg.api_key_env}=<your key> to .env (https://elevenlabs.io/app/settings/api-keys)",
            )
        self._client = ElevenLabs(api_key=key, timeout=REQUEST_TIMEOUT_S)

    def _warm_blocking(self) -> None:
        """Nothing to load locally; a warm-up request would cost characters."""

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        try:
            stream = self._client.text_to_speech.convert(
                self.cfg.voice_id,
                text=text,
                model_id=self.cfg.model,
                output_format=OUTPUT_FORMAT,
            )
            raw = b"".join(stream)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = f" (HTTP {status})" if status is not None else ""
            raise ElevenLabsError(
                f"elevenlabs request failed: {exc.__class__.__name__}{detail}"
            ) from None
        if len(raw) % 2:
            raw = raw[:-1]
        return np.frombuffer(raw, dtype="<i2").astype(np.int16), OUTPUT_RATE
