"""Local TTS with Kokoro via ONNX (optional extra; better voice than Piper, slower on CPU).

Files: ``models_path/kokoro/<model_file>`` and ``<voices_file>`` (defaults
``kokoro-v1.0.onnx`` / ``voices-v1.0.bin``), fetched by
``uv run python -m spark.models download --kokoro``. Verified against ``kokoro-onnx`` 0.6.1:
``Kokoro(model_path, voices_path)`` and ``create(text, voice, speed, lang) ->
(float32 samples, sample_rate)``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.audio.stream import to_int16
from spark.interfaces import BackendUnavailable
from spark.models import kokoro_paths
from spark.tts.base import SentenceTTS

if TYPE_CHECKING:
    from spark.config import Settings

__all__ = ["KokoroTTS", "kokoro_lang"]

KOKORO_DOWNLOAD_FIX = "uv run python -m spark models download --kokoro"


def kokoro_lang(cfg: Any) -> str:
    """``cfg.lang``, or the voice's own accent: ``bf_``/``bm_`` voices are British (en-gb), the
    rest American (en-us)."""
    return cfg.lang or ("en-gb" if str(cfg.voice).startswith("b") else "en-us")


class KokoroTTS(SentenceTTS):
    backend = "tts/kokoro"

    def __init__(self, settings: Settings, **deps: Any) -> None:
        super().__init__(settings, **deps)
        self.cfg = settings.tts.options.kokoro
        try:
            import kokoro_onnx
        except ImportError as exc:
            raise BackendUnavailable(
                self.backend,
                f"kokoro-onnx is not installed ({exc.__class__.__name__})",
                "uv sync --extra kokoro",
            ) from exc
        self._kokoro_mod = kokoro_onnx
        self.model_path, self.voices_path = kokoro_paths(settings)
        missing = [p for p in (self.model_path, self.voices_path) if not p.is_file()]
        if missing:
            raise BackendUnavailable(
                self.backend,
                f"Kokoro files not found ({', '.join(str(p) for p in missing)})",
                KOKORO_DOWNLOAD_FIX,
            )
        self._engine: Any = None
        self._load_lock = threading.Lock()

    def load(self) -> Any:
        with self._load_lock:
            if self._engine is None:
                self._engine = self._kokoro_mod.Kokoro(str(self.model_path), str(self.voices_path))
            return self._engine

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        engine = self.load()
        samples, rate = engine.create(
            text, voice=self.cfg.voice, speed=self.cfg.speed, lang=kokoro_lang(self.cfg)
        )
        return to_int16(np.asarray(samples, dtype=np.float32)), int(rate)
