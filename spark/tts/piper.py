"""Local TTS with Piper (piper-tts 1.8.0, ONNX on the CPU). The default backend (spec §4).

The voice is ``models_path/piper/<voice>.onnx`` plus ``<voice>.onnx.json``, fetched by
``uv run python -m spark models download``. ``piper`` is imported in the constructor; the voice
is loaded (about 3 to 5 s) by :meth:`warmup` or on first use, in a worker thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.interfaces import BackendUnavailable
from spark.models import DOWNLOAD_FIX, piper_voice_paths
from spark.tts.base import SentenceTTS

if TYPE_CHECKING:
    from spark.config import Settings

__all__ = ["PiperTTS"]


class PiperTTS(SentenceTTS):
    backend = "tts/piper"

    def __init__(self, settings: Settings, **deps: Any) -> None:
        super().__init__(settings, **deps)
        self.cfg = settings.tts.options.piper
        try:
            import piper
        except ImportError as exc:
            raise BackendUnavailable(
                self.backend, f"piper-tts is not installed ({exc})", "uv sync"
            ) from exc
        self._piper = piper
        self.model_path, self.config_path = piper_voice_paths(settings)
        missing = [p for p in (self.model_path, self.config_path) if not p.is_file()]
        if missing:
            raise BackendUnavailable(
                self.backend,
                f"Piper voice {self.cfg.voice!r} not found ({', '.join(str(p) for p in missing)})",
                DOWNLOAD_FIX,
            )
        self._voice: Any = None
        self._load_lock = threading.Lock()
        self._syn_config = piper.SynthesisConfig(
            speaker_id=self.cfg.speaker_id, length_scale=self.cfg.length_scale
        )

    @property
    def sample_rate(self) -> int:
        return int(self.load().config.sample_rate)

    def load(self) -> Any:
        with self._load_lock:
            if self._voice is None:
                self._voice = self._piper.PiperVoice.load(
                    str(self.model_path), config_path=str(self.config_path)
                )
            return self._voice

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        voice = self.load()
        rate = int(voice.config.sample_rate)
        parts = [chunk.audio_int16_array for chunk in voice.synthesize(text, self._syn_config)]
        if not parts:
            return np.zeros(0, dtype=np.int16), rate
        return np.concatenate(parts).astype(np.int16, copy=False), rate
