"""Local speech-to-text with faster-whisper (CTranslate2) on the CPU (spec §4, §9).

The model is read from ``models_path/whisper/<model>`` (see :mod:`spark.models`); it is never
downloaded at runtime. ``faster_whisper`` is imported in the constructor only.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.audio.stream import to_float32, to_mono
from spark.interfaces import SAMPLE_RATE, STT, BackendUnavailable
from spark.models import DOWNLOAD_FIX, whisper_model_dir, whisper_present

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["FasterWhisperSTT", "pcm_to_float32"]

BACKEND = "stt/faster_whisper"


def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    """1-D float32 in [-1, 1] (what WhisperModel.transcribe expects) from int16 or float audio."""
    return np.ascontiguousarray(to_float32(to_mono(pcm)), dtype=np.float32).ravel()


class FasterWhisperSTT(STT):
    """``base.en`` / int8 / CPU by default. All inference runs in ``asyncio.to_thread``."""

    def __init__(self, settings: Settings, **deps: Any) -> None:
        self.settings = settings
        self.cfg = settings.stt.options.faster_whisper
        try:
            import faster_whisper
        except ImportError as exc:
            raise BackendUnavailable(
                BACKEND, f"faster-whisper is not installed ({exc})", "uv sync"
            ) from exc
        self._fw = faster_whisper
        self.model_dir = whisper_model_dir(settings)
        if not whisper_present(self.model_dir):
            raise BackendUnavailable(
                BACKEND,
                f"whisper model {self.cfg.model!r} not found at {self.model_dir}",
                DOWNLOAD_FIX,
            )
        self._model: Any = None
        self._lock = threading.Lock()

    # --- model --------------------------------------------------------------------------------

    def _load(self) -> Any:
        """Build the WhisperModel once (blocking; call from a worker thread)."""
        with self._lock:
            if self._model is None:
                self._model = self._fw.WhisperModel(
                    str(self.model_dir),
                    device=self.cfg.device,
                    compute_type=self.cfg.compute_type,
                    cpu_threads=self.cfg.cpu_threads,
                    download_root=str(self.model_dir.parent),
                    local_files_only=True,
                )
            return self._model

    def _transcribe_blocking(self, audio: np.ndarray) -> str:
        model = self._load()
        segments, _info = model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        # ``segments`` is a lazy generator: decoding happens here, inside the worker thread.
        return "".join(seg.text for seg in segments).strip()

    def _warm_blocking(self) -> None:
        self._transcribe_blocking(np.zeros(SAMPLE_RATE, dtype=np.float32))

    # --- STT ------------------------------------------------------------------------------------

    async def warmup(self) -> None:
        """Load the model and run one 1 s silence transcription to hide first-call latency."""
        await asyncio.to_thread(self._warm_blocking)

    async def transcribe(self, pcm: np.ndarray) -> str:
        audio = pcm_to_float32(pcm)
        if audio.size == 0:
            return ""
        return await asyncio.to_thread(self._transcribe_blocking, audio)

    async def close(self) -> None:
        with self._lock:
            self._model = None
