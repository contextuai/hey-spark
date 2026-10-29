"""Streaming Silero VAD using the ONNX model bundled with faster-whisper (no torch).

faster-whisper's ``SileroVADModel.__call__`` resets the LSTM state on every call, so for
streaming we drive the ONNX session directly and carry ``h``/``c`` plus the 64-sample
context between 512-sample frames (verified in docs/sdk_notes.md §2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.interfaces import VAD, BackendUnavailable

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

BACKEND = "vad/silero"
MODEL_FILE = "silero_vad_v6.onnx"
FRAME = 512
CONTEXT = 64
STATE_SHAPE = (1, 1, 128)
_INSTALL_FIX = "uv sync  (faster-whisper ships the Silero ONNX model and onnxruntime)"


def default_model_path() -> Path:
    """Path of the Silero ONNX file inside the installed faster-whisper package."""
    try:
        from faster_whisper.utils import get_assets_path
    except ImportError as exc:
        raise BackendUnavailable(
            BACKEND, f"faster-whisper is not installed ({exc})", _INSTALL_FIX
        ) from exc
    return Path(get_assets_path()) / MODEL_FILE


class SileroVAD(VAD):
    """Stateful Silero VAD on 512-sample 16 kHz int16 frames."""

    frame_length = FRAME

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_path: str | Path | None = None,
        threshold: float | None = None,
        **deps: Any,
    ) -> None:
        if threshold is None:
            threshold = settings.vad.options.silero.threshold if settings is not None else 0.5
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"silero threshold must be in (0, 1), got {threshold}")
        self.threshold = float(threshold)
        self.deps = deps

        path = Path(model_path) if model_path is not None else default_model_path()
        if not path.is_file():
            raise BackendUnavailable(BACKEND, f"model file not found: {path}", _INSTALL_FIX)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailable(
                BACKEND, f"onnxruntime is not installed ({exc})", _INSTALL_FIX
            ) from exc

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # corrupt file, incompatible opset, ...
            raise BackendUnavailable(
                BACKEND, f"could not load {path.name}: {exc}", _INSTALL_FIX
            ) from exc
        self.model_path = path
        self.last_probability = 0.0
        self.reset()
        log.debug("Silero VAD loaded from %s (threshold %.2f)", path, self.threshold)

    def reset(self) -> None:
        """Clear LSTM state and context; call at the start of every utterance."""
        self._h = np.zeros(STATE_SHAPE, np.float32)
        self._c = np.zeros(STATE_SHAPE, np.float32)
        self._ctx = np.zeros((1, CONTEXT), np.float32)
        self._input = np.zeros((1, CONTEXT + FRAME), np.float32)
        self.last_probability = 0.0

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for exactly 512 samples (int16, or float in [-1, 1])."""
        x = np.asarray(frame)
        if x.ndim != 1 or x.size != FRAME:
            raise ValueError(f"SileroVAD needs 1-D frames of {FRAME} samples, got shape {x.shape}")
        inp = self._input
        inp[0, :CONTEXT] = self._ctx[0]
        if x.dtype == np.int16:
            np.multiply(x, 1.0 / 32768.0, out=inp[0, CONTEXT:], casting="unsafe")
        else:
            inp[0, CONTEXT:] = x
        out, self._h, self._c = self._session.run(None, {"input": inp, "h": self._h, "c": self._c})
        self._ctx[0] = inp[0, -CONTEXT:]
        self.last_probability = float(np.asarray(out).reshape(-1)[0])
        return self.last_probability

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.probability(frame) >= self.threshold
