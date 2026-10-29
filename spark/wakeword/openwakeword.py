"""openWakeWord backend (optional extra ``oww``), fully open source (spec §4).

Uses a custom ``.onnx`` when ``model_path`` is set, otherwise the pretrained ``model_name``
(``hey_jarvis`` dev placeholder). openwakeword 0.6.0 ships no model files, so pretrained and
feature models are downloaded into ``<models_dir>/openwakeword`` on first construction.
``inference_framework="onnx"`` is required on Windows (tflite-runtime is Linux-only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.interfaces import BackendUnavailable, WakeWordDetector

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

BACKEND = "wakeword/openwakeword"
FRAME = 1280  # 80 ms at 16 kHz
FEATURE_FILES = ("melspectrogram.onnx", "embedding_model.onnx")
INSTALL_FIX = "uv sync --extra oww"


def find_model(directory: Path, model_name: str) -> Path | None:
    """The newest ``<model_name>*.onnx`` in ``directory`` (e.g. hey_jarvis_v0.1.onnx)."""
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{model_name}*.onnx"))
    return matches[-1] if matches else None


class OpenWakeWord(WakeWordDetector):
    """openWakeWord on 1280-sample 16 kHz int16 frames, thresholded on the best score."""

    frame_length = FRAME

    def __init__(self, settings: Settings, **deps: Any) -> None:
        opts = settings.wakeword.options.openwakeword
        self.deps = deps
        self.threshold = float(opts.threshold)
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"openwakeword threshold must be in (0, 1), got {self.threshold}")
        self.model_dir = settings.models_path / "openwakeword"
        self.last_score = 0.0

        try:
            import openwakeword.utils as oww_utils
            from openwakeword.model import Model
        except ImportError as exc:
            raise BackendUnavailable(
                BACKEND, f"openwakeword is not installed ({exc})", INSTALL_FIX
            ) from exc

        if opts.model_path:
            wake_model = settings.path(opts.model_path)
            if not wake_model.is_file():
                raise BackendUnavailable(
                    BACKEND,
                    f"custom model not found: {wake_model}",
                    "Train a 'Hey Spark' openWakeWord model (see assets/wakewords/README.txt) and "
                    f"save the .onnx to {wake_model}, or set "
                    "wakeword.options.openwakeword.model_path: null to use the placeholder",
                )
            to_download: list[str] = []
        else:
            wake_model = find_model(self.model_dir, opts.model_name)
            to_download = [] if wake_model is not None else [opts.model_name]

        features = [self.model_dir / name for name in FEATURE_FILES]
        if to_download or not all(p.is_file() for p in features):
            self._download(oww_utils, to_download)
            if wake_model is None:
                wake_model = find_model(self.model_dir, opts.model_name)
        missing = [p for p in features if not p.is_file()]
        if wake_model is None or missing:
            what = [str(p) for p in missing] + ([] if wake_model else [opts.model_name])
            raise BackendUnavailable(
                BACKEND,
                f"model files missing after download: {', '.join(what)}",
                f"Check network access to github.com, or set model_name to one of "
                f"alexa, hey_jarvis, hey_mycroft, hey_rhasspy (models dir: {self.model_dir})",
            )

        try:
            self._model = Model(
                wakeword_models=[str(wake_model)],
                inference_framework="onnx",
                melspec_model_path=str(features[0]),
                embedding_model_path=str(features[1]),
            )
        except Exception as exc:
            raise BackendUnavailable(
                BACKEND, f"could not load {wake_model.name}: {exc}", INSTALL_FIX
            ) from exc
        self.model_file = wake_model
        log.info("openWakeWord ready: %s (threshold %.2f)", wake_model.name, self.threshold)

    def _download(self, oww_utils: Any, names: list[str]) -> None:
        log.info(
            "Downloading openWakeWord models %s into %s", names or "(features)", self.model_dir
        )
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            oww_utils.download_models(names, target_directory=str(self.model_dir))
        except Exception as exc:
            raise BackendUnavailable(
                BACKEND,
                f"model download failed: {exc}",
                "Check network access to github.com and retry, or copy the .onnx files into "
                f"{self.model_dir}",
            ) from exc

    def score(self, frame: np.ndarray) -> float:
        """Best wake-word score in [0, 1] for exactly 1280 int16 samples."""
        pcm = np.asarray(frame)
        if pcm.shape != (FRAME,):
            raise ValueError(f"openWakeWord needs frames of {FRAME} samples, got shape {pcm.shape}")
        if pcm.dtype != np.int16:
            raise ValueError(f"openWakeWord needs int16 samples, got {pcm.dtype}")
        if self._model is None:
            raise RuntimeError("OpenWakeWord is closed")
        scores = self._model.predict(pcm)
        self.last_score = float(max(scores.values())) if scores else 0.0
        return self.last_score

    def process(self, frame: np.ndarray) -> bool:
        if self.score(frame) >= self.threshold:
            self.reset()  # avoid re-triggering on the same utterance
            return True
        return False

    def reset(self) -> None:
        """Clear openWakeWord's prediction and feature buffers."""
        if self._model is not None:
            self._model.reset()
        self.last_score = 0.0

    def close(self) -> None:
        """ONNX sessions are released with the model object."""
        self._model = None
