"""Porcupine wake word ("Hey Spark" custom .ppn), the default backend (spec §4).

The AccessKey is read from the environment variable named in
``wakeword.options.porcupine.access_key_env`` and is never logged or put in an error message.
Porcupine's own error text echoes the key back ("Failed to parse AccessKey `...`"), so every
message we surface is redacted first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spark.interfaces import BackendUnavailable, WakeWordDetector

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

BACKEND = "wakeword/porcupine"
KEY_FIX = "Get a key at https://console.picovoice.ai and set PICOVOICE_ACCESS_KEY in .env"


def _ppn_fix(path: Path) -> str:
    return (
        'Train "Hey Spark" for Windows (x86_64) in Picovoice Console '
        f"(https://console.picovoice.ai > Porcupine) and save the .ppn to {path}"
    )


def _key_fix(env_name: str) -> str:
    return KEY_FIX.replace("PICOVOICE_ACCESS_KEY", env_name)


def redact(text: str, secret: str) -> str:
    """Remove ``secret`` from ``text`` (no-op for an empty secret)."""
    return text.replace(secret, "***") if secret else text


class PorcupineWakeWord(WakeWordDetector):
    """Picovoice Porcupine on 512-sample 16 kHz int16 frames."""

    def __init__(self, settings: Settings, **deps: Any) -> None:
        opts = settings.wakeword.options.porcupine
        self.deps = deps
        self.keyword_path = settings.path(opts.keyword_path)
        self.sensitivity = float(opts.sensitivity)
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError(f"porcupine sensitivity must be in [0, 1], got {self.sensitivity}")

        access_key = os.environ.get(opts.access_key_env, "").strip()
        if not access_key:
            raise BackendUnavailable(
                BACKEND,
                f"environment variable {opts.access_key_env} is not set",
                _key_fix(opts.access_key_env),
            )
        if not self.keyword_path.is_file():
            raise BackendUnavailable(
                BACKEND,
                f"keyword file not found: {self.keyword_path}",
                _ppn_fix(self.keyword_path),
            )
        try:
            import pvporcupine
        except ImportError as exc:
            raise BackendUnavailable(
                BACKEND, f"pvporcupine is not installed ({exc})", "uv sync"
            ) from exc

        try:
            self._handle = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[str(self.keyword_path)],
                sensitivities=[self.sensitivity],
                device="cpu",
            )
        except Exception as exc:
            reason, fix = _map_error(pvporcupine, exc, self.keyword_path, opts.access_key_env)
            # ``from None``: the original exception text may contain the key.
            raise BackendUnavailable(BACKEND, redact(reason, access_key), fix) from None
        self.frame_length = int(self._handle.frame_length)
        self.sample_rate = int(self._handle.sample_rate)
        log.info(
            "Porcupine ready: %s (frame %d, sensitivity %.2f)",
            self.keyword_path.name,
            self.frame_length,
            self.sensitivity,
        )

    def process(self, frame: np.ndarray) -> bool:
        if self._handle is None:
            raise RuntimeError("PorcupineWakeWord is closed")
        pcm = np.asarray(frame)
        if pcm.shape != (self.frame_length,):
            raise ValueError(
                f"Porcupine needs frames of {self.frame_length} samples, got shape {pcm.shape}"
            )
        if pcm.dtype != np.int16:
            raise ValueError(f"Porcupine needs int16 samples, got {pcm.dtype}")
        return self._handle.process(pcm) >= 0

    def reset(self) -> None:
        """Porcupine keeps no state across frames that needs clearing."""

    def close(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle is not None:
            handle.delete()


def _map_error(pv: Any, exc: Exception, ppn: Path, env_name: str) -> tuple[str, str]:
    """Turn a Porcupine exception into (reason, fix). Callers redact the reason."""
    lines = str(exc).splitlines()
    message = getattr(exc, "message", None) or (lines[0] if lines else "") or type(exc).__name__
    key_fix = _key_fix(env_name)

    def is_(name: str) -> bool:
        cls = getattr(pv, name, None)
        return isinstance(cls, type) and isinstance(exc, cls)

    if is_("PorcupineActivationLimitError"):
        return f"AccessKey activation limit reached ({message})", (
            "Check device/usage limits for your key at https://console.picovoice.ai"
        )
    if is_("PorcupineActivationThrottledError"):
        return f"AccessKey activation throttled ({message})", "Wait a few minutes and retry"
    if is_("PorcupineActivationRefusedError"):
        return f"AccessKey refused ({message})", key_fix
    if is_("PorcupineActivationError"):
        return f"AccessKey activation failed; check network and key ({message})", key_fix
    if is_("PorcupineIOError"):
        return f"could not read keyword file {ppn} ({message})", _ppn_fix(ppn)
    if is_("PorcupineKeyError") or is_("PorcupineInvalidArgumentError"):
        detail = " ".join(getattr(exc, "message_stack", []) or [])
        if "AccessKey" in message or "AccessKey" in detail:
            return f"invalid AccessKey in {env_name}", key_fix
        return (
            f"invalid argument; the .ppn may be for another platform or Porcupine version "
            f"({message})",
            _ppn_fix(ppn),
        )
    if is_("PorcupineError"):
        return f"Porcupine failed to initialise ({type(exc).__name__}: {message})", key_fix
    if isinstance(exc, ValueError):
        return f"invalid Porcupine arguments ({message})", key_fix
    return f"unexpected error {type(exc).__name__}: {message}", key_fix
