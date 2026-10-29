"""macOS microphone permission for the desktop app.

macOS asks the user for microphone access (TCC) on an app's first recording. PortAudio only
triggers that prompt when it opens a stream, and an app without access may see no usable input
device at all, so the app can fail with "no usable input device" before the prompt ever shows.
:func:`ensure` asks for access up front through AVFoundation, before PortAudio starts.

Uses pyobjc-core (already in the macOS app: pywebview needs it) and loads AVFoundation at
runtime, so no extra dependency. Everything is best effort: when pyobjc or AVFoundation isn't
available the status is ``"unknown"`` and the caller carries on as before.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

log = logging.getLogger(__name__)

AV_FOUNDATION = "/System/Library/Frameworks/AVFoundation.framework"
AV_MEDIA_TYPE_AUDIO = "soun"  # the value of AVMediaTypeAudio
# AVAuthorizationStatus
STATUSES = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}
REQUEST_TIMEOUT_S = 120.0  # how long to wait for the user to answer the prompt

SETTINGS_FIX = (
    "open System Settings > Privacy & Security > Microphone, turn on Hey Spark, "
    "then quit and reopen Hey Spark"
)

_capture_device: Any = None


def _av_capture_device() -> Any:
    """AVFoundation's ``AVCaptureDevice`` class, loaded once (raises if unavailable)."""
    global _capture_device
    if _capture_device is None:
        import objc

        objc.loadBundle("AVFoundation", {}, bundle_path=AV_FOUNDATION)
        objc.registerMetaDataForSelector(
            b"AVCaptureDevice",
            b"requestAccessForMediaType:completionHandler:",
            {
                "arguments": {
                    3: {
                        "callable": {
                            "retval": {"type": b"v"},
                            "arguments": {0: {"type": b"^v"}, 1: {"type": b"Z"}},
                        }
                    }
                }
            },
        )
        _capture_device = objc.lookUpClass("AVCaptureDevice")
    return _capture_device


def status() -> str:
    """``authorized``, ``denied``, ``restricted``, ``not_determined`` or ``unknown``."""
    if sys.platform != "darwin":
        return "unknown"
    try:
        code = _av_capture_device().authorizationStatusForMediaType_(AV_MEDIA_TYPE_AUDIO)
    except Exception as exc:
        log.debug("microphone permission status unavailable: %s", exc)
        return "unknown"
    return STATUSES.get(int(code), "unknown")


def request(timeout: float = REQUEST_TIMEOUT_S) -> bool:
    """Show the macOS microphone prompt and wait for the answer. True if access was granted."""
    answered = threading.Event()
    granted: list[bool] = []

    def done(ok: bool) -> None:
        granted.append(bool(ok))
        answered.set()

    try:
        _av_capture_device().requestAccessForMediaType_completionHandler_(AV_MEDIA_TYPE_AUDIO, done)
    except Exception as exc:
        log.warning("could not ask for microphone access: %s", exc)
        return False
    if not answered.wait(timeout):
        log.warning("no answer to the microphone prompt after %.0f s", timeout)
        return False
    return granted[0]


def ensure() -> str:
    """Ask for microphone access if macOS hasn't asked yet. Returns the resulting status.

    Only the frozen macOS app asks: from a terminal, macOS attributes the mic to the terminal
    app, which prompts on its own when the stream opens.
    """
    if sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return "unknown"
    current = status()
    if current == "not_determined":
        log.info("asking macOS for microphone access")
        current = "authorized" if request() else status()
        log.info("microphone access: %s", current)
    return current
