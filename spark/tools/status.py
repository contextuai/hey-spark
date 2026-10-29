"""``spark_status``: "what are you working on?" as a short speakable answer.

The app implements a :data:`StatusProvider`: a zero-argument callable (sync or async) returning a
plain dict. Recognized keys, all optional:

- ``state``: current app state, e.g. "thinking" or "idle"
- ``working_on``: a short description of the task in progress
- ``model``: the active brain model name
- ``backends``: ``{"wakeword": "porcupine", "stt": "faster_whisper", ...}``
- ``uptime_s``: seconds since start
- ``connectors``: list of connected MCP server names

Anything that looks like a secret (by key name or value shape) is dropped, never spoken.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from spark.tools.timers import TimerService, speak_duration

log = logging.getLogger(__name__)

StatusProvider = Callable[[], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]

_SECRET_KEY = re.compile(r"key|token|secret|password|passwd|auth|credential|cookie|bearer", re.I)
_SECRET_VALUE = re.compile(
    r"(?:sk-|ghp_|gho_|github_pat_|xox[bap]-|Bearer\s)|[A-Za-z0-9+/=_]{32,}|[0-9a-fA-F]{16,}"
)
_COMPONENT_NAMES = {
    "wakeword": "wake word",
    "wake_word": "wake word",
    "vad": "voice detection",
    "stt": "speech recognition",
    "tts": "voice",
    "brain": "brain",
    "audio_in": "microphone",
    "audio_out": "speaker",
}


def _is_secret(key: str, value: Any) -> bool:
    if _SECRET_KEY.search(key):
        return True
    return isinstance(value, str) and bool(_SECRET_VALUE.search(value))


def _speakable(value: Any) -> str:
    text = str(value).replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _clean_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in data.items():
        key = str(key)
        if _is_secret(key, value):
            continue
        if isinstance(value, Mapping):
            clean[key] = _clean_mapping(value)
        elif isinstance(value, list | tuple):
            clean[key] = [v for v in value if not _is_secret(key, v)]
        else:
            clean[key] = value
    return clean


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def describe_status(status: Mapping[str, Any]) -> str:
    """Render a status dict (already fetched) as speakable sentences, secrets removed."""
    data = _clean_mapping(status)
    sentences: list[str] = []

    working_on = data.get("working_on")
    state = data.get("state")
    if working_on:
        sentences.append(f"I'm working on {_speakable(working_on)}.")
    elif state:
        sentences.append(f"I'm {_speakable(state).lower()} right now.")

    backends = data.get("backends")
    if isinstance(backends, Mapping) and backends:
        parts = [
            f"{_speakable(name)} for {_COMPONENT_NAMES.get(str(comp), _speakable(comp))}"
            for comp, name in backends.items()
            if name
        ]
        if parts:
            sentences.append(f"I'm using {_join(parts)}.")

    model = data.get("model")
    if model:
        # Drop a trailing snapshot date ("-20251001") so it isn't read out as a number.
        model_name = re.sub(r"[-_]\d{8}$", "", str(model))
        sentences.append(f"My model is {_speakable(model_name)}.")

    connectors = data.get("connectors")
    if isinstance(connectors, list | tuple) and connectors:
        names = [_speakable(c) for c in connectors]
        sentences.append(f"Connected to {_join(names)}.")

    uptime = data.get("uptime_s")
    if isinstance(uptime, int | float) and not isinstance(uptime, bool) and uptime >= 0:
        sentences.append(f"I've been up for {speak_duration(uptime)}.")

    return " ".join(sentences)


def describe_timers(timers: TimerService | None) -> str:
    """ "No timers running." / "You have 2 timers: pasta with 4 minutes left, and ..."."""
    active = timers.list() if timers is not None else []
    if not active:
        return "No timers running."
    parts = [f"{t.spoken_name} with {speak_duration(t.remaining)} left" for t in active]
    if len(parts) == 1:
        return f"You have one timer: {parts[0]}."
    return f"You have {len(parts)} timers: {_join(parts)}."


async def spark_status_text(provider: StatusProvider | None, timers: TimerService | None) -> str:
    """Full ``spark_status`` answer. Never raises; a broken provider still yields timers."""
    status_text = ""
    if provider is not None:
        try:
            result = provider()
            if inspect.isawaitable(result):
                result = await result
            status_text = describe_status(result or {})
        except Exception:
            log.exception("status provider failed")
            status_text = "I couldn't read my own status just now."
    if not status_text:
        status_text = "I'm up and listening."
    return f"{status_text} {describe_timers(timers)}"
