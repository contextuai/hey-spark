"""Backend registry: the only place a backend name is mapped to code.

Entries are ``"module:Class"`` strings so nothing is imported until a backend is selected.
Every backend class is constructed as ``Class(settings, **deps)``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from spark import interfaces

if TYPE_CHECKING:
    from spark.config import Settings

REGISTRY: dict[str, dict[str, str | type]] = {
    "wakeword": {
        "porcupine": "spark.wakeword.porcupine:PorcupineWakeWord",
        "openwakeword": "spark.wakeword.openwakeword:OpenWakeWord",
    },
    "vad": {
        "silero": "spark.vad.silero:SileroVAD",
    },
    "stt": {
        "faster_whisper": "spark.stt.faster_whisper:FasterWhisperSTT",
        "deepgram": "spark.stt.deepgram:DeepgramSTT",
    },
    "tts": {
        "piper": "spark.tts.piper:PiperTTS",
        "kokoro": "spark.tts.kokoro_onnx:KokoroTTS",
        "elevenlabs": "spark.tts.elevenlabs:ElevenLabsTTS",
        "print": "spark.tts.console:PrintTTS",
    },
    "brain": {
        "claude_agent": "spark.brain.claude_agent:ClaudeAgentBrain",
        "openai_compat": "spark.brain.openai_compat:OpenAICompatBrain",
        "mock": "spark.brain.mock:MockBrain",
    },
    "audio_in": {
        "sounddevice": "spark.audio.stream:SoundDeviceMic",
    },
    "audio_out": {
        "sounddevice": "spark.audio.player:SoundDevicePlayer",
    },
}

BASES: dict[str, type] = {
    "wakeword": interfaces.WakeWordDetector,
    "vad": interfaces.VAD,
    "stt": interfaces.STT,
    "tts": interfaces.TTS,
    "brain": interfaces.Brain,
    "audio_in": interfaces.AudioInput,
    "audio_out": interfaces.AudioOutput,
}


class UnknownBackend(KeyError):
    """The configured backend name has no registry entry."""

    def __init__(self, kind: str, name: str) -> None:
        options = ", ".join(sorted(REGISTRY.get(kind, {}))) or "none"
        super().__init__(f"unknown {kind} backend {name!r} (choices: {options})")
        self.kind = kind
        self.name = name


def register(kind: str, name: str, target: str | type) -> None:
    """Add or replace a backend (used by tests and plugins)."""
    if kind not in BASES:
        raise KeyError(f"unknown component kind {kind!r}")
    REGISTRY.setdefault(kind, {})[name] = target


def unregister(kind: str, name: str) -> None:
    REGISTRY.get(kind, {}).pop(name, None)


def resolve(kind: str, name: str) -> type:
    """Import and return the class for ``kind``/``name`` without constructing it."""
    try:
        target = REGISTRY[kind][name]
    except KeyError as exc:
        raise UnknownBackend(kind, name) from exc
    if isinstance(target, type):
        cls = target
    else:
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, attr)
    base = BASES[kind]
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(f"{kind} backend {name!r} ({cls!r}) is not a {base.__name__}")
    return cls


def selected_name(kind: str, settings: Settings) -> str:
    """The backend name ``settings`` selects for ``kind``."""
    match kind:
        case "audio_in":
            return settings.audio.input_backend
        case "audio_out":
            return settings.audio.output_backend
        case _:
            return getattr(settings, kind).backend


def build(kind: str, settings: Settings, name: str | None = None, **deps: Any) -> Any:
    """Construct the configured backend for ``kind`` (or ``name`` if given)."""
    cls = resolve(kind, name or selected_name(kind, settings))
    return cls(settings, **deps)
