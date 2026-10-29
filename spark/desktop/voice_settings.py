"""The Personality & Voice window's API: ``pywebview.api.<name>()`` from ``ui/voice.html``.

How Spark talks (the persona, what it calls you, the spoken reply to the wake word) and the
voice it talks in. The choices go to ``prefs.json`` (:mod:`spark.prefs`, ``VOICE_KEYS``), which
overrides ``config.yaml``; ``config.yaml`` is never written. Saving applies them to the shared
settings and asks the controller to rebuild Spark (``on_apply``), which first downloads a voice
that isn't on this computer yet.

pywebview calls these methods on its own worker threads, so each is plain blocking code that
returns a JSON-able dict with ``ok`` and, on failure, a plain ``error``. All state stays in
underscore names (pywebview exposes every public attribute).
"""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import threading
import wave
from collections.abc import Callable, Mapping
from typing import Any

from spark.config import Settings
from spark.persona import PERSONAS, VOICES, Voice, persona_for, voice_by_id, voice_id
from spark.prefs import Prefs, apply_voice_prefs

log = logging.getLogger(__name__)

__all__ = ["VoiceSettingsApi"]

MAX_ADDRESS_CHARS = 40


class VoiceSettingsApi:
    """``settings`` is the app's shared Settings: a save changes it in place, like the Brain
    window. ``on_apply`` rebuilds Spark with it and returns None or what went wrong."""

    def __init__(
        self,
        settings: Settings,
        *,
        on_apply: Callable[[], str | None] | None = None,
        prefs: Prefs | None = None,
    ) -> None:
        self._settings = settings
        self._on_apply = on_apply
        self._prefs_obj = prefs
        self._engines: dict[str, Any] = {}  # voice id -> loaded TTS, for previews
        self._lock = threading.Lock()

    def _prefs(self) -> Prefs:
        return self._prefs_obj or Prefs.from_settings(self._settings)

    # --- views ------------------------------------------------------------------------------

    @staticmethod
    def _available(backend: str) -> bool:
        module = {"piper": "piper", "kokoro": "kokoro_onnx"}.get(backend)
        return module is not None and importlib.util.find_spec(module) is not None

    def _voice_settings(self, voice: Voice) -> Settings:
        """A copy of the settings speaking with ``voice``."""
        s = self._settings.model_copy(deep=True)
        s.tts.backend = voice.backend
        if voice.backend == "piper":
            s.tts.options.piper.voice = voice.voice
        else:
            s.tts.options.kokoro.voice = voice.voice
            s.tts.options.kokoro.lang = None
        return s

    def _downloaded(self, voice: Voice) -> bool:
        from spark.models import model_status

        return bool(model_status(self._voice_settings(voice))[voice.backend]["present"])

    def load(self) -> dict[str, Any]:
        s = self._settings
        voices = persona_for(s)
        current = voice_id(s)
        listed = [
            {
                "id": v.id,
                "label": v.label,
                "accent": v.accent,
                "size": v.size,
                "license": v.license,
                "available": self._available(v.backend),
                "downloaded": self._downloaded(v),
            }
            for v in VOICES
        ]
        if voice_by_id(current) is None:  # set in config.yaml: keep it selectable
            listed.insert(
                0,
                {
                    "id": current,
                    "label": f"{current} (from config.yaml)",
                    "accent": "",
                    "size": "",
                    "license": "",
                    "available": True,
                    "downloaded": True,
                    "custom": True,
                },
            )
        return {
            "ok": True,
            "name": s.app.name,
            "personas": [
                {
                    "key": p.key,
                    "label": p.label,
                    "description": p.description,
                    "voice": p.voice,
                    "wake_reply": p.wake_reply,
                    "wake_line": p.phrases.get("wake", ""),
                }
                for p in PERSONAS.values()
            ],
            "persona": voices.persona.key,
            "address_as": s.app.address_as,
            "wake_reply": voices.wake_reply_on,
            "voice": current,
            "voices": listed,
        }

    # --- preview ----------------------------------------------------------------------------

    def preview(self, form: Mapping[str, Any]) -> dict[str, Any]:
        """Speak the persona's sample line in a voice: a WAV data URL for the page to play."""
        vid = str(form.get("voice") or "")
        voice = voice_by_id(vid)
        if voice is None:
            if vid != voice_id(self._settings):
                return {"ok": False, "error": "Unknown voice."}
            settings = self._settings
        else:
            if not self._available(voice.backend):
                return {"ok": False, "error": f"{voice.label} isn't installed in this build."}
            if not self._downloaded(voice):
                return {
                    "ok": False,
                    "error": f"{voice.label} downloads when you save (about {voice.size}). "
                    "You can preview it after that.",
                }
            settings = self._voice_settings(voice)
        try:
            text = self._sample_text(form)
            pcm, rate = self._engine(vid, settings).render(text)
        except Exception as exc:
            log.warning("voice preview failed: %s", exc)
            return {"ok": False, "error": f"Couldn't play the sample: {exc}"}
        return {"ok": True, "text": text, "audio": _wav_data_url(pcm, rate)}

    def _sample_text(self, form: Mapping[str, Any]) -> str:
        s = self._settings.model_copy(deep=True)
        self._apply_form(s, form)
        return persona_for(s).line("sample")

    def _engine(self, vid: str, settings: Settings) -> Any:
        from spark import registry

        with self._lock:
            engine = self._engines.get(vid)
            if engine is None:
                engine = registry.build("tts", settings)
                self._engines = {vid: engine}  # keep one loaded voice, not all of them
            return engine

    # --- save -------------------------------------------------------------------------------

    @staticmethod
    def _apply_form(settings: Settings, form: Mapping[str, Any]) -> dict[str, Any]:
        """Validate ``form`` and apply it to ``settings``; returns the prefs to save."""
        persona = str(form.get("persona") or "standard")
        if persona not in PERSONAS:
            raise ValueError("Unknown personality.")
        address = " ".join(str(form.get("address_as") or "").split())
        if len(address) > MAX_ADDRESS_CHARS:
            raise ValueError(f"Keep what it calls you under {MAX_ADDRESS_CHARS} characters.")
        values: dict[str, Any] = {
            "persona": persona,
            "address_as": address,
            "wake_reply": bool(form.get("wake_reply")),
        }
        vid = str(form.get("voice") or "")
        if voice_by_id(vid) is not None:
            values["tts_voice"] = vid
        elif vid and vid != voice_id(settings):
            raise ValueError("Unknown voice.")
        apply_voice_prefs(settings, values)
        return values

    def save(self, form: Mapping[str, Any]) -> dict[str, Any]:
        try:
            values = self._apply_form(self._settings, form)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self._prefs().update(**values)
        log.info(
            "voice settings: persona %s, voice %s, wake reply %s",
            values["persona"],
            voice_id(self._settings),
            values["wake_reply"],
        )
        error = None
        if self._on_apply is not None:
            try:
                error = self._on_apply()
            except Exception as exc:
                log.exception("voice settings: apply failed")
                error = str(exc)
        view = self.load()
        if error:
            return {**view, "ok": False, "error": error, "saved": True}
        return {**view, "saved": True}


def _wav_data_url(pcm: Any, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.astype("<i2").tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
