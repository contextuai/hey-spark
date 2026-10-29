"""Choices made in the app that must survive a restart (Phase 5, P5.7c).

``config.yaml`` belongs to the user (comments and all), so the app never rewrites it. What the
user picks in the window or by voice goes to ``prefs.json`` next to it instead, and is applied
on top of the loaded settings at startup:

* ``model_mode``: ``auto`` (fast/deep routing), ``fast`` or ``deep`` (every request on that model)
* ``fast_model`` / ``deep_model``: the model id behind each route
* ``worker_model``: the researcher's model ('' = the same as ``fast_model``)
* ``openai_fast_model`` / ``openai_deep_model`` / ``openai_worker_model``: the same for
  ``brain.backend: openai_compat`` (P5.7e), kept apart so switching backends never puts a Claude
  id on another provider or the other way round
* ``brain_backend`` (``claude_agent`` / ``openai_compat``), ``openai_provider``,
  ``openai_base_url`` and ``openai_api_version``: the Brain window's choice of provider
  (P5.7f). ``''`` for the URL or version means "the preset's". API keys are never here; they go
  to ``.env``.
* ``openai_saved``: per provider, what was last saved in the Brain window (to prefill it when
  the user switches back), and ``brain_previous``: the brain keys before the last save (revert).
* ``persona``, ``address_as``, ``wake_reply`` and ``tts_voice`` (``piper:<voice>`` or
  ``kokoro:<voice>``): the Personality & Voice window's choices (spark/persona.py).

Also here: the model choices offered in the window, and friendly spoken names for model ids.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = [
    "BRAIN_KEYS",
    "MODEL_CHOICES",
    "MODES",
    "VOICE_KEYS",
    "Prefs",
    "apply_prefs",
    "apply_voice_prefs",
    "model_choices",
    "model_info",
    "model_label",
    "pref_updates",
    "reset_brain",
    "route_models_cfg",
    "set_route_models",
]

MODES = ("auto", "fast", "deep")
BACKENDS = ("claude_agent", "openai_compat")

#: Everything the Brain window saves (and restores on revert).
BRAIN_KEYS = (
    "brain_backend",
    "openai_provider",
    "openai_base_url",
    "openai_api_version",
    "openai_fast_model",
    "openai_deep_model",
    "openai_worker_model",
    "fast_model",
    "deep_model",
    "worker_model",
)

#: Everything the Personality & Voice window saves.
VOICE_KEYS = ("persona", "address_as", "wake_reply", "tts_voice")

#: Offered in the window. Any other id can still be set in config.yaml.
MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("claude-haiku-5-5", "Haiku 5.5 (fastest)"),
    ("claude-sonnet-5", "Sonnet 5"),
    ("claude-sonnet-5-5", "Sonnet 5.5"),
    ("claude-opus-5-5", "Opus 5.5 (most capable)"),
    ("claude-fable-5-1", "Fable 5.1"),
)

_FAMILY = re.compile(r"claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$")


def model_label(model_id: str) -> str:
    """A spoken name: ``claude-sonnet-5`` -> "Sonnet 5", ``claude-haiku-4-5-20251001`` ->
    "Haiku 4.5". Unknown shapes come back unchanged."""
    m = _FAMILY.match(model_id.strip())
    if m is None:
        return model_id
    family, major, minor = m.groups()
    version = f"{major}.{minor}" if minor else major
    return f"{family.capitalize()} {version}"


def _openai(settings: Settings) -> bool:
    return settings.brain.backend == "openai_compat"


def route_models_cfg(settings: Settings) -> Any:
    """The config section holding the active backend's model per role: ``brain`` (Claude ids)
    or ``brain.openai_compat``."""
    return settings.brain.openai_compat if _openai(settings) else settings.brain


def set_route_models(
    settings: Settings, fast: str | None, deep: str | None, worker: str | None
) -> None:
    """Set the active backend's models in ``settings`` (None = unchanged; worker '' = same as
    fast)."""
    cfg = route_models_cfg(settings)
    if fast:
        cfg.fast_model = fast
    if deep:
        cfg.deep_model = deep
    if worker is not None:
        cfg.worker_model = worker.strip() or None


def pref_updates(
    settings: Settings, fast: str | None, deep: str | None, worker: str | None
) -> dict[str, Any]:
    """``prefs.update(**...)`` keyword arguments for a model change on the active backend."""
    prefix = "openai_" if _openai(settings) else ""
    return {
        f"{prefix}fast_model": fast,
        f"{prefix}deep_model": deep,
        f"{prefix}worker_model": worker,
    }


def model_info(settings: Settings, mode: str | None = None) -> dict[str, Any]:
    """What the window shows: the mode, each route's model id and its spoken label."""
    from spark.brain.openai_presets import resolved_models

    fast, deep, worker_id = resolved_models(settings)
    worker = worker_id or ""
    info = {
        "mode": mode or settings.brain.mode,
        "fast_model": fast,
        "deep_model": deep,
        "worker_model": worker,  # '' = same as fast
        "fast_label": model_label(fast),
        "deep_label": model_label(deep),
        "worker_label": model_label(worker or fast),
        "backend": settings.brain.backend,
    }
    if _openai(settings):
        from spark.brain.openai_presets import resolve_provider

        info["provider"] = resolve_provider(settings.brain.openai_compat).label
    return info


def model_choices(settings: Settings, extra: Any = ()) -> list[list[str]]:
    """``[[id, label], ...]`` for the window's pickers: the Claude choices on Claude; on
    ``openai_compat`` the configured names, the preset's suggestions and ``extra`` (what the
    endpoint's ``/models`` listed). The configured ids are always included."""
    from spark.brain.openai_presets import PRESETS, resolved_models

    if _openai(settings):
        preset = PRESETS.get(settings.brain.openai_compat.provider)
        base = [m for m in (preset.fast_model, preset.deep_model) if m] if preset else []
        ids = [*resolved_models(settings)[:2], *base, *(m for m in extra if isinstance(m, str))]
        choices: list[list[str]] = []
    else:
        ids = [settings.brain.fast_model, settings.brain.deep_model]
        choices = [[mid, label] for mid, label in MODEL_CHOICES]
    known = {mid for mid, _ in choices}
    for mid in ids:
        if mid and mid not in known:
            choices.append([mid, model_label(mid)])
            known.add(mid)
    return choices


class Prefs:
    """``prefs.json``: read once, written on every change. Unknown keys are kept."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = self._read()

    @classmethod
    def from_settings(cls, settings: Settings) -> Prefs:
        return cls(settings.path(settings.app.prefs_file))

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("prefs: %s is unreadable (%s); using config.yaml", self.path.name, exc)
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, **values: Any) -> None:
        """Set keys (None values are skipped). The file is read again first, so a change saved
        by another :class:`Prefs` on the same file (the app's, a window's) is kept."""
        self.data = {**self._read(), **{k: v for k, v in values.items() if v is not None}}
        self._write()

    def delete(self, *keys: str) -> None:
        self.data = {k: v for k, v in self._read().items() if k not in keys}
        self._write()

    def _write(self) -> None:
        try:
            from spark.util.files import write_text_atomic

            write_text_atomic(self.path, json.dumps(self.data, indent=2) + "\n")
        except OSError as exc:
            log.warning("prefs: could not save %s: %s", self.path.name, exc)


def apply_prefs(settings: Settings, prefs: Prefs) -> None:
    """Overlay the saved choices on ``settings`` (in place). Saved values win over
    ``config.yaml``; keys that were never saved leave ``config.yaml``'s values alone."""
    from spark.brain.openai_presets import PRESETS

    mode = prefs.get("model_mode")
    if mode in MODES:
        settings.brain.mode = mode
    apply_voice_prefs(settings, prefs.data)
    backend = prefs.get("brain_backend")
    if backend in BACKENDS and not settings.is_pinned("brain.backend"):
        settings.brain.backend = backend
    oc = settings.brain.openai_compat
    provider = prefs.get("openai_provider")
    if isinstance(provider, str) and provider in PRESETS and provider != oc.provider:
        # config.yaml's key variable and extra headers/body belong to its own provider.
        oc.provider = provider
        oc.api_key_env = None
        oc.extra_headers = {}
        oc.extra_body = {}
    for key in ("base_url", "api_version"):
        value = prefs.get("openai_" + key)
        if isinstance(value, str):
            setattr(oc, key, value.strip() or None)  # '' = the preset's
    for prefix, cfg in (("", settings.brain), ("openai_", settings.brain.openai_compat)):
        for key in ("fast_model", "deep_model"):
            value = prefs.get(prefix + key)
            if isinstance(value, str) and value.strip():
                setattr(cfg, key, value.strip())
        worker = prefs.get(prefix + "worker_model")
        if isinstance(worker, str):
            cfg.worker_model = worker.strip() or None  # '' = same as the fast model


def apply_voice_prefs(settings: Settings, prefs: Mapping[str, Any]) -> None:
    """The Personality & Voice window's choices (``prefs.data``; part of :func:`apply_prefs`)."""
    from spark.persona import PERSONAS

    persona = prefs.get("persona")
    if isinstance(persona, str) and persona in PERSONAS:
        settings.app.persona = persona
    address = prefs.get("address_as")
    if isinstance(address, str):
        settings.app.address_as = address.strip()
    wake = prefs.get("wake_reply")
    if isinstance(wake, bool):
        settings.app.wake_reply = wake
    voice = prefs.get("tts_voice")
    backend, _, name = voice.partition(":") if isinstance(voice, str) else ("", "", "")
    if name and backend == "piper":
        settings.tts.backend = "piper"
        settings.tts.options.piper.voice = name
    elif name and backend == "kokoro":
        settings.tts.backend = "kokoro"
        settings.tts.options.kokoro.voice = name
        settings.tts.options.kokoro.lang = None  # from the voice


#: The fields :func:`apply_prefs` may change, per section. :func:`reset_brain` puts them back.
_BRAIN_FIELDS = ("backend", "mode", "fast_model", "deep_model", "worker_model")
_OPENAI_FIELDS = (
    "provider",
    "base_url",
    "api_version",
    "api_key_env",
    "extra_headers",
    "extra_body",
    "fast_model",
    "deep_model",
    "worker_model",
)


def reset_brain(settings: Settings, baseline: Any) -> None:
    """Put the fields prefs can override back to ``baseline`` (a copy of ``settings.brain``
    taken before any prefs were applied, i.e. ``config.yaml``), in place. Followed by
    :func:`apply_prefs`, a pref that was *removed* (a revert) falls back to ``config.yaml``
    instead of keeping the value applied last."""
    for name in _BRAIN_FIELDS:
        setattr(settings.brain, name, getattr(baseline, name))
    for name in _OPENAI_FIELDS:
        value = copy.deepcopy(getattr(baseline.openai_compat, name))
        setattr(settings.brain.openai_compat, name, value)
