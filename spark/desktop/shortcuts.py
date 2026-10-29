"""The Shortcuts window's API (Phase 5, P5.7b): ``pywebview.api.<name>()`` from
``ui/shortcuts.html``.

Every method returns a JSON-able dict with ``ok`` and, on failure, a plain ``error`` the page
shows. Saving writes ``shortcuts.json``; the running app re-reads it on change, so nothing needs
a restart. pywebview exposes every public attribute, so all state stays in underscore names.
"""

from __future__ import annotations

import logging
from typing import Any

from spark.config import Settings
from spark.shortcuts import ShortcutError, ShortcutStore, expand, match_shortcut

log = logging.getLogger(__name__)


class ShortcutsApi:
    def __init__(self, settings: Settings, store: ShortcutStore | None = None) -> None:
        self._settings = settings
        self._store = store or ShortcutStore.from_settings(settings)

    def _view(self) -> dict[str, Any]:
        return {
            "ok": True,
            "shortcuts": [s.to_dict() for s in self._store.all()],
            "file": str(self._store.path),
            "name": self._settings.app.name,
            "max_extra_words": self._settings.shortcuts.max_extra_words,
        }

    def list(self) -> dict[str, Any]:
        return self._view()

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = self._store.save(dict(data or {}))
        except ShortcutError as exc:
            return {"ok": False, "error": str(exc)}
        except (OSError, TypeError, ValueError) as exc:
            log.warning("shortcuts: save failed: %s", exc)
            return {"ok": False, "error": f"Couldn't save: {exc}"}
        log.info("shortcuts: saved %r", saved.name)
        return {**self._view(), "saved": saved.id}

    def remove(self, shortcut_id: str) -> dict[str, Any]:
        try:
            self._store.delete(shortcut_id)
        except OSError as exc:
            return {"ok": False, "error": f"Couldn't delete: {exc}"}
        return self._view()

    def set_enabled(self, shortcut_id: str, enabled: bool) -> dict[str, Any]:
        try:
            self._store.set_enabled(shortcut_id, bool(enabled))
        except OSError as exc:
            return {"ok": False, "error": f"Couldn't save: {exc}"}
        return self._view()

    def test(self, phrase: str) -> dict[str, Any]:
        """Which shortcut ``phrase`` would fire, and the prompt Spark would send."""
        phrase = str(phrase or "").strip()
        if not phrase:
            return {"ok": True, "match": None}
        match = match_shortcut(
            phrase,
            self._store.all(),
            names=self._settings.app.wake_names,
            max_extra_words=self._settings.shortcuts.max_extra_words,
        )
        if match is None:
            return {"ok": True, "match": None}
        return {
            "ok": True,
            "match": {
                "id": match.shortcut.id,
                "name": match.shortcut.name,
                "trigger": match.trigger,
                "extra": match.extra,
                "route": match.shortcut.route,
                "prompt": expand(match, phrase),
            },
        }
