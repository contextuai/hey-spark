"""Spark's memory: Markdown files the user can read and edit (Phase 5, P5.9; decision D5).

This first part is **connector notes**: what Spark learned about a connected system so it
doesn't have to rediscover it on every question ("orders are in WAREHOUSE table X, column
move_out_date"). Without them, every data question starts with list databases, list schemas,
list tables and guesses. With them, the deep model can query directly.

Layout under ``memory.dir`` (next to ``config.yaml``; the installed app's user folder)::

    memory/
      connectors/
        WAREHOUSE.md        # one file per MCP server

A connector file is plain Markdown: a heading, then one bullet per note. Each bullet ends with
where it came from, ``(2026-10-02, c-20261002-161303-ab12)``, the conversation id in the
activity log. The user may edit or delete anything. Spark writes through the in-process tools
``save_connector_note`` (append one note) and ``rewrite_connector_notes`` (replace the file, to
tidy up or correct). Notes are about structure (tables, columns, filters, query patterns that
worked), never secrets: everything is redacted before it is written.

Every new session's briefing (:mod:`spark.briefing`) includes the notes of the connectors that
are switched on, within ``memory.briefing_chars``.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["MemoryStore", "MemoryWriteError"]

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_SOURCE = re.compile(r"\s*\((\d{4}-\d{2}-\d{2})(?:,\s*(c-[\w-]+))?\)\s*$")


class MemoryWriteError(ValueError):
    """A memory write that can't be done; the message is spoken back through the model."""


def _redact(text: str) -> str:
    from spark.util.logging import redact

    return redact(text)


class MemoryStore:
    """Reads and writes the memory folder. Build with :meth:`from_settings`.

    Args:
        directory: the memory folder.
        note_chars: longest single note.
        file_chars: largest connector file; past it, Spark is asked to rewrite (consolidate).
        today: date source (tests pass a fixed one).
        conversation: returns the open conversation id for a note's source (the activity log's).
    """

    def __init__(
        self,
        directory: Path,
        *,
        note_chars: int = 600,
        file_chars: int = 6000,
        today: Callable[[], dt.date] = dt.date.today,
        conversation: Callable[[], str | None] = lambda: None,
    ) -> None:
        self.dir = Path(directory)
        self.note_chars = note_chars
        self.file_chars = file_chars
        self._today = today
        self.conversation = conversation

    @classmethod
    def from_settings(cls, settings: Settings) -> MemoryStore:
        cfg = settings.memory
        return cls(
            settings.path(cfg.dir),
            note_chars=cfg.note_chars,
            file_chars=cfg.connector_file_chars,
        )

    # -- connector notes ----------------------------------------------------------------------

    def connector_path(self, server: str) -> Path:
        name = _SAFE.sub("_", server.strip()) or "connector"
        return self.dir / "connectors" / f"{name}.md"

    def connector_notes(self, server: str) -> str:
        """The connector's notes file as written ('' if none)."""
        path = self.connector_path(server)
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
        except OSError as exc:
            log.warning("memory: could not read %s: %s", path.name, exc)
            return ""

    def connector_bullets(self, server: str) -> list[str]:
        """The note texts (bullets) without their source suffix."""
        out = []
        for line in self.connector_notes(server).splitlines():
            if (m := _BULLET.match(line)) is not None:
                out.append(_SOURCE.sub("", m.group(1)).strip())
        return [b for b in out if b]

    def _source(self) -> str:
        conv = self.conversation()
        day = self._today().isoformat()
        return f"({day}, {conv})" if conv else f"({day})"

    def _write(self, path: Path, text: str) -> None:
        from spark.util.files import write_text_atomic

        write_text_atomic(path, text)

    def add_connector_note(self, server: str, note: str) -> str:
        """Append one note. Returns a short confirmation for the model."""
        note = " ".join(_redact(str(note or "")).split())
        if not note:
            raise MemoryWriteError("The note is empty.")
        if len(note) > self.note_chars:
            raise MemoryWriteError(
                f"Keep a note under {self.note_chars} characters: one fact, the table, the "
                "columns, the filter."
            )
        if note.lower() in (b.lower() for b in self.connector_bullets(server)):
            return f"Already noted for {server}."
        path = self.connector_path(server)
        current = self.connector_notes(server)
        if len(current) + len(note) > self.file_chars:
            raise MemoryWriteError(
                f"The notes for {server} are full. Rewrite them shorter with "
                "rewrite_connector_notes, keeping what is still true, then add this."
            )
        head = current or f"# {server} notes\n\nWhat Spark learned about {server}. Edit freely."
        self._write(path, f"{head}\n- {note} {self._source()}\n")
        log.info("memory: note added for %s", server)
        return f"Noted for {server}."

    def rewrite_connector_notes(self, server: str, notes: Iterable[str] | str) -> str:
        """Replace a connector's notes with ``notes`` (bullets or lines). Returns a confirmation."""
        if isinstance(notes, str):
            lines = [ln for ln in notes.splitlines() if ln.strip()]
        else:
            lines = [str(n) for n in notes if str(n).strip()]
        cleaned = []
        for line in lines:
            m = _BULLET.match(line)
            text = " ".join(_redact(m.group(1) if m else line).split())
            if text.startswith("#"):
                continue  # headings are ours
            if text:
                cleaned.append(text[: self.note_chars])
        body = "\n".join(
            f"- {t}" if _SOURCE.search(t) else f"- {t} {self._source()}" for t in cleaned
        )
        text = f"# {server} notes\n\nWhat Spark learned about {server}. Edit freely.\n{body}\n"
        if len(text) > self.file_chars + 200:
            raise MemoryWriteError(f"Still too long; keep the notes for {server} shorter.")
        self._write(self.connector_path(server), text)
        log.info("memory: notes rewritten for %s (%d)", server, len(cleaned))
        return f"Rewrote the notes for {server} ({len(cleaned)} notes)."

    def write_connector_notes(self, server: str, text: str) -> None:
        """Replace a connector's notes file with ``text`` as the user wrote it (the window)."""
        text = _redact(text).strip()
        path = self.connector_path(server)
        if not text:
            path.unlink(missing_ok=True)
            return
        self._write(path, text + "\n")

    def connectors_with_notes(self) -> list[str]:
        folder = self.dir / "connectors"
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.md"))

    def connector_briefing(self, servers: Iterable[str], max_chars: int) -> str:
        """Notes for ``servers`` (the enabled connectors), within ``max_chars``."""
        parts: list[str] = []
        used = 0
        for server in servers:
            bullets = self.connector_bullets(server)
            if not bullets:
                continue
            block = f"{server}:\n" + "\n".join(f"- {b}" for b in bullets)
            if used + len(block) > max_chars:
                room = max_chars - used
                if room < 200:
                    break
                block = block[:room].rsplit("\n", 1)[0]
            parts.append(block)
            used += len(block)
        if not parts:
            return ""
        return (
            "What you already know about the connected systems (your own notes from earlier "
            "work; the user can edit them). Use them first: query directly instead of listing "
            "databases, schemas and tables again. If a note turns out wrong, correct it with "
            "rewrite_connector_notes.\n" + "\n\n".join(parts)
        )
