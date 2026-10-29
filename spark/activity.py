"""The activity log: an append-only, hash-chained, redacted record of everything (Phase 5, P5.5).

One JSON object per line, one file per local day: ``<activity.dir>/YYYY-MM-DD.jsonl``. It is the
permanent record behind the History window, conversation summaries and memory. It is kept with
no time limit unless ``activity.retention_days`` is set (decision D6), and holds full tool
results unless ``activity.tool_results`` says otherwise (D7). Secrets are redacted before
anything is written (:func:`spark.util.logging.redact`).

Every event has::

    {"seq": 12, "ts": "2026-10-02T12:04:17.123-05:00", "kind": "tool_call",
     "conv": "c-20261002-120319-3f2a", ...fields..., "prev": "<hash>", "hash": "<hash>"}

``hash = sha256(prev + canonical JSON of the event without "hash")`` and ``prev`` is the hash
of the line before it: the previous day's last line for a day's first line, and 64 zeros for the
very first line (D8, tamper-evident). :func:`verify_file` and :meth:`ActivityLog.verify`
recompute the chain.

Kinds written by the app and the brain (fields beyond the common ones):

* ``conversation_start`` (source) / ``conversation_end`` (reason)
* ``user`` (text, mode: voice | follow_up | text | answer) / ``assistant`` (text, interrupted)
* ``command`` (command, text) / ``ignored`` (text, reason, by)
* ``route`` (route, model, reason)
* ``tool_call`` (tool, id, input) / ``tool_result`` (tool, id, is_error, chars, content)
* ``confirm`` (tool, outcome: yes | no | timeout | denied | ...)
* ``turn_end`` (session_id, num_turns, cost_usd, route, error)
* ``announce`` (text) / ``error`` (error, message)

Writing never raises into the voice loop: a failed write is logged once and dropped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import secrets
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["GENESIS", "ActivityLog", "NullActivityLog", "verify_file"]

GENESIS = "0" * 64
_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")

Now = Callable[[], dt.datetime]


def _local_now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def _canonical(event: Mapping[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def event_hash(prev: str, event: Mapping[str, Any]) -> str:
    """The chain hash of ``event`` (without its own ``hash`` field) after ``prev``."""
    body = {k: v for k, v in event.items() if k != "hash"}
    return hashlib.sha256((prev + _canonical(body)).encode("utf-8")).hexdigest()


def _redact_value(value: Any, redact: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(k): _redact_value(v, redact) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v, redact) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def tool_result_text(content: Any) -> str:
    """The text of a ``ToolResultBlock.content`` (a string or MCP content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(f"[{block.get('type', 'content')}]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _read_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify_file(path: Path, prev: str | None = None) -> tuple[bool, int | None, str]:
    """Check one day's chain. ``prev`` is the previous day's last hash (None: don't check the
    first link). Returns ``(ok, bad_line_number_or_None, last_hash)``."""
    last = prev
    n = 0
    try:
        for n, event in enumerate(_read_lines(path), start=1):
            if last is not None and event.get("prev") != last:
                return False, n, last
            if event.get("hash") != event_hash(str(event.get("prev")), event):
                return False, n, last or ""
            last = str(event["hash"])
    except (OSError, ValueError, KeyError):
        return False, n + 1, last or ""
    return True, None, last or (prev or GENESIS)


class ActivityLog:
    """Append-only activity log. Build with :meth:`from_settings`.

    Args:
        directory: where the day files live.
        enabled: False makes every method a no-op (``activity.enabled``).
        tool_results: ``full`` | ``excerpt`` | ``meta`` (how much of a tool result is kept).
        excerpt_chars: the excerpt length for ``excerpt``.
        retention_days: 0 keeps everything; otherwise :meth:`purge` deletes older day files.
        redact: the redaction function (default :func:`spark.util.logging.redact`).
        now: local time source (tests pass a fixed clock).
    """

    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool = True,
        tool_results: str = "full",
        excerpt_chars: int = 2000,
        retention_days: int = 0,
        redact: Callable[[str], str] | None = None,
        now: Now = _local_now,
    ) -> None:
        if redact is None:
            from spark.util.logging import redact as _redact

            redact = _redact
        self.dir = Path(directory)
        self.enabled = enabled
        self.tool_results = tool_results
        self.excerpt_chars = excerpt_chars
        self.retention_days = retention_days
        self._redact = redact
        self._now = now
        self._seq: int | None = None
        self._last_hash: str | None = None
        self._warned = False
        self.conversation: str | None = None  # the open conversation id, if any
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> ActivityLog:
        cfg = settings.activity
        return cls(
            settings.path(cfg.dir),
            enabled=cfg.enabled,
            tool_results=cfg.tool_results,
            excerpt_chars=cfg.excerpt_chars,
            retention_days=cfg.retention_days,
        )

    # -- files --------------------------------------------------------------------------------

    def day_path(self, day: dt.date) -> Path:
        return self.dir / f"{day.isoformat()}.jsonl"

    def days(self) -> list[dt.date]:
        """Days with a log file, oldest first."""
        if not self.dir.is_dir():
            return []
        out = []
        for p in self.dir.iterdir():
            if (m := _DAY_FILE.match(p.name)) is not None:
                out.append(dt.date.fromisoformat(m.group(1)))
        return sorted(out)

    def read_day(self, day: dt.date) -> list[dict[str, Any]]:
        path = self.day_path(day)
        if not path.exists():
            return []
        try:
            return list(_read_lines(path))
        except (OSError, ValueError) as exc:
            log.warning("activity: could not read %s: %s", path.name, exc)
            return []

    def iter_events(self, days: list[dt.date] | None = None) -> Iterator[dict[str, Any]]:
        for day in days if days is not None else self.days():
            yield from self.read_day(day)

    def _load_tail(self) -> None:
        """Find the last sequence number and hash (the newest day file's last line)."""
        self._seq, self._last_hash = 0, GENESIS
        for day in reversed(self.days()):
            events = self.read_day(day)
            if events:
                last = events[-1]
                self._seq = int(last.get("seq", 0))
                self._last_hash = str(last.get("hash", GENESIS))
                return

    # -- writing ------------------------------------------------------------------------------

    def new_conversation_id(self) -> str:
        stamp = self._now().strftime("%Y%m%d-%H%M%S")
        return f"c-{stamp}-{secrets.token_hex(2)}"

    def start_conversation(self, source: str) -> str:
        """Open a conversation (closing any open one first). Returns its id."""
        if self.conversation is not None:
            self.end_conversation("superseded")
        self.conversation = self.new_conversation_id()
        self.record("conversation_start", source=source)
        return self.conversation

    def end_conversation(self, reason: str) -> str | None:
        conv = self.conversation
        if conv is None:
            return None
        self.record("conversation_end", reason=reason)
        self.conversation = None
        return conv

    def ensure_conversation(self, source: str) -> str:
        return self.conversation or self.start_conversation(source)

    def record(self, kind: str, **fields: Any) -> dict[str, Any] | None:
        """Append one event. Returns it (as written), or None when disabled or on failure."""
        if not self.enabled:
            return None
        try:
            if self._seq is None:
                self._load_tail()
            assert self._seq is not None and self._last_hash is not None
            now = self._now()
            event: dict[str, Any] = {
                "seq": self._seq + 1,
                "ts": now.isoformat(timespec="milliseconds"),
                "kind": kind,
                "conv": fields.pop("conv", self.conversation),
            }
            for key, value in fields.items():
                if value is not None:
                    event[key] = _redact_value(value, self._redact)
            event["prev"] = self._last_hash
            event["hash"] = event_hash(self._last_hash, event)
            path = self.day_path(now.date())
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._seq = event["seq"]
            self._last_hash = event["hash"]
        except Exception as exc:
            if not self._warned:
                self._warned = True
                log.warning("activity: could not write (%s); this event is lost", exc)
            return None
        for listener in list(self.listeners):
            try:
                listener(event)
            except Exception as exc:
                log.debug("activity listener failed: %s", exc)
        return event

    # convenience writers -----------------------------------------------------------------------

    def user(self, text: str, mode: str) -> None:
        self.record("user", text=text, mode=mode)

    def assistant(self, text: str, *, interrupted: bool = False) -> None:
        self.record("assistant", text=text, interrupted=interrupted or None)

    def tool_call(self, tool: str, tool_id: str | None, tool_input: Any) -> None:
        self.record("tool_call", tool=tool, id=tool_id, input=tool_input)

    def tool_result(
        self, tool: str | None, tool_id: str | None, content: Any, is_error: bool | None
    ) -> None:
        text = tool_result_text(content)
        fields: dict[str, Any] = {"tool": tool, "id": tool_id, "is_error": bool(is_error)}
        fields["chars"] = len(text)
        if self.tool_results == "full":
            fields["content"] = text
        elif self.tool_results == "excerpt":
            fields["content"] = text[: self.excerpt_chars]
            if len(text) > self.excerpt_chars:
                fields["truncated"] = True
        self.record("tool_result", **fields)

    def error(self, exc: BaseException) -> None:
        self.record("error", error=type(exc).__name__, message=str(exc)[:2000])

    # -- reading and checking -------------------------------------------------------------------

    def verify(self) -> tuple[bool, str | None]:
        """Check the whole chain, day by day. Returns ``(ok, problem)``."""
        prev: str | None = None  # the oldest kept day may follow purged ones
        for day in self.days():
            ok, line, prev_hash = verify_file(self.day_path(day), prev)
            if not ok:
                return False, f"{day.isoformat()}.jsonl line {line}"
            prev = prev_hash
        return True, None

    def purge(self) -> list[dt.date]:
        """Delete day files older than ``retention_days`` (nothing when it is 0)."""
        if self.retention_days <= 0:
            return []
        cutoff = self._now().date() - dt.timedelta(days=self.retention_days)
        removed = []
        for day in self.days():
            if day < cutoff:
                try:
                    self.day_path(day).unlink()
                    removed.append(day)
                except OSError as exc:
                    log.warning("activity: could not delete %s: %s", day, exc)
        if removed:
            log.info("activity: purged %d day(s) older than %s", len(removed), cutoff)
        return removed


class NullActivityLog(ActivityLog):
    """A disabled log, for components built without one."""

    def __init__(self) -> None:
        super().__init__(Path("."), enabled=False)
