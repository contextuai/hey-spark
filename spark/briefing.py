"""The briefing a new Claude session starts with (Phase 5, P5.6; memory joins it in P5.9).

Every wake starts a fresh Claude conversation (decision D2), so the user never pays for an
all-day context. Continuity comes from this briefing, appended to the system prompt when the
session connects:

* **Connector notes** (P5.9, :mod:`spark.memory`): what Spark learned earlier about each enabled
  connector (where data lives, queries that worked), within ``memory.briefing_chars``.
* **Carry-over:** if the previous conversation ended less than ``attention.carry_over_s``
  ago, its last ``attention.carry_over_exchanges`` exchanges, word for word, so "what was that
  number again?" still works after a short sleep.

It is built from the activity log, never from the CLI's own session files.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.activity import ActivityLog
    from spark.config import Settings
    from spark.memory import MemoryStore

__all__ = ["Conversation", "build_briefing", "conversations", "last_conversation"]

#: Longest single line quoted in the carry-over (characters).
MAX_LINE_CHARS = 500


@dataclass
class Conversation:
    """One conversation reassembled from activity events."""

    id: str
    started: dt.datetime | None = None
    ended: dt.datetime | None = None
    end_reason: str | None = None
    exchanges: list[tuple[str, str]] = field(default_factory=list)  # ("user"|"assistant", text)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        return self.ended is not None


def _ts(event: dict[str, Any]) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(event.get("ts")))
    except ValueError:
        return None


def conversations(events: Iterable[dict[str, Any]]) -> list[Conversation]:
    """Group events by conversation id, in order of first appearance."""
    by_id: dict[str, Conversation] = {}
    for event in events:
        conv_id = event.get("conv")
        if not conv_id:
            continue
        conv = by_id.get(conv_id)
        if conv is None:
            conv = by_id[conv_id] = Conversation(conv_id, started=_ts(event))
        conv.events.append(event)
        kind = event.get("kind")
        if kind == "conversation_end":
            conv.ended = _ts(event)
            conv.end_reason = event.get("reason")
        elif kind == "user" and event.get("text"):
            conv.exchanges.append(("user", str(event["text"])))
        elif kind == "assistant" and event.get("text"):
            conv.exchanges.append(("assistant", str(event["text"])))
    return list(by_id.values())


def last_conversation(activity: ActivityLog, now: dt.datetime) -> Conversation | None:
    """The most recent closed conversation from today's or yesterday's log."""
    days = [d for d in activity.days() if d >= now.date() - dt.timedelta(days=1)]
    closed = [c for c in conversations(activity.iter_events(days)) if c.closed and c.exchanges]
    return closed[-1] if closed else None


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LINE_CHARS else text[: MAX_LINE_CHARS - 1] + "…"


def _ago(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "less than a minute ago"
    if minutes == 1:
        return "a minute ago"
    return f"about {minutes} minutes ago"


def carry_over(
    activity: ActivityLog,
    *,
    now: dt.datetime,
    within_s: float,
    exchanges: int,
    name: str = "Spark",
) -> str:
    conv = last_conversation(activity, now)
    if conv is None or conv.ended is None:
        return ""
    age = (now - conv.ended).total_seconds()
    if age < 0 or age > within_s:
        return ""
    lines = [
        f"{'User' if role == 'user' else name}: {_clip(text)}"
        for role, text in conv.exchanges[-2 * exchanges :]
    ]
    return (
        f"Earlier conversation with the user (it ended {_ago(age)}). It is context, not a new "
        'request: use it when the user refers back to it ("what was that number again?", '
        '"and for last month?"), and don\'t bring it up on your own.\n' + "\n".join(lines)
    )


def build_briefing(
    settings: Settings,
    activity: ActivityLog,
    *,
    now: dt.datetime | None = None,
    memory: MemoryStore | None = None,
    connectors: Iterable[str] = (),
) -> str:
    """The text appended to a new session's system prompt ('' when there is nothing)."""
    now = now or dt.datetime.now().astimezone()
    cfg = settings.attention
    parts = []
    if memory is not None:
        parts.append(memory.connector_briefing(connectors, settings.memory.briefing_chars))
    parts += [
        carry_over(
            activity,
            now=now,
            within_s=cfg.carry_over_s,
            exchanges=cfg.carry_over_exchanges,
            name=settings.app.name,
        )
    ]
    return "\n\n".join(p for p in parts if p)
