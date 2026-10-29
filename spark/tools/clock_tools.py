"""``current_time``: the local date, time, weekday and timezone as one speakable sentence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

NowFn = Callable[[], datetime]


def local_now() -> datetime:
    """The current local time as an aware datetime (carries the OS timezone)."""
    return datetime.now().astimezone()


def _spoken_clock(dt: datetime) -> str:
    hour12 = dt.hour % 12 or 12
    suffix = "AM" if dt.hour < 12 else "PM"
    if dt.minute == 0:
        if dt.hour == 12:
            return "12 noon"
        if dt.hour == 0:
            return "12 midnight"
        return f"{hour12} {suffix}"
    return f"{hour12}:{dt.minute:02d} {suffix}"


def _spoken_timezone(dt: datetime) -> str | None:
    if dt.tzinfo is None:
        return None
    name = (dt.tzname() or "").strip()
    offset = dt.utcoffset()
    # Names like "+0530" or "UTC+05:30" read badly; say the offset instead.
    if name and not any(ch.isdigit() for ch in name):
        return name
    if offset is None:
        return None
    minutes = int(offset.total_seconds() // 60)
    if minutes == 0:
        return "UTC"
    sign = "plus" if minutes > 0 else "minus"
    hours, mins = divmod(abs(minutes), 60)
    return f"UTC {sign} {hours}" + (f" {mins:02d}" if mins else "")


def current_time_text(now: NowFn = local_now) -> str:
    """E.g. "It's 3:07 PM on Thursday, September 24, 2026, Pacific Daylight Time."."""
    dt = now()
    date = f"{dt.strftime('%A')}, {dt.strftime('%B')} {dt.day}, {dt.year}"
    text = f"It's {_spoken_clock(dt)} on {date}"
    zone = _spoken_timezone(dt)
    return f"{text}, {zone}." if zone else f"{text}."
