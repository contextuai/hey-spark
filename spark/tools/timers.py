"""Voice timers: set / list / cancel, each one an asyncio task sleeping on an injectable clock.

When a timer expires, ``on_fire(timer)`` is called (sync or async). The app uses that callback
to play an earcon and speak "Your pasta timer is done." Every user-facing error is a
:class:`TimerError` whose message is a short sentence that can be read aloud as-is.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from spark.util.clock import Clock, SystemClock

log = logging.getLogger(__name__)

MIN_SECONDS = 1.0
MAX_SECONDS = 24 * 3600.0
MAX_LABEL_CHARS = 40

OnFire = Callable[["Timer"], Awaitable[None] | None]


class TimerError(ValueError):
    """A timer request that can't be honored. ``str(err)`` is a speakable sentence."""


@dataclass(frozen=True)
class Timer:
    """One timer. ``id`` is a small number as a string ("1", "2"); ``label`` is optional."""

    id: str
    label: str | None
    seconds: float
    due: float
    remaining: float = 0.0

    @property
    def spoken_name(self) -> str:
        """How to say this timer out loud: "pasta timer" or "timer 2"."""
        return f"{self.label} timer" if self.label else f"timer {self.id}"

    @property
    def spoken_ref(self) -> str:
        """The name with an article where English wants one: "the pasta timer", "timer 2"."""
        return f"the {self.spoken_name}" if self.label else self.spoken_name


# --------------------------------------------------------------------------- duration parsing

_UNITS: dict[str, float] = {
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), 1.0),
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), 60.0),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), 3600.0),
}
_SMALL_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]  # fmt: skip
_TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_WORD_NUMBERS: dict[str, float] = {w: float(i) for i, w in enumerate(_SMALL_WORDS)}
_WORD_NUMBERS.update({w: float(20 + 10 * i) for i, w in enumerate(_TENS_WORDS)})
_WORD_NUMBERS.update({"couple": 2.0, "few": 3.0})
_FILLER = {"and", "plus", "for", "of", "in", "about", "just", "timer", "the"}
_ARTICLES = {"a", "an"}
_NEXT_SMALLER_UNIT = {3600.0: 60.0, 60.0: 1.0}
_CLOCK_FORM = re.compile(r"^(\d+):(\d{1,2})(?::(\d{1,2}))?$")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")

_UNPARSEABLE = "I couldn't understand that duration. Try something like five minutes."


def parse_duration(value: float | int | str) -> float:
    """Turn ``300``, ``"300"``, ``"5 minutes"``, ``"1h30m"``, ``"an hour and a half"``,
    ``"1:30"`` (minutes:seconds) or ``"1:30:00"`` into seconds.

    Does not range-check; :meth:`TimerService.set` does. Raises :class:`TimerError`.
    """
    if isinstance(value, bool):
        raise TimerError(_UNPARSEABLE)
    if isinstance(value, int | float):
        if not math.isfinite(value):
            raise TimerError(_UNPARSEABLE)
        return float(value)
    if not isinstance(value, str):
        raise TimerError(_UNPARSEABLE)

    text = value.strip().lower()
    if not text:
        raise TimerError(_UNPARSEABLE)
    if text.startswith("-"):
        raise TimerError("Timers need a positive length.")

    clock_match = _CLOCK_FORM.match(text)
    if clock_match:
        a, b, c = clock_match.groups()
        if c is None:
            return int(a) * 60.0 + int(b)
        return int(a) * 3600.0 + int(b) * 60.0 + int(c)

    tokens = _TOKEN.findall(text.replace("-", " "))
    leftover = _TOKEN.sub("", text.replace("-", " "))
    if not tokens or leftover.strip(" ,.&"):
        raise TimerError(_UNPARSEABLE)
    return _sum_tokens(tokens)


def _sum_tokens(tokens: list[str]) -> float:
    total = 0.0
    pending: float | None = None
    pending_from_article = False
    pending_is_tens_word = False
    last_unit: float | None = None

    def next_meaningful(i: int) -> str | None:
        for tok in tokens[i + 1 :]:
            if tok not in _FILLER:
                return tok
        return None

    for i, tok in enumerate(tokens):
        if tok in _FILLER:
            continue
        if tok in _ARTICLES:
            if pending is None:
                pending, pending_from_article, pending_is_tens_word = 1.0, True, False
            continue
        if tok == "half":
            nxt = next_meaningful(i)
            if nxt in _UNITS or nxt in _ARTICLES:
                base = 0.0 if pending is None or pending_from_article else pending
                pending, pending_from_article, pending_is_tens_word = base + 0.5, False, False
            elif last_unit is not None:
                total += 0.5 * last_unit
                pending, pending_from_article = None, False
            else:
                raise TimerError(_UNPARSEABLE)
            continue
        if tok in _UNITS:
            unit = _UNITS[tok]
            if pending is None:
                raise TimerError(_UNPARSEABLE)
            total += pending * unit
            last_unit = unit
            pending, pending_from_article, pending_is_tens_word = None, False, False
            continue
        number = _WORD_NUMBERS.get(tok)
        is_word = number is not None
        if number is None:
            try:
                number = float(tok)
            except ValueError:
                raise TimerError(_UNPARSEABLE) from None
        if pending is not None and not pending_from_article:
            if pending_is_tens_word and is_word and number < 10:
                pending += number  # "twenty five"
                pending_is_tens_word = False
                continue
            raise TimerError(_UNPARSEABLE)
        pending = number
        pending_from_article = False
        pending_is_tens_word = is_word and tok in _TENS_WORDS

    if pending is not None and not pending_from_article:
        # A bare trailing number: seconds on its own, or the next smaller unit ("1 hour 30").
        unit = 1.0 if last_unit is None else _NEXT_SMALLER_UNIT.get(last_unit, 1.0)
        total += pending * unit
    elif last_unit is None:
        raise TimerError(_UNPARSEABLE)
    return total


# --------------------------------------------------------------------------- speaking durations


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def speak_duration(seconds: float) -> str:
    """``330`` -> "5 minutes and 30 seconds"; ``5400`` -> "1 hour and 30 minutes".

    Rounds up to whole seconds; drops seconds once the duration is 10 minutes or more.
    """
    total = max(0, math.ceil(seconds - 1e-9))
    if total >= 600:
        total = round(total / 60.0) * 60
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts: list[str] = []
    if hours:
        parts.append(_plural(hours, "hour"))
    if minutes:
        parts.append(_plural(minutes, "minute"))
    if secs and not hours:
        parts.append(_plural(secs, "second"))
    if not parts:
        return "less than a second"
    return " and ".join(parts) if len(parts) <= 2 else f"{parts[0]}, {parts[1]} and {parts[2]}"


# --------------------------------------------------------------------------- labels and lookup

_LABEL_JUNK = re.compile(r"[^a-z0-9 '\-]")
_LEADING_WORDS = ("the ", "my ", "a ", "an ")


def normalize_label(label: str | None) -> str | None:
    """ "The Pasta Timer" -> "pasta". Returns None for an empty label."""
    if label is None:
        return None
    text = _LABEL_JUNK.sub(" ", label.lower())
    text = " ".join(text.split())
    changed = True
    while changed:
        changed = False
        for word in _LEADING_WORDS:
            if text.startswith(word):
                text, changed = text[len(word) :], True
        if text.endswith(" timer") or text == "timer":
            text, changed = text[: -len("timer")].strip(), True
    text = text[:MAX_LABEL_CHARS].strip()
    return text or None


def _normalize_query(query: str) -> str:
    text = normalize_label(query) or ""
    for prefix in ("timer ", "number ", "no "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.lstrip("#").strip()
    if text in _WORD_NUMBERS and _WORD_NUMBERS[text] >= 1:
        text = str(int(_WORD_NUMBERS[text]))
    return text


# --------------------------------------------------------------------------- the service


class TimerService:
    """Owns the running timers. Must be used from within a running asyncio loop."""

    def __init__(
        self,
        clock: Clock | None = None,
        on_fire: OnFire | None = None,
        max_timers: int = 20,
    ) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self.on_fire: OnFire | None = on_fire
        self.max_timers = max_timers
        self._timers: dict[str, Timer] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._next_id = 1
        self._closed = False

    # ---- queries

    def list(self) -> list[Timer]:
        """Active timers, soonest first, each with ``remaining`` seconds filled in."""
        now = self._clock.monotonic()
        snapshot = [
            dataclasses.replace(t, remaining=max(0.0, t.due - now)) for t in self._timers.values()
        ]
        return sorted(snapshot, key=lambda t: t.due)

    def __len__(self) -> int:
        return len(self._timers)

    def find(self, query: str | None) -> list[Timer]:
        """Timers matching a spoken reference: "pasta", "the pasta timer", "2", "timer two".

        An empty query matches every timer (so "cancel the timer" works when there's one).
        Exact id or label wins; otherwise any timer whose label contains the query.
        """
        timers = self.list()
        text = _normalize_query(query or "")
        if not text:
            return timers
        exact = [t for t in timers if t.id == text or t.label == text]
        if exact:
            return exact
        return [t for t in timers if t.label and (text in t.label or t.label in text)]

    # ---- commands

    def set(self, duration: float | int | str, label: str | None = None) -> Timer:
        """Start a timer. ``duration`` is seconds or text like "5 minutes"."""
        if self._closed:
            raise TimerError("Timers are shutting down right now.")
        seconds = parse_duration(duration)
        if seconds < MIN_SECONDS:
            raise TimerError("Timers need to be at least one second long.")
        if seconds > MAX_SECONDS:
            raise TimerError("Timers can be at most 24 hours long.")
        if len(self._timers) >= self.max_timers:
            raise TimerError(
                f"You already have {self.max_timers} timers running. Cancel one first."
            )
        name = normalize_label(label)
        if name is not None and any(t.label == name for t in self._timers.values()):
            raise TimerError(
                f"You already have a {name} timer. Cancel it first or pick another name."
            )

        loop = asyncio.get_running_loop()
        if not self._timers:
            self._next_id = 1
        timer_id = str(self._next_id)
        self._next_id += 1
        timer = Timer(
            id=timer_id,
            label=name,
            seconds=seconds,
            due=self._clock.monotonic() + seconds,
            remaining=seconds,
        )
        self._timers[timer_id] = timer
        task = loop.create_task(self._run(timer), name=f"spark-timer-{timer_id}")
        self._tasks[timer_id] = task
        task.add_done_callback(lambda t, tid=timer_id: self._task_done(tid, t))
        log.info("timer set: id=%s label=%s seconds=%.0f", timer_id, name, seconds)
        return timer

    def cancel(self, id_or_label: str) -> Timer | None:
        """Cancel the one timer matching ``id_or_label``. None if no match or ambiguous."""
        matches = self.find(id_or_label)
        if len(matches) != 1:
            return None
        return self._cancel_id(matches[0].id)

    def cancel_all(self) -> list[Timer]:
        """Cancel every timer. Returns what was cancelled."""
        return [t for t in (self._cancel_id(tid) for tid in list(self._timers)) if t is not None]

    async def close(self) -> None:
        """Cancel all timers and wait for their tasks to finish. Idempotent."""
        self._closed = True
        tasks = list(self._tasks.values())
        self._timers.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # ---- internals

    def _cancel_id(self, timer_id: str) -> Timer | None:
        timer = self._timers.pop(timer_id, None)
        task = self._tasks.pop(timer_id, None)
        if task is not None and not task.done():
            task.cancel()
        if timer is None:
            return None
        log.info("timer cancelled: id=%s label=%s", timer.id, timer.label)
        return dataclasses.replace(timer, remaining=max(0.0, timer.due - self._clock.monotonic()))

    async def _run(self, timer: Timer) -> None:
        while True:
            left = timer.due - self._clock.monotonic()
            if left <= 0:
                break
            await self._clock.sleep(left)
        if self._timers.pop(timer.id, None) is None:
            return  # cancelled while we were waking up
        log.info("timer fired: id=%s label=%s", timer.id, timer.label)
        callback = self.on_fire
        if callback is None:
            return
        try:
            result = callback(dataclasses.replace(timer, remaining=0.0))
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("timer on_fire callback failed for timer %s", timer.id)

    def _task_done(self, timer_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(timer_id) is task:
            del self._tasks[timer_id]
        if not task.cancelled() and task.exception() is not None:
            log.error("timer task %s crashed", timer_id, exc_info=task.exception())
