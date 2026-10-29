"""Per-turn latency tracking (spec §9).

A :class:`TurnTimer` records when each pipeline stage first happened in a turn::

    wake → speech_end → stt_done → brain_first_token → first_sentence → first_audio → turn_done

``breakdown()`` turns the marks into millisecond deltas between consecutive stages present,
plus ``total_ms`` (wake → first audio when both exist). :class:`LatencyLog` appends one JSON
line per turn to ``settings.log_path / settings.latency.file``; :func:`summarize` reduces many
records to median / p90 per stage for ``spark bench latency``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings
    from spark.util.clock import Clock

__all__ = [
    "BRAIN_FIRST_TOKEN",
    "FIRST_AUDIO",
    "FIRST_SENTENCE",
    "SPEECH_END",
    "STAGES",
    "STT_DONE",
    "TURN_DONE",
    "WAKE",
    "LatencyLog",
    "TurnTimer",
    "format_breakdown",
    "percentile",
    "summarize",
]

WAKE = "wake"
SPEECH_END = "speech_end"
STT_DONE = "stt_done"
BRAIN_FIRST_TOKEN = "brain_first_token"
FIRST_SENTENCE = "first_sentence"
FIRST_AUDIO = "first_audio"
TURN_DONE = "turn_done"

STAGES: tuple[str, ...] = (
    WAKE,
    SPEECH_END,
    STT_DONE,
    BRAIN_FIRST_TOKEN,
    FIRST_SENTENCE,
    FIRST_AUDIO,
    TURN_DONE,
)

TOTAL_KEY = "total_ms"
_SCALARS = (str, int, float, bool, type(None))


class _MonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


def _check_meta_value(key: str, value: Any) -> Any:
    """Metadata must be small JSON values (never raw audio / bytes / arrays)."""
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, list | tuple):
        return [_check_meta_value(key, v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _check_meta_value(key, v) for k, v in value.items()}
    raise TypeError(
        f"latency metadata {key!r} must be a JSON scalar/list/dict, got {type(value).__name__}"
    )


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


class TurnTimer:
    """Marks pipeline stages for one turn using an injectable monotonic clock."""

    def __init__(self, clock: Clock | None = None, **meta: Any) -> None:
        self._clock = clock if clock is not None else _MonotonicClock()
        self._marks: dict[str, float] = {}
        self.meta: dict[str, Any] = {}
        self.annotate(**meta)

    def mark(self, name: str, *, overwrite: bool = False) -> float:
        """Record ``name`` now. Repeats are ignored unless ``overwrite``; returns the kept time."""
        if not name:
            raise ValueError("stage name must be non-empty")
        if name in self._marks and not overwrite:
            return self._marks[name]
        self._marks[name] = self._clock.monotonic()
        return self._marks[name]

    def has(self, name: str) -> bool:
        return name in self._marks

    @property
    def marks(self) -> dict[str, float]:
        """Raw monotonic timestamps, in the order they were first recorded."""
        return dict(self._marks)

    def annotate(self, **meta: Any) -> None:
        """Attach metadata (backends, route, text lengths ...). Values must be JSON-able."""
        for key, value in meta.items():
            self.meta[key] = _check_meta_value(key, value)

    def elapsed_ms(self, start: str, end: str) -> float | None:
        """Milliseconds between two marks, or None if either is missing."""
        if start not in self._marks or end not in self._marks:
            return None
        return _ms(self._marks[end] - self._marks[start])

    def breakdown(self) -> dict[str, Any]:
        """Deltas between consecutive standard stages present, total, offsets, metadata.

        Returns ``{"stages": {"wake->speech_end": ms, ...}, "total_ms": ms | None,
        "marks_ms": {name: ms since first mark}, "meta": {...}}``.
        """
        present = [s for s in STAGES if s in self._marks]
        stages: dict[str, float] = {}
        for prev, cur in pairwise(present):
            stages[f"{prev}->{cur}"] = _ms(self._marks[cur] - self._marks[prev])

        total: float | None = None
        if self._marks:
            ordered = sorted(self._marks.values())
            start = self._marks.get(WAKE, ordered[0])
            end = self._marks.get(FIRST_AUDIO, ordered[-1])
            total = _ms(end - start)

        origin = min(self._marks.values()) if self._marks else 0.0
        marks_ms = {name: _ms(t - origin) for name, t in self._marks.items()}
        return {
            "stages": stages,
            TOTAL_KEY: total,
            "marks_ms": marks_ms,
            "meta": dict(self.meta),
        }


def format_breakdown(breakdown: Mapping[str, Any]) -> str:
    """One console line, e.g. ``latency 1450ms | wake->speech_end 900 · ... | route=fast``."""
    total = breakdown.get(TOTAL_KEY)
    head = "latency " + (f"{total:.0f}ms" if total is not None else "n/a")
    stages = breakdown.get("stages") or {}
    parts = [head]
    if stages:
        parts.append(" · ".join(f"{k} {v:.0f}" for k, v in stages.items()))
    meta = breakdown.get("meta") or {}
    if meta:
        parts.append(" ".join(f"{k}={v}" for k, v in meta.items()))
    return " | ".join(parts)


class LatencyLog:
    """Appends one JSON object per turn to a ``.jsonl`` file (lazily creates the directory)."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    @classmethod
    def from_settings(cls, settings: Settings) -> LatencyLog:
        return cls(settings.log_path / settings.latency.file, enabled=settings.latency.enabled)

    def append(self, turn: TurnTimer | Mapping[str, Any], *, ts: float | None = None) -> None:
        """Write one record (a timer's breakdown, or an already-built breakdown dict)."""
        if not self.enabled:
            return
        record = dict(turn.breakdown() if isinstance(turn, TurnTimer) else turn)
        record.setdefault("ts", time.time() if ts is None else ts)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        """All well-formed records in the file (malformed lines are skipped)."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), ``q`` in [0, 100]."""
    data = sorted(values)
    if not data:
        raise ValueError("percentile of empty data")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    pos = (len(data) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Median and p90 (plus n/min/max) per stage delta and for ``total_ms``.

    Stages appear in canonical pipeline order, then ``total_ms`` last. Turns cut short by
    barge-in (``meta.interrupted``) are skipped: they never reached a complete reply.
    """
    samples: dict[str, list[float]] = {}
    for rec in records:
        if (rec.get("meta") or {}).get("interrupted"):
            continue
        for key, value in (rec.get("stages") or {}).items():
            if isinstance(value, int | float):
                samples.setdefault(key, []).append(float(value))
        total = rec.get(TOTAL_KEY)
        if isinstance(total, int | float):
            samples.setdefault(TOTAL_KEY, []).append(float(total))

    def order(key: str) -> tuple[int, int, str]:
        if key == TOTAL_KEY:
            return (len(STAGES) + 1, 0, key)
        first, _, second = key.partition("->")
        rank = STAGES.index(first) if first in STAGES else len(STAGES)
        rank2 = STAGES.index(second) if second in STAGES else len(STAGES)
        return (rank, rank2, key)

    return {
        key: {
            "n": float(len(vals)),
            "median": round(percentile(vals, 50), 1),
            "p90": round(percentile(vals, 90), 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
        }
        for key, vals in sorted(samples.items(), key=lambda kv: order(kv[0]))
    }
