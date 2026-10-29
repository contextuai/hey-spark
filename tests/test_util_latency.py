"""Tests for TurnTimer, LatencyLog, summarize, and format_breakdown."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spark.config import Settings
from spark.util import latency as L


class StepClock:
    """Tiny fake clock: time only moves when the test says so."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _full_turn(clock: StepClock, **meta: object) -> L.TurnTimer:
    t = L.TurnTimer(clock, **meta)
    t.mark(L.WAKE)
    for stage, dt in [
        (L.SPEECH_END, 1.2),
        (L.STT_DONE, 0.4),
        (L.BRAIN_FIRST_TOKEN, 0.5),
        (L.FIRST_SENTENCE, 0.3),
        (L.FIRST_AUDIO, 0.15),
        (L.TURN_DONE, 2.0),
    ]:
        clock.advance(dt)
        t.mark(stage)
    return t


def test_stage_constants() -> None:
    assert L.STAGES == (
        "wake",
        "speech_end",
        "stt_done",
        "brain_first_token",
        "first_sentence",
        "first_audio",
        "turn_done",
    )


def test_breakdown_full_turn() -> None:
    clock = StepClock()
    t = _full_turn(clock, stt="faster_whisper", route="fast")
    b = t.breakdown()
    assert b["stages"] == {
        "wake->speech_end": 1200.0,
        "speech_end->stt_done": 400.0,
        "stt_done->brain_first_token": 500.0,
        "brain_first_token->first_sentence": 300.0,
        "first_sentence->first_audio": 150.0,
        "first_audio->turn_done": 2000.0,
    }
    assert b["total_ms"] == 2550.0  # wake -> first_audio, not turn_done
    assert b["marks_ms"]["wake"] == 0.0
    assert b["marks_ms"]["turn_done"] == 4550.0
    assert b["meta"] == {"stt": "faster_whisper", "route": "fast"}
    json.dumps(b)  # JSON-serialisable


def test_first_mark_wins_unless_overwrite() -> None:
    clock = StepClock()
    t = L.TurnTimer(clock)
    first = t.mark(L.WAKE)
    clock.advance(1.0)
    assert t.mark(L.WAKE) == first
    clock.advance(1.0)
    assert t.mark(L.WAKE, overwrite=True) == first + 2.0
    assert t.has(L.WAKE)
    with pytest.raises(ValueError):
        t.mark("")


def test_missing_stages_are_skipped_and_total_uses_first_to_last() -> None:
    clock = StepClock()
    t = L.TurnTimer(clock)
    t.mark(L.STT_DONE)  # --text mode: no wake/speech_end
    clock.advance(0.6)
    t.mark(L.BRAIN_FIRST_TOKEN)
    clock.advance(0.4)
    t.mark(L.TURN_DONE)
    b = t.breakdown()
    assert b["stages"] == {
        "stt_done->brain_first_token": 600.0,
        "brain_first_token->turn_done": 400.0,
    }
    assert b["total_ms"] == 1000.0
    assert t.elapsed_ms(L.STT_DONE, L.TURN_DONE) == 1000.0
    assert t.elapsed_ms(L.WAKE, L.TURN_DONE) is None


def test_empty_timer() -> None:
    b = L.TurnTimer(StepClock()).breakdown()
    assert b["stages"] == {}
    assert b["total_ms"] is None
    assert L.format_breakdown(b).startswith("latency n/a")


def test_custom_marks_recorded_but_not_in_stages() -> None:
    clock = StepClock()
    t = L.TurnTimer(clock)
    t.mark(L.WAKE)
    clock.advance(0.1)
    t.mark("tool_call")
    clock.advance(0.1)
    t.mark(L.FIRST_AUDIO)
    b = t.breakdown()
    assert b["stages"] == {"wake->first_audio": 200.0}
    assert b["marks_ms"]["tool_call"] == 100.0


def test_metadata_rejects_raw_audio() -> None:
    t = L.TurnTimer(StepClock())
    t.annotate(text_len=12, backends={"tts": "piper"}, tags=["a", "b"])
    assert t.meta["backends"] == {"tts": "piper"}
    with pytest.raises(TypeError):
        t.annotate(audio=np.zeros(16, dtype=np.int16))
    with pytest.raises(TypeError):
        t.annotate(pcm=b"\x00\x01")


def test_default_clock_is_real() -> None:
    t = L.TurnTimer()
    t.mark(L.WAKE)
    t.mark(L.FIRST_AUDIO)
    assert t.breakdown()["total_ms"] >= 0.0


def test_format_breakdown_one_line() -> None:
    line = L.format_breakdown(_full_turn(StepClock(), route="fast").breakdown())
    assert "\n" not in line
    assert line.startswith("latency 2550ms | wake->speech_end 1200")
    assert "first_sentence->first_audio 150" in line
    assert line.endswith("route=fast")


# --- LatencyLog ----------------------------------------------------------------------------


def _settings(tmp_path: Path, **latency: object) -> Settings:
    return Settings(config_dir=tmp_path, logging={"dir": "logs"}, latency=latency)


def test_latency_log_appends_jsonl(tmp_path: Path) -> None:
    log = L.LatencyLog.from_settings(_settings(tmp_path))
    assert log.path == tmp_path / "logs" / "latency.jsonl"
    assert not log.path.parent.exists()  # created lazily
    clock = StepClock()
    log.append(_full_turn(clock, route="fast"), ts=1.0)
    log.append(_full_turn(clock, route="deep").breakdown(), ts=2.0)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    recs = [json.loads(x) for x in lines]
    assert [r["ts"] for r in recs] == [1.0, 2.0]
    assert recs[1]["meta"]["route"] == "deep"
    assert log.read() == recs


def test_latency_log_disabled_writes_nothing(tmp_path: Path) -> None:
    log = L.LatencyLog.from_settings(_settings(tmp_path, enabled=False))
    log.append(_full_turn(StepClock()))
    assert not log.path.exists()
    assert not (tmp_path / "logs").exists()
    assert log.read() == []


def test_latency_log_read_skips_garbage(tmp_path: Path) -> None:
    path = tmp_path / "lat.jsonl"
    path.write_text('{"total_ms": 1}\nnot json\n\n[1, 2]\n{"total_ms": 2}\n', encoding="utf-8")
    assert L.LatencyLog(path).read() == [{"total_ms": 1}, {"total_ms": 2}]


# --- summarize -----------------------------------------------------------------------------


def test_percentile_matches_numpy() -> None:
    data = [5.0, 1.0, 9.0, 3.0, 7.0, 2.5]
    for q in (0, 10, 50, 90, 100):
        assert L.percentile(data, q) == pytest.approx(float(np.percentile(data, q)))
    with pytest.raises(ValueError):
        L.percentile([], 50)
    with pytest.raises(ValueError):
        L.percentile([1.0], 101)


def test_summarize_median_p90_per_stage(tmp_path: Path) -> None:
    log = L.LatencyLog(tmp_path / "lat.jsonl")
    for i in range(10):
        clock = StepClock()
        t = L.TurnTimer(clock)
        t.mark(L.WAKE)
        clock.advance(1.0 + i * 0.1)  # 1000..1900 ms
        t.mark(L.SPEECH_END)
        clock.advance(0.5)
        t.mark(L.FIRST_AUDIO)
        log.append(t)
    summary = L.summarize(log.read())
    assert list(summary) == ["wake->speech_end", "speech_end->first_audio", "total_ms"]
    ws = summary["wake->speech_end"]
    assert ws["n"] == 10
    assert ws["median"] == pytest.approx(1450.0)
    assert ws["p90"] == pytest.approx(1810.0)
    assert ws["min"] == 1000.0 and ws["max"] == 1900.0
    assert summary["speech_end->first_audio"]["median"] == 500.0
    assert summary["total_ms"]["median"] == pytest.approx(1950.0)


def test_summarize_ignores_missing_values() -> None:
    recs = [{"stages": {"wake->speech_end": 100}}, {"total_ms": None}, {}]
    assert L.summarize(recs) == {
        "wake->speech_end": {"n": 1.0, "median": 100.0, "p90": 100.0, "min": 100.0, "max": 100.0}
    }
    assert L.summarize([]) == {}


def test_summarize_skips_interrupted_turns():
    done = {"stages": {"wake->speech_end": 100.0}, "total_ms": 900.0, "meta": {}}
    cut = {
        "stages": {"wake->speech_end": 5000.0},
        "total_ms": 9000.0,
        "meta": {"interrupted": True},
    }
    out = L.summarize([done, cut])
    assert out["total_ms"]["n"] == 1 and out["total_ms"]["median"] == 900.0
