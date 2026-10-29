"""State-machine tests for spark.app with fakes (spec §3, §6)."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest

from spark import registry
from spark.app import TIMER_EARCON, SparkApp, State, strip_wake_phrase
from spark.config import Settings
from spark.tools.status import describe_status
from spark.tools.timers import TimerService
from tests.fakes import (
    FakeClock,
    FakeMic,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWakeWord,
    MockBrain,
    concat,
    make_settings,
    silence,
    tone,
)

BLOCK = 512


def _settings(tmp_path, **overrides):
    overrides.setdefault("logging", {"dir": str(tmp_path / "logs")})
    overrides.setdefault("debug_audio", {"enabled": False, "dir": str(tmp_path / "dbg")})
    overrides.setdefault("attention", {"awake_s": 8.0})  # short awake window: fast tests
    return make_settings(**overrides)


def _app(tmp_path, *, stt=None, brain=None, tts=None, settings=None, **kw):
    settings = settings or _settings(tmp_path)
    player = FakePlayer()
    return SparkApp(
        settings,
        mic=FakeMic(),
        player=player,
        wakeword=FakeWakeWord(),
        vad=FakeVAD(),
        stt=stt or FakeSTT(transcripts=["what is two plus two"]),
        tts=tts or FakeTTS(),
        brain=brain or MockBrain(),
        **kw,
    )


async def _feed(app: SparkApp, pcm: np.ndarray, block: int = BLOCK) -> None:
    for i in range(0, len(pcm), block):
        await app.feed(pcm[i : i + block])


async def _wake(app: SparkApp) -> None:
    app.wakeword.trigger()
    await _feed(app, silence(BLOCK / 16000))


async def _utterance(app: SparkApp, speech_s: float = 0.6, tail_s: float = 0.8) -> None:
    await _feed(app, concat(tone(speech_s), silence(tail_s)))


async def _until(app: SparkApp, state: State) -> None:
    if app.state is state:
        return
    reached = asyncio.Event()
    app.on_state = lambda _old, new: reached.set() if new is state else None
    try:
        await asyncio.wait_for(reached.wait(), 5.0)
    finally:
        app.on_state = None


async def _finish_turn(app: SparkApp) -> None:
    assert app.turn_task is not None
    await asyncio.wait_for(app.turn_task, 5)


async def test_full_cycle_idle_to_follow_up_to_idle(tmp_path):
    app = _app(tmp_path)
    assert app.state is State.IDLE
    await _wake(app)
    assert app.state is State.LISTENING
    assert app.player.earcons == ["wake"]
    await _utterance(app)
    assert app.state is State.THINKING
    await _finish_turn(app)
    assert app.state is State.FOLLOW_UP
    assert app.brain.calls == ["what is two plus two"]
    assert app.tts.spoken == ["Two plus two is four."]
    assert app.player.earcons == ["wake", "thinking"]
    assert app.state_history == [
        State.IDLE,
        State.LISTENING,
        State.THINKING,
        State.SPEAKING,
        State.FOLLOW_UP,
    ]
    await _feed(app, silence(7.5))
    assert app.state is State.FOLLOW_UP
    await _feed(app, silence(0.6))
    assert app.state is State.IDLE


async def test_follow_up_question_needs_no_wake_word(tmp_path):
    stt = FakeSTT(transcripts=["what is two plus two", "and what about three"])
    app = _app(tmp_path, stt=stt)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.FOLLOW_UP
    await _feed(app, silence(2.0))
    await _utterance(app)
    assert app.state is State.THINKING
    await _finish_turn(app)
    assert app.brain.calls == ["what is two plus two", "and what about three"]
    assert app.wakeword.detections == [0]
    assert app.last_timer is not None
    assert app.last_timer.meta["mode"] == "follow_up"


async def test_no_speech_after_wake_returns_to_idle(tmp_path):
    app = _app(tmp_path)
    await _wake(app)
    await _feed(app, silence(29.9), block=16000)
    assert app.state is State.LISTENING  # 30 s to think after the wake word
    await _feed(app, silence(0.2))
    assert app.state is State.IDLE
    assert app.stt.received == []
    assert app.brain.calls == []


async def test_a_long_pause_to_think_after_the_wake_word_still_gets_the_question(tmp_path):
    app = _app(tmp_path)
    await _wake(app)
    await _feed(app, silence(25.0), block=16000)
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == ["what is two plus two"]
    # the 25 s of thinking time is not sent to speech-to-text (only the pre-roll is kept)
    assert app.stt.received[0] < 16000 * 3


async def test_empty_transcript_returns_to_idle_without_brain(tmp_path):
    app = _app(tmp_path, stt=FakeSTT(transcripts=[""]))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.IDLE
    assert app.brain.calls == []


async def test_wake_phrase_in_transcript_is_stripped(tmp_path):
    app = _app(tmp_path, stt=FakeSTT(transcripts=["Hey Spark, what is two plus two"]))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == ["what is two plus two"]


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Hey Spark, what's the time?", "what's the time?"),
        ("hey spark what's the time", "what's the time"),
        ("Spark. New topic", "New topic"),
        ("sparkling water is nice", "sparkling water is nice"),
        ("", ""),
    ],
)
def test_strip_wake_phrase(raw, clean):
    assert strip_wake_phrase(raw) == clean


async def test_pre_roll_is_sent_to_stt(tmp_path):
    app = _app(tmp_path)
    await _feed(app, tone(0.5, freq=300))  # audio before the wake word, kept in the ring
    await _wake(app)
    await _utterance(app, speech_s=0.6, tail_s=0.8)
    await _finish_turn(app)
    pre_roll = int(16000 * app.settings.recorder.pre_roll_ms / 1000)
    end_silence = app.settings.recorder.end_silence_ms / 1000
    # pre-roll + 0.6 s speech + end-of-speech silence (the rest of the tail is not needed)
    assert app.stt.received[0] >= pre_roll + int(16000 * (0.6 + end_silence)) - BLOCK


async def test_stt_error_plays_error_earcon_and_goes_idle(tmp_path):
    app = _app(tmp_path, stt=FakeSTT(fail_with=RuntimeError("boom")))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.IDLE
    assert app.player.earcons[-1] == "error"
    assert isinstance(app.errors[0], RuntimeError)


async def test_brain_error_plays_error_earcon_and_goes_idle(tmp_path):
    app = _app(tmp_path, brain=MockBrain(fail_with=ConnectionError("offline")))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.IDLE
    assert app.player.earcons[-1] == "error"
    # a later wake still works
    app.brain.fail_with = None
    await _wake(app)
    assert app.state is State.LISTENING


async def test_reply_is_sanitized_before_tts(tmp_path):
    brain = MockBrain(replies=["**Tokyo** is the capital. Details at https://example.com/x ok."])
    app = _app(tmp_path, brain=brain)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    spoken = " ".join(app.tts.spoken)
    assert "Tokyo is the capital." in spoken
    assert "*" not in spoken
    assert "http" not in spoken


async def test_latency_breakdown_logged(tmp_path):
    app = _app(tmp_path)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    lines = (tmp_path / "logs" / "latency.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    for stage in ("wake", "speech_end", "stt_done", "brain_first_token", "first_audio"):
        assert stage in record["marks_ms"], stage
    assert record["meta"]["brain"] == "mock"


async def test_transcripts_logged_and_can_be_disabled(tmp_path):
    app = _app(tmp_path)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    app.transcripts.close()
    text = (tmp_path / "logs" / "transcripts.log").read_text(encoding="utf-8")
    assert "what is two plus two" in text and "Two plus two is four." in text

    off = tmp_path / "off"
    settings = _settings(off, logging={"dir": str(off / "logs"), "transcripts": False})
    app2 = _app(off, settings=settings)
    await _wake(app2)
    await _utterance(app2)
    await _finish_turn(app2)
    assert not (off / "logs" / "transcripts.log").exists()


async def test_debug_audio_saved_only_when_enabled(tmp_path):
    app = _app(tmp_path)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert not (tmp_path / "dbg").exists()

    settings = _settings(tmp_path, debug_audio={"enabled": True, "dir": str(tmp_path / "dbg")})
    app2 = _app(tmp_path, settings=settings)
    await _wake(app2)
    await _utterance(app2)
    await _finish_turn(app2)
    assert (tmp_path / "dbg" / "last_utterance.wav").stat().st_size > 1000


async def test_wake_word_ignored_while_speaking_when_barge_in_off(tmp_path):
    brain = MockBrain(replies=["One. Two. Three. Four."])
    app = _app(tmp_path, brain=brain, tts=FakeTTS(per_chunk_s=0.05))
    await _wake(app)
    await _utterance(app)
    await _until(app, State.SPEAKING)
    await _wake(app)
    assert app.state is State.SPEAKING
    await _finish_turn(app)
    assert app.tts.spoken == ["One.", "Two.", "Three.", "Four."]


async def test_barge_in_stops_speech_fast_and_listens(tmp_path):
    settings = _settings(tmp_path, app={"barge_in": True})
    brain = MockBrain(replies=["One. Two. Three. Four."])
    tts = FakeTTS(per_chunk_s=2.0)
    app = _app(tmp_path, settings=settings, brain=brain, tts=tts)
    await _wake(app)
    await _utterance(app)
    await _until(app, State.SPEAKING)
    t0 = time.perf_counter()
    await _wake(app)
    elapsed = time.perf_counter() - t0
    assert app.state is State.LISTENING
    assert elapsed < 0.3
    assert tts.stopped
    assert app.player.stops >= 1
    assert app.turn_task is None
    assert app.stop_latency_ms is not None and app.stop_latency_ms < 300
    await app.wait_interrupted()  # brain.interrupt() runs off the mic loop
    assert brain.interrupts >= 1
    assert len(tts.spoken) <= 1  # the rest of the reply was dropped


async def test_text_turn(tmp_path):
    app = _app(tmp_path)
    reply = await app.text_turn("what is two plus two")
    assert reply == "Two plus two is four."
    assert app.state is State.IDLE
    assert await app.text_turn("   ") == ""


async def test_run_consumes_mic_until_end(tmp_path):
    audio = concat(silence(0.1), tone(0.6), silence(0.8))
    settings = _settings(tmp_path)
    app = SparkApp(
        settings,
        mic=FakeMic(audio),
        player=FakePlayer(),
        wakeword=FakeWakeWord(fire_at={1}),
        vad=FakeVAD(),
        stt=FakeSTT(transcripts=["what is two plus two"]),
        tts=FakeTTS(),
        brain=MockBrain(),
    )
    await app.run()
    await _finish_turn(app)
    assert app.mic.started and app.mic.stopped
    assert app.brain.calls == ["what is two plus two"]


async def test_openwakeword_frame_size_is_rechunked(tmp_path):
    app = _app(tmp_path)
    app.wakeword = FakeWakeWord(frame_length=1280, fire_at={2})
    from spark.audio.stream import Rechunker

    app._wake_chunker = Rechunker(1280)
    await _feed(app, silence(0.5), block=480)  # odd mic block size
    assert app.state is State.LISTENING


async def test_from_settings_builds_configured_backends(tmp_path):
    registry.register("audio_in", "fake", FakeMic)
    registry.register("audio_out", "fake", FakePlayer)
    registry.register("wakeword", "fake", FakeWakeWord)
    registry.register("vad", "fake", FakeVAD)
    registry.register("stt", "fake", FakeSTT)
    registry.register("tts", "fake", FakeTTS)
    try:
        settings = _settings(
            tmp_path,
            audio={"input_backend": "fake", "output_backend": "fake"},
            wakeword={"backend": "fake"},
            vad={"backend": "fake"},
            stt={"backend": "fake"},
            tts={"backend": "fake"},
        )
        app = SparkApp.from_settings(settings)
        assert isinstance(app.mic, FakeMic)
        assert isinstance(app.stt, FakeSTT)
        assert isinstance(app.tts, FakeTTS)
        assert app.tts.player is app.player
        assert isinstance(app.brain, MockBrain)
        await app.warmup()
        assert app.stt.warmups == 1 and app.tts.warmups == 1
        await app.close()

        text_app = SparkApp.from_settings(settings, audio=False)
        assert text_app.mic is None and text_app.player is None and text_app.stt is None
    finally:
        for kind in ("audio_in", "audio_out", "wakeword", "vad", "stt", "tts"):
            registry.unregister(kind, "fake")


async def test_fenced_code_is_never_spoken(tmp_path):
    reply = "Sure, run this:\n```bash\nuv sync && uv run python -m spark --text\n```\nThat's it."
    app = _app(tmp_path, brain=MockBrain(replies=[reply]))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    spoken = " ".join(app.tts.spoken)
    assert "uv sync" not in spoken and "python" not in spoken and "`" not in spoken
    assert "Sure, run this" in spoken and "That's it." in spoken


async def test_follow_up_keeps_leading_spark_word(tmp_path):
    stt = FakeSTT(transcripts=["what is two plus two", "Spark design is hard"])
    app = _app(tmp_path, stt=stt)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls[-1] == "Spark design is hard"


# --- timers and status (P2.4) -------------------------------------------------------------------


async def _settle(predicate, limit_s: float = 5.0) -> None:
    """Yield to the loop until ``predicate()`` holds (timers on a FakeClock fire within ticks)."""
    deadline = time.perf_counter() + limit_s
    while not predicate():
        if time.perf_counter() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def _timer_app(tmp_path, **kw):
    timers = TimerService(clock=FakeClock())
    app = _app(tmp_path, timers=timers, **kw)
    return app, timers


async def test_timer_fired_while_idle_is_announced_at_once(tmp_path):
    app, timers = _timer_app(tmp_path)
    try:
        timers.set("5 minutes", "pasta")
        await _settle(lambda: app.announcements)
        await _settle(lambda: app.tts.spoken)
        assert app.state is State.IDLE
        assert app.player.earcons == [TIMER_EARCON]
        assert app.tts.spoken == ["Your pasta timer is done."]
        assert app.announcements == ["Your pasta timer is done."]
    finally:
        await app.close()


async def test_timer_fired_while_speaking_waits_for_follow_up(tmp_path):
    brain = MockBrain(replies=["One. Two. Three."])
    app, timers = _timer_app(tmp_path, brain=brain, tts=FakeTTS(per_chunk_s=0.05))
    try:
        await _wake(app)
        await _utterance(app)
        await _until(app, State.SPEAKING)
        timers.set(90)  # unnamed: "timer 1"
        await _settle(lambda: app._pending_announcements)  # fired, but Spark is busy
        assert app.state is State.SPEAKING
        assert app.announcements == [] and "Your timer 1 is done." not in app.tts.spoken
        await _finish_turn(app)
        assert app.state is State.FOLLOW_UP
        await _settle(lambda: "Your timer 1 is done." in app.tts.spoken)
        # the reply was never cut: the announcement came after it, with its earcon
        assert app.tts.spoken == ["One.", "Two.", "Three.", "Your timer 1 is done."]
        assert app.player.earcons == ["wake", "thinking", TIMER_EARCON]
        await _settle(lambda: app._announce_task is not None and app._announce_task.done())
        # the follow-up window restarts after Spark has spoken
        assert app.state is State.FOLLOW_UP and app.recorder is not None
        assert not app.recorder.done
    finally:
        await app.close()


async def test_timer_fired_during_text_turn_is_announced_after_it(tmp_path):
    brain = MockBrain(replies=["Sure."], delay_s=0.05)
    app, timers = _timer_app(tmp_path, brain=brain)
    try:
        turn = asyncio.create_task(app.text_turn("set a timer"))
        await _until(app, State.THINKING)
        timers.set("1 second", "tea")
        await _settle(lambda: app._pending_announcements)
        assert app.tts.spoken == []
        assert await turn == "Sure."
        await _settle(lambda: len(app.tts.spoken) == 2)
        assert app.tts.spoken == ["Sure.", "Your tea timer is done."]
    finally:
        await app.close()


async def test_on_announce_callback_gets_the_text(tmp_path):
    printed: list[str] = []
    app, timers = _timer_app(tmp_path, on_announce=printed.append)
    try:
        timers.set(10, "eggs")
        await _settle(lambda: printed)
        assert printed == ["Your eggs timer is done."]
    finally:
        await app.close()


async def test_close_cancels_running_timers(tmp_path):
    timers = TimerService()  # real clock: would take an hour to fire
    app = _app(tmp_path, timers=timers)
    timers.set("1 hour", "roast")
    assert len(timers) == 1
    await app.close()
    assert len(timers) == 0
    assert app.announcements == []


async def test_status_snapshot_has_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_GITHUB_PAT", "ghp_16C7e42F292c6912E7710c838347Ae178B4a")
    app = _app(tmp_path)
    app.brain.connectors = ["spark", "github"]  # type: ignore[attr-defined]
    app.brain.model = "claude-haiku-4-5-20251001"  # type: ignore[attr-defined]
    snap = app.status_snapshot()
    assert snap["state"] == "idle"
    assert snap["backends"] == {
        "wakeword": "openwakeword",
        "stt": "faster_whisper",
        "tts": "kokoro",
        "brain": "mock",
    }
    assert snap["connectors"] == ["github"]  # the in-process spark server isn't a connector
    assert snap["model"] == "claude-haiku-4-5-20251001"
    assert snap["uptime_s"] >= 0
    assert "ghp_" not in json.dumps(snap)
    text = describe_status(snap)
    assert "idle" in text and "github" in text


async def test_from_settings_gives_the_brain_timers_and_status(tmp_path):
    settings = _settings(tmp_path, tts={"backend": "print"})
    app = SparkApp.from_settings(settings, audio=False)
    try:
        assert isinstance(app.brain, MockBrain)
        assert app.brain.deps["timers"] is app.timers
        assert app.timers is not None and app.timers.on_fire == app._on_timer_fired
        status = app.brain.deps["status_provider"]()
        assert status["state"] == "idle" and "stt" not in status["backends"]  # text mode
    finally:
        await app.close()


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Boomi, what's the capital of Japan?", "what's the capital of Japan?"),
        ("Hey Boomy, set a timer.", "set a timer."),
        ("Hey Spark.", ""),
        ("Hey Spark, hi", "hi"),
        ("Boomerang facts, please", "Boomerang facts, please"),
    ],
)
def test_strip_wake_phrase_with_a_custom_name(raw, clean):
    names = Settings(app={"name": "Hey Spark", "name_aliases": ["Boomi", "Boomy"]}).app.wake_names
    assert strip_wake_phrase(raw, names) == clean


# --- P5.1: a failed turn is explained out loud ------------------------------------------------


async def test_failed_turn_speaks_a_plain_sentence_after_the_beep(tmp_path):
    from spark.app import failure_line
    from spark.brain.claude_agent import BrainError

    err = BrainError("Claude returned an error (error_max_turns): Reached maximum number of turns")
    app = _app(tmp_path, brain=MockBrain(fail_with=err))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.player.earcons[-1] == "error"
    line = failure_line(err)
    await _settle(lambda: line in app.tts.spoken)
    assert "more steps" in line
    assert app.announcements == [line]
    assert app.tts.spoken[-1] == line
    assert "error_max_turns" not in " ".join(app.tts.spoken)


def test_failure_lines_never_carry_the_error_text():
    from spark.app import failure_line
    from spark.interfaces import BackendUnavailable

    secret = "C:/Users/x/secret-path tok_1234567890abcdef"
    assert secret not in failure_line(RuntimeError(secret))
    assert "too long" in failure_line(TimeoutError())
    assert "API key" in failure_line(RuntimeError("401 Unauthorized"))
    assert "isn't working" in failure_line(BackendUnavailable("stt", "missing", "install it"))


# --- P5.2: awake window ----------------------------------------------------------------------


def _awake_app(tmp_path, transcripts, **kw):
    settings = _settings(tmp_path, attention={"awake_s": 180.0})
    return _app(tmp_path, stt=FakeSTT(transcripts=transcripts), settings=settings, **kw)


async def test_spark_stays_awake_for_three_minutes_then_sleeps(tmp_path):
    slept: list[str] = []
    app = _awake_app(tmp_path, ["what is two plus two", "and what about three"])
    app.on_sleep = slept.append
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.FOLLOW_UP
    assert app.awake_remaining() == pytest.approx(180.0, abs=1.0)
    await _feed(app, silence(120.0), block=16000)
    assert app.state is State.FOLLOW_UP  # two minutes later: no wake word needed
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == ["what is two plus two", "and what about three"]
    await _feed(app, silence(170.0), block=16000)
    assert app.state is State.FOLLOW_UP  # the second request restarted the window
    await _feed(app, silence(11.0), block=16000)
    assert app.state is State.IDLE
    assert app.player.earcons[-1] == "sleep"
    assert slept == ["awake window over"]
    assert app.awake_remaining() is None


async def test_go_to_sleep_while_asleep_is_quiet(tmp_path):
    slept: list[str] = []
    app = _awake_app(tmp_path, ["hi"])
    app.on_sleep = slept.append
    app.go_to_sleep("command")
    assert app.state is State.IDLE
    assert slept == [] and "sleep" not in app.player.earcons


async def test_a_failed_turn_leaves_spark_asleep(tmp_path):
    app = _awake_app(tmp_path, ["hi"], brain=MockBrain(fail_with=ConnectionError("offline")))
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.IDLE
    assert app.awake_remaining() is None


async def test_sleep_earcon_can_be_turned_off(tmp_path):
    settings = _settings(tmp_path, attention={"awake_s": 2.0, "sleep_earcon": ""})
    app = _app(tmp_path, settings=settings)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    await _feed(app, silence(3.0), block=16000)
    assert app.state is State.IDLE
    assert "sleep" not in app.player.earcons


def test_sleep_chime_is_short_and_quieter_than_the_wake_chime():
    from spark.audio.earcons import sleep_chime, wake_chime

    sleep, wake = sleep_chime(), wake_chime()
    assert 0.2 < sleep.size / 16000 < 0.4
    assert np.abs(sleep).max() < np.abs(wake).max()


# --- P5.3: local voice commands ----------------------------------------------------------------


class _ResetBrain(MockBrain):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.resets = 0

    async def reset(self) -> None:
        self.resets += 1


async def _awake_turn(app):
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.FOLLOW_UP


async def test_go_to_sleep_command_answers_locally_and_sleeps(tmp_path):
    slept: list[str] = []
    app = _awake_app(tmp_path, ["what is two plus two", "Spark, go to sleep."])
    app.on_sleep = slept.append
    await _awake_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == ["what is two plus two"]  # the command never reached Claude
    assert app.tts.spoken[-1] == "Okay."
    assert app.state is State.IDLE and app.awake_remaining() is None
    assert app.player.earcons[-1] == "sleep"
    assert slept == ["command"]
    assert [c.kind for c in app.commands] == ["sleep"]


async def test_never_mind_sleeps_without_a_word(tmp_path):
    app = _awake_app(tmp_path, ["what is two plus two", "never mind"])
    await _awake_turn(app)
    spoken = list(app.tts.spoken)
    await _utterance(app)
    await _finish_turn(app)
    assert app.tts.spoken == spoken
    assert app.state is State.IDLE
    assert app.player.earcons[-1] == "sleep"


async def test_new_conversation_resets_the_brain_and_stays_awake(tmp_path):
    brain = _ResetBrain()
    app = _awake_app(
        tmp_path, ["what is two plus two", "new conversation", "what time is it"], brain=brain
    )
    await _awake_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    await _settle(lambda: brain.resets == 1)
    assert app.tts.spoken[-1] == "Okay, fresh start."
    assert app.state is State.FOLLOW_UP  # still awake: the next request needs no wake word
    await _utterance(app)
    await _finish_turn(app)
    assert brain.calls == ["what is two plus two", "what time is it"]


async def test_a_command_from_asleep_works_too(tmp_path):
    app = _awake_app(tmp_path, ["go to sleep"])
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == []
    assert app.tts.spoken == ["Okay."]
    assert app.state is State.IDLE


async def test_text_mode_commands(tmp_path):
    brain = _ResetBrain()
    app = _app(tmp_path, brain=brain)
    assert await app.text_turn("new topic") == "Okay, fresh start."
    assert brain.resets == 1
    assert await app.text_turn("never mind") == ""
    assert await app.text_turn("what is two plus two") == "Two plus two is four."
    assert brain.calls == ["what is two plus two"]


# --- P5.4: "is this for me?" while awake --------------------------------------------------------


async def _live(app, seconds: float) -> None:
    """Feed silence in 1 s blocks while the app's FakeClock moves with it (real-time audio)."""
    for _ in range(int(seconds)):
        app.clock.now += 1.0
        await _feed(app, silence(1.0), block=16000)


async def test_chatter_while_awake_is_ignored_and_does_not_extend_the_window(tmp_path):
    ignored: list[str] = []
    app = _awake_app(
        tmp_path,
        ["what is two plus two", "I told him the meeting moved", "and what about three"],
        clock=FakeClock(),
    )
    app.on_ignored = lambda text, verdict: ignored.append(verdict.reason)
    await _awake_turn(app)
    earcons = list(app.player.earcons)
    await _live(app, 100)
    await _utterance(app)  # a colleague talking
    await _finish_turn(app)
    assert app.state is State.FOLLOW_UP
    assert app.brain.calls == ["what is two plus two"]
    assert ignored == ["no request"]
    assert app.player.earcons == earcons  # no thinking tick for speech that isn't for Spark
    assert app.awake_remaining() == pytest.approx(80.0, abs=1.0)  # not restarted
    await _live(app, 75)
    assert app.state is State.FOLLOW_UP
    await _live(app, 10)
    assert app.state is State.IDLE  # asleep on the original schedule


async def test_after_the_wake_word_everything_is_addressed(tmp_path):
    app = _awake_app(tmp_path, ["lunch was good today"])
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == ["lunch was good today"]
    assert app.ignored == []


async def test_a_yes_to_sparks_question_gets_through(tmp_path):
    brain = MockBrain(replies=["There are 42. Want the breakdown?", "Phoenix has 12."])
    app = _awake_app(tmp_path, ["how many open orders", "yes"], brain=brain)
    await _awake_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    assert brain.calls == ["how many open orders", "yes"]
    assert app.player.earcons.count("thinking") == 2  # one per answered request


# --- P5.6: a fresh session per wake --------------------------------------------------------------


async def test_falling_asleep_prepares_the_next_session(tmp_path):
    brain = _ResetBrain()
    app = _awake_app(tmp_path, ["what is two plus two"], brain=brain)
    await _awake_turn(app)
    assert brain.resets == 0
    app.go_to_sleep("command")
    await _settle(lambda: brain.resets == 1)
    app.go_to_sleep("command")  # already asleep: nothing more
    await asyncio.sleep(0.01)
    assert brain.resets == 1 and app.brain_resets == 1
    await app.close()


async def test_an_unused_session_is_not_reset(tmp_path):
    brain = _ResetBrain()
    app = _awake_app(tmp_path, ["go to sleep"], brain=brain)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.state is State.IDLE
    await asyncio.sleep(0.01)
    assert brain.resets == 0  # the command never touched the brain
    await app.close()


async def test_a_failed_turn_also_prepares_a_fresh_session(tmp_path):
    class FailingReset(_ResetBrain):
        pass

    brain = FailingReset(fail_with=ConnectionError("offline"))
    app = _awake_app(tmp_path, ["hi there what's up"], brain=brain)
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    await _settle(lambda: brain.resets == 1)
    await app.close()
