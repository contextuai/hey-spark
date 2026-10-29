"""Barge-in (P3.3, spec §3 and Phase 3: "Barge-in stops speech within 300 ms").

Every test drives :meth:`SparkApp.feed` with fakes. Stop latency is measured in wall time
(``time.perf_counter``) from the wake frame being processed to ``player.stop()`` returning
(``BargeIn.stop_latency_ms``) and to the interrupted ``speak()`` returning (``speak_return_ms``).
The measured values are printed; run with ``-s`` to see them.

The ``hardware`` test at the end plays Piper through the real speaker and is skipped by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from spark.app import SparkApp, State
from spark.audio.player import BLOCK_MS, SoundDevicePlayer
from spark.brain.confirm import ConfirmationGate, VoiceConfirmer
from spark.brain.mock import default_reply
from spark.config import Settings
from spark.tools.timers import TimerService
from spark.tts.base import SentenceTTS
from tests.fakes import (
    FakeClock,
    FakeMic,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWakeWord,
    MockBrain,
    ToolCallingBrain,
    concat,
    make_settings,
    silence,
    tone,
)
from tests.test_audio_player import RealtimeFakeStream
from tests.test_confirm import notes_setup

REPO = Path(__file__).resolve().parents[1]
BLOCK = 512
BUDGET_MS = 300.0
LONG_REPLY = "This is the first sentence. Here is the second one. A third. And a fourth one."


def _settings(tmp_path: Path, *, barge_in: bool = True, earcons: bool = True) -> Settings:
    return make_settings(
        logging={"dir": str(tmp_path / "logs")},
        debug_audio={"enabled": False, "dir": str(tmp_path / "dbg")},
        app={"barge_in": barge_in, "earcons": earcons},
    )


def _app(tmp_path: Path, *, settings=None, player=None, tts=None, brain=None, stt=None, **kw):
    settings = settings or _settings(tmp_path)
    player = player if player is not None else FakePlayer()
    return SparkApp(
        settings,
        mic=FakeMic(),
        player=player,
        wakeword=FakeWakeWord(),
        vad=FakeVAD(),
        stt=stt or FakeSTT(transcripts=["tell me a story", "what is two plus two"]),
        tts=tts if tts is not None else FakeTTS(player=player),
        brain=brain or MockBrain(replies=[LONG_REPLY]),
        **kw,
    )


async def _feed(app: SparkApp, pcm: np.ndarray) -> None:
    for i in range(0, len(pcm), BLOCK):
        await app.feed(pcm[i : i + BLOCK])


async def _wake(app: SparkApp) -> None:
    app.wakeword.trigger()
    await _feed(app, silence(BLOCK / 16000))


async def _utterance(app: SparkApp) -> None:
    await _feed(app, concat(tone(0.6), silence(0.8)))


async def _until(pred, limit_s: float = 5.0) -> None:
    async with asyncio.timeout(limit_s):
        while not pred():  # noqa: ASYNC110 - polling app state that has no event
            await asyncio.sleep(0.002)


def _records(tmp_path: Path) -> list[dict]:
    path = tmp_path / "logs" / "latency.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _after_last_barge(app: SparkApp) -> list[State]:
    history = app.state_history
    i = max(i for i, s in enumerate(history) if s is State.LISTENING)
    return history[i:]


async def _speaking(app: SparkApp) -> None:
    """Wake, ask, and wait until the first sentence is audibly playing."""
    await _wake(app)
    await _utterance(app)
    await _until(lambda: app.state is State.SPEAKING)
    await _until(lambda: app.player.is_playing)


def _report(name: str, app: SparkApp, extra: str = "") -> None:
    b = app.last_barge_in
    assert b is not None
    print(
        f"\n[barge-in] {name}: wake->player.stop {b.stop_latency_ms} ms, "
        f"wake->speak() returned {b.speak_return_ms} ms, unwound {b.unwind_ms} ms {extra}"
    )


# --- SPEAKING ------------------------------------------------------------------------------------


async def test_barge_in_with_realtime_fake_player(tmp_path):
    """FakeTTS playing each sentence as 1 s of audio through a real-time FakePlayer."""
    player = FakePlayer(speed=1.0)
    tts = FakeTTS(per_chunk_s=1.0, player=player)
    brain = MockBrain(replies=[LONG_REPLY])
    app = _app(tmp_path, player=player, tts=tts, brain=brain)
    try:
        await _speaking(app)
        await asyncio.sleep(0.2)  # mid-sentence
        spoken, played = list(tts.spoken), len(player.played)
        await _wake(app)

        assert app.state is State.LISTENING
        assert app.player.earcons[-1] == "wake"
        barge = app.last_barge_in
        assert barge is not None and barge.source == "speaking"
        assert barge.stop_latency_ms < BUDGET_MS
        assert app.stop_latency_ms == barge.stop_latency_ms
        assert app.turn_task is None
        await app.wait_interrupted()
        assert brain.interrupts >= 1 and barge.brain_interrupted
        assert barge.speak_return_ms is not None and barge.speak_return_ms < BUDGET_MS
        assert player.stop_latency is not None and player.stop_latency < BUDGET_MS / 1000
        _report(
            "FakeTTS + FakePlayer(speed=1)", app, f"(player {player.stop_latency * 1e3:.1f} ms)"
        )

        await asyncio.sleep(0.3)  # nothing else from the old reply comes out
        assert tts.spoken == spoken == ["This is the first sentence."]
        assert len(player.played) == played

        tts.per_chunk_s = 0.05
        await _utterance(app)  # the next request is heard and answered normally
        assert app.state is State.THINKING
        await asyncio.wait_for(app.turn_task, 5)
        assert brain.calls == ["tell me a story", "what is two plus two"]
        assert tts.spoken[-1] == "Two plus two is four."
        assert app.state is State.FOLLOW_UP
        assert _after_last_barge(app) == [
            State.LISTENING,
            State.THINKING,
            State.SPEAKING,
            State.FOLLOW_UP,
        ]
        assert app.state_history.count(State.FOLLOW_UP) == 1  # no stale FOLLOW_UP

        interrupted, answered = _records(tmp_path)
        assert interrupted["meta"]["interrupted"] is True
        assert interrupted["meta"]["barge_in_from"] == "speaking"
        assert interrupted["meta"]["stop_latency_ms"] == barge.stop_latency_ms
        assert interrupted["meta"]["speak_return_ms"] == barge.speak_return_ms
        assert "interrupted" not in answered["meta"]
        assert answered["meta"]["after_barge_in"] is True
        assert app.last_reply == "Two plus two is four."
    finally:
        await app.close()


class ToneSynthTTS(SentenceTTS):
    """A real :class:`SentenceTTS` whose "synthesis" is ``seconds`` of 440 Hz per sentence."""

    backend = "tts/tone"

    def __init__(self, settings, *, seconds: float = 1.5, **deps):
        super().__init__(settings, **deps)
        self.seconds = seconds
        self.synthesized: list[str] = []

    def synthesize(self, text):
        self.synthesized.append(text)
        return tone(self.seconds, amplitude=12000), 16_000


async def test_barge_in_with_sentence_tts_and_realtime_fake_player(tmp_path):
    settings = _settings(tmp_path)
    player = FakePlayer(settings, speed=1.0)
    tts = ToneSynthTTS(settings, player=player)
    brain = MockBrain(replies=[LONG_REPLY])
    app = _app(tmp_path, settings=settings, player=player, tts=tts, brain=brain)
    try:
        await _speaking(app)
        await asyncio.sleep(0.3)
        played = len(player.played)
        await _wake(app)

        barge = app.last_barge_in
        assert app.state is State.LISTENING
        assert barge is not None and barge.stop_latency_ms < BUDGET_MS
        await app.wait_interrupted()
        assert barge.speak_return_ms is not None and barge.speak_return_ms < BUDGET_MS
        assert brain.interrupts >= 1
        _report("SentenceTTS + FakePlayer(speed=1)", app)

        await asyncio.sleep(0.3)
        assert len(player.played) == played == 1  # only the interrupted sentence ever played
        assert not tts.speaking

        await _utterance(app)
        await asyncio.wait_for(app.turn_task, 10)
        assert brain.calls[-1] == "what is two plus two"
        assert tts.synthesized[-1] == "Two plus two is four."
        assert app.state is State.FOLLOW_UP
    finally:
        await app.close()


class TimedStream(RealtimeFakeStream):
    """Real-time fake output stream that notes when each block started and how loud it was."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.blocks: list[tuple[float, int]] = []  # (perf_counter at callback start, nonzero)
        self._lock = threading.Lock()

    def tick(self):
        started = time.perf_counter()
        out = super().tick()
        with self._lock:
            self.blocks.append((started, int(np.count_nonzero(out[:, 0]))))
        return out

    def nonzero_since(self, t: float) -> int:
        with self._lock:
            return sum(n for start, n in self.blocks if start >= t)

    def nonzero_before(self, t: float) -> int:
        with self._lock:
            return sum(n for start, n in self.blocks if start < t)


async def test_playback_buffer_plays_at_most_one_block_after_barge_in(tmp_path):
    """The real SoundDevicePlayer / PlaybackBuffer path with a fake 48 kHz device."""
    settings = _settings(tmp_path, earcons=False)  # nothing but speech reaches the device
    streams: list[TimedStream] = []

    def factory(**kw):
        streams.append(TimedStream(**kw))
        return streams[-1]

    player = SoundDevicePlayer(settings, stream_factory=factory, device_rate=48_000, channels=1)
    tts = ToneSynthTTS(settings, player=player, seconds=2.0)
    app = _app(tmp_path, settings=settings, player=player, tts=tts)
    try:
        await app.warmup()
        await _speaking(app)
        await asyncio.sleep(0.5)
        await _wake(app)
        barge = app.last_barge_in
        assert barge is not None and app.state is State.LISTENING
        await app.wait_interrupted()
        await asyncio.sleep(0.2)  # ten more device blocks

        stream = streams[0]
        block = stream.blocksize  # 960 samples = 20 ms
        assert stream.nonzero_before(barge.detected_at) > 10 * block  # it really was speaking
        leaked = stream.nonzero_since(barge.detected_at)
        assert leaked <= block
        assert barge.stop_latency_ms < BUDGET_MS
        assert barge.speak_return_ms is not None and barge.speak_return_ms < BUDGET_MS
        assert not player.is_playing
        _report(
            "SentenceTTS + SoundDevicePlayer(PlaybackBuffer)",
            app,
            f"(samples after the wake frame: {leaked}, one block = {block} = {BLOCK_MS} ms)",
        )
    finally:
        await app.close()


class StubbornTTS(FakeTTS):
    """Once stopped, swallows the turn's cancellation and returns 200 ms late, like a backend
    that unwinds slowly. The old turn then carries on after the barge-in, as it would if a
    cancel got lost."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.late_returns = 0

    async def speak(self, text_chunks):
        with contextlib.suppress(asyncio.CancelledError):
            await super().speak(text_chunks)
        if self.stopped:
            self.late_returns += 1
            await asyncio.sleep(0.2)


async def test_stale_turn_cannot_flip_state_after_barge_in(tmp_path):
    tts = StubbornTTS(per_chunk_s=2.0)
    brain = MockBrain(replies=[LONG_REPLY, "Sure."])
    app = _app(tmp_path, tts=tts, brain=brain)
    try:
        await _wake(app)
        await _utterance(app)
        await _until(lambda: tts.spoken)
        old_turn = app.turn_task
        await _wake(app)
        assert app.state is State.LISTENING
        tts.per_chunk_s = 0.0
        await _utterance(app)  # the new turn is THINKING while the old one is still unwinding
        assert app.state is State.THINKING
        assert old_turn is not None and not old_turn.done()
        await asyncio.wait_for(app.turn_task, 5)
        assert old_turn.done() and tts.late_returns == 1
        # Without the generation check the old turn would have jumped THINKING -> FOLLOW_UP.
        assert _after_last_barge(app) == [
            State.LISTENING,
            State.THINKING,
            State.SPEAKING,
            State.FOLLOW_UP,
        ]
        assert app.state_history.count(State.FOLLOW_UP) == 1
        assert tts.spoken == ["This is the first sentence.", "Sure."]
        assert [r["meta"].get("interrupted", False) for r in _records(tmp_path)] == [True, False]
    finally:
        await app.close()


# --- THINKING -------------------------------------------------------------------------------------


@pytest.mark.parametrize("slow", ["stt", "brain"])
async def test_barge_in_while_thinking_aborts_the_turn(tmp_path, slow):
    stt = FakeSTT(
        transcripts=["tell me a story", "what is two plus two"],
        delay_s=0.5 if slow == "stt" else 0.0,
    )
    brain = MockBrain(delay_s=0.5 if slow == "brain" else 0.0)  # canned replies
    app = _app(tmp_path, stt=stt, brain=brain)
    try:
        await _wake(app)
        await _utterance(app)
        assert app.state is State.THINKING
        await asyncio.sleep(0.05)
        await _wake(app)
        assert app.state is State.LISTENING
        assert app.last_barge_in is not None and app.last_barge_in.source == "thinking"
        await app.wait_interrupted()
        assert brain.interrupts >= 1
        assert app.tts.spoken == []
        record = _records(tmp_path)[-1]
        assert record["meta"]["interrupted"] is True
        assert record["meta"]["barge_in_from"] == "thinking"

        brain.delay_s = stt.delay_s = 0.0
        await _utterance(app)
        await asyncio.wait_for(app.turn_task, 5)
        # (a cancelled FakeSTT never consumed its first transcript, so it returns it now)
        heard = stt.results[-1]
        assert brain.calls[-1] == heard
        assert app.tts.spoken == [default_reply(heard)]  # only the new turn was ever spoken
        assert app.state is State.FOLLOW_UP
    finally:
        await app.close()


# --- disabled by config ---------------------------------------------------------------------------


async def test_wake_word_is_ignored_while_busy_when_barge_in_is_off(tmp_path):
    settings = _settings(tmp_path, barge_in=False)
    brain = MockBrain(replies=["One. Two. Three."], delay_s=0.02)
    app = _app(tmp_path, settings=settings, brain=brain, tts=FakeTTS(per_chunk_s=0.05))
    try:
        await _wake(app)
        await _utterance(app)
        assert app.state is State.THINKING
        calls = app.wakeword.calls
        await _wake(app)  # THINKING
        await _until(lambda: app.state is State.SPEAKING)
        await _wake(app)  # SPEAKING
        assert app.wakeword.calls == calls  # the detector isn't even run
        await asyncio.wait_for(app.turn_task, 5)
        assert app.tts.spoken == ["One.", "Two.", "Three."]
        assert app.state is State.FOLLOW_UP
        assert app.last_barge_in is None and brain.interrupts == 0
        assert app.state_history.count(State.LISTENING) == 1
    finally:
        await app.close()


# --- CONFIRMING -----------------------------------------------------------------------------------


def _confirm_app(tmp_path: Path):
    settings = _settings(tmp_path)
    brain = ToolCallingBrain(settings)
    stt = FakeSTT(transcripts=["make a note called groceries", "make a note called groceries"])
    app = _app(tmp_path, settings=settings, brain=brain, stt=stt, tts=FakeTTS())
    confirmer = VoiceConfirmer(
        listen=app.listen_for_answer,
        say=app.say_for_confirmation,
        cue=app.confirmation_cue,
        timeout_s=settings.confirm.timeout_s,
    )
    app.confirmer = confirmer
    gate = ConfirmationGate(notes_setup().policy, confirmer)
    brain.gate = gate
    return app, brain, gate


async def test_wake_word_while_confirming_cancels_and_denies(tmp_path):
    app, brain, gate = _confirm_app(tmp_path)
    try:
        await _wake(app)
        await _utterance(app)
        await _until(lambda: app.state is State.CONFIRMING)
        turn = app.turn_task
        await _wake(app)  # "Hey Spark" instead of yes/no
        assert app.state is State.LISTENING
        assert app.last_barge_in is not None and app.last_barge_in.source == "confirming"
        await app.wait_interrupted()
        assert turn is not None and turn.done()
        assert brain.executed == []  # the wake word was not taken as an answer
        assert gate.decisions[-1][1] in {"cancelled", "timeout"}  # denied either way
        assert app.stt.results == ["make a note called groceries"]  # no answer transcribed
        assert brain.interrupts >= 1

        # Spark listens for a new request: it asks again, and silence denies it.
        await _utterance(app)
        await _until(lambda: app.state is State.CONFIRMING)
        await _feed(app, silence(app.settings.confirm.timeout_s + 0.2))
        await asyncio.wait_for(app.turn_task, 5)
        assert brain.executed == []
        assert gate.decisions[-1] == ("mcp__notes__create_note", "timeout")
        assert app.state is State.FOLLOW_UP
    finally:
        await app.close()


# --- timer announcements --------------------------------------------------------------------------


async def test_barge_in_during_timer_announcement_in_idle(tmp_path):
    timers = TimerService(clock=FakeClock())
    player = FakePlayer(speed=1.0)
    tts = FakeTTS(per_chunk_s=2.0, player=player)  # tts.stop() also stops the player
    app = _app(tmp_path, player=player, tts=tts, timers=timers)
    try:
        timers.set("5 minutes", "pasta")
        await _until(lambda: tts.speaking and tts.spoken == ["Your pasta timer is done."])
        await _wake(app)
        assert app.state is State.LISTENING
        barge = app.last_barge_in
        assert barge is not None and barge.source == "idle"
        assert barge.stop_latency_ms < BUDGET_MS
        stops = player.stops
        await app.wait_interrupted()
        await asyncio.sleep(0.05)
        assert not tts.speaking and barge.speak_return_ms is not None
        assert barge.speak_return_ms < BUDGET_MS
        assert player.earcons[-1] == "wake"
        assert player.stops == stops  # nothing stopped the player after the wake earcon
        assert app.brain.interrupts == 0  # there was no turn to interrupt
        _report("timer announcement (IDLE)", app)
    finally:
        await app.close()


async def test_barge_in_during_timer_announcement_in_follow_up(tmp_path):
    timers = TimerService(clock=FakeClock())
    tts = FakeTTS(per_chunk_s=0.0)
    app = _app(tmp_path, tts=tts, timers=timers, brain=MockBrain(replies=["Okay."]))
    try:
        await _wake(app)
        await _utterance(app)
        await asyncio.wait_for(app.turn_task, 5)
        assert app.state is State.FOLLOW_UP
        tts.per_chunk_s = 2.0
        timers.set(60, "tea")
        await _until(lambda: tts.speaking and tts.spoken[-1] == "Your tea timer is done.")
        await _wake(app)
        assert app.state is State.LISTENING
        assert app.last_barge_in is not None and app.last_barge_in.source == "follow_up"
        await app.wait_interrupted()
        assert not tts.speaking
        tts.per_chunk_s = 0.0
        await _utterance(app)
        await asyncio.wait_for(app.turn_task, 5)
        assert app.brain.calls == ["tell me a story", "what is two plus two"]
        assert app.state is State.FOLLOW_UP
    finally:
        await app.close()


# --- real speaker (skipped by default: -m hardware) ---------------------------------------------


def _piper_present() -> bool:
    from spark.models import piper_voice_paths

    return all(p.is_file() for p in piper_voice_paths(Settings(config_dir=REPO)))


@pytest.mark.hardware
@pytest.mark.skipif(not _piper_present(), reason="Piper voice absent: spark models download")
async def test_hardware_barge_in_stops_piper_on_the_real_speaker(tmp_path):
    """Plays a long Piper sentence out loud, barges in after 1 s, and measures the stop.

    Run by hand: ``uv run pytest -m hardware -s tests/test_barge_in.py``.
    """
    from spark.tts.piper import PiperTTS

    settings = Settings(
        config_dir=REPO,
        brain={"backend": "mock"},
        logging={"dir": str(tmp_path / "logs")},
        app={"barge_in": True, "earcons": False},
    )
    player = SoundDevicePlayer(settings)
    tts = PiperTTS(settings, player=player)
    reply = (
        "This is a deliberately long sentence that keeps going for several seconds so that "
        "the barge-in can cut it off in the middle while it is still clearly audible."
    )
    app = _app(
        tmp_path, settings=settings, player=player, tts=tts, brain=MockBrain(replies=[reply])
    )
    try:
        await app.warmup()
        await _wake(app)
        await _utterance(app)
        await _until(lambda: tts.first_audio_at is not None, limit_s=30)
        await asyncio.sleep(1.0)
        await _wake(app)
        barge = app.last_barge_in
        assert barge is not None and app.state is State.LISTENING
        await app.wait_interrupted()
        stream = player._stream
        device_ms = float(getattr(stream, "latency", 0.0) or 0.0) * 1000
        worst = barge.stop_latency_ms + BLOCK_MS + device_ms
        _report("Piper on the real speaker", app, f"(device buffer {device_ms:.1f} ms)")
        print(f"[barge-in] worst-case audible stop: {worst:.1f} ms")
        assert not player.is_playing
        assert worst < BUDGET_MS
    finally:
        await app.close()
