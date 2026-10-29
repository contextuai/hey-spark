"""The mute button (Phase 5, P5.7k): the mic is ignored, work under way goes on."""

from __future__ import annotations

import pytest

from spark.app import State
from tests.fakes import FakeClock, FakeSTT, MockBrain, silence
from tests.test_app import (
    _app,
    _awake_app,
    _awake_turn,
    _finish_turn,
    _settings,
    _settle,
    _utterance,
    _wake,
)


async def test_muted_ignores_the_wake_word_and_keeps_nothing(tmp_path):
    seen: list[bool] = []
    app = _app(tmp_path, on_mute=seen.append)
    app.set_muted(True)
    await _wake(app)  # the detector would fire, but it never sees the audio
    await _utterance(app)
    assert app.state is State.IDLE
    assert app.stt.received == [] and app.brain.calls == []
    assert app._ring.read_latest(16000).size == 0 or not app._ring.read_latest(16000).any()
    app.set_muted(False)
    await _wake(app)
    assert app.state is State.LISTENING
    assert seen == [True, False]
    await app.close()


async def test_muting_while_listening_drops_the_half_heard_request(tmp_path):
    app = _app(tmp_path)
    await _wake(app)
    assert app.state is State.LISTENING
    app.set_muted(True)
    assert app.state is State.IDLE
    await _utterance(app)
    app.set_muted(False)
    assert app.brain.calls == [] and app.state is State.IDLE
    await app.close()


async def test_unmute_with_awake_time_left_listens_without_the_wake_word(tmp_path):
    clock = FakeClock()
    app = _awake_app(tmp_path, ["what is two plus two", "and what about three"], clock=clock)
    await _awake_turn(app)
    app.set_muted(True)
    await _utterance(app)  # the visitor talking: never heard
    clock.now += 60
    app.set_muted(False)
    assert app.state is State.FOLLOW_UP
    await _utterance(app)
    await _finish_turn(app)
    assert app.brain.calls == ["what is two plus two", "and what about three"]
    assert app.stt.received.__len__() == 2
    await app.close()


async def test_the_awake_window_runs_out_while_muted(tmp_path):
    clock = FakeClock()
    slept: list[str] = []
    app = _awake_app(tmp_path, ["what is two plus two"], clock=clock)
    app.on_sleep = slept.append
    await _awake_turn(app)
    app.set_muted(True)
    clock.now += 181
    await app.feed(silence(0.1))
    assert app.state is State.IDLE and slept == ["awake window over"]
    app.set_muted(False)
    assert app.state is State.IDLE  # asleep: the wake word is needed again
    await app.close()


async def test_a_reply_under_way_still_finishes(tmp_path):
    brain = MockBrain(replies=["Two plus two is four."], delay_s=0.01)
    app = _app(tmp_path, brain=brain)
    await _wake(app)
    await _utterance(app)
    app.set_muted(True)  # muted while Spark is thinking
    await _finish_turn(app)
    assert app.tts.spoken == ["Two plus two is four."]
    await app.close()


async def test_a_confirmation_while_muted_gets_no_answer(tmp_path):
    app = _app(tmp_path)
    app.set_state(State.THINKING)
    app.set_muted(True)
    assert await app.listen_for_answer(5.0) == ""
    assert app.state is State.THINKING
    await app.close()


async def test_mute_by_voice(tmp_path):
    seen: list[bool] = []
    app = _awake_app(
        tmp_path, ["what is two plus two", "Spark, mute yourself"], on_mute=seen.append
    )
    await _awake_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    assert app.tts.spoken[-1] == "Muted. Use the mic button to unmute me."
    assert app.muted and seen == [True]
    assert app.state is State.FOLLOW_UP  # awake time is kept for after unmuting
    assert app.brain.calls == ["what is two plus two"]
    await app.close()


async def test_mute_is_recorded_in_the_activity_log(tmp_path):
    from spark.activity import ActivityLog

    activity = ActivityLog(tmp_path / "history")
    app = _app(tmp_path, activity=activity)
    app.set_muted(True)
    app.set_muted(True)  # no change, no record
    app.set_muted(False)
    events = [(e["kind"], e.get("muted")) for e in activity.iter_events()]
    assert events == [("mute", True), ("mute", False)]
    await app.close()


# --- the window ---------------------------------------------------------------------------------


async def test_controller_mute_before_and_after_start(tmp_path):
    from tests.fakes import concat, tone
    from tests.test_desktop import _controller, _until

    c, events, built = _controller(tmp_path)
    await c.aset_muted(True)
    assert c.snapshot()["muted"] is True
    assert {"type": "muted", "muted": True} in events
    await c.astart()
    app = built.apps[0]
    assert app.muted  # applied when the app was built
    c._on_level(0.7)
    assert events[-1] == {"type": "level", "level": 0.0}  # the ring stays flat while muted
    await c.aset_muted(False)
    assert not app.muted and events[-1] == {"type": "muted", "muted": False}
    app.wakeword.trigger()
    app.mic.inner.push(concat(silence(0.032), tone(0.6), silence(0.8)))
    await _until(events, lambda e: e.get("state") == "follow_up")
    await c.aclose()


@pytest.mark.parametrize("text", ["mute", "Spark, go on mute", "mute yourself please"])
def test_mute_phrases(text):
    from spark.commands import MUTE, match_command

    cmd = match_command(text, ("spark",))
    assert cmd is not None and cmd.kind == MUTE


def test_unmute_is_not_a_voice_command():
    from spark.commands import match_command

    assert match_command("unmute", ("spark",)) is None
    assert match_command("mute the email from Sam", ("spark",)) is None


async def test_text_mode_ignores_nothing_when_muted(tmp_path):
    app = _app(tmp_path, stt=FakeSTT(transcripts=[]), settings=_settings(tmp_path))
    app.set_muted(True)
    assert await app.text_turn("what is two plus two") == "Two plus two is four."
    await _settle(lambda: True)
    await app.close()
