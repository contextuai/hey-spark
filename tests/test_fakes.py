"""The fakes behave as documented (they are the foundation of every state-machine test)."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from spark import registry
from spark.interfaces import STT, TTS, VAD, AudioInput, AudioOutput, Brain, WakeWordDetector
from tests.fakes import (
    FakeClock,
    FakeConfirmer,
    FakeMic,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWakeWord,
    MockBrain,
    concat,
    make_settings,
    rms,
    silence,
    speech_like,
    tone,
)


async def _collect(stream):
    return [x async for x in stream]


async def _aiter(items, delay_s=0.0):
    for item in items:
        if delay_s:
            await asyncio.sleep(delay_s)
        yield item


# --- helpers ---------------------------------------------------------------------------------


def test_audio_helpers_shape_and_level():
    assert silence(0.5).dtype == np.int16 and len(silence(0.5)) == 8000
    t = tone(1.0, freq=440, amplitude=8000)
    assert t.dtype == np.int16 and len(t) == 16000
    assert rms(t) == pytest.approx(8000 / np.sqrt(2), rel=0.01)
    s = speech_like(1.0)
    assert s.dtype == np.int16 and len(s) == 16000
    np.testing.assert_array_equal(s, speech_like(1.0))  # seeded, deterministic
    joined = concat(silence(0.1), tone(0.2), silence(0.1))
    assert joined.dtype == np.int16 and len(joined) == 6400


def test_make_settings_selects_mock_brain_and_merges():
    s = make_settings(app={"follow_up_s": 3.0}, brain={"idle_reset_s": 5.0})
    assert s.brain.backend == "mock"
    assert s.brain.idle_reset_s == 5.0
    assert s.app.follow_up_s == 3.0
    assert isinstance(registry.build("brain", s), MockBrain)


def test_every_fake_implements_its_interface():
    assert isinstance(FakeMic(), AudioInput)
    assert isinstance(FakeWakeWord(), WakeWordDetector)
    assert isinstance(FakeVAD(), VAD)
    assert isinstance(FakeSTT(), STT)
    assert isinstance(FakeTTS(), TTS)
    assert isinstance(FakePlayer(), AudioOutput)
    assert isinstance(MockBrain(), Brain)


# --- clock -----------------------------------------------------------------------------------


async def test_fake_clock_advances_without_real_sleep():
    clock = FakeClock()
    start = time.perf_counter()
    await clock.sleep(8.0)
    clock.advance(1.5)
    assert clock.monotonic() == 9.5
    assert clock.sleeps == [8.0]
    assert time.perf_counter() - start < 0.1


# --- mic -------------------------------------------------------------------------------------


async def test_mic_cuts_array_into_padded_blocks():
    mic = FakeMic(tone(0.1), block_size=512)  # 1600 samples -> 4 blocks, last padded
    await mic.start()
    blocks = await _collect(mic.frames())
    assert [len(b) for b in blocks] == [512] * 4
    assert all(b.dtype == np.int16 for b in blocks)
    assert not blocks[-1][1600 - 3 * 512 :].any()
    assert mic.started and mic.blocks_yielded == 4


async def test_mic_yields_block_list_as_given():
    mic = FakeMic([np.ones(100, np.int16), np.ones(1280, np.int16)])
    assert [len(b) for b in await _collect(mic.frames())] == [100, 1280]


async def test_mic_hold_open_waits_for_push_then_stop():
    mic = FakeMic(silence(0.064), block_size=512, hold_open=True)
    got: list[int] = []

    async def consume():
        async for block in mic.frames():
            got.append(len(block))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    assert got == [512, 512] and not task.done()
    mic.push(tone(0.032))
    await asyncio.sleep(0.01)
    assert got == [512, 512, 512] and not task.done()
    await mic.stop()
    await asyncio.wait_for(task, 0.5)
    assert mic.stopped


async def test_mic_realtime_uses_injected_clock():
    clock = FakeClock()
    mic = FakeMic(silence(1.0), block_size=1600, realtime=True, clock=clock)
    assert len(await _collect(mic.frames())) == 10
    assert clock.monotonic() == pytest.approx(1.0)


# --- wake word / VAD -------------------------------------------------------------------------


def test_wakeword_fires_at_indices_and_on_trigger():
    ww = FakeWakeWord(fire_at={2, 4})
    frame = np.zeros(512, np.int16)
    assert [ww.process(frame) for _ in range(6)] == [False, False, True, False, True, False]
    ww.trigger()
    assert ww.process(frame) is True
    assert ww.process(frame) is False
    assert ww.detections == [2, 4, 6]
    ww.reset()
    assert ww.resets == 1 and ww.calls == 8


def test_wakeword_marker_tone_mode_and_frame_length_check():
    ww = FakeWakeWord(frame_length=1280, marker_hz=1000)
    assert ww.process(tone(0.08, freq=1000)) is True
    assert ww.process(tone(0.08, freq=300)) is False
    assert ww.process(silence(0.08)) is False
    with pytest.raises(ValueError):
        ww.process(np.zeros(512, np.int16))


def test_vad_energy_on_tone_speech_and_silence():
    vad = FakeVAD()
    assert vad.is_speech(tone(0.032)) is True
    assert vad.is_speech(silence(0.032)) is False
    audio = speech_like(1.0)
    assert all(vad.is_speech(audio[i : i + 512]) for i in range(0, 16000 - 512, 512))
    quiet = tone(0.032, amplitude=300)
    assert vad.is_speech(quiet) is False


def test_vad_mask_mode():
    vad = FakeVAD(mask=[True, False, True])
    frame = np.zeros(512, np.int16)
    assert [vad.is_speech(frame) for _ in range(4)] == [True, False, True, False]
    assert vad.speech_frames == 2


# --- STT -------------------------------------------------------------------------------------


async def test_stt_scripted_list_then_default():
    stt = FakeSTT(transcripts=["hello", "what is two plus two"])
    await stt.warmup()
    assert await stt.transcribe(silence(1.0)) == "hello"
    assert await stt.transcribe(silence(0.5)) == "what is two plus two"
    assert await stt.transcribe(silence(0.1)) == ""
    assert stt.received == [16000, 8000, 1600]
    assert stt.warmups == 1


async def test_stt_callable_delay_and_failure():
    clock = FakeClock()
    stt = FakeSTT(transcripts=lambda pcm: f"{len(pcm)} samples", delay_s=2.0, clock=clock)
    assert await stt.transcribe(silence(0.25)) == "4000 samples"
    assert clock.monotonic() == 2.0
    with pytest.raises(RuntimeError, match="boom"):
        await FakeSTT(fail_with=RuntimeError("boom")).transcribe(silence(0.1))


# --- TTS -------------------------------------------------------------------------------------


async def test_tts_records_chunks_with_times():
    tts = FakeTTS()
    await tts.speak(_aiter(["Hello there. ", "  ", "How are you?"]))
    assert tts.spoken == ["Hello there. ", "How are you?"]
    assert len(tts.chunk_times) == 2
    assert tts.first_chunk_at == tts.chunk_times[0][1]
    assert not tts.speaking and not tts.stopped


async def test_tts_stop_while_playing_returns_fast():
    tts = FakeTTS(per_chunk_s=5.0)
    task = asyncio.create_task(tts.speak(_aiter(["One.", "Two."])))
    for _ in range(100):  # wait up to 1 s for the first chunk (robust under CPU load)
        if tts.spoken:
            break
        await asyncio.sleep(0.01)
    assert tts.speaking and tts.spoken == ["One."]
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert tts.stopped and not tts.speaking
    assert tts.spoken == ["One."]
    assert tts.stop_latency is not None and tts.stop_latency < 0.05


async def test_tts_stop_while_waiting_for_brain_returns_fast():
    tts = FakeTTS()
    task = asyncio.create_task(tts.speak(_aiter(["First.", "Second."], delay_s=5.0)))
    await asyncio.sleep(0.02)
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert tts.spoken == []
    assert tts.stop_latency < 0.05


async def test_tts_plays_through_player_and_stop_stops_player():
    player = FakePlayer(speed=1.0)
    tts = FakeTTS(per_chunk_s=5.0, player=player)
    task = asyncio.create_task(tts.speak(_aiter(["Long sentence."])))
    for _ in range(100):  # wait up to 1 s for playback to start (robust under CPU load)
        if player.is_playing:
            break
        await asyncio.sleep(0.01)
    assert player.is_playing
    await tts.stop()
    await asyncio.wait_for(task, 1.0)
    assert not player.is_playing and player.stops == 1 and player.interrupted_plays == 1
    assert player.played == [(80000, 16000)]


# --- player ----------------------------------------------------------------------------------


async def test_player_records_and_is_instant_by_default():
    player = FakePlayer()
    player.play_earcon("wake")
    await player.play(tone(2.0), 16000)
    player.play_earcon("thinking")
    assert player.earcons == ["wake", "thinking"]
    assert player.played == [(32000, 16000)]
    assert not player.is_playing


async def test_player_stop_is_sync_and_interrupts_play():
    player = FakePlayer(speed=1.0)
    task = asyncio.create_task(player.play(tone(3.0), 16000))
    await asyncio.sleep(0.02)
    assert player.is_playing
    player.stop()
    await asyncio.wait_for(task, 1.0)
    assert not player.is_playing
    assert player.stop_latency < 0.05


async def test_player_virtual_time_with_clock():
    clock = FakeClock()
    player = FakePlayer(speed=1.0, clock=clock)
    await player.play(tone(1.5), 16000)
    assert clock.monotonic() == pytest.approx(1.5)


# --- MockBrain -------------------------------------------------------------------------------


async def test_mock_brain_default_replies_stream_in_several_deltas():
    brain = MockBrain()
    deltas = await _collect(brain.respond("What is two plus two?"))
    assert len(deltas) >= 3
    assert "".join(deltas) == "Two plus two is four."
    echo = "".join(await _collect(brain.respond("turn on the lights")))
    assert echo == "You said: turn on the lights."
    assert brain.calls == ["What is two plus two?", "turn on the lights"]


async def test_mock_brain_scripted_list_callable_and_chunk_size():
    brain = MockBrain(replies=["First reply. Second sentence.", "Another."], chunk_size=4)
    deltas = await _collect(brain.respond("a"))
    assert all(len(d) <= 4 for d in deltas)
    assert "".join(deltas) == "First reply. Second sentence."
    assert "".join(await _collect(brain.respond("b"))) == "Another."
    assert "".join(await _collect(brain.respond("hello"))).startswith("Hi, I'm Spark")

    upper = MockBrain(replies=lambda text: text.upper() + ".")
    assert "".join(await _collect(upper.respond("ok then"))) == "OK THEN."


async def test_mock_brain_interrupt_stops_stream():
    brain = MockBrain(replies=["one two three four five six seven eight"], delay_s=0.05)
    got: list[str] = []

    async def consume():
        async for delta in brain.respond("go"):
            got.append(delta)
            if len(got) == 2:
                await brain.interrupt()

    await asyncio.wait_for(consume(), 1.0)
    assert got == ["one ", "two "]
    assert brain.interrupts == 1 and not brain.in_flight


async def test_mock_brain_delay_uses_clock():
    clock = FakeClock()
    brain = MockBrain(replies=["a b c"], delay_s=1.0, clock=clock)
    assert "".join(await _collect(brain.respond("x"))) == "a b c"
    assert clock.monotonic() == pytest.approx(3.0)


async def test_mock_brain_fail_with_before_and_mid_stream():
    brain = MockBrain(fail_with=RuntimeError("brain down"))
    with pytest.raises(RuntimeError, match="brain down"):
        await _collect(brain.respond("hi"))

    mid = MockBrain(replies=["one two three"], fail_with=ConnectionError, fail_after=2)
    got: list[str] = []
    with pytest.raises(ConnectionError):
        async for delta in mid.respond("x"):
            got.append(delta)
    assert got == ["one ", "two "]


async def test_mock_brain_reset_warmup_close():
    brain = MockBrain(make_settings())
    await brain.warmup()
    await brain.reset()
    await brain.reset()
    await brain.close()
    assert (brain.warmups, brain.resets, brain.closed) == (1, 2, True)


# --- confirmer -------------------------------------------------------------------------------


async def test_confirmer_scripted_then_default_no_answer():
    confirm = FakeConfirmer([True, False])
    assert await confirm.ask("send the email") is True
    assert await confirm.ask("create the repo") is False
    assert await confirm.ask("delete the file") is None  # default: no answer, so denied
    assert confirm.asked == ["send the email", "create the repo", "delete the file"]
