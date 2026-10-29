"""Bench commands, driven by fakes (the real runs are human-run on the laptop)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from spark import bench
from spark.app import SparkApp
from spark.audio.wavio import read_wav
from tests.fakes import (
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


def test_stats():
    s = bench.stats([100, 200, 300, 400])
    assert s["median"] == 250 and s["n"] == 4 and s["min"] == 100 and s["max"] == 400
    with pytest.raises(ValueError):
        bench.stats([])


async def test_bench_stt_with_fake(tmp_path):
    stt = FakeSTT(transcripts="the capital of japan")
    inputs = [("cmd_01.wav", tone(1.0)), ("cmd_02.wav", tone(0.5))]
    result = await bench.bench_stt(make_settings(), 2, stt=stt, inputs=inputs)
    assert result["transcribe_ms"]["n"] == 4
    assert stt.warmups == 1
    assert result["texts"]["cmd_01.wav"] == "the capital of japan"


async def test_bench_stt_without_fixtures_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"generate.py"):
        await bench.bench_stt(make_settings(), 1, stt=FakeSTT(), inputs=[])


class _SynthTTS(FakeTTS):
    def synthesize(self, text):
        return np.zeros(22050, dtype=np.int16), 22050


async def test_bench_tts_uses_synthesize():
    result = await bench.bench_tts(make_settings(), 3, tts=_SynthTTS())
    assert result["synth_ms"]["n"] == 3
    assert result["audio_s"] == 1.0


async def test_bench_tts_rejects_backend_without_synthesize():
    with pytest.raises(TypeError):
        await bench.bench_tts(make_settings(), 1, tts=FakeTTS())


async def test_bench_brain_with_mock():
    brain = MockBrain(replies=lambda text: "Tokyo is the capital of Japan.")
    result = await bench.bench_brain(make_settings(), 3, brain=brain)
    assert result["ttft_ms"]["n"] == 3
    assert result["last_reply"] == "Tokyo is the capital of Japan."
    assert brain.closed


async def test_bench_latency_counts_voice_turns(tmp_path):
    # each turn: a 1 kHz marker tone (the fake wake word), a question, then enough silence
    # for end-of-speech (0.7 s) and the follow-up window (0.5 s) to expire back to IDLE
    one_turn = concat(tone(0.128, freq=1000), tone(0.6), silence(2.5))
    trials = 3
    mic = FakeMic(concat(silence(0.2), *[one_turn] * (trials + 1)), hold_open=True)
    settings = make_settings(logging={"dir": str(tmp_path)}, app={"follow_up_s": 0.5})
    app = SparkApp(
        settings,
        mic=mic,
        player=FakePlayer(),
        wakeword=FakeWakeWord(marker_hz=1000),
        vad=FakeVAD(),
        stt=FakeSTT(default="what is two plus two"),
        tts=FakeTTS(),
        brain=MockBrain(),
    )
    seen = []
    result = await asyncio.wait_for(
        bench.bench_latency(app, trials, on_turn=lambda n, t: seen.append(n)), 10
    )
    assert result["trials"] == trials
    assert seen == [1, 2, 3]
    assert result["summary"]["total_ms"]["n"] == trials
    assert result["passed"] is True  # fakes are instant
    assert mic.stopped
    assert result["response_ms"]["n"] == trials


async def test_bench_false_wake_counts_by_audio_time():
    minutes = 0.05  # 3 s of audio
    mic = FakeMic(silence(4.0))
    wake = FakeWakeWord(fire_at={10})
    result = await bench.bench_false_wake(make_settings(), minutes, mic=mic, wakeword=wake)
    assert result["wakes"] == 1
    assert result["complete"] is True
    assert result["passed"] is False
    assert result["minutes"] == pytest.approx(0.05, abs=0.01)
    assert wake.closed

    quiet = await bench.bench_false_wake(
        make_settings(), minutes, mic=FakeMic(silence(4.0)), wakeword=FakeWakeWord()
    )
    assert quiet["wakes"] == 0 and quiet["passed"] is True


async def test_bench_false_wake_incomplete_run_does_not_pass():
    result = await bench.bench_false_wake(
        make_settings(), 1.0, mic=FakeMic(silence(2.0)), wakeword=FakeWakeWord()
    )
    assert result["complete"] is False and result["passed"] is False


async def test_record_fixture(tmp_path):
    mic = FakeMic(tone(3.0))
    path = await bench.record_fixture(make_settings(), "cmd_01", 2.0, mic=mic, out_dir=tmp_path)
    pcm, rate = read_wav(path)
    assert path.name == "cmd_01.wav" and rate == 16000 and len(pcm) == 32000
    with pytest.raises(ValueError):
        await bench.record_fixture(make_settings(), "../evil", 1.0, mic=FakeMic(), out_dir=tmp_path)
