"""Recorder endpointing on synthetic speech and silence, plus an optional real-Silero test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spark.audio.recorder import Recorder
from spark.audio.wavio import read_wav
from spark.config import Settings
from spark.interfaces import VAD

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000


class EnergyVAD(VAD):
    """Speech = frame RMS above a threshold. Counts calls and resets."""

    def __init__(self, frame_length: int = 512, threshold: float = 500.0) -> None:
        self.frame_length = frame_length
        self.threshold = threshold
        self.calls = 0
        self.resets = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        assert frame.size == self.frame_length
        self.calls += 1
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) > self.threshold

    def reset(self) -> None:
        self.resets += 1


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


def _settings(**recorder) -> Settings:
    return Settings(config_dir=ROOT, recorder=recorder) if recorder else Settings(config_dir=ROOT)


def speech(ms: float) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000)) / SR
    return (6000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def silence(ms: float) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


def feed(rec: Recorder, pcm: np.ndarray, block: int = 320) -> int | None:
    """Feed ``block``-sized pieces; return the input offset where the recorder finished."""
    for i in range(0, pcm.size, block):
        rec.add(pcm[i : i + block])
        if rec.done:
            return i + min(block, pcm.size - i)
    return None


def test_end_of_speech_fires_within_one_frame_of_700ms():
    vad = EnergyVAD()
    rec = Recorder(_settings(), vad, clock=FakeClock())
    rec.start()
    speech_end = 1024 * 10  # 640 ms of speech, frame-aligned
    audio = np.concatenate([speech(speech_end / SR * 1000), silence(2000)])
    assert feed(rec, audio, block=512) is not None
    assert rec.end_of_speech and not rec.timed_out and rec.heard_speech
    expected = speech_end + int(0.7 * SR)  # 700 ms of silence after speech
    assert expected <= rec.end_sample <= expected + vad.frame_length
    assert vad.resets == 1


@pytest.mark.parametrize("block", [160, 441, 1280, 3000])
def test_endpoint_is_independent_of_mic_block_size(block):
    vad = EnergyVAD()
    rec = Recorder(_settings(), vad)
    rec.start()
    audio = np.concatenate([silence(300), speech(800), silence(1500)])
    feed(rec, audio, block=block)
    assert rec.end_of_speech
    speech_end = int(1.1 * SR)
    expected = speech_end + int(0.7 * SR)
    assert expected <= rec.end_sample <= expected + 2 * vad.frame_length


def test_short_pause_does_not_end_utterance():
    rec = Recorder(_settings(), EnergyVAD())
    rec.start()
    feed(rec, np.concatenate([speech(500), silence(400), speech(500)]))
    assert not rec.done
    feed(rec, silence(800))
    assert rec.end_of_speech


def test_blip_shorter_than_min_speech_is_not_speech():
    rec = Recorder(_settings(min_speech_ms=150, no_speech_timeout_s=2.0), EnergyVAD())
    rec.start()
    feed(rec, np.concatenate([speech(64), silence(1000)]), block=512)
    assert not rec.heard_speech and not rec.end_of_speech
    feed(rec, silence(1500))
    assert rec.timed_out and rec.reason == "no_speech"


def test_no_speech_timeout_and_follow_up_override():
    rec = Recorder(_settings(no_speech_timeout_s=5.0), EnergyVAD())
    rec.start()
    feed(rec, silence(4900))
    assert not rec.done
    feed(rec, silence(200))
    assert rec.timed_out and rec.reason == "no_speech" and not rec.heard_speech

    rec.start(no_speech_timeout_s=8.0)  # FOLLOW_UP window
    feed(rec, silence(7900))
    assert not rec.done
    feed(rec, silence(200))
    assert rec.reason == "no_speech"
    assert rec.end_sample == pytest.approx(8 * SR, abs=512)


def test_max_utterance_length():
    rec = Recorder(_settings(max_utterance_s=3.0), EnergyVAD())
    rec.start()
    feed(rec, speech(5000))
    assert rec.timed_out and rec.reason == "max_length" and rec.heard_speech
    assert rec.end_sample == pytest.approx(3 * SR, abs=512)
    n = rec.pcm().size
    rec.add(speech(500))  # finished recorders ignore input
    assert rec.pcm().size == n


def test_pre_roll_included_in_pcm_but_not_judged():
    vad = EnergyVAD()
    rec = Recorder(_settings(), vad)
    pre = speech(300)  # loud pre-roll must not count as heard speech
    rec.start(pre_roll=pre)
    assert rec.pre_roll_samples == pre.size
    body = np.concatenate([silence(100), speech(600), silence(800)])
    feed(rec, body)
    assert rec.end_of_speech
    pcm = rec.pcm()
    assert np.array_equal(pcm[: pre.size], pre)
    assert np.array_equal(pcm[pre.size : pre.size + 1600], silence(100))
    assert rec.duration_s == pytest.approx(pcm.size / SR)
    assert vad.calls * vad.frame_length == rec.samples_consumed


def test_pre_roll_alone_does_not_satisfy_min_speech():
    rec = Recorder(_settings(no_speech_timeout_s=1.0), EnergyVAD())
    rec.start(pre_roll=speech(300))
    feed(rec, silence(1100))
    assert rec.reason == "no_speech"


def test_restart_clears_state(tmp_path):
    rec = Recorder(_settings(), EnergyVAD(frame_length=1280))
    rec.start()
    feed(rec, np.concatenate([speech(500), silence(900)]))
    assert rec.end_of_speech
    path = rec.save_wav(tmp_path / "debug" / "last.wav")
    saved, rate = read_wav(path)
    assert rate == SR and np.array_equal(saved, rec.pcm())
    rec.start()
    assert not rec.done and not rec.heard_speech and rec.pcm().size == 0
    assert rec.started_at is not None


def test_add_before_start_is_ignored():
    rec = Recorder(_settings(), EnergyVAD())
    rec.add(speech(100))
    assert rec.pcm().size == 0 and not rec.done


def _silero_model_path() -> Path | None:
    try:
        import faster_whisper
    except ImportError:
        return None
    p = Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"
    return p if p.is_file() else None


@pytest.mark.skipif(
    _silero_model_path() is None, reason="Silero model not available (faster-whisper not installed)"
)
def test_recorder_with_real_silero_on_silence():
    """Real Silero: pure silence never counts as speech, so the no-speech timeout fires."""
    from spark import registry

    vad = registry.build("vad", _settings(), name="silero")
    rec = Recorder(_settings(no_speech_timeout_s=1.0), vad)
    rec.start()
    feed(rec, silence(1200), block=512)
    assert rec.reason == "no_speech" and not rec.heard_speech


@pytest.mark.skipif(
    _silero_model_path() is None, reason="Silero model not available (faster-whisper not installed)"
)
def test_recorder_with_real_silero_ignores_steady_tone():
    """Real Silero: a steady 220 Hz tone (loud but not speech) must not open an utterance."""
    from spark import registry

    vad = registry.build("vad", _settings(), name="silero")
    rec = Recorder(_settings(no_speech_timeout_s=1.5), vad)
    rec.start()
    feed(rec, np.concatenate([speech(1000), silence(700)]), block=441)
    assert rec.reason == "no_speech" and not rec.heard_speech


# --- P5.2: long waits for speech (the awake window) -------------------------------------------


def test_max_utterance_counts_from_the_first_speech_not_from_start():
    rec = Recorder(_settings(max_utterance_s=3.0), EnergyVAD())
    rec.start(no_speech_timeout_s=180.0)
    assert feed(rec, silence(60_000), block=1600) is None  # a minute of waiting is fine
    assert not rec.done
    feed(rec, speech(5000))
    assert rec.reason == "max_length"
    assert rec.end_sample - rec.first_speech_sample == pytest.approx(3 * SR, abs=512)


def test_waiting_silence_is_trimmed_to_the_pre_roll():
    rec = Recorder(_settings(pre_roll_ms=300), EnergyVAD())
    pre = speech(300)
    rec.start(pre_roll=pre, no_speech_timeout_s=180.0)
    feed(rec, silence(120_000), block=1600)
    feed(rec, np.concatenate([speech(600), silence(800)]))
    assert rec.end_of_speech
    pcm = rec.pcm()
    assert np.array_equal(pcm[: pre.size], pre)  # the explicit pre-roll is kept whole
    # pre-roll + at most ~2 x (300 ms + one frame) of the waiting silence + the utterance
    assert pcm.size / SR < 0.3 + 2 * (0.3 + 512 / SR) + 0.6 + 0.8 + 0.05
    assert rec.first_speech_sample == pytest.approx(120 * SR, abs=512)


def test_no_speech_timeout_still_counts_from_start():
    rec = Recorder(_settings(), EnergyVAD())
    rec.start(no_speech_timeout_s=10.0)
    feed(rec, silence(12_000), block=1600)
    assert rec.reason == "no_speech"
    assert rec.end_sample == pytest.approx(10 * SR, abs=1600)
