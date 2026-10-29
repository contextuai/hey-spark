"""Tests on the committed audio fixtures (spec §11, spark-audio-fixtures skill).

* Sanity: every generated file exists, is 16 kHz mono int16, the folder is under 5 MB, and the
  silence and noise levels match what the generator promised.
* The generator itself: idempotent, and exits 0 with the download hint when the voice is missing.
* STT accuracy with the real faster-whisper model on the 10 commands (>= 9/10).
* VAD endpointing with the real Silero model on ``utterance_with_tail.wav``.
* Wake word with the real backends, only when their model / key is available.

Real recordings in ``tests/audio_fixtures/recorded/`` replace the synthetic file of the same name.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import wave
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from spark import registry
from spark.audio.recorder import Recorder
from spark.audio.stream import Rechunker, resample_to_16k
from spark.audio.wavio import read_wav
from spark.config import Settings, load_settings
from spark.interfaces import BackendUnavailable

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "audio_fixtures"
RECORDED = FIXTURES / "recorded"
MAX_TOTAL_BYTES = 5 * 1024 * 1024
HAS_FW = importlib.util.find_spec("faster_whisper") is not None
HAS_OWW = importlib.util.find_spec("openwakeword") is not None


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "spark_fixture_generate", FIXTURES / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()
META: dict = json.loads((FIXTURES / gen.META_NAME).read_text(encoding="utf-8"))


def settings() -> Settings:
    return load_settings(ROOT / "config.yaml", load_env_file=False)


def fixture_path(name: str) -> Path:
    """``recorded/<name>`` when a human recorded it, else the synthetic file."""
    recorded = RECORDED / name
    return recorded if recorded.is_file() else FIXTURES / name


def load_16k(path: Path) -> np.ndarray:
    pcm, rate = read_wav(path)
    return pcm if rate == 16_000 else resample_to_16k(pcm, rate)


def normalize(text: str) -> str:
    """Lowercase, drop hyphens/apostrophes, turn other punctuation into spaces."""
    text = re.sub(r"[-'’]", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def missing_keywords(transcript: str, keywords: list[str]) -> list[str]:
    said = normalize(transcript)
    return [kw for kw in keywords if not any(normalize(alt) in said for alt in kw.split("|"))]


# --- sanity ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", gen.expected_files())
def test_fixture_is_16k_mono_int16(name: str) -> None:
    path = FIXTURES / name
    assert path.is_file(), f"{name} missing: run  uv run python tests/audio_fixtures/generate.py"
    with wave.open(str(path), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (16_000, 1, 2)
        assert wf.getnframes() > 0
    assert name in META, f"{name} has no entry in {gen.META_NAME}"


def test_recorded_fixtures_are_16k_mono_int16() -> None:
    for path in sorted(RECORDED.glob("*.wav")):
        with wave.open(str(path), "rb") as wf:
            got = (wf.getframerate(), wf.getnchannels(), wf.getsampwidth())
        assert got == (16_000, 1, 2), f"{path.name}: {got}"


def test_total_size_under_5mb() -> None:
    total = sum(p.stat().st_size for p in FIXTURES.rglob("*.wav"))
    assert total < MAX_TOTAL_BYTES, f"audio fixtures total {total / 1e6:.2f} MB (limit 5 MB)"


def test_commands_yaml_matches_cmd_files() -> None:
    commands = gen.load_commands()
    assert len(commands) == 10
    for i, cmd in enumerate(commands, 1):
        meta = META[f"cmd_{i:02d}.wav"]
        assert meta["text"] == cmd["text"]
        assert meta["keywords"] == cmd["keywords"]
        assert cmd["keywords"], f"command {i} has no keywords"
    japan = next(c for c in commands if "Japan" in c["text"])
    assert japan["keywords"] == ["japan"]


def test_silence_is_quiet() -> None:
    pcm, _ = read_wav(FIXTURES / "silence_3s.wav")
    assert pcm.size == 3 * 16_000
    assert gen.rms_dbfs(pcm) < -55.0
    assert np.max(np.abs(pcm.astype(np.int32))) < int(32768 * 10 ** (-40 / 20))


@pytest.mark.parametrize(
    ("name", "target_dbfs"),
    [("noise_pink_10s.wav", gen.PINK_DBFS), ("babble_10s.wav", gen.BABBLE_SPEECH_DBFS)],
)
def test_noise_level_roughly_as_specified(name: str, target_dbfs: float) -> None:
    pcm, _ = read_wav(FIXTURES / name)
    assert pcm.size == 10 * 16_000
    level = gen.rms_dbfs(pcm)
    # babble adds a -35 dBFS pink bed on top of the -25 dBFS speech mix (+0.4 dB)
    assert target_dbfs - 1.5 <= level <= target_dbfs + 1.5, f"{name}: {level:.1f} dBFS"


def test_pink_noise_is_pink() -> None:
    """PSD falls 3 dB per octave (equal energy per octave, unlike white noise, which is flat):
    mean power density in 250-500 Hz is about 9 dB above 2-4 kHz, three octaves up."""
    pcm, _ = read_wav(FIXTURES / "noise_pink_10s.wav")
    spec = np.abs(np.fft.rfft(pcm.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(pcm.size, 1 / 16_000)

    def band(lo: float, hi: float) -> float:
        return float(spec[(freqs >= lo) & (freqs < hi)].mean())

    ratio_db = 10 * np.log10(band(250, 500) / band(2000, 4000))
    assert 6.0 < ratio_db < 12.0, ratio_db


def test_speech_fixtures_are_louder_than_silence() -> None:
    for name in [f"cmd_{i:02d}.wav" for i in range(1, 11)] + ["wake_hey_spark_2.wav"]:
        pcm, _ = read_wav(FIXTURES / name)
        assert gen.rms_dbfs(pcm) > -35.0, name


def test_utterance_metadata_consistent() -> None:
    meta = META["utterance_with_tail.wav"]
    pcm, _ = read_wav(FIXTURES / "utterance_with_tail.wav")
    assert meta["samples"] == pcm.size
    assert meta["speech_start_s"] == pytest.approx(gen.UTT_LEAD_S)
    assert meta["speech_end_s"] + meta["tail_silence_s"] == pytest.approx(pcm.size / 16_000)
    tail = pcm[int(meta["speech_end_s"] * 16_000) :]
    head = pcm[: int(meta["speech_start_s"] * 16_000)]
    assert gen.rms_dbfs(tail) < -55.0 and gen.rms_dbfs(head) < -55.0


# --- generator -------------------------------------------------------------------------------


def test_generator_without_voice_exits_0_with_hint(tmp_path: Path, capsys) -> None:
    out = tmp_path / "fx"
    rc = gen.main(["--out", str(out), "--voice", str(tmp_path / "missing.onnx")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "spark models download" in text
    assert sorted(p.name for p in out.glob("*.wav")) == ["noise_pink_10s.wav", "silence_3s.wav"]

    before = {p.name: p.stat().st_mtime_ns for p in out.glob("*.wav")}
    report = gen.generate(out, None)
    assert sorted(report["skipped"]) == ["noise_pink_10s.wav", "silence_3s.wav"]
    assert report["written"] == []
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.wav")} == before

    report = gen.generate(out, None, force=True)
    assert sorted(report["written"]) == ["noise_pink_10s.wav", "silence_3s.wav"]


def test_generator_is_deterministic_for_noise(tmp_path: Path) -> None:
    gen.generate(tmp_path, None)
    for name in ("silence_3s.wav", "noise_pink_10s.wav"):
        assert (tmp_path / name).read_bytes() == (FIXTURES / name).read_bytes(), name


def test_keyword_matching_normalizes() -> None:
    assert missing_keywords("Do I have any new E-mails?", ["email"]) == []
    assert missing_keywords("Set a timer for 5 minutes.", ["timer", "five|5", "minute"]) == []
    assert missing_keywords("What's the capital of Japan?", ["japan"]) == []
    assert missing_keywords("What's the capital of China?", ["japan"]) == ["japan"]


# --- STT accuracy (real faster-whisper) ------------------------------------------------------


@pytest.fixture(scope="module")
def transcripts(shared_whisper) -> dict[str, str]:
    """Transcribe the 10 commands once with the configured faster-whisper model."""
    if not HAS_FW:
        pytest.skip("faster-whisper is not installed (uv sync)")
    stt = shared_whisper  # loaded once per session (tests/conftest.py)

    async def run() -> dict[str, str]:
        out = {}
        for i in range(1, 11):
            name = f"cmd_{i:02d}.wav"
            out[name] = await stt.transcribe(load_16k(fixture_path(name)))
        return out

    return asyncio.run(run())


def test_stt_accuracy_on_commands(transcripts: dict[str, str]) -> None:
    commands = gen.load_commands()
    misses = {}
    for i, cmd in enumerate(commands, 1):
        name = f"cmd_{i:02d}.wav"
        missing = missing_keywords(transcripts[name], cmd["keywords"])
        if missing:
            misses[name] = (transcripts[name], missing)
    score = len(commands) - len(misses)
    print(f"\nSTT accuracy {score}/10")
    for name, text in transcripts.items():
        src = "recorded" if (RECORDED / name).is_file() else "synthetic"
        flag = f"  MISSING {misses[name][1]}" if name in misses else ""
        print(f"  {name} [{src}]: {text!r}{flag}")
    assert score >= 9, f"STT accuracy {score}/10, misses: {misses}"


def test_stt_capital_of_japan(transcripts: dict[str, str]) -> None:
    assert "japan" in normalize(transcripts["cmd_07.wav"])


# --- VAD endpointing (real Silero) -----------------------------------------------------------


@pytest.fixture(scope="module")
def silero():
    if not HAS_FW:
        pytest.skip("faster-whisper (bundled Silero ONNX) is not installed (uv sync)")
    from spark.vad.silero import SileroVAD

    try:
        return SileroVAD(settings())
    except BackendUnavailable as exc:
        pytest.skip(f"Silero VAD unavailable: {exc}")


@pytest.mark.parametrize("block", [512, 1600])
def test_vad_endpointing_on_utterance_with_tail(silero, block: int) -> None:
    s = settings()
    meta = META["utterance_with_tail.wav"]
    pcm, _ = read_wav(FIXTURES / "utterance_with_tail.wav")
    rec = Recorder(s, silero)
    rec.start()
    for start in range(0, pcm.size, block):
        rec.add(pcm[start : start + block])
        if rec.done:
            break
    assert rec.reason == "end_of_speech", f"recorder stopped with {rec.reason!r}"
    fired_s = rec.end_sample / 16_000
    expected_s = meta["speech_end_s"] + s.recorder.end_silence_ms / 1000
    error_ms = (fired_s - expected_s) * 1000
    print(
        f"\nVAD endpoint (block {block}): fired at {fired_s * 1000:.0f} ms, speech_end "
        f"{meta['speech_end_s'] * 1000:.0f} ms + {s.recorder.end_silence_ms} ms silence "
        f"-> error {error_ms:+.0f} ms"
    )
    assert -100.0 <= error_ms <= 300.0, f"endpoint error {error_ms:+.0f} ms"


def test_silero_hears_no_speech_in_silence(silero) -> None:
    pcm, _ = read_wav(FIXTURES / "silence_3s.wav")
    silero.reset()
    frames = Rechunker(silero.frame_length).feed(pcm)
    assert not any(silero.is_speech(f) for f in frames)


def test_silero_hears_speech_in_command(silero) -> None:
    pcm, _ = read_wav(FIXTURES / "cmd_07.wav")
    silero.reset()
    frames = Rechunker(silero.frame_length).feed(pcm)
    assert sum(silero.is_speech(f) for f in frames) * 512 / 16_000 > 0.5


# --- wake word (real backends, when available) -----------------------------------------------

WAKE_POSITIVES = ["wake_hey_spark_1.wav", "wake_hey_spark_2.wav", "wake_hey_spark_3.wav"]
WAKE_NEGATIVES = [
    "wake_negative_1.wav",
    "wake_negative_2.wav",
    "wake_negative_3.wav",
    "noise_pink_10s.wav",
    "babble_10s.wav",
    "silence_3s.wav",
]


def _fires(detector, pcm: np.ndarray) -> bool:
    detector.reset()
    fired = False
    for frame in Rechunker(detector.frame_length).feed(pcm):
        fired = detector.process(frame) or fired
    return fired


def _build_wakeword(name: str, s: Settings):
    try:
        return registry.build("wakeword", s, name=name)
    except BackendUnavailable as exc:
        pytest.skip(f"{name} unavailable: {exc}")


@pytest.fixture(scope="module")
def oww():
    if not HAS_OWW:
        pytest.skip("openwakeword is not installed (optional backend, not in the venv)")
    s = settings()
    detector = _build_wakeword("openwakeword", s)
    yield s, detector
    detector.close()


@pytest.fixture(scope="module")
def porcupine():
    from dotenv import dotenv_values

    s = settings()
    key_env = s.wakeword.options.porcupine.access_key_env
    with pytest.MonkeyPatch.context() as mp:
        # the AccessKey may live in .env; expose it only for this module's detector
        env_key = dotenv_values(ROOT / ".env").get(key_env) if (ROOT / ".env").is_file() else None
        if env_key:
            mp.setenv(key_env, env_key)
        detector = _build_wakeword("porcupine", s)
        yield detector
        detector.close()


@pytest.mark.parametrize("name", WAKE_NEGATIVES)
def test_openwakeword_rejects_negatives(oww, name: str) -> None:
    _s, detector = oww
    assert not _fires(detector, load_16k(fixture_path(name))), f"false wake on {name}"


@pytest.mark.parametrize("name", WAKE_POSITIVES)
def test_openwakeword_accepts_hey_spark(oww, name: str) -> None:
    s, detector = oww
    model_path = s.wakeword.options.openwakeword.model_path
    if not model_path:
        pytest.skip(
            "no custom 'Hey Spark' openWakeWord model (model_path is null; the hey_jarvis "
            "placeholder cannot recognize 'Hey Spark')"
        )
    if "hey_spark" not in Path(model_path).stem.lower():
        pytest.skip(f"wake model {Path(model_path).name} isn't 'Hey Spark' (the positives are)")
    assert _fires(detector, load_16k(fixture_path(name))), f"missed wake on {name}"


@pytest.mark.parametrize("name", WAKE_NEGATIVES)
def test_porcupine_rejects_negatives(porcupine, name: str) -> None:
    assert not _fires(porcupine, load_16k(fixture_path(name))), f"false wake on {name}"


@pytest.mark.parametrize("name", WAKE_POSITIVES)
def test_porcupine_accepts_hey_spark(porcupine, name: str) -> None:
    assert _fires(porcupine, load_16k(fixture_path(name))), f"missed wake on {name}"
