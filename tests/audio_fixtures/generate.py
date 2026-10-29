"""Generate Spark's synthetic audio fixtures (spark-audio-fixtures skill, spec §11).

Usage::

    uv run python tests/audio_fixtures/generate.py            # create missing files only
    uv run python tests/audio_fixtures/generate.py --force    # regenerate everything

Every file is 16 kHz mono int16 PCM. Speech comes from the Piper voice configured in
``config.yaml`` (``<models_dir>/piper/<voice>.onnx``, 22.05 kHz, resampled with
:func:`spark.audio.stream.resample`); silence and noise come from a seeded numpy RNG. Facts the
tests need (speech start/end in ``utterance_with_tail.wav``, target levels, the text spoken) are
written to the ``fixtures.json`` sidecar.

Without the Piper voice the noise and silence files are still written, the speech files are
skipped, a ``spark models download`` hint is printed, and the exit code is 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allow ``python tests/audio_fixtures/generate.py`` from anywhere
    sys.path.insert(0, str(ROOT))

from spark.audio.stream import resample, to_int16  # noqa: E402
from spark.audio.wavio import write_wav  # noqa: E402

HERE = Path(__file__).resolve().parent
COMMANDS_FILE = HERE / "commands.yaml"
META_NAME = "fixtures.json"
SR = 16_000
FULL_SCALE = 32768.0

SILENCE_DBFS = -60.0  # dither level for "silent" stretches
PINK_DBFS = -30.0
BABBLE_SPEECH_DBFS = -25.0
BABBLE_PINK_DBFS = -35.0
SPEECH_PEAK_DBFS = -3.0
WAKE_LENGTH_SCALES = (0.9, 1.0, 1.15)
WAKE_NEGATIVES = ("Hey Mark", "Hey Sam", "Park the car")
WAKE_PAD_S = 0.5
CMD_LEAD_S = 0.3
CMD_TAIL_S = 0.5
UTT_LEAD_S = 0.3
UTT_TAIL_S = 1.5
UTT_COMMAND_INDEX = 7  # "What's the capital of Japan?" (1-based index into commands.yaml)
BABBLE_SENTENCES = (
    "And in other news tonight, the city council voted to approve the new budget.",
    "The home team scored twice in the second half to take the lead.",
    "Stay tuned after the break for the full weather forecast for the weekend.",
    "Researchers say the new battery design could double the range of electric cars.",
)
DOWNLOAD_HINT = "uv run python -m spark models download"

Synth = Callable[[str, float], np.ndarray]


# --- levels and signals ----------------------------------------------------------------------


def dbfs_to_amp(dbfs: float) -> float:
    """Amplitude as a fraction of int16 full scale for a level in dBFS."""
    return float(10.0 ** (dbfs / 20.0))


def rms_dbfs(pcm: np.ndarray) -> float:
    """RMS level of int16 (or [-1, 1] float) audio in dBFS; -inf for digital silence."""
    x = np.asarray(pcm, dtype=np.float64)
    if np.asarray(pcm).dtype == np.int16:
        x = x / FULL_SCALE
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")


def set_rms(x: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale float audio to the given RMS level (dBFS)."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    return x if rms == 0 else x * (dbfs_to_amp(dbfs) / rms)


def dither(seconds: float, rng: np.random.Generator, dbfs: float = SILENCE_DBFS) -> np.ndarray:
    """Near-silence: Gaussian dither at ``dbfs`` RMS (float in [-1, 1])."""
    return rng.normal(0.0, dbfs_to_amp(dbfs), round(seconds * SR))


def pink_noise(seconds: float, rng: np.random.Generator, dbfs: float = PINK_DBFS) -> np.ndarray:
    """1/f noise (spectral shaping of white Gaussian noise) at ``dbfs`` RMS, float."""
    n = round(seconds * SR)
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    shape = np.zeros_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])  # power ~ 1/f, DC removed
    return set_rms(np.fft.irfft(spectrum * shape, n=n), dbfs)


def trim_speech(x: np.ndarray, floor_db: float = 40.0) -> np.ndarray:
    """Drop leading/trailing near-silence from TTS output (10 ms frames, ``floor_db`` below the
    loudest frame, never above -50 dBFS absolute), so speech start/end are known exactly."""
    x = np.asarray(x, dtype=np.float64)
    hop = SR // 100
    n = x.size // hop
    if n == 0:
        return x
    frames = x[: n * hop].reshape(n, hop)
    level = 20 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12)
    loud = np.flatnonzero(level > min(level.max() - floor_db, -50.0))
    if loud.size == 0:
        return x[:0]
    return x[loud[0] * hop : (loud[-1] + 1) * hop]


def normalize_peak(x: np.ndarray, dbfs: float = SPEECH_PEAK_DBFS) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x if peak == 0 else x * (dbfs_to_amp(dbfs) / peak)


# --- Piper -----------------------------------------------------------------------------------


def default_voice_path() -> Path:
    """``<models_dir>/piper/<voice>.onnx`` from the repo's ``config.yaml``."""
    from spark.config import load_settings
    from spark.models import piper_voice_paths

    settings = load_settings(ROOT / "config.yaml", load_env_file=False)
    return piper_voice_paths(settings)[0]


def piper_synth(voice_path: Path) -> Synth:
    """Load the Piper voice once; return ``synth(text, length_scale) -> 16 kHz float audio``."""
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(voice_path))

    def synth(text: str, length_scale: float = 1.0) -> np.ndarray:
        cfg = SynthesisConfig(length_scale=length_scale)
        parts = [
            resample(chunk.audio_int16_array, chunk.sample_rate, SR)
            for chunk in voice.synthesize(text, cfg)
        ]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return normalize_peak(trim_speech(audio))

    return synth


# --- fixture builders ------------------------------------------------------------------------


def load_commands(path: Path = COMMANDS_FILE) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    commands = data["commands"]
    if len(commands) != 10:
        raise ValueError(f"{path}: expected 10 commands, found {len(commands)}")
    return commands


def build_silence(_synth: Synth | None) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(1)
    return dither(3.0, rng), {"kind": "silence", "dither_dbfs": SILENCE_DBFS}


def build_pink(_synth: Synth | None) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(2)
    return pink_noise(10.0, rng), {"kind": "noise", "rms_dbfs": PINK_DBFS, "seed": 2}


def build_babble(synth: Synth) -> tuple[np.ndarray, dict[str, Any]]:
    """Four sentences looping at staggered offsets (TV-like), plus a pink-noise bed."""
    rng = np.random.default_rng(3)
    n = 10 * SR
    mix = np.zeros(n)
    for i, text in enumerate(BABBLE_SENTENCES):
        voice = synth(text, (0.9, 1.0, 1.1, 1.2)[i])
        gap = np.zeros(int(0.4 * SR))
        track = np.concatenate([np.zeros(int((0.2 + 1.3 * i) * SR))] + [voice, gap] * 6)[:n]
        mix[: track.size] += track
    mix = set_rms(mix, BABBLE_SPEECH_DBFS) + pink_noise(10.0, rng, BABBLE_PINK_DBFS)
    meta = {
        "kind": "babble",
        "speech_rms_dbfs": BABBLE_SPEECH_DBFS,
        "pink_rms_dbfs": BABBLE_PINK_DBFS,
        "sentences": list(BABBLE_SENTENCES),
    }
    return mix, meta


def padded(speech: np.ndarray, lead_s: float, tail_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    body = speech + dither(speech.size / SR, rng)
    return np.concatenate([dither(lead_s, rng), body, dither(tail_s, rng)])


def build_wake(i: int) -> Callable[[Synth], tuple[np.ndarray, dict[str, Any]]]:
    scale = WAKE_LENGTH_SCALES[i - 1]

    def build(synth: Synth) -> tuple[np.ndarray, dict[str, Any]]:
        speech = synth("Hey Spark", scale)
        meta = {"kind": "wake_positive", "text": "Hey Spark", "length_scale": scale}
        return padded(speech, WAKE_PAD_S, WAKE_PAD_S, 10 + i), meta

    return build


def build_wake_negative(i: int) -> Callable[[Synth], tuple[np.ndarray, dict[str, Any]]]:
    text = WAKE_NEGATIVES[i - 1]

    def build(synth: Synth) -> tuple[np.ndarray, dict[str, Any]]:
        speech = synth(text, 1.0)
        meta = {"kind": "wake_negative", "text": text}
        return padded(speech, WAKE_PAD_S, WAKE_PAD_S, 20 + i), meta

    return build


def build_command(i: int, cmd: dict[str, Any]) -> Callable[[Synth], tuple[np.ndarray, dict]]:
    def build(synth: Synth) -> tuple[np.ndarray, dict[str, Any]]:
        speech = synth(cmd["text"], 1.0)
        meta = {"kind": "command", "text": cmd["text"], "keywords": list(cmd["keywords"])}
        return padded(speech, CMD_LEAD_S, CMD_TAIL_S, 30 + i), meta

    return build


def build_utterance(cmd: dict[str, Any]) -> Callable[[Synth], tuple[np.ndarray, dict]]:
    def build(synth: Synth) -> tuple[np.ndarray, dict[str, Any]]:
        speech = synth(cmd["text"], 1.0)
        lead = round(UTT_LEAD_S * SR)
        meta = {
            "kind": "utterance_with_tail",
            "text": cmd["text"],
            "lead_silence_s": UTT_LEAD_S,
            "tail_silence_s": UTT_TAIL_S,
            "speech_start_s": lead / SR,
            "speech_end_s": (lead + speech.size) / SR,
        }
        return padded(speech, UTT_LEAD_S, UTT_TAIL_S, 50), meta

    return build


def fixture_plan() -> list[tuple[str, bool, Callable[..., tuple[np.ndarray, dict[str, Any]]]]]:
    """``(file name, needs_voice, builder)`` for every generated fixture."""
    commands = load_commands()
    plan: list[tuple[str, bool, Callable[..., tuple[np.ndarray, dict[str, Any]]]]] = [
        ("silence_3s.wav", False, build_silence),
        ("noise_pink_10s.wav", False, build_pink),
        ("babble_10s.wav", True, build_babble),
    ]
    plan += [(f"wake_hey_spark_{i}.wav", True, build_wake(i)) for i in (1, 2, 3)]
    plan += [(f"wake_negative_{i}.wav", True, build_wake_negative(i)) for i in (1, 2, 3)]
    plan += [(f"cmd_{i:02d}.wav", True, build_command(i, cmd)) for i, cmd in enumerate(commands, 1)]
    plan.append(("utterance_with_tail.wav", True, build_utterance(commands[UTT_COMMAND_INDEX - 1])))
    return plan


def expected_files() -> list[str]:
    return [name for name, _needs_voice, _build in fixture_plan()]


# --- main ------------------------------------------------------------------------------------


def generate(out_dir: Path, voice_path: Path | None, force: bool = False) -> dict[str, list[str]]:
    """Write missing (or, with ``force``, all) fixtures into ``out_dir``.

    Returns ``{"written": [...], "skipped": [...], "needs_voice": [...]}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / META_NAME
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    have_voice = voice_path is not None and voice_path.is_file()
    synth: Synth | None = None
    report: dict[str, list[str]] = {"written": [], "skipped": [], "needs_voice": []}

    for name, needs_voice, build in fixture_plan():
        path = out_dir / name
        if path.is_file() and name in meta and not force:
            report["skipped"].append(name)
            continue
        if needs_voice and not have_voice:
            report["needs_voice"].append(name)
            continue
        if needs_voice and synth is None:
            assert voice_path is not None
            print(f"loading Piper voice {voice_path.name} ...", flush=True)
            synth = piper_synth(voice_path)
        audio, info = build(synth)
        pcm = to_int16(np.clip(audio, -1.0, 1.0))
        write_wav(path, pcm, SR)
        info.update(
            {"samples": int(pcm.size), "duration_s": pcm.size / SR, "rms_dbfs": rms_dbfs(pcm)}
        )
        meta[name] = info
        report["written"].append(name)
        print(f"  wrote {name:26s} {pcm.size / SR:5.2f} s  {path.stat().st_size / 1024:6.1f} KB")

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="regenerate files that exist")
    parser.add_argument("--out", type=Path, default=HERE, help="output directory")
    parser.add_argument("--voice", type=Path, default=None, help="Piper .onnx (default: config)")
    args = parser.parse_args(argv)

    voice = args.voice if args.voice is not None else default_voice_path()
    report = generate(args.out, voice, force=args.force)
    total = sum(p.stat().st_size for p in args.out.glob("*.wav"))
    print(
        f"{len(report['written'])} written, {len(report['skipped'])} already present, "
        f"total wav size {total / 1e6:.2f} MB"
    )
    if report["needs_voice"]:
        print(
            f"Piper voice not found at {voice}; skipped {len(report['needs_voice'])} speech "
            f"fixtures. Download it with:  {DOWNLOAD_HINT}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
