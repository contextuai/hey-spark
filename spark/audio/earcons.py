"""Generate Spark's earcons (wake chime, thinking tick, error tone) as 16 kHz mono int16 WAVs.

Regenerate with ``uv run python -m spark.audio.earcons [out_dir]`` (default ``assets/sounds``).
Every sound has raised-cosine fades so it starts and ends without a click.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from spark.audio.stream import to_int16
from spark.audio.wavio import write_wav
from spark.interfaces import SAMPLE_RATE

__all__ = ["error_tone", "generate", "sleep_chime", "thinking_tick", "wake_chime"]


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE


def _fade(x: np.ndarray, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    y = x.copy()
    for ms, sl in ((fade_in_ms, slice(None)), (fade_out_ms, slice(None, None, -1))):
        n = min(int(ms * SAMPLE_RATE / 1000), y.size // 2)
        if n > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
            view = y[sl]
            view[:n] *= ramp
    return y


def _tone(freq: float, seconds: float, harmonics: tuple[float, ...] = (1.0, 0.25)) -> np.ndarray:
    t = _t(seconds)
    out = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    return np.asarray(out) / sum(harmonics)


def wake_chime() -> np.ndarray:
    """Rising two-note chime (E5 → A5), ~220 ms, bell-like decay."""
    first = _tone(659.25, 0.11) * np.exp(-_t(0.11) * 12)
    second = _tone(880.0, 0.16) * np.exp(-_t(0.16) * 10)
    out = np.zeros(int(0.22 * SAMPLE_RATE))
    out[: first.size] += _fade(first, 4, 15)
    start = int(0.06 * SAMPLE_RATE)
    out[start : start + second.size] += _fade(second, 4, 40)
    return to_int16(0.6 * out / np.max(np.abs(out)))


def sleep_chime() -> np.ndarray:
    """Soft falling two-note chime (A5 → E5), the wake chime in reverse and quieter: Spark
    stopped listening without the wake word."""
    first = _tone(880.0, 0.12) * np.exp(-_t(0.12) * 12)
    second = _tone(659.25, 0.2) * np.exp(-_t(0.2) * 9)
    out = np.zeros(int(0.28 * SAMPLE_RATE))
    out[: first.size] += _fade(first, 6, 20)
    start = int(0.08 * SAMPLE_RATE)
    out[start : start + second.size] += _fade(second, 6, 60)
    return to_int16(0.4 * out / np.max(np.abs(out)))


def thinking_tick() -> np.ndarray:
    """Soft, short tick (~60 ms): a 1.2 kHz blip with a fast decay, quieter than the chime."""
    x = _tone(1200.0, 0.06, (1.0,)) * np.exp(-_t(0.06) * 70)
    return to_int16(0.35 * _fade(x, 2, 20))


def error_tone() -> np.ndarray:
    """Low descending tone (~300 ms), a glide from 440 Hz to 220 Hz."""
    t = _t(0.3)
    freq = 440.0 * (0.5 ** (t / t[-1]))  # exponential glide to one octave down
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    x = (np.sin(phase) + 0.3 * np.sin(2 * phase)) / 1.3
    return to_int16(0.55 * _fade(x, 8, 80))


def generate(out_dir: str | Path = "assets/sounds") -> list[Path]:
    """Write wake.wav, thinking.wav, error.wav and sleep.wav into ``out_dir``."""
    out = Path(out_dir)
    sounds = {
        "wake": wake_chime(),
        "thinking": thinking_tick(),
        "error": error_tone(),
        "sleep": sleep_chime(),
    }
    return [write_wav(out / f"{name}.wav", pcm, SAMPLE_RATE) for name, pcm in sounds.items()]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    for path in generate(args[0] if args else "assets/sounds"):
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
