"""Tiny WAV read/write helpers (stdlib ``wave`` + numpy, no extra dependencies)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from spark.audio.stream import to_int16, to_mono


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file. Returns ``(int16 mono samples, sample_rate)``."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2")
    elif width == 1:
        data = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    elif width == 4:
        data = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"{path}: unsupported sample width {width} bytes")
    if channels > 1:
        data = data.reshape(-1, channels)
    return to_int16(to_mono(data)), rate


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int = 16_000) -> Path:
    """Write mono int16 (or float in [-1, 1]) samples as a 16-bit PCM WAV. Creates parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_int16(to_mono(pcm))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data.astype("<i2").tobytes())
    return path
