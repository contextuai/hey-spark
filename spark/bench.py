"""Benchmarks behind ``python -m spark bench ...`` (spec §9, §10).

``stt``, ``tts`` and ``brain`` run without a microphone. ``latency`` and ``false-wake`` need the
real mic and are meant to be run by a human on the target laptop; they take injectable
components so they are tested with fakes.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spark import registry
from spark.audio.stream import Rechunker
from spark.interfaces import SAMPLE_RATE
from spark.util.latency import percentile, summarize

if TYPE_CHECKING:
    from spark.app import SparkApp
    from spark.config import Settings
    from spark.interfaces import STT, TTS, AudioInput, Brain, WakeWordDetector
    from spark.util.latency import TurnTimer

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "audio_fixtures"
BENCH_SENTENCE = "The capital of Japan is Tokyo, a city of about fourteen million people."
BENCH_PROMPT = "What's the capital of Japan?"


def stats(samples_ms: list[float]) -> dict[str, float]:
    """Median/p90/min/max of millisecond samples."""
    if not samples_ms:
        raise ValueError("no samples")
    return {
        "n": float(len(samples_ms)),
        "median": round(percentile(samples_ms, 50), 1),
        "p90": round(percentile(samples_ms, 90), 1),
        "min": round(min(samples_ms), 1),
        "max": round(max(samples_ms), 1),
    }


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def bench_inputs(fixtures: Path = FIXTURES) -> list[tuple[str, np.ndarray]]:
    """Command utterances to transcribe: recorded ones first, then synthesized fixtures."""
    from spark.audio.wavio import read_wav

    found = sorted((fixtures / "recorded").glob("cmd_*.wav")) or sorted(fixtures.glob("cmd_*.wav"))
    out = []
    for path in found:
        pcm, rate = read_wav(path)
        if rate != SAMPLE_RATE:
            from spark.audio.stream import resample

            pcm = resample(pcm, rate, SAMPLE_RATE)
        out.append((path.name, pcm))
    return out


async def bench_stt(
    settings: Settings,
    trials: int = 5,
    *,
    stt: STT | None = None,
    inputs: list[tuple[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Time ``transcribe`` on the command fixtures (``trials`` passes over at most 3 files)."""
    stt = stt or registry.build("stt", settings)
    inputs = inputs if inputs is not None else bench_inputs()
    if not inputs:
        raise FileNotFoundError(
            "no cmd_*.wav fixtures; run: uv run python tests/audio_fixtures/generate.py"
        )
    t0 = time.perf_counter()
    await stt.warmup()
    warmup_ms = _ms(time.perf_counter() - t0)
    samples: list[float] = []
    texts: dict[str, str] = {}
    for _ in range(trials):
        for name, pcm in inputs[:3]:
            t = time.perf_counter()
            texts[name] = await stt.transcribe(pcm)
            samples.append(_ms(time.perf_counter() - t))
    audio_s = [len(pcm) / SAMPLE_RATE for _, pcm in inputs[:3]]
    return {
        "backend": settings.stt.backend,
        "warmup_ms": round(warmup_ms, 1),
        "audio_s": [round(a, 2) for a in audio_s],
        "transcribe_ms": stats(samples),
        "texts": texts,
    }


async def _synthesize(tts: TTS, text: str) -> tuple[np.ndarray, int]:
    synth = getattr(tts, "synthesize", None)
    if synth is None:
        raise TypeError(f"{type(tts).__name__} has no synthesize(); cannot benchmark it")
    if inspect.iscoroutinefunction(synth):
        return await synth(text)
    return await asyncio.to_thread(synth, text)


async def bench_tts(
    settings: Settings, trials: int = 5, *, tts: TTS | None = None, text: str = BENCH_SENTENCE
) -> dict[str, Any]:
    """Time synthesis of one sentence (the first-audio cost of a reply)."""
    tts = tts or registry.build("tts", settings, player=None)
    t0 = time.perf_counter()
    await tts.warmup()
    warmup_ms = _ms(time.perf_counter() - t0)
    samples: list[float] = []
    audio_s = 0.0
    for _ in range(trials):
        t = time.perf_counter()
        pcm, rate = await _synthesize(tts, text)
        samples.append(_ms(time.perf_counter() - t))
        audio_s = len(pcm) / rate
    median = percentile(samples, 50)
    return {
        "backend": settings.tts.backend,
        "warmup_ms": round(warmup_ms, 1),
        "text_chars": len(text),
        "audio_s": round(audio_s, 2),
        "synth_ms": stats(samples),
        "real_time_factor": round(median / 1000.0 / audio_s, 3) if audio_s else None,
    }


async def bench_brain(
    settings: Settings, trials: int = 5, *, brain: Brain | None = None, prompt: str = BENCH_PROMPT
) -> dict[str, Any]:
    """Time-to-first-token and full-reply time for a short question, on a warm session."""
    brain = brain or registry.build("brain", settings)
    t0 = time.perf_counter()
    await brain.warmup()
    warmup_ms = _ms(time.perf_counter() - t0)
    ttft: list[float] = []
    total: list[float] = []
    reply = ""
    try:
        for _ in range(trials):
            t = time.perf_counter()
            first: float | None = None
            parts: list[str] = []
            async for delta in brain.respond(prompt):
                if first is None and delta:
                    first = time.perf_counter()
                parts.append(delta)
            end = time.perf_counter()
            if first is None:
                raise RuntimeError("brain returned no text")
            ttft.append(_ms(first - t))
            total.append(_ms(end - t))
            reply = "".join(parts)
    finally:
        await brain.close()
    return {
        "backend": settings.brain.backend,
        "model": settings.brain.fast_model,
        "warmup_ms": round(warmup_ms, 1),
        "ttft_ms": stats(ttft),
        "reply_ms": stats(total),
        "last_reply": reply,
    }


async def bench_latency(
    app: SparkApp,
    trials: int = 10,
    *,
    on_turn: Callable[[int, TurnTimer], None] | None = None,
) -> dict[str, Any]:
    """Run the live voice loop until ``trials`` turns complete; summarize wake->first audio."""
    records: list[dict[str, Any]] = []
    done = asyncio.Event()

    def collect(timer: TurnTimer) -> None:
        if timer.meta.get("mode") != "voice":
            return  # follow-ups have no wake mark
        records.append(timer.breakdown())
        if on_turn is not None:
            on_turn(len(records), timer)
        if len(records) >= trials:
            done.set()

    app.on_turn = collect
    runner = asyncio.create_task(app.run())
    waiter = asyncio.create_task(done.wait())
    try:
        await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        await app.stop()
        waiter.cancel()
        await asyncio.gather(runner, waiter, return_exceptions=True)
        await app.cancel_turn()
    summary = summarize(records)
    total = summary.get("total_ms")
    # speech_end -> first_audio: Spark's own response time, excluding how long the user talked
    response = [
        r["marks_ms"]["first_audio"] - r["marks_ms"]["speech_end"]
        for r in records
        if {"first_audio", "speech_end"} <= set(r.get("marks_ms", {}))
    ]
    return {
        "trials": len(records),
        "summary": summary,
        "response_ms": stats(response) if response else None,
        "median_total_ms": total["median"] if total else None,
        "target_ms": 2000.0,
        "passed": bool(total and len(records) >= trials and total["median"] < 2000.0),
    }


async def bench_false_wake(
    settings: Settings,
    minutes: float = 30.0,
    *,
    mic: AudioInput | None = None,
    wakeword: WakeWordDetector | None = None,
    on_wake: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Count wake-word detections over ``minutes`` of mic audio (measured in samples)."""
    mic = mic or registry.build("audio_in", settings)
    wakeword = wakeword or registry.build("wakeword", settings)
    chunker = Rechunker(wakeword.frame_length)
    target = int(minutes * 60 * SAMPLE_RATE)
    consumed = 0
    wakes: list[float] = []
    await mic.start()
    try:
        async for block in mic.frames():
            for frame in chunker.feed(block):
                consumed += frame.size
                if wakeword.process(frame):
                    at = consumed / SAMPLE_RATE
                    wakes.append(round(at, 2))
                    wakeword.reset()
                    if on_wake is not None:
                        on_wake(at)
            if consumed >= target:
                break
    finally:
        await mic.stop()
        wakeword.close()
    minutes_heard = consumed / SAMPLE_RATE / 60
    return {
        "backend": settings.wakeword.backend,
        "minutes": round(minutes_heard, 2),
        "wakes": len(wakes),
        "wake_times_s": wakes,
        "complete": consumed >= target,
        "passed": consumed >= target and len(wakes) < 1,
    }


async def record_fixture(
    settings: Settings,
    name: str,
    seconds: float = 4.0,
    *,
    mic: AudioInput | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Record ``seconds`` of 16 kHz mono mic audio to ``tests/audio_fixtures/recorded``."""
    from spark.audio.wavio import write_wav

    if not name or any(c in name for c in "/\\:") or name.startswith("."):
        raise ValueError(f"bad fixture name {name!r}")
    mic = mic or registry.build("audio_in", settings)
    target = int(seconds * SAMPLE_RATE)
    blocks: list[np.ndarray] = []
    got = 0
    await mic.start()
    try:
        async for block in mic.frames():
            blocks.append(np.asarray(block, dtype=np.int16).copy())
            got += len(block)
            if got >= target:
                break
    finally:
        await mic.stop()
    pcm = np.concatenate(blocks)[:target] if blocks else np.zeros(0, dtype=np.int16)
    dest = (out_dir or FIXTURES / "recorded") / f"{name.removesuffix('.wav')}.wav"
    return write_wav(dest, pcm, SAMPLE_RATE)
