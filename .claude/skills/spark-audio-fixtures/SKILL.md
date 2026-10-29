---
name: spark-audio-fixtures
description: Generate and manage Spark's test audio fixtures (wake-word positives/negatives, 10 sample commands, silence, babble/TV-like noise) without recording a human — synthesized with Piper plus numpy noise — and keep them small. Use when creating tests/audio_fixtures or any test that needs real audio.
---

# Audio fixtures for Spark

The spec (§11) asks for recorded wav fixtures. Automated tooling can't record a human, so we **synthesize** them, and a person can later drop real recordings into `tests/audio_fixtures/recorded/` (the tests pick those up automatically when present).

## Generator: `tests/audio_fixtures/generate.py`
Run with `uv run python tests/audio_fixtures/generate.py`. It's idempotent: skip files that already exist, `--force` regenerates them. It needs the Piper voice in `models_dir`. If the voice is missing, print the `spark models download` hint and exit 0.

Produce **16 kHz mono int16 wav** (resample Piper's 22.05 kHz output with numpy linear or polyphase interpolation):
| File | Content |
|---|---|
| `silence_3s.wav` | Zeros plus −60 dBFS dither |
| `noise_pink_10s.wav` | Pink noise at −30 dBFS (seeded RNG) |
| `babble_10s.wav` | 4 overlapping Piper sentences at −25 dBFS plus pink noise (a TV-like negative) |
| `wake_hey_spark_{1..3}.wav` | "Hey Spark" at 3 lengths/speeds (Piper `length_scale` 0.9/1.0/1.15) plus 0.5 s of silence padding |
| `wake_negative_{1..3}.wav` | "Hey Mark", "Hey Sam", "Park the car" |
| `cmd_{01..10}.wav` | The 10 sample commands in `commands.yaml` (text plus expected keywords), e.g. "what's the capital of Japan" → ["japan"] |
| `utterance_with_tail.wav` | 0.3 s silence + command + 1.5 s silence (for VAD endpointing) |

Keep the total under about 5 MB. Commit the generated wavs **and** the generator (so the tests don't need Piper at test time).

## Test usage
- STT accuracy: transcribe each `cmd_*.wav` and require every expected keyword in ≥ 9/10 (skip if the whisper model is missing).
- VAD endpointing: on `utterance_with_tail.wav`, end-of-speech fires between speech_end + silence_ms − 100 ms and + 300 ms.
- Wake word: positives must fire and negatives, noise, and babble must not. Skip if the model or key is missing. Note that a synthesized voice is a weak proxy, so the real false-wake check lives in the manual QA checklist, `docs/dev/MANUAL_QA.md` (`spark bench false-wake`).

## Real recordings (human)
`python -m spark record-fixture <name>` records 16 kHz mono from the mic into `tests/audio_fixtures/recorded/<name>.wav`. Tests prefer recorded files when they exist.
