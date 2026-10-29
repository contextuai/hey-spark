# Audio fixtures

Test audio for Spark: 16 kHz, mono, 16-bit PCM WAV. Keep the whole folder under about 5 MB.

## Synthetic fixtures (committed)
`generate.py` builds them with the Piper voice (`.models/piper/en_US-lessac-medium.onnx`) plus
seeded numpy noise, so no human has to record anything:

```bash
uv run python tests/audio_fixtures/generate.py           # create missing files only
uv run python tests/audio_fixtures/generate.py --force   # regenerate everything
```

If the voice is missing, the script writes only the silence and noise files, prints the
`uv run python -m spark models download` hint, and exits 0. The wavs are committed, so tests
never need Piper.

| File | Content |
|---|---|
| `silence_3s.wav` | −60 dBFS dither |
| `noise_pink_10s.wav` | pink noise at −30 dBFS |
| `babble_10s.wav` | 4 overlapping sentences at −25 dBFS plus pink noise at −35 dBFS (TV-like) |
| `wake_hey_spark_{1..3}.wav` | "Hey Spark" at length_scale 0.9 / 1.0 / 1.15, 0.5 s padding |
| `wake_negative_{1..3}.wav` | "Hey Mark", "Hey Sam", "Park the car" |
| `cmd_{01..10}.wav` | the commands in `commands.yaml` (text plus expected keywords) |
| `utterance_with_tail.wav` | 0.3 s silence, a command, 1.5 s silence (VAD endpointing) |

`fixtures.json` is written by the generator and records each file's text, levels, and, for
`utterance_with_tail.wav`, the exact speech start and end times the endpointing test uses.

## Real recordings (preferred)
A synthesized voice is a weak proxy for a real person in a real room. Record your own with:

```bash
uv run python -m spark record-fixture cmd_07 --seconds 4
```

That writes `tests/audio_fixtures/recorded/cmd_07.wav` (16 kHz mono from the configured mic).
Use the same names as the synthetic files (`cmd_01` .. `cmd_10`, saying the text from
`commands.yaml`). When `recorded/<name>.wav` exists, the tests use it instead of the synthetic
file. Recordings count toward the 5 MB budget, so keep them short.
