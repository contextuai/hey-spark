# Phase 1 report — 2026-09-24

Voice round-trip with no connectors (SPARK_SPEC §10, Phase 1). Built by the Ralph loop on branch `build/phases-1-3`.
Audited by `spec-reviewer` in two rounds. Round 1 found 2 FAILs (both fixed, see "Known issues and fixes"). Round 2 verdict: **CODE-COMPLETE** (0 FAIL). Its one non-blocking finding, a failed priming turn during an in-turn reconnect erroring the user's turn, is fixed: priming now runs only at warm-up/reset.

## Acceptance checklist
| Criterion (SPARK_SPEC §10) | Status | Evidence / how to verify |
|---|---|---|
| Mic, wake word, VAD, STT, and Claude (no tools) feed Piper TTS | PASS (code) | `spark/app.py` builds every component through `spark/registry.py`. The brain runs with `tools=[]`, `setting_sources=[]`, `strict_mcp_config=True`. Tests: `tests/test_app.py::test_full_cycle_idle_to_follow_up_to_idle`, `tests/test_brain_claude_agent.py::test_build_options` |
| Earcons (wake chime, thinking tick, error tone) | PASS (code) / NEEDS-HUMAN (audible) | `assets/sounds/*.wav`, `app.py` `_earcon`. Tests assert `["wake","thinking"]` and `"error"` on STT and brain failures. Listening check: HUMAN_CHECKLIST "Earcons are audible" |
| FOLLOW_UP window (8 s, no wake word needed) | PASS (code) | `app.follow_up_s: 8.0`. Tests: still FOLLOW_UP at 7.5 s, IDLE at 8.1 s; a follow-up question is answered without the wake word |
| Config-driven backends | PASS | `spark/registry.py` (string paths, lazy imports). No `if backend ==` anywhere in app code |
| "Hey Spark, what's the capital of Japan?" gets a spoken answer | NEEDS-HUMAN | Run `uv run python -m spark`. Supporting evidence: the live brain test `pytest -m live -k capital` passed on this laptop ("Tokyo is the capital of Japan."). The Piper→whisper round-trip test transcribes "…capital of Japan". The wake word can't run without `PICOVOICE_ACCESS_KEY` and the `.ppn` |
| Median wake-to-first-audio < 2 s over 10 trials, breakdown logged | NEEDS-HUMAN | `uv run python -m spark bench latency --trials 10` (tested with fakes). It prints the per-stage breakdown and end-of-speech→first-audio. Per-turn breakdown goes to `logs/latency.jsonl`. **Risk: likely over 2 s with local STT; see Latency** |
| `python -m spark doctor` passes on the target laptop | NEEDS-HUMAN | Observed here on 2026-09-24: all required checks ✔ (mic: Echo Cancelling Speakerphone via WASAPI at 16 kHz, speaker, models, claude CLI 2.1.281, brain reachable), except ✖ Porcupine: `PICOVOICE_ACCESS_KEY` not set. Exit code 1 until the key and `.ppn` are added |
| < 1 false wake in 30 min of TV/background noise | NEEDS-HUMAN | `uv run python -m spark bench false-wake --minutes 30` (tested with fakes; counts by audio time) |
| STT and TTS swappable in `config.yaml` without code changes | PASS | `tests/test_swap.py` builds the app from two config files that differ only in `stt.backend`/`tts.backend` (and via `SPARK_STT__BACKEND` env); all real backend names resolve |
| §11: `--text`, `--no-tts`, `--debug-audio` (off by default) | PASS | `--text` prints only (no audio devices opened; `--speak` opts in). `--no-tts` selects the `print` TTS backend. `--debug-audio` writes `debug_audio/last_utterance.wav`. Tests: `tests/test_cli.py`, `tests/test_app.py::test_debug_audio_saved_only_when_enabled` |
| §11: VAD endpointing and STT accuracy on wav fixtures | PASS | Synthesized fixtures (`tests/audio_fixtures/`, 1.9 MB). whisper `base.en` 9/10 commands (miss: "pull requests" → "poll requests"). Real Silero end-of-speech fired +2 ms from the expected sample |
| §11: wake-word positive/negative fixtures | NEEDS-HUMAN | The tests exist (`tests/test_fixtures_audio.py`) but skip without the Porcupine key and `.ppn` (or the openWakeWord extra plus a custom model) |
| §11: MockBrain state-machine tests, plus a recorded-transcript no-markdown test | PASS | `tests/test_app.py`. `tests/test_brain_prompts.py` recorded-transcript test. `test_fenced_code_is_never_spoken` |
| §12: secrets never logged, no audio stored by default, transcript logging can be disabled | PASS | Redaction filter (`spark/util/logging.py`) with tests. Porcupine key scrubbed from its own error text. `debug_audio.enabled: false`. `logging.transcripts` toggle with tests |

## Latency (this machine, no mic)
Measured on 2026-09-24 on a mid-range Windows laptop (CPU only). All medians are real measurements from the listed commands. Nothing is estimated.

| Step | Median | p90 | Budget (§9) | Notes |
|---|---|---|---|---|
| End-of-speech detection | 700 ms (config) | — | ~700 ms | `recorder.end_silence_ms`. Real Silero fired within +2 ms of the target on the fixture |
| STT (faster-whisper `base.en` int8, 1.6–2.4 s utterances) | 707 ms | 751 ms | < 600 ms | `spark bench stt --trials 5` (n=15). **Over budget.** A run under heavy CPU load during P1.6 measured a 1322 ms median. Warm-up 1.3 s |
| Brain time-to-first-token (Haiku 4.5, warm session) | 1037 ms | 1352 ms | < 600 ms | `spark bench brain --trials 5`, after adding `brain.prime_on_connect`. Before priming: median 2153 ms, p90 3969 ms (the first turn after connect cost ~4–5 s extra). **Over budget.** Connect/warm-up 5–13 s at startup |
| TTS first chunk (Piper `en_US-lessac-medium`) | 334 ms | 349 ms | < 200 ms | `spark bench tts --trials 5`, on a 71-character sentence (4.45 s of audio, real-time factor 0.075). A shorter 1.9 s first sentence measured a 254 ms median (P1.6). **Over budget**; the synthesis cost scales with sentence length |
| VAD per 512-sample frame (Silero ONNX) | 0.43 ms | — | — | average over 500 frames |
| Barge-in / playback stop | ~0.25 ms to `stop()`; first silent block ~16 ms | — | < 300 ms (Phase 3) | fake 20 ms output device (P1.4 tests) |
| **Wake → first audio (end to end)** | not measured | — | < 2000 ms | Needs the mic plus the wake word: `spark bench latency --trials 10` (human) |

The sum of the component medians *after the user stops speaking* is 700 + 707 + 1037 + 334 ≈ 2.8 s. That is arithmetic on the rows above, not an end-to-end measurement, and it excludes the time the user spends speaking (which wake→first-audio includes). On this hardware the 2 s Phase 1 target is therefore **unlikely to be met with local STT**. The levers, in order of expected impact: Deepgram STT (cloud, post-wake audio only), whisper `tiny.en`, a shorter `end_silence_ms` (e.g. 500 ms), and Piper's `-low` voice. This is the spec §13 open decision, and the human checklist has the exact comparison commands.

## What was built
- `spark/interfaces.py` (ABCs plus `BackendUnavailable`), `spark/registry.py` (string-path registry, `Cls(settings, **deps)`), and `spark/config.py` (pydantic settings: `config.yaml` < `.env` < `SPARK_*` env < CLI overrides). `config.yaml` has commented defaults for every key.
- Audio: `SoundDeviceMic` (WASAPI, 16 kHz, bounded queue, resample fallback), `RingBuffer`/`Rechunker` (512/1280 frames from any block size), an interruptible `SoundDevicePlayer` with generated earcons, and a `Recorder` (pre-roll, sample-timed endpointing, max length, no-speech timeout).
- Backends: Silero VAD (faster-whisper's bundled ONNX, streaming state, no torch), Porcupine (default) and openWakeWord (extra), faster-whisper (default) and Deepgram (extra), Piper (default), Kokoro ONNX and ElevenLabs (extras), and a `print` TTS for `--no-tts`/`--text`.
- Brain: `ClaudeAgentBrain` (one worker task owns a long-lived `ClaudeSDKClient`; partial streaming; idle reset after 10 min; "new topic"; interrupt; session priming), a voice system prompt, and `sanitize_for_speech` plus the fence-aware `SpeechFilter`. `MockBrain` for tests and offline smoke runs.
- App: an asyncio state machine (IDLE→LISTENING→THINKING→SPEAKING→FOLLOW_UP) whose mic loop never blocks, with earcons, the 8 s follow-up window, errors → error tone → IDLE, a barge-in hook (off until Phase 3), a per-turn `TurnTimer` → `logs/latency.jsonl`, rotating transcripts, and `--debug-audio`.
- CLI: `python -m spark` with `run`/`--text`/`--speak`/`--no-tts`/`--debug-audio`/backend overrides, `doctor`, `models download`, `bench stt|tts|brain|latency|false-wake`, and `record-fixture`.
- Tests: 570+ offline tests (about 45 s), plus live tests (`-m live`) and hardware tests (`-m hardware`).

## Deviations from spec (and why)
- **STT runs in THINKING, not LISTENING** (§6 sketch). The mic loop hands each finished utterance to a turn task so it never blocks (needed later for barge-in). Because of this, the thinking tick plays before STT.
- **Deepgram uses the pre-recorded REST API, not streaming** (§4). The utterance is already endpointed locally, so streaming would add complexity for little gain in Phase 1.
- **Wake-word fixtures are synthesized with Piper, not recorded** (§11). The loop can't record a human; `record-fixture` plus `tests/audio_fixtures/recorded/` let a human add real ones, and the tests prefer them automatically.
- **`--text` gained `--speak`.** Plain `--text` never opens audio devices, per §11 and CLAUDE.md.
- **Brain session priming** (`brain.prime_on_connect`, on by default) adds one throwaway exchange ("OK") to each new conversation's history, to move a measured ~4–5 s first-turn cost into warm-up.

## Open decisions touched (§13)
- **STT: local `base.en` vs `small.en` vs Deepgram.** `base.en` already misses the 600 ms budget on this CPU (707 ms idle, 1.3 s loaded), so `small.en` is ruled out. Deepgram or `tiny.en` need to be evaluated with the human latency run.
- **TTS voice:** Piper `en_US-lessac-medium` misses 200 ms on long first sentences. Kokoro wasn't benchmarked (the extra isn't installed, and the model wasn't downloaded because of disk constraints).
- **Wake word "Hey Spark":** needs Picovoice Console training (human).

## Known issues and fixes
- Round-1 audit FAIL (fixed): fenced code blocks could reach TTS. The app now uses a per-turn `SpeechFilter` (test `test_fenced_code_is_never_spoken`).
- Round-1 audit FAIL (fixed): `--text` opened the speaker. It now prints only, with `--speak` to opt in (tests in `tests/test_cli.py`).
- `first_audio` is marked when audio is handed to the output stream, not when the device actually plays it, so it's optimistic by the output buffer (~20 ms blocks).
- The output stream takes ~0.64 s to open on this laptop. It's opened at warm-up.
- The first turn after each brain reconnect (idle reset, "new topic") pays the priming turn (~4 s) inside that reset.
- openWakeWord's placeholder `hey_jarvis` model can't recognise "Hey Spark". A custom model is needed for that backend.
- The inline triple-backtick case ("Run ```echo hi``` then relax") still speaks the code's words. Cosmetic.
- Raw IDs (e.g. `01HXYZABCDEF`) aren't stripped yet. That's a Phase 2 criterion, handled in P2.4.

## Verdict
**CODE-COMPLETE.** Every Phase 1 row is PASS or NEEDS-HUMAN; none is FAIL.
