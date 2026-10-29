# Hey Spark — project guide for Claude Code

`docs/design/SPEC.md` is the source of truth for **what** Hey Spark does. This file covers **how** to work in this repo. If the two conflict, the spec wins.

## Development environment
- Windows 11 (Git Bash or PowerShell) or macOS. Both ship as desktop apps: keep platform code
  behind `sys.platform` checks, and use `pathlib` and forward slashes everywhere.
  macOS-only glue lives in `spark/desktop/macos.py`.
- Python **3.11**, managed by [uv](https://docs.astral.sh/uv/). Don't call a system `python` directly; always use `uv run ...`.
- The Claude brain drives the `claude` CLI through `claude-agent-sdk`, so the CLI must be installed and logged in (or `ANTHROPIC_API_KEY` set) for live use.
- Node.js and `npx` are needed for stdio MCP servers.
- Model downloads (Whisper, Piper, ...) go to `config.models_dir` (default `./.models`, gitignored).

## Commands
```bash
uv sync --all-extras                      # install (plain `uv sync` REMOVES the desktop/oww/kokoro/... extras)
uv run pytest -q                          # all offline tests (must pass before any commit)
uv run pytest -q -m live                  # tests that hit real Claude / network (opt-in)
uv run ruff check . && uv run ruff format --check .
uv run python -m spark --text             # type-in/print-out mode, no audio
uv run python -m spark doctor             # environment checks
```
`/check` runs the full quality gate (skill: `spark-quality-gate`). Checks that need a real mic, speaker, or credentials are in `docs/dev/MANUAL_QA.md`.

## Architecture rules
- Every component (wake word, VAD, STT, TTS, brain, audio in and out) is an ABC in `spark/interfaces.py`. Backends are chosen **only** through `config.yaml` via a registry in `spark/registry.py`, never through `if backend == ...` scattered in app code. Skill: `spark-backend-pattern`.
- Heavy or optional imports (`pvporcupine`, `openwakeword`, `faster_whisper`, `piper`, `kokoro_onnx`, `deepgram`, `elevenlabs`, `sounddevice`) happen **inside** the backend's constructor or factory, never at module top level. Tests and `--text` mode must import cleanly without them.
- Audio is 16 kHz mono int16 internally. Frame sizes differ by consumer (Porcupine 512, openWakeWord 1280, Silero 512). Re-chunk from a ring buffer; never assume one size.
- The app loop is `asyncio`. Blocking CPU work (STT, TTS synthesis, wake-word inference if slow) runs in `asyncio.to_thread`. The PortAudio callback only enqueues audio.
- Secrets come only from env or `.env`. Never log them. `mcp_servers.json` values use `${VAR}` expansion.
- Side-effecting tools always go through the spoken confirmation hook (`spark/brain/confirm.py`). Default is deny.

## Dependencies
Add dependencies with `uv add`, and keep the footprint small:
- Optional backends (openWakeWord, Kokoro, Deepgram, ElevenLabs, the desktop window, ...) go in an **optional extra** (`uv add --optional <extra> <pkg>`) and are imported lazily, never as core dependencies.
- No torch (Silero VAD runs on ONNX through faster-whisper), and no new GUI toolkit or cloud-service SDK without discussing it first.
- Anything not already in `pyproject.toml` needs the maintainer's OK: ask before adding it, and give the one-line reason in the commit message.

## Testing rules
- Test with fakes (`tests/fakes.py`: FakeMic, FakeWakeWord, FakeVAD, FakeSTT, FakeTTS, MockBrain), not with real hardware.
- Tests that need model files skip cleanly with a reason when the files are absent (`pytest.mark.skipif`). Tests that need network or API keys are marked `@pytest.mark.live`.
- Keep the offline test suite under about 60 s.

## Subagents (`.claude/agents/`)
| Agent | Use for |
|---|---|
| `sdk-verifier` | Verify claude-agent-sdk / MCP / backend library APIs against the **installed** package source before writing code that depends on them |
| `audio-engineer` | `spark/audio/*`, `wakeword/*`, `vad/*`, earcons, barge-in timing |
| `speech-engineer` | `stt/*`, `tts/*`, model download/caching, latency benchmarks |
| `brain-engineer` | `brain/*`, `tools/*`, MCP config loading, allowlists, confirmation hook, routing |
| `test-engineer` | Fakes, fixtures, unit/state-machine tests, running the suite and triaging failures |
| `spec-reviewer` | Read-only audit of the codebase against `docs/design/SPEC.md` acceptance criteria, e.g. before a release |

Only run implementation agents in parallel when their files do not overlap.

## Background
Hey Spark was originally built with Claude Code in an autonomous loop, phase by phase against the spec. The original plans and acceptance reports are kept in `docs/dev/history/` for reference.
