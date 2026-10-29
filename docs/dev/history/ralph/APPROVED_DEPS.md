# Pre-approved dependencies

The spec (§14.3) says to ask before adding unlisted dependencies. The Ralph loop can't ask, so this list is the approval.
Add each one with `uv add` at the point it's first needed, and write the one-line reason in the commit message.
Anything not listed here → do NOT add. Record it under "Blocked / needs human" in PLAN.md.

## Core (`uv add`)
| Package | Why |
|---|---|
| `claude-agent-sdk` | Brain (§8) |
| `pydantic`, `pydantic-settings` | Config models (§5 config.py) |
| `pyyaml` | `config.yaml` |
| `python-dotenv` | `.env` loading |
| `numpy` | PCM buffers, resampling, RMS |
| `sounddevice` | Mic and speaker I/O (WASAPI) |
| `rich` | Terminal logging and status (§1 non-goals: terminal only) |
| `faster-whisper` | Default STT. Also ships the **Silero VAD ONNX** model (`faster_whisper.vad`) plus `onnxruntime`, which we use for VAD so we don't pull in torch (a disk constraint) |
| `piper-tts` | Default TTS |
| `pvporcupine` | Default wake word |
| `huggingface-hub` | Model download into `models_dir` (faster-whisper already depends on it) |

## Optional extras (`uv add --optional <extra> <pkg>`)
| Extra | Package | Why |
|---|---|---|
| `oww` | `openwakeword` | Alternative wake word; its bundled `hey_jarvis` model works as a dev placeholder |
| `deepgram` | `deepgram-sdk` | Cloud STT alternative |
| `kokoro` | `kokoro-onnx` | Alternative TTS |
| `elevenlabs` | `elevenlabs` | Cloud TTS alternative |
| `silero-torch` | `silero-vad` | ONLY if the faster-whisper ONNX Silero route turns out to be unusable. Document why first |

## Dev (`uv add --dev`)
`pytest`, `pytest-asyncio`, `ruff`

## Not allowed without a human
torch (except via the `silero-torch` extra above), any local LLM, any GUI toolkit, any cloud service SDK not listed above.

## Phase 4 desktop app (approved by the user 2026-09-29)
| Where | Package | Why |
|---|---|---|
| extra `desktop` | `pywebview` | The desktop window (OS web view). Pulls in pythonnet on Windows |
| group `build` | `pyinstaller` | Freezing the app for the installer |
External tool (not pip): Inno Setup 6, for the Windows installer.

## P5.7e OpenAI-compatible brain (approved by the user 2026-10-06)
| Where | Package | Why |
|---|---|---|
| core | `httpx~=0.28.1` | Streaming Chat Completions (SSE) for `brain.backend: openai_compat`. Was already locked transitively; now declared directly. Imported lazily inside the backend |
| core | `mcp~=2.2` | MCP client sessions for `openai_compat` (the Claude CLI does this itself on the Claude brain). Was already locked through claude-agent-sdk; now declared directly. Imported lazily |
