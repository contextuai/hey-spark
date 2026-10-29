# Changelog

All notable changes to Hey Spark are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-releases are tagged
`vX.Y.Z-N`.

## [Unreleased]

## [0.1.0] - 2026-10-31

The first public release: desktop apps for macOS (Apple Silicon) and Windows, and the source.

### Added

- **Hermes Agent as a brain.** Pick Hermes Agent under "Recommended" in the Brain window, enter
  its address (this computer or your server) and the API_SERVER_KEY from Hermes's .env, and
  Spark talks to Hermes's OpenAI-compatible API server. Hermes answers with its own model,
  tools, memory and skills.
- **Film-style commands and shortcuts.** Thirteen new built-in shortcuts ("sitrep", "what's
  next?", "prep me", "check the post", "what's happening in the workshop?", "I'm home", "wrap up
  the day", "run a diagnostic", "time check", "you up?", "drop my needle", "house party
  protocol", "start a new project"), "rise and shine" for the morning briefing, and new command
  phrases ("stand down", "belay that", "clean slate protocol", "full power", "conserve power",
  "go dark"). A shortcut can be set
  to fire only when its phrase is said on its own, and new built-in shortcuts are added to an
  existing `shortcuts.json` once. See `docs/commands.md`.
- **macOS app.** A drag-to-Applications disk image for Apple Silicon Macs on macOS 14 or later,
  built alongside the Windows installer (`scripts/build_macos.sh`, `build-apps.yml`). The app
  asks for microphone access, opens Claude Code in Terminal for new projects, follows the system
  dark mode, and picks up the commands from your shell's `PATH` (Homebrew, npm, `claude`) even
  when it is opened from Finder. Developer ID signing and notarization switch on when the
  signing secrets are configured.
- **App icon.** A glowing orange core inside waveform rings on a charcoal tile, with a simplified
  version for the small taskbar and title-bar sizes, for macOS, Windows and the window.
- **Frozen-app self-test.** `HeySpark --self-test` loads every native library and checks the
  bundled files; every CI build runs it on both platforms.
- **Personalities.** `app.persona: butler` turns Spark into a composed British butler with a dry
  wit: it calls you "sir" (`app.address_as`), answers the wake word out loud ("Yes, sir?"), and
  words its own lines ("Shall I proceed, sir?") to match. A Personality & Voice window picks the
  personality and the voice, with a sample for each voice. See `docs/persona.md`.
- **George is the default voice.** Kokoro's British male voice George (`bm_george`) replaces
  Piper's LJ as the default. The Kokoro engine ships in the installers and its 340 MB model
  downloads on first run ("Downloading the default voice, George"). The Personality & Voice
  window offers the seven Kokoro voices; Piper voices can still be set in `config.yaml`.
- **Wake word.** A bundled openWakeWord model for "Hey Spark" (`assets/wakewords/hey_spark.onnx`,
  CC BY-NC-SA 4.0, non-commercial), plus Picovoice Porcupine as an alternative backend.
- **Wake-word trainer.** `tools/wakeword_trainer.ipynb`, a Google Colab notebook that trains an
  openWakeWord model for any phrase from synthetic Piper voices on Colab's free T4 GPU (one click
  from the "Open in Colab" badge), with a step-by-step guide in `docs/wakeword.md`.
- **Voice pipeline.** 16 kHz mic capture, Silero VAD end-of-speech detection, sentence-streamed
  speech output, earcons, a follow-up window without the wake word, and optional barge-in.
- **Speech-to-text backends:** faster-whisper (local, default) and Deepgram (cloud, optional).
- **Text-to-speech backends:** Kokoro ONNX (local, default, voice George), Piper (local) and
  ElevenLabs (cloud, optional).
- **Brains.** Claude through the Claude Agent SDK and an installed Claude Code (its own sign-in,
  or an Anthropic API key),
  with fast/deep model routing and a read-only researcher sub-agent; and any OpenAI-compatible
  Chat Completions endpoint (presets for OpenAI, Azure OpenAI, OpenRouter, Groq, Together,
  Ollama, LM Studio, Hermes Agent and vLLM). In the Brain window, Ollama models are picked from
  Ollama's own list or downloaded there, and local models are warmed up and kept loaded for
  faster first answers. A mock brain for offline testing.
- **MCP connectors** (`mcp_servers.json` with `${VAR}` expansion) with per-server tool policy:
  read-only tools pre-approved, side-effecting tools gated by a **spoken "yes"** (silence or an
  unclear answer means no), destructive tools blocked by default, everything else denied.
  Connector health checks and per-connector notes. GitHub in two minutes: paste GitHub's hosted
  server and a token into the Connectors window (README); reading runs freely and creating an
  issue asks first. The Connectors window:
  - says when a server refuses the token (HTTP 401/403) instead of a bare "Server returned an
    error response";
  - refuses VS Code's `${input:...}` prompts rather than saving one as the token, and turns
    `${env:NAME}` into `${NAME}`;
  - lets a pasted server take the name of a switched-off one in `config.yaml` (such as the
    shipped `github` example);
  - deletes a removed server's token from `.env` when nothing else uses it.

  The bundled local `github` server no longer starts with `--read-only`, so its issue tools work
  after a spoken yes, as `config.yaml` lists them.
- **Local tools:** timers, the time, and a "start a new project" flow (folder, git repo, private
  GitHub repo and tracking issue) behind the confirmation gate.
- **Awareness and conversations:** an awake window after each reply, local voice commands
  ("go to sleep", "new conversation", "stop", "mute"), an "is this for me?" check, a new
  conversation per wake with a short carry-over briefing, and voice shortcuts.
- **Activity log:** an append-only, hash-chained, secret-redacted JSONL record of conversations,
  tool calls and confirmations.
- **Desktop app** (pywebview) with a Start/Stop button, a status light, mute, and windows for
  connectors, shortcuts and brain/model settings.
- **Windows installer:** a per-user Inno Setup installer built by `scripts/build_windows.ps1`
  (PyInstaller one-folder build; speech models download on first start). Not code-signed yet.
- **CLI:** voice mode, `--text` mode, `spark doctor` environment checks, `spark models download`,
  and `spark bench` latency benchmarks.
- **Security:** secrets only from `.env` or the environment, a redacting log filter, and
  scrubbed environments for child processes Spark starts itself.
- **Project infrastructure:** CI on macOS, Windows and Ubuntu (lint, format, offline tests,
  smoke run), app builds with a frozen-app self-test on every packaging change, a release
  workflow that builds and publishes the macOS disk image and the Windows installer,
  Dependabot, and community files (contributing guide, code of conduct, security policy, issue
  and PR templates).

### Changed

- **Release defaults.** `brain.mode` defaults to `auto`. The desktop apps don't bundle the
  Claude Code CLI; they use the one you install, signed in to your Claude account or with an
  Anthropic API key. The claude.ai Gmail and Calendar connectors are off by default. The
  GPL-3.0 and LGPL-2.1 texts ship with the apps, and `THIRD_PARTY_NOTICES.md` links the
  corresponding sources.
- The project was renamed from **Chip** to **Hey Spark** (package `spark`, command `spark`,
  distribution `hey-spark`). `CHIP_*` environment variables (including `CHIP_CONFIG`) still work
  for this release with a deprecation warning; use the `SPARK_*` names instead. An explicit
  `SPARK_*` variable wins over its `CHIP_*` counterpart.

[Unreleased]: https://github.com/contextuai/hey-spark/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/contextuai/hey-spark/releases/tag/v0.1.0
