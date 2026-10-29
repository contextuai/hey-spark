# Contributing to Hey Spark

Thanks for your interest in Hey Spark, a wake-word voice assistant by
[ContextuAI](https://contextuai.com). This guide covers setting up, testing, the conventions we
follow, and how releases are cut.

By taking part you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Security problems
go through private reporting, not issues: see [SECURITY.md](SECURITY.md).

## Getting started

Hey Spark targets **Windows 10/11** and **Python 3.11**. The tests and the text mode also run on
Linux. You need [uv](https://docs.astral.sh/uv/), which manages Python and the virtualenv for you.
Don't use a system `python` directly; always go through `uv run`.

```bash
git clone https://github.com/<you>/hey-spark.git
cd hey-spark
uv sync --all-extras          # all optional backends + the dev tools
```

> **Use `uv sync --all-extras`.** A plain `uv sync` *removes* the optional extras (`desktop`,
> `oww`, `deepgram`, `kokoro`, `elevenlabs`), and parts of the test suite exercise them.

To build the Windows installer you also need the `build` group and Inno Setup 6:
`uv sync --all-extras --group build`, then `powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1`.

## The quality gate

Every PR must pass these. CI runs the same commands on Windows and Ubuntu
(`.github/workflows/pr-checks.yml`).

```bash
uv run ruff check .
uv run ruff format --check .           # `uv run ruff format .` to fix
uv run pytest -q                       # offline suite, no keys, no hardware, about a minute
echo "what is two plus two" | uv run spark --brain mock --text   # smoke run, fully offline
```

Opt-in suites, which CI does not run:

```bash
uv run pytest -q -m live       # hits real Claude / cloud APIs (needs keys or a `claude` login)
uv run pytest -q -m hardware   # needs a real microphone or speaker
uv run spark doctor            # checks your local environment
```

## Writing tests

- **Test with fakes, not hardware.** `tests/fakes.py` has `FakeMic`, `FakeWakeWord`, `FakeVAD`,
  `FakeSTT`, `FakeTTS` and `MockBrain`. The state machine, barge-in and confirmation flows are all
  tested this way. Nothing in the default suite may open a real audio device.
- Tests that need downloaded model files must **skip cleanly** with a reason when the files are
  absent (`pytest.mark.skipif`). CI has no models.
- Tests that need the network or API keys are marked `@pytest.mark.live`.
- Keep the offline suite fast (under about 60 s).

## Architecture rules worth knowing

- Every component (wake word, VAD, STT, TTS, brain, audio in/out) is an ABC in
  `spark/interfaces.py`. Backends are picked **only** through `config.yaml` via the registry in
  `spark/registry.py`, never with `if backend == ...` in app code.
- Heavy or optional imports (`openwakeword`, `faster_whisper`, `piper`, `sounddevice`, cloud SDKs)
  happen **inside** the backend's constructor, never at module top level, so tests and `--text`
  mode import cleanly without them.
- Audio is 16 kHz mono int16 internally. The app loop is `asyncio`; blocking work runs in
  `asyncio.to_thread`.
- Secrets come only from the environment or `.env`, and are never logged.
- Side-effecting tools always go through the spoken confirmation hook
  (`spark/brain/confirm.py`). The default answer is no.

### Adding a backend

Follow [`.claude/skills/spark-backend-pattern/SKILL.md`](.claude/skills/spark-backend-pattern/SKILL.md):
implement the ABC from `spark/interfaces.py`, register it in `spark/registry.py`, keep its imports
lazy, add its options to `config.yaml` with a comment, add a `spark doctor` check, and add tests
(with fakes or a skip when its models or keys are missing). A new third-party dependency usually
belongs in an optional extra (`uv add --optional <extra> <package>`); discuss it in the issue or
PR first.

## Wake-word models and licensing

The code is MIT, but the bundled wake-word model `assets/wakewords/hey_spark.onnx` is
**CC BY-NC-SA 4.0 (non-commercial)**, because the background audio used to train it carries mixed
licenses. See [`assets/wakewords/LICENSE`](assets/wakewords/LICENSE).

If you contribute a new or retrained wake-word model (for example one made with
[`tools/wakeword_trainer.ipynb`](tools/wakeword_trainer.ipynb), described in
[`docs/wakeword.md`](docs/wakeword.md)), your PR must state:

- the phrase, the trainer settings and the openWakeWord version used;
- **every dataset used for training** (TTS voices, background noise, music, room impulse
  responses, negative speech) and **its license**;
- the resulting license of the model, which can be no more permissive than the most restrictive
  training input.

Models with unknown or incompatible training-data licenses can't be accepted.

## Commits, branches and pull requests

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | For |
|---|---|
| `feat:` | a new feature |
| `fix:` | a bug fix |
| `docs:` | documentation only |
| `test:` | adding or fixing tests |
| `refactor:` | code changes that neither fix a bug nor add a feature |
| `perf:` | performance |
| `ci:` | workflows and CI config |
| `build:` | packaging, the installer, dependencies |
| `chore:` | maintenance (including release bumps) |

Add a scope when it helps (`feat(tts): ...`) and `!` for breaking changes (`refactor!: ...`).

The flow:

1. Open or find an issue for anything bigger than a small fix, so we can agree on the approach.
2. Fork, then branch from `main` (`feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`).
3. Make the change with tests, and run the quality gate locally.
4. Open a PR against `main` and fill in the template. **The PR title must be a conventional
   commit**: PRs are squash-merged, and the title becomes the commit on `main`.
5. A maintainer reviews it. CI must be green before merge.

## How releases work

Releases are automated by `.github/workflows/release.yml`. Versions are `X.Y.Z`, with an optional
numeric suffix `X.Y.Z-N` for pre-releases. To cut one, a maintainer:

1. Opens a PR that bumps the version **in all three places**:
   - `pyproject.toml` (`[project].version`)
   - `spark/__init__.py` (`__version__`)
   - `uv.lock` (run `uv lock` after editing `pyproject.toml`)

   and moves the `## [Unreleased]` notes in `CHANGELOG.md` under the new version.
2. Titles the PR exactly `chore: release vX.Y.Z` (or `chore: release vX.Y.Z-N`).
3. Squash-merges it. The push to `main` triggers the release workflow, which checks that the
   three version files agree with the title, creates the `vX.Y.Z` tag, builds the Windows
   installer (PyInstaller + Inno Setup) and publishes a GitHub Release with
   `HeySpark-Setup-X.Y.Z.exe`, its SHA-256 checksum, and generated notes. A `-N` version is
   published as a pre-release.

Maintainers can also run the workflow by hand (Actions, "Release", "Run workflow", with a version)
or push a `vX.Y.Z` tag; both paths run the same checks. The installer is not code-signed yet.

## Questions

Open an issue, or start a discussion on the pull request. Thanks for helping make Hey Spark
better.
