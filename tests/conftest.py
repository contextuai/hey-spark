"""Shared fixtures. The real whisper model is loaded once per test session (it's the slowest
setup in the suite; several test files use it)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def shared_whisper():
    """A warmed-up FasterWhisperSTT for the repo's config, or skip with the fix."""
    from spark.config import Settings
    from spark.interfaces import BackendUnavailable

    try:
        from spark.stt.faster_whisper import FasterWhisperSTT

        stt = FasterWhisperSTT(Settings(config_dir=REPO))
        asyncio.run(stt.warmup())
    except (BackendUnavailable, ImportError) as exc:
        pytest.skip(f"whisper unavailable ({exc}); run: uv run python -m spark models download")
    return stt


@pytest.fixture(autouse=True)
def _isolated_user_files(tmp_path, monkeypatch):
    """The activity log (P5.5), shortcuts (P5.7a), prefs and the seen claude.ai connectors live
    next to config.yaml by default: keep
    every test's copies in its own tmp folder, never in the repo."""
    monkeypatch.setenv("SPARK_ACTIVITY__DIR", str(tmp_path / "history"))
    monkeypatch.setenv("SPARK_SHORTCUTS__FILE", str(tmp_path / "shortcuts.json"))
    monkeypatch.setenv("SPARK_MEMORY__DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("SPARK_APP__PREFS_FILE", str(tmp_path / "prefs.json"))
    monkeypatch.setenv("SPARK_CONNECTORS__CLAUDEAI_STATE_FILE", str(tmp_path / "claudeai.json"))


@pytest.fixture(autouse=True)
def _never_the_repo_env_file(tmp_path, monkeypatch):
    """The Connectors and Brain windows write secrets to ``.env`` next to config.yaml
    (``connector_store.env_path``). A test whose settings point at the repo (``Settings()``
    defaults ``config_dir`` to the working directory) gets a tmp ``.env`` instead, so no test
    can read or change the developer's real keys."""
    from spark.brain import connector_store

    real = connector_store.env_path
    repo_env = (REPO / ".env").resolve()

    def env_path(settings):
        path = real(settings)
        return tmp_path / ".env" if path.resolve() == repo_env else path

    monkeypatch.setattr(connector_store, "env_path", env_path)


@pytest.fixture(autouse=True)
def _close_transcript_logs(monkeypatch):
    """Many tests build a SparkApp and never close it, leaving its transcript file open until GC,
    where the ResourceWarning lands on whichever test runs then. Close them per test."""
    from spark.util.logging import TranscriptLog

    opened: list[TranscriptLog] = []
    original = TranscriptLog.__init__

    def tracking_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(TranscriptLog, "__init__", tracking_init)
    yield
    for log in opened:
        log.close()
