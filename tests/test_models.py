"""spark.models: paths, idempotent downloads (monkeypatched, no network), disk guard, CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from spark import models
from spark.config import Settings
from spark.models import (
    InsufficientDiskSpace,
    download_models,
    kokoro_paths,
    model_status,
    piper_hf_path,
    piper_voice_paths,
    whisper_model_dir,
    whisper_present,
    whisper_repo_id,
)

BIG = 100 * 1024**3


class _Quiet:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args, **kwargs) -> None:
        self.lines.append(" ".join(str(a) for a in args))


class _Downloads:
    """Records download calls and writes small fake files like the real helpers would."""

    def __init__(self) -> None:
        self.snapshots: list[tuple[str, Path, list[str]]] = []
        self.files: list[tuple[str, str]] = []
        self.urls: list[str] = []
        self.fail_snapshot = False

    def snapshot(self, repo_id: str, local_dir: Path, allow_patterns: list[str]) -> None:
        self.snapshots.append((repo_id, local_dir, allow_patterns))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model.bin").write_bytes(b"m" * 1000)
        if self.fail_snapshot:
            raise ConnectionError("network dropped")
        for name in ("config.json", "tokenizer.json", "vocabulary.txt"):
            (local_dir / name).write_bytes(b"{}")
        (local_dir / ".cache" / "huggingface").mkdir(parents=True)

    def hf_file(self, repo_id: str, filename: str, local_dir: Path) -> Path:
        self.files.append((repo_id, filename))
        dest = local_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"v" * 500)
        return dest

    def url(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        dest.write_bytes(b"k" * 300)


@pytest.fixture
def dl(monkeypatch):
    d = _Downloads()
    monkeypatch.setattr(models, "_snapshot_download", d.snapshot)
    monkeypatch.setattr(models, "_hf_file_download", d.hf_file)
    monkeypatch.setattr(models, "_url_download", d.url)
    monkeypatch.setattr(models, "free_bytes", lambda path: BIG)
    return d


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(config_dir=tmp_path, models_dir="./m")


# --- paths ------------------------------------------------------------------------------------


def test_piper_hf_path():
    assert piper_hf_path("en_US-lessac-medium") == "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    assert piper_hf_path("en_US-lessac-low") == "en/en_US/lessac/low/en_US-lessac-low.onnx"
    assert (
        piper_hf_path("en_GB-northern_english_male-medium")
        == "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx"
    )
    assert piper_hf_path("de_DE-thorsten-high").startswith("de/de_DE/thorsten/high/")
    for bad in ("lessac", "en_US-lessac", "enUS-lessac-medium"):
        with pytest.raises(ValueError):
            piper_hf_path(bad)


def test_whisper_repo_and_dir(settings, tmp_path):
    assert whisper_repo_id("base.en") == "Systran/faster-whisper-base.en"
    assert whisper_repo_id("org/custom-model") == "org/custom-model"
    assert whisper_model_dir(settings) == (tmp_path / "m" / "whisper" / "base.en").resolve()
    s2 = Settings(
        config_dir=tmp_path,
        models_dir="./m",
        stt={"options": {"faster_whisper": {"model": "org/custom"}}},
    )
    assert whisper_model_dir(s2).name == "org--custom"


def test_voice_and_kokoro_paths(settings, tmp_path):
    onnx, cfg = piper_voice_paths(settings)
    assert onnx == (tmp_path / "m" / "piper" / "en_US-ljspeech-medium.onnx").resolve()
    assert cfg.name == "en_US-ljspeech-medium.onnx.json"
    model, voices = kokoro_paths(settings)
    assert model.parent.name == "kokoro" and model.name == "kokoro-v1.0.onnx"
    assert voices.name == "voices-v1.0.bin"


# --- download ---------------------------------------------------------------------------------


def test_download_places_files_where_backends_look_and_is_idempotent(settings, dl):
    out = _Quiet()
    result = download_models(settings, console=out)

    wdir = whisper_model_dir(settings)
    onnx, cfg = piper_voice_paths(settings)
    assert result == {"whisper": wdir, "piper": onnx}
    assert whisper_present(wdir) and onnx.is_file() and cfg.is_file()
    assert not (wdir / ".cache").exists()
    assert dl.snapshots[0][0] == "Systran/faster-whisper-base.en"
    assert "model.bin" in dl.snapshots[0][2]
    assert dl.files == [
        ("rhasspy/piper-voices", "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json"),
        ("rhasspy/piper-voices", "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx"),
    ]
    leftovers = [p for p in settings.models_path.rglob("*") if ".partial" in p.name]
    assert leftovers == []
    assert any("free disk" in line for line in out.lines)
    assert any("MB" in line for line in out.lines)

    download_models(settings, console=_Quiet())
    assert len(dl.snapshots) == 1 and len(dl.files) == 2  # nothing re-downloaded

    status = model_status(settings)
    assert status["whisper"]["present"] and status["piper"]["present"]
    assert not status["kokoro"]["present"]
    assert status["whisper"]["size_bytes"] > 0
    # Shape spark.doctor relies on: every entry is a dict with present/path/required.
    for info in status.values():
        assert isinstance(info["present"], bool) and isinstance(info["path"], str)
        assert isinstance(info["required"], bool)
    assert status["whisper"]["required"] and not status["piper"]["required"]
    assert status["kokoro"]["required"]  # the default voice, George


def test_required_follows_selected_backends(tmp_path):
    s = Settings(config_dir=tmp_path, stt={"backend": "deepgram"}, tts={"backend": "kokoro"})
    status = model_status(s)
    assert not status["whisper"]["required"] and not status["piper"]["required"]
    assert status["kokoro"]["required"]


def test_interrupted_whisper_download_is_not_mistaken_for_complete(settings, dl):
    dl.fail_snapshot = True
    with pytest.raises(ConnectionError):
        download_models(settings, piper=False, console=_Quiet())
    wdir = whisper_model_dir(settings)
    assert not wdir.exists()
    assert not model_status(settings)["whisper"]["present"]

    dl.fail_snapshot = False
    download_models(settings, piper=False, console=_Quiet())
    assert whisper_present(wdir)
    assert not wdir.with_name(wdir.name + ".partial").exists()


def test_piper_only_json_present_counts_as_missing(settings, dl):
    onnx, cfg = piper_voice_paths(settings)
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}", encoding="utf-8")
    assert not model_status(settings)["piper"]["present"]
    download_models(settings, whisper=False, console=_Quiet())
    assert onnx.is_file() and len(dl.files) == 2


def test_refuses_when_disk_is_low(settings, dl, monkeypatch):
    monkeypatch.setattr(models, "free_bytes", lambda path: 1024**3)
    with pytest.raises(InsufficientDiskSpace):
        download_models(settings, console=_Quiet())
    assert dl.snapshots == [] and dl.files == []


def test_low_disk_is_fine_when_nothing_is_missing(settings, dl, monkeypatch):
    download_models(settings, console=_Quiet())
    monkeypatch.setattr(models, "free_bytes", lambda path: 1024**3)
    download_models(settings, console=_Quiet())
    assert len(dl.snapshots) == 1


def test_kokoro_download(settings, dl):
    download_models(settings, whisper=False, piper=False, kokoro=True, console=_Quiet())
    model, voices = kokoro_paths(settings)
    assert model.is_file() and voices.is_file()
    assert dl.urls == [
        f"{models.KOKORO_URL_BASE}/voices-v1.0.bin",
        f"{models.KOKORO_URL_BASE}/kokoro-v1.0.onnx",
    ]
    download_models(settings, whisper=False, piper=False, kokoro=True, console=_Quiet())
    assert len(dl.urls) == 2


def test_free_bytes_handles_missing_dirs(tmp_path):
    assert models.free_bytes(tmp_path / "does" / "not" / "exist") > 0


def test_cli_download_and_status(tmp_path, dl, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models_dir: ./mm\n", encoding="utf-8")
    assert models.main(["download", "--config", str(cfg), "--no-piper"]) == 0
    assert whisper_present(tmp_path / "mm" / "whisper" / "base.en")
    assert dl.files == []
    assert models.main(["status", "--config", str(cfg)]) == 0
    assert "whisper" in capsys.readouterr().out


def test_cli_reports_low_disk(tmp_path, dl, monkeypatch, capsys):
    monkeypatch.setattr(models, "free_bytes", lambda path: 10)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models_dir: ./mm\n", encoding="utf-8")
    assert models.main(["download", "--config", str(cfg)]) == 2
    assert "need at least" in capsys.readouterr().err


def test_stt_backend_reads_the_download_location(settings, dl, monkeypatch):
    """FasterWhisperSTT must look exactly where download_models writes."""
    import sys
    import types

    download_models(settings, piper=False, console=_Quiet())
    seen: list[str] = []
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = lambda path, **kw: seen.append(path)
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    from spark.stt.faster_whisper import FasterWhisperSTT

    stt = FasterWhisperSTT(settings)
    stt._load()
    assert seen == [str(whisper_model_dir(settings))]


def test_url_download_verifies_with_certifis_bundle(tmp_path, monkeypatch):
    # The macOS app's Python has no system CA store: downloads must pass certifi's bundle.
    import io

    import certifi

    seen = {}

    def fake_urlopen(url, timeout, context):
        seen["url"], seen["context"] = url, context
        return io.BytesIO(b"model bytes")

    loaded = []
    monkeypatch.setattr(
        models.ssl,
        "create_default_context",
        lambda cafile=None: loaded.append(cafile) or "ctx",
    )
    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "kokoro.onnx"
    models._url_download("https://example.com/kokoro.onnx", dest)
    assert dest.read_bytes() == b"model bytes"
    assert loaded == [certifi.where()] and seen["context"] == "ctx"
