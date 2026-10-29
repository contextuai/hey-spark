"""Doctor: every check reports ✔/✖ with a fix, never crashes, and sets the exit code."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from rich.console import Console

from spark import doctor
from spark.doctor import Check
from spark.interfaces import BackendUnavailable
from tests.fakes import MockBrain, make_settings


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=200, highlight=False), buf


def test_exit_code_only_counts_required_failures():
    assert doctor.exit_code([Check("a", True), Check("b", False, required=False)]) == 0
    assert doctor.exit_code([Check("a", True), Check("b", False)]) == 1


def test_symbols():
    assert Check("a", True).symbol == "✔"
    assert Check("a", False).symbol == "✖"
    assert Check("a", False, required=False).symbol == "!"


def test_main_prints_fix_and_returns_nonzero():
    def ok(settings):
        return Check("python", True, "3.11")

    def bad(settings):
        return Check("microphone", False, "no device", "plug in a mic")

    def warn(settings):
        return [Check("npx", False, "missing", "install node", required=False)]

    console, buf = _console()
    code = doctor.main(make_settings(), console=console, checks=[ok, bad, warn])
    out = buf.getvalue()
    assert code == 1
    assert "✔ python" in out and "✖ microphone" in out and "! npx" in out
    assert "fix: plug in a mic" in out and "fix: install node" in out
    assert "1 required check(s) failed" in out


def test_crashing_check_becomes_failure_not_crash():
    def boom(settings):
        raise RuntimeError("kaput")

    async def aboom(settings):
        raise OSError("nope")

    console, buf = _console()
    code = doctor.main(make_settings(), console=console, checks=[boom, aboom])
    assert code == 1
    assert "kaput" in buf.getvalue() and "nope" in buf.getvalue()


def test_all_pass_exit_zero():
    console, buf = _console()
    assert doctor.main(make_settings(), console=console, checks=[lambda s: Check("x", True)]) == 0
    assert "All required checks passed" in buf.getvalue()


def test_check_python():
    assert doctor.check_python(make_settings()).ok is (sys.version_info >= (3, 11))


def test_check_dotenv(tmp_path):
    s = make_settings(config_dir=tmp_path)
    c = doctor.check_dotenv(s)
    assert not c.ok and not c.required and ".env.example" in c.fix
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
    assert doctor.check_dotenv(s).ok


def test_check_disk(tmp_path, monkeypatch):
    s = make_settings(models_dir=str(tmp_path / "models"))
    assert doctor.check_disk(s).ok

    class Usage:
        free = int(0.5e9)

    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage)
    c = doctor.check_disk(s)
    assert not c.ok and "GB" in c.fix


def test_check_mic_maps_unavailable(monkeypatch):
    import spark.audio.stream as stream

    def fail(settings, seconds=0.5):
        raise BackendUnavailable("audio_in", "no input device", "plug in a mic")

    monkeypatch.setattr(stream, "probe_input", fail)
    c = doctor.check_mic(make_settings())
    assert not c.ok and c.required and c.fix == "plug in a mic"

    def good(settings, seconds=0.5):
        return {
            "name": "Jabra",
            "hostapi": "Windows WASAPI",
            "samplerate": 16000,
            "resampled": False,
            "samples": 4800,
        }

    monkeypatch.setattr(stream, "probe_input", good)
    c = doctor.check_mic(make_settings())
    assert c.ok and "Jabra" in c.detail and "WASAPI" in c.detail


def test_check_wakeword_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("PICOVOICE_ACCESS_KEY", raising=False)
    c = doctor.check_wakeword(make_settings(config_dir=tmp_path, wakeword={"backend": "porcupine"}))
    assert not c.ok and c.required
    assert c.fix


def test_check_models_reports_each(monkeypatch):
    import spark.models

    monkeypatch.setattr(
        spark.models,
        "model_status",
        lambda s: {
            "whisper base.en": {"present": True, "path": "/m/whisper"},
            "piper en_US-lessac-medium": {"present": False, "path": "/m/piper"},
        },
        raising=False,
    )
    checks = doctor.check_models(make_settings())
    assert [c.ok for c in checks] == [True, False]
    assert "models download" in checks[1].fix


def test_check_claude_cli_missing(monkeypatch, tmp_path):
    # The frozen app has no bundled CLI: nothing on PATH or in ~/.local/bin means "install it".
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor, "_bundled_cli", lambda: None)
    monkeypatch.setattr(doctor.Path, "home", lambda: tmp_path)
    c = doctor.check_claude_cli(make_settings())
    assert not c.ok and "install Claude Code" in c.fix and "cli_path" in c.fix


def test_find_claude_cli_follows_the_sdk_order(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(doctor, "_bundled_cli", lambda: None)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert doctor.find_claude_cli(make_settings()) is None
    # the native installer's folder
    name = "claude.exe" if sys.platform == "win32" else "claude"
    native = tmp_path / ".local" / "bin" / name
    native.parent.mkdir(parents=True)
    native.write_bytes(b"")
    assert doctor.find_claude_cli(make_settings()) == str(native)
    # PATH wins over that; on Windows only a native .exe (the SDK refuses npm's claude.cmd)
    on_path = "C:/tools/claude.exe" if sys.platform == "win32" else "/opt/bin/claude"
    monkeypatch.setattr(doctor.shutil, "which", lambda n: on_path if n == "claude" else None)
    assert doctor.find_claude_cli(make_settings()) == on_path
    if sys.platform == "win32":
        monkeypatch.setattr(doctor.shutil, "which", lambda n: "C:/npm/claude.cmd")
        assert doctor.find_claude_cli(make_settings()) == str(native)
    # the SDK's bundled CLI (source installs) and then brain.cli_path win over everything
    monkeypatch.setattr(doctor, "_bundled_cli", lambda: Path("/sdk/_bundled/claude"))
    assert doctor.find_claude_cli(make_settings()) == str(Path("/sdk/_bundled/claude"))
    s = make_settings(brain={"backend": "mock", "cli_path": "D:/claude.exe"})
    assert doctor.find_claude_cli(s) == "D:/claude.exe"


async def test_check_brain_with_mock():
    c = await doctor.check_brain(make_settings())
    assert c.ok, c.detail


async def test_check_brain_failure_gives_auth_fix(monkeypatch):
    from spark import registry

    monkeypatch.setattr(registry, "build", lambda kind, settings: _BrokenBrain())
    c = await doctor.check_brain(make_settings(brain={"backend": "claude_agent"}))
    assert not c.ok and "/login" in c.fix and "ANTHROPIC_API_KEY" in c.fix
    # another brain's failure doesn't send the user to Anthropic
    c = await doctor.check_brain(make_settings(brain={"backend": "openai_compat"}))
    assert not c.ok and "ANTHROPIC" not in c.fix


class _BrokenBrain(MockBrain):
    async def warmup(self):
        raise ConnectionError("401 unauthorized")


def test_optional_backend_required_only_when_selected(monkeypatch):
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)
    checks = {c.name: c for c in doctor.check_optional_backends(make_settings())}
    assert not checks["optional: deepgram"].required
    checks = {
        c.name: c
        for c in doctor.check_optional_backends(make_settings(stt={"backend": "deepgram"}))
    }
    assert checks["optional: deepgram"].required and not checks["optional: deepgram"].ok


def test_check_npx_is_optional(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    c = doctor.check_npx(make_settings())
    assert not c.ok and not c.required


@pytest.mark.hardware
def test_real_doctor_runs():
    from spark.config import load_settings

    console, buf = _console()
    code = doctor.main(load_settings(), console=console)
    # the console may be cp1252 under `pytest -s`; the CLI itself forces UTF-8 (spark.__main__)
    print(buf.getvalue().encode("ascii", "replace").decode("ascii"))
    assert "Spark doctor" in buf.getvalue()
    assert code in (0, 1)


def test_render_redacts_secrets(monkeypatch):
    monkeypatch.setenv("BAR_API_KEY", "envsecret987654")
    console, buf = _console()
    doctor.render(
        [Check("x", False, "saw envsecret987654 and sk-ant-abcdefghijkl", "use sk-ant-zzzzzzzzzz")],
        console,
    )
    out = buf.getvalue()
    assert "envsecret987654" not in out and "sk-ant-abc" not in out and "sk-ant-zzz" not in out
