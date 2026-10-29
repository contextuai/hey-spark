"""CLI parsing and the offline commands of ``python -m spark`` (spec §11)."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from spark import __main__ as cli
from spark.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_parser_defaults_to_run():
    args = cli.build_parser().parse_args([])
    assert args.cmd is None and not args.text and not args.no_tts and not args.debug_audio


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["doctor"],
        ["models", "download"],
        ["bench", "stt"],
        ["bench", "tts", "--trials", "3"],
        ["bench", "brain"],
        ["bench", "latency", "--trials", "10"],
        ["bench", "false-wake", "--minutes", "30"],
        ["record-fixture", "cmd_01", "--seconds", "3"],
        ["--text", "--no-tts", "--brain", "mock"],
        ["--debug-audio", "--stt", "deepgram", "--tts", "kokoro"],
    ],
)
def test_parser_accepts(argv):
    cli.build_parser().parse_args(argv)


def test_overrides_flow_into_settings(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args(
        ["--config", str(cfg), "--brain", "mock", "--stt", "deepgram", "--debug-audio"]
    )
    s = cli.load(args)
    assert isinstance(s, Settings)
    assert s.brain.backend == "mock"
    assert s.stt.backend == "deepgram"
    assert s.debug_audio.enabled is True
    assert s.tts.backend == "kokoro"


def test_no_tts_selects_print_backend(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    s = cli.load(cli.build_parser().parse_args(["--config", str(cfg), "--no-tts"]))
    assert s.tts.backend == "print"


def test_debug_audio_off_by_default(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    s = cli.load(cli.build_parser().parse_args(["--config", str(cfg)]))
    assert s.debug_audio.enabled is False


def test_text_mode_smoke_subprocess(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"logging:\n  dir: {tmp_path.as_posix()}/logs\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "spark",
            "--config",
            str(cfg),
            "--text",
            "--no-tts",
            "--brain",
            "mock",
        ],
        input="what is two plus two\nhello\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Two plus two is four." in proc.stdout
    assert "Spark" in proc.stdout


async def test_run_text_in_process(tmp_path, monkeypatch, capsys):
    from rich.console import Console

    s = Settings(
        brain={"backend": "mock"},
        tts={"backend": "print"},
        logging={"dir": str(tmp_path / "logs")},
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("what is two plus two\nquit\nignored\n"))
    code = await cli.run_text(s, speak=False, console=Console(file=io.StringIO()))
    assert code == 0
    out = capsys.readouterr().out
    assert "Two plus two is four." in out
    assert "ignored" not in out


def test_unknown_backend_reports_choices(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"logging:\n  dir: {tmp_path.as_posix()}/logs\n", encoding="utf-8")
    monkey_stdin = io.StringIO("hi\n")
    old = sys.stdin
    sys.stdin = monkey_stdin
    try:
        code = cli.main(["--config", str(cfg), "--text", "--no-tts", "--brain", "nope"])
    finally:
        sys.stdin = old
    assert code == cli.EXIT_UNAVAILABLE
    assert "mock" in capsys.readouterr().out


def test_backend_unavailable_prints_fix(capsys):
    from rich.console import Console

    from spark.interfaces import BackendUnavailable

    buf = io.StringIO()
    code = cli._unavailable(
        Console(file=buf), BackendUnavailable("porcupine", "no key", "set PICOVOICE_ACCESS_KEY")
    )
    assert code == cli.EXIT_UNAVAILABLE
    assert "set PICOVOICE_ACCESS_KEY" in buf.getvalue()


def test_bad_config_exits_nonzero(tmp_path):
    assert cli.main(["--config", str(tmp_path / "missing.yaml"), "doctor"]) == cli.EXIT_FAIL


def test_text_mode_prints_without_audio_by_default(tmp_path, monkeypatch):
    from spark.app import SparkApp

    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args(["--config", str(cfg), "--text"])
    s = cli.load(args)
    assert s.tts.backend == "print"
    app = SparkApp.from_settings(s.model_copy(update={}), audio=False, speaker=False)
    assert app.player is None and app.mic is None


def test_text_speak_opt_in_keeps_configured_tts(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    s = cli.load(cli.build_parser().parse_args(["--config", str(cfg), "--text", "--speak"]))
    assert s.tts.backend == "kokoro"


def test_cli_unavailable_redacts_and_escapes():
    from rich.console import Console

    from spark.interfaces import BackendUnavailable

    buf = io.StringIO()
    cli._unavailable(
        Console(file=buf), BackendUnavailable("x", "bad key ghp_abcdefghijklmnopqrstu [red]", "fix")
    )
    out = buf.getvalue()
    assert "ghp_abc" not in out and "[red]" in out  # masked, and markup shown literally
