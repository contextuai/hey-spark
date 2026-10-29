"""spark.desktop.macos: pure helpers, tested with a fake ``subprocess.run`` on any OS."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from spark.desktop import macos


def fake_run(stdout: str = "", returncode: int = 0, exc: Exception | None = None):
    calls: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_prefers_dark() -> None:
    assert macos.prefers_dark(fake_run("Dark\n")) is True
    # Light mode: the key doesn't exist and `defaults` fails
    assert macos.prefers_dark(fake_run("", returncode=1)) is False
    assert macos.prefers_dark(fake_run(exc=FileNotFoundError())) is False


def test_applescript_string_escapes() -> None:
    assert macos.applescript_string('say "hi" \\ bye') == '"say \\"hi\\" \\\\ bye"'


def test_show_alert_quotes_the_message() -> None:
    run = fake_run()
    macos.show_alert("Hey Spark", 'couldn\'t start: "boom"', run)
    [argv] = run.calls
    assert argv[:2] == ["osascript", "-e"]
    assert 'message "couldn\'t start: \\"boom\\"" as critical' in argv[2]
    macos.show_alert("t", "m", fake_run(exc=OSError()))  # never raises


def test_login_shell_path_skips_banners() -> None:
    run = fake_run("Welcome to zsh!\n__SPARK_PATH__=/opt/homebrew/bin:/usr/bin")
    assert macos.login_shell_path({"SHELL": "/bin/bash"}, run) == "/opt/homebrew/bin:/usr/bin"
    assert run.calls[0][:2] == ["/bin/bash", "-ilc"]
    assert macos.login_shell_path({}, fake_run("no marker")) is None
    assert macos.login_shell_path({}, fake_run(exc=subprocess.TimeoutExpired("zsh", 5))) is None


def test_merged_path_orders_and_dedupes(tmp_path: Path) -> None:
    brew = tmp_path / "brew"
    local = tmp_path / "home" / ".local" / "bin"
    brew.mkdir()
    local.mkdir(parents=True)
    current = os.pathsep.join(["/usr/bin", "/bin"])
    shell = os.pathsep.join([str(brew), "/usr/bin", str(tmp_path / "gone")])
    merged = macos.merged_path(
        current, shell, home=tmp_path / "home", extra=(str(brew), "~/.local/bin", "/nope")
    )
    # shell entries first (missing folders dropped), current kept as-is, extras only if present
    assert merged.split(os.pathsep) == [str(brew), "/usr/bin", "/bin", str(local)]


def test_fix_gui_path_updates_the_env(tmp_path: Path) -> None:
    env = {"PATH": "/usr/bin", "SHELL": "/bin/zsh"}
    run = fake_run(f"__SPARK_PATH__={tmp_path}")
    new = macos.fix_gui_path(env, run)
    assert env["PATH"] == new
    assert new.split(os.pathsep)[:2] == [str(tmp_path), "/usr/bin"]
