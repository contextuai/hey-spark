"""Tests for spark.util.logging: redaction, console setup, transcript log."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from spark.config import Settings
from spark.util import logging as clog
from spark.util.logging import REDACTED, SecretRedactingFilter, TranscriptLog, redact

ANTHROPIC = "sk-ant-api03-abcdefGHIJKL0123456789_xyz"
PICO = "pv1234567890ABCDEFabcdef=="


@pytest.fixture
def secret_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC)
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", PICO)
    monkeypatch.setenv("MY_SERVICE_SECRET", "hunter2hunter2")
    monkeypatch.setenv("DB_PASSWORD_MAIN", "correcthorse")
    monkeypatch.setenv("SHORT_TOKEN", "abc")  # too short to redact
    monkeypatch.setenv("SPARK_LOGGING__LEVEL", "INFO")  # not secret-looking


# --- redact() ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ANTHROPIC_API_KEY",
        "PICOVOICE_ACCESS_KEY",
        "GITHUB_TOKEN",
        "SLACK_BOT_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "CLIENT_SECRET",
        "SMTP_PASSWORD",
        "my_password_2",
        "DEEPGRAM_API_KEY",
        "AZURE_CREDENTIALS",
    ],
)
def test_secret_names(name: str) -> None:
    assert clog.is_secret_env_name(name)


@pytest.mark.parametrize("name", ["PATH", "HOME", "SPARK_LOGGING__LEVEL", "PWD", "KEYBOARD"])
def test_non_secret_names(name: str) -> None:
    assert not clog.is_secret_env_name(name)


def test_redact_env_values(secret_env: None) -> None:
    text = f"key={ANTHROPIC} pico {PICO} pw correcthorse s hunter2hunter2 short abc level INFO"
    out = redact(text)
    for secret in (ANTHROPIC, PICO, "correcthorse", "hunter2hunter2"):
        assert secret not in out
    assert "short abc" in out
    assert "level INFO" in out


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("token sk-ant-oat01-ZZZZZZZZZZZZZZZZ here", "sk-ant-oat01-ZZZZZZZZZZZZZZZZ"),
        ("gh: ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
        ("pat github_pat_11ABCDEFG0123456_abcdefXYZ", "github_pat_11ABCDEFG0123456_abcdefXYZ"),
        ("Authorization: Bearer eyJhbGciOi.JIUzI1NiJ9.abc", "eyJhbGciOi.JIUzI1NiJ9.abc"),
        ("slack xoxb-1234-5678-abcdefghij", "xoxb-1234-5678-abcdefghij"),
        ("slack xoxp-1234-5678-abcdefghij", "xoxp-1234-5678-abcdefghij"),
        ("url?api_key=abcdef123456&x=1", "abcdef123456"),
        ('{"password": "s3cr3t-value"}', "s3cr3t-value"),
    ],
)
def test_redact_token_patterns(text: str, secret: str) -> None:
    out = redact(text, environ={})
    assert secret not in out
    assert REDACTED in out


def test_redact_keeps_bearer_word_and_plain_text() -> None:
    assert redact("Bearer abcdefghijklmnop", environ={}) == f"Bearer {REDACTED}"
    plain = "What is the capital of Japan? Tokyo. The token limit is 8000."
    assert redact(plain, environ={}) == plain
    assert redact("", environ={}) == ""


def test_redact_extra_secrets() -> None:
    assert redact("x custom-secret-value y", extra_secrets=["custom-secret-value"], environ={}) == (
        f"x {REDACTED} y"
    )


# --- the filter ----------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def captured() -> Iterator[tuple[logging.Logger, _ListHandler]]:
    logger = logging.getLogger("spark.test.redaction")
    handler = _ListHandler()
    handler.addFilter(SecretRedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, handler
    logger.removeHandler(handler)


def test_filter_redacts_percent_args(
    secret_env: None, captured: tuple[logging.Logger, _ListHandler]
) -> None:
    logger, handler = captured
    logger.info("connecting with %s and %r", ANTHROPIC, {"k": PICO})
    logger.info("literal %s", "ghp_abcdefghijklmnopqrstuvwxyz0123")
    joined = "\n".join(handler.lines)
    assert ANTHROPIC not in joined and PICO not in joined and "ghp_" not in joined
    assert handler.lines[0].startswith("connecting with [REDACTED] and ")


def test_filter_redacts_exception_text(
    secret_env: None, captured: tuple[logging.Logger, _ListHandler]
) -> None:
    logger, handler = captured
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        raise RuntimeError(f"auth failed for {ANTHROPIC}")
    except RuntimeError:
        logger.exception("boom")
    assert "RuntimeError" in handler.lines[0]
    assert ANTHROPIC not in handler.lines[0]


def test_filter_survives_bad_format_args(captured: tuple[logging.Logger, _ListHandler]) -> None:
    logger, handler = captured
    logger.info("needs two %s %s", "only-one")
    assert "needs two" in handler.lines[0]


# --- setup_logging -------------------------------------------------------------------------


def _spark_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "_spark_console_handler", False)]


@pytest.fixture
def restore_root() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    for h in _spark_handlers():
        root.removeHandler(h)
        h.close()
    root.setLevel(level)


def test_setup_logging_idempotent(tmp_path: Path, restore_root: None) -> None:
    settings = Settings(config_dir=tmp_path, logging={"level": "DEBUG"})
    clog.setup_logging(settings)
    clog.setup_logging(settings)
    handlers = _spark_handlers()
    assert len(handlers) == 1
    assert any(isinstance(f, SecretRedactingFilter) for f in handlers[0].filters)
    assert handlers[0].level == logging.DEBUG
    assert logging.getLogger().level == logging.DEBUG
    assert not (tmp_path / "logs").exists()  # console only; no files


def test_setup_logging_console_false(tmp_path: Path, restore_root: None) -> None:
    settings = Settings(config_dir=tmp_path, logging={"level": "warning"})
    clog.setup_logging(settings)
    clog.setup_logging(settings, console=False)
    [fallback] = _spark_handlers()  # a quiet, redacting stderr fallback (not the rich console)
    assert fallback.level == logging.WARNING
    assert any(isinstance(f, SecretRedactingFilter) for f in fallback.filters)
    assert logging.getLogger().level == logging.WARNING


def test_text_mode_warning_is_redacted(
    tmp_path: Path, restore_root: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("FOO_TOKEN", "supersecret123")
    clog.setup_logging(Settings(config_dir=tmp_path), console=False)
    logging.getLogger("spark.test").warning("x %s ghp_abcdefghijklmnopqrstu", "supersecret123")
    err = capsys.readouterr().err
    assert "supersecret123" not in err and "ghp_abc" not in err
    assert REDACTED in err


def test_child_env_drops_secrets_but_keeps_requested() -> None:
    env = {
        "PATH": "/bin",
        "GITHUB_TOKEN": "ghp_x",
        "ANTHROPIC_API_KEY": "sk-ant-x",
        "COMPOSIO_MCP_URL": "https://backend.composio.dev/v3/mcp/abc?user_id=me",
        "HOME": "/home/me",
    }
    out = clog.child_env(environ=env)
    assert out == {"PATH": "/bin", "HOME": "/home/me"}
    assert "GITHUB_TOKEN" in clog.child_env(keep=["github_token"], environ=env)


def test_mcp_urls_count_as_secrets() -> None:
    assert clog.is_secret_env_name("COMPOSIO_MCP_URL")
    assert clog.is_secret_env_name("SOMETHING_MCP_URL")


def test_setup_logging_rejects_bad_level(tmp_path: Path, restore_root: None) -> None:
    with pytest.raises(ValueError):
        clog.setup_logging(Settings(config_dir=tmp_path, logging={"level": "LOUD"}))


# --- transcripts ---------------------------------------------------------------------------


def test_transcript_writes_user_and_assistant(tmp_path: Path, secret_env: None) -> None:
    settings = Settings(config_dir=tmp_path)
    with clog.get_transcript_logger(settings) as tlog:
        assert tlog.path == tmp_path / "logs" / "transcripts.log"
        assert not tlog.path.exists()  # lazy
        tlog.user("what is the capital\nof Japan")
        tlog.assistant(f"Tokyo. Also your key is {ANTHROPIC}")
    lines = tlog.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("user: what is the capital of Japan")
    assert lines[1].endswith(f"assistant: Tokyo. Also your key is {REDACTED}")


def test_transcript_disabled_creates_nothing(tmp_path: Path) -> None:
    settings = Settings(config_dir=tmp_path, logging={"transcripts": False})
    tlog = TranscriptLog.from_settings(settings)
    tlog.user("hello")
    tlog.assistant("hi")
    tlog.close()
    assert not (tmp_path / "logs").exists()


def test_transcript_rotation(tmp_path: Path) -> None:
    settings = Settings(
        config_dir=tmp_path, logging={"max_bytes": 300, "backup_count": 2, "dir": "t/logs"}
    )
    tlog = TranscriptLog.from_settings(settings)
    try:
        for i in range(60):
            tlog.user(f"utterance number {i} with some padding text")
    finally:
        tlog.close()
    log_dir = tmp_path / "t" / "logs"
    names = sorted(p.name for p in log_dir.iterdir())
    assert names == ["transcripts.log", "transcripts.log.1", "transcripts.log.2"]
    for p in log_dir.iterdir():
        assert p.stat().st_size <= 300
    assert "utterance number 59" in (log_dir / "transcripts.log").read_text(encoding="utf-8")


def test_transcript_close_is_idempotent_and_reopens(tmp_path: Path) -> None:
    tlog = TranscriptLog(tmp_path / "x.log")
    tlog.close()
    tlog.user("a")
    tlog.close()
    tlog.close()
    tlog.assistant("b")
    tlog.close()
    assert len((tmp_path / "x.log").read_text(encoding="utf-8").splitlines()) == 2
