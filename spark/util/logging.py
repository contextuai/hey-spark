"""Logging for Spark: rich console output, secret redaction, and a rotating transcript log.

Every handler Spark creates carries :class:`SecretRedactingFilter`, which masks

* the values of environment variables whose *names* look secret (``*_KEY``, ``*_TOKEN``,
  ``*_SECRET``, ``*PASSWORD*`` ...; values shorter than 6 characters are ignored), and
* well-known token shapes (``sk-ant-...``, ``ghp_...``, ``github_pat_...``, ``Bearer <token>``,
  ``xoxb-...``, ``api_key=...``),

in the fully formatted message (args included) and in exception text (spec §12).

Transcripts (what the user said, what Spark answered) go to
``settings.log_path / settings.logging.transcript_file`` through a ``RotatingFileHandler``. When
``settings.logging.transcripts`` is false nothing is written and no file or directory is created.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spark.config import Settings

__all__ = [
    "REDACTED",
    "SecretRedactingFilter",
    "TranscriptLog",
    "child_env",
    "get_transcript_logger",
    "is_secret_env_name",
    "redact",
    "secret_env_values",
    "setup_logging",
]

REDACTED = "[REDACTED]"
MIN_SECRET_LEN = 6

# Marker attribute on handlers created by setup_logging (used for idempotency).
_SPARK_HANDLER_ATTR = "_spark_console_handler"

_SECRET_NAME_RE = re.compile(
    r"""(
        (_|^)(API_?)?KEY$          # *_KEY, *_API_KEY, APIKEY
      | (_|^)TOKENS?$              # *_TOKEN
      | SECRET                     # *_SECRET, CLIENT_SECRET_ID ...
      | PASSWORD | PASSWD | (_|^)PASS$ | (_|^)PWD$
      | CREDENTIALS?$
      | (_|^)PAT$                  # personal access tokens
      | ACCESS_KEY | PRIVATE_KEY | AUTH_TOKEN
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Explicit names, in case a naming convention above ever misses them.
_KNOWN_SECRET_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "PICOVOICE_ACCESS_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY",
        "OPENAI_API_KEY",
        "SLACK_BOT_TOKEN",
        "NOTION_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "COMPOSIO_API_KEY",
        # MCP URLs can embed server ids and user ids (Composio): treat them as secrets.
        "COMPOSIO_MCP_URL",
        "COMPOSIO_MCP_URL_OUTLOOK",
    }
)

# Environment variables whose names match the patterns but never hold secrets.
_NOT_SECRET_NAMES = frozenset({"PWD", "OLDPWD"})

_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), REDACTED),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}"), REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{8,}"), REDACTED),
    (re.compile(r"\b(Bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*", re.IGNORECASE), r"\1 " + REDACTED),
    (
        re.compile(
            r"""(\b(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|token|secret|"""
            r"""password|passwd|pwd)\b["']?\s*[=:]\s*["']?)([^\s"',;&]{6,})""",
            re.IGNORECASE,
        ),
        r"\1" + REDACTED,
    ),
)


def is_secret_env_name(name: str) -> bool:
    """True if an environment variable called ``name`` probably holds a secret."""
    upper = name.upper()
    if upper in _NOT_SECRET_NAMES:
        return False
    return (
        upper in _KNOWN_SECRET_NAMES
        or upper.endswith("_MCP_URL")
        or bool(_SECRET_NAME_RE.search(upper))
    )


def child_env(keep: Iterable[str] = (), environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of the environment for child processes, without secret-looking variables.

    ``gh``, ``git`` and an interactive ``claude`` window must use their own logins, not Spark's
    keys from ``.env`` (e.g. a ``GITHUB_TOKEN`` would override ``gh``'s login, and
    ``ANTHROPIC_API_KEY`` would switch Claude Code to API billing). ``keep`` names survive.
    """
    env = os.environ if environ is None else environ
    kept = {k.upper() for k in keep}
    return {k: v for k, v in env.items() if k.upper() in kept or not is_secret_env_name(k)}


def secret_env_values(environ: Mapping[str, str] | None = None) -> list[str]:
    """Values of secret-looking env vars (at least 6 chars), longest first."""
    env = os.environ if environ is None else environ
    values = {
        value
        for name, value in env.items()
        if value and len(value) >= MIN_SECRET_LEN and is_secret_env_name(name)
    }
    return sorted(values, key=len, reverse=True)


def redact(
    text: str,
    *,
    extra_secrets: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> str:
    """Mask secrets in ``text`` (env-var secret values, extra literals, and token patterns)."""
    if not text:
        return text
    literals = secret_env_values(environ)
    literals.extend(s for s in extra_secrets if s and len(s) >= MIN_SECRET_LEN)
    for secret in sorted(set(literals), key=len, reverse=True):
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern, replacement in _TOKEN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Rewrites each record so its formatted message (and exception text) is redacted.

    The record's ``msg`` becomes the redacted, fully formatted message and ``args`` is cleared,
    so secrets passed through ``%s`` arguments are masked too. Never drops a record.
    """

    def __init__(self, extra_secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self.extra_secrets = tuple(extra_secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # malformed format args: keep something readable
            message = f"{record.msg!s} {record.args!r}"
        record.msg = redact(message, extra_secrets=self.extra_secrets)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text, extra_secrets=self.extra_secrets)
        if record.stack_info:
            record.stack_info = redact(record.stack_info, extra_secrets=self.extra_secrets)
        return True


def _level(name: str | int) -> int:
    if isinstance(name, int):
        return name
    level = logging.getLevelName(name.strip().upper())
    if not isinstance(level, int):
        raise ValueError(f"unknown log level: {name!r}")
    return level


def setup_logging(settings: Settings, *, console: bool = True) -> None:
    """Configure the root logger: a rich console handler at ``settings.logging.level``.

    Idempotent: handlers created by an earlier call are removed (and closed) before new ones are
    added, so calling this twice never duplicates output. With ``console=False`` no console
    handler is installed (the level is still applied).
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _SPARK_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()

    level = _level(settings.logging.level)
    root.setLevel(level)
    if not console:
        # Without this, Python's lastResort handler would print WARNING+ records unredacted.
        fallback = logging.StreamHandler()
        fallback.setLevel(logging.WARNING)
        fallback.addFilter(SecretRedactingFilter())
        setattr(fallback, _SPARK_HANDLER_ATTR, True)
        root.addHandler(fallback)
        return

    from rich.console import Console
    from rich.logging import RichHandler

    handler = RichHandler(
        console=Console(stderr=True),
        level=level,
        show_path=False,
        rich_tracebacks=False,
        markup=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.addFilter(SecretRedactingFilter())
    setattr(handler, _SPARK_HANDLER_ATTR, True)
    root.addHandler(handler)


class TranscriptLog:
    """Rotating, redacted log of the conversation (``user`` / ``assistant`` lines).

    The file and its directory are created lazily on the first write. When transcripts are
    disabled every method is a no-op.
    """

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = True,
        max_bytes: int = 1_000_000,
        backup_count: int = 5,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._handler: RotatingFileHandler | None = None
        # An unmanaged logger: not registered with logging.getLogger, never propagates.
        self._logger = logging.Logger(f"spark.transcript[{self.path}]", logging.INFO)
        self._logger.propagate = False

    @classmethod
    def from_settings(cls, settings: Settings) -> TranscriptLog:
        cfg = settings.logging
        return cls(
            settings.log_path / cfg.transcript_file,
            enabled=cfg.transcripts,
            max_bytes=cfg.max_bytes,
            backup_count=cfg.backup_count,
        )

    def _ensure_handler(self) -> None:
        if self._handler is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            self.path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        handler.addFilter(SecretRedactingFilter())
        self._logger.addHandler(handler)
        self._handler = handler

    def write(self, role: str, text: str) -> None:
        if not self.enabled:
            return
        self._ensure_handler()
        one_line = " ".join(text.split())  # one line per utterance
        self._logger.info("%s: %s", role, one_line)

    def user(self, text: str) -> None:
        self.write("user", text)

    def assistant(self, text: str) -> None:
        self.write("assistant", text)

    def close(self) -> None:
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def __enter__(self) -> TranscriptLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def get_transcript_logger(settings: Settings) -> TranscriptLog:
    """The transcript log configured by ``settings.logging``."""
    return TranscriptLog.from_settings(settings)
