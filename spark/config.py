"""Spark settings: ``config.yaml`` < ``.env`` < environment < explicit overrides.

Environment overrides use the ``SPARK_`` prefix with ``__`` between nested keys, e.g.
``SPARK_STT__BACKEND=deepgram``. Secrets (API keys) are never stored in settings; backends
read them from the environment variable named in their options (``*_env`` fields).

The project was called Chip, so a ``CHIP_*`` variable still works for one release: it is copied
to its ``SPARK_*`` name (an explicit ``SPARK_*`` wins) with a deprecation warning.
"""

from __future__ import annotations

import contextvars
import logging
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

log = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("config.yaml")
_yaml_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar("_yaml_path", default=None)
LEGACY_ENV_PREFIX = "CHIP_"
ENV_PREFIX = "SPARK_"
_warned_legacy: set[str] = set()


def apply_legacy_env(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Copy each ``CHIP_X`` to ``SPARK_X`` unless ``SPARK_X`` is already set; return the names
    copied. Warns once per name (names only, never values)."""
    env = os.environ if environ is None else environ
    copied = []
    for key in [k for k in env if k.upper().startswith(LEGACY_ENV_PREFIX)]:
        new = ENV_PREFIX + key[len(LEGACY_ENV_PREFIX) :]
        if new in env:
            continue
        env[new] = env[key]
        copied.append(key)
        if key not in _warned_legacy:
            _warned_legacy.add(key)
            log.warning("%s is deprecated; rename it to %s (support ends after 0.1)", key, new)
    return copied


class _Section(BaseModel):
    model_config = {"extra": "forbid"}


# --- audio -----------------------------------------------------------------------------------


class AudioConfig(_Section):
    sample_rate: int = 16_000
    input_backend: str = "sounddevice"
    output_backend: str = "sounddevice"
    input_device: str | int | None = None
    output_device: str | int | None = None
    host_api: str | None = "Windows WASAPI"
    block_size: int = 512
    ring_seconds: float = 10.0
    earcon_volume: float = 0.5


# --- wake word -------------------------------------------------------------------------------


class PorcupineConfig(_Section):
    access_key_env: str = "PICOVOICE_ACCESS_KEY"
    keyword_path: str = "assets/wakewords/hey_spark_windows.ppn"
    sensitivity: float = 0.5


class OpenWakeWordConfig(_Section):
    model_path: str | None = None
    model_name: str = "hey_jarvis"
    threshold: float = 0.5


class WakeWordOptions(_Section):
    porcupine: PorcupineConfig = Field(default_factory=PorcupineConfig)
    openwakeword: OpenWakeWordConfig = Field(default_factory=OpenWakeWordConfig)


class WakeWordConfig(_Section):
    # MVP: custom openWakeWord "Hey Spark" model. The spec default is porcupine (no Picovoice key).
    backend: str = "openwakeword"
    options: WakeWordOptions = Field(default_factory=WakeWordOptions)


# --- VAD and recorder ------------------------------------------------------------------------


class SileroConfig(_Section):
    threshold: float = 0.5


class VADOptions(_Section):
    silero: SileroConfig = Field(default_factory=SileroConfig)


class VADConfig(_Section):
    backend: str = "silero"
    options: VADOptions = Field(default_factory=VADOptions)


class RecorderConfig(_Section):
    end_silence_ms: int = 700
    pre_roll_ms: int = 300
    min_speech_ms: int = 150
    max_utterance_s: float = 15.0
    no_speech_timeout_s: float = 30.0  # after the wake word: time to think before asking


# --- STT -------------------------------------------------------------------------------------


class FasterWhisperConfig(_Section):
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 1
    language: str | None = "en"


class DeepgramConfig(_Section):
    api_key_env: str = "DEEPGRAM_API_KEY"
    model: str = "nova-3"
    language: str = "en"
    timeout_s: float = 10.0


class STTOptions(_Section):
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)
    deepgram: DeepgramConfig = Field(default_factory=DeepgramConfig)


class STTConfig(_Section):
    backend: str = "faster_whisper"
    options: STTOptions = Field(default_factory=STTOptions)


# --- TTS -------------------------------------------------------------------------------------


class PiperConfig(_Section):
    voice: str = "en_US-ljspeech-medium"
    length_scale: float = 1.0
    speaker_id: int | None = None


class KokoroConfig(_Section):
    model_file: str = "kokoro-v1.0.onnx"
    voices_file: str = "voices-v1.0.bin"
    voice: str = "bm_george"
    speed: float = 1.0
    lang: str | None = None  # None: from the voice (a..: en-us, b..: en-gb)


class ElevenLabsConfig(_Section):
    api_key_env: str = "ELEVENLABS_API_KEY"
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model: str = "eleven_flash_v2_5"


class TTSOptions(_Section):
    piper: PiperConfig = Field(default_factory=PiperConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)


class TTSConfig(_Section):
    backend: str = "kokoro"
    options: TTSOptions = Field(default_factory=TTSOptions)


# --- brain -----------------------------------------------------------------------------------


class OpenAICompatConfig(_Section):
    """``brain.openai_compat``: any OpenAI-compatible Chat Completions endpoint (P5.7e).

    ``provider`` picks a preset (spark/brain/openai_presets.py) for the base URL, the key's env
    var and suggested models; every field set here overrides it. The key itself only ever comes
    from the environment or ``.env``: ``api_key_env`` is the variable's NAME."""

    provider: str = Field(
        "openai",
        pattern="^(openai|azure|openrouter|groq|together|ollama|lmstudio|hermes|vllm|custom)$",
    )
    base_url: str | None = None  # None = the preset's
    api_key_env: str | None = None  # None = the preset's (e.g. OPENAI_API_KEY); "" = no key
    api_version: str | None = None  # Azure classic endpoints: sent as ?api-version=
    extra_headers: dict[str, str] = Field(default_factory=dict)  # values may use ${VAR}
    extra_body: dict[str, Any] = Field(default_factory=dict)  # e.g. {reasoning_effort: minimal}
    # Model names on this provider (None = the preset's suggestion). Kept apart from the
    # Claude ids above so switching backends needs no edits.
    fast_model: str | None = None
    deep_model: str | None = None
    worker_model: str | None = None  # the researcher; None = fast_model
    timeout_s: float = Field(60.0, gt=0)  # per request: connect + time between streamed chunks
    max_tokens: int | None = Field(1024, ge=1)  # per reply; None = the provider's default
    temperature: float | None = Field(None, ge=0, le=2)  # None = the provider's default
    parallel_tool_calls: bool | None = None  # None = don't send (not every server knows it)
    max_tool_result_chars: int = Field(16_000, ge=500)  # longer results are cut for the model


class BrainConfig(_Section):
    backend: str = "claude_agent"  # claude_agent | openai_compat | mock
    # auto: fast/deep routing per request. fast / deep: every request on that model (P5.7c).
    # The desktop window and voice ("use the deep model") change it; the choice is kept in
    # prefs.json, which overrides this.
    mode: str = Field("auto", pattern="^(auto|fast|deep)$")
    fast_model: str = "claude-haiku-5-5"
    deep_model: str = "claude-opus-5-5"
    # The researcher worker (P5.7d): a sub-agent on a faster model that does tool-heavy lookups
    # in the connected systems in the background and reports back to the voice model.
    workers: bool = True
    worker_model: str | None = None  # None = fast_model
    worker_max_turns: int = Field(25, ge=1)
    idle_reset_s: float = 600.0
    # Agent steps (tool rounds) per request. Over budget, the tool gate tells the model to stop
    # and answer with what it has, so the user hears an answer instead of an error.
    max_turns: int = Field(8, ge=1)  # fast route
    deep_max_turns: int = Field(25, ge=1)  # deep route (data lookups need schema discovery)
    connect_timeout_s: float = 60.0
    prime_on_connect: bool = True
    cli_path: str | None = None
    new_topic_phrases: list[str] = Field(
        default_factory=lambda: ["new topic", "start over", "forget that", "reset conversation"]
    )
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)


# --- connectors (MCP) ------------------------------------------------------------------------
# How these keys map to SDK options is documented in spark/brain/mcp_config.py and
# docs/connectors.md ("Implemented in P2.2").

_WILDCARD_CHARS = frozenset("*?")


def _tool_part(entry: str) -> str | None:
    """The tool part of a list entry: ``X`` for ``X`` or ``mcp__srv__X``; None for ``mcp__srv``."""
    if entry.startswith("mcp__"):
        _, sep, tool = entry[len("mcp__") :].partition("__")
        return tool if sep else None
    return entry


class ConnectorServerConfig(_Section):
    """Per-server tool policy. Entries are bare tool names (``get_me``) or full
    ``mcp__<server>__<tool>`` names, optionally with ``*``/``?`` globs."""

    enabled: bool = True
    allowed_tools: list[str] = Field(default_factory=list)
    side_effecting: list[str] = Field(default_factory=list)
    destructive: list[str] = Field(default_factory=list)
    allow_destructive: bool = False
    # "deep": a request that names this server (or any route_keywords), and every later turn of
    # a conversation that has used it, goes to the deep model with its larger step budget.
    route: str = Field("auto", pattern="^(auto|deep)$")
    route_keywords: list[str] = Field(default_factory=list)

    @field_validator("allowed_tools", "side_effecting", "destructive")
    @classmethod
    def _non_empty_entries(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value]
        if any(not v for v in cleaned):
            raise ValueError("tool names must be non-empty strings")
        return cleaned

    @field_validator("allowed_tools")
    @classmethod
    def _no_server_wide_auto_approval(cls, value: list[str]) -> list[str]:
        for entry in value:
            tool = _tool_part(entry)
            if tool is None or not tool or set(tool) <= _WILDCARD_CHARS:
                raise ValueError(
                    f"allowed_tools entry {entry!r} would auto-approve a whole server; "
                    "list the read-only tools explicitly (prefix globs like GMAIL_FETCH_* are ok)"
                )
        return value


def _default_connector_servers() -> dict[str, ConnectorServerConfig]:
    """Defaults from docs/connectors.md §3.2. Keep in sync with config.yaml."""
    return {
        "github": ConnectorServerConfig(
            enabled=False,
            allowed_tools=[
                "get_me",
                "list_pull_requests",
                "pull_request_read",
                "search_pull_requests",
                "list_issues",
                "issue_read",
                "search_issues",
                "get_file_contents",
                "list_commits",
                "get_commit",
                "list_branches",
            ],
            side_effecting=[
                "create_repository",
                "issue_write",
                "add_issue_comment",
                "create_pull_request",
            ],
            destructive=[
                "merge_pull_request",
                "delete_repository",
                "delete_file",
                "push_files",
                "create_or_update_file",
                "fork_repository",
                "actions_run_trigger",
                "update_pull_request_branch",
            ],
        ),
        **_default_claudeai_servers(),
    }


#: Gmail tools of the claude.ai connector (verified live 2026-10-05). Read-only ones run, the
#: draft and label ones ask first, everything else (sending, trash, spam, label admin) is off.
CLAUDEAI_GMAIL_READ = (
    "search_threads",
    "get_thread",
    "get_message",
    "list_labels",
    "list_drafts",
    "get_draft",
)
CLAUDEAI_GMAIL_ASK = (
    "create_draft",
    "update_draft",
    "label_thread",
    "unlabel_thread",
    "label_message",
    "unlabel_message",
)
CLAUDEAI_GMAIL_OFF = (
    "send_message",
    "reply",
    "forward",
    "delete_draft",
    "trash_thread",
    "untrash_thread",
    "trash_message",
    "untrash_message",
    "mark_thread_spam",
    "unmark_thread_spam",
    "mark_message_spam",
    "unmark_message_spam",
    "apply_sensitive_thread_label",
    "apply_sensitive_message_label",
    "update_message_labels",
    "create_label",
    "update_label",
    "delete_label",
)
CLAUDEAI_CALENDAR_READ = (
    "list_events",
    "get_event",
    "list_calendars",
    "search_events",
    "suggest_time",
)
CLAUDEAI_CALENDAR_ASK = ("create_event", "update_event", "respond_to_event")
CLAUDEAI_CALENDAR_OFF = ("delete_event",)


def _default_claudeai_servers() -> dict[str, ConnectorServerConfig]:
    """claude.ai account connectors (docs/connectors.md §4.7). The CLI injects them, so they
    need no mcp_servers.json entry. Keep in sync with config.yaml."""
    return {
        # off by default: they ride on a claude.ai login in the Claude CLI, which a public
        # app must not offer (Agent SDK terms); for personal setups, see docs/connectors.md
        "claude_ai_Gmail": ConnectorServerConfig(
            enabled=False,
            allowed_tools=list(CLAUDEAI_GMAIL_READ),
            side_effecting=list(CLAUDEAI_GMAIL_ASK),
            destructive=list(CLAUDEAI_GMAIL_OFF),
            route="deep",
            route_keywords=["email", "emails", "gmail", "inbox", "mail"],
        ),
        "claude_ai_Google_Calendar": ConnectorServerConfig(
            enabled=False,
            allowed_tools=list(CLAUDEAI_CALENDAR_READ),
            side_effecting=list(CLAUDEAI_CALENDAR_ASK),
            destructive=list(CLAUDEAI_CALENDAR_OFF),
            route="deep",
            route_keywords=["calendar", "meeting", "meetings"],
        ),
    }


class ConnectorsConfig(_Section):
    mcp_config: str = "mcp_servers.json"
    # Servers added in the desktop Connectors window (spark/brain/connector_store.py).
    ui_file: str = "connectors.json"
    # claude.ai account connectors (servers keyed claude_ai_*) load only when at least one is
    # enabled and this is false. True forces them all off (ENABLE_CLAUDEAI_MCP_SERVERS=false).
    block_claudeai_connectors: bool = False
    # The claude.ai connectors the CLI reported last (app-managed): unconfigured ones are hidden.
    claudeai_state_file: str = "claudeai_connectors.json"
    servers: dict[str, ConnectorServerConfig] = Field(default_factory=_default_connector_servers)

    @field_validator("servers")
    @classmethod
    def _valid_server_keys(
        cls, value: dict[str, ConnectorServerConfig]
    ) -> dict[str, ConnectorServerConfig]:
        for key in value:
            if (
                not key
                or "__" in key
                or not all((c.isascii() and c.isalnum()) or c in "_-" for c in key)
            ):
                raise ValueError(
                    f"connector server key {key!r} must use only letters, digits, '_' or '-' "
                    "and must not contain '__' (tool names are split on it)"
                )
        return value


# --- projects (start_project, Phase 3) -------------------------------------------------------


class ProjectsConfig(_Section):
    """The "start a new project" flow (spark/tools/project.py). Keep in sync with config.yaml."""

    root_dir: str = "~/Projects"
    repo_host: str = Field("gh", pattern=r"^(gh|none)$")
    github_owner: str | None = Field(None, pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
    github_private: bool = True
    tracker: str = Field("github_issue", pattern=r"^(github_issue|none)$")
    open_claude_code: bool = False
    gh_path: str = "gh"
    claude_path: str = "claude"
    command_timeout_s: float = Field(60.0, gt=0)


# --- app, logging, latency -------------------------------------------------------------------


class AppConfig(_Section):
    name: str = "Spark"  # what the assistant calls itself; also stripped as a wake-word prefix
    # other spellings speech-to-text may produce for the name (e.g. "Sparks" for "Spark")
    name_aliases: list[str] = Field(default_factory=list)
    # what you say to wake it, shown in the desktop window (e.g. "Hey Spark"); empty = the name
    wake_phrase: str = ""
    # how it talks (spark/persona.py): standard | butler
    persona: str = Field("standard", pattern="^(standard|butler)$")
    address_as: str = "sir"  # what the butler calls you ("" = no title)
    # say a short reply ("Yes, sir?") instead of the wake chime; None = the persona's default
    wake_reply: bool | None = None
    # Deprecated: use attention.awake_s. When set, it overrides it (old configs keep working).
    follow_up_s: float | None = None
    barge_in: bool = False
    earcons: bool = True
    thinking_earcon: bool = True
    sounds_dir: str = "assets/sounds"
    prefs_file: str = "prefs.json"  # choices made in the app (model mode, models), app-managed

    @property
    def wake_names(self) -> tuple[str, ...]:
        """Lowercase names a transcript may start with: the name and its aliases."""
        names = [self.name, *self.name_aliases]
        return tuple(dict.fromkeys(n.strip().lower() for n in names if n.strip()))

    @property
    def wake_prompt(self) -> str:
        """The phrase to show the user: ``wake_phrase``, or the name when it's unset."""
        return self.wake_phrase.strip() or self.name


class AttentionConfig(_Section):
    """Awake and asleep (Phase 5, P5.2-P5.6). Keep in sync with config.yaml."""

    # After a reply Spark stays awake this long with no wake word; every request restarts it.
    awake_s: float = Field(180.0, gt=0)
    sleep_earcon: str = "sleep"  # played when the awake window ends ("" = none)
    # "Is this for me?" check on speech heard while awake (spark/attention.py, P5.4).
    addressee: str = Field("rules", pattern="^(rules|fast_model|off)$")
    addressee_timeout_s: float = Field(1.5, gt=0)  # fast_model: then the rules decide
    question_window_s: float = Field(30.0, ge=0)  # after Spark asks a question, any reply counts
    # Each wake starts a fresh Claude session (P5.6). The last conversation's exchanges are
    # carried into it when it ended less than carry_over_s ago.
    carry_over_s: float = Field(1800.0, ge=0)
    carry_over_exchanges: int = Field(6, ge=0)


class ActivityConfig(_Section):
    """The activity log (spark/activity.py, P5.5). Keep in sync with config.yaml."""

    enabled: bool = True
    dir: str = "history"  # one JSONL file per day, next to config.yaml (or the user data folder)
    tool_results: str = Field("full", pattern="^(full|excerpt|meta)$")
    excerpt_chars: int = Field(2000, ge=100)
    retention_days: int = Field(0, ge=0)  # 0 = keep forever; the user deletes


class ShortcutsConfig(_Section):
    """Voice shortcuts (spark/shortcuts.py, P5.7a). Keep in sync with config.yaml."""

    enabled: bool = True
    file: str = "shortcuts.json"  # next to config.yaml; edited in the Shortcuts window
    max_extra_words: int = Field(6, ge=0)  # words allowed beyond the trigger phrases
    seed_examples: bool = True  # write a few examples when the file doesn't exist


class MemoryConfig(_Section):
    """Spark's memory (spark/memory.py, P5.9). Keep in sync with config.yaml."""

    enabled: bool = True
    dir: str = "memory"  # Markdown files next to config.yaml (or in the user data folder)
    note_chars: int = Field(600, ge=50)  # longest single connector note
    connector_file_chars: int = Field(6000, ge=500)  # past this, Spark rewrites the notes shorter
    briefing_chars: int = Field(8000, ge=0)  # connector notes in a new session's briefing


class DesktopConfig(_Section):
    """The desktop window (spec §10 Phase 4). Keep in sync with config.yaml."""

    autostart_listening: bool = False  # start listening as soon as the window opens
    log_file: str = "spark.log"  # app log under logging.dir (the window has no console)


class LoggingConfig(_Section):
    level: str = "INFO"
    dir: str = "logs"
    transcripts: bool = True
    transcript_file: str = "transcripts.log"
    max_bytes: int = 1_000_000
    backup_count: int = 5


class LatencyConfig(_Section):
    enabled: bool = True
    file: str = "latency.jsonl"


class DebugAudioConfig(_Section):
    enabled: bool = False
    dir: str = "debug_audio"


# --- spoken confirmation (spark/brain/confirm.py, Phase 3) ------------------------------------


class ConfirmConfig(_Section):
    """Confirming side-effecting tools by voice. Keep in sync with config.yaml."""

    timeout_s: float = Field(8.0, gt=0)  # voice: time for the user to start answering
    text_timeout_s: float = Field(60.0, gt=0)  # --text: time to type y/n
    hook_timeout_s: float = Field(120.0, gt=0)  # HookMatcher timeout; the CLI gives up after it
    reprompt_unclear: bool = True  # ask "yes or no?" once more after an unclear answer
    earcon: str = "wake"  # cue played before listening for the answer ("" = none)
    yes_phrases: list[str] = Field(default_factory=list)  # extra phrases, on top of built-ins
    no_phrases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _hook_outlives_the_question(self) -> ConfirmConfig:
        need = max(self.timeout_s, self.text_timeout_s) + 30.0
        if self.hook_timeout_s < need:
            raise ValueError(
                f"confirm.hook_timeout_s must be at least {need:.0f} s (the longest answer "
                "timeout plus 30 s for speaking the question and transcribing the answer)"
            )
        return self

    @property
    def ask_budget_s(self) -> float:
        """How long the hook waits for an answer before deciding "no answer" on its own,
        a little under the CLI's hook timeout."""
        return max(1.0, self.hook_timeout_s - 10.0)


# --- fast/deep routing (spark/brain/router.py, Phase 3) ---------------------------------------

DEFAULT_DEEP_PHRASES = [
    "new project",
    "start a project",
    "summarize this week",
    "summarize the week",
    "summarise this week",
    "draft a plan",
    "make a plan",
    "step by step",
    "in detail",
    "pros and cons",
]
# Creation and multi-step work, plus every action verb that changes something outside Spark
# (send, delete, schedule, ...): the fast route may only use read-only tools (``fast_tools``),
# so a request that needs a side effect must route deep. Multi-word entries ("set up") match
# the start of a clause; words that are often nouns in questions ("open PRs?", "schedule for
# today?", "email from Sam?") are listed only with an article ("open a", "schedule a").
# Timer requests stay fast: the "timer" fast keyword is checked first.
DEFAULT_DEEP_VERBS = [
    "add",
    "analyse",
    "analyze",
    "archive",
    "assign",
    "book a",
    "book an",
    "brainstorm",
    "build",
    "cancel",
    "close",
    "commit",
    "compare",
    "create",
    "debug",
    "delete",
    "design",
    "draft",
    "edit",
    "email the",
    "forward",
    "implement",
    "investigate",
    "invite",
    "merge",
    "move",
    "open a",
    "open an",
    "organize",
    "outline",
    "plan",
    "post a",
    "post an",
    "prepare",
    "publish",
    "push",
    "refactor",
    "remove",
    "rename",
    "reply",
    "research",
    "review",
    "schedule a",
    "schedule an",
    "send",
    "set up",
    "share a",
    "share my",
    "share the",
    "start a",
    "start an",
    "start the",
    "submit",
    "summarise",
    "summarize",
    "update",
    "upload",
    "write",
]
DEFAULT_FAST_KEYWORDS = [
    "what time",
    "time is it",
    "what day",
    "what's the date",
    "today's date",
    "timer",
    "timers",
    "working on",
    "your status",
    "how are you",
    "hello",
    "thank you",
    "thanks",
    "good morning",
    "good night",
]


class RoutingConfig(_Section):
    """Fast/deep model routing (spec §8.4). Keep in sync with config.yaml."""

    enabled: bool = True  # false: every turn uses brain.fast_model
    classifier: str = Field("rules", pattern="^(rules|fast_model)$")
    classifier_timeout_s: float = Field(1.5, gt=0)  # fast_model classifier; then rules decide
    ack_after_s: float = Field(5.0, ge=0)  # deep turn, no text yet: say ack_text (0 = never)
    ack_text: str | None = None  # None: the persona's ("On it, give me a moment.")
    max_fast_words: int = Field(25, ge=1)  # longer requests go deep
    deep_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_DEEP_PHRASES))
    deep_verbs: list[str] = Field(default_factory=lambda: list(DEFAULT_DEEP_VERBS))
    fast_keywords: list[str] = Field(default_factory=lambda: list(DEFAULT_FAST_KEYWORDS))
    # Tools a fast-routed turn may use (spec §8.4 "fast path ... with a limited tool set"),
    # enforced per call by the PreToolUse gate (spark/brain/confirm.py). Empty = every read-only
    # tool (each server's allowed_tools plus Spark's read-only tools). Otherwise exactly these
    # full names ("mcp__spark__current_time"). Deep turns use the full tool set.
    fast_tools: list[str] = Field(default_factory=list)


# --- root ------------------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPARK_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
    )

    config_dir: Path = Path(".")
    models_dir: str = "./.models"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    connectors: ConnectorsConfig = Field(default_factory=ConnectorsConfig)
    projects: ProjectsConfig = Field(default_factory=ProjectsConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    debug_audio: DebugAudioConfig = Field(default_factory=DebugAudioConfig)
    confirm: ConfirmConfig = Field(default_factory=ConfirmConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    desktop: DesktopConfig = Field(default_factory=DesktopConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    shortcuts: ShortcutsConfig = Field(default_factory=ShortcutsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    # Settings given on the command line ("brain.backend" for --brain): the app's saved choices
    # (prefs.json) don't override them.
    _pinned: set[str] = PrivateAttr(default_factory=set)

    def pin(self, *keys: str) -> None:
        self._pinned.update(keys)

    def is_pinned(self, key: str) -> bool:
        return key in self._pinned

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        yaml_file = _yaml_path.get()
        if yaml_file is not None:
            sources.append(
                YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file, deep_merge=True)
            )
        return tuple(sources)

    def path(self, value: str | os.PathLike[str]) -> Path:
        """Resolve a configured path relative to the directory holding ``config.yaml``."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.config_dir / p).resolve()

    @property
    def models_path(self) -> Path:
        return self.path(self.models_dir)

    @property
    def log_path(self) -> Path:
        return self.path(self.logging.dir)

    @property
    def awake_s(self) -> float:
        """How long Spark stays awake after a reply: ``attention.awake_s``, unless the deprecated
        ``app.follow_up_s`` is set."""
        legacy = self.app.follow_up_s
        return legacy if legacy is not None else self.attention.awake_s


def load_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    load_env_file: bool = True,
    **overrides: Any,
) -> Settings:
    """Load settings from ``config_path`` (default: ``$SPARK_CONFIG`` or ./config.yaml).

    ``.env`` next to the config file is loaded into ``os.environ`` first (without replacing
    variables that are already set) so backends can read their API keys from the environment.
    ``overrides`` take precedence over everything (nested dicts are allowed).
    """
    apply_legacy_env()
    path = Path(config_path or os.environ.get("SPARK_CONFIG") or DEFAULT_CONFIG)
    if config_path is not None and not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    config_dir = path.parent.resolve()
    if load_env_file:
        from dotenv import load_dotenv

        load_dotenv(config_dir / ".env", override=False)
        apply_legacy_env()
    token = _yaml_path.set(path if path.exists() else None)
    try:
        overrides.setdefault("config_dir", config_dir)
        return Settings(**overrides)
    finally:
        _yaml_path.reset(token)
