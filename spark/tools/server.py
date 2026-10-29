"""The ``spark`` in-process SDK MCP server: current_time, timers, spark_status, start_project, and
(with a memory store) the connector-notes tools.

``claude_agent_sdk`` is imported lazily inside :func:`build_spark_tools` so importing this module
stays cheap (tests and ``--text`` startup don't pay for it until the brain is built).

Every tool returns short speakable text, never JSON or IDs. Bad input comes back as an
``is_error`` result with a speakable message rather than an exception.

``save_connector_note`` / ``rewrite_connector_notes`` (P5.9) write Spark's own Markdown notes
about a connected system under ``memory/connectors/``: local, undoable by the user, so like the
timers they are pre-approved, never confirmed. They only exist when a memory store is passed.

``start_project`` (spark/tools/project.py) is the only side-effecting tool. It is left out of the
pre-approved names so the PreToolUse confirmation hook asks first, and
:func:`spark_confirmation_summarizers` gives that hook its one spoken line.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from spark import __version__
from spark.tools.clock_tools import NowFn, current_time_text, local_now
from spark.tools.project import ProjectError, ProjectService, summarize_start_project
from spark.tools.status import StatusProvider, describe_timers, spark_status_text
from spark.tools.timers import TimerError, TimerService, speak_duration

if TYPE_CHECKING:
    from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool

    from spark.activity import ActivityLog
    from spark.config import ProjectsConfig
    from spark.memory import MemoryStore

log = logging.getLogger(__name__)

SERVER_NAME = "spark"

CURRENT_TIME = "current_time"
SET_TIMER = "set_timer"
LIST_TIMERS = "list_timers"
CANCEL_TIMER = "cancel_timer"
SPARK_STATUS = "spark_status"
START_PROJECT = "start_project"
SAVE_CONNECTOR_NOTE = "save_connector_note"
REWRITE_CONNECTOR_NOTES = "rewrite_connector_notes"
#: Read-only (pre-approved) tools that exist only with a memory store.
MEMORY_TOOLS: tuple[str, ...] = (SAVE_CONNECTOR_NOTE, REWRITE_CONNECTOR_NOTES)

# Permission classification (spec §7 "Tool permissions", §8.3). The spoken-confirmation rule is
# about tools that send, create, update or delete things in the outside world. set_timer and
# cancel_timer only touch Spark's own in-memory timers, are trivially undone, and asking "Should I
# go ahead?" before "set a 5 minute timer" would be absurd in a voice UI, so they are deliberately
# classed as read-only (pre-approved, no confirmation). start_project creates a folder, a GitHub
# repo and an issue, so it is side-effecting: never pre-approved, always confirmed by voice.
READ_ONLY_TOOLS: tuple[str, ...] = (
    CURRENT_TIME,
    SET_TIMER,
    LIST_TIMERS,
    CANCEL_TIMER,
    SPARK_STATUS,
)
SIDE_EFFECTING_TOOLS: tuple[str, ...] = (START_PROJECT,)
TOOL_NAMES: tuple[str, ...] = READ_ONLY_TOOLS + SIDE_EFFECTING_TOOLS

_GENERIC_FAILURE = "Sorry, something went wrong with that."


def full_tool_name(tool: str, server: str = SERVER_NAME) -> str:
    """The name the model and ``allowed_tools`` see: ``mcp__spark__current_time``."""
    return f"mcp__{server}__{tool}"


def spark_tool_names(server: str = SERVER_NAME) -> list[str]:
    """Full names of the spark server's read-only (pre-approved) tools."""
    return [full_tool_name(t, server) for t in READ_ONLY_TOOLS]


def spark_side_effecting_names(server: str = SERVER_NAME) -> list[str]:
    """Full names of the spark tools that need a spoken confirmation."""
    return [full_tool_name(t, server) for t in SIDE_EFFECTING_TOOLS]


Summarizer = Callable[[Mapping[str, Any]], str]


def spark_confirmation_summarizers(
    projects: ProjectsConfig | ProjectService | None = None, server: str = SERVER_NAME
) -> dict[str, Summarizer]:
    """``{full tool name: fn(tool_input) -> one spoken line}`` for the confirmation hook (P3.1).

    Pass the same :class:`ProjectService` (or ``settings.projects``) the server was built with,
    so the line describes exactly what will run."""
    if isinstance(projects, ProjectService):
        return {full_tool_name(START_PROJECT, server): projects.summarize}
    from spark.config import ProjectsConfig

    cfg = projects if projects is not None else ProjectsConfig()
    return {full_tool_name(START_PROJECT, server): lambda args: summarize_start_project(args, cfg)}


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _guarded(handler: Handler) -> Handler:
    """Turn TimerError/ProjectError/MemoryWriteError into a speakable is_error result and hide
    any other crash."""
    from spark.memory import MemoryWriteError

    async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return await handler(args or {})
        except (TimerError, ProjectError, MemoryWriteError) as err:
            return _error(str(err))
        except Exception:
            log.exception("spark tool %s failed", getattr(handler, "__name__", "?"))
            return _error(_GENERIC_FAILURE)

    wrapper.__name__ = getattr(handler, "__name__", "wrapper")
    return wrapper


_SET_TIMER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "duration": {
            "type": ["number", "string"],
            "description": (
                "How long: a number of seconds (300) or text like '5 minutes', '1h30m', "
                "'an hour and a half'. Between 1 second and 24 hours."
            ),
        },
        "label": {
            "type": "string",
            "description": "Optional short name the user gave the timer, e.g. 'pasta'.",
        },
    },
    "required": ["duration"],
}

_CANCEL_TIMER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "timer": {
            "type": "string",
            "description": (
                "Which timer: its name ('pasta'), its number ('2'), or 'all'. "
                "Leave out when only one timer is running."
            ),
        },
    },
}

_START_PROJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The project name the user said, e.g. 'weather-bot' or 'weather bot'.",
        },
        "private": {
            "type": "boolean",
            "description": "Make the GitHub repo private. Leave out to use the configured default.",
        },
        "open_claude_code": {
            "type": "boolean",
            "description": "Open a Claude Code window in the new folder. Only if the user asks.",
        },
        "description": {
            "type": "string",
            "description": "Optional one-line description of the project.",
        },
    },
    "required": ["name"],
}

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

_SAVE_NOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "description": "The connector the note is about, as in its tool names, e.g. WAREHOUSE.",
        },
        "note": {
            "type": "string",
            "description": (
                "One fact about the system's structure that will save exploring next time: "
                "which database, schema, table and columns hold something, a filter or join "
                "that worked, a query pattern. Not the user's data values, not secrets."
            ),
        },
    },
    "required": ["server", "note"],
}

_REWRITE_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "server": {"type": "string", "description": "The connector, e.g. WAREHOUSE."},
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The complete new list of notes (everything still true, corrected).",
        },
    },
    "required": ["server", "notes"],
}


def _default_project_service() -> ProjectService:
    from spark.config import ProjectsConfig

    log.info("start_project: no project settings passed, using the config defaults")
    return ProjectService.from_config(ProjectsConfig())


def build_spark_tools(
    timers: TimerService,
    status_provider: StatusProvider | None = None,
    now: NowFn = local_now,
    projects: ProjectService | None = None,
    memory: MemoryStore | None = None,
    connectors: Callable[[], list[str]] | None = None,
    activity: ActivityLog | None = None,
) -> list[SdkMcpTool[Any]]:
    """The ``@tool``-decorated tools. Each exposes its coroutine as ``.handler``.

    ``projects`` runs ``start_project``. The app should pass
    ``ProjectService.from_settings(settings)``; tests pass one built with fakes. Without it the
    config defaults are used."""
    from claude_agent_sdk import ToolAnnotations, tool

    project_service = projects if projects is not None else _default_project_service()

    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    local_write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    @tool(
        CURRENT_TIME,
        "Get the user's current local date, time, weekday and timezone.",
        _EMPTY_SCHEMA,
        annotations=read_only,
    )
    @_guarded
    async def current_time(args: dict[str, Any]) -> dict[str, Any]:
        return _text(current_time_text(now))

    @tool(
        SET_TIMER,
        "Start a countdown timer. When it ends, Spark chimes and says the timer is done.",
        _SET_TIMER_SCHEMA,
        annotations=local_write,
    )
    @_guarded
    async def set_timer(args: dict[str, Any]) -> dict[str, Any]:
        if "duration" not in args:
            return _error("How long should the timer be?")
        timer = timers.set(args["duration"], args.get("label"))
        return _text(f"Started {timer.spoken_ref} for {speak_duration(timer.seconds)}.")

    @tool(
        LIST_TIMERS,
        "List running timers with the time left on each.",
        _EMPTY_SCHEMA,
        annotations=read_only,
    )
    @_guarded
    async def list_timers(args: dict[str, Any]) -> dict[str, Any]:
        return _text(describe_timers(timers))

    @tool(
        CANCEL_TIMER,
        "Cancel a running timer by name or number, or all timers.",
        _CANCEL_TIMER_SCHEMA,
        annotations=local_write,
    )
    @_guarded
    async def cancel_timer(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("timer") or "").strip()
        if not len(timers):
            return _error("There are no timers running.")
        if query.lower() in {"all", "all timers", "every timer", "everything"}:
            cancelled = timers.cancel_all()
            count = len(cancelled)
            return _text(
                "Cancelled your timer." if count == 1 else f"Cancelled all {count} timers."
            )
        matches = timers.find(query)
        if not matches:
            return _error(f"I don't have a timer called {query}. {describe_timers(timers)}")
        if len(matches) > 1:
            refs = [t.spoken_ref for t in matches]
            names = ", ".join(refs[:-1]) + " or " + refs[-1]
            return _error(f"Which one, {names}?")
        cancelled_timer = timers.cancel(matches[0].id)
        if cancelled_timer is None:
            return _error("That timer already finished.")
        return _text(f"Cancelled {cancelled_timer.spoken_ref}.")

    @tool(
        SPARK_STATUS,
        "What Spark is doing right now: its state, task in progress, backends, uptime and "
        "running timers. Use for 'what are you working on' or 'status'.",
        _EMPTY_SCHEMA,
        annotations=read_only,
    )
    @_guarded
    async def spark_status(args: dict[str, Any]) -> dict[str, Any]:
        return _text(await spark_status_text(status_provider, timers))

    @tool(
        START_PROJECT,
        "Start a new software project: a local folder with git and a first commit, a GitHub "
        "repo, a tracking issue, and optionally a Claude Code window there. The user confirms "
        "by voice before it runs. Tell them the result briefly.",
        _START_PROJECT_SCHEMA,
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
        ),
    )
    @_guarded
    async def start_project(args: dict[str, Any]) -> dict[str, Any]:
        plan = project_service.plan(args)
        result = await asyncio.to_thread(project_service.run, plan)
        text = result.summary()
        return _text(text) if result.ok else _error(text)

    tools = [current_time, set_timer, list_timers, cancel_timer, spark_status, start_project]
    if memory is None:
        return tools

    def server_name(raw: Any) -> str:
        """The configured name for ``raw`` (case-insensitive), or an error the model can fix."""
        from spark.memory import MemoryWriteError

        name = str(raw or "").strip()
        if name.startswith("mcp__"):
            name = name[len("mcp__") :].partition("__")[0]
        known = connectors() if connectors is not None else []
        for k in known:
            if k.lower() == name.lower():
                return k
        if not known:
            return name
        raise MemoryWriteError(
            f"There is no connector called {name}. Use one of: {', '.join(known)}."
        )

    @tool(
        SAVE_CONNECTOR_NOTE,
        "Save one short note about where things are in a connected system, so next time you "
        "can query directly instead of exploring. Use it after you had to discover tables, "
        "columns or a query pattern to answer. Structure only, never data values or secrets.",
        _SAVE_NOTE_SCHEMA,
        annotations=local_write,
    )
    @_guarded
    async def save_connector_note(args: dict[str, Any]) -> dict[str, Any]:
        server = server_name(args.get("server"))
        result = memory.add_connector_note(server, str(args.get("note") or ""))
        if activity is not None:
            activity.record("memory", action="note", server=server, text=args.get("note"))
        return _text(result)

    @tool(
        REWRITE_CONNECTOR_NOTES,
        "Replace all notes about a connected system with a corrected, shorter list. Use it when "
        "a note turned out wrong or the notes are full.",
        _REWRITE_NOTES_SCHEMA,
        annotations=local_write,
    )
    @_guarded
    async def rewrite_connector_notes(args: dict[str, Any]) -> dict[str, Any]:
        server = server_name(args.get("server"))
        notes = args.get("notes") or []
        result = memory.rewrite_connector_notes(server, notes)
        if activity is not None:
            activity.record("memory", action="rewrite", server=server, notes=notes)
        return _text(result)

    return [*tools, save_connector_note, rewrite_connector_notes]


def build_spark_server(
    timers: TimerService,
    status_provider: StatusProvider | None = None,
    now: NowFn = local_now,
    version: str = __version__,
    projects: ProjectService | None = None,
    memory: MemoryStore | None = None,
    connectors: Callable[[], list[str]] | None = None,
    activity: ActivityLog | None = None,
) -> tuple[McpSdkServerConfig, list[str], list[str]]:
    """Build the in-process ``spark`` server.

    Returns ``(server_config, read_only_tool_names, side_effecting_tool_names)`` where names are
    full ``mcp__spark__<tool>`` names. Pass ``{"spark": server_config}`` in ``mcp_servers`` (the
    key must be ``SERVER_NAME`` for the names to match) and the read-only names in
    ``allowed_tools``. The side-effecting names (``mcp__spark__start_project``) go to
    ``McpSetup.add_sdk_server(..., side_effecting=...)`` so they are confirmed, never
    pre-approved.
    """
    from claude_agent_sdk import create_sdk_mcp_server

    tools = build_spark_tools(timers, status_provider, now, projects, memory, connectors, activity)
    config = create_sdk_mcp_server(name=SERVER_NAME, version=version, tools=tools)
    return (config, *spark_tool_lists(memory=memory is not None))


def spark_tool_lists(*, memory: bool) -> tuple[list[str], list[str]]:
    """``(read_only, side_effecting)`` full names of the spark tools, for ``add_sdk_server``.
    Both brains use it, so the classification lives in one place."""
    read_only = READ_ONLY_TOOLS + (MEMORY_TOOLS if memory else ())
    return (
        [full_tool_name(t) for t in read_only],
        [full_tool_name(t) for t in SIDE_EFFECTING_TOOLS],
    )
