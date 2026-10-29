"""ClaudeAgentBrain: Spark's brain on the Claude Agent SDK (spec §8).

Design (see docs/sdk_notes.md §1):

* One long-lived ``ClaudeSDKClient`` per conversation. The SDK client holds an anyio task group
  from ``connect()`` until ``disconnect()`` and must be used in the task that connected it
  (``claude_agent_sdk/client.py:58-64``). So a single **session worker** task owns the client:
  it connects, runs every turn, resets, and disconnects. ``respond()``/``reset()``/``warmup()``/
  ``close()`` talk to it over an ``asyncio.Queue``; each turn streams its text deltas back over
  its own queue.
* Text deltas come from ``StreamEvent`` (``include_partial_messages=True``) events of type
  ``content_block_delta``/``text_delta``. If a turn produced no partials, the ``TextBlock``s
  of the ``AssistantMessage`` are used instead. A turn ends at the ``ResultMessage``.
* ``claude_agent_sdk`` is imported lazily (in ``__init__``), never at module import time.
* MCP (P2.4): the external servers come from ``load_mcp_servers(settings)`` (or an injected
  ``mcp=`` :class:`McpSetup`), and the in-process ``spark`` server (time, timers, status) is
  merged in when a ``timers`` dep is given. The long-lived client keeps every stdio server
  process and in-process server alive between turns ("keep MCP servers warm", spec §9); a
  reset reconnects them with the new client. Only read-only tools are in ``allowed_tools``.
* Tool permissions (P3.1): every tool call passes the ``PreToolUse`` hook
  :class:`spark.brain.confirm.ConfirmationGate`. Read-only tools pass. Side-effecting ones are
  confirmed out loud through the injected ``confirmer``. Destructive and unknown ones are
  denied. Before it asks, the gate waits until this worker has seen the tool call in the stream
  and pushed a sentence break to the app. That way "Sure, I'll set that up." is spoken before
  "I'll create a GitHub repo called weather-bot. Should I go ahead?", not after.
* Routing (P3.2): :class:`spark.brain.router.Router` picks fast or deep per turn (logged as
  ``route: deep (<reason>)``). The worker switches the ONE session to the route's model with
  ``await client.set_model(model)`` before the query (``claude_agent_sdk/client.py:331``, a
  control request, no reconnect), so context carries across fast and deep turns.
* Fast tool limit (spec §8.4 "fast path ... with a limited tool set"): ``allowed_tools`` and
  ``mcp_servers`` are spawn-time CLI options, and the only per-server runtime switch,
  ``toggle_mcp_server``, disconnects and reconnects the server (docs/sdk_notes.md "Routing
  (P3.2)"). So the session is spawned with the full tool set and the limit is enforced per call
  in the ``PreToolUse`` gate: the worker records the active turn's route (:attr:`current_route`,
  set when the turn starts, cleared when it ends), and on a fast turn the gate denies any tool
  outside ``routing.fast_tools`` (default: the read-only tools) without asking. Deep turns get the
  full set, with side effects confirmed as usual. After such a refusal the next turn is routed
  deep (``Router.escalate``), so "ask again" reaches the deep path.
* Long deep turns: if no reply text has arrived ``routing.ack_after_s`` after a deep turn was
  queued (clock time via the injected clock, paused while a confirmation is asking the
  user), ``respond()`` yields ``routing.ack_text`` once as its own sentence, then streams the
  reply. The voice prompt asks multi-step answers to end with a one-sentence summary. A fast
  turn is acknowledged the same way, but only once it has started a tool call (a slow
  connector), so a plain quick answer never gets an "On it".
* Step budget (P5.1): each request may use ``brain.max_turns`` agent steps on the fast route and
  ``brain.deep_max_turns`` on the deep route. The CLI's own ``max_turns`` is set a little above
  the larger one as a backstop. The gate's ``precheck`` counts the turn's tool calls and, at the
  budget, refuses the next one with "answer with what you found", so the user hears an answer
  instead of an ``error_max_turns`` failure.
* Voice and worker (P5.7d): with ``brain.workers`` on and read-only connector tools present, the
  voice model gets the built-in ``Agent`` tool and a ``researcher`` sub-agent
  (``ClaudeAgentOptions.agents``) on ``brain.worker_model`` (default the fast model) holding those
  read-only tools plus the connector-notes tools. The voice model delegates multi-step lookups
  and speaks the result. Worker messages carry ``parent_tool_use_id`` and are never spoken; their
  tool calls still pass the gate (with ``agent_id``) and are logged with ``agent``. The step
  budget counts only the voice model's own calls; a worker has ``worker_max_turns``. In Auto,
  once a turn has gone deep, the rest of the conversation stays deep.
* Connector health (P5.7i): after every connect and every turn the worker asks the CLI for the
  live MCP status (``client.get_mcp_status()``). Changes are logged (status plus a short error,
  never the server config or URL) and passed to ``on_mcp_status``. A connector that is
  ``failed`` or ``needs-auth`` is mentioned once per session to the model, as a status line
  after the user's next request, so it says so instead of trying. Each wake starts a new
  session, which reconnects every server.
* claude.ai account connectors: enabled ``claude_ai_*`` servers (``McpSetup.claudeai``) are
  injected by the logged-in CLI, which needs ``ENABLE_CLAUDEAI_MCP_SERVERS=true`` and
  ``strict_mcp_config=False`` (verified live, docs/sdk_notes.md). The session is non-strict only
  then. The status check maps ``"claude.ai Gmail"`` to ``claude_ai_Gmail``, saves every claude.ai
  server it sees (:func:`spark.brain.claudeai.save_state`) so the next session hides the
  unconfigured ones, and reports only Spark's own servers. ``needs-auth`` asks the user to sign
  in on claude.ai.
* Deep connectors (P5.1): a ``route: deep`` server (``McpSetup.deep_route``) sends a request
  that names it deep, and once a conversation has used one of its tools, every later turn of
  that conversation is deep too ("how many orders shipped this month" after "connect to
  WAREHOUSE"). A reset (new conversation) clears it. A *fast* turn that reaches a deep connector's
  tool anyway ("how many orders are shipping today?", which names no server) is stopped at
  that first call and the same request is re-run on the deep route inside the same
  ``respond()``: the user hears one answer, from the deep model, with the deep step budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

from spark.activity import ActivityLog, NullActivityLog
from spark.brain.confirm import ConfirmationGate, Confirmer, HookCallback, Summarizer
from spark.brain.core import (
    REASON_ESCALATING,
    RESEARCHER_DESCRIPTION,
    RESEARCHER_PROMPT,
    BrainClosed,
    BrainError,
    TurnStats,
    VoiceBrain,
    _Turn,
)
from spark.brain.mcp_config import McpSetup, env_for_sdk
from spark.brain.prompts import WORKER_RULES
from spark.brain.router import Router
from spark.interfaces import BackendUnavailable
from spark.util.clock import SystemClock

if TYPE_CHECKING:
    from spark.config import Settings
    from spark.memory import MemoryStore
    from spark.tools.project import ProjectService
    from spark.tools.status import StatusProvider
    from spark.tools.timers import TimerService
    from spark.util.clock import Clock

__all__ = [
    "REASON_ESCALATING",
    "RESEARCHER_PROMPT",
    "BrainClosed",
    "BrainError",
    "ClaudeAgentBrain",
    "TurnStats",
]

log = logging.getLogger(__name__)

BACKEND = "claude_agent"
#: What to do when the Claude Code CLI is missing (the frozen app doesn't ship it).
INSTALL_CLAUDE_CODE = (
    "install Claude Code (https://code.claude.com/docs/en/setup), then restart Hey Spark; "
    "or set brain.cli_path to the claude executable"
)
#: How long the CLI may take to report the MCP status before we skip the check.
MCP_STATUS_TIMEOUT_S = 5.0
#: How long a set_model control request may take before the turn goes ahead on the old model.
SET_MODEL_TIMEOUT_S = 15.0


PRIME_TEXT = "Connection check before we start. Reply with just: OK"


@dataclass
class _Control:
    kind: str  # "connect" | "reset" | "close"
    future: asyncio.Future[None]


def _load_sdk() -> ModuleType:
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise BackendUnavailable(
            BACKEND,
            f"claude-agent-sdk is not importable ({exc})",
            "uv sync  (claude-agent-sdk is a core dependency)",
        ) from exc
    return claude_agent_sdk


def _text_delta(event: Any) -> str | None:
    """The text of a raw Anthropic stream event if it is a ``text_delta``, else None."""
    if not isinstance(event, dict) or event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta")
    if isinstance(delta, dict) and delta.get("type") == "text_delta":
        text = delta.get("text")
        return text if isinstance(text, str) and text else None
    return None


def _is_text_block_start(event: Any) -> bool:
    if not isinstance(event, dict) or event.get("type") != "content_block_start":
        return False
    block = event.get("content_block")
    return isinstance(block, dict) and block.get("type") == "text"


def _tool_block_start_id(event: Any) -> str | None:
    """The tool_use id if ``event`` starts a ``tool_use`` content block, else None."""
    if not isinstance(event, dict) or event.get("type") != "content_block_start":
        return None
    block = event.get("content_block")
    if isinstance(block, dict) and block.get("type") == "tool_use":
        tool_id = block.get("id")
        return tool_id if isinstance(tool_id, str) and tool_id else None
    return None


class ClaudeAgentBrain(VoiceBrain):
    """The default brain. Built by the registry as ``ClaudeAgentBrain(settings, **deps)``.

    Args:
        settings: Spark settings (uses ``settings.brain``).
        clock: injectable clock for idle reset and timing (default: real clock).
        sdk: the ``claude_agent_sdk`` module (tests inject a fake; default: lazy import).
        turn_timeout_s: give up on a turn that hasn't finished after this long.
        interrupt_timeout_s: how long ``interrupt()`` waits for the turn to drain.
        close_timeout_s: how long ``close()`` waits for a clean disconnect before cancelling.
        mcp: a prepared :class:`McpSetup` (default: ``load_mcp_servers(settings)``). The brain
            merges the ``spark`` server into it.
        timers: the app's :class:`TimerService`; when given, the in-process ``spark`` server
            (current_time, timers, spark_status) is added.
        status_provider: what ``spark_status`` reports (the app's ``status_snapshot``).
        confirmer: asks the user to approve side-effecting tools (voice or ``--text``). If it is
            None, the default gate denies them without asking.
        summarizers: extra per-tool confirmation lines, keyed by the full or the bare tool name.
        projects: the ``ProjectService`` behind ``start_project`` (default: built from
            ``settings.projects``). It is shared with its confirmation summary.
        tool_gate: override the ``PreToolUse`` hook (default: :class:`ConfirmationGate`). A
            custom gate should call :meth:`note_confirming` around its question so the
            acknowledgement timer pauses, and read :attr:`current_route` for the fast tool
            limit. An injected :class:`ConfirmationGate` without a ``route_provider`` gets one.
        router: the fast/deep :class:`Router` (default: ``Router.from_settings(settings)``).
        **deps: extra registry dependencies, kept in ``self.deps``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Clock | None = None,
        sdk: ModuleType | None = None,
        turn_timeout_s: float = 180.0,
        interrupt_timeout_s: float = 10.0,
        close_timeout_s: float = 15.0,
        mcp: McpSetup | None = None,
        timers: TimerService | None = None,
        status_provider: StatusProvider | None = None,
        confirmer: Confirmer | None = None,
        summarizers: Mapping[str, Summarizer] | None = None,
        projects: ProjectService | None = None,
        tool_gate: HookCallback | None = None,
        router: Router | None = None,
        activity: ActivityLog | None = None,
        context_provider: Callable[[], str] | None = None,
        memory: MemoryStore | None = None,
        **deps: Any,
    ) -> None:
        self.settings = settings
        self.config = settings.brain
        self.models_cfg = settings.brain  # Claude model ids per role
        self.deps = deps
        self.activity: ActivityLog = activity or NullActivityLog()
        # Called at every connect (warm-up, reset): text appended to the system prompt (P5.6).
        self.context_provider = context_provider
        self.memory = memory  # connector notes tools (P5.9); None: no memory tools
        self._sdk = sdk if sdk is not None else _load_sdk()
        self._clock: Clock = clock or SystemClock()
        self.turn_timeout_s = turn_timeout_s
        self.interrupt_timeout_s = interrupt_timeout_s
        self.close_timeout_s = close_timeout_s

        # Deep mode connects on the voice model, so no session starts with a model switch.
        self.model: str = (
            self.config.deep_model if self.config.mode == "deep" else self.config.fast_model
        )
        self.timers = timers
        self.confirmer = confirmer
        self.summarizers: dict[str, Summarizer] = dict(summarizers or {})
        self.mcp: McpSetup = mcp if mcp is not None else self._load_mcp()
        if timers is not None:
            self._add_spark_server(timers, status_provider, projects)
        self.hook_timeout_s: float = settings.confirm.hook_timeout_s
        self.routing = settings.routing
        self.router: Router = router or Router.from_settings(settings, sdk=self._sdk)
        self._init_core(tool_gate=tool_gate, delegate_tool="Agent", worker_rules=WORKER_RULES)

        self._client: Any | None = None
        self._requests: asyncio.Queue[_Turn | _Control] | None = None
        self._worker: asyncio.Task[None] | None = None

    # -- MCP ----------------------------------------------------------------------------------

    def _add_spark_server(
        self,
        timers: TimerService,
        status_provider: StatusProvider | None,
        projects: ProjectService | None,
    ) -> None:
        from spark.tools.project import ProjectService
        from spark.tools.server import (
            SERVER_NAME,
            build_spark_server,
            spark_confirmation_summarizers,
        )

        if SERVER_NAME in self.mcp.servers:
            log.info("MCP: in-process %r server already configured", SERVER_NAME)
            return
        service = projects if projects is not None else ProjectService.from_settings(self.settings)
        server, read_only, side_effecting = build_spark_server(
            timers,
            status_provider,
            projects=service,
            memory=self.memory,
            connectors=lambda: self.external_connectors,
            activity=self.activity,
        )
        self.mcp.add_sdk_server(
            SERVER_NAME, server, read_only=read_only, side_effecting=side_effecting
        )
        for name, fn in spark_confirmation_summarizers(service).items():
            self.summarizers.setdefault(name, fn)

    # -- connector health (P5.7i) ----------------------------------------------------------------

    async def _check_mcp_status(self) -> None:
        """Ask the CLI for the live MCP status; log and report changes. Best effort, in the
        worker task (the client's owner)."""
        from spark.tools.server import SERVER_NAME

        client = self._client
        get_status = getattr(client, "get_mcp_status", None)
        if get_status is None:
            return
        try:
            async with asyncio.timeout(MCP_STATUS_TIMEOUT_S):
                response = await get_status()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("brain: MCP status unavailable: %s", exc)
            return
        from spark.brain import claudeai

        entries = [e for e in (response or {}).get("mcpServers", []) or [] if isinstance(e, dict)]
        seen = claudeai.parse_status(entries)
        if seen and claudeai.save_state(self.settings, seen):
            log.info("claude.ai connectors seen: %s", ", ".join(sorted(seen)))
        mine = set(self.mcp.servers) | set(self.mcp.claudeai)
        current: dict[str, dict[str, str]] = {}
        for entry in entries:
            raw = str(entry.get("name") or "")
            name = claudeai.server_key(raw) or raw
            if not name or name == SERVER_NAME or (mine and name not in mine):
                continue  # another claude.ai or user-level server: hidden, not Spark's
            info = {"status": str(entry.get("status") or "unknown")}
            if entry.get("error"):
                info["error"] = " ".join(str(entry["error"]).split())[:200]
            if claudeai.is_claudeai(name) and info["status"] == "needs-auth":
                info.setdefault("error", claudeai.SIGN_IN_HINT)
            current[name] = info
        self._apply_mcp_status(current)

    def _agents_option(self) -> dict[str, Any]:
        definition = getattr(self._sdk, "AgentDefinition", None)
        spec: dict[str, Any] = {
            "description": RESEARCHER_DESCRIPTION,
            "prompt": RESEARCHER_PROMPT,
            "tools": self.worker_tools(),
            "model": self.worker_model,
            "maxTurns": self.config.worker_max_turns,
        }
        return {"researcher": definition(**spec) if definition is not None else spec}

    def _mcp_option_kwargs(self) -> dict[str, Any]:
        allowed = [t for t in self.mcp.allowed_tools if not any(c in t for c in "*?[")]
        if len(allowed) != len(self.mcp.allowed_tools):
            log.warning("MCP: dropped non-exact allowed_tools entries (globs never pre-approve)")
        # timeout: the CLI's per-hook budget (default 60 s, types.py:622) must outlast a spoken
        # question plus the answer; on overrun the CLI cancels the hook and skips the tool.
        gate = self._sdk.HookMatcher(
            matcher=None, hooks=[self.tool_gate], timeout=self.hook_timeout_s
        )
        return {
            "mcp_servers": dict(self.mcp.servers),
            "allowed_tools": allowed,
            "disallowed_tools": list(self.mcp.disallowed_tools),
            # Every tool call passes the gate: read-only ones defer to allowed_tools,
            # side-effecting ones are confirmed by voice, destructive and unknown are denied.
            "hooks": {"PreToolUse": [gate]},
            "env": env_for_sdk(self.settings, claudeai=bool(self.mcp.claudeai)),
        }

    # -- options ------------------------------------------------------------------------------

    def _option_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``ClaudeAgentOptions``. Field names are verified in
        docs/sdk_notes.md §1 and the P2.4 section (``claude_agent_sdk/types.py:1962-2349``)."""
        workers = self.workers_active
        kwargs: dict[str, Any] = {
            "system_prompt": self._session_prompt(),
            "model": self.model,
            "include_partial_messages": True,
            # No built-in Claude Code tools (Bash, Read, ...): MCP tools only, plus Agent for
            # the researcher worker when it exists.
            "tools": ["Agent"] if workers else [],
            "setting_sources": [],  # don't inherit user/project settings, hooks, or CLAUDE.md
            # Only MCP servers we pass explicitly, unless claude.ai connectors are in use: the
            # CLI loads those only without --strict-mcp-config (the unconfigured ones are hidden
            # by disallowed_tools, and the gate denies any tool outside the policy).
            "strict_mcp_config": not self.mcp.claudeai,
            # Backstop only: the gate's step budget answers before this is reached.
            "max_turns": max(self.config.max_turns, self.config.deep_max_turns) + 2,
            "stderr": self._on_cli_stderr,
            **self._mcp_option_kwargs(),
        }
        if workers:
            kwargs["agents"] = self._agents_option()
            kwargs["allowed_tools"] = [*kwargs["allowed_tools"], "Agent"]
        if self.config.cli_path:
            kwargs["cli_path"] = self.config.cli_path
        return kwargs

    def _build_options(self) -> Any:
        return self._sdk.ClaudeAgentOptions(**self._option_kwargs())

    @staticmethod
    def _on_cli_stderr(line: str) -> None:
        log.debug("claude cli: %s", line.rstrip())

    # -- public API ---------------------------------------------------------------------------

    async def warmup(self) -> None:
        """Start the session worker and connect (spawns the CLI) ahead of the first turn. An
        optional route classifier warms up alongside."""
        await asyncio.gather(self._control("connect"), self.router.warmup())

    async def reset(self) -> None:
        """Start a fresh conversation: disconnect, then connect a new client."""
        self._reset_state()
        if self._worker is None or self._worker.done():
            self._client = None  # nothing connected yet; the next turn connects fresh
            return
        await self._control("reset")

    async def close(self) -> None:
        """Disconnect and stop the worker. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.router.aclose()
        except Exception as exc:
            log.debug("brain: router close failed: %s", exc)
        for turn in [*self._queued, *([self._active] if self._active else [])]:
            turn.interrupted = True
            turn.interrupt.set()
        worker = self._worker
        if worker is None or worker.done():
            if worker is not None:
                with contextlib.suppress(BaseException):
                    await worker
            return
        assert self._requests is not None
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._requests.put_nowait(_Control("close", fut))
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=self.close_timeout_s)
        except TimeoutError:
            log.warning(
                "brain: worker did not stop within %.1f s; cancelling", self.close_timeout_s
            )
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker
        except Exception:
            log.exception("brain: worker failed during close")
        finally:
            if not fut.done():
                fut.cancel()

    # -- helpers ------------------------------------------------------------------------------

    def _enqueue(self, turn: _Turn) -> None:
        self._ensure_worker()
        assert self._requests is not None
        self._queued.append(turn)
        self._requests.put_nowait(turn)

    def _ensure_worker(self) -> None:
        self._check_open()
        if self._worker is None or self._worker.done():
            self._requests = asyncio.Queue()
            self._worker = asyncio.create_task(self._run_worker(), name="spark-brain-session")

    async def _control(self, kind: str) -> None:
        self._ensure_worker()
        assert self._requests is not None
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._requests.put_nowait(_Control(kind, fut))
        await fut

    # -- session worker (the only task that touches the client) -------------------------------

    async def _run_worker(self) -> None:
        assert self._requests is not None
        requests = self._requests
        try:
            while True:
                item = await requests.get()
                if isinstance(item, _Control):
                    if item.kind == "close":
                        await self._disconnect()
                        self._fail_pending(requests)
                        if not item.future.done():
                            item.future.set_result(None)
                        return
                    try:
                        if item.kind == "reset":
                            await self._disconnect()
                        # prime only here (warm-up / reset), never inside a user's turn
                        await self._ensure_connected(prime=self.config.prime_on_connect)
                    except asyncio.CancelledError:
                        item.future.cancel()
                        raise
                    except Exception as exc:
                        if not item.future.done():
                            item.future.set_exception(exc)
                    else:
                        if not item.future.done():
                            item.future.set_result(None)
                else:
                    await self._handle_turn(item)
        finally:
            # Cancelled or closed: never leave the CLI subprocess behind.
            await self._disconnect()

    def _fail_pending(self, requests: asyncio.Queue[_Turn | _Control]) -> None:
        while not requests.empty():
            item = requests.get_nowait()
            if isinstance(item, _Control):
                if not item.future.done():
                    item.future.set_exception(BrainClosed("the brain has been closed"))
            else:
                self._finish(item, BrainClosed("the brain has been closed"))

    async def _ensure_connected(self, *, prime: bool = False) -> None:
        if self._client is not None:
            return
        sdk = self._sdk
        client = sdk.ClaudeSDKClient(options=self._build_options())
        start = self._clock.monotonic()
        try:
            # asyncio.timeout (not wait_for, which runs the coroutine in a *new* task on 3.11):
            # connect() must run in this worker task, which then owns the client's task group.
            async with asyncio.timeout(self.config.connect_timeout_s):
                await client.connect()
        except TimeoutError as exc:
            raise BackendUnavailable(
                BACKEND,
                f"connecting to Claude timed out after {self.config.connect_timeout_s:.0f} s",
                "sign in to Claude Code (run claude, then /login) or set ANTHROPIC_API_KEY, "
                "check the network, or raise brain.connect_timeout_s",
            ) from exc
        except sdk.CLINotFoundError as exc:
            # The frozen app doesn't ship the CLI: the user installs Claude Code themselves.
            log.debug("brain: %s", exc)
            raise BackendUnavailable(
                BACKEND,
                "Claude Code isn't installed (Spark's Claude brain runs it)",
                INSTALL_CLAUDE_CODE,
            ) from exc
        except sdk.CLIConnectionError as exc:
            raise BackendUnavailable(
                BACKEND,
                f"could not start the Claude Code CLI ({exc})",
                "run `uv run python -m spark doctor`; check brain.cli_path and `claude --version`",
            ) from exc
        self._client = client
        log.debug(
            "brain: connected (model=%s) in %.2f s", self.model, self._clock.monotonic() - start
        )
        if prime:
            await self._prime()
            await self._prewarm_models()
        await self._check_mcp_status()

    async def _prewarm_models(self) -> None:
        """Visit every routed model once with ``set_model`` and come back, at warm-up.

        Measured live (docs/sdk_notes.md "Routing (P3.2)"): the first ``set_model`` to a given
        name costs ~0.45-0.7 s (the CLI validates it) and later ones ~0 ms. Doing it here keeps
        that off the first deep turn, and a rejected model name (``Model '...' not found``) is
        reported at startup instead of mid-conversation. Best effort."""
        client = self._client
        if client is None or not self.routing.enabled:
            return
        home = self.model
        for model in dict.fromkeys((self.config.fast_model, self.config.deep_model)):
            if model == home:
                continue
            try:
                async with asyncio.timeout(SET_MODEL_TIMEOUT_S):
                    await client.set_model(model)
                    await client.set_model(home)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "brain: model %s was not accepted at warm-up (%s: %s); turns routed to it "
                    "will stay on %s",
                    model,
                    type(exc).__name__,
                    exc,
                    home,
                )
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(SET_MODEL_TIMEOUT_S):
                        await client.set_model(home)

    async def _prime(self) -> None:
        """Run one throwaway turn so the session's first-request cost isn't paid by the user.

        Measured live: the first turn after connect has ~4-5 s extra time-to-first-token; later
        turns ~1-1.5 s (docs/sdk_notes.md "Verified live"). Best effort: on failure the client
        is dropped and the next turn reconnects.
        """
        turn = _Turn(PRIME_TEXT, interrupted=True)  # output is discarded
        saved = self.last_turn
        start = self._clock.monotonic()
        try:
            async with asyncio.timeout(self.config.connect_timeout_s):
                await self._run_turn(turn)
            log.debug("brain: primed in %.2f s", self._clock.monotonic() - start)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("brain: priming turn failed (%s); will reconnect on the next turn", exc)
            await self._disconnect()
        finally:
            self.last_turn = saved

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.disconnect()
        except Exception as exc:
            log.debug("brain: disconnect failed: %s", exc)

    async def _handle_turn(self, turn: _Turn) -> None:
        if turn.interrupt.is_set():  # interrupted while still queued
            self._finish(turn)
            return
        self._active = turn
        error: BaseException | None = None
        try:
            await self._ensure_connected()
            switched = await self._switch_model(turn)
            # The gate reads this for every tool call of the turn (fast tool limit).
            self._current_route = self._turn_route(turn)
            await self._run_turn(turn, model_switched=switched)
        except asyncio.CancelledError:
            self._current_route = None
            self._finish(turn, BrainClosed("the brain was shut down mid-turn"))
            self._active = None
            raise
        except BrainError as exc:
            error = exc
        except TimeoutError:
            error = BrainError(f"Claude did not finish within {self.turn_timeout_s:.0f} s")
            await self._disconnect()  # the stream is in an unknown state; start clean
        except BackendUnavailable as exc:
            error = exc
        except Exception as exc:
            log.debug("brain: turn failed", exc_info=True)
            error = BrainError(f"Claude request failed: {type(exc).__name__}: {exc}")
            await self._disconnect()
        self._current_route = None
        self._active = None
        self._finish(turn, error)
        if self._client is not None:
            await self._check_mcp_status()  # after the reply: never delays it

    async def _switch_model(self, turn: _Turn) -> bool:
        """Point the session at the turn's routed model with ``set_model`` (verified at
        ``claude_agent_sdk/client.py:331``: ``async set_model(model: str | None = None)``, a
        ``set_model`` control request, ``_internal/query.py:722``). Same client, same
        conversation. On failure the turn goes ahead on the current model. Returns True if the
        model changed."""
        decision = turn.decision
        if decision is None or decision.model == self.model:
            return False
        client = self._client
        assert client is not None
        start = self._clock.monotonic()
        try:
            async with asyncio.timeout(SET_MODEL_TIMEOUT_S):
                await client.set_model(decision.model)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "brain: switching to %s failed (%s: %s); this turn stays on %s",
                decision.model,
                type(exc).__name__,
                exc,
                self.model,
            )
            return False
        log.info(
            "brain: model %s -> %s (%.0f ms)",
            self.model,
            decision.model,
            (self._clock.monotonic() - start) * 1000,
        )
        self.model = decision.model
        return True

    async def _run_turn(self, turn: _Turn, *, model_switched: bool = False) -> None:
        sdk = self._sdk
        client = self._client
        assert client is not None
        stats = TurnStats(model=self.model, model_switched=model_switched, acked=turn.acked)
        if turn.decision is not None:
            stats.route = turn.decision.route
            stats.route_reason = turn.decision.reason
        turn.stats = stats
        fallback: list[str] = []
        emitted = 0
        need_sep = False
        last_char = ""
        assistant_error: str | None = None

        def emit(text: str) -> None:
            nonlocal emitted, need_sep, last_char
            if need_sep and emitted and not last_char.isspace() and not text[:1].isspace():
                text = " " + text
            need_sep = False
            if stats.ttft_s is None:
                stats.ttft_s = self._clock.monotonic() - start
            emitted += 1
            last_char = text[-1]
            stats.chars += len(text)
            if not turn.interrupted:
                turn.out.put_nowait(text)

        start = self._clock.monotonic()
        watcher: asyncio.Task[None] | None = None
        result = None
        record = turn.decision is not None  # never log the priming turn
        tool_names: dict[str, str] = {}
        gate = self.tool_gate if isinstance(self.tool_gate, ConfirmationGate) else None
        decisions_before = len(gate.decisions) if gate is not None else 0
        user_message = getattr(sdk, "UserMessage", None)
        result_block = getattr(sdk, "ToolResultBlock", None)
        try:
            async with asyncio.timeout(self.turn_timeout_s):
                await client.query(turn.text)
                watcher = asyncio.create_task(self._watch_interrupt(client, turn))
                async for msg in client.receive_response():
                    if isinstance(msg, sdk.StreamEvent):
                        if msg.parent_tool_use_id is not None:
                            continue
                        if _is_text_block_start(msg.event):
                            need_sep = True
                            continue
                        tool_id = _tool_block_start_id(msg.event)
                        if tool_id is not None:
                            if self._announce_tool(turn, tool_id, emitted > 0):
                                last_char = "\n"
                            continue
                        delta = _text_delta(msg.event)
                        if delta:
                            stats.partial = True
                            emit(delta)
                    elif isinstance(msg, sdk.AssistantMessage):
                        if msg.parent_tool_use_id is not None:
                            # The researcher at work: never spoken, but its calls are logged.
                            if record:
                                self._record_worker_calls(msg, tool_names)
                            continue
                        served = getattr(msg, "model", None)
                        if stats.served_model is None and isinstance(served, str) and served:
                            stats.served_model = served
                        if getattr(msg, "error", None):
                            assistant_error = str(msg.error)
                        for block in msg.content:
                            if isinstance(block, sdk.ToolUseBlock):
                                # names only: arguments and results can hold personal data.
                                # The activity log (redacted, on disk) keeps the details.
                                log.info("tool call: %s", block.name)
                                tool_names[block.id] = block.name
                                if record:
                                    self.activity.tool_call(block.name, block.id, block.input)
                                if self._announce_tool(turn, block.id, emitted > 0):
                                    last_char = "\n"
                        texts = [b.text for b in msg.content if isinstance(b, sdk.TextBlock)]
                        if texts:
                            fallback.extend(texts)
                            if not stats.partial:
                                # No partials this turn: speak whole blocks as they arrive.
                                for text in texts:
                                    need_sep = True
                                    emit(text)
                    elif user_message is not None and isinstance(msg, user_message):
                        if record:
                            self._record_tool_results(msg, result_block, tool_names)
                    elif isinstance(msg, sdk.ResultMessage):
                        result = msg
        finally:
            self._tool_seen.clear()
            if gate is not None and record:
                for name, outcome in gate.decisions[decisions_before:]:
                    self.activity.record("confirm", tool=name, outcome=outcome)
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher

        stats.total_s = self._clock.monotonic() - start
        stats.interrupted = turn.interrupted
        if result is not None:
            stats.duration_ms = result.duration_ms
            stats.duration_api_ms = result.duration_api_ms
            stats.num_turns = result.num_turns
            stats.total_cost_usd = result.total_cost_usd
            stats.usage = result.usage
            stats.is_error = bool(result.is_error)
            stats.subtype = result.subtype
            stats.stop_reason = result.stop_reason
            stats.terminal_reason = getattr(result, "terminal_reason", None)
            stats.session_id = result.session_id
        self.last_turn = stats
        log.debug("brain turn: %s", stats.as_meta())
        if record:
            self.activity.record(
                "turn_end",
                session_id=stats.session_id,
                num_turns=stats.num_turns,
                cost_usd=stats.total_cost_usd,
                route=stats.route,
                model=stats.served_model or stats.model,
                error=stats.subtype if stats.is_error else None,
                interrupted=stats.interrupted or None,
            )

        if result is None:
            raise BrainError("Claude's response stream ended without a result", stats)
        if result.is_error and not turn.interrupted:
            detail = result.result or "; ".join(result.errors or []) or assistant_error
            status = getattr(result, "api_error_status", None)
            raise BrainError(
                f"Claude returned an error ({result.subtype}"
                + (f", HTTP {status}" if status else "")
                + (f"): {detail}" if detail else ")"),
                stats,
            )

    def _record_worker_calls(self, msg: Any, tool_names: dict[str, str]) -> None:
        sdk = self._sdk
        for block in msg.content:
            if isinstance(block, sdk.ToolUseBlock):
                log.info("researcher tool call: %s", block.name)
                tool_names[block.id] = block.name
                self.activity.record(
                    "tool_call",
                    tool=block.name,
                    id=block.id,
                    input=block.input,
                    agent="researcher",
                    model=getattr(msg, "model", None),
                )

    def _record_tool_results(
        self, msg: Any, result_block: Any, tool_names: Mapping[str, str]
    ) -> None:
        content = msg.content if isinstance(msg.content, list) else []
        for block in content:
            if result_block is not None and isinstance(block, result_block):
                tool_id = block.tool_use_id
                self.activity.tool_result(
                    tool_names.get(tool_id), tool_id, block.content, block.is_error
                )

    # -- tool-call ordering for the confirmation hook ------------------------------------------

    async def _watch_interrupt(self, client: Any, turn: _Turn) -> None:
        await turn.interrupt.wait()
        try:
            await client.interrupt()
        except Exception as exc:
            log.debug("brain: interrupt request failed: %s", exc)
