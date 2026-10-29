"""What every tool-using voice brain shares, whichever model provider runs it (P5.7e).

:class:`VoiceBrain` holds the behaviour that is about Spark, not about Claude or OpenAI:

* the voice system prompt plus the briefing (``context_provider``) for each new session;
* the MCP setup (:class:`~spark.brain.mcp_config.McpSetup`) and the ONE tool gate
  (:class:`~spark.brain.confirm.ConfirmationGate`): read-only tools pass, side-effecting ones are
  confirmed out loud, destructive and unknown ones are denied, and the step budget is enforced
  in its ``precheck``;
* fast/deep routing per request, the model mode (auto / fast / deep), shortcut routes, deep
  connectors and the "stay deep" rules, the escalation re-run;
* the "On it, give me a moment." acknowledgement timer, paused while a confirmation is asking;
* idle reset and "new topic", interruption (barge-in), connector health (P5.7i) and the status
  line told to the model about unreachable connectors;
* the researcher worker's tool list (P5.7d).

A backend (:class:`spark.brain.claude_agent.ClaudeAgentBrain`,
:class:`spark.brain.openai_compat.OpenAICompatBrain`) supplies how a turn actually runs: it
implements :meth:`VoiceBrain._enqueue` (start the turn; push text deltas to ``turn.out`` and
call :meth:`_finish` at the end) and the connection lifecycle. ``respond()`` is shared, so both
backends route, acknowledge, escalate and stream exactly the same way.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from spark.activity import ActivityLog
from spark.brain.confirm import ConfirmationGate, Confirmer, HookCallback, Summarizer
from spark.brain.mcp_config import McpConfigError, McpSetup, load_mcp_servers
from spark.brain.prompts import split_new_topic, voice_system_prompt
from spark.brain.router import DEEP, RouteDecision, Router
from spark.interfaces import Brain
from spark.persona import persona_for

if TYPE_CHECKING:
    from spark.config import Settings
    from spark.memory import MemoryStore
    from spark.util.clock import Clock

__all__ = [
    "REASON_ESCALATING",
    "RESEARCHER_DESCRIPTION",
    "RESEARCHER_PROMPT",
    "BrainClosed",
    "BrainError",
    "ModelRoles",
    "TurnStats",
    "VoiceBrain",
]

log = logging.getLogger(
    "spark.brain.claude_agent"
)  # one logger name for both backends' shared code

_END = object()  # end-of-turn marker on a turn's output queue
_ACK = object()  # "no text yet, say the acknowledgement" (respond() only)

#: The researcher worker's instructions (P5.7d).
RESEARCHER_PROMPT = """\
You are the researcher behind a voice assistant. The assistant hands you a lookup; you do it with \
the read-only tools of the connected systems and report back to the assistant, not to the user.
- Check the connector notes you were given first and query directly when they cover the request. \
Otherwise explore: list databases, schemas and tables, describe the likely tables, then query.
- Prefer one precise query over many broad ones. Count and aggregate in the query when you can.
- After you had to explore, save where the data lives with save_connector_note (database, \
schema, table, key columns, the filter that worked). Structure only, never data values.
- Report in a few plain lines: the answer with its numbers, the filters and date range you used, \
the table it came from, and anything uncertain. No markdown tables, no raw dumps."""

#: How the voice model sees the researcher (the Agent definition / the research function).
RESEARCHER_DESCRIPTION = (
    "Researcher. Looks things up in the connected systems (databases, email, "
    "calendar, repositories): explores where data lives, runs and refines read-only "
    "queries, and reports a compact factual result with the numbers and where they "
    "came from. Use it for any lookup that needs more than one or two tool calls."
)

#: Deny reason for the first deep-connector call of a fast turn (the turn is re-run deep).
REASON_ESCALATING = "Not run here. Stop now and say nothing; this request is being handed over."


class BrainError(RuntimeError):
    """The model finished a turn with an error (or the stream failed).

    ``spoken``: an optional sentence for the user (the app says it instead of its generic
    failure line). It never contains the raw error, which can hold paths, IDs or secrets."""

    def __init__(
        self, message: str, stats: TurnStats | None = None, *, spoken: str | None = None
    ) -> None:
        super().__init__(message)
        self.stats = stats
        self.spoken = spoken


class BrainClosed(RuntimeError):
    """The brain was closed while (or before) a request was handled."""


class ModelRoles(Protocol):
    """The model behind each role: ``BrainConfig`` (Claude ids) or ``OpenAICompatConfig``."""

    fast_model: Any
    deep_model: Any
    worker_model: Any


@dataclass
class TurnStats:
    """Metadata of the last completed turn, for the latency log (spec §9)."""

    ttft_s: float | None = None  # query sent -> first text delta
    total_s: float | None = None  # query sent -> ResultMessage
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    is_error: bool = False
    subtype: str | None = None
    stop_reason: str | None = None
    terminal_reason: str | None = None
    session_id: str | None = None
    model: str | None = None
    partial: bool = False  # True when deltas came from StreamEvents
    interrupted: bool = False
    chars: int = 0
    route: str | None = None  # "fast" | "deep" (spark.brain.router)
    route_reason: str | None = None
    model_switched: bool = False  # set_model was called before this turn
    served_model: str | None = None  # AssistantMessage.model (types.py:1144): who answered
    acked: bool = False  # the "On it, give me a moment." acknowledgement was yielded

    def as_meta(self) -> dict[str, Any]:
        """Flat JSON-friendly dict (``TurnTimer.annotate(**stats.as_meta())``)."""
        out: dict[str, Any] = {
            "brain_ttft_ms": None if self.ttft_s is None else round(self.ttft_s * 1000, 1),
            "brain_total_ms": None if self.total_s is None else round(self.total_s * 1000, 1),
            "brain_duration_ms": self.duration_ms,
            "brain_api_ms": self.duration_api_ms,
            "brain_num_turns": self.num_turns,
            "brain_cost_usd": self.total_cost_usd,
            "brain_is_error": self.is_error,
            "brain_model": self.model,
            "brain_served_model": self.served_model,
            "brain_partial": self.partial,
            "brain_interrupted": self.interrupted,
            "brain_route": self.route,
            "brain_route_reason": self.route_reason,
            "brain_model_switched": self.model_switched,
            "brain_acked": self.acked,
        }
        if self.usage:
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                if isinstance(self.usage.get(key), int):
                    out[f"brain_{key}"] = self.usage[key]
        return out


@dataclass
class _Turn:
    text: str
    out: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    interrupt: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: bool = False  # interrupt() or abandonment: drop output, don't raise
    decision: RouteDecision | None = None  # None: stay on the current model (priming)
    acked: bool = False
    stats: TurnStats | None = None
    mode_forced: bool = False  # brain.mode fast/deep: the user picked the model (P5.7c)
    tool_calls: int = 0  # calls that reached the PreToolUse gate (step budget)
    tool_started: asyncio.Event = field(default_factory=asyncio.Event)
    escalate: str | None = None  # a fast turn reached this deep connector: re-run it deep


class VoiceBrain(Brain):
    """Shared base of the tool-using brains. See the module docstring.

    A subclass sets, before calling :meth:`_init_core`: ``settings``, ``config``
    (``settings.brain``), ``models_cfg`` (:class:`ModelRoles`), ``activity``,
    ``context_provider``, ``memory``, ``_clock``, ``confirmer``, ``summarizers``, ``mcp``,
    ``routing``, ``router`` and ``interrupt_timeout_s``.
    """

    settings: Settings
    config: Any
    models_cfg: ModelRoles
    activity: ActivityLog
    context_provider: Callable[[], str] | None
    memory: MemoryStore | None
    _clock: Clock
    confirmer: Confirmer | None
    summarizers: dict[str, Summarizer]
    mcp: McpSetup
    router: Router
    interrupt_timeout_s: float

    def _init_core(
        self,
        *,
        tool_gate: HookCallback | None,
        delegate_tool: str,
        worker_rules: str,
        worker_rules_on: bool | None = None,
    ) -> None:
        """Build the gate, routing state and system prompt.

        ``delegate_tool``: the name the voice model calls to start the researcher (``Agent`` on
        Claude, ``research`` elsewhere). It is pre-approved like a read-only tool, and still
        counts as one step of the budget. ``worker_rules``: the prompt section about the
        researcher, added when it exists (``worker_rules_on`` overrides that check)."""
        self.last_context = ""
        #: How long the gate waits for the tool call to show up in the stream before asking.
        self.tool_announce_wait_s: float = 2.0
        self._tool_seen: dict[str, asyncio.Event] = {}
        # Confirmation questions in flight (the ack timer is paused while any are).
        self._confirming = 0
        self._confirm_idle = asyncio.Event()
        self._confirm_idle.set()
        self._confirm_busy = asyncio.Event()
        # The active turn's route for the gate's fast tool limit (None outside routed turns).
        self._current_route: str | None = None
        self._escalate_next = False  # a fast turn had a tool refused: route the next one deep
        workers = self.workers_active
        self.delegate_tool = delegate_tool
        self.tool_gate: HookCallback = tool_gate or ConfirmationGate(
            self.mcp.policy,
            self.confirmer,
            summarizers=self.summarizers,
            ask_timeout_s=self.settings.confirm.ask_budget_s,
            before_ask=self._await_tool_announced,
            on_asking=self.note_confirming,
            route_provider=self._route_for_gate,
            fast_tools=self.routing.fast_tools,
            on_fast_denied=self._note_fast_denied,
            precheck=self._step_precheck,
            extra_read=(delegate_tool,) if workers else (),
        )
        if isinstance(self.tool_gate, ConfirmationGate):
            if self.tool_gate.route_provider is None:
                self.tool_gate.route_provider = self._route_for_gate
            if self.tool_gate.on_fast_denied is None:
                self.tool_gate.on_fast_denied = self._note_fast_denied
            if self.tool_gate.precheck is None:
                self.tool_gate.precheck = self._step_precheck
            if workers and not self.tool_gate.extra_read:
                self.tool_gate.extra_read = frozenset({delegate_tool})
        # The deep connector this conversation has used (route every later turn deep).
        self._sticky_deep: str | None = None
        self._went_deep = False  # auto: a turn of this conversation went deep (stay deep)
        # One-shot route for the next respond() ("fast" | "deep"), e.g. a voice shortcut's.
        self.next_route: str | None = None
        self.next_route_reason = ""
        self.next_route_wins = False  # True: next_route beats a fixed model mode (asked for)
        # Live MCP status per external server (P5.7i): {"WAREHOUSE": {"status": ..., "error": ...}}
        self.mcp_status: dict[str, dict[str, str]] = {}
        self.on_mcp_status: Callable[[dict[str, dict[str, str]]], None] | None = None
        self._status_noted: set[str] = set()  # unreachable servers already told to the model
        self._check_fast_tools()
        self._log_connectors()
        extras = []
        if self.memory is not None and self.external_connectors:
            from spark.brain.prompts import CONNECTOR_NOTES_RULES

            extras.append(CONNECTOR_NOTES_RULES)
        if workers if worker_rules_on is None else worker_rules_on:
            extras.append(worker_rules)
        self.voices = persona_for(self.settings)
        ack = self.routing.ack_text
        self.ack_text: str = (ack if ack is not None else self.voices.line("ack")).strip()
        self.system_prompt: str = voice_system_prompt(
            tools_available=bool(self.mcp.servers or self.mcp.claudeai),
            name=self.settings.app.name,
            extra="\n\n".join(extras),
            persona=self.voices.prompt(),
            ack=self.ack_text,
        )
        self.last_turn: TurnStats | None = None
        self.resets = 0
        self._active: _Turn | None = None
        self._queued: list[_Turn] = []
        self._last_turn_end: float | None = None
        self._closed = False

    # -- backend hooks --------------------------------------------------------------------------

    def _enqueue(self, turn: _Turn) -> None:
        """Start (or queue) ``turn``. The backend streams its text to ``turn.out`` and calls
        :meth:`_finish` when it is over. Turns run one at a time, in order."""
        raise NotImplementedError

    # -- MCP ----------------------------------------------------------------------------------

    def _load_mcp(self) -> McpSetup:
        try:
            return load_mcp_servers(self.settings)
        except McpConfigError as exc:
            # A broken mcp_servers.json must not take the voice loop down: run without external
            # connectors (the in-process spark server still works) and say why.
            log.error("MCP config unusable, running without external connectors: %s", exc)
            setup = McpSetup()
            setup.warnings.append(str(exc))
            return setup

    def _log_connectors(self) -> None:
        """Which servers are on or off. The loader already logged each disabled server's reason
        (it names variables, never values), so this is the one-line summary."""
        enabled = sorted({*self.mcp.servers, *self.mcp.claudeai})
        log.info("MCP servers enabled: %s", ", ".join(enabled) if enabled else "none")
        if self.mcp.disabled:
            log.info("MCP servers disabled: %s", ", ".join(sorted(self.mcp.disabled)))
        log.debug("MCP pre-approved tools: %s", ", ".join(self.mcp.allowed_tools) or "none")

    def _check_fast_tools(self) -> None:
        """Warn about ``routing.fast_tools`` entries that can never match a tool call."""
        for name in self.routing.fast_tools:
            if any(c in name for c in "*?["):
                log.warning(
                    "routing.fast_tools: %r is a pattern; only exact full tool names match", name
                )
            elif self.mcp.policy.classify(name) == "unknown":
                log.warning(
                    "routing.fast_tools: %s is not a known tool (a typo, or its server is off)",
                    name,
                )

    @property
    def external_connectors(self) -> list[str]:
        """Enabled MCP servers other than Spark's own in-process one (claude.ai ones included)."""
        from spark.tools.server import SERVER_NAME

        return sorted({s for s in self.mcp.servers if s != SERVER_NAME} | set(self.mcp.claudeai))

    @property
    def connectors(self) -> list[str]:
        """Enabled MCP server names (external and in-process), for ``spark_status``."""
        return sorted(self.mcp.servers)

    # -- researcher worker (P5.7d) ----------------------------------------------------------------

    @property
    def worker_model(self) -> str:
        return self.models_cfg.worker_model or self.models_cfg.fast_model

    def worker_tools(self) -> list[str]:
        """What the researcher may use: the connectors' read-only tools plus the notes tools."""
        from spark.tools.server import MEMORY_TOOLS, SERVER_NAME, full_tool_name

        external = [
            t
            for t in self.mcp.allowed_tools
            if t.startswith("mcp__")
            and not any(c in t for c in "*?[")
            and t.split("__")[1] != SERVER_NAME
        ]
        notes = [full_tool_name(t) for t in MEMORY_TOOLS] if self.memory is not None else []
        return external + notes

    @property
    def workers_active(self) -> bool:
        """The researcher exists: workers are on and some connector has read-only tools."""
        if not self.config.workers:
            return False
        from spark.tools.server import SERVER_NAME

        return any(
            t.startswith("mcp__") and t.split("__")[1] != SERVER_NAME
            for t in self.mcp.allowed_tools
        )

    # -- routing state for the tool gate ------------------------------------------------------

    @property
    def current_route(self) -> str | None:
        """The route of the turn being run ("fast" or "deep"), or None between turns, while
        priming, or with routing disabled. The ``PreToolUse`` gate reads it per tool call."""
        return self._current_route

    def _route_for_gate(self) -> str | None:
        return self._current_route

    def _turn_route(self, turn: _Turn) -> str | None:
        if turn.decision is None or not self.router.config.enabled or turn.mode_forced:
            return None  # priming, routing disabled, or a model the user picked: no limit
        return turn.decision.route

    def _note_fast_denied(self, tool_name: str) -> None:
        """The gate refused a tool on the fast route and told the model to have the user ask
        again. Route the next turn deep so that retry can actually run it."""
        self._escalate_next = True

    def step_budget(self, turn: _Turn | None = None) -> int:
        """Agent steps one request may use: ``deep_max_turns`` on the deep route, otherwise
        ``max_turns``."""
        turn = turn if turn is not None else self._active
        if turn is not None and turn.decision is not None and turn.decision.route == DEEP:
            return self.config.deep_max_turns
        return self.config.max_turns

    def _step_precheck(self, tool_name: str, data: Mapping[str, Any] | None = None) -> str | None:
        """The gate's ``precheck``: count the active turn's tool calls, note deep connectors,
        and refuse a call past the step budget (the final answer needs one step too). A worker
        sub-agent's calls (``agent_id`` in the hook input) only note the connector."""
        from spark.brain.confirm import REASON_STEP_BUDGET

        server = self._deep_server_of(tool_name)
        if server is not None and self._sticky_deep is None:
            log.info("brain: conversation uses %s; later turns go deep", server)
            self._sticky_deep = server
        turn = self._active
        if turn is None or (data or {}).get("agent_id"):
            return None  # between turns, or the researcher's own call (it has worker_max_turns)
        fast = turn.decision is not None and turn.decision.route != DEEP
        if server is not None and fast and self.router.config.enabled and not turn.mode_forced:
            # Too big for the fast route: stop this turn quietly; respond() re-runs it deep.
            log.info("brain: fast turn reached deep connector %s; re-running it deep", server)
            turn.escalate = server
            turn.interrupted = True  # drop anything more this turn says
            turn.interrupt.set()
            return REASON_ESCALATING
        turn.tool_calls += 1
        turn.tool_started.set()
        if turn.tool_calls > max(1, self.step_budget(turn) - 1):
            return REASON_STEP_BUDGET
        return None

    def _deep_server_of(self, tool_name: str) -> str | None:
        if not tool_name.startswith("mcp__"):
            return None
        server = tool_name[len("mcp__") :].partition("__")[0]
        return server if server in self.mcp.deep_route else None

    def _deep_mention(self, text: str) -> str | None:
        """The ``route: deep`` server a request names (whole words), or None."""
        low = text.lower()
        for server, words in self.mcp.deep_route.items():
            for word in words:
                if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", low):
                    return server
        return None

    def _route_deep_connectors(self, text: str, decision: RouteDecision) -> RouteDecision:
        if decision.route == DEEP:
            self._went_deep = True
            return decision
        if self._went_deep:
            return self.router.escalate(decision, "the conversation already went deep")
        if (server := self._deep_mention(text)) is not None:
            return self.router.escalate(decision, f"names deep connector {server}")
        if self._sticky_deep is not None:
            return self.router.escalate(
                decision, f"conversation is using deep connector {self._sticky_deep}"
            )
        return decision

    # -- model mode (P5.7c) ---------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """``auto`` (routing), ``fast`` or ``deep`` (every request on that model)."""
        return self.config.mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("auto", "fast", "deep"):
            raise ValueError(f"unknown model mode {mode!r}")
        if mode != self.config.mode:
            log.info("brain: model mode %s -> %s", self.config.mode, mode)
        self.config.mode = mode

    def set_models(
        self, fast: str | None = None, deep: str | None = None, worker: str | None = None
    ) -> None:
        """Change the model behind a route. The next turn on that route uses it. ``worker``:
        the researcher's model ('' = same as fast; None = unchanged)."""
        roles = self.models_cfg
        if fast and fast.strip():
            roles.fast_model = self.router.fast_model = fast.strip()
        if deep and deep.strip():
            roles.deep_model = self.router.deep_model = deep.strip()
        if worker is not None:
            roles.worker_model = worker.strip() or None
        log.info(
            "brain: models fast=%s deep=%s researcher=%s",
            roles.fast_model,
            roles.deep_model,
            self.worker_model,
        )

    def _model_for(self, route: str) -> str:
        return self.models_cfg.deep_model if route == DEEP else self.models_cfg.fast_model

    # -- connector health (P5.7i) ----------------------------------------------------------------

    def unreachable_connectors(self) -> list[str]:
        return sorted(
            name
            for name, info in self.mcp_status.items()
            if info.get("status") in ("failed", "needs-auth")
        )

    def _with_status_note(self, text: str) -> str:
        """Tell the model once per session which connectors can't be reached right now."""
        from spark.brain import claudeai

        new = [n for n in self.unreachable_connectors() if n not in self._status_noted]
        if not new:
            return text
        self._status_noted.update(new)
        names = " and ".join(new)
        verb = "is" if len(new) == 1 else "are"
        signin = [
            claudeai.display_name(n)
            for n in new
            if claudeai.is_claudeai(n) and self.mcp_status[n].get("status") == "needs-auth"
        ]
        extra = (
            f" For {' and '.join(signin)} the user needs to {claudeai.SIGN_IN_HINT}; say that."
            if signin
            else ""
        )
        return (
            f"{text}\n\n(Status from {self.settings.app.name}, not from the user: {names} "
            f"{verb} unreachable right now. If this request needs it, say so in one sentence "
            f"instead of trying its tools.{extra})"
        )

    def _apply_mcp_status(self, current: dict[str, dict[str, str]]) -> None:
        """Store the live status per server; log changes once and tell ``on_mcp_status``."""
        if current == self.mcp_status:
            return
        for name, info in sorted(current.items()):
            before = self.mcp_status.get(name, {}).get("status")
            if before == info["status"]:
                continue
            if info["status"] in ("failed", "needs-auth"):
                log.warning(
                    "MCP %s: %s%s",
                    name,
                    info["status"],
                    f" ({info['error']})" if info.get("error") else "",
                )
            else:
                log.info("MCP %s: %s", name, info["status"])
        self.mcp_status = current
        if self.on_mcp_status is not None:
            try:
                self.on_mcp_status({k: dict(v) for k, v in current.items()})
            except Exception as exc:
                log.debug("brain: on_mcp_status failed: %s", exc)

    # -- session prompt -----------------------------------------------------------------------

    def _session_prompt(self) -> str:
        """The system prompt plus the briefing for a new session (``context_provider``)."""
        context = ""
        if self.context_provider is not None:
            try:
                context = (self.context_provider() or "").strip()
            except Exception as exc:
                log.warning("brain: briefing failed (%s); starting without it", exc)
        self.last_context = context
        if context:
            log.info("brain: new session with a %d-character briefing", len(context))
            return f"{self.system_prompt}\n\n{context}"
        return self.system_prompt

    # -- public API ---------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """What every reset clears, before the backend starts its fresh session."""
        self.resets += 1
        self._last_turn_end = None
        self._escalate_next = False  # a fresh conversation has nothing to retry
        self._sticky_deep = None
        self._went_deep = False
        self._status_noted = set()

    async def respond(self, text: str) -> AsyncIterator[str]:
        self._check_open()
        if self._idle_expired():
            log.info(
                "brain: idle for more than %.0f s, starting a fresh conversation",
                self.config.idle_reset_s,
            )
            await self.reset()
        matched, rest = split_new_topic(
            text, self.config.new_topic_phrases, self.settings.app.wake_names
        )
        if matched:
            log.info("brain: new topic requested")
            await self.reset()
            if not rest:
                self._last_turn_end = self._clock.monotonic()
                yield self.voices.line("fresh_start")
                return
            text = rest

        forced, self.next_route = self.next_route, None
        wins, self.next_route_wins = self.next_route_wins, False
        mode = self.config.mode
        if wins and forced in ("fast", "deep"):
            mode = forced  # "use the deep model to ...": this request, whatever the mode
            decision = RouteDecision(
                forced, self._model_for(forced), self.next_route_reason or f"asked {forced}"
            )
            log.info("route: %s (%s)", decision.route, decision.reason)
        elif mode in ("fast", "deep"):
            # The user picked the model: no routing, escalation, or shortcut route.
            decision = RouteDecision(mode, self._model_for(mode), f"model mode: always {mode}")
            log.info("route: %s (%s)", decision.route, decision.reason)
            self._escalate_next = False
        elif forced in ("fast", "deep") and self.router.config.enabled:
            decision = RouteDecision(
                forced, self._model_for(forced), self.next_route_reason or f"forced {forced}"
            )
            log.info("route: %s (%s)", decision.route, decision.reason)
        else:
            decision = await self.router.decide(text)
        if self._escalate_next:
            self._escalate_next = False
            decision = self.router.escalate(decision, "retry after a fast-route tool refusal")
        if mode == "auto":
            decision = self._route_deep_connectors(text, decision)
        self.activity.record(
            "route", route=decision.route, model=decision.model, reason=decision.reason
        )
        self._check_open()
        text = self._with_status_note(text)
        turn = _Turn(text, decision=decision, mode_forced=mode != "auto")
        self._enqueue(turn)
        ack_pending = self.routing.ack_after_s > 0
        pending_get: asyncio.Future[Any] | None = None
        try:
            while True:
                if ack_pending:
                    ack_pending = False
                    item, pending_get = await self._first_item_or_ack(
                        turn, after_tool=decision.route != DEEP
                    )
                    if item is _ACK:
                        turn.acked = True
                        if turn.stats is not None:
                            turn.stats.acked = True
                        log.info(
                            "brain: no reply after %.1f s, acknowledging", self.routing.ack_after_s
                        )
                        yield self.ack_text + " "
                        continue
                elif pending_get is not None:
                    item = await pending_get
                    pending_get = None
                else:
                    item = await turn.out.get()
                if item is _END:
                    if turn.escalate is None:
                        return
                    decision = self.router.escalate(
                        decision, f"fast turn reached deep connector {turn.escalate}"
                    )
                    self.activity.record(
                        "route", route=decision.route, model=decision.model, reason=decision.reason
                    )
                    acked = turn.acked
                    turn = _Turn(text, decision=decision, acked=acked)
                    self._enqueue(turn)
                    ack_pending = not acked and self.routing.ack_after_s > 0
                    continue
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if pending_get is not None and not pending_get.done():
                pending_get.cancel()
            if not turn.done.is_set():
                # The consumer stopped early (barge-in or cancellation): stop generating.
                turn.interrupted = True
                turn.interrupt.set()
            self._last_turn_end = self._clock.monotonic()

    async def _first_item_or_ack(
        self, turn: _Turn, *, after_tool: bool = False
    ) -> tuple[Any, asyncio.Future[Any] | None]:
        """The turn's first output item, or ``_ACK`` once ``routing.ack_after_s`` of clock time
        passed without one. Time while a confirmation question is pending doesn't count.
        ``after_tool`` (fast turns): the clock only starts once the turn calls a tool.

        Returns ``(item, None)``, or ``(_ACK, pending_get)`` where ``pending_get`` is the still
        running ``out.get()`` to await next (so no item is lost to a cancelled get)."""
        remaining = self.routing.ack_after_s
        get: asyncio.Future[Any] = asyncio.ensure_future(turn.out.get())
        waiters: list[asyncio.Future[Any]] = []
        try:
            if after_tool and not turn.tool_started.is_set():
                waiters = [asyncio.ensure_future(turn.tool_started.wait())]
                await asyncio.wait([get, *waiters], return_when=asyncio.FIRST_COMPLETED)
                if get.done():
                    return get.result(), None
            while True:
                for waiter in waiters:
                    waiter.cancel()
                if not self._confirm_idle.is_set():
                    waiters = [asyncio.ensure_future(self._confirm_idle.wait())]
                    await asyncio.wait([get, *waiters], return_when=asyncio.FIRST_COMPLETED)
                    if get.done():
                        return get.result(), None
                    continue
                start = self._clock.monotonic()
                waiters = [
                    asyncio.ensure_future(self._clock.sleep(remaining)),
                    asyncio.ensure_future(self._confirm_busy.wait()),
                ]
                await asyncio.wait([get, *waiters], return_when=asyncio.FIRST_COMPLETED)
                if get.done():
                    return get.result(), None
                remaining -= self._clock.monotonic() - start
                if remaining <= 1e-9 and self._confirm_idle.is_set():
                    return _ACK, get
        except BaseException:
            get.cancel()
            raise
        finally:
            for waiter in waiters:
                waiter.cancel()

    def note_confirming(self, active: bool) -> None:
        """A confirmation question started (True) or ended (False). Pauses the ack timer."""
        self._confirming = max(0, self._confirming + (1 if active else -1))
        if self._confirming:
            self._confirm_idle.clear()
            self._confirm_busy.set()
        else:
            self._confirm_busy.clear()
            self._confirm_idle.set()

    async def interrupt(self) -> None:
        """Abort the in-flight (and any queued) turn and wait until it has drained."""
        for turn in list(self._queued):
            turn.interrupted = True
            turn.interrupt.set()
        turn = self._active
        if turn is None or turn.done.is_set():
            return
        turn.interrupted = True
        turn.interrupt.set()
        try:
            await asyncio.wait_for(turn.done.wait(), timeout=self.interrupt_timeout_s)
        except TimeoutError:
            log.warning(
                "brain: interrupted turn did not finish within %.1f s", self.interrupt_timeout_s
            )

    # -- helpers ------------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise BrainClosed("the brain has been closed")

    def _idle_expired(self) -> bool:
        if self._last_turn_end is None:
            return False
        return self._clock.monotonic() - self._last_turn_end > self.config.idle_reset_s

    def _finish(self, turn: _Turn, error: BaseException | None = None) -> None:
        if turn in self._queued:
            self._queued.remove(turn)
        turn.done.set()
        if error is not None and not turn.interrupted:
            turn.out.put_nowait(error)
        turn.out.put_nowait(_END)

    # -- tool-call ordering for the confirmation hook ------------------------------------------

    def _tool_event(self, tool_use_id: str) -> asyncio.Event:
        event = self._tool_seen.get(tool_use_id)
        if event is None:
            event = self._tool_seen[tool_use_id] = asyncio.Event()
        return event

    def _announce_tool(self, turn: _Turn, tool_use_id: Any, text_emitted: bool) -> bool:
        """The stream reached a tool call. Push a sentence break so the app's splitter
        releases any unfinished sentence, then wake a gate waiting on this call.

        Returns True if a newline was emitted."""
        if not isinstance(tool_use_id, str) or not tool_use_id:
            return False
        event = self._tool_event(tool_use_id)
        if event.is_set():
            return False
        pushed = text_emitted and not turn.interrupted
        if pushed:
            turn.out.put_nowait("\n")
        event.set()
        return pushed

    async def _await_tool_announced(self, tool_use_id: str | None) -> None:
        """``before_ask`` for the gate. The tool call may reach the gate before the text that
        preceded it reached the app. Wait (briefly) until it has, so the app speaks that text
        before the question."""
        if tool_use_id:
            try:
                async with asyncio.timeout(self.tool_announce_wait_s):
                    await self._tool_event(tool_use_id).wait()
            except TimeoutError:
                log.debug("brain: tool call %s not seen in the stream; asking anyway", tool_use_id)
        # One more loop pass lets the app's speech pump take the flushed sentence first.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
