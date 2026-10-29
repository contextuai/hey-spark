"""OpenAICompatBrain: Spark's brain on any OpenAI-compatible Chat Completions endpoint (P5.7e).

Selected with ``brain.backend: openai_compat`` (OpenAI, Azure OpenAI, OpenRouter, Groq,
Together, Ollama, LM Studio, vLLM; presets in :mod:`spark.brain.openai_presets`). Everything that
is about Spark rather than the provider comes from :class:`spark.brain.core.VoiceBrain`, the same
code the Claude brain runs: the voice prompt and briefing, routing and model mode, the step
budget, the "On it" acknowledgement, deep connectors, idle reset and "new topic", connector
health, and the ONE tool gate (:class:`spark.brain.confirm.ConfirmationGate`).

How a turn runs:

* The conversation is a list of Chat Completions messages, kept between turns (the session).
  The system message is the voice prompt plus the briefing, built when a session starts.
  A reset (idle, "new topic", new conversation) clears the list and rebuilds the briefing.
* Each request streams (:class:`spark.brain.openai_chat.ChatClient`). Reply text goes to the app
  as it arrives, so sentence streaming works exactly as on Claude. When a tool call starts, a
  sentence break is pushed so "Sure, I'll set that up." is spoken before any confirmation.
* Every tool call passes the gate with the same hook input the CLI sends (``tool_name``,
  ``tool_input``, ``agent_id`` for the researcher), and :func:`spark.brain.confirm.hook_verdict`
  applies the result the way the CLI does: allow runs, deny returns the reason to the model,
  no decision runs only pre-approved (read-only) tools. Results go back as ``role: tool``
  messages (cut to ``max_tool_result_chars``; the activity log keeps the whole result).
* Step budget: the gate's precheck refuses calls past ``max_turns`` / ``deep_max_turns`` with
  "answer with what you found". As a backstop, after budget + 2 rounds a last request is sent
  with no tools, so the model has to answer.
* Only tools the policy allows are offered (read-only, side-effecting, and destructive ones on
  servers with ``allow_destructive``), mirroring the Claude brain's ``disallowed_tools``.
* MCP servers are run by Spark itself (:class:`spark.brain.mcp_client.McpManager`), with ``${VAR}``
  expanded in this process. Spark's own tools (spark/tools) are called through their handlers.
* Researcher (``brain.workers``): the voice model gets a ``research`` function. It runs a nested
  tool loop on ``worker_model`` with only the read-only connector tools and the connector-notes
  tools, ``worker_max_turns`` rounds, every call gated and logged with ``agent: researcher``,
  and returns a report the voice model interprets. It is never spoken directly.
* Barge-in: :meth:`interrupt` cancels the turn's task, which closes the HTTP stream at once.
* Not here: claude.ai account connectors (they come from the Claude CLI login). Enabled ones
  are skipped with one warning, and shown as unreachable so the model says so.

``httpx`` and ``mcp`` are imported lazily (in :class:`ChatClient` and when a server connects).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from spark.activity import ActivityLog, NullActivityLog
from spark.brain.confirm import ConfirmationGate, Confirmer, HookCallback, Summarizer, hook_verdict
from spark.brain.core import (
    RESEARCHER_DESCRIPTION,
    RESEARCHER_PROMPT,
    BrainClosed,
    BrainError,
    TurnStats,
    VoiceBrain,
    _Turn,
)
from spark.brain.mcp_client import ClientFactory, InProcessServer, McpManager
from spark.brain.mcp_config import McpSetup
from spark.brain.openai_presets import Provider, resolve_provider
from spark.brain.openai_tools import ToolNameMap, clip_result, function_tool
from spark.brain.prompts import worker_rules
from spark.brain.router import DEEP, Router
from spark.interfaces import BackendUnavailable
from spark.util.clock import SystemClock

if TYPE_CHECKING:
    from spark.brain.openai_chat import ChatClient, ChatResult, ToolCall
    from spark.config import Settings
    from spark.memory import MemoryStore
    from spark.tools.project import ProjectService
    from spark.tools.status import StatusProvider
    from spark.tools.timers import TimerService
    from spark.util.clock import Clock

__all__ = ["RESEARCH_TOOL", "OpenAICompatBrain"]

log = logging.getLogger("spark.brain.openai")

BACKEND = "openai_compat"
RESEARCH_TOOL = "research"
CLAUDEAI_ONLY = "claude.ai connectors work only with the Claude brain"
#: Above this many characters of history, old tool results are shortened before a request.
HISTORY_SOFT_LIMIT = 150_000
OLD_RESULT_CHARS = 400
REASON_INTERRUPTED = "Not run: the user interrupted."
REASON_BAD_ARGUMENTS = (
    "Error: the arguments were not a valid JSON object. Call the tool again with valid JSON."
)
BUDGET_NOTE = (
    "(Step budget used up: no more tools. Answer now with what you found, in one or two short "
    "spoken sentences, and offer to keep digging.)"
)
WORKER_BUDGET_NOTE = "Step budget used up: no more tools. Report what you found so far."

_RESEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "brief": {
            "type": "string",
            "description": (
                "A precise brief: what to find, for which time range or filter, and what to "
                "report back."
            ),
        }
    },
    "required": ["brief"],
}


class OpenAICompatBrain(VoiceBrain):
    """Built by the registry as ``OpenAICompatBrain(settings, **deps)``.

    Takes the same dependencies as :class:`spark.brain.claude_agent.ClaudeAgentBrain` (timers,
    status_provider, confirmer, activity, context_provider, memory, ...). Test seams:

    Args:
        transport: an ``httpx`` transport for the chat client (``httpx.MockTransport``).
        chat: a ready :class:`ChatClient` (overrides ``transport``).
        mcp_client_factory: builds MCP clients for :class:`McpManager` (tests pass fakes).
        environ: where the API key and ``${VAR}`` values come from (default ``os.environ``).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Clock | None = None,
        transport: Any | None = None,
        chat: ChatClient | None = None,
        mcp_client_factory: ClientFactory | None = None,
        environ: Mapping[str, str] | None = None,
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
        self.oc = settings.brain.openai_compat
        self.deps = deps
        self.provider: Provider = resolve_provider(self.oc)
        self._check_provider(environ)
        # Fill in the preset's suggested names, so the window, prefs and "which model" see them.
        self.oc.fast_model = self.provider.fast_model
        self.oc.deep_model = self.provider.deep_model
        self.models_cfg = self.oc
        self.activity: ActivityLog = activity or NullActivityLog()
        self.context_provider = context_provider
        self.memory = memory
        self._clock: Clock = clock or SystemClock()
        self.turn_timeout_s = turn_timeout_s
        self.interrupt_timeout_s = interrupt_timeout_s
        self.close_timeout_s = close_timeout_s
        if chat is None:
            from spark.brain.openai_chat import ChatClient

            chat = ChatClient(
                self.provider, self.oc, transport=transport, clock=self._clock, environ=environ
            )
        self.chat = chat
        self.timers = timers
        self.confirmer = confirmer
        self.summarizers: dict[str, Summarizer] = dict(summarizers or {})
        self.mcp: McpSetup = mcp if mcp is not None else self._load_mcp()
        self.claudeai_skipped: list[str] = self._skip_claudeai()
        if timers is not None:
            self._add_spark_tools(timers, status_provider, projects)
        self.routing = settings.routing
        self.router: Router = router or self._build_router()
        self._init_core(
            tool_gate=tool_gate,
            delegate_tool=RESEARCH_TOOL,
            worker_rules=worker_rules("the research tool"),
        )
        self.model: str = self._model_for(DEEP if self.config.mode == "deep" else "fast")
        self.manager = McpManager(
            self.mcp.servers,
            base_dir=settings.config_dir,
            environ=environ,
            client_factory=mcp_client_factory,
            connect_timeout_s=min(30.0, self.config.connect_timeout_s),
        )
        self.manager.on_status = self._on_manager_status
        self.available_models: list[str] = []  # what GET /models listed (the window's picker)
        self._messages: list[dict[str, Any]] = []
        self._system: str | None = None
        self._turn_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._prime_task: asyncio.Task[None] | None = None
        log.info(
            "brain: %s at %s (fast=%s, deep=%s, researcher=%s)",
            self.provider.label,
            self.provider.base_url,
            self.oc.fast_model,
            self.oc.deep_model,
            self.worker_model,
        )

    # -- setup ------------------------------------------------------------------------------------

    def _check_provider(self, environ: Mapping[str, str] | None) -> None:
        prov = self.provider
        if prov.problems:
            raise BackendUnavailable(
                BACKEND,
                "; ".join(prov.problems),
                "fix it in the Brain window, or edit brain.openai_compat in config.yaml",
            )
        missing = prov.missing_env(environ)
        if missing:
            raise BackendUnavailable(
                BACKEND,
                f"{', '.join(missing)} is not set (needed for {prov.label})",
                f"enter the key in the Brain window, or put {missing[0]}=... in .env",
            )

    def _skip_claudeai(self) -> list[str]:
        """claude.ai account connectors come from the Claude CLI login: not available here."""
        skipped = list(self.mcp.claudeai)
        if skipped:
            log.warning(
                "brain: %s skipped: %s (switch brain.backend to claude_agent to use them)",
                ", ".join(skipped),
                CLAUDEAI_ONLY,
            )
            self.mcp.claudeai = []
        return skipped

    def _add_spark_tools(
        self,
        timers: TimerService,
        status_provider: StatusProvider | None,
        projects: ProjectService | None,
    ) -> None:
        from spark.tools.project import ProjectService
        from spark.tools.server import (
            SERVER_NAME,
            build_spark_tools,
            spark_confirmation_summarizers,
            spark_tool_lists,
        )

        if SERVER_NAME in self.mcp.servers:
            log.info("MCP: in-process %r server already configured", SERVER_NAME)
            return
        service = projects if projects is not None else ProjectService.from_settings(self.settings)
        tools = build_spark_tools(
            timers,
            status_provider,
            projects=service,
            memory=self.memory,
            connectors=lambda: self.external_connectors,
            activity=self.activity,
        )
        read_only, side_effecting = spark_tool_lists(memory=self.memory is not None)
        self.mcp.add_sdk_server(
            SERVER_NAME, InProcessServer(tools), read_only=read_only, side_effecting=side_effecting
        )
        for name, fn in spark_confirmation_summarizers(service).items():
            self.summarizers.setdefault(name, fn)

    def _build_router(self) -> Router:
        cfg = self.routing
        classifier = None
        if cfg.enabled and cfg.classifier == "fast_model":
            from spark.brain.openai_side import OpenAIClassifier

            classifier = OpenAIClassifier(self.chat, lambda: self.models_cfg.fast_model)
        return Router(
            cfg, fast_model=self.oc.fast_model, deep_model=self.oc.deep_model, classifier=classifier
        )

    def _on_manager_status(self, status: dict[str, dict[str, str]]) -> None:
        current = {k: dict(v) for k, v in status.items()}
        for key in self.claudeai_skipped:
            current[key] = {"status": "failed", "error": CLAUDEAI_ONLY}
        self._apply_mcp_status(current)

    # -- public API -------------------------------------------------------------------------------

    async def warmup(self) -> None:
        """Connect the MCP servers, list the endpoint's models, build the session prompt."""
        self._check_open()
        self._on_manager_status(self.manager.status)
        await asyncio.gather(
            self.manager.start(wait_s=self.config.connect_timeout_s),
            self._fetch_models(),
            self.router.warmup(),
        )
        if self._system is None:
            self._system = self._session_prompt()
        self._start_prime()

    def _start_prime(self) -> None:
        """Local servers: read the session prompt and tools into the model in the background.

        Every request starts with the voice prompt and the tool list (~3,000 tokens). A local
        model on a CPU or a small GPU needs 10-30 s to read that, and Ollama, LM Studio and
        llama.cpp keep the result (a prompt cache) for the next request with the same start.
        Priming pays that cost now instead of on the user's first question, and loads the model.
        Cloud APIs are fast at this, so they aren't primed.
        """
        if not (self.provider.local and self.config.prime_on_connect) or self._closed:
            return
        if self._prime_task is not None and not self._prime_task.done():
            self._prime_task.cancel()
        self._prime_task = asyncio.create_task(self._prime(), name="spark-brain-prime")
        self._tasks.add(self._prime_task)
        self._prime_task.add_done_callback(self._tasks.discard)

    async def _prime(self) -> None:
        tools, _names = self._offered_tools()
        messages = [{"role": "system", "content": self._system}, {"role": "user", "content": "Hi"}]
        # Only the model most turns use: priming a second one can push the first out of memory.
        model = self.model
        start = self._clock.monotonic()
        try:
            await self.chat.stream(model, messages, tools, max_tokens=1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.info("brain: priming %s failed (%s); the first answer may be slow", model, exc)
            return
        log.info("brain: primed %s in %.1f s", model, self._clock.monotonic() - start)

    async def _fetch_models(self) -> None:
        from spark.brain.openai_chat import ProviderError

        try:
            self.available_models = await self.chat.list_models()
        except ProviderError as exc:
            log.warning("brain: could not list %s models: %s", self.provider.label, exc)
            return
        except Exception as exc:
            log.debug("brain: listing models failed: %s", exc)
            return
        if not self.available_models:
            return
        listed = set(self.available_models)
        for role, model in (("fast", self.oc.fast_model), ("deep", self.oc.deep_model)):
            if model and model not in listed:
                log.warning(
                    "brain: %s model %s is not in %s's model list; requests to it may fail",
                    role,
                    model,
                    self.provider.label,
                )

    async def reset(self) -> None:
        """Start a fresh conversation: clear the history and rebuild the briefing."""
        self._reset_state()
        async with self._turn_lock:
            self._messages = []
            self._system = self._session_prompt()
        self._start_prime()

    async def close(self) -> None:
        """Stop any turn, close the MCP sessions and the HTTP client. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for turn in [*self._queued, *([self._active] if self._active else [])]:
            turn.interrupted = True
            turn.interrupt.set()
        tasks = list(self._tasks)
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=self.close_timeout_s)
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(BaseException):
                    await task
        for closer in (self.router.aclose, self.manager.aclose, self.chat.aclose):
            try:
                await closer()
            except Exception as exc:
                log.debug("brain: close step failed: %s", exc)

    # -- turns ------------------------------------------------------------------------------------

    def _enqueue(self, turn: _Turn) -> None:
        self._queued.append(turn)
        task = asyncio.create_task(self._handle_turn(turn), name="spark-brain-turn")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_turn(self, turn: _Turn) -> None:
        async with self._turn_lock:
            if turn.interrupt.is_set() or self._closed:
                self._finish(
                    turn, BrainClosed("the brain has been closed") if self._closed else None
                )
                return
            self._active = turn
            self._current_route = self._turn_route(turn)
            error: BaseException | None = None
            inner = asyncio.create_task(self._run_turn(turn), name="spark-brain-request")
            stop = asyncio.create_task(turn.interrupt.wait())
            try:
                done, _ = await asyncio.wait(
                    {inner, stop}, timeout=self.turn_timeout_s, return_when=asyncio.FIRST_COMPLETED
                )
                if inner in done:
                    inner.result()
                else:
                    inner.cancel()  # barge-in, escalation or timeout: close the stream now
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await inner
                    if not turn.interrupt.is_set():
                        error = BrainError(
                            f"{self.provider.label} did not finish within "
                            f"{self.turn_timeout_s:.0f} s"
                        )
            except asyncio.CancelledError:
                inner.cancel()
                self._current_route = None
                self._active = None
                self._finish(turn, BrainClosed("the brain was shut down mid-turn"))
                raise
            except (BrainError, BackendUnavailable) as exc:
                error = exc
            except Exception as exc:
                log.debug("brain: turn failed", exc_info=True)
                error = BrainError(
                    f"{self.provider.label} request failed: {type(exc).__name__}: {exc}"
                )
            finally:
                stop.cancel()
            self._current_route = None
            self._active = None
            self._finish(turn, error)

    def _offered_tools(
        self, allowed: set[str] | None = None
    ) -> tuple[list[dict[str, Any]], ToolNameMap]:
        """The ``tools`` list for a request and its name map. Voice model (``allowed`` None):
        every tool the policy permits plus ``research``; the researcher: only ``allowed``."""
        policy = self.mcp.policy
        specs = [
            s
            for s in self.manager.tools()
            if (
                s.name in allowed
                if allowed is not None
                else policy.classify(s.name) == "read" or policy.needs_confirmation(s.name)
            )
        ]
        names = ToolNameMap(s.name for s in specs)
        tools = [
            function_tool(names.function(s.name), s.description, s.input_schema) for s in specs
        ]
        if allowed is None and self.workers_active:
            names.add(RESEARCH_TOOL)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": RESEARCH_TOOL,
                        "description": RESEARCHER_DESCRIPTION,
                        "parameters": _RESEARCH_PARAMETERS,
                    },
                }
            )
        return tools, names

    def _history(self) -> list[dict[str, Any]]:
        """The conversation so far, with old tool results shortened past the soft limit."""
        total = sum(len(str(m.get("content") or "")) for m in self._messages)
        if total <= HISTORY_SOFT_LIMIT:
            return list(self._messages)
        out = []
        for m in self._messages:
            content = m.get("content")
            if (
                m.get("role") == "tool"
                and isinstance(content, str)
                and len(content) > OLD_RESULT_CHARS
                and total > HISTORY_SOFT_LIMIT
            ):
                total -= len(content) - OLD_RESULT_CHARS
                m = {**m, "content": content[:OLD_RESULT_CHARS] + " [older result shortened]"}
            out.append(m)
        return out

    async def _run_turn(self, turn: _Turn) -> None:
        if self._system is None:
            self._system = self._session_prompt()
        await self.manager.start(wait_s=self.config.connect_timeout_s)
        self.manager.ensure()
        decision = turn.decision
        model = decision.model if decision is not None else self.models_cfg.fast_model
        self.model = model
        stats = TurnStats(model=model, acked=turn.acked, partial=True)
        if decision is not None:
            stats.route = decision.route
            stats.route_reason = decision.reason
        turn.stats = stats
        start = self._clock.monotonic()
        gate = self.tool_gate if isinstance(self.tool_gate, ConfirmationGate) else None
        decisions_before = len(gate.decisions) if gate is not None else 0
        pending: list[dict[str, Any]] = [{"role": "user", "content": turn.text}]
        partial: list[str] = []  # text of the request being streamed (for a barge-in)
        emitted = 0
        need_sep = False
        last_char = ""
        announced = False
        usage_in = usage_out = 0
        rounds = 0

        def emit(text: str) -> None:
            nonlocal emitted, need_sep, last_char
            partial.append(text)
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

        def tool_started(index: int) -> None:
            nonlocal announced, last_char
            if not announced:
                announced = True
                if self._announce_tool(turn, f"r{rounds}-{index}", emitted > 0):
                    last_char = "\n"

        tools, names = self._offered_tools()
        hard_cap = self.step_budget(turn) + 2
        try:
            while not turn.interrupt.is_set():
                rounds += 1
                final = rounds > hard_cap
                messages = [{"role": "system", "content": self._system}, *self._history(), *pending]
                if final:
                    messages.append({"role": "user", "content": BUDGET_NOTE})
                need_sep, announced, partial = emitted > 0, False, []
                result = await self.chat.stream(
                    model,
                    messages,
                    None if final else tools or None,
                    on_text=emit,
                    on_tool_start=tool_started,
                )
                partial = []
                usage_in, usage_out = _add_usage(result, usage_in, usage_out)
                if stats.served_model is None and result.model:
                    stats.served_model = result.model
                stats.stop_reason = result.finish_reason
                if result.tool_calls and not final:
                    pending.append(_assistant_tool_message(result, names))
                    for call in result.tool_calls:
                        if turn.interrupt.is_set():
                            break
                        text = await self._run_tool_call(turn, call, names)
                        pending.append({"role": "tool", "tool_call_id": call.id, "content": text})
                    continue
                pending.append({"role": "assistant", "content": result.text})
                if result.finish_reason == "length":
                    log.info("brain: reply cut at max_tokens (%s)", self.oc.max_tokens)
                break
        finally:
            if partial and not turn.escalate:
                pending.append({"role": "assistant", "content": "".join(partial)})
            if turn.escalate is None:
                self._messages.extend(_repair(pending))
            self._tool_seen.clear()
            if gate is not None and decision is not None:
                for name, outcome in gate.decisions[decisions_before:]:
                    self.activity.record("confirm", tool=name, outcome=outcome)
            stats.total_s = self._clock.monotonic() - start
            stats.duration_ms = round(stats.total_s * 1000)
            stats.num_turns = rounds
            stats.interrupted = turn.interrupted
            if usage_in or usage_out:
                stats.usage = {"input_tokens": usage_in, "output_tokens": usage_out}
            self.last_turn = stats
            log.debug("brain turn: %s", stats.as_meta())
            if decision is not None:
                self.activity.record(
                    "turn_end",
                    num_turns=rounds,
                    route=stats.route,
                    model=stats.served_model or stats.model,
                    interrupted=stats.interrupted or None,
                )

    async def _run_tool_call(
        self,
        turn: _Turn,
        call: ToolCall,
        names: ToolNameMap,
        *,
        agent: str | None = None,
        model: str | None = None,
        allowed: set[str] | None = None,
    ) -> str:
        """Gate, run and log one tool call; returns the text for the ``role: tool`` message."""
        name = names.tool(call.name) or call.name
        try:
            args = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            log.info("tool call: %s (arguments are not a JSON object)", name)
            return REASON_BAD_ARGUMENTS
        if agent:
            log.info("researcher tool call: %s", name)
            self.activity.record(
                "tool_call", tool=name, id=call.id, input=args, agent=agent, model=model
            )
        else:
            log.info("tool call: %s", name)
            self.activity.tool_call(name, call.id, args)
        if allowed is not None and name not in allowed:
            reason = "Not available to the researcher: only read-only lookups are."
            log.info("tool denied: %s (researcher)", name)
            self.activity.tool_result(name, call.id, reason, True)
            return reason
        self._tool_event(call.id).set()  # the sentence before it was already pushed
        data: dict[str, Any] = {"tool_name": name, "tool_input": args, "tool_use_id": call.id}
        if agent:
            data["agent_id"] = agent
        output = await self.tool_gate(data, call.id, None)
        preapproved = name in self.mcp.allowed_tools or (
            name == RESEARCH_TOOL and self.workers_active
        )
        ok, reason = hook_verdict(output, preapproved=preapproved)
        if not ok:
            self.activity.tool_result(name, call.id, reason, True)
            return reason or "Not allowed."
        if name == RESEARCH_TOOL:
            text, is_error = await self._research(turn, str(args.get("brief") or args)), False
        else:
            outcome = await self.manager.call(name, args)
            text, is_error = outcome.text, outcome.is_error
        self.activity.tool_result(name, call.id, text, is_error)
        if is_error and not text.lower().startswith("error"):
            text = f"Error: {text}"
        return clip_result(text, self.oc.max_tool_result_chars)

    async def _research(self, turn: _Turn, brief: str) -> str:
        """The researcher: a nested tool loop on the worker model with read-only tools."""
        allowed = set(self.worker_tools())
        tools, names = self._offered_tools(allowed)
        model = self.worker_model
        system = RESEARCHER_PROMPT
        if self.last_context:
            system = f"{system}\n\n{self.last_context}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]
        budget = self.config.worker_max_turns
        log.info("brain: researcher on %s (%d tools)", model, len(tools))
        for round_no in range(budget + 1):
            final = round_no >= budget or not tools
            if round_no >= budget:
                messages.append({"role": "user", "content": WORKER_BUDGET_NOTE})
            result = await self.chat.stream(model, messages, None if final else tools)
            if result.tool_calls and not final:
                messages.append(_assistant_tool_message(result, names))
                for call in result.tool_calls:
                    if turn.interrupt.is_set():
                        break
                    text = await self._run_tool_call(
                        turn, call, names, agent="researcher", model=model, allowed=allowed
                    )
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
                if turn.interrupt.is_set():
                    break
                continue
            return result.text.strip() or "The researcher found nothing to report."
        return "The researcher was stopped before it finished."


def _add_usage(result: ChatResult, tokens_in: int, tokens_out: int) -> tuple[int, int]:
    usage = result.usage or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion = usage.get("completion_tokens") or usage.get("output_tokens")
    return (
        tokens_in + (prompt if isinstance(prompt, int) else 0),
        tokens_out + (completion if isinstance(completion, int) else 0),
    )


def _assistant_tool_message(result: ChatResult, names: ToolNameMap) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.text or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments or "{}"},
            }
            for c in result.tool_calls
        ],
    }


def _repair(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every assistant tool call must be answered by a ``role: tool`` message before the next
    request (an interrupted turn may have left some unanswered)."""
    out: list[dict[str, Any]] = []
    open_ids: list[str] = []

    def close_open() -> None:
        for call_id in open_ids:
            out.append({"role": "tool", "tool_call_id": call_id, "content": REASON_INTERRUPTED})
        open_ids.clear()

    for m in messages:
        if m.get("role") == "tool":
            if m.get("tool_call_id") in open_ids:
                open_ids.remove(m["tool_call_id"])
            out.append(m)
            continue
        close_open()
        out.append(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            open_ids = [c["id"] for c in m["tool_calls"]]
    close_open()
    return out
