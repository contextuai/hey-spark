"""Fast/deep routing (spec §8.4).

:class:`Router` decides, per user turn, whether the fast model (status questions, lookups, time
and timers, small talk) or the deep model (creation and multi-step work) answers it. The brain
then switches the ONE long-lived session to that model with ``client.set_model`` before the
query, so the conversation's context carries across fast and deep turns (docs/sdk_notes.md
"Routing (P3.2)").

Rules (``Router.route``, pure and instant; the order is the precedence):

1. a deep phrase anywhere ("new project", "summarize this week", "draft a plan", ...) -> deep
2. two or more imperative clauses ("check my email and draft replies") -> deep
3. more than ``routing.max_fast_words`` words -> deep
4. a fast keyword ("what time", "timer", "working on", "hello", ...) -> fast
5. a clause that starts with a deep verb ("write ...", "send ...", "let's set up ...") -> deep
6. otherwise -> fast

The fast route may only use a limited tool set (``routing.fast_tools``, by default the read-only
tools), enforced per call by :class:`spark.brain.confirm.ConfirmationGate`. So every verb that asks
for a side effect (create, send, delete, schedule, set up, ...) is a deep verb, and after a tool
call was refused on the fast route the brain makes the next turn deep (:meth:`Router.escalate`).

Optionally (``routing.classifier: fast_model``) a :class:`FastModelClassifier` asks the fast
model for one word, "fast" or "deep", within ``routing.classifier_timeout_s``; on timeout, error
or an unclear answer the rules decide. With the default ``rules`` no call is made, so routing
adds no latency.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from spark.config import RoutingConfig, Settings

__all__ = [
    "DEEP",
    "FAST",
    "FastModelClassifier",
    "RouteClassifier",
    "RouteDecision",
    "Router",
    "count_imperative_clauses",
    "normalize_request",
]

log = logging.getLogger(__name__)

Route = Literal["fast", "deep"]
FAST: Route = "fast"
DEEP: Route = "deep"

#: Verbs that open an imperative clause ("check my mail", "set a timer"), on top of the
#: configured deep verbs. Used to count clauses; one alone does not make a request deep.
IMPERATIVE_VERBS = frozenset(
    {
        "add",
        "book",
        "call",
        "cancel",
        "change",
        "check",
        "close",
        "delete",
        "email",
        "find",
        "fix",
        "get",
        "give",
        "list",
        "look",
        "make",
        "move",
        "open",
        "put",
        "read",
        "remind",
        "reply",
        "schedule",
        "send",
        "set",
        "show",
        "start",
        "stop",
        "tell",
        "turn",
        "update",
    }
)

#: Stripped from the start of a clause before looking at its first word.
_CLAUSE_PREFIXES = (
    "please",
    "can you",
    "could you",
    "would you",
    "will you",
    "let's",
    "lets",
    "let us",
    "help me",
    "i want you to",
    "i need you to",
    "i'd like you to",
    "i would like you to",
    "i want to",
    "i'd like to",
    "i would like to",
    "i need to",
    "we need to",
    "we should",
    "i wanna",
    "go ahead and",
    "and then",
    "then",
    "also",
    "and",
    "now",
    "just",
)
_WAKE_PREFIX = re.compile(r"^\s*(?:(?:hey|hi|ok|okay)\s+)?spark\b[\s,.!:;-]*", re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"[,.;:!?]+|\s+(?:and then|and|then|after that)\s+")
_WORD = re.compile(r"[a-z0-9']+")


def normalize_request(text: str) -> str:
    """Lowercase, straight apostrophes, no wake-word prefix ("Hey Spark, ..."), single spaces."""
    t = text.replace("’", "'").replace("‘", "'")
    t = _WAKE_PREFIX.sub("", t, count=1)
    return " ".join(t.lower().split())


def _strip_prefixes(clause: str) -> str:
    changed = True
    while changed and clause:
        changed = False
        for prefix in _CLAUSE_PREFIXES:
            if clause == prefix:
                return ""
            if clause.startswith(prefix + " "):
                clause = clause[len(prefix) + 1 :].lstrip()
                changed = True
    return clause


def _clauses(normalized: str) -> list[str]:
    out = []
    for raw in _CLAUSE_SPLIT.split(normalized):
        clause = _strip_prefixes(raw.strip())
        if clause:
            out.append(clause)
    return out


def _first_word(clause: str) -> str:
    match = _WORD.match(clause)
    return match.group(0) if match else ""


def count_imperative_clauses(normalized: str, verbs: Iterable[str] = ()) -> int:
    """How many clauses start with an imperative verb (after "please", "let's", ...)."""
    known = IMPERATIVE_VERBS | {v.lower() for v in verbs}
    return sum(1 for clause in _clauses(normalized) if _first_word(clause) in known)


def _phrase_pattern(phrases: Sequence[str]) -> re.Pattern[str] | None:
    cleaned = sorted({" ".join(p.lower().split()) for p in phrases if p.strip()}, key=len)
    if not cleaned:
        return None
    alts = "|".join(re.escape(p) for p in reversed(cleaned))  # longest first
    return re.compile(rf"(?<![a-z0-9'])(?:{alts})(?![a-z0-9'])")


@dataclass(frozen=True)
class RouteDecision:
    """Which model answers a turn, and why (logged, and kept in the turn stats)."""

    route: Route
    model: str
    reason: str

    def as_meta(self) -> dict[str, Any]:
        return {"route": self.route, "route_model": self.model, "route_reason": self.reason}


class RouteClassifier(Protocol):
    """Optional model-based classifier. Returns "fast", "deep", or None (undecided)."""

    async def classify(self, text: str) -> str | None: ...

    async def warmup(self) -> None: ...

    async def aclose(self) -> None: ...


class Router:
    """Picks fast or deep per turn. Build with :meth:`from_settings`.

    Args:
        config: the ``routing`` settings section.
        fast_model / deep_model: the model ids from ``brain`` (never hard-coded here).
        classifier: optional :class:`RouteClassifier`; None means rules only.
    """

    def __init__(
        self,
        config: RoutingConfig,
        *,
        fast_model: str,
        deep_model: str,
        classifier: RouteClassifier | None = None,
    ) -> None:
        self.config = config
        self.fast_model = fast_model
        self.deep_model = deep_model
        self.classifier = classifier
        self.decisions: list[RouteDecision] = []
        self._deep_phrases = _phrase_pattern(config.deep_phrases)
        self._fast_keywords = _phrase_pattern(config.fast_keywords)
        verbs = {" ".join(v.lower().split()) for v in config.deep_verbs if v.strip()}
        self._deep_verbs = frozenset(v for v in verbs if " " not in v)
        # multi-word verbs ("set up") match the start of a clause, longest first
        self._deep_verb_phrases = tuple(sorted((v for v in verbs if " " in v), key=len)[::-1])

    @classmethod
    def from_settings(cls, settings: Settings, *, sdk: ModuleType | None = None) -> Router:
        cfg = settings.routing
        classifier: RouteClassifier | None = None
        if cfg.enabled and cfg.classifier == "fast_model":
            classifier = FastModelClassifier(settings, sdk=sdk)
        return cls(
            cfg,
            fast_model=settings.brain.fast_model,
            deep_model=settings.brain.deep_model,
            classifier=classifier,
        )

    def _decision(self, route: Route, reason: str) -> RouteDecision:
        model = self.deep_model if route == DEEP else self.fast_model
        return RouteDecision(route, model, reason)

    def route(self, text: str) -> RouteDecision:
        """The rule-based decision (instant, no I/O). Does not log; :meth:`decide` does."""
        if not self.config.enabled:
            return self._decision(FAST, "routing disabled")
        t = normalize_request(text)
        if not t:
            return self._decision(FAST, "empty")
        if self._deep_phrases is not None:
            match = self._deep_phrases.search(t)
            if match:
                return self._decision(DEEP, f"phrase '{match.group(0)}'")
        clauses = count_imperative_clauses(t, self._deep_verbs)
        if clauses >= 2:
            return self._decision(DEEP, f"{clauses} imperative clauses")
        words = len(_WORD.findall(t))
        if words > self.config.max_fast_words:
            return self._decision(DEEP, f"{words} words")
        if self._fast_keywords is not None:
            match = self._fast_keywords.search(t)
            if match:
                return self._decision(FAST, f"keyword '{match.group(0)}'")
        for clause in _clauses(t):
            verb = self._clause_deep_verb(clause)
            if verb:
                return self._decision(DEEP, f"verb '{verb}'")
        return self._decision(FAST, "default")

    def _clause_deep_verb(self, clause: str) -> str | None:
        for phrase in self._deep_verb_phrases:
            if clause == phrase or clause.startswith(phrase + " "):
                return phrase
        verb = _first_word(clause)
        return verb if verb in self._deep_verbs else None

    def escalate(self, decision: RouteDecision, reason: str) -> RouteDecision:
        """Turn a fast decision into a deep one, logged like :meth:`decide`. The brain uses it
        for the turn after a tool call was refused on the fast route, so "ask again" works.
        With routing disabled (or an already deep decision) the decision is returned as is."""
        if not self.config.enabled or decision.route == DEEP:
            return decision
        deep = self._decision(DEEP, f"{reason}; rules: {decision.reason}")
        self.decisions.append(deep)
        log.info("route: %s (%s)", deep.route, deep.reason)
        return deep

    async def decide(self, text: str) -> RouteDecision:
        """The decision for a turn: the classifier if configured (bounded by
        ``classifier_timeout_s``, rules on any failure), else the rules. Logged at INFO."""
        decision = self.route(text)
        if self.classifier is not None and self.config.enabled:
            decision = await self._classify(text, decision)
        self.decisions.append(decision)
        log.info("route: %s (%s)", decision.route, decision.reason)
        return decision

    async def _classify(self, text: str, rules: RouteDecision) -> RouteDecision:
        assert self.classifier is not None
        try:
            async with asyncio.timeout(self.config.classifier_timeout_s):
                label = await self.classifier.classify(text)
        except TimeoutError:
            return self._decision(rules.route, f"classifier timeout; rules: {rules.reason}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("router: classifier failed: %s", exc)
            return self._decision(rules.route, f"classifier error; rules: {rules.reason}")
        if label in (FAST, DEEP):
            return self._decision(label, "classifier")  # type: ignore[arg-type]
        return self._decision(rules.route, f"classifier unsure; rules: {rules.reason}")

    async def warmup(self) -> None:
        if self.classifier is not None:
            try:
                await self.classifier.warmup()
            except Exception as exc:
                log.warning("router: classifier warm-up failed (%s); rules will decide", exc)

    async def aclose(self) -> None:
        if self.classifier is not None:
            await self.classifier.aclose()


# --------------------------------------------------------------------------------------------
# Optional fast-model classifier
# --------------------------------------------------------------------------------------------

CLASSIFIER_PROMPT = """\
You label requests to a voice assistant. Reply with exactly one lowercase word and nothing else.
Reply deep if the request needs multi-step work, creating something, or any action that changes \
something: starting or setting up a project, sending, deleting, scheduling, booking, posting, \
updating, planning, drafting or writing, summarizing many items, comparing, analyzing, \
refactoring, or several actions in one request.
Reply fast for everything else: a status question, a single lookup, the time or a timer \
(setting, listing or cancelling one), a simple fact, or small talk."""

_LABEL = re.compile(r"\b(fast|deep)\b")


@dataclass
class _Ask:
    text: str
    future: asyncio.Future[str | None]


@dataclass
class _ClassifierState:
    client: Any | None = None
    calls: int = 0


class FastModelClassifier:
    """Asks ``brain.fast_model`` for "fast" or "deep" on a separate, tool-less client.

    The session client can't be used for this (the label exchange would enter the user's
    conversation), so this keeps its own ``ClaudeSDKClient``, owned by its own worker task (the
    SDK client must be used from the task that connected it, docs/sdk_notes.md §1). It is
    reconnected every ``reset_every`` calls so its context stays tiny. It only exists when
    ``routing.classifier: fast_model``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        sdk: ModuleType | None = None,
        reset_every: int = 20,
    ) -> None:
        self.settings = settings
        self.model = settings.brain.fast_model
        self.reset_every = reset_every
        self._sdk = sdk
        self._requests: asyncio.Queue[_Ask | None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._state = _ClassifierState()

    def _load_sdk(self) -> ModuleType:
        if self._sdk is None:
            import claude_agent_sdk

            self._sdk = claude_agent_sdk
        return self._sdk

    def _options(self) -> Any:
        from spark.brain.mcp_config import env_for_sdk

        sdk = self._load_sdk()
        kwargs: dict[str, Any] = {
            "system_prompt": CLASSIFIER_PROMPT,
            "model": self.model,
            "tools": [],
            "setting_sources": [],
            "strict_mcp_config": True,
            "max_turns": 1,
            "env": env_for_sdk(self.settings, claudeai=False),
        }
        if self.settings.brain.cli_path:
            kwargs["cli_path"] = self.settings.brain.cli_path
        return sdk.ClaudeAgentOptions(**kwargs)

    def _ensure_worker(self) -> asyncio.Queue[_Ask | None]:
        if self._worker is None or self._worker.done():
            self._requests = asyncio.Queue()
            self._worker = asyncio.create_task(self._run(), name="spark-route-classifier")
        assert self._requests is not None
        return self._requests

    async def warmup(self) -> None:
        """Connect ahead of the first turn (the CLI spawn takes seconds)."""
        fut: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._ensure_worker().put_nowait(_Ask("", fut))
        await fut

    async def classify(self, text: str) -> str | None:
        fut: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._ensure_worker().put_nowait(_Ask(text, fut))
        return await fut

    async def aclose(self) -> None:
        worker = self._worker
        if worker is None or worker.done():
            return
        assert self._requests is not None
        self._requests.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=10.0)
        except TimeoutError:
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker

    async def _run(self) -> None:
        assert self._requests is not None
        requests = self._requests
        try:
            while True:
                ask = await requests.get()
                if ask is None:
                    return
                try:
                    label = await self._handle(ask.text)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("router: classifier call failed: %s", exc)
                    await self._disconnect()
                    if not ask.future.done():
                        ask.future.set_exception(exc)
                    continue
                if not ask.future.done():  # the caller may have timed out already
                    ask.future.set_result(label)
        finally:
            await self._disconnect()
            while not requests.empty():
                left = requests.get_nowait()
                if left is not None and not left.future.done():
                    left.future.cancel()

    async def _handle(self, text: str) -> str | None:
        state = self._state
        if state.client is not None and state.calls >= self.reset_every:
            await self._disconnect()
        if state.client is None:
            client = self._load_sdk().ClaudeSDKClient(options=self._options())
            async with asyncio.timeout(self.settings.brain.connect_timeout_s):
                await client.connect()
            state.client = client
            state.calls = 0
        if not text:
            return None  # warm-up only
        sdk = self._load_sdk()
        client = state.client
        state.calls += 1
        await client.query(f"Request: {text}")
        chunks: list[str] = []
        async for msg in client.receive_response():
            if isinstance(msg, sdk.AssistantMessage):
                chunks.extend(b.text for b in msg.content if isinstance(b, sdk.TextBlock))
        match = _LABEL.search(" ".join(chunks).lower())
        return match.group(1) if match else None

    async def _disconnect(self) -> None:
        client, self._state.client = self._state.client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as exc:
                log.debug("router: classifier disconnect failed: %s", exc)
