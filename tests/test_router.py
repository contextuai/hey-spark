"""Fast/deep routing rules, the optional classifier and its fallbacks (spark/brain/router.py)."""

from __future__ import annotations

import asyncio
import logging
import types
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from spark.brain.router import (
    DEEP,
    FAST,
    FastModelClassifier,
    Router,
    count_imperative_clauses,
    normalize_request,
)
from spark.config import RoutingConfig, Settings

FAST_MODEL = "claude-haiku-4-5-20251001"
DEEP_MODEL = "claude-sonnet-5"


def make_router(classifier: Any = None, **routing: Any) -> Router:
    return Router(
        RoutingConfig(**routing),
        fast_model=FAST_MODEL,
        deep_model=DEEP_MODEL,
        classifier=classifier,
    )


# --- rules ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hey Spark, let's start a new project called weather-bot",
        "Summarize this week and draft a plan for next week",
        "draft a plan for the launch",
        "Compare the two job offers",
        "Please analyze last month's spending",
        "Write a short poem about autumn",
        "Can you refactor the audio module",
        "check my email and draft replies to anything urgent",
        "Look up the weather, then send Sam a message about the picnic",
        "Let's brainstorm names for the band",
        "Walk me through everything I need to know about moving to a new city, including the "
        "paperwork, the utilities, the schools, the doctors and anything else I might forget",
    ],
)
def test_deep_requests(text):
    decision = make_router().route(text)
    assert decision.route == DEEP, decision
    assert decision.model == DEEP_MODEL


@pytest.mark.parametrize(
    "text",
    [
        "what time is it",
        "Hey Spark, what time is it?",
        "set a timer for ten minutes",
        "cancel the pasta timer",
        "what are you working on",
        "What's on my calendar tomorrow?",
        "Any unread email from Sam?",
        "How many open PRs on the spark repo?",
        "What's the capital of Japan?",
        "hello",
        "thanks",
        "tell me a joke",
        "",
    ],
)
def test_fast_requests(text):
    decision = make_router().route(text)
    assert decision.route == FAST, decision
    assert decision.model == FAST_MODEL


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Hey Spark, let's start a new project called weather-bot", "phrase 'new project'"),
        ("check my email and draft replies", "2 imperative clauses"),
        ("what time is it", "keyword 'what time'"),
        ("write a haiku", "verb 'write'"),
        ("what's the capital of Japan", "default"),
    ],
)
def test_reasons_name_the_rule(text, reason):
    assert make_router().route(text).reason == reason


def test_long_requests_go_deep():
    text = " ".join(["word"] * 26)
    assert make_router().route(text).reason == "26 words"
    assert make_router(max_fast_words=30).route(text).route == FAST


def test_keyword_lists_come_from_config():
    router = make_router(deep_phrases=["grocery list"], deep_verbs=[], fast_keywords=["poem"])
    assert router.route("make me a grocery list").route == DEEP
    assert router.route("write a poem").route == FAST  # "write" no longer a deep verb


def test_routing_disabled_is_always_fast():
    decision = make_router(enabled=False).route("draft a plan for the launch")
    assert (decision.route, decision.reason) == (FAST, "routing disabled")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        # The fast route can't use side-effecting tools, so every side-effect request is deep.
        ("let's start a new project called weather-bot", "phrase 'new project'"),
        ("Create a repo called weather-bot", "verb 'create'"),
        ("send Sam the meeting notes", "verb 'send'"),
        ("draft an email to the landlord", "verb 'draft'"),
        ("delete the old branch", "verb 'delete'"),
        ("set up a repo for the band", "verb 'set up'"),
        ("let's set up the new laptop", "verb 'set up'"),
        ("schedule a meeting with Sam on Friday", "verb 'schedule a'"),
        ("book a table for two", "verb 'book a'"),
        ("email the team the agenda", "verb 'email the'"),
        ("reply to Sam's last message", "verb 'reply'"),
        ("post an update in the team channel", "verb 'post an'"),
        ("open a pull request for the fix", "verb 'open a'"),
        ("add milk to my shopping list", "verb 'add'"),
        ("I want to create an issue for the crash", "verb 'create'"),
        ("I'd like to cancel my dentist appointment", "verb 'cancel'"),
        ("could you update the README", "verb 'update'"),
    ],
)
def test_side_effect_requests_route_deep(text, reason):
    decision = make_router().route(text)
    assert (decision.route, decision.reason) == (DEEP, reason)


@pytest.mark.parametrize(
    "text",
    [
        "set a timer for five minutes",
        "start a timer for the pasta",
        "cancel the pasta timer",
        "set up a timer for ten minutes",
        "list my timers",
        "what time is it",
        "any open pull requests on the spark repo?",
        "did Sam send the notes?",
        "open PRs on the spark repo?",
        "email from Sam today?",
        "schedule for tomorrow?",
        "start time of the standup?",
        "book recommendations for the trip?",
    ],
)
def test_timers_and_lookups_stay_fast(text):
    assert make_router().route(text).route == FAST


def test_multi_word_deep_verbs_match_the_clause_start_only():
    router = make_router(deep_verbs=["set up"], fast_keywords=[])
    assert router.route("set up the printer").reason == "verb 'set up'"
    assert router.route("set upstairs lights").route == FAST  # whole words only
    assert router.route("the set up is done").route == FAST  # not at the clause start


async def test_escalate_turns_a_fast_decision_deep_and_logs_it(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.router")
    router = make_router()
    fast = await router.decide("okay")
    deep = router.escalate(fast, "retry after a fast-route tool refusal")
    assert (deep.route, deep.model) == (DEEP, DEEP_MODEL)
    assert deep.reason == "retry after a fast-route tool refusal; rules: default"
    assert router.decisions[-1] is deep
    assert "route: deep (retry after a fast-route tool refusal; rules: default)" in caplog.text
    assert router.escalate(deep, "again") is deep  # already deep: unchanged
    off = make_router(enabled=False)
    decision = off.route("okay")
    assert off.escalate(decision, "x") is decision  # routing disabled: never deep


def test_normalize_and_count_clauses():
    assert normalize_request("Hey Spark, What’s  UP") == "what's up"
    assert normalize_request("Spark: set a timer") == "set a timer"
    assert normalize_request("chipmunks are cute") == "chipmunks are cute"
    assert count_imperative_clauses("please check my mail and then send the report") == 2
    assert count_imperative_clauses("what's the weather and the time") == 0
    assert count_imperative_clauses("could you draft a note", ["draft"]) == 1


def test_decision_meta():
    decision = make_router().route("what time is it")
    assert decision.as_meta() == {
        "route": FAST,
        "route_model": FAST_MODEL,
        "route_reason": "keyword 'what time'",
    }


def test_from_settings_uses_brain_models_and_no_classifier_by_default():
    s = Settings(brain={"fast_model": "f-model", "deep_model": "d-model"})
    router = Router.from_settings(s)
    assert router.classifier is None
    assert router.route("draft a plan").model == "d-model"
    assert router.route("hello").model == "f-model"


def test_from_settings_builds_the_fast_model_classifier_when_configured():
    s = Settings(routing={"classifier": "fast_model"})
    router = Router.from_settings(s)
    assert isinstance(router.classifier, FastModelClassifier)
    assert router.classifier.model == s.brain.fast_model


def test_config_rejects_an_unknown_classifier():
    with pytest.raises(ValueError):
        RoutingConfig(classifier="gpt")


# --- decide(): logging and the classifier -----------------------------------------------------


async def test_decide_logs_every_decision(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.router")
    router = make_router()
    await router.decide("what time is it")
    await router.decide("Hey Spark, let's start a new project called weather-bot")
    assert "route: fast (keyword 'what time')" in caplog.text
    assert "route: deep (phrase 'new project')" in caplog.text
    assert [d.route for d in router.decisions] == [FAST, DEEP]


class FakeClassifier:
    def __init__(self, label: str | None = None, delay: float = 0.0, error: bool = False):
        self.label = label
        self.delay = delay
        self.error = error
        self.calls: list[str] = []
        self.closed = False
        self.warmed = False

    async def classify(self, text: str) -> str | None:
        self.calls.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise RuntimeError("boom")
        return self.label

    async def warmup(self) -> None:
        self.warmed = True

    async def aclose(self) -> None:
        self.closed = True


async def test_classifier_decides_when_it_answers():
    clf = FakeClassifier("deep")
    decision = await make_router(clf).decide("hello there")  # rules would say fast
    assert (decision.route, decision.model, decision.reason) == (DEEP, DEEP_MODEL, "classifier")
    assert clf.calls == ["hello there"]


async def test_classifier_timeout_falls_back_to_rules(caplog):
    caplog.set_level(logging.INFO, logger="spark.brain.router")
    clf = FakeClassifier("deep", delay=10.0)
    router = make_router(clf, classifier_timeout_s=0.05)
    loop = asyncio.get_running_loop()
    start = loop.time()
    decision = await router.decide("what time is it")
    assert loop.time() - start < 1.0  # bounded by classifier_timeout_s
    assert decision.route == FAST
    assert decision.reason == "classifier timeout; rules: keyword 'what time'"
    assert "route: fast (classifier timeout; rules: keyword 'what time')" in caplog.text


async def test_classifier_error_or_unclear_answer_falls_back_to_rules():
    decision = await make_router(FakeClassifier(error=True)).decide("draft a plan")
    assert (decision.route, decision.reason) == (
        DEEP,
        "classifier error; rules: phrase 'draft a plan'",
    )
    decision = await make_router(FakeClassifier("maybe")).decide("hello")
    assert (decision.route, decision.reason) == (FAST, "classifier unsure; rules: keyword 'hello'")


async def test_router_warms_and_closes_the_classifier():
    clf = FakeClassifier("fast")
    router = make_router(clf)
    await router.warmup()
    await router.aclose()
    assert clf.warmed and clf.closed


async def test_rules_only_router_makes_no_calls():
    router = make_router()
    await router.warmup()
    decision = await router.decide("draft a plan")
    await router.aclose()
    assert decision.reason == "phrase 'draft a plan'"


# --- FastModelClassifier on a fake SDK ----------------------------------------------------------


@dataclass
class _TextBlock:
    text: str


@dataclass
class _AssistantMessage:
    content: list[Any]
    model: str


@dataclass
class _ResultMessage:
    is_error: bool = False


class _Options:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Client:
    instances: ClassVar[list[_Client]] = []
    reply = "deep"

    def __init__(self, options: _Options) -> None:
        self.options = options
        self.owner: asyncio.Task[Any] | None = None
        self.queries: list[str] = []
        self.disconnected = False
        _Client.instances.append(self)

    async def connect(self) -> None:
        self.owner = asyncio.current_task()

    async def query(self, prompt: str) -> None:
        assert asyncio.current_task() is self.owner, "used outside the connecting task"
        self.queries.append(prompt)

    async def receive_response(self):
        assert asyncio.current_task() is self.owner
        yield _AssistantMessage([_TextBlock(_Client.reply)], "claude-haiku")
        yield _ResultMessage()

    async def disconnect(self) -> None:
        self.disconnected = True


def _fake_sdk() -> types.ModuleType:
    mod = types.ModuleType("claude_agent_sdk")
    mod.ClaudeSDKClient = _Client
    mod.ClaudeAgentOptions = _Options
    mod.AssistantMessage = _AssistantMessage
    mod.TextBlock = _TextBlock
    return mod


async def test_fast_model_classifier_asks_the_fast_model_on_its_own_client():
    _Client.instances = []
    s = Settings(routing={"classifier": "fast_model"})
    clf = FastModelClassifier(s, sdk=_fake_sdk(), reset_every=2)
    try:
        await clf.warmup()
        _Client.reply = "Deep."
        assert await clf.classify("draft a plan") == "deep"
        _Client.reply = "fast"
        assert await clf.classify("hello") == "fast"
        _Client.reply = "I am not sure"
        assert await clf.classify("hmm") is None  # third call: reconnected (reset_every=2)
    finally:
        await clf.aclose()
    first, second = _Client.instances
    opts = first.options.kwargs
    assert opts["model"] == s.brain.fast_model
    assert opts["tools"] == [] and opts["max_turns"] == 1 and opts["setting_sources"] == []
    assert "exactly one lowercase word" in opts["system_prompt"]
    assert first.queries == ["Request: draft a plan", "Request: hello"]
    assert first.disconnected and second.disconnected
    assert second.queries == ["Request: hmm"]


async def test_fast_model_classifier_in_a_router_falls_back_on_timeout():
    class SlowClient(_Client):
        async def connect(self) -> None:
            self.owner = asyncio.current_task()
            await asyncio.sleep(10)

    sdk = _fake_sdk()
    sdk.ClaudeSDKClient = SlowClient
    s = Settings(routing={"classifier": "fast_model", "classifier_timeout_s": 0.05})
    router = Router.from_settings(s, sdk=sdk)
    try:
        decision = await router.decide("Hey Spark, let's start a new project called weather-bot")
        assert decision.route == DEEP
        assert decision.reason.startswith("classifier timeout; rules:")
    finally:
        assert isinstance(router.classifier, FastModelClassifier)
        router.classifier._worker.cancel()  # the slow connect never finishes on its own
        await asyncio.gather(router.classifier._worker, return_exceptions=True)
