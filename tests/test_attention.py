"""'Is this for me?' while awake (Phase 5, P5.4): rules, the optional model check, and the app."""

from __future__ import annotations

import asyncio

import pytest

from spark.attention import Addressee, rules_verdict
from tests.fakes import make_settings

NAMES = ("spark", "sparks", "sparky")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Spark, what's next?", "name"),
        ("I think sparky... no, Spark knows", "name"),
        ("and last month?", "continuation"),
        ("What about the Denver stores?", "continuation"),
        ("How many orders shipped?", "question"),
        ("is the build green", "question"),
        ("the orders for Phoenix?", "question"),
        ("check my calendar for tomorrow", "request"),
        ("Please send it to Sam", "request"),
        ("can you pull the numbers again", "request"),
        ("I need the sales report for June", "request"),
    ],
)
def test_requests_are_addressed(text, reason):
    v = rules_verdict(text, names=NAMES)
    assert v.addressed and v.reason == reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("okay", "filler"),
        ("hmm yeah right", "filler"),
        ("Thank you.", "filler"),
        ("I told him the meeting moved to Thursday", "no request"),
        ("Lunch was good today", "no request"),
        ("...", "empty"),
    ],
)
def test_chatter_is_ignored(text, reason):
    v = rules_verdict(text, names=NAMES)
    assert not v.addressed and v.reason == reason


def test_an_answer_to_sparks_question_counts_only_soon_after():
    asked = "There are 42 open orders. Want the breakdown by city?"
    assert rules_verdict("yes", names=NAMES, last_reply=asked, since_reply_s=4).addressed
    assert not rules_verdict("yes", names=NAMES, last_reply=asked, since_reply_s=90).addressed
    statement = "There are 42 open orders."
    assert not rules_verdict("yes", names=NAMES, last_reply=statement, since_reply_s=4).addressed


class FakeSide:
    def __init__(self, answer: str = "yes", delay: float = 0.0, error: bool = False):
        self.answer, self.delay, self.error = answer, delay, error
        self.prompts: list[str] = []
        self.closed = False

    async def warmup(self) -> None:
        pass

    async def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise RuntimeError("cli died")
        return self.answer

    async def aclose(self) -> None:
        self.closed = True


def _model_addressee(side: FakeSide, timeout: float = 1.0) -> Addressee:
    s = make_settings(attention={"addressee": "fast_model", "addressee_timeout_s": timeout})
    return Addressee(s, side=side)


async def test_model_mode_asks_with_the_last_reply():
    side = FakeSide("No.")
    a = _model_addressee(side)
    v = await a.decide("is the build green", last_reply="Your PR merged.", since_reply_s=5)
    assert not v.addressed and v.by == "model" and v.reason == "model said no"
    assert "Your PR merged." in side.prompts[0] and "is the build green" in side.prompts[0]


async def test_model_mode_skips_the_model_when_the_rules_are_certain():
    side = FakeSide("no")
    a = _model_addressee(side)
    assert (await a.decide("Spark, what's next")).addressed
    assert not (await a.decide("okay")).addressed
    assert side.prompts == []


@pytest.mark.parametrize("side", [FakeSide(delay=5.0), FakeSide(error=True), FakeSide("maybe")])
async def test_model_mode_falls_back_to_the_rules(side):
    a = _model_addressee(side, timeout=0.05)
    v = await a.decide("how many orders shipped")
    assert v.addressed and v.by == "rules" and v.reason == "question"


async def test_off_mode_answers_everything():
    a = Addressee(make_settings(attention={"addressee": "off"}))
    assert (await a.decide("lunch was good")).addressed


# --- SideModel against the fake SDK ------------------------------------------------------------

from tests.test_brain_claude_agent import FakeClient, fake_sdk, streamed  # noqa: E402, F401


async def test_side_model_asks_on_its_own_tool_less_client(fake_sdk):  # noqa: F811
    from spark.brain.side import SideModel

    FakeClient.script = staticmethod(lambda text, client: streamed("y", "es"))
    side = SideModel(make_settings(), "Answer yes or no.", sdk=fake_sdk, reset_every=2)
    try:
        await side.warmup()
        assert await side.ask("q1") == "yes"
        assert await side.ask("q2") == "yes"
        assert await side.ask("q3") == "yes"  # reconnected after 2 questions
        assert [c.queries for c in FakeClient.instances] == [["q1", "q2"], ["q3"]]
        opts = FakeClient.instances[0].options
        assert opts.system_prompt == "Answer yes or no." and opts.tools == []
        assert opts.max_turns == 1 and opts.strict_mcp_config is True
    finally:
        await side.aclose()
    assert FakeClient.instances[-1].disconnected
