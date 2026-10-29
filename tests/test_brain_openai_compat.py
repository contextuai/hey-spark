"""OpenAICompatBrain offline (httpx.MockTransport + fake MCP clients) plus opt-in live tests.

Offline tests cover: streaming text, tool calls split across chunks and in parallel, the shared
confirmation gate (yes / no / destructive / unknown), the step budget, provider errors as spoken
lines, barge-in, the researcher's nested loop, escalation, resets, connector health, and the
registry. Live tests (``-m live``) need Ollama with a model, or OPENAI_API_KEY."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from spark import registry
from spark.app import failure_line
from spark.brain.confirm import DENY_REASONS, REASON_DECLINED, REASON_STEP_BUDGET
from spark.brain.core import REASON_ESCALATING
from spark.brain.openai_chat import ProviderError
from spark.brain.openai_compat import BUDGET_NOTE, CLAUDEAI_ONLY, OpenAICompatBrain
from spark.brain.prompts import FRESH_START_ACK, MARKDOWN_CHARS
from spark.config import Settings
from spark.interfaces import BackendUnavailable
from tests.fakes import FakeClock, FakeConfirmer
from tests.openai_fakes import (
    GITHUB_TOOLS,
    KEY_ENV,
    FakeMcpClient,
    HangingStream,
    RecActivity,
    Script,
    call,
    factory,
    finish_chunk,
    github_setup,
    reply,
    sse,
    text_chunk,
    tool_chunk,
)


def settings(**brain) -> Settings:
    oc = {"provider": "openai", "fast_model": "fast-m", "deep_model": "deep-m"}
    oc.update(brain.pop("openai_compat", {}))
    cfg = {"backend": "openai_compat", "mode": "fast", "workers": False, "openai_compat": oc}
    cfg.update(brain)
    return Settings(brain=cfg, routing={"ack_after_s": 0})


def make(
    script: Script,
    s: Settings | None = None,
    *,
    setup=None,
    gh: FakeMcpClient | None = None,
    **kw,
) -> OpenAICompatBrain:
    gh = gh or FakeMcpClient(GITHUB_TOOLS, results={"list_issues": "Two open issues: A and B."})
    return OpenAICompatBrain(
        s or settings(),
        transport=script.transport(),
        environ=KEY_ENV,
        mcp=setup if setup is not None else github_setup(),
        mcp_client_factory=factory({"github": gh}),
        clock=FakeClock(),
        **kw,
    )


async def say(brain: OpenAICompatBrain, text: str) -> str:
    return "".join([d async for d in brain.respond(text)])


def tool_messages(body: dict) -> list[dict]:
    return [m for m in body["messages"] if m["role"] == "tool"]


# --- streaming --------------------------------------------------------------------------------


async def test_streams_text_deltas_with_the_voice_prompt_and_briefing():
    script = Script(reply("Hello", " there."))
    brain = make(script, context_provider=lambda: "BRIEFING: it is Monday.")
    deltas = [d async for d in brain.respond("hi")]
    assert deltas == ["Hello", " there."]
    body = script.bodies[0]
    system = body["messages"][0]
    assert system["role"] == "system"
    assert "BRIEFING: it is Monday." in system["content"]
    assert body["messages"][-1] == {"role": "user", "content": "hi"}
    assert body["model"] == "fast-m"
    offered = {t["function"]["name"] for t in body["tools"]}
    assert offered == {"mcp__github__list_issues", "mcp__github__create_issue"}  # policy only
    assert brain.last_turn.served_model == "served"
    await brain.close()


async def test_history_carries_across_turns_and_new_topic_clears_it():
    script = Script(reply("First."), reply("Second."))
    brain = make(script)
    await say(brain, "one")
    await say(brain, "two")
    roles = [m["role"] for m in script.bodies[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert await say(brain, "Spark, new topic") == FRESH_START_ACK
    assert brain._messages == []
    await brain.close()


async def test_tool_call_split_across_chunks_runs_and_is_logged():
    gh = FakeMcpClient(GITHUB_TOOLS, results={"list_issues": "Two open issues."})
    script = Script(
        call("mcp__github__list_issues", {"state": "open"}, split=5), reply("You have two.")
    )
    activity = RecActivity()
    brain = make(script, gh=gh, activity=activity)
    assert await say(brain, "any issues?") == "You have two."
    assert gh.calls == [("list_issues", {"state": "open"})]
    second = script.bodies[1]["messages"]
    assistant = second[-2]
    assert assistant["tool_calls"][0]["function"]["name"] == "mcp__github__list_issues"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "Two open issues."}
    assert activity.kinds("tool_call")[0]["tool"] == "mcp__github__list_issues"
    result = activity.kinds("tool_result")[0]
    assert result["tool"] == "mcp__github__list_issues" and not result["is_error"]
    assert activity.kinds("turn_end")[0]["num_turns"] == 2
    await brain.close()


async def test_parallel_tool_calls_each_get_a_result_in_order():
    both = sse(
        text_chunk("Checking both."),
        tool_chunk(0, id="a", name="mcp__github__list_issues", args='{"state":'),
        tool_chunk(1, id="b", name="mcp__github__list_issues", args='{"state": "closed"}'),
        tool_chunk(0, args=' "open"}'),
        finish_chunk("tool_calls"),
    )
    gh = FakeMcpClient(GITHUB_TOOLS)
    script = Script(both, reply("Done."))
    brain = make(script, gh=gh)
    out = [d async for d in brain.respond("open and closed issues")]
    assert out[0] == "Checking both." and "\n" in out  # sentence break before the tools
    assert gh.calls == [("list_issues", {"state": "open"}), ("list_issues", {"state": "closed"})]
    assert [m["tool_call_id"] for m in tool_messages(script.bodies[1])] == ["a", "b"]
    await brain.close()


# --- the gate --------------------------------------------------------------------------------


@pytest.mark.parametrize("answer", [True, False])
async def test_side_effecting_tool_needs_a_spoken_yes(answer):
    gh = FakeMcpClient(GITHUB_TOOLS)
    confirmer = FakeConfirmer([answer])
    script = Script(call("mcp__github__create_issue", {"title": "Fix login"}), reply("Okay."))
    brain = make(script, settings(mode="deep"), gh=gh, confirmer=confirmer)
    await say(brain, "create an issue called fix login")
    assert len(confirmer.asked) == 1 and "Fix login" in confirmer.asked[0]
    [msg] = tool_messages(script.bodies[1])
    if answer:
        assert gh.calls == [("create_issue", {"title": "Fix login"})]
        assert msg["content"] == "create_issue done"
    else:
        assert gh.calls == []
        assert msg["content"] == REASON_DECLINED
    await brain.close()


async def test_destructive_and_unknown_tools_are_denied_without_running():
    gh = FakeMcpClient(GITHUB_TOOLS)
    confirmer = FakeConfirmer([True, True])
    two = sse(
        tool_chunk(0, id="a", name="mcp__github__delete_repository", args="{}"),
        tool_chunk(1, id="b", name="mcp__github__not_listed", args="{}"),
        finish_chunk("tool_calls"),
    )
    script = Script(two, reply("I can't."))
    brain = make(script, settings(mode="deep"), gh=gh, confirmer=confirmer)
    await say(brain, "delete the repo")
    assert gh.calls == [] and confirmer.asked == []
    contents = [m["content"] for m in tool_messages(script.bodies[1])]
    assert contents == [DENY_REASONS["destructive"], DENY_REASONS["unknown"]]
    await brain.close()


async def test_bad_json_arguments_are_returned_to_the_model():
    script = Script(call("mcp__github__list_issues", "{not json"), reply("Sorry."))
    gh = FakeMcpClient(GITHUB_TOOLS)
    brain = make(script, gh=gh)
    await say(brain, "issues")
    assert gh.calls == []
    assert "valid JSON" in tool_messages(script.bodies[1])[0]["content"]
    await brain.close()


async def test_step_budget_refuses_then_forces_an_answer_without_tools():
    s = settings(max_turns=2)  # fast mode: one tool call, then answer
    loop = [call("mcp__github__list_issues", {}, id=f"c{i}") for i in range(4)]
    script = Script(*loop, reply("Here is what I found."))
    gh = FakeMcpClient(GITHUB_TOOLS)
    brain = make(script, s, gh=gh)
    assert await say(brain, "dig") == "Here is what I found."
    assert len(gh.calls) == 1
    assert tool_messages(script.bodies[2])[-1]["content"] == REASON_STEP_BUDGET
    final = script.bodies[-1]
    assert "tools" not in final and final["messages"][-1]["content"] == BUDGET_NOTE
    await brain.close()


# --- errors and barge-in -----------------------------------------------------------------------


async def test_401_raises_a_spoken_provider_error():
    script = Script(httpx.Response(401, json={"error": {"message": "bad key"}}))
    brain = make(script)
    with pytest.raises(ProviderError) as info:
        await say(brain, "hi")
    assert "API key" in failure_line(info.value)
    await brain.close()


async def test_ollama_down_says_is_it_running():
    s = settings(openai_compat={"provider": "ollama"})
    brain = make(Script(httpx.ConnectError("refused")), s)
    with pytest.raises(ProviderError) as info:
        await say(brain, "hi")
    assert failure_line(info.value) == "I can't reach Ollama on this computer. Is it running?"
    await brain.close()


async def test_barge_in_closes_the_stream_and_keeps_the_partial_reply():
    stream = HangingStream(sse(text_chunk("Well, let me"), done=False))
    script = Script(httpx.Response(200, stream=stream), reply("Next."))
    brain = make(script)
    gen = brain.respond("tell me a long story")
    assert await gen.__anext__() == "Well, let me"
    await asyncio.wait_for(brain.interrupt(), 2)
    await asyncio.wait_for(stream.closed.wait(), 2)
    await gen.aclose()
    assert brain.last_turn.interrupted
    assert brain._messages[-1] == {"role": "assistant", "content": "Well, let me"}
    assert await say(brain, "never mind") == "Next."
    await brain.close()


# --- researcher, escalation, status ------------------------------------------------------------


async def test_researcher_runs_a_nested_loop_on_the_worker_model():
    s = settings(workers=True, mode="deep", openai_compat={"worker_model": "worker-m"})
    gh = FakeMcpClient(GITHUB_TOOLS, results={"list_issues": "Issue A, issue B."})
    script = Script(
        call("research", {"brief": "count open issues"}, id="r1"),
        call("mcp__github__list_issues", {"state": "open"}, id="w1"),
        call("mcp__github__create_issue", {"title": "x"}, id="w2"),
        reply("Two open issues: A and B."),
        reply("You have two open issues."),
    )
    activity = RecActivity()
    confirmer = FakeConfirmer([True])
    brain = make(script, s, gh=gh, activity=activity, confirmer=confirmer)
    assert "research" in {t["function"]["name"] for t in script_tools(brain)}
    assert await say(brain, "how many open issues") == "You have two open issues."
    models = [b["model"] for b in script.bodies]
    assert models == ["deep-m", "worker-m", "worker-m", "worker-m", "deep-m"]
    worker_tools = {t["function"]["name"] for t in script.bodies[1]["tools"]}
    assert worker_tools == {"mcp__github__list_issues"}  # read-only only
    assert gh.calls == [("list_issues", {"state": "open"})]  # create_issue never ran
    assert confirmer.asked == []
    report = tool_messages(script.bodies[-1])[-1]
    assert report["content"] == "Two open issues: A and B."
    agents = [e.get("agent") for e in activity.kinds("tool_call")]
    assert agents == [None, "researcher", "researcher"]
    await brain.close()


def script_tools(brain: OpenAICompatBrain) -> list[dict]:
    return brain._offered_tools()[0]


async def test_fast_turn_reaching_a_deep_connector_is_rerun_deep():
    s = settings(mode="auto")
    s.routing.max_fast_words = 50
    script = Script(
        call("mcp__github__list_issues", {}),
        call("mcp__github__list_issues", {}),
        reply("Deep answer."),
    )
    gh = FakeMcpClient(GITHUB_TOOLS)
    brain = make(script, s, setup=github_setup(deep=True), gh=gh)
    await brain.manager.start(wait_s=2)
    out = await say(brain, "what's up")
    assert out == "Deep answer."
    assert [b["model"] for b in script.bodies] == ["fast-m", "deep-m", "deep-m"]
    assert len(gh.calls) == 1  # the fast attempt never ran the tool
    assert all(REASON_ESCALATING not in json.dumps(m) for m in brain._messages)
    await brain.close()


async def test_unreachable_connector_and_claudeai_status():
    setup = github_setup()
    setup.claudeai = ["claude_ai_Gmail"]
    seen: list[dict] = []
    gh = FakeMcpClient(GITHUB_TOOLS, fail_connect=ConnectionRefusedError("connection refused"))
    script = Script(reply("GitHub is down."))
    brain = make(script, setup=setup, gh=gh)
    brain.on_mcp_status = seen.append
    assert brain.claudeai_skipped == ["claude_ai_Gmail"]
    await brain.warmup()
    assert brain.mcp_status["github"]["status"] == "failed"
    assert brain.mcp_status["claude_ai_Gmail"] == {"status": "failed", "error": CLAUDEAI_ONLY}
    assert seen and seen[-1]["github"]["status"] == "failed"
    await say(brain, "my issues?")
    user = script.bodies[0]["messages"][-1]["content"]
    assert "unreachable right now" in user and "github" in user
    await brain.close()


async def test_warmup_lists_models_for_the_window():
    script = Script(models=["fast-m", "other"])
    brain = make(script)
    await brain.warmup()
    assert brain.available_models == ["fast-m", "other"]
    await brain.close()


def _ollama_settings(**brain) -> Settings:
    return settings(openai_compat={"provider": "ollama"}, **brain)


async def _prime_done(brain: OpenAICompatBrain) -> None:
    if brain._prime_task is not None:
        await asyncio.wait_for(brain._prime_task, timeout=2)


async def test_local_server_is_primed_at_warmup():
    script = Script(reply("x"), reply("Paris."))
    brain = make(script, _ollama_settings())
    await brain.warmup()
    await _prime_done(brain)
    (prime,) = script.bodies
    assert prime["model"] == "fast-m" and prime["max_tokens"] == 1
    assert prime["messages"][0] == {"role": "system", "content": brain._system}
    assert prime["tools"]  # the same tool list the turns send, so the prompt cache matches
    assert await say(brain, "capital of France?") == "Paris."
    turn = script.bodies[1]
    assert turn["messages"][0] == prime["messages"][0] and turn["tools"] == prime["tools"]
    assert brain._messages[0] == {"role": "user", "content": "capital of France?"}  # not "Hi"
    await brain.close()


async def test_reset_primes_the_new_prompt():
    script = Script(reply("x"), reply("y"))
    brain = make(script, _ollama_settings())
    await brain.warmup()
    await _prime_done(brain)
    await brain.reset()
    await _prime_done(brain)
    assert len(script.bodies) == 2 and script.bodies[1]["max_tokens"] == 1
    await brain.close()


@pytest.mark.parametrize("s", [settings(), _ollama_settings(prime_on_connect=False)])
async def test_cloud_or_opted_out_is_not_primed(s):
    script = Script()
    brain = make(script, s)
    await brain.warmup()
    await _prime_done(brain)
    assert script.bodies == []
    await brain.close()


async def test_failed_prime_does_not_break_turns():
    script = Script(httpx.ConnectError("refused"), reply("Still here."))
    brain = make(script, _ollama_settings())
    await brain.warmup()
    await _prime_done(brain)
    assert await say(brain, "hello?") == "Still here."
    await brain.close()


async def test_in_process_spark_tools_and_timers():
    from spark.tools.timers import TimerService

    timers = TimerService(on_fire=lambda t: None)
    script = Script(call("mcp__spark__set_timer", {"duration": "1 minute"}), reply("Timer set."))
    brain = make(script, timers=timers)
    assert await say(brain, "set a one minute timer") == "Timer set."
    assert len(timers) == 1
    assert "1 minute" in tool_messages(script.bodies[1])[0]["content"]
    timers.cancel_all()
    await brain.close()


# --- registry and construction ----------------------------------------------------------------


def test_registry_selects_the_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = settings()
    brain = registry.build("brain", s)
    assert type(brain) is OpenAICompatBrain
    assert brain.model == "fast-m"


def test_missing_key_and_missing_model_are_backend_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable) as info:
        OpenAICompatBrain(settings(), environ={})
    assert "OPENAI_API_KEY" in info.value.reason and ".env" in info.value.fix
    with pytest.raises(BackendUnavailable) as info:
        OpenAICompatBrain(
            settings(
                openai_compat={"provider": "lmstudio", "fast_model": None, "deep_model": None}
            ),
            environ={},
        )
    assert "fast_model" in info.value.reason


def test_model_switching_and_mode():
    brain = make(Script())
    brain.set_models("f2", "d2", "w2")
    assert (brain.oc.fast_model, brain.oc.deep_model, brain.worker_model) == ("f2", "d2", "w2")
    assert brain.router.fast_model == "f2"
    brain.set_mode("deep")
    assert brain.mode == "deep"


# --- live --------------------------------------------------------------------------------------


def _ollama_model() -> str | None:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None
    return models[0] if models else None


async def _live_capital(s: Settings) -> None:
    brain = OpenAICompatBrain(s, mcp=github_setup().__class__())
    try:
        out = await asyncio.wait_for(say(brain, "What's the capital of Japan?"), 120)
    finally:
        await brain.close()
    assert "Tokyo" in out
    assert not any(c in out for c in MARKDOWN_CHARS)


@pytest.mark.live
async def test_live_ollama_capital():
    model = _ollama_model()
    if model is None:
        pytest.skip("Ollama not running on localhost:11434 or no model pulled")
    s = settings(openai_compat={"provider": "ollama", "fast_model": model, "deep_model": model})
    await _live_capital(s)


@pytest.mark.live
async def test_live_openai_capital():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    s = settings(openai_compat={"provider": "openai", "fast_model": None, "deep_model": None})
    await _live_capital(s)
