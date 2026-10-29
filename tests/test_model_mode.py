"""Choosing the model: auto, always fast, always deep (Phase 5, P5.7c)."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest

from spark.prefs import Prefs, apply_prefs, model_label
from tests.fakes import FakeClock, make_settings

# --- prefs and labels ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "label"),
    [
        ("claude-haiku-5-5", "Haiku 5.5"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5"),
        ("claude-sonnet-5", "Sonnet 5"),
        ("claude-sonnet-5-5", "Sonnet 5.5"),
        ("claude-opus-5-5", "Opus 5.5"),
        ("claude-fable-5-1", "Fable 5.1"),
        ("my-custom-model", "my-custom-model"),
    ],
)
def test_model_labels(model_id, label):
    assert model_label(model_id) == label


def test_prefs_survive_a_restart_and_override_config(tmp_path):
    path = tmp_path / "prefs.json"
    Prefs(path).update(model_mode="deep", deep_model="claude-opus-5-5", fast_model=None)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "model_mode": "deep",
        "deep_model": "claude-opus-5-5",
    }
    s = make_settings()
    apply_prefs(s, Prefs(path))
    assert s.brain.mode == "deep" and s.brain.deep_model == "claude-opus-5-5"
    assert s.brain.fast_model == "claude-haiku-5-5"


def test_bad_prefs_are_ignored(tmp_path, caplog):
    path = tmp_path / "prefs.json"
    path.write_text("{oops", encoding="utf-8")
    s = make_settings()
    apply_prefs(s, Prefs(path))
    assert s.brain.mode == "auto" and "unreadable" in caplog.text
    path.write_text(json.dumps({"model_mode": "turbo", "deep_model": " "}), encoding="utf-8")
    apply_prefs(s, Prefs(path))
    assert s.brain.mode == "auto" and s.brain.deep_model == "claude-opus-5-5"


# --- the brain ----------------------------------------------------------------------------------


@pytest.fixture
def sdk(monkeypatch):
    from tests.test_brain_claude_agent import FakeClient, make_fake_sdk, streamed

    FakeClient.instances = []
    FakeClient.script = staticmethod(lambda text, client: streamed("Ok."))
    FakeClient.connect_error = None
    FakeClient.connect_delay = FakeClient.step_delay = 0.0
    FakeClient.set_model_error = None
    FakeClient.mcp_status = None
    mod = make_fake_sdk()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def _brain(**kw):
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.tools.timers import TimerService
    from tests.test_brain_claude_agent import ManualClock, settings

    return ClaudeAgentBrain(
        settings(), clock=ManualClock(), timers=TimerService(clock=FakeClock()), **kw
    )


async def _ask(brain, text: str) -> str:
    return "".join([d async for d in brain.respond(text)])


async def test_always_deep_and_always_fast(sdk, caplog):
    caplog.set_level(logging.INFO, logger="spark.brain")
    b = _brain()
    try:
        b.set_mode("deep")
        await _ask(b, "what time is it")  # a fast keyword, but the user picked deep
        assert b.last_turn.route == "deep" and b.last_turn.model == b.config.deep_model
        assert "route: deep (model mode: always deep)" in caplog.text
        b.set_mode("fast")
        await _ask(b, "Draft a plan for the product launch")  # a deep phrase
        assert b.last_turn.route == "fast"
        b.set_mode("auto")
        await _ask(b, "Draft a plan for the product launch")
        assert b.last_turn.route == "deep"
        with pytest.raises(ValueError):
            b.set_mode("turbo")
    finally:
        await b.close()


async def test_set_models_changes_the_model_used(sdk):
    from tests.test_brain_claude_agent import FakeClient

    b = _brain()
    try:
        b.set_models(deep="claude-opus-5-5")
        await _ask(b, "Draft a plan for the product launch")
        assert b.last_turn.model == "claude-opus-5-5"
        assert FakeClient.instances[0].models[-1] == "claude-opus-5-5"
        assert b.router.deep_model == "claude-opus-5-5"
    finally:
        await b.close()


async def test_a_fixed_mode_ignores_shortcut_routes_but_not_an_explicit_ask(sdk):
    b = _brain()
    try:
        b.set_mode("fast")
        b.next_route, b.next_route_reason = "deep", "shortcut 'Morning briefing'"
        await _ask(b, "good morning")
        assert b.last_turn.route == "fast"  # the user's mode wins over a shortcut
        b.next_route, b.next_route_reason, b.next_route_wins = "deep", "asked for deep", True
        await _ask(b, "check the open orders")
        assert b.last_turn.route == "deep"  # "use the deep model to ..." wins for one request
        await _ask(b, "and now?")
        assert b.last_turn.route == "fast"
    finally:
        await b.close()


async def test_always_fast_never_hands_over_or_limits_tools(sdk):
    from tests.test_brain_claude_agent import (
        LIST_PRS,
        START_PROJECT,
        _decision,
        _deep_github,
        hook_turns,
    )
    from tests.test_brain_claude_agent import (
        FakeClient as FC,
    )

    script, turns = hook_turns(LIST_PRS, START_PROJECT)
    FC.script = staticmethod(script)

    class Yes:
        async def ask(self, summary: str) -> bool:
            return True

    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.tools.timers import TimerService
    from tests.test_brain_claude_agent import ManualClock, settings

    b = ClaudeAgentBrain(
        settings(),
        clock=ManualClock(),
        mcp=_deep_github(),
        timers=TimerService(clock=FakeClock()),
        confirmer=Yes(),
    )
    b.tool_announce_wait_s = 0.1
    try:
        b.set_mode("fast")
        await _ask(b, "anything new for me?")
        assert len(FC.instances[0].queries) == 1  # no re-run on the deep model
        # read-only passes; the side-effecting one is asked (yes), not refused as "too big"
        assert [_decision(c.output) for c in turns[0]] == ["pass", "allow"]
        assert b.last_turn.route == "fast"
    finally:
        await b.close()


# --- the app: voice commands and "use the deep model to ..." -------------------------------------


class ModeBrain:
    def __new__(cls, **kw: Any):
        from spark.brain.mock import MockBrain

        brain = MockBrain(**kw)
        brain.mode = "auto"
        brain.next_route, brain.next_route_reason, brain.next_route_wins = None, "", False
        brain.models: list[tuple[str | None, str | None]] = []

        def set_mode(mode: str) -> None:
            brain.mode = mode

        def set_models(
            fast: str | None = None, deep: str | None = None, worker: str | None = None
        ) -> None:
            brain.models.append((fast, deep))

        brain.set_mode, brain.set_models = set_mode, set_models
        return brain


async def test_voice_mode_commands_switch_save_and_stay_awake(tmp_path):
    from tests.test_app import _awake_app, _awake_turn, _finish_turn, _utterance

    prefs = Prefs(tmp_path / "prefs.json")
    seen: list[str] = []
    brain = ModeBrain()
    app = _awake_app(
        tmp_path,
        ["what is two plus two", "use the deep model", "which model are you using", "back to auto"],
        brain=brain,
        prefs=prefs,
        on_model=lambda info: seen.append(info["mode"]),
    )
    await _awake_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    assert brain.mode == "deep" and prefs.get("model_mode") == "deep"
    assert (
        app.tts.spoken[-1] == "Okay, I'll use Opus 5.5 for everything until you say back to auto."
    )
    assert app.state.value == "follow_up"
    await _utterance(app)
    await _finish_turn(app)
    assert app.tts.spoken[-1] == "I'm using Opus 5.5 for everything."
    await _utterance(app)
    await _finish_turn(app)
    assert brain.mode == "auto"
    assert app.tts.spoken[-1].startswith("Okay, back to auto: Haiku 5.5 for quick questions")
    assert seen == ["deep", "auto"]
    assert brain.calls == ["what is two plus two"]  # none of the commands reached Claude
    await app.close()


async def test_use_the_deep_model_to_routes_one_request(tmp_path):
    from tests.fakes import FakeSTT
    from tests.test_app import _app, _finish_turn, _settings, _utterance, _wake

    brain = ModeBrain()
    routes: list[tuple[str | None, bool]] = []
    original = brain.respond

    def respond(text):
        routes.append((brain.next_route, brain.next_route_wins))
        return original(text)

    brain.respond = respond
    app = _app(
        tmp_path,
        stt=FakeSTT(transcripts=["Use the deep model to check how many orders ship today"]),
        brain=brain,
        settings=_settings(tmp_path),
    )
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    assert brain.calls == ["check how many orders ship today"]
    assert routes == [("deep", True)]
    await app.close()


async def test_text_mode_mode_commands(tmp_path):
    from tests.test_app import _app

    brain = ModeBrain()
    app = _app(tmp_path, brain=brain)
    assert (await app.text_turn("fast mode")).startswith("Okay, I'll use Haiku 5.5")
    assert brain.mode == "fast"
    await app.close()


# --- the window ---------------------------------------------------------------------------------


async def test_controller_saves_the_mode_before_and_after_start(tmp_path):
    from tests.test_desktop import _controller

    c, events, built = _controller(tmp_path)
    await c.aset_mode("deep")
    assert c.prefs.get("model_mode") == "deep" and c.settings.brain.mode == "deep"
    assert events[-1]["type"] == "model" and events[-1]["mode"] == "deep"
    await c.aset_models(None, "claude-opus-5-5")
    assert c.snapshot()["model"]["deep_label"] == "Opus 5.5"
    await c.astart()
    app = built.apps[0]
    await c.aset_mode("auto")
    assert app.settings.brain.mode == "auto" and events[-1]["mode"] == "auto"
    await c.aset_mode("warp")  # ignored
    assert events[-1]["mode"] == "auto"
    await c.aclose()


# --- P5.7h: the researcher's model ----------------------------------------------------------------


def test_researcher_model_prefs(tmp_path):
    from spark.prefs import model_info

    path = tmp_path / "prefs.json"
    Prefs(path).update(worker_model="claude-sonnet-5")
    s = make_settings()
    apply_prefs(s, Prefs(path))
    assert s.brain.worker_model == "claude-sonnet-5"
    assert model_info(s)["worker_label"] == "Sonnet 5"
    Prefs(path).update(worker_model="")  # back to "same as Fast"
    apply_prefs(s, Prefs(path))
    assert s.brain.worker_model is None
    info = model_info(s)
    assert info["worker_model"] == "" and info["worker_label"] == "Haiku 5.5"


async def test_researcher_model_applies_from_the_next_session(sdk):
    from spark.brain.claude_agent import ClaudeAgentBrain
    from spark.tools.timers import TimerService
    from tests.test_brain_claude_agent import FakeClient, ManualClock, github_setup, settings

    b = ClaudeAgentBrain(
        settings(mode="deep", workers=True),
        clock=ManualClock(),
        mcp=github_setup(),
        timers=TimerService(clock=FakeClock()),
    )
    try:
        await _ask(b, "how many PRs are open")
        assert FakeClient.instances[0].options.agents["researcher"].model == "claude-haiku-5-5"
        b.set_models(worker="claude-sonnet-5")
        await b.reset()
        assert FakeClient.instances[-1].options.agents["researcher"].model == "claude-sonnet-5"
        b.set_models(worker="")
        assert b.worker_model == "claude-haiku-5-5"
        b.set_models(fast="claude-sonnet-5")  # worker unchanged (None), follows fast
        assert b.worker_model == "claude-sonnet-5"
    finally:
        await b.close()


async def test_app_saves_the_researcher_and_reports_it(tmp_path):
    from tests.test_app import _app

    prefs = Prefs(tmp_path / "prefs.json")
    brain = ModeBrain()
    brain.workers_active = True
    app = _app(tmp_path, brain=brain, prefs=prefs)
    app.set_models(worker="claude-opus-5-5")
    assert prefs.get("worker_model") == "claude-opus-5-5"
    assert app.model_info()["worker_label"] == "Opus 5.5"
    assert brain.models[-1] == (None, None)  # ModeBrain records (fast, deep) only
    reply = await app.text_turn("which model are you using")
    assert reply.endswith("Opus 5.5 does the research.")
    await app.close()


async def test_controller_researcher_before_start(tmp_path):
    from tests.test_desktop import _controller

    c, events, _ = _controller(tmp_path)
    await c.aset_models(None, None, "claude-sonnet-5")
    assert c.settings.brain.worker_model == "claude-sonnet-5"
    assert events[-1]["worker_label"] == "Sonnet 5"
    await c.aset_models(None, None, "")
    assert c.settings.brain.worker_model is None and c.prefs.get("worker_model") == ""
    await c.aclose()
