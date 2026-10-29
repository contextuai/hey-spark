"""Personalities (spark/persona.py): the prompt section, Spark's fixed lines in the persona's
words, the spoken wake reply, the voice prefs, and the Personality & Voice window's API."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from spark.app import SparkApp, State, failure_line, strip_spoken_prefix, timer_announcement
from spark.audio.recorder import Recorder
from spark.brain.prompts import VOICE_SYSTEM_PROMPT, voice_system_prompt
from spark.commands import NEW_CONVERSATION, SLEEP
from spark.config import Settings
from spark.persona import DEFAULT_GO_AHEAD, PERSONAS, VOICES, Voices, persona_for, voice_id
from spark.prefs import Prefs, apply_prefs
from spark.tools.timers import TimerService
from spark.tts.kokoro_onnx import kokoro_lang
from tests.fakes import (
    FakeClock,
    FakeMic,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    FakeVAD,
    FakeWakeWord,
    MockBrain,
    concat,
    make_settings,
    silence,
    tone,
)

SR = 16_000
BLOCK = 512


def butler(**app) -> Settings:
    return make_settings(app={"persona": "butler", **app})


# --- lines -------------------------------------------------------------------------------------


def test_standard_lines_are_the_old_fixed_strings():
    v = persona_for(Settings())
    assert v.persona.key == "standard"
    assert v.line("ack") == "On it, give me a moment."
    assert v.line("fresh_start") == "Okay, fresh start."
    assert v.line("timer_done", timer="pasta timer") == "Your pasta timer is done."
    assert v.wake_reply == ""  # the chime, as before
    assert v.prompt() == ""
    assert v.confirm_line("I'll create it. Should I go ahead?") == (
        "I'll create it. Should I go ahead?"
    )


def test_butler_lines_use_the_address_and_drop_it_cleanly():
    v = persona_for(butler())
    assert v.line("ack") == "Right away, sir. One moment."
    assert v.wake_reply == "Yes, sir?"
    assert v.line("timer_done", timer="pasta timer") == "Your pasta timer has finished, sir."
    nameless = persona_for(butler(address_as=""))
    assert nameless.line("ack") == "Right away. One moment."
    assert nameless.wake_reply == "Yes?"
    assert persona_for(butler(address_as="ma'am")).line("sleep") == "Very good, ma'am."
    assert persona_for(butler(wake_reply=False)).wake_reply == ""


def test_every_persona_has_every_standard_line():
    keys = set(PERSONAS["standard"].phrases)
    for p in PERSONAS.values():
        assert set(p.phrases) == keys, p.key
        v = Voices(p, address="sir")
        for key in keys:
            assert "{" not in v.line(key, timer="t"), (p.key, key)


def test_butler_confirmations_in_its_own_words():
    v = persona_for(butler())
    assert v.confirm_line(f"I'll create a branch called x. {DEFAULT_GO_AHEAD}") == (
        "I'll create a branch called x. Shall I proceed, sir?"
    )
    assert v.confirm_line("Sorry, was that a yes or a no?") == (
        "I beg your pardon, sir. Was that a yes or a no?"
    )


def test_failure_and_timer_lines_follow_the_persona():
    v = persona_for(butler())
    assert failure_line(TimeoutError(), v).startswith("My apologies, sir")
    assert failure_line(RuntimeError("401 Unauthorized"), v).startswith("Claude didn't accept")
    assert failure_line(TimeoutError()) == "Sorry, that took too long. Try again in a moment."

    class T:
        spoken_name = "tea timer"

    assert timer_announcement(T(), v) == "Your tea timer has finished, sir."


# --- the prompt --------------------------------------------------------------------------------


def test_standard_prompt_is_unchanged():
    v = persona_for(Settings())
    assert voice_system_prompt(persona=v.prompt(), ack=v.line("ack")) == VOICE_SYSTEM_PROMPT


def test_butler_prompt_adds_the_personality_and_quotes_its_ack():
    v = persona_for(butler())
    p = voice_system_prompt(persona=v.prompt(), ack=v.line("ack"))
    assert p.startswith(VOICE_SYSTEM_PROMPT.split("\n")[0])
    assert "British butler" in p
    assert 'Address the user as "sir"' in p
    assert '"Right away, sir. One moment"' in p
    assert "On it, give me a moment" not in p
    assert "Every rule above still applies" in p  # after the rules, never instead of them
    assert "Don't use a title" in persona_for(butler(address_as="")).prompt()


async def test_brain_uses_the_persona_prompt_ack_and_fresh_start():
    from spark.brain.openai_compat import OpenAICompatBrain
    from tests.openai_fakes import KEY_ENV, Script, github_setup, reply

    s = Settings(
        app={"persona": "butler"},
        brain={
            "backend": "openai_compat",
            "mode": "fast",
            "workers": False,
            "openai_compat": {"provider": "openai", "fast_model": "f", "deep_model": "d"},
        },
        routing={"ack_after_s": 0},
    )
    script = Script(reply("Very good."))
    brain = OpenAICompatBrain(
        s, transport=script.transport(), environ=KEY_ENV, mcp=github_setup(), clock=FakeClock()
    )
    try:
        assert brain.ack_text == "Right away, sir. One moment."
        assert "British butler" in brain.system_prompt
        assert "".join([d async for d in brain.respond("new topic")]) == (
            "Of course, sir. A clean slate."
        )
    finally:
        await brain.close()
    s.routing.ack_text = "Hold on."
    brain = OpenAICompatBrain(
        s, transport=script.transport(), environ=KEY_ENV, mcp=github_setup(), clock=FakeClock()
    )
    assert brain.ack_text == "Hold on."  # config.yaml's own line wins
    await brain.close()


# --- the recorder skips Spark's own wake reply -------------------------------------------------


def test_recorder_ignores_and_drops_the_reply_window():
    rec = Recorder(Settings(), FakeVAD())
    rec.start(np.zeros(800, dtype=np.int16), ignore_s=0.5)
    loud = tone(0.5)  # Spark saying "Yes, sir?": loud, but never speech
    for i in range(0, len(loud), BLOCK):
        rec.add(loud[i : i + BLOCK])
    assert rec.first_speech_sample is None and not rec.done
    for chunk in (silence(0.2), tone(0.6), silence(0.8)):
        for i in range(0, len(chunk), BLOCK):
            rec.add(chunk[i : i + BLOCK])
    assert rec.end_of_speech
    # the reply's audio was not kept: pre-roll + at most a block of it + the rest
    assert rec.pcm().size < 800 + BLOCK + int(SR * 1.6)


def test_recorder_ignore_window_still_counts_toward_no_speech():
    rec = Recorder(Settings(recorder={"no_speech_timeout_s": 1.0}), FakeVAD())
    rec.start(None, ignore_s=5.0)
    quiet = silence(1.2)
    for i in range(0, len(quiet), BLOCK):
        rec.add(quiet[i : i + BLOCK])
    assert rec.reason == "no_speech"


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Yes, sir? What time is it?", "What time is it?"),
        ("yes sir what time is it", "what time is it"),
        ("Yes sir.", ""),
        ("Yesterday was fine", "Yesterday was fine"),
        ("What time is it?", "What time is it?"),
    ],
)
def test_strip_spoken_prefix(raw, clean):
    assert strip_spoken_prefix(raw, "Yes, sir?") == clean


# --- the app -----------------------------------------------------------------------------------


def _app(settings: Settings, **kw) -> SparkApp:
    return SparkApp(
        settings,
        mic=FakeMic(),
        player=FakePlayer(),
        wakeword=FakeWakeWord(),
        vad=FakeVAD(),
        stt=kw.pop("stt", None) or FakeSTT(transcripts=["what is two plus two"]),
        tts=FakeTTS(),
        brain=kw.pop("brain", None) or MockBrain(),
        **kw,
    )


def _settings(tmp_path, **app) -> Settings:
    return make_settings(
        app=app,
        logging={"dir": str(tmp_path / "logs")},
        attention={"awake_s": 8.0},
    )


async def _feed(app: SparkApp, pcm: np.ndarray) -> None:
    for i in range(0, len(pcm), BLOCK):
        await app.feed(pcm[i : i + BLOCK])


async def test_butler_answers_the_wake_word_out_loud_instead_of_the_chime(tmp_path):
    stt = FakeSTT(transcripts=["Yes, sir? What is two plus two"])
    app = _app(_settings(tmp_path, persona="butler"), stt=stt)
    await app.warmup()
    assert app.tts.rendered == ["Yes, sir?"]
    app.wakeword.trigger()
    await _feed(app, silence(BLOCK / SR))
    assert app.state is State.LISTENING
    assert app.player.clips == [(int(SR * app.tts.render_s), SR)]
    assert "wake" not in app.player.earcons
    assert app.recorder.ignore_samples == int(SR * (app.tts.render_s + 0.15))
    await _feed(app, concat(tone(0.6), silence(0.2)))  # the reply itself: not the request
    assert app.state is State.LISTENING
    await _feed(app, concat(tone(0.6), silence(0.8)))
    await asyncio.wait_for(app.turn_task, 5)
    assert app.brain.calls == ["What is two plus two"]
    await app.close()


async def test_standard_persona_keeps_the_chime(tmp_path):
    app = _app(_settings(tmp_path))
    await app.warmup()
    assert app.tts.rendered == []
    app.wakeword.trigger()
    await _feed(app, silence(BLOCK / SR))
    assert app.player.earcons == ["wake"] and app.player.clips == []
    assert app.recorder.ignore_samples == 0
    await app.close()


async def test_wake_reply_falls_back_to_the_chime_when_it_cannot_be_made(tmp_path):
    app = _app(_settings(tmp_path, persona="butler"))

    def broken(_text):
        raise RuntimeError("no voice")

    app.tts.render = broken
    await app.warmup()
    app.wakeword.trigger()
    await _feed(app, silence(BLOCK / SR))
    assert app.player.earcons == ["wake"]
    await app.close()


async def test_butler_command_replies_and_confirmations(tmp_path):
    app = _app(_settings(tmp_path, persona="butler"))
    assert app._command_reply(SLEEP) == "Very good, sir."
    assert app._command_reply(NEW_CONVERSATION) == "Of course, sir. A clean slate."
    assert await app.text_turn("go to sleep") == "Very good, sir."
    await app.say_for_confirmation(f"I'll create a repo. {DEFAULT_GO_AHEAD}")
    assert app.confirm_questions == ["I'll create a repo. Shall I proceed, sir?"]
    await app.close()


async def test_butler_timer_announcement(tmp_path):
    timers = TimerService(clock=FakeClock())
    app = _app(_settings(tmp_path, persona="butler"), timers=timers)
    try:
        timers.set("5 minutes", "pasta")
        for _ in range(500):
            if app.tts.spoken:
                break
            await asyncio.sleep(0.005)
        assert app.tts.spoken == ["Your pasta timer has finished, sir."]
    finally:
        await app.close()


# --- prefs and voices --------------------------------------------------------------------------


def test_voice_prefs_override_config(tmp_path):
    prefs = Prefs(tmp_path / "prefs.json")
    prefs.update(persona="butler", address_as="boss", wake_reply=False, tts_voice="kokoro:bm_fable")
    s = Settings()
    apply_prefs(s, prefs)
    assert (s.app.persona, s.app.address_as, s.app.wake_reply) == ("butler", "boss", False)
    assert s.tts.backend == "kokoro" and s.tts.options.kokoro.voice == "bm_fable"
    assert voice_id(s) == "kokoro:bm_fable"
    prefs.update(tts_voice="piper:en_GB-cori-high", persona="nobody")
    apply_prefs(s, prefs)
    assert s.tts.backend == "piper" and voice_id(s) == "piper:en_GB-cori-high"
    assert s.app.persona == "butler"  # an unknown persona is ignored


def test_kokoro_lang_follows_the_voice():
    s = Settings()
    k = s.tts.options.kokoro
    assert kokoro_lang(k) == "en-gb"  # bm_george
    k.voice = "af_heart"
    assert kokoro_lang(k) == "en-us"
    k.lang = "en-us"
    assert kokoro_lang(k) == "en-us"


def test_voices_are_well_formed():
    ids = [v.id for v in VOICES]
    assert len(ids) == len(set(ids))
    for v in VOICES:
        assert v.id == f"{v.backend}:{v.voice}"
        assert "lessac" not in v.voice
    assert PERSONAS["butler"].voice in ids


# --- the Personality & Voice window ------------------------------------------------------------


def _api(tmp_path, applied: list[int] | None = None):
    from spark.desktop.voice_settings import VoiceSettingsApi

    s = Settings(config_dir=tmp_path, models_dir=str(tmp_path / "models"))

    def on_apply():
        if applied is not None:
            applied.append(1)
        return None

    return s, VoiceSettingsApi(s, on_apply=on_apply)


def test_window_load_lists_personas_and_voices(tmp_path):
    _s, api = _api(tmp_path)
    view = api.load()
    assert view["ok"] and view["persona"] == "standard"
    assert view["voice"] == "kokoro:bm_george"
    assert all(v["id"].startswith("kokoro:") for v in view["voices"])
    assert [p["key"] for p in view["personas"]] == ["standard", "butler"]
    george = next(v for v in view["voices"] if v["id"] == "kokoro:bm_george")
    assert george["downloaded"] is False and george["size"] == "340 MB"
    json.dumps(view)


def test_window_save_applies_writes_prefs_and_rebuilds(tmp_path):
    applied: list[int] = []
    s, api = _api(tmp_path, applied)
    r = api.save(
        {
            "persona": "butler",
            "address_as": "  ma'am ",
            "wake_reply": True,
            "voice": "kokoro:bm_george",
        }
    )
    assert r["ok"] and r["saved"] and applied == [1]
    assert (s.app.persona, s.app.address_as, s.app.wake_reply) == ("butler", "ma'am", True)
    assert s.tts.backend == "kokoro" and s.tts.options.kokoro.voice == "bm_george"
    saved = json.loads((tmp_path / "prefs.json").read_text(encoding="utf-8"))
    assert saved["tts_voice"] == "kokoro:bm_george" and saved["persona"] == "butler"
    assert r["persona"] == "butler" and r["voice"] == "kokoro:bm_george"


def test_window_rejects_bad_input_and_reports_apply_errors(tmp_path):
    s, api = _api(tmp_path)
    assert api.save({"persona": "pirate"})["ok"] is False
    assert api.save({"persona": "butler", "voice": "piper:en_GB-alan-medium"})["ok"] is False
    assert api.save({"persona": "butler", "address_as": "x" * 41})["ok"] is False
    assert s.app.persona == "standard"  # nothing applied
    api._on_apply = lambda: "Spark couldn't start."
    r = api.save({"persona": "butler", "voice": "kokoro:bf_emma"})
    assert r["ok"] is False and r["error"] == "Spark couldn't start." and r["saved"]


def test_window_preview_needs_the_voice_downloaded(tmp_path):
    _s, api = _api(tmp_path)
    r = api.preview({"persona": "butler", "voice": "kokoro:bm_george"})
    assert r["ok"] is False and "downloads when you save" in r["error"]
    assert api.preview({"voice": "nope:x"})["ok"] is False


def test_window_preview_returns_a_wav_with_the_sample_line(tmp_path, monkeypatch):
    s, api = _api(tmp_path)
    monkeypatch.setattr(api, "_downloaded", lambda _v: True)
    tts = FakeTTS()
    monkeypatch.setattr(api, "_engine", lambda _vid, _settings: tts)
    r = api.preview({"persona": "butler", "address_as": "sir", "voice": "kokoro:bm_lewis"})
    assert r["ok"], r
    assert r["text"] == "Good evening, sir. Spark, at your service. Shall I check your calendar?"
    assert r["audio"].startswith("data:audio/wav;base64,")
    assert tts.rendered == [r["text"]]
    assert s.app.persona == "standard"  # a preview changes nothing
