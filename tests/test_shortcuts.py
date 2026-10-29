"""Voice shortcuts (Phase 5, P5.7a): matching, the store, and the app sending the prompt."""

from __future__ import annotations

import json
import os

import pytest

from spark.shortcuts import (
    DEFAULT_SHORTCUTS,
    Shortcut,
    ShortcutError,
    ShortcutStore,
    expand,
    match_shortcut,
    normalize,
)

NAMES = ("spark", "spark")


def _defaults() -> list[Shortcut]:
    return [Shortcut.from_dict(dict(d)).validate() for d in DEFAULT_SHORTCUTS]


def test_normalize_expands_contractions():
    assert normalize("How’s today looking?") == ["how", "is", "today", "looking"]
    assert normalize("What's on, Spark!") == ["what", "is", "on", "spark"]


@pytest.mark.parametrize(
    "said",
    [
        "Good morning, how is today looking like",  # the user's own example
        "good morning",
        "Good morning Spark!",
        "Hey Spark, how's today looking?",
        "so what does today look like",
        "good morning, how is today looking for the Denver team",
    ],
)
def test_the_morning_briefing_fires(said):
    match = match_shortcut(said, _defaults(), names=NAMES)
    assert match is not None and match.shortcut.name == "Morning briefing"


@pytest.mark.parametrize(
    "said",
    [
        "good morning, can you also move my 3pm meeting to Friday and tell Sam",
        "morning",
        "what is the weather today",
        "",
    ],
)
def test_requests_that_only_contain_a_trigger_do_not_fire(said):
    assert match_shortcut(said, _defaults(), names=NAMES) is None


def test_extra_words_are_kept_for_context():
    match = match_shortcut("good morning, how is today looking for Denver", _defaults())
    assert match is not None and match.extra == "for denver"
    prompt = expand(match, "good morning, how is today looking for Denver")
    assert prompt.startswith("Give me my morning briefing")
    assert 'The user said: "good morning, how is today looking for Denver"' in prompt


def test_disabled_shortcuts_never_fire_and_the_longest_trigger_wins():
    items = _defaults()
    items.append(Shortcut("Standup", ["good morning team"], "Draft my standup notes.").validate())
    assert match_shortcut("good morning team", items).shortcut.name == "Standup"
    items[0].enabled = False
    assert match_shortcut("good morning", items) is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "", "triggers": ["a b"], "prompt": "x"}, "name"),
        ({"name": "x", "triggers": [], "prompt": "x"}, "phrase"),
        ({"name": "x", "triggers": ["hi"], "prompt": "x"}, "two words"),
        ({"name": "x", "triggers": ["a b"], "prompt": " "}, "prompt"),
        ({"name": "x", "triggers": ["a b"], "prompt": "x", "route": "warp"}, "Route"),
    ],
)
def test_validation(data, message):
    with pytest.raises(ShortcutError, match=message):
        Shortcut.from_dict(data).validate()


def test_store_seeds_saves_and_reloads_hand_edits(tmp_path):
    store = ShortcutStore(tmp_path / "shortcuts.json")
    assert [s.name for s in store.all()] == [d["name"] for d in DEFAULT_SHORTCUTS]
    saved = store.save(
        {"name": "Wind down", "triggers": "wind down\nclose the books", "prompt": "Summarize."}
    )
    assert saved.triggers == ["wind down", "close the books"] and saved.id.startswith("s-")
    assert [s.name for s in ShortcutStore(store.path).all()][-1] == "Wind down"
    updated = store.save({**saved.to_dict(), "prompt": "Summarize today."})
    assert updated.id == saved.id and updated.created_at == saved.created_at
    assert len(store.all()) == len(DEFAULT_SHORTCUTS) + 1
    with pytest.raises(ShortcutError, match="already starts"):
        store.save({"name": "Dup", "triggers": ["Good morning!"], "prompt": "x"})
    store.set_enabled(saved.id, False)
    assert not next(s for s in store.all() if s.id == saved.id).enabled
    # a hand edit is picked up without a restart
    data = json.loads(store.path.read_text(encoding="utf-8"))
    data["shortcuts"] = data["shortcuts"][:1]
    store.path.write_text(json.dumps(data), encoding="utf-8")
    os.utime(store.path, (1, 1))
    assert [s.name for s in store.all()] == ["Morning briefing"]
    store.delete(store.all()[0].id)
    assert store.all() == []


def test_default_triggers_never_clash():
    triggers = [" ".join(normalize(t)) for d in DEFAULT_SHORTCUTS for t in d["triggers"]]
    assert len(triggers) == len(set(triggers))


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("Rise and shine!", "Morning briefing"),
        ("Wake up, daddy's home", "Morning briefing"),
        ("Spark, give me a rundown", "What needs me"),
        ("Sitrep", "Sitrep"),
        ("Status report, please", "Sitrep"),
        ("What's next?", "Next meeting"),
        ("Hey Spark, who's next", "Next meeting"),
        ("Prep me for my next meeting with legal", "Meeting prep"),
        ("Check the post", "Inbox"),
        ("What's happening in the workshop?", "Workshop report"),
        ("Honey, I'm home!", "Home"),
        ("Let's wrap up the day", "End of day"),
        ("Run a diagnostic", "Diagnostic"),
        ("Time check", "Time check"),
        ("Spark, you up?", "You up"),
        ("Drop my needle", "Drop the needle"),
        ("House party protocol", "House party protocol"),
        ("Start a new project called weather station", "New project"),
    ],
)
def test_the_film_style_shortcuts_fire(said, name):
    match = match_shortcut(said, _defaults(), names=NAMES)
    assert match is not None and match.shortcut.name == name


@pytest.mark.parametrize(
    "said",
    [
        "what's next week looking like",
        "who's next in line for the promotion",
        "I'm home sick today, can you tell the team",
        "you up for a game of chess",
        "what is the situation with the build",
    ],
)
def test_exact_shortcuts_only_fire_on_their_own(said):
    match = match_shortcut(said, _defaults(), names=NAMES)
    assert match is None or not match.shortcut.exact


def test_one_word_phrases_only_for_exact_shortcuts():
    assert Shortcut("x", ["sitrep"], "x", exact=True).validate().triggers == ["sitrep"]
    with pytest.raises(ShortcutError, match="two words"):
        Shortcut("x", ["sitrep"], "x").validate()


def _legacy_file(path, shortcuts):
    path.write_text(json.dumps({"shortcuts": shortcuts}), encoding="utf-8")


def test_new_defaults_are_added_to_an_existing_file_once(tmp_path):
    path = tmp_path / "shortcuts.json"
    morning = Shortcut.from_dict(dict(DEFAULT_SHORTCUTS[0])).validate()
    morning.triggers = ["good morning", "morning report"]  # the user's own edit
    mine = Shortcut("Code", ["code report"], "My own code report.").validate()
    # written before this release: "What needs me" was deleted by the user
    _legacy_file(path, [morning.to_dict(), mine.to_dict()])

    names = [s.name for s in ShortcutStore(path).all()]
    assert "What needs me" not in names  # deleted defaults stay deleted
    assert {"Sitrep", "Time check", "New project"} <= set(names)
    items = {s.name: s for s in ShortcutStore(path).all()}
    assert items["Morning briefing"].triggers == [
        "good morning",
        "morning report",
        "rise and shine",
        "wake up daddy's home",
    ]
    assert "code report" not in items["Workshop report"].triggers  # the user's shortcut has it
    assert json.loads(path.read_text(encoding="utf-8"))["defaults"]["Sitrep"]

    # deleting a new default sticks, too
    store = ShortcutStore(path)
    store.delete(next(s.id for s in store.all() if s.name == "Sitrep"))
    assert "Sitrep" not in [s.name for s in ShortcutStore(path).all()]


def test_a_broken_file_means_no_shortcuts(tmp_path, caplog):
    path = tmp_path / "shortcuts.json"
    path.write_text("{nope", encoding="utf-8")
    assert ShortcutStore(path).all() == []
    assert "unreadable" in caplog.text


def test_no_seed_when_turned_off(tmp_path):
    store = ShortcutStore(tmp_path / "shortcuts.json", seed=False)
    assert store.all() == [] and not store.path.exists()


# --- the app sends the prompt -----------------------------------------------------------------


class RouteBrain:
    """MockBrain plus the brain's one-shot route attribute."""

    def __new__(cls, **kw):
        from spark.brain.mock import MockBrain

        brain = MockBrain(**kw)
        brain.next_route = None
        brain.next_route_reason = ""
        return brain


async def test_a_shortcut_sends_the_detailed_prompt(tmp_path):
    from tests.fakes import FakeSTT
    from tests.test_app import _app, _finish_turn, _settings, _utterance, _wake

    store = ShortcutStore(tmp_path / "shortcuts.json")
    brain = RouteBrain(replies=["You have three meetings."])
    seen_routes: list[str | None] = []
    original = brain.respond

    def respond(text):
        seen_routes.append(brain.next_route)
        return original(text)

    brain.respond = respond
    said = "Good morning, how is today looking like"
    app = _app(
        tmp_path,
        stt=FakeSTT(transcripts=[said]),
        brain=brain,
        shortcuts=store,
        settings=_settings(tmp_path),
    )
    await _wake(app)
    await _utterance(app)
    await _finish_turn(app)
    (sent,) = brain.calls
    assert sent.startswith("Give me my morning briefing") and said in sent
    assert seen_routes == ["deep"] and brain.next_route_reason == "shortcut 'Morning briefing'"
    assert app.shortcuts_fired[0].shortcut.name == "Morning briefing"
    assert app.last_reply == "You have three meetings."
    await app.close()


async def test_a_shortcut_while_awake_skips_the_addressee_check(tmp_path):
    from tests.test_app import _awake_app, _awake_turn, _finish_turn, _utterance

    store = ShortcutStore(tmp_path / "shortcuts.json")
    app = _awake_app(tmp_path, ["what is two plus two", "good morning"], shortcuts=store)
    await _awake_turn(app)
    await _utterance(app)
    await _finish_turn(app)
    assert len(app.brain.calls) == 2 and app.brain.calls[1].startswith("Give me my morning")
    assert app.ignored == []
    await app.close()


async def test_text_mode_shortcuts(tmp_path):
    from tests.test_app import _app

    app = _app(tmp_path, shortcuts=ShortcutStore(tmp_path / "shortcuts.json"))
    await app.text_turn("anything urgent?")
    assert app.brain.calls[0].startswith("Check what needs my attention")
    await app.close()
