"""Local voice commands (Phase 5, P5.3): spark.commands plus the app handling them."""

from __future__ import annotations

import pytest

from spark.commands import (
    CANCEL,
    MODE_DEEP,
    MODE_FAST,
    MUTE,
    NEW_CONVERSATION,
    SLEEP,
    match_command,
)

NAMES = ("spark", "sparks", "sparky")


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Go to sleep.", SLEEP),
        ("Now go for a sleep.", SLEEP),  # what the user actually said on 2026-10-02
        ("Spark, that's all, thanks!", SLEEP),
        ("Okay that’s it for now", SLEEP),
        ("Goodbye Spark.", SLEEP),
        ("Hey Sparky, stop listening please", SLEEP),
        ("Never mind.", CANCEL),
        ("Stop.", CANCEL),
        ("Cancel that please", CANCEL),
        ("New conversation.", NEW_CONVERSATION),
        ("Let's start a new conversation.", NEW_CONVERSATION),
        ("Okay, new topic please.", NEW_CONVERSATION),
        ("Spark, start over", NEW_CONVERSATION),
        ("Stand down.", SLEEP),
        ("Lock it down, Spark", SLEEP),
        ("Belay that", CANCEL),
        ("Abort!", CANCEL),
        ("Clean slate protocol", NEW_CONVERSATION),
        ("Wipe the board", NEW_CONVERSATION),
        ("Full power", MODE_DEEP),
        ("Bring out the big guns", MODE_DEEP),
        ("Conserve power", MODE_FAST),
        ("Go dark", MUTE),
        ("Silent running", MUTE),
    ],
)
def test_commands_match(text, kind):
    cmd = match_command(text, NAMES)
    assert cmd is not None and cmd.kind == kind


@pytest.mark.parametrize(
    "text",
    [
        "stop the timer",
        "cancel my 3pm meeting",
        "new topic, what's the weather tomorrow?",
        "how do I go to sleep faster",
        "that's all the open PRs?",
        "okay",
        "thanks",
        "Spark",
        "",
        "   ",
        "never mind the email, check my calendar",
    ],
)
def test_requests_are_never_commands(text):
    assert match_command(text, NAMES) is None


def test_custom_phrases():
    cmd = match_command("lights out", NAMES, phrases={SLEEP: ["lights out"]})
    assert cmd is not None and cmd.kind == SLEEP
    assert match_command("go to sleep", NAMES, phrases={SLEEP: ["lights out"]}) is None
