"""Local voice commands, handled without Claude (Phase 5, P5.3).

A command is the *whole* utterance, give or take the assistant's name, politeness and filler:
"go to sleep", "Spark, that's all, thanks", "okay new conversation please". Anything with more
in it ("stop the timer", "new topic: what's the weather?") is not a command and goes to the brain
as before, so a command can never swallow a real request.

Kinds:

* ``sleep``: "go to sleep", "that's all", "goodbye", "stop listening", "stand down" ... Spark
  says "Okay." and falls asleep (sleep chime; the wake word is needed again).
* ``cancel``: "never mind", "cancel that", "stop", "forget it", "belay that" ... Spark falls
  asleep without saying anything.
* ``new_conversation``: "new conversation", "new topic", "start over", "clean slate" ... Spark
  starts a fresh Claude conversation and says "Okay, fresh start." (the brain is not asked).
* ``mode_deep`` / ``mode_fast`` / ``mode_auto``: "use the deep model", "fast mode", "back to
  auto" ... which model answers (P5.7c). ``mode_status``: "which model are you using?".
* ``mute``: "mute", "mute yourself", "go on mute", "go dark" ... Spark stops listening until the
  mic button in the window unmutes it (P5.7k); it can't hear an "unmute".

Pure Python, no heavy imports.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "COMMAND_PHRASES",
    "Command",
    "match_command",
    "normalize_command_text",
    "split_model_request",
]

SLEEP = "sleep"
CANCEL = "cancel"
NEW_CONVERSATION = "new_conversation"
MODE_DEEP = "mode_deep"
MODE_FAST = "mode_fast"
MODE_AUTO = "mode_auto"
MODE_STATUS = "mode_status"
MODE_COMMANDS = (MODE_DEEP, MODE_FAST, MODE_AUTO, MODE_STATUS)
MUTE = "mute"

COMMAND_PHRASES: dict[str, tuple[str, ...]] = {
    SLEEP: (
        "go to sleep",
        "go for a sleep",
        "go for sleep",
        "go back to sleep",
        "you can go to sleep",
        "you can sleep",
        "sleep",
        "sleep now",
        "time to sleep",
        "stop listening",
        "that's all",
        "that is all",
        "that's it",
        "that is it",
        "that's it for now",
        "that's all for now",
        "that will be all",
        "that'll be all",
        "nothing else",
        "we're done",
        "we are done",
        "i'm done",
        "i am done",
        "goodbye",
        "good bye",
        "bye",
        "bye bye",
        "see you",
        "see you later",
        "talk to you later",
        "later",
        "stand down",
        "lock it down",
        "power down",
    ),
    CANCEL: (
        "never mind",
        "nevermind",
        "cancel",
        "cancel that",
        "stop",
        "stop it",
        "stop that",
        "forget it",
        "no thanks",
        "no thank you",
        "nothing",
        "not now",
        "belay that",
        "scratch that",
        "abort",
    ),
    NEW_CONVERSATION: (
        "new conversation",
        "start a new conversation",
        "start new conversation",
        "new topic",
        "change of topic",
        "change the topic",
        "different topic",
        "start over",
        "start fresh",
        "fresh start",
        "reset conversation",
        "reset the conversation",
        "clear the conversation",
        "clean slate",
        "clean slate protocol",
        "wipe the board",
        "wipe the slate",
    ),
    MODE_DEEP: (
        "use the deep model",
        "use deep model",
        "use the deep mode",
        "deep mode",
        "switch to deep mode",
        "switch to the deep model",
        "always use the deep model",
        "use the smart model",
        "use the smarter model",
        "use the best model",
        "use the big model",
        "use your best model",
        "think harder",
        "full power",
        "bring out the big guns",
    ),
    MODE_FAST: (
        "use the fast model",
        "use fast model",
        "fast mode",
        "switch to fast mode",
        "switch to the fast model",
        "always use the fast model",
        "use the quick model",
        "quick mode",
        "conserve power",
        "power saving mode",
    ),
    MODE_AUTO: (
        "auto mode",
        "automatic mode",
        "back to auto",
        "go back to auto",
        "switch to auto",
        "switch to auto mode",
        "use auto mode",
        "pick the model yourself",
        "choose the model yourself",
        "pick the model automatically",
    ),
    MUTE: (
        "mute",
        "mute yourself",
        "mute the mic",
        "mute your mic",
        "go on mute",
        "go mute",
        "mute please",
        "go dark",
        "silent running",
    ),
    MODE_STATUS: (
        "which model are you using",
        "what model are you using",
        "which model is this",
        "what model is this",
        "what mode are you in",
        "which mode are you in",
    ),
}

#: Words that may surround a command without making it a request.
_FILLER = (
    "hey",
    "hi",
    "ok",
    "okay",
    "so",
    "um",
    "uh",
    "alright",
    "all right",
    "right",
    "yeah",
    "yes",
    "no",
    "please",
    "thanks",
    "thank you",
    "now",
    "just",
    "then",
    "let's",
    "lets",
    "let us",
    "can you",
    "could you",
    "you can",
)


#: Longer utterances are never commands (a command plus filler fits easily).
MAX_COMMAND_WORDS = 12


@dataclass(frozen=True)
class Command:
    kind: str  # SLEEP | CANCEL | NEW_CONVERSATION
    phrase: str  # the phrase that matched (normalized)


def normalize_command_text(text: str) -> str:
    """Lowercase, straight apostrophes, words separated by single spaces, no punctuation."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def _stripped_forms(words: list[str], strip: Sequence[tuple[str, ...]]) -> list[str]:
    """Every phrase reachable by dropping filler/name word sequences from either end ("okay
    that's it for now" -> "that's it for now", "that's it for", ...), shortest last."""
    seen: dict[str, None] = {}
    frontier = [tuple(words)]
    while frontier:
        current = frontier.pop()
        key = " ".join(current)
        if key in seen:
            continue
        seen[key] = None
        for seq in strip:
            n = len(seq)
            if len(current) > n and current[:n] == seq:
                frontier.append(current[n:])
            if len(current) > n and current[-n:] == seq:
                frontier.append(current[:-n])
    return list(seen)


def _phrase_index(phrases: dict[str, Iterable[str]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for kind, items in phrases.items():
        for phrase in items:
            index.setdefault(normalize_command_text(phrase), kind)
    return index


_INDEX = _phrase_index(COMMAND_PHRASES)


_ONE_SHOT = re.compile(
    r"^\W*(?:(?:hey|ok|okay)\W+)?(?:\w+\W+)??(?:please\W+)?(?:can you\W+)?"
    r"(?:use|with|ask)\W+(?:the\W+|your\W+)?"
    r"(deep|smart|smarter|best|big|bigger|fast|quick|quicker)\W+model\W+"
    r"(?:to\W+|and\W+|for\W+|on\W+)?(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def split_model_request(text: str) -> tuple[str, str] | None:
    """ "Use the deep model to check WAREHOUSE's orders" -> ``("deep", "check WAREHOUSE's
    orders")``: one request on the named route, without changing the mode. None if ``text``
    doesn't start that way."""
    m = _ONE_SHOT.match(text.strip())
    if m is None or not re.search(r"\w", m.group(2)):
        return None
    route = "fast" if m.group(1).lower() in ("fast", "quick", "quicker") else "deep"
    return route, m.group(2).strip()


def match_command(
    text: str,
    names: Sequence[str] = ("spark",),
    *,
    phrases: dict[str, Iterable[str]] | None = None,
) -> Command | None:
    """The local command ``text`` is, or None if it is (or contains) a real request.

    ``names`` are the lowercase assistant names (``app.wake_names``) that may lead or trail the
    command ("Spark, go to sleep", "goodbye Spark")."""
    norm = normalize_command_text(text)
    if not norm:
        return None
    index = _INDEX if phrases is None else _phrase_index(phrases)
    if norm in index:
        return Command(index[norm], norm)
    if len(norm.split()) > MAX_COMMAND_WORDS:
        return None  # a real request; don't search
    strip = [tuple(w.split()) for w in (*_FILLER, *names) if w.strip()]
    strip.sort(key=len, reverse=True)  # "thank you" before "you"
    for core in _stripped_forms(norm.split(), strip):
        if core in index:
            return Command(index[core], core)
    return None
