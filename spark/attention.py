"""Is this utterance meant for Spark? (Phase 5, P5.4)

While Spark is awake (no wake word needed), the mic also hears colleagues, phone calls and the
user thinking out loud. Every utterance heard in the awake window passes :class:`Addressee`
before it reaches the brain. The rule is "when unsure, stay silent" (decision D1): only clear
requests, questions, answers to Spark's own question and continuations are answered. Ignored
speech does not restart the awake window, and the app logs it.

Modes (``attention.addressee``):

* ``rules`` (default): :func:`rules_verdict`, instant and offline.
* ``fast_model``: a tool-less :class:`~spark.brain.side.SideModel` sees Spark's last reply and the
  utterance and answers yes or no. On a timeout or error the rules decide. A name mention or an
  answer to Spark's own question is decided by the rules first, without asking.
* ``off``: everything heard while awake is answered (the Phase 1-3 behaviour).

Utterances after the wake word are never checked: saying the wake word is the address.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = ["Addressee", "Verdict", "rules_verdict"]

_FILLER_WORDS = frozenset(
    [
        "ok",
        "okay",
        "hmm",
        "hm",
        "mm",
        "mhm",
        "uh",
        "um",
        "ah",
        "oh",
        "huh",
        "yeah",
        "yep",
        "yup",
        "yes",
        "no",
        "nope",
        "right",
        "sure",
        "alright",
        "cool",
        "great",
        "nice",
        "wow",
        "thanks",
        "thank",
        "you",
        "fine",
        "good",
        "got",
        "it",
        "i",
        "see",
        "well",
        "so",
        "anyway",
        "lol",
        "haha",
    ]
)
_QUESTION_START = frozenset(
    [
        "what",
        "what's",
        "whats",
        "when",
        "when's",
        "where",
        "where's",
        "who",
        "who's",
        "whom",
        "whose",
        "why",
        "how",
        "how's",
        "which",
        "is",
        "isn't",
        "are",
        "aren't",
        "am",
        "was",
        "wasn't",
        "were",
        "can",
        "can't",
        "could",
        "couldn't",
        "do",
        "don't",
        "does",
        "doesn't",
        "did",
        "didn't",
        "will",
        "won't",
        "would",
        "wouldn't",
        "should",
        "shouldn't",
        "shall",
        "have",
        "haven't",
        "has",
        "hasn't",
        "had",
        "any",
        "anything",
    ]
)
_REQUEST_START = (
    "please",
    "can you",
    "could you",
    "would you",
    "will you",
    "i need",
    "i want",
    "i'd like",
    "i would like",
    "let's",
    "lets",
    "let me know",
    "go ahead",
    "tell me",
    "remind me",
    "help me",
)
_CONTINUATION_START = (
    "and",
    "also",
    "what about",
    "how about",
    "then",
    "but",
    "plus",
    "okay and",
    "ok and",
    "so what",
    "what else",
    "one more",
    "another",
    "same for",
    "do the same",
)
_VERBS = frozenset(
    [
        "add",
        "analyse",
        "analyze",
        "ask",
        "book",
        "build",
        "cancel",
        "change",
        "check",
        "close",
        "compare",
        "count",
        "create",
        "debug",
        "delete",
        "describe",
        "draft",
        "email",
        "explain",
        "find",
        "fix",
        "forward",
        "get",
        "give",
        "list",
        "look",
        "make",
        "move",
        "open",
        "pull",
        "push",
        "put",
        "read",
        "remember",
        "remind",
        "rename",
        "repeat",
        "reply",
        "run",
        "say",
        "schedule",
        "search",
        "send",
        "set",
        "share",
        "show",
        "start",
        "stop",
        "summarise",
        "summarize",
        "text",
        "try",
        "turn",
        "update",
        "write",
    ]
)


@dataclass(frozen=True)
class Verdict:
    addressed: bool
    reason: str  # short, for the log: "name", "question", "filler", "no request", ...
    by: str = "rules"  # "rules" | "model" | "off"


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    return " ".join(re.sub(r"[^\w\s'?]", " ", text).split())


def _starts(words: str, prefixes: Sequence[str]) -> str | None:
    for p in prefixes:
        if words == p or words.startswith(p + " "):
            return p
    return None


def rules_verdict(
    text: str,
    *,
    names: Sequence[str] = ("spark",),
    last_reply: str = "",
    since_reply_s: float | None = None,
    question_window_s: float = 30.0,
) -> Verdict:
    """Rule-based check. ``last_reply`` is what Spark said last and ``since_reply_s`` how long
    ago it finished speaking."""
    norm = _norm(text)
    words = norm.replace("?", " ").split()
    if not words:
        return Verdict(False, "empty")
    plain = " ".join(words)
    for name in names:
        if name and re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", plain):
            return Verdict(True, "name")
    asked = last_reply.rstrip().endswith("?")
    if asked and since_reply_s is not None and since_reply_s <= question_window_s:
        return Verdict(True, "answers Spark's question")
    if all(w in _FILLER_WORDS for w in words):
        return Verdict(False, "filler")
    if _starts(plain, _CONTINUATION_START):
        return Verdict(True, "continuation")
    if _starts(plain, _REQUEST_START):
        return Verdict(True, "request")
    if words[0] in _VERBS:
        return Verdict(True, "request")
    if words[0] in _QUESTION_START or norm.rstrip().endswith("?"):
        return Verdict(True, "question")
    return Verdict(False, "no request")


ADDRESSEE_PROMPT = """\
You decide whether a sentence overheard by a voice assistant named {name} was said TO the \
assistant. The user and the assistant were just talking, so the assistant is listening without \
its wake word, but the microphone also hears other people, phone calls, TV, and the user talking \
to someone else or to themselves.
You get the assistant's last reply and the overheard sentence. Reply with exactly one word: yes \
if the sentence is a request, question, or follow-up meant for the assistant; no if it is meant \
for someone else, is background talk, or you are not sure."""

_YES_NO = re.compile(r"\b(yes|no)\b")


class Addressee:
    """Decides whether awake-window speech is meant for Spark. Build with
    :meth:`from_settings`; the app calls :meth:`decide` per utterance."""

    def __init__(
        self,
        settings: Settings,
        *,
        mode: str | None = None,
        side: Any | None = None,
    ) -> None:
        cfg = settings.attention
        self.settings = settings
        self.mode = mode or cfg.addressee
        self.names = settings.app.wake_names
        self.question_window_s = cfg.question_window_s
        self.timeout_s = cfg.addressee_timeout_s
        self._side = side
        if self.mode == "fast_model" and self._side is None:
            from spark.brain.side import make_side_model

            self._side = make_side_model(
                settings,
                ADDRESSEE_PROMPT.format(name=settings.app.name),
                name="addressee",
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> Addressee:
        return cls(settings)

    async def warmup(self) -> None:
        if self._side is not None:
            try:
                await self._side.warmup()
            except Exception as exc:
                log.warning("addressee: model warm-up failed (%s); rules will decide", exc)

    async def aclose(self) -> None:
        if self._side is not None:
            await self._side.aclose()

    async def decide(
        self, text: str, *, last_reply: str = "", since_reply_s: float | None = None
    ) -> Verdict:
        if self.mode == "off":
            return Verdict(True, "check off", by="off")
        rules = rules_verdict(
            text,
            names=self.names,
            last_reply=last_reply,
            since_reply_s=since_reply_s,
            question_window_s=self.question_window_s,
        )
        if self.mode != "fast_model" or self._side is None:
            return rules
        if rules.reason in ("name", "answers Spark's question", "empty", "filler"):
            return rules  # certain either way: no need to ask
        prompt = f"Assistant's last reply: {last_reply or '(nothing yet)'}\nOverheard: {text}"
        try:
            async with asyncio.timeout(self.timeout_s):
                answer = await self._side.ask(prompt)
        except Exception as exc:  # a timeout too
            log.info("addressee: model gave no answer (%s); rules decide", type(exc).__name__)
            return rules
        match = _YES_NO.search(answer.lower())
        if match is None:
            return rules
        return Verdict(match.group(1) == "yes", f"model said {match.group(1)}", by="model")
