"""Personalities: how Spark talks (``app.persona``), and the voices it can talk in.

A persona is two things:

* a section added to the voice system prompt (the model's own replies), and
* Spark's fixed lines, the ones the model never writes: the "On it" acknowledgement, the
  replies to "go to sleep" or "new conversation", "Should I go ahead?", timer announcements,
  error lines, and the optional spoken reply to the wake word.

``standard`` is Spark as it always was, so its lines are exactly the old fixed strings. ``butler``
is a composed British butler with a dry wit who calls you "sir" (``app.address_as``).

In the phrases, ``{addr}`` becomes ", sir" (", " plus ``app.address_as``) or nothing when that is
empty, ``{name}`` is ``app.name`` and ``{timer}`` the timer's spoken name. Pure Python, no heavy
imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spark.config import Settings

__all__ = [
    "DEFAULT_GO_AHEAD",
    "DEFAULT_REPROMPT",
    "DEFAULT_VOICE",
    "PERSONAS",
    "VOICES",
    "Persona",
    "Voice",
    "Voices",
    "persona_for",
    "voice_by_id",
    "voice_id",
]

DEFAULT_GO_AHEAD = "Should I go ahead?"
DEFAULT_REPROMPT = "Sorry, was that a yes or a no?"
DEFAULT_VOICE = "kokoro:bm_george"

_STANDARD_PHRASES = {
    "ack": "On it, give me a moment.",
    "sleep": "Okay.",
    "fresh_start": "Okay, fresh start.",
    "muted": "Muted. Use the mic button to unmute me.",
    "go_ahead": DEFAULT_GO_AHEAD,
    "reprompt": DEFAULT_REPROMPT,
    "timer_done": "Your {timer} is done.",
    "too_many_steps": (
        "Sorry, that needed more steps than I'm allowed. Try asking something narrower."
    ),
    "too_long": "Sorry, that took too long. Try again in a moment.",
    "broken": "Sorry, part of me isn't working right now. The details are in the log.",
    "failed": "Sorry, something went wrong with that.",
    "wake": "Yes?",
    "sample": "Hi, I'm {name}. How can I help?",
}

_BUTLER_PHRASES = {
    "ack": "Right away{addr}. One moment.",
    "sleep": "Very good{addr}.",
    "fresh_start": "Of course{addr}. A clean slate.",
    "muted": "Muted{addr}. The mic button will bring me back.",
    "go_ahead": "Shall I proceed{addr}?",
    "reprompt": "I beg your pardon{addr}. Was that a yes or a no?",
    "timer_done": "Your {timer} has finished{addr}.",
    "too_many_steps": (
        "My apologies{addr}, that needed more steps than I'm permitted. Perhaps something narrower."
    ),
    "too_long": "My apologies{addr}, that took rather too long. Do try again in a moment.",
    "broken": (
        "My apologies{addr}, part of me isn't working at the moment. The details are in the log."
    ),
    "failed": "My apologies{addr}, something went wrong there.",
    "wake": "Yes{addr}?",
    "sample": "Good evening{addr}. {name}, at your service. Shall I check your calendar?",
}

_BUTLER_PROMPT = """\
Personality:
- You speak as a composed, impeccably polite British butler with a dry, understated wit: calm, \
unflappable, quietly confident, never servile or gushing.
- {address_rule}
- A touch of deadpan humour is welcome when it fits, never at the expense of the answer and \
never more than a few words.
- Be quietly anticipatory: when there is an obvious next step, offer it in a few words, for \
example "Shall I set a reminder?"
- Prefer understatement and British phrasing: "Very good", "Right away", "I'm afraid", "rather".
- Every rule above still applies: short spoken sentences, the answer first, no lists or \
markdown."""


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    description: str
    phrases: dict[str, str]
    prompt: str = ""  # added to the voice system prompt; {address_rule} filled in
    voice: str = ""  # the voice the settings window suggests for it (a VOICES id)
    wake_reply: bool = False  # say the "wake" phrase instead of the wake chime by default


PERSONAS: dict[str, Persona] = {
    "standard": Persona(
        key="standard",
        label="Standard",
        description="Friendly, plain and brief.",
        phrases=_STANDARD_PHRASES,
    ),
    "butler": Persona(
        key="butler",
        label="Butler",
        description="A composed British butler with a dry wit, who calls you sir.",
        phrases=_BUTLER_PHRASES,
        prompt=_BUTLER_PROMPT,
        voice=DEFAULT_VOICE,
        wake_reply=True,
    ),
}


class _Blank(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class Voices:
    """The persona applied to one :class:`~spark.config.Settings`: ready-to-speak lines."""

    persona: Persona
    name: str = "Spark"
    address: str = ""
    wake_reply_on: bool = False
    _cache: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def line(self, key: str, **values: str) -> str:
        """The persona's line ``key``, filled in. Unknown keys fall back to the standard line."""
        template = self.persona.phrases.get(key) or _STANDARD_PHRASES.get(key, "")
        addr = f", {self.address}" if self.address else ""
        return template.format_map(_Blank(addr=addr, name=self.name, **values))

    @property
    def wake_reply(self) -> str:
        """What to say when the wake word fires, or '' for just the chime."""
        return self.line("wake") if self.wake_reply_on else ""

    def prompt(self) -> str:
        """The persona's section of the voice system prompt ('' for none)."""
        if not self.persona.prompt:
            return ""
        if self.address:
            rule = (
                f'Address the user as "{self.address}" now and then, usually once in a reply '
                "and never in every sentence."
            )
        else:
            rule = "Don't use a title or name for the user."
        return self.persona.prompt.format(address_rule=rule)

    def confirm_line(self, text: str) -> str:
        """A confirmation line in the persona's words: the standard "Should I go ahead?" ending
        and the "yes or no?" reprompt are swapped for the persona's own."""
        if text == DEFAULT_REPROMPT:
            return self.line("reprompt")
        if text.endswith(DEFAULT_GO_AHEAD):
            return text[: -len(DEFAULT_GO_AHEAD)] + self.line("go_ahead")
        return text


def persona_for(settings: Settings) -> Voices:
    app = settings.app
    persona = PERSONAS.get(app.persona, PERSONAS["standard"])
    wake = persona.wake_reply if app.wake_reply is None else app.wake_reply
    return Voices(persona, name=app.name, address=app.address_as.strip(), wake_reply_on=wake)


# --- voices ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Voice:
    id: str  # "<backend>:<voice>"
    backend: str  # tts backend: piper | kokoro
    voice: str
    label: str
    accent: str
    size: str  # one-off download
    license: str


#: Offered in the Personality & Voice window: the Kokoro voices (one 340 MB download for all of
#: them; George is the default). Any other voice, Piper's included, can still be set in
#: config.yaml.
VOICES: tuple[Voice, ...] = (
    Voice("kokoro:bm_george", "kokoro", "bm_george", "George (Kokoro)",
          "British male", "340 MB", "Apache-2.0"),
    Voice("kokoro:bm_fable", "kokoro", "bm_fable", "Fable (Kokoro)",
          "British male", "340 MB", "Apache-2.0"),
    Voice("kokoro:bm_lewis", "kokoro", "bm_lewis", "Lewis (Kokoro)",
          "British male", "340 MB", "Apache-2.0"),
    Voice("kokoro:bm_daniel", "kokoro", "bm_daniel", "Daniel (Kokoro)",
          "British male", "340 MB", "Apache-2.0"),
    Voice("kokoro:bf_emma", "kokoro", "bf_emma", "Emma (Kokoro)",
          "British female", "340 MB", "Apache-2.0"),
    Voice("kokoro:af_heart", "kokoro", "af_heart", "Heart (Kokoro)",
          "American female", "340 MB", "Apache-2.0"),
    Voice("kokoro:am_michael", "kokoro", "am_michael", "Michael (Kokoro)",
          "American male", "340 MB", "Apache-2.0"),
)  # fmt: skip


def voice_by_id(vid: str) -> Voice | None:
    return next((v for v in VOICES if v.id == vid), None)


def voice_id(settings: Settings) -> str:
    """The configured voice as a VOICES-style id (it may not be in VOICES)."""
    backend = settings.tts.backend
    if backend == "piper":
        return f"piper:{settings.tts.options.piper.voice}"
    if backend == "kokoro":
        return f"kokoro:{settings.tts.options.kokoro.voice}"
    return backend
