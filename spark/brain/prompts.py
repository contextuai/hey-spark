"""Voice system prompt, the speech sanitizer, and "new topic" detection (spec §8.1, §8.2).

Everything here is pure Python with no heavy imports, so the app, ``--text`` mode, and tests can
use it freely.

``sanitize_for_speech`` is the safety net behind the prompt: models still leak markdown, URLs,
IDs, and emoji now and then, and a TTS engine would read them aloud literally. It is applied per
sentence (after :mod:`spark.util.sentences`), so it must be safe on partial text (an unmatched
``**`` or ``[``), and it is idempotent: ``s(s(x)) == s(x)``.

Symbol policy (kept consistent everywhere):

* ``&`` -> "and", ``%`` -> "percent", ``+`` between operands -> "plus", ``=`` -> "equals",
  ``@`` -> "at", ``#5`` -> "number 5", ``~5`` -> "about 5", ``$5`` -> "5 dollars",
  ``5 * 3`` -> "5 times 3", ``->``/``→`` -> "to", ``°`` -> "degrees".
* ``/`` between two words -> " or " (``yes/no`` -> "yes or no"); ``/`` between digits
  (dates, fractions) is left for the TTS engine.
* Anything that is only markup (``* _ ` # | [ ] { } < > ^ \\ ~``) is removed.

ID policy (for raw tool results that leak into a reply):

* Removed entirely: URLs, email addresses, UUIDs, hex hashes of 7+ chars with a digit and a
  letter (git SHAs), 16+ char letter-and-digit tokens (ULIDs, base64-ish IDs), GitHub node IDs
  (``PR_kwDO...``), JWTs, and credentials with a known prefix (``ghp_``, ``sk-``, ``xoxb-``...).
* Kept, because people say them: issue and PR numbers (``#1234`` -> "number 1234", and
  ``acme/api#12`` -> "number 12"), ticket keys like ``PROJ-123``, plain numbers and dates.
  ISO date-times keep only the date (``2026-09-24T10:15:00Z`` -> ``2026-09-24``).
* Code in triple backticks is dropped, whether fenced over lines or inline on one line.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

__all__ = [
    "CONNECTOR_NOTES_RULES",
    "FRESH_START_ACK",
    "MARKDOWN_CHARS",
    "NO_TOOLS_NOTE",
    "TOOL_RULES",
    "VOICE_RULES",
    "VOICE_SYSTEM_PROMPT",
    "WORKER_RULES",
    "SpeechFilter",
    "is_new_topic",
    "sanitize_for_speech",
    "split_new_topic",
    "voice_system_prompt",
    "worker_rules",
]

# --------------------------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------------------------

VOICE_RULES = """\
You are Spark, a voice assistant running on the user's laptop. You are speaking, not writing: \
every word you produce is read aloud by a text-to-speech engine, so write exactly what should \
be heard.

How to answer:
- Answer in one to three short spoken sentences unless the user asks for more detail.
- Lead with the answer. No preamble, no restating the question, no sign-off.
- Use plain conversational sentences only. Never use markdown, headings, bold, italics, bullet \
points, numbered lists, tables, code, code blocks, links, URLs, email addresses, or emoji.
- Spell out symbols the way a person would say them: say "and" instead of an ampersand, \
"percent" instead of the percent sign, "plus", "equals", "degrees", "dollars". Write "yes or \
no" instead of "yes slash no". Say numbers and times naturally, for example "about three \
thirty in the afternoon".
- When a list is unavoidable, say how many items there are and name at most three in a \
sentence, then offer the rest.
- When you use a tool or look something up, summarize the result in one to three spoken \
sentences. Never read raw data, IDs, hashes, file paths, URLs, email addresses, or long lists \
aloud. Offer more, for example "Want the details?"
- If a request is ambiguous, ask one short clarifying question instead of guessing.
- Before any action with side effects, such as creating, sending, deleting, or changing \
something, the user must hear exactly what will happen and "Should I go ahead?", and the action \
only proceeds after a clear yes. For actions done through a tool, Spark does this step for you \
automatically at the moment you call the tool, so call it directly and do not ask first yourself.
- When a request takes several steps, such as starting a project, making a plan, or \
summarizing a week, finish with one short sentence that sums up what you did or found. Spark \
may already have told the user "On it, give me a moment" while you worked, so don't say that \
yourself.
- Work quietly. Don't narrate what you are about to do or which step you are on ("let me \
check", "let me search the tables", "let me try a query"), and don't ask permission to try \
something; just do it. Everything you write is spoken, so speak only the result. Spark already \
tells the user "On it, give me a moment" when the work takes a while.
- If you can't finish without something from the user, ask one short question at the end, \
after saying briefly what you found.
- If you don't know something or can't do it, say so in one sentence.
- Your replies may be interrupted. Keep them short so the user can follow up."""

TOOL_RULES = """\
Tools:
- You have tools for the current time, timers, your own status, and whichever services are \
connected, such as calendar, email, or code repositories. When the answer depends on live data, \
use a tool instead of guessing. If no tool covers the request, say briefly that you can't do \
that yet.
- Tool results are raw data for you, not text to read out. Turn them into one to three short \
spoken sentences: give the count first, then name the top one or two items by title, person, or \
time, then offer the rest. For example: "You have five open pull requests. The newest is Fix \
the login timeout, from Sam. Want the others?"
- Never say IDs, node IDs, commit hashes, URLs, links, email addresses, tokens, file paths, or \
raw timestamps, even if the tool returned them. Refer to things by name, by person, or by a \
short number, such as "pull request 42". Say dates and times the way a person would, such as \
"tomorrow at nine" or "yesterday afternoon".
- If there is nothing, say so plainly, for example "No unread email from Sam."
- Tool actions are confirmed for you. When you call a tool that creates, sends, or changes \
something, Spark itself tells the user exactly what will happen and asks "Should I go ahead?" \
before the tool runs. So when the user asks for such an action, call the tool right away with \
the details they gave. Don't ask for confirmation in your own words first, or they will be \
asked twice.
- If the user says no to an action, or doesn't answer, the tool call comes back refused. \
Acknowledge it in a few words, such as "Okay, I won't.", and don't try it again.
- If a tool fails or is not allowed, say so in one short sentence in plain words and don't \
retry it more than once. If a tool isn't available, tell the user you can't do that yet."""

#: Added when Spark has memory tools and at least one external connector (P5.9).
CONNECTOR_NOTES_RULES = """\
Connector notes:
- You keep short notes about each connected system with save_connector_note. Whenever you had \
to explore a system to answer (list databases, schemas or tables, describe a table, try several \
queries), save one or two notes before you answer: where that kind of data lives (database, \
schema, table, the key columns) and the query or filter that worked. Structure only, never the \
user's data values, people's details, or secrets.
- Your notes from earlier come with the conversation under "What you already know about the \
connected systems". Use them first and query directly. If one turns out wrong, fix it with \
rewrite_connector_notes.
- Don't mention the notes to the user unless they ask."""

#: Added when the researcher worker is on (P5.7d). ``{helper}`` names how the voice model starts
#: it: the Agent tool on Claude, the research function on other providers (P5.7e).
_WORKER_RULES = """\
Your researcher:
- You have a researcher helper ({helper}) that runs on a faster model \
with read-only access to the connected systems and your connector notes. Use it for any lookup \
that needs exploring or more than one or two tool calls: finding where data lives, running and \
refining queries, gathering items from several places. Give it a precise brief: what to find, \
for which time range or filter, and what to report.
- It reports facts back to you, not to the user. Check that the report answers the question, \
then answer in your own short spoken words. If it came back empty or unsure, say so plainly.
- Do single quick lookups and every action that changes something yourself."""


def worker_rules(helper: str = 'the Agent tool, agent "researcher"') -> str:
    """The researcher section of the voice prompt, naming how to start the helper."""
    return _WORKER_RULES.format(helper=helper)


WORKER_RULES = worker_rules()

#: Kept for a brain built with no tools at all (no MCP servers and no in-process server).
NO_TOOLS_NOTE = """\
Right now you have no tools: you can't check calendars, email, files, code repositories, the \
web, or the current time. Answer from what you know. If asked for something that needs a tool, \
say briefly that you can't do that yet."""


#: The acknowledgement the voice rules quote (what Spark says when a turn runs long).
DEFAULT_ACK = "On it, give me a moment"


def voice_system_prompt(
    *,
    tools_available: bool = True,
    extra: str = "",
    name: str = "Spark",
    persona: str = "",
    ack: str = "",
) -> str:
    """Build the voice system prompt: the §8.2 rules plus the tool-result rules.

    ``tools_available=False`` swaps the tool rules for :data:`NO_TOOLS_NOTE` (a brain with no
    servers at all). ``name`` (``app.name``) replaces "Spark" as the assistant's name.
    ``persona`` is the personality section (spark/persona.py), placed after the rules, and
    ``ack`` the acknowledgement Spark really says, quoted in the rules so the model doesn't
    repeat it."""
    rules = VOICE_RULES
    ack = ack.strip().rstrip(".!")
    if ack and ack != DEFAULT_ACK:
        rules = rules.replace(f'"{DEFAULT_ACK}"', f'"{ack}"')
    parts = [rules, TOOL_RULES if tools_available else NO_TOOLS_NOTE]
    if extra.strip():
        parts.append(extra.strip())
    if persona.strip():
        parts.append(persona.strip())
    prompt = "\n\n".join(parts)
    return prompt if name == "Spark" else re.sub(r"\bSpark\b", name, prompt)


#: Phase 2 prompt: all §8.2 rules, named Spark, plus how to speak tool results.
VOICE_SYSTEM_PROMPT = voice_system_prompt()

FRESH_START_ACK = "Okay, fresh start."

# --------------------------------------------------------------------------------------------
# sanitize_for_speech
# --------------------------------------------------------------------------------------------

#: Characters that must never reach TTS (used by tests and the recorded-transcript check).
MARKDOWN_CHARS = frozenset("*_`#|[]{}<>^\\~")

# ```code``` on one line is code too: drop it whole (before _FENCED_BLOCK, which would otherwise
# treat the opening fence as the start of a block running to the next line's fence).
_INLINE_FENCE = re.compile(r"(```+|~~~+)[^\n]*?\1")
_FENCED_BLOCK = re.compile(r"(```|~~~)[^\n]*\n.*?(?:\1|\Z)", re.DOTALL)
_FENCE_MARKER = re.compile(r"(```+|~~~+)[\w+.-]*")
_IMAGE = re.compile(r"!\[([^\]\n]*)\]\([^)\n]*\)?")
_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)?")
_REF_LINK = re.compile(r"\[([^\]\n]+)\]\[[^\]\n]*\]")
_AUTOLINK = re.compile(r"<(?:https?://|mailto:)[^>\s]*>?", re.IGNORECASE)
# URLs stop before trailing sentence punctuation so "see https://x.io." keeps its full stop.
_URL_TAIL = r"(?:[^\s<>\"')\]]*[^\s<>\"')\].,;:!?])?"
_URL = re.compile(r"(?:\b(?:https?|ftp)://|\bwww\.)" + _URL_TAIL, re.IGNORECASE)
_PATH_URL = re.compile(
    r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|ai|app|co|gov|edu)/" + _URL_TAIL,
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
# Hex with at least one digit and one letter, 7+ chars: short and full git SHAs, hashes.
_HEX_ID = re.compile(r"\b(?:0x)?(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}\b", re.IGNORECASE)
# 16+ word chars mixing letters and digits: ULIDs, base64-ish IDs, msg_01..., MDExOl... node IDs.
_TOKEN_ID = re.compile(r"\b(?=\w*\d\w*\d)(?=\w*[A-Za-z]\w*[A-Za-z])\w{16,}\b")
# GitHub GraphQL node IDs: PR_kwDO..., I_kwDO..., R_kgDO..., IC_kwDO...
_NODE_ID = re.compile(r"\b[A-Z]{1,4}_[a-z]{2}[A-Z]{2}[A-Za-z0-9_-]{2,}")
# Credentials by their well-known prefixes (never speak a token, even a partial one).
_API_TOKEN = re.compile(
    r"\b(?:sk|pk|rk|ghp|gho|ghu|ghs|ghr|github_pat|glpat|xox[abpors])[-_][A-Za-z0-9_-]{6,}"
    r"|\bAKIA[0-9A-Z]{12,}\b"
)
_JWT = re.compile(r"\beyJ[\w-]{4,}\.[\w-]{4,}\.[\w-]*")
# "acme/api#412" -> "#412" (spoken "number 412"); the slug itself is noise.
_REPO_REF = re.compile(r"\b[\w.-]+/[\w.-]+#(?=\d)")
# ISO 8601 date-times keep only the date: "2026-09-24T10:15:00Z" -> "2026-09-24".
_ISO_DATETIME = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?(?![\w:])"
)
_HEADER = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}(?:>\s?)+", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+•▪◦‣]\s+(?:\[[ xX]\]\s+)?", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d{1,3}[.)]\s+", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_HRULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_ITALIC_UNDER = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_STRIKE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_SNAKE = re.compile(r"(?<=[A-Za-z0-9])_+(?=[A-Za-z0-9])")
_TIMES = re.compile(r"(?<=\d)\s*[*×]\s*(?=\d)")
_ARROW = re.compile(r"\s*(?:->|=>|→|⟶|➔|➜)\s*")
_CPP = re.compile(r"\b([Cc])\+\+")
_PLUS = re.compile(r"(?<=[\w)])\s*\+\s*(?=[\w($])")
_LEADING_PLUS = re.compile(r"(?<![\w)])\+(?=\d)")
_EQUALS = re.compile(r"(?<=[\w)])\s*={1,3}\s*(?=[\w($-])")
_PERCENT = re.compile(r"\s*%")
_DOLLARS = re.compile(r"\$(\d[\d,]*(?:\.\d+)?)(\s?(?:[kKmMbB]|million|billion|thousand)\b)?")
_POUNDS = re.compile(r"£(\d[\d,]*(?:\.\d+)?)")
_EUROS = re.compile(r"€(\d[\d,]*(?:\.\d+)?)")
_NUMBER_SIGN = re.compile(r"#(?=\d)")
_ABOUT = re.compile(r"(?<!\w)~\s?(?=\d)")
_DEGREES = re.compile(r"\s*°\s*([CF])\b|\s*°")
_SLASH_WORDS = re.compile(r"(?<=[A-Za-z])\s*/\s*(?=[A-Za-z])")
_AMP = re.compile(r"\s*&+\s*")
_AT = re.compile(r"\s+@\s+|(?<=\w)@(?=\w)")
_LESS_GREATER = re.compile(r"(?<=\d)\s*(<=|>=|<|>)\s*(?=\d)")
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, symbols & pictographs ext.
    "\U0001fb00-\U0001fbff"
    "☀-➿"  # misc symbols, dingbats (☀ ✅ ✨ ❤)
    "⬀-⯿"  # arrows / stars (⭐ ⬆)
    "⌀-⏿"  # misc technical (⌛ ⏰)
    "■-◿"  # geometric shapes (▶ ◀ ▪)
    "←-⇿"  # arrows
    "⤀-⥿"
    "〰〽㊗㊙"
    "©®™"  # © ® ™
    "︀-️"  # variation selectors
    "‍"  # zero-width joiner
    "⃣"  # keycap
    "\U000e0020-\U000e007f"  # tag characters (flags)
    "]+"
)
_LEFTOVER = re.compile(r"[*_`#|\[\]{}<>^\\~]")
_EMPTY_BRACKETS = re.compile(r"\(\s*[,;:]?\s*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")
_SPACE_AFTER_OPEN = re.compile(r"\(\s+")
_OPEN_THEN_PUNCT = re.compile(r"\(\s*[,;:](?:\s*[,;:])*\s*")
_DUP_PUNCT = re.compile(r"([,;:])(?:\s*[,;:])+")
_LEADING_PUNCT = re.compile(r"^(?:[\s,;:.)]|[-–—](?=\s))+(?=\S)")
_ORPHAN_PUNCT = re.compile(r"^[\s,;:.)\-–—!?]+$")
_WS = re.compile(r"[ \t\r\f\v]+")
_NEWLINES = re.compile(r"\s*\n\s*")


def _dollars(m: re.Match[str]) -> str:
    return f"{m.group(1)}{m.group(2) or ''} dollars"


def _newline(m: re.Match[str]) -> str:
    """A line break becomes a sentence break unless the line already ended with punctuation."""
    before = m.string[: m.start()].rstrip()
    if not before or before[-1] in ".!?:;,":
        return " "
    return ". "


def sanitize_for_speech(text: str) -> str:
    """Make ``text`` safe to hand to TTS: no markdown, URLs, emails, IDs, or emoji.

    Safe on partial sentences and idempotent. Newlines are turned into sentence breaks.
    """
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")

    # 1. Code and structure.
    s = _INLINE_FENCE.sub(" ", s)
    s = _FENCED_BLOCK.sub(" ", s)
    s = _FENCE_MARKER.sub(" ", s)
    s = _TABLE_RULE.sub("", s)
    s = _HRULE.sub("", s)
    s = _HEADER.sub("", s)
    s = _QUOTE.sub("", s)
    s = _BULLET.sub("", s)
    s = _NUMBERED.sub("", s)

    # 2. Links, URLs, addresses, IDs (before emphasis so underscores in URLs don't matter).
    s = _IMAGE.sub(r"\1", s)
    s = _LINK.sub(r"\1", s)
    s = _REF_LINK.sub(r"\1", s)
    s = _AUTOLINK.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _PATH_URL.sub(" ", s)
    s = _EMAIL.sub(" ", s)
    s = _REPO_REF.sub("#", s)
    s = _ISO_DATETIME.sub(r"\1", s)
    s = _JWT.sub(" ", s)
    s = _API_TOKEN.sub(" ", s)
    s = _NODE_ID.sub(" ", s)
    s = _UUID.sub(" ", s)
    s = _HEX_ID.sub(" ", s)
    s = _TOKEN_ID.sub(" ", s)

    # 3. Emphasis and inline code.
    s = _BOLD.sub(r"\2", s)
    s = _STRIKE.sub(r"\1", s)
    s = _ITALIC_STAR.sub(r"\1", s)
    s = _ITALIC_UNDER.sub(r"\1", s)
    s = s.replace("`", "")
    s = _SNAKE.sub(" ", s)

    # 4. Symbols spoken as words.
    s = _ARROW.sub(" to ", s)
    s = _TIMES.sub(" times ", s)
    s = _CPP.sub(r"\1 plus plus", s)
    s = _LEADING_PLUS.sub("plus ", s)
    s = _PLUS.sub(" plus ", s)
    s = _LESS_GREATER.sub(
        lambda m: {
            "<": " less than ",
            ">": " greater than ",
            "<=": " at most ",
            ">=": " at least ",
        }[m.group(1)],
        s,
    )
    s = _EQUALS.sub(" equals ", s)
    s = _PERCENT.sub(" percent", s)
    s = _DOLLARS.sub(_dollars, s)
    s = _POUNDS.sub(lambda m: f"{m.group(1)} pounds", s)
    s = _EUROS.sub(lambda m: f"{m.group(1)} euros", s)
    s = _NUMBER_SIGN.sub("number ", s)
    s = _ABOUT.sub("about ", s)
    s = _DEGREES.sub(lambda m: f" degrees {m.group(1)}" if m.group(1) else " degrees", s)
    s = _SLASH_WORDS.sub(" or ", s)
    s = _AMP.sub(" and ", s)
    s = _AT.sub(" at ", s)

    # 5. Emoji and leftover markup characters.
    s = _EMOJI.sub(" ", s)
    s = _LEFTOVER.sub(" ", s)
    for ch in "=+$@•":
        s = s.replace(ch, " ")

    # 6. Tidy whitespace and punctuation left behind by removals.
    s = _NEWLINES.sub(_newline, s)
    s = _WS.sub(" ", s)
    s = _EMPTY_BRACKETS.sub(" ", s)
    s = _OPEN_THEN_PUNCT.sub("(", s)
    s = _SPACE_AFTER_OPEN.sub("(", s)
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _DUP_PUNCT.sub(r"\1", s)
    s = _WS.sub(" ", s).strip()
    s = _LEADING_PUNCT.sub("", s)
    if _ORPHAN_PUNCT.match(s):
        return ""
    return s


class SpeechFilter:
    """Stateful per-sentence sanitizer that also drops lines inside code fences.

    The sentence splitter cuts at newlines, so a fenced block reaches TTS one line at a time and
    :func:`sanitize_for_speech` alone can't tell that ``print(x)`` sits inside a fence. Feed
    sentences in order; ``feed`` returns the speakable text ('' to skip).
    """

    def __init__(self) -> None:
        self._in_fence = False

    def feed(self, sentence: str) -> str:
        stripped = sentence.strip()
        fences = len(re.findall(r"```|~~~", stripped))
        if self._in_fence:
            if fences % 2 == 1:
                self._in_fence = False
                tail = re.split(r"```|~~~", stripped)[-1]
                return sanitize_for_speech(tail)
            return ""
        if fences % 2 == 1:
            head = re.split(r"```|~~~", stripped)[0]
            self._in_fence = True
            return sanitize_for_speech(head)
        return sanitize_for_speech(sentence)

    def reset(self) -> None:
        self._in_fence = False

    def filter(self, sentences: Iterable[str]) -> list[str]:
        """Sanitize a sequence of sentences, dropping ones that end up empty."""
        return [out for s in sentences if (out := self.feed(s))]


# --------------------------------------------------------------------------------------------
# "new topic"
# --------------------------------------------------------------------------------------------

_SEP = r"[\s,.:;!?\-–—]*"


def _names_alt(names: Sequence[str]) -> str:
    return "|".join(re.escape(n) for n in names if n) or "spark"


def _prefix(names: Sequence[str]) -> str:
    return (
        r"^\s*(?:(?:hey|hi|ok|okay|yo)\s*,?\s+)?"
        r"(?:(?:" + _names_alt(names) + r")\b" + _SEP + r")?"
        r"(?:(?:ok|okay|so|um|uh|alright|all right)\b" + _SEP + r")?"
        r"(?:(?:let'?s|let\s+us|please|can\s+we|could\s+we)\s+)?"
        r"(?:(?:do|have|start|begin)\s+(?:a\s+)?)?"
    )


def _phrase_pattern(phrase: str) -> str:
    words = [re.escape(w) for w in re.findall(r"[\w']+", phrase.lower())]
    return r"[\s,\-]+".join(words)


def split_new_topic(
    text: str, phrases: Sequence[str], names: Sequence[str] = ("spark",)
) -> tuple[bool, str]:
    """Detect a "new topic" command at the start of ``text``.

    Returns ``(matched, rest)`` where ``rest`` is whatever the user said after the phrase
    (``"Spark, new topic. What's the weather?"`` -> ``(True, "What's the weather?")``).
    Matching is case-insensitive and tolerates "Spark," / "hey Spark" / "let's" / "please".
    ``names`` are the lowercase assistant names accepted in place of "spark" (``app.wake_names``).
    """
    alts = [_phrase_pattern(p) for p in phrases if p.strip()]
    if not alts or not text or not text.strip():
        return False, text
    pattern = re.compile(
        _prefix(names)
        + r"(?:"
        + "|".join(alts)
        + r")\b"
        + _SEP
        + r"(?:(?:please|now|thanks|thank\s+you|"
        + _names_alt(names)
        + r")\b"
        + _SEP
        + r")*(.*)$",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.match(text)
    if m is None:
        return False, text
    rest = m.group(1).strip()
    if not re.search(r"\w", rest):
        rest = ""
    return True, rest


def is_new_topic(text: str, phrases: Sequence[str], names: Sequence[str] = ("spark",)) -> bool:
    """True if ``text`` starts with (or is) a "new topic" phrase such as "Spark, new topic"."""
    return split_new_topic(text, phrases, names)[0]
