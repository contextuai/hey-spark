"""Spoken confirmation for side-effecting tools: the ``PreToolUse`` hook and the Confirmer interface
(spec §7 "Tool permissions", §8.3; Phase 3 "every side-effecting action is confirmed by voice, and
saying 'no' prevents it").

How a tool call is decided (``ToolPolicy.classify`` from :mod:`spark.brain.mcp_config`):

* ``read``: passes with ``{}``. The CLI's own permission rules then run the tool, because every
  read-only tool is in ``allowed_tools``.
* ``side_effect`` (and ``destructive`` on a server with ``allow_destructive: true``): Spark says one
  line ("I'll create a GitHub repo called weather-bot. Should I go ahead?") and waits for yes or no.
  A yes returns an explicit ``permissionDecision: "allow"``. ``{}`` would **not** be enough:
  verified live with SDK 0.2.159 / CLI 2.1.281, a tool outside ``allowed_tools`` whose hook
  returned ``{}`` was refused by the CLI ("Claude requested permissions to use ..., but you
  haven't granted it yet") and never ran (docs/sdk_notes.md, P3.1 section). A no, or no answer,
  returns a deny whose reason the model reads back to the user.
* ``destructive`` (by default) and ``unknown``: denied without asking.

Fast route (spec §8.4 "fast path ... with a limited tool set"): while the brain runs a turn routed
to the fast model (``route_provider()`` returns ``"fast"``), a tool outside the fast tool set is
denied **without asking**, with :data:`REASON_FAST_ROUTE`, and logged as
``tool denied: <name> (fast route)``. The fast set is ``routing.fast_tools`` when it lists names,
else every read-only tool. ``allowed_tools`` can't change per turn without reconnecting (it is a
spawn-time CLI flag, docs/sdk_notes.md "Routing (P3.2)"), so the limit is enforced here, per call,
and the one session keeps its context. Deep turns, turns outside routing and a disabled router
(``route_provider()`` returns None) are unaffected.

Defaults are fail-closed. A confirmer error, a timeout, or an unclear answer all count as "no".
If the hook itself overruns ``HookMatcher.timeout``, the CLI cancels it and does not run the tool
(verified live).

Only tool **names** are logged, never arguments (they can hold personal data).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, TextIO

from spark.brain.mcp_config import ToolPolicy, split_tool_name
from spark.brain.prompts import sanitize_for_speech

log = logging.getLogger("spark.brain.tools")

__all__ = [
    "DENY_REASONS",
    "REASON_ALLOWED",
    "REASON_DECLINED",
    "REASON_FAST_ROUTE",
    "REASON_NO_ANSWER",
    "REASON_STEP_BUDGET",
    "ConfirmationAborted",
    "ConfirmationGate",
    "Confirmer",
    "HookCallback",
    "LineReader",
    "Summarizer",
    "TextConfirmer",
    "VoiceConfirmer",
    "allow",
    "classify_yes_no",
    "deny",
    "generic_summary",
    "hook_verdict",
    "humanize_tool_name",
    "speakable_value",
    "summarize_tool_call",
]

HookCallback = Callable[[dict[str, Any], "str | None", Any], Awaitable[dict[str, Any]]]
Summarizer = Callable[[Mapping[str, Any]], "str | None"]

# --- hook outputs ------------------------------------------------------------------------------

#: Reasons the model reads back to the user, so they are phrased to be relayed, not logged.
REASON_DECLINED = "The user said no. Don't do it; acknowledge briefly."
REASON_NO_ANSWER = "No confirmation was heard, so I didn't do it."
REASON_ALLOWED = "The user confirmed this action out loud."
REASON_FAST_ROUTE = (
    "Not done: that needs a bigger step than a quick answer. Don't try another tool. Tell the "
    "user in one short sentence that it's a multi-step task and to just ask again, the same "
    "way; no extra detail is needed."
)
REASON_STEP_BUDGET = (
    "Not run: this request has used all the steps it is allowed. Don't call any more tools. "
    "Tell the user in one or two short sentences what you found so far, and offer to keep "
    "digging if they want."
)

DENY_REASONS: dict[str, str] = {
    "side_effect": (
        "Blocked: this action would change something, and Spark has no way to confirm it right "
        "now. Tell the user you can't do that right now."
    ),
    "destructive": (
        "Blocked: this kind of action is turned off in Spark's settings. Tell the user you can't "
        "do that."
    ),
    "unknown": (
        "Blocked: this tool isn't on Spark's allowlist. Tell the user you can't do that yet."
    ),
}


def deny(reason: str) -> dict[str, Any]:
    """The ``PreToolUse`` deny output (``PreToolUseHookSpecificOutput``, types.py:437)."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def allow(reason: str = REASON_ALLOWED) -> dict[str, Any]:
    """The explicit ``PreToolUse`` allow output. It is required for tools outside
    ``allowed_tools``: ``{}`` leaves them to the CLI's permission check, which refuses them."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
        }
    }


def hook_verdict(output: Mapping[str, Any] | None, *, preapproved: bool) -> tuple[bool, str | None]:
    """What the CLI would do with a ``PreToolUse`` output, for brains that run tools
    themselves (P5.7e): ``(run it, deny reason)``. An explicit allow runs, a deny doesn't, and
    no decision (``{}``) defers to ``allowed_tools``: it runs only when ``preapproved``."""
    spec = (output or {}).get("hookSpecificOutput")
    decision = spec.get("permissionDecision") if isinstance(spec, Mapping) else None
    if decision == "deny":
        reason = spec.get("permissionDecisionReason") if isinstance(spec, Mapping) else None
        return False, str(reason or DENY_REASONS["unknown"])
    if decision == "allow" or preapproved:
        return True, None
    return False, DENY_REASONS["unknown"]


# --- yes / no ----------------------------------------------------------------------------------

YES_PHRASES: tuple[str, ...] = (
    "yes",
    "yeah",
    "yea",
    "yep",
    "yup",
    "ya",
    "sure",
    "go ahead",
    "go for it",
    "do it",
    "please do",
    "ok",
    "okay",
    "alright",
    "all right",
    "affirmative",
    "confirm",
    "confirmed",
    "absolutely",
    "definitely",
    "of course",
    "sounds good",
)
NO_PHRASES: tuple[str, ...] = (
    "no",
    "nope",
    "nah",
    "don't",
    "dont",
    "do not",
    "stop",
    "cancel",
    "wait",
    "not now",
    "not yet",
    "never mind",
    "nevermind",
    "hold on",
    "hold off",
    "negative",
    "abort",
    "forget it",
    "skip it",
)
#: Yes-idioms that contain a "no" word. They are masked before the no-scan.
YES_IDIOMS: tuple[str, ...] = ("no problem", "no worries", "not a problem", "why not")
#: Hedges. They are masked before the no-scan, and they make an otherwise positive answer unclear.
UNSURE_PHRASES: tuple[str, ...] = (
    "not sure",
    "don't know",
    "dont know",
    "do not know",
    "maybe",
    "i guess",
    "perhaps",
    "not certain",
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def _normalize(text: str) -> str:
    s = (text or "").lower().translate(_APOSTROPHES)
    s = _NON_WORD.sub(" ", s)
    return " " + " ".join(s.split()) + " "


def _phrase_re(phrases: Iterable[str]) -> re.Pattern[str]:
    words = sorted({" ".join(p.lower().translate(_APOSTROPHES).split()) for p in phrases if p})
    words = [w for w in words if w]
    if not words:
        return re.compile(r"(?!x)x")  # matches nothing
    alts = "|".join(r"\s+".join(re.escape(part) for part in w.split()) for w in words)
    return re.compile(rf"(?<![\w'])(?:{alts})(?![\w'])")


class _YesNo:
    def __init__(self, extra_yes: Sequence[str] = (), extra_no: Sequence[str] = ()) -> None:
        self.yes = _phrase_re([*YES_PHRASES, *extra_yes])
        self.no = _phrase_re([*NO_PHRASES, *extra_no])
        self.idioms = _phrase_re(YES_IDIOMS)
        self.unsure = _phrase_re(UNSURE_PHRASES)

    def classify(self, text: str) -> bool | None:
        s = _normalize(text)
        if not s.strip():
            return None
        unsure = bool(self.unsure.search(s))
        s = self.unsure.sub(" ", s)
        idiom = bool(self.idioms.search(s))
        s = self.idioms.sub(" ", s)
        if self.no.search(s):
            return False  # any refusal wins: "no, don't", "yes... actually no", "yes, wait"
        if unsure:
            return None
        if idiom or self.yes.search(s):
            return True
        return None


_DEFAULT_YES_NO = _YesNo()


def classify_yes_no(
    text: str, *, extra_yes: Sequence[str] = (), extra_no: Sequence[str] = ()
) -> bool | None:
    """True for a clear yes, False for any no, None for silence or anything unclear.

    A refusal anywhere wins ("yes... actually no" is no). A hedge ("I'm not sure", "maybe")
    makes the answer unclear. Phrases match whole words, so "know" is not "no".
    """
    matcher = _DEFAULT_YES_NO if not (extra_yes or extra_no) else _YesNo(extra_yes, extra_no)
    return matcher.classify(text)


# --- spoken summaries --------------------------------------------------------------------------

_MAX_VALUE_CHARS = 40
_MAX_VALUE_WORDS = 6
_SECRET_KEY = re.compile(
    r"token|secret|passw|api_?key|apikey|auth|credential|cookie|session|private|otp|\bpin\b|key$",
    re.I,
)
_ID_KEY = re.compile(
    r"(^|_)ids?$|Ids?$|^sha$|url|uri|href|link|path|email|mail_?to|hash|number$|^owner$", re.I
)
_UNSPEAKABLE = re.compile(
    r"://|www\.|@|[\\/]|\b[0-9a-f]{7,}\b|\b(?=\w*\d)\w{16,}\b|^\s*[\[{]", re.I
)
_NAME_KEYS = ("name", "title", "summary", "subject", "branch", "repo", "label", "topic")
_VERBS = frozenset(
    {
        *("add", "archive", "assign", "book", "cancel", "close", "comment", "create"),
        *("delete", "draft", "edit", "fork", "invite", "label", "make", "merge", "move"),
        *("open", "post", "publish", "push", "remove", "rename", "reply", "run", "schedule"),
        *("send", "set", "share", "start", "submit", "trigger", "update", "upload", "write"),
    }
)
_WORD_MAP = {"googlecalendar": "calendar", "gmail": "Gmail", "github": "GitHub", "pr": "PR"}
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def humanize_tool_name(tool_name: str) -> str:
    """``mcp__composio__GMAIL_CREATE_EMAIL_DRAFT`` -> ``"Gmail create email draft"``."""
    parts = split_tool_name(tool_name)
    bare = parts[1] if parts else tool_name
    words = re.split(r"[_\-\s.]+", _CAMEL.sub("_", bare))
    out = [_WORD_MAP.get(w.lower(), w.lower()) for w in words if w]
    return " ".join(out) or "that tool"


def speakable_value(value: Any) -> str | None:
    """A short argument value that is safe to say out loud, or None.

    Anything long, URL-, path-, email- or ID-like is rejected, as is anything that
    ``sanitize_for_speech`` would change beyond whitespace and punctuation."""
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > _MAX_VALUE_CHARS or len(text.split()) > _MAX_VALUE_WORDS:
        return None
    if _UNSPEAKABLE.search(text):
        return None
    clean = sanitize_for_speech(text).rstrip(".!?,;: ")
    if not clean:
        return None
    if re.sub(r"\W+", "", clean.lower()) != re.sub(r"\W+", "", text.lower()):
        return None  # the sanitizer removed an ID, token, emoji or markup
    return clean


def _usable(key: str) -> bool:
    return not (_SECRET_KEY.search(key) or _ID_KEY.search(key))


def _pick_args(args: Mapping[str, Any], limit: int = 2) -> list[tuple[str, str]]:
    ordered = [k for k in _NAME_KEYS if k in args] + [k for k in args if k not in _NAME_KEYS]
    picked: list[tuple[str, str]] = []
    for key in ordered:
        if not isinstance(key, str) or not _usable(key):
            continue
        value = speakable_value(args.get(key))
        if value is not None:
            picked.append((key, value))
        if len(picked) >= limit:
            break
    return picked


def generic_summary(tool_name: str, args: Mapping[str, Any] | None) -> str:
    """The fallback line: the humanized tool name plus at most two short argument values."""
    action = humanize_tool_name(tool_name)
    first = action.split()[0].lower() if action else ""
    line = f"I'll {action}" if first in _VERBS else f"I'll use {action}"
    for i, (key, value) in enumerate(_pick_args(args or {})):
        label = " ".join(key.lower().split("_"))
        if i == 0 and key in _NAME_KEYS:
            line += f" called {value}"
        else:
            line += f", {label} {value}"
    return line + "."


def _arg(args: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = speakable_value(args.get(key))
        if value:
            return value
    return None


def _when(value: Any) -> str | None:
    """``2026-09-24T15:30:00`` -> ``"on Thursday at 3:30 PM"``; None if unparseable."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    hour = dt.hour % 12 or 12
    minute = f":{dt.minute:02d}" if dt.minute else ""
    return f"on {dt.strftime('%A')} at {hour}{minute} {'AM' if dt.hour < 12 else 'PM'}"


def _repo(args: Mapping[str, Any]) -> str | None:
    name = _arg(args, "name")
    return f"I'll create a GitHub repo called {name}." if name else "I'll create a GitHub repo."


def _pull_request(args: Mapping[str, Any]) -> str | None:
    title = _arg(args, "title")
    return f"I'll open a pull request titled {title}." if title else "I'll open a pull request."


def _issue(args: Mapping[str, Any]) -> str | None:
    title = _arg(args, "title")
    return f"I'll create an issue titled {title}." if title else "I'll create an issue."


def _comment(args: Mapping[str, Any]) -> str | None:
    return "I'll post a comment there."


def _branch(args: Mapping[str, Any]) -> str | None:
    branch = _arg(args, "branch", "name")
    return f"I'll create a branch called {branch}." if branch else "I'll create a branch."


def _email_draft(args: Mapping[str, Any]) -> str | None:
    subject = _arg(args, "subject")
    return f"I'll save an email draft about {subject}." if subject else "I'll save an email draft."


def _calendar_event(args: Mapping[str, Any]) -> str | None:
    what = _arg(args, "summary", "title")
    when = _when(args.get("start_datetime") or args.get("start") or args.get("start_time"))
    line = f"I'll add {what} to your calendar" if what else "I'll add an event to your calendar"
    return line + (f" {when}." if when else ".")


#: Built-in per-tool summarizers, keyed by the bare tool name (any server) or the full
#: ``mcp__server__tool`` name. Spark's own tools register theirs through the brain.
BUILTIN_SUMMARIZERS: dict[str, Summarizer] = {
    "create_repository": _repo,
    "create_pull_request": _pull_request,
    "create_issue": _issue,
    "add_issue_comment": _comment,
    "create_branch": _branch,
    "GMAIL_CREATE_EMAIL_DRAFT": _email_draft,
    "GOOGLECALENDAR_CREATE_EVENT": _calendar_event,
}


def _finish_question(line: str, *, destructive: bool) -> str:
    text = sanitize_for_speech(line).strip()
    if not text:
        text = "I'll go ahead with that."
    if destructive:
        text = "This can't be undone. " + text
    if not text.endswith("?"):
        if text[-1] not in ".!":
            text += "."
        text += " Should I go ahead?"
    return text


def summarize_tool_call(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    summarizers: Mapping[str, Summarizer] | None = None,
    *,
    destructive: bool = False,
) -> str:
    """One spoken line that ends in a question, with no URLs, IDs or secrets.

    A per-tool summarizer is looked up by full name, then by bare tool name. If it is missing,
    fails, or returns nothing, the generic summary is used instead."""
    args = tool_input if isinstance(tool_input, Mapping) else {}
    table: dict[str, Summarizer] = {**BUILTIN_SUMMARIZERS, **(summarizers or {})}
    parts = split_tool_name(tool_name)
    fn = table.get(tool_name) or (table.get(parts[1]) if parts else None)
    line: str | None = None
    if fn is not None:
        try:
            line = fn(args)
        except Exception as exc:
            log.warning("summarizer for %s failed (%s); using the generic line", tool_name, exc)
    return _finish_question(line or generic_summary(tool_name, args), destructive=destructive)


# --- confirmers --------------------------------------------------------------------------------


class ConfirmationAborted(RuntimeError):
    """The question could not be asked (the turn was cancelled). This counts as no answer."""


class Confirmer(ABC):
    """Asks the user to approve one action."""

    @abstractmethod
    async def ask(self, summary: str) -> bool | None:
        """True = yes, False = no, None = no answer or unclear (treated as no)."""


class LineReader:
    """Reads stdin lines on one daemon thread and hands them to asyncio.

    The ``--text`` REPL and :class:`TextConfirmer` share it, so they never compete for the same
    line. A prompt that timed out also leaves no blocked thread behind to swallow the next line.
    ``readline()`` returns ``""`` at end of input and ``None`` on timeout.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream
        self._queue: asyncio.Queue[str] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._eof = False

    def _start(self) -> asyncio.Queue[str]:
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            self._queue = asyncio.Queue()
            self._loop = loop
            self._thread = threading.Thread(
                target=self._pump, args=(loop, self._queue), name="spark-stdin", daemon=True
            )
            self._thread.start()
        return self._queue

    def _pump(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]) -> None:
        stream = self._stream if self._stream is not None else sys.stdin
        while True:
            try:
                line = stream.readline()
            except (OSError, ValueError):
                line = ""
            try:
                loop.call_soon_threadsafe(queue.put_nowait, line)
            except RuntimeError:  # the loop is closed: nobody is listening any more
                return
            if not line:
                return

    async def readline(self, timeout: float | None = None) -> str | None:  # noqa: ASYNC109
        if self._eof:
            return ""
        queue = self._start()
        try:
            async with asyncio.timeout(timeout):
                line = await queue.get()
        except TimeoutError:
            return None
        if not line:
            self._eof = True
        return line


class TextConfirmer(Confirmer):
    """``--text`` mode: print the question and read y/n from stdin.

    ``y``, ``yes``, ``sure``, ``go ahead`` (and the other yes phrases) mean yes. Any other
    typed answer means no. No answer within ``timeout_s``, or end of input, returns None.

    Args:
        reader: the shared :class:`LineReader` (default: a new one on stdin).
        out: where the prompt is written (default: stdout).
        say: optional coroutine that delivers the question through the app (so it lands after
            any reply text already printed or spoken); the prompt then only shows the choices.
    """

    def __init__(
        self,
        *,
        reader: LineReader | None = None,
        out: TextIO | None = None,
        say: Callable[[str], Awaitable[None]] | None = None,
        timeout_s: float = 60.0,
        yes_phrases: Sequence[str] = (),
        no_phrases: Sequence[str] = (),
    ) -> None:
        self.reader = reader or LineReader()
        self.out = out
        self.say = say
        self.timeout_s = timeout_s
        self._yes_no = _YesNo(yes_phrases, no_phrases)

    def _write(self, text: str) -> None:
        out = self.out or sys.stdout
        out.write(text)
        out.flush()

    async def ask(self, summary: str) -> bool | None:
        if self.say is not None:
            await self.say(summary)
            self._write("(yes/no)> ")
        else:
            self._write(f"{summary} (yes/no)> ")
        line = await self.reader.readline(self.timeout_s)
        if line is None:
            self._write("\n(no answer)\n")
            return None
        if not line:
            return None  # end of input
        answer = line.strip().lower()
        return answer == "y" or self._yes_no.classify(answer) is True


class VoiceConfirmer(Confirmer):
    """Say the question, then listen for one utterance and classify it as yes or no.

    Args:
        listen: ``await listen(timeout_s) -> str``, provided by the app. It records one utterance
            with the app's mic, recorder and STT, and returns its transcript, or ``""`` if nobody
            spoke within ``timeout_s``.
        say: ``await say(text)`` speaks and returns once playback has finished, so the mic never
            hears Spark's own question. The default is ``tts.speak`` of a single sentence.
        tts: used when ``say`` is not given.
        cue: optional ``await cue()`` played between the question and listening (the app plays
            the listening earcon and waits for it to finish).
        timeout_s: how long to wait for the user to start answering.
        answer_grace_s: extra time for the answer itself plus transcription.
        reprompt: on an unclear (non-empty) answer, ask "yes or no?" once more.
    """

    REPROMPT = "Sorry, was that a yes or a no?"

    def __init__(
        self,
        *,
        listen: Callable[[float], Awaitable[str]],
        say: Callable[[str], Awaitable[None]] | None = None,
        tts: Any | None = None,
        cue: Callable[[], Awaitable[None]] | None = None,
        timeout_s: float = 8.0,
        answer_grace_s: float = 25.0,
        reprompt: bool = True,
        yes_phrases: Sequence[str] = (),
        no_phrases: Sequence[str] = (),
    ) -> None:
        if say is None and tts is None:
            raise ValueError("VoiceConfirmer needs say= or tts=")
        self.listen = listen
        self._say = say
        self.tts = tts
        self.cue = cue
        self.timeout_s = timeout_s
        self.answer_grace_s = answer_grace_s
        self.reprompt = reprompt
        self._yes_no = _YesNo(yes_phrases, no_phrases)
        self.heard: list[str] = []  # transcripts of the answers (for tests and debugging)

    async def say(self, text: str) -> None:
        if self._say is not None:
            await self._say(text)
            return

        async def one() -> Any:
            yield text

        await self.tts.speak(one())

    async def _hear(self) -> str:
        if self.cue is not None:
            with contextlib.suppress(Exception):
                await self.cue()
        try:
            async with asyncio.timeout(self.timeout_s + self.answer_grace_s):
                text = await self.listen(self.timeout_s)
        except TimeoutError:
            return ""
        text = (text or "").strip()
        self.heard.append(text)
        return text

    async def ask(self, summary: str) -> bool | None:
        await self.say(summary)
        text = await self._hear()
        if not text:
            return None
        answer = self._yes_no.classify(text)
        if answer is None and self.reprompt:
            await self.say(self.REPROMPT)
            text = await self._hear()
            if not text:
                return None
            answer = self._yes_no.classify(text)
        return answer


# --- the hook ----------------------------------------------------------------------------------

_OUTCOME = {True: "yes", False: "no", None: "timeout"}


class ConfirmationGate:
    """The ``PreToolUse`` hook. Its signature is ``await gate(input_data, tool_use_id, context)``,
    and the SDK awaits it the same way (``_internal/query.py:565``).

    Args:
        policy: classifies tool names (read / side_effect / destructive / unknown).
        confirmer: asks the user. With None, every side-effecting call is denied without asking.
        summarizers: per-tool summary functions, keyed by the full or the bare tool name.
        ask_timeout_s: overall budget for one confirmation. Keep it below the ``HookMatcher``
            timeout so Spark answers "no answer" before the CLI gives up on the hook.
        before_ask: ``await before_ask(tool_use_id)`` runs before speaking. The brain uses it to
            let the reply text before the tool call reach the speaker first.
        on_asking: ``on_asking(True)`` when a confirmation starts waiting on the user and
            ``on_asking(False)`` when it ends (always paired). The brain pauses its "On it,
            give me a moment" timer in between, since the user is being asked a question.
        route_provider: ``route_provider()`` returns the active turn's route, ``"fast"`` or
            ``"deep"``, or None when no routed turn is running or routing is disabled. On
            ``"fast"`` only the fast tool set may run. None (the default) means no fast limit.
        fast_tools: the fast tool set as exact full tool names. Empty or None means every
            read-only tool (``policy.classify(name) == "read"``).
        on_fast_denied: ``on_fast_denied(name)`` after a call was refused on the fast route.
            The brain uses it to route the next turn deep.
        precheck: ``precheck(name, input_data)`` runs first for every call and returns a deny
            reason, or None to go on. The brain uses it for the per-request step budget (the
            hook input tells a worker sub-agent's call, ``agent_id``, from the voice model's).
        extra_read: built-in tool names that are pre-approved like read-only tools (``Agent``,
            which starts the researcher worker; its own tool calls pass this gate too).
    """

    def __init__(
        self,
        policy: ToolPolicy,
        confirmer: Confirmer | None = None,
        *,
        summarizers: Mapping[str, Summarizer] | None = None,
        ask_timeout_s: float | None = None,
        before_ask: Callable[[str | None], Awaitable[None]] | None = None,
        on_asking: Callable[[bool], None] | None = None,
        route_provider: Callable[[], str | None] | None = None,
        fast_tools: Iterable[str] | None = None,
        on_fast_denied: Callable[[str], None] | None = None,
        precheck: Callable[[str, Mapping[str, Any]], str | None] | None = None,
        extra_read: Iterable[str] = (),
    ) -> None:
        self.policy = policy
        self.precheck = precheck
        self.extra_read: frozenset[str] = frozenset(extra_read)
        self.on_asking = on_asking
        self.confirmer = confirmer
        self.summarizers: dict[str, Summarizer] = dict(summarizers or {})
        self.ask_timeout_s = ask_timeout_s
        self.before_ask = before_ask
        self.route_provider = route_provider
        self.fast_tools: frozenset[str] = frozenset(
            t.strip() for t in fast_tools or () if t.strip()
        )
        self.on_fast_denied = on_fast_denied
        # (tool name, yes/no/timeout/denied/fast-denied/cancelled)
        self.decisions: list[tuple[str, str]] = []
        self._lock = asyncio.Lock()  # parallel tool calls are asked about one at a time

    @property
    def current_route(self) -> str | None:
        """The active turn's route from ``route_provider`` ("fast", "deep" or None)."""
        if self.route_provider is None:
            return None
        try:
            route = self.route_provider()
        except Exception as exc:
            # Deep behaviour still confirms every side effect, so this stays safe.
            log.warning("confirm: route_provider failed (%s); no fast-route limit", exc)
            return None
        return route if isinstance(route, str) else None

    def in_fast_set(self, tool_name: str) -> bool:
        """Whether a fast-routed turn may call ``tool_name``."""
        if self.fast_tools:
            return tool_name in self.fast_tools
        return self.policy.classify(tool_name) == "read"

    def _fast_blocked(self, tool_name: str) -> bool:
        return self.current_route == "fast" and not self.in_fast_set(tool_name)

    def _deny_fast(self, tool_name: str) -> dict[str, Any]:
        log.info("tool denied: %s (fast route)", tool_name)
        self.decisions.append((tool_name, "fast-denied"))
        if self.on_fast_denied is not None:
            try:
                self.on_fast_denied(tool_name)
            except Exception as exc:
                log.debug("confirm: on_fast_denied failed: %s", exc)
        return deny(REASON_FAST_ROUTE)

    def summarize(self, tool_name: str, tool_input: Mapping[str, Any] | None) -> str:
        return summarize_tool_call(
            tool_name,
            tool_input,
            self.summarizers,
            destructive=self.policy.classify(tool_name) == "destructive",
        )

    async def __call__(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        data = input_data or {}
        name = str(data.get("tool_name") or "")
        kind = self.policy.classify(name) if name else "unknown"
        if self.precheck is not None and name:
            try:
                reason = self.precheck(name, data)
            except Exception as exc:
                log.debug("confirm: precheck failed: %s", exc)
                reason = None
            if reason:
                log.info("tool denied: %s (step budget)", name)
                self.decisions.append((name, "budget"))
                return deny(reason)
        if name in self.extra_read:
            return {}  # pre-approved built-in (allowed_tools)
        if kind == "read":
            if self._fast_blocked(name):
                return self._deny_fast(name)  # an explicit fast_tools list leaves it out
            return {}  # allowed_tools pre-approves it in the CLI
        if not name or not self.policy.needs_confirmation(name):
            # Blocked on every route, so say why, not "ask again".
            log.info("tool denied: %s (%s)", name or "<unnamed>", kind)
            self.decisions.append((name, "denied"))
            return deny(DENY_REASONS.get(kind, DENY_REASONS["unknown"]))
        if self._fast_blocked(name):
            return self._deny_fast(name)  # no question on the fast route
        if self.confirmer is None:
            log.info("tool denied: %s (%s)", name, kind)  # no confirmer to ask
            self.decisions.append((name, "denied"))
            return deny(DENY_REASONS["side_effect"])

        tool_input = data.get("tool_input")
        summary = self.summarize(name, tool_input if isinstance(tool_input, Mapping) else {})
        self._notify_asking(True)
        try:
            async with self._lock:
                if self.before_ask is not None:
                    try:
                        await self.before_ask(tool_use_id)
                    except Exception as exc:
                        log.debug("confirm: before_ask failed: %s", exc)
                answer = await self._ask(name, summary)
        except asyncio.CancelledError:
            # The CLI cancelled the hook (timeout or interrupt). It won't run the tool.
            log.info("tool confirm: %s -> cancelled", name)
            self.decisions.append((name, "cancelled"))
            raise
        finally:
            self._notify_asking(False)
        outcome = _OUTCOME[answer]
        log.info("tool confirm: %s -> %s", name, outcome)
        self.decisions.append((name, outcome))
        if answer is True:
            return allow()
        return deny(REASON_DECLINED if answer is False else REASON_NO_ANSWER)

    def _notify_asking(self, active: bool) -> None:
        if self.on_asking is None:
            return
        try:
            self.on_asking(active)
        except Exception as exc:
            log.debug("confirm: on_asking failed: %s", exc)

    async def _ask(self, name: str, summary: str) -> bool | None:
        assert self.confirmer is not None
        try:
            async with asyncio.timeout(self.ask_timeout_s):
                result = await self.confirmer.ask(summary)
        except TimeoutError:
            return None
        except Exception as exc:
            log.warning("tool confirm: %s: asking failed (%s); treating as no answer", name, exc)
            return None
        if result is True:
            return True
        if result is False:
            return False
        return None
