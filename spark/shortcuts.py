"""Voice shortcuts: a short phrase stands for a detailed prompt (Phase 5, P5.7a).

"Good morning, how is today looking?" is a lot to put into words every day. A shortcut maps one
or more trigger phrases to the detailed prompt the user wrote once ("Give me my morning
briefing: today's meetings with times and who's in them, unread email that needs me, pull
requests waiting for my review ... under thirty seconds."). When a trigger is heard, the brain
gets the detailed prompt instead of the words, plus the words themselves so it knows what was
actually said.

Shortcuts live in ``shortcuts.json`` next to ``config.yaml`` (user-owned, gitignored), edited in
the desktop Shortcuts window (P5.7b) or by hand. The file is re-read when it changes, so edits
apply without a restart. A missing file is seeded with a few examples.

Matching is forgiving but strict about intent: case, punctuation and common contractions are
ignored ("how's" = "how is"), a trigger must appear as whole words, and after removing every
matched trigger, filler and the assistant's name, at most ``shortcuts.max_extra_words`` words
may be left. So "good morning, how is today looking like" fires the morning briefing, but "good
morning, can you also move my 3pm meeting to Friday and tell Sam" does not.

Pure Python, no heavy imports.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spark.config import Settings

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SHORTCUTS",
    "Shortcut",
    "ShortcutError",
    "ShortcutMatch",
    "ShortcutStore",
    "expand",
    "match_shortcut",
    "normalize",
]

ROUTES = ("auto", "fast", "deep")

_CONTRACTIONS = {
    "what's": "what is",
    "how's": "how is",
    "where's": "where is",
    "who's": "who is",
    "when's": "when is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "today's": "today is",
    "i'm": "i am",
    "let's": "let us",
    "what're": "what are",
    "how're": "how are",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "i've": "i have",
    "i'd": "i would",
    "you're": "you are",
    "we're": "we are",
}
_FILLER = frozenset(
    [
        "hey",
        "hi",
        "hello",
        "ok",
        "okay",
        "so",
        "um",
        "uh",
        "well",
        "please",
        "thanks",
        "thank",
        "you",
        "just",
        "like",
        "the",
        "a",
        "an",
        "and",
        "now",
        "then",
        "yeah",
        "right",
        "alright",
    ]
)


class ShortcutError(ValueError):
    """A shortcut that can't be saved (message is shown in the window)."""


def normalize(text: str) -> list[str]:
    """Lowercase words with straight apostrophes and contractions expanded."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    out: list[str] = []
    for word in text.split():
        word = word.strip("'")
        if not word:
            continue
        out.extend(_CONTRACTIONS.get(word, word).split())
    return out


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass
class Shortcut:
    name: str
    triggers: list[str]
    prompt: str
    route: str = "auto"  # auto | fast | deep
    enabled: bool = True
    exact: bool = False  # fire only when a trigger is all that was said (give or take filler)
    id: str = field(default_factory=lambda: f"s-{secrets.token_hex(4)}")
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def validate(self) -> Shortcut:
        self.name = " ".join(str(self.name or "").split())
        self.triggers = [" ".join(str(t).split()) for t in self.triggers or [] if str(t).strip()]
        self.triggers = list(dict.fromkeys(self.triggers))
        self.prompt = str(self.prompt or "").strip()
        if not self.name:
            raise ShortcutError("Give the shortcut a name.")
        if not self.triggers:
            raise ShortcutError("Add at least one phrase to say.")
        self.exact = bool(self.exact)
        if any(len(normalize(t)) < (1 if self.exact else 2) for t in self.triggers):
            raise ShortcutError(
                "Each phrase needs at least two words, so it isn't said by accident."
            )
        if not self.prompt:
            raise ShortcutError("Write the prompt Spark should send.")
        if self.route not in ROUTES:
            raise ShortcutError(f"Route must be one of {', '.join(ROUTES)}.")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Shortcut:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        triggers = known.get("triggers")
        if isinstance(triggers, str):
            known["triggers"] = [t for t in re.split(r"[\n;|]", triggers) if t.strip()]
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SKIP = "Skip any area you have no tool for without apologising."
_NEEDS_ME = (
    "Check what needs my attention right now across the connected tools: email and messages "
    "waiting for my reply, reviews requested from me, and meetings in the next two hours. Lead "
    "with the most urgent item. Three items at most, then offer the rest."
)

#: Written to a new shortcuts.json, and added once to an existing one (see
#: :meth:`ShortcutStore._add_new_defaults`). Several are nods to a certain film AI butler; the
#: model answers them in the current personality.
DEFAULT_SHORTCUTS: tuple[dict[str, Any], ...] = (
    {
        "name": "Morning briefing",
        "triggers": [
            "good morning",
            "how is today looking",
            "how does today look",
            "what does today look like",
            "what is on today",
            "rise and shine",
            "wake up daddy's home",
        ],
        "prompt": (
            "Give me my morning briefing for today. Use the connected tools. Cover, in this "
            "order: my meetings today with their times and who is in them, flagging anything "
            "that overlaps or needs preparation; unread email that needs a reply from me; pull "
            "requests or tickets waiting on me. Skip any area you have no tool for without "
            "apologising. Keep it under thirty seconds of speech and end with the one thing I "
            "should do first."
        ),
        "route": "deep",
    },
    {
        "name": "What needs me",
        "triggers": [
            "what needs my attention",
            "anything urgent",
            "what is waiting on me",
            "give me a rundown",
            "what is the rundown",
        ],
        "prompt": _NEEDS_ME,
        "route": "deep",
    },
    {
        "name": "Sitrep",
        "triggers": ["sitrep", "situation report", "status report", "what is the situation"],
        "prompt": _NEEDS_ME,
        "route": "deep",
        "exact": True,
    },
    {
        "name": "Next meeting",
        "triggers": [
            "what is next",
            "who is next",
            "what is my next meeting",
            "when is my next meeting",
        ],
        "prompt": (
            "Check my calendar for my next meeting: when it starts, how long until then, who is "
            "in it, and anything I should prepare. Two or three sentences. If nothing else is "
            "on today, say so and mention the first meeting tomorrow."
        ),
        "exact": True,
    },
    {
        "name": "Meeting prep",
        "triggers": [
            "brief me on my next meeting",
            "prep me for my next meeting",
            "prep me for the meeting",
            "prep me",
        ],
        "prompt": (
            "Prepare me for my next meeting. Use the connected tools: who is in it and their "
            f"roles if known, recent email threads with them, and what the meeting is for. {_SKIP} "
            "Under thirty seconds, ending with the one thing I should raise or decide."
        ),
        "route": "deep",
    },
    {
        "name": "Inbox",
        "triggers": [
            "check the post",
            "check my post",
            "anything in the inbox",
            "anything in my inbox",
        ],
        "prompt": (
            "Go through my unread email. Tell me how many there are, then the ones that need me: "
            "who it is from and what they want, most important first. Five at most, and skip "
            "newsletters and notifications. If no email tool is connected, say so in one line."
        ),
        "route": "deep",
    },
    {
        "name": "Workshop report",
        "triggers": [
            "what is happening in the workshop",
            "what is happening in the lab",
            "code report",
        ],
        "prompt": (
            "Give me a code report from the connected tools: pull requests waiting for my review, "
            "my own pull requests with failing checks or new comments, and issues newly assigned "
            "to me. Most urgent first, under thirty seconds. If no code tool is connected, say so "
            "in one line."
        ),
        "route": "deep",
    },
    {
        "name": "Home",
        "triggers": ["i am home", "honey i am home", "i am back home"],
        "prompt": (
            "Welcome me home, then give me a short evening wrap-up from the connected tools: "
            "anything that came in since this morning that still needs me, what is left on "
            f"today's calendar, and my first meeting tomorrow. {_SKIP} Under twenty seconds."
        ),
        "route": "deep",
        "exact": True,
    },
    {
        "name": "End of day",
        "triggers": ["wrap up the day", "end of day report", "end of the day report"],
        "prompt": (
            "Give me an end-of-day report from the connected tools: what I got done today "
            "(meetings held, pull requests merged, email answered), what is still waiting on me, "
            f"and the first thing on tomorrow's calendar. {_SKIP} Under thirty seconds."
        ),
        "route": "deep",
    },
    {
        "name": "Diagnostic",
        "triggers": ["run a diagnostic", "run diagnostics", "systems check", "system check"],
        "prompt": (
            "Run a quick self-check with your status tool and report like a ship's computer "
            "would: which brain and mode you are on, which connectors are working, and any "
            "running timers. If everything is fine, keep it to one or two sentences."
        ),
    },
    {
        "name": "Time check",
        "triggers": ["time check"],
        "prompt": "Tell me the time, and any running timers with the time left. One sentence.",
        "route": "fast",
        "exact": True,
    },
    {
        "name": "You up",
        "triggers": ["you up", "are you up", "you awake", "are you awake"],
        "prompt": (
            "I'm checking that you're here. Answer in one short, warm line that you are, in your "
            "own personality. Nothing else."
        ),
        "route": "fast",
        "exact": True,
    },
    {
        "name": "Drop the needle",
        "triggers": ["drop my needle", "drop the needle"],
        "prompt": (
            "Start my music: something classic and loud, a rock anthem unless I named "
            "something. Use a music tool if one is connected. If none is, say in one line that "
            "you can't play music yet and that a music connector would let you."
        ),
    },
    {
        "name": "House party protocol",
        "triggers": ["house party protocol", "all hands on deck"],
        "prompt": (
            "All hands on deck: sweep every connected tool for everything that needs me today: "
            "meetings, email waiting for a reply, reviews, failing builds, assigned issues. "
            f"Group it by urgency, most urgent first. {_SKIP} Under forty-five seconds."
        ),
        "route": "deep",
    },
    {
        "name": "New project",
        "triggers": ["start a new project", "create a new project", "new project file"],
        "prompt": (
            "I want to start a new project. If what I said includes its name, set it up with "
            "your start_project tool. If not, ask me in one short question what to call it and "
            "what it's for."
        ),
    },
)

#: The defaults a shortcuts.json written before the "defaults" record existed was seeded with.
_LEGACY_DEFAULTS: dict[str, list[str]] = {
    "Morning briefing": [
        "good morning",
        "how is today looking",
        "how does today look",
        "what does today look like",
        "what is on today",
    ],
    "What needs me": ["what needs my attention", "anything urgent", "what is waiting on me"],
}


@dataclass(frozen=True)
class ShortcutMatch:
    shortcut: Shortcut
    trigger: str  # the trigger phrase that matched (as written)
    extra: str  # words the user said beyond the triggers ('' if none)


def _find(words: Sequence[str], phrase: Sequence[str]) -> int:
    n = len(phrase)
    for i in range(len(words) - n + 1):
        if tuple(words[i : i + n]) == tuple(phrase):
            return i
    return -1


def match_shortcut(
    text: str,
    shortcuts: Iterable[Shortcut],
    *,
    names: Sequence[str] = (),
    max_extra_words: int = 6,
) -> ShortcutMatch | None:
    """The shortcut ``text`` fires, or None. The longest matched trigger wins."""
    words = normalize(text)
    if not words:
        return None
    best: tuple[int, ShortcutMatch] | None = None
    name_words = {w for n in names for w in normalize(n)}
    for shortcut in shortcuts:
        if not shortcut.enabled:
            continue
        covered = [False] * len(words)
        longest, first_trigger = 0, ""
        for trigger in shortcut.triggers:
            phrase = normalize(trigger)
            if not phrase:
                continue
            start = _find(words, phrase)
            if start < 0:
                continue
            for i in range(start, start + len(phrase)):
                covered[i] = True
            if len(phrase) > longest:
                longest, first_trigger = len(phrase), trigger
        if not longest:
            continue
        rest = [w for w, c in zip(words, covered, strict=True) if not c]
        meaningful = [w for w in rest if w not in _FILLER and w not in name_words]
        if len(meaningful) > (0 if shortcut.exact else max_extra_words):
            continue
        extra = " ".join(meaningful)
        candidate = ShortcutMatch(shortcut, first_trigger, extra)
        if best is None or longest > best[0]:
            best = (longest, candidate)
    return best[1] if best else None


def expand(match: ShortcutMatch, said: str) -> str:
    """What the brain gets: the shortcut's prompt plus what was actually said."""
    return (
        f"{match.shortcut.prompt}\n\n"
        f'(Voice shortcut "{match.shortcut.name}". The user said: "{said.strip()}".)'
    )


class ShortcutStore:
    """``shortcuts.json`` with change detection. Build with :meth:`from_settings`."""

    def __init__(self, path: Path, *, seed: bool = True) -> None:
        self.path = Path(path)
        self.seed = seed
        self._mtime: float | None = None
        self._items: list[Shortcut] = []
        self._warned = False
        # default name -> its triggers this file has been offered (the file's "defaults" key)
        self._defaults: dict[str, list[str]] | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ShortcutStore:
        return cls(settings.path(settings.shortcuts.file), seed=settings.shortcuts.seed_examples)

    def _seed_if_missing(self) -> None:
        if self.path.exists() or not self.seed:
            return
        try:
            self._defaults = {d["name"]: list(d["triggers"]) for d in DEFAULT_SHORTCUTS}
            self.write([Shortcut.from_dict(dict(d)).validate() for d in DEFAULT_SHORTCUTS])
            log.info("shortcuts: seeded %s with examples", self.path.name)
        except OSError as exc:
            log.warning("shortcuts: could not seed %s: %s", self.path, exc)

    def all(self) -> list[Shortcut]:
        """Every shortcut, re-read when the file changed."""
        self._seed_if_missing()
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._items, self._mtime = [], None
            return []
        if mtime != self._mtime:
            self._items = self._read()
            self._mtime = mtime
        return list(self._items)

    def _add_new_defaults(self, items: list[Shortcut]) -> bool:
        """Offer defaults (and default triggers) this file hasn't been offered yet, once: a
        default the user deleted or a trigger they removed stays gone. A new default is added
        only if no shortcut has its name; a trigger another shortcut uses is skipped. True if
        anything changed (the caller writes the file)."""
        seen = self._defaults if self._defaults is not None else dict(_LEGACY_DEFAULTS)
        taken = {" ".join(normalize(t)) for s in items for t in s.triggers}
        changed = self._defaults is None
        for default in DEFAULT_SHORTCUTS:
            name, triggers = default["name"], list(default["triggers"])
            offered = seen.get(name)
            new = [t for t in triggers if t not in (offered or [])]
            if not new:
                continue
            fresh = [t for t in new if " ".join(normalize(t)) not in taken]
            mine = next((s for s in items if s.name == name), None)
            if fresh and offered is None and mine is None:
                items.append(Shortcut.from_dict({**default, "triggers": fresh}).validate())
            elif fresh and offered is not None and mine is not None:
                mine.triggers = [*mine.triggers, *fresh]
                mine.updated_at = _utc_now()
            taken.update(" ".join(normalize(t)) for t in fresh)
            seen = {**seen, name: triggers}
            changed = True
        self._defaults = seen
        return changed

    def _read(self) -> list[Shortcut]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data.get("shortcuts", []) if isinstance(data, dict) else data
            items = []
            for entry in raw:
                try:
                    items.append(Shortcut.from_dict(entry).validate())
                except (ShortcutError, TypeError) as exc:
                    log.warning("shortcuts: skipped %r: %s", entry.get("name", "?"), exc)
            self._warned = False
            record = data.get("defaults") if isinstance(data, dict) else None
            self._defaults = (
                {str(k): [str(t) for t in v] for k, v in record.items() if isinstance(v, list)}
                if isinstance(record, dict)
                else None
            )
            if self.seed and self._add_new_defaults(items):
                try:
                    self.write(items)
                    log.info("shortcuts: added the new default shortcuts to %s", self.path.name)
                except OSError as exc:
                    log.warning("shortcuts: could not update %s: %s", self.path, exc)
            return items
        except (OSError, ValueError, AttributeError) as exc:
            if not self._warned:
                self._warned = True
                log.warning("shortcuts: %s is unreadable (%s); no shortcuts", self.path.name, exc)
            return []

    def write(self, items: Iterable[Shortcut]) -> None:
        items = list(items)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": "Voice shortcuts: say a trigger phrase and Spark sends the prompt. "
            "Edited in the Shortcuts window; hand edits are fine too.",
            "shortcuts": [s.to_dict() for s in items],
        }
        if self._defaults is not None:
            payload["defaults"] = self._defaults
        from spark.util.files import write_text_atomic

        write_text_atomic(self.path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        self._items = items
        try:
            self._mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime = None

    def save(self, data: dict[str, Any]) -> Shortcut:
        """Create (no ``id``) or update a shortcut from window data. Returns it, validated."""
        items = self.all()
        incoming = Shortcut.from_dict(data)
        existing = next((s for s in items if s.id == data.get("id")), None)
        if existing is not None:
            incoming.id, incoming.created_at = existing.id, existing.created_at
        incoming.updated_at = _utc_now()
        incoming.validate()
        clash = self._trigger_clash(incoming, items)
        if clash:
            raise ShortcutError(clash)
        if existing is not None:
            items = [incoming if s.id == existing.id else s for s in items]
        else:
            items.append(incoming)
        self.write(items)
        return incoming

    @staticmethod
    def _trigger_clash(new: Shortcut, items: Iterable[Shortcut]) -> str | None:
        mine = {" ".join(normalize(t)) for t in new.triggers}
        for other in items:
            if other.id == new.id:
                continue
            for trigger in other.triggers:
                if " ".join(normalize(trigger)) in mine:
                    return f'"{trigger}" already starts the shortcut "{other.name}".'
        return None

    def delete(self, shortcut_id: str) -> None:
        self.write([s for s in self.all() if s.id != shortcut_id])

    def set_enabled(self, shortcut_id: str, enabled: bool) -> None:
        items = self.all()
        for s in items:
            if s.id == shortcut_id:
                s.enabled = bool(enabled)
                s.updated_at = _utc_now()
        self.write(items)
