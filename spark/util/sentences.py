"""Streaming sentence splitter: turns brain text deltas into speakable sentences (spec §6, §9).

Each sentence is yielded as soon as it is complete, so TTS can start on the first one while the
brain is still streaming. Rules:

* ``.``, ``!``, ``?`` and ellipses (plus closing quotes/brackets) end a sentence when followed by
  whitespace. A boundary split across deltas (``"Hello"`` / ``"."`` / ``" World"``) waits for the
  whitespace before deciding.
* Newlines always end a sentence (blank lines are dropped).
* No split inside decimals / versions (``3.5``, ``v2.1``), after titles and Latin abbreviations
  (``Mr.``, ``Dr.``, ``St.``, ``vs.``, ``e.g.``, ``i.e.``), after single-letter initials, or after a
  bare list number (``1.``). ``etc.``, ``a.m.``/``p.m.``, dotted acronyms (``U.S.``) and ellipses
  end a sentence only when the next word starts with a capital letter.
* A long run without a boundary is force-split once it passes ``first_max_chars`` (first
  sentence, for latency) or ``max_chars`` (later ones): at the last clause mark (``, ; : —``),
  else the last space, never leaving a piece shorter than ``min_chars``.
* Short sentences ("Ok.") are emitted as-is; empty strings are never emitted.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator

__all__ = ["SentenceSplitter", "split_sentences", "split_text"]

DEFAULT_MIN_CHARS = 20
DEFAULT_MAX_CHARS = 220
DEFAULT_FIRST_MAX_CHARS = 120

_TERMINALS = ".!?…"
_CLOSERS = "\"'”’)]}»"
_OPENERS = "\"'“‘([{«"
_CLAUSE_MARKS = ",;:—–"

# Never end a sentence after these (lowercase, without the final dot).
_NEVER_SPLIT = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "vs",
        "e.g",
        "i.e",
        "cf",
        "approx",
        "fig",
        "dept",
        "gen",
        "gov",
        "rev",
        "capt",
        "lt",
        "col",
        "sgt",
    }
)

# End a sentence after these only if the next word is capitalized.
_CONDITIONAL = frozenset(
    {
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "a.m",
        "p.m",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "no",
    }
)

_DOTTED_ACRONYM = re.compile(r"^(?:[a-z]\.)+[a-z]$", re.IGNORECASE)

_WAIT = -1  # sentinel: a boundary may be at the end of the buffer; need more text


class SentenceSplitter:
    """Incremental splitter. ``feed()`` text, get complete sentences; ``flush()`` at the end."""

    def __init__(
        self,
        *,
        min_chars: int = DEFAULT_MIN_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
        first_max_chars: int | None = DEFAULT_FIRST_MAX_CHARS,
    ) -> None:
        first = max_chars if first_max_chars is None else first_max_chars
        if min_chars < 1:
            raise ValueError("min_chars must be >= 1")
        if max_chars <= min_chars or first <= min_chars:
            raise ValueError("max_chars and first_max_chars must be greater than min_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_max_chars = first
        self._buf = ""
        self._emitted = 0

    # -- public ---------------------------------------------------------------------------

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[str]:
        out = self._drain(final=True)
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            out.append(rest)
            self._emitted += 1
        return out

    # -- internals ------------------------------------------------------------------------

    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            self._buf = self._buf.lstrip()
            if not self._buf:
                return out
            cut, skip = self._find_boundary(self._buf, final=final)
            # A sentence (complete or still growing) longer than the limit is force-split so
            # TTS can start; otherwise wait for more text.
            span = len(self._buf) if cut is None or cut == _WAIT else cut
            forced = self._force_cut(self._buf[:span])
            if forced is not None:
                cut, skip = forced, 0
            elif cut is None or cut == _WAIT:
                return out
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut + skip :]
            if sentence:
                out.append(sentence)
                self._emitted += 1

    def _force_cut(self, buf: str) -> int | None:
        """Where to split ``buf`` (one pending sentence) if it is over the length limit."""
        limit = self.first_max_chars if self._emitted == 0 else self.max_chars
        if len(buf.rstrip()) <= limit:
            return None
        window = buf[: limit + 1]
        clause = None
        space = None
        for i in range(self.min_chars, len(window)):
            ch = window[i]
            if not ch.isspace():
                continue
            space = i
            if window[i - 1] in _CLAUSE_MARKS:
                clause = i
        if clause is not None:
            return clause
        if space is not None:
            return space
        # No space inside the window (e.g. a long URL): cut at the next space, if any.
        for i in range(limit + 1, len(buf)):
            if buf[i].isspace():
                return i
        return None

    def _find_boundary(self, buf: str, *, final: bool) -> tuple[int | None, int]:
        """Return (end index of first sentence, chars to skip after it) or (None|_WAIT, 0)."""
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch == "\n" or ch == "\r":
                return i, 1
            if ch not in _TERMINALS:
                i += 1
                continue
            run_end = i
            while run_end < n and buf[run_end] in _TERMINALS:
                run_end += 1
            end = run_end
            while end < n and buf[end] in _CLOSERS:
                end += 1
            if end >= n:
                return (end, 0) if final else (_WAIT, 0)
            if not buf[end].isspace():
                i = end  # "3.5", "e.g" inner dots, "U.S" inner dots, "?!x"
                continue
            decision = self._decide(buf, i, run_end, end, final=final)
            if decision == _WAIT:
                return _WAIT, 0
            if decision:
                return end, 0
            i = end
        return None, 0

    def _decide(self, buf: str, start: int, run_end: int, end: int, *, final: bool) -> int:
        """1 = boundary, 0 = not a boundary, _WAIT = need the next word to decide."""
        run = buf[start:run_end]
        if "!" in run or "?" in run:
            return 1
        if run != ".":  # ellipsis ("..." or "…"): a pause unless a new sentence starts
            return self._next_word_capitalized(buf, end, final=final)
        word = _word_before(buf, start)
        lower = word.lower()
        if not lower:
            return 1
        if lower in _NEVER_SPLIT:
            return 0
        if len(lower) == 1 and lower.isalpha():  # initials: "J. R. R. Tolkien"
            return 0
        if lower.isdigit() and buf[:start].strip(_OPENERS + " \t") == word:
            return 0  # bare list number "1."
        if lower in _CONDITIONAL or _DOTTED_ACRONYM.match(lower):
            return self._next_word_capitalized(buf, end, final=final)
        return 1

    @staticmethod
    def _next_word_capitalized(buf: str, end: int, *, final: bool) -> int:
        k = end
        n = len(buf)
        while k < n and buf[k].isspace():
            if buf[k] in "\r\n":
                return 1
            k += 1
        while k < n and buf[k] in _OPENERS:
            k += 1
        if k >= n:
            return 1 if final else _WAIT
        return 1 if buf[k].isupper() else 0


def _word_before(buf: str, idx: int) -> str:
    """The token immediately before ``buf[idx]`` (stripped of opening quotes/brackets)."""
    j = idx
    while j > 0 and not buf[j - 1].isspace():
        j -= 1
    return buf[j:idx].lstrip(_OPENERS)


async def split_sentences(
    chunks: AsyncIterable[str],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    first_max_chars: int | None = DEFAULT_FIRST_MAX_CHARS,
) -> AsyncIterator[str]:
    """Yield sentences from a stream of text deltas as soon as each one is complete."""
    splitter = SentenceSplitter(
        min_chars=min_chars, max_chars=max_chars, first_max_chars=first_max_chars
    )
    async for chunk in chunks:
        for sentence in splitter.feed(chunk):
            yield sentence
    for sentence in splitter.flush():
        yield sentence


def split_text(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    first_max_chars: int | None = DEFAULT_FIRST_MAX_CHARS,
) -> list[str]:
    """Split a complete text into sentences (same rules as :func:`split_sentences`)."""
    splitter = SentenceSplitter(
        min_chars=min_chars, max_chars=max_chars, first_max_chars=first_max_chars
    )
    return splitter.feed(text) + splitter.flush()
