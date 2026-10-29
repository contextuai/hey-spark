"""Tests for the streaming sentence splitter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

import pytest

from spark.util.sentences import SentenceSplitter, split_sentences, split_text


async def _stream(chunks: Iterable[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


async def _collect(chunks: Iterable[str], **kw: int) -> list[str]:
    return [s async for s in split_sentences(_stream(chunks), **kw)]


def _char_stream(text: str) -> list[str]:
    return list(text)


# --- basic boundaries ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there. How are you?", ["Hello there.", "How are you?"]),
        ("Wow! That is great. Right?", ["Wow!", "That is great.", "Right?"]),
        ("Really?! Yes.", ["Really?!", "Yes."]),
        ('He said "hi." Then he left.', ['He said "hi."', "Then he left."]),
        ("It works (mostly.) Next one.", ["It works (mostly.)", "Next one."]),
        ("Ok. Sure.", ["Ok.", "Sure."]),
        ("No trailing punctuation", ["No trailing punctuation"]),
        ("", []),
        ("   \n\t  ", []),
    ],
)
def test_basic_boundaries(text: str, expected: list[str]) -> None:
    assert split_text(text) == expected


def test_newlines_are_boundaries_and_blank_lines_dropped() -> None:
    text = "First line\n\nSecond line\r\nThird line."
    assert split_text(text) == ["First line", "Second line", "Third line."]


def test_list_numbers_do_not_split() -> None:
    text = "1. Buy milk\n2. Get eggs"
    assert split_text(text) == ["1. Buy milk", "2. Get eggs"]


def test_number_at_sentence_end_splits() -> None:
    assert split_text("The answer is 42. Next question.") == [
        "The answer is 42.",
        "Next question.",
    ]


# --- abbreviations, decimals, ellipses ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Smith met Dr. Jones on Main St. yesterday.",
        "Bring fruit, e.g. apples, i.e. the red ones.",
        "It is cats vs. dogs today.",
        "Ask Prof. Lee about it.",
        "J. R. R. Tolkien wrote it.",
        "We shipped it at 5 p.m. yesterday.",
        "Pens, paper, etc. are in the drawer.",
        "The U.S. economy grew.",
        "Pi is about 3.14 and e is 2.72 roughly.",
        "Upgrade to v2.1 before Friday.",
        "Version 1.2.3 is out now.",
        "Well... maybe not.",
        "Hmm… let me think.",
    ],
)
def test_no_false_split(text: str) -> None:
    assert split_text(text) == [text]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I live in the U.S. The weather is nice.", ["I live in the U.S.", "The weather is nice."]),
        ("We met at 9 a.m. Then we left.", ["We met at 9 a.m.", "Then we left."]),
        ("Apples, pears, etc. Oranges too.", ["Apples, pears, etc.", "Oranges too."]),
        ("Wait for it... Here it is.", ["Wait for it...", "Here it is."]),
        ("Hmm… Okay then.", ["Hmm…", "Okay then."]),
        ("Update to v2.1. It fixes bugs.", ["Update to v2.1.", "It fixes bugs."]),
        ("It costs 3.5 dollars. Cheap!", ["It costs 3.5 dollars.", "Cheap!"]),
    ],
)
def test_conditional_splits(text: str, expected: list[str]) -> None:
    assert split_text(text) == expected


# --- streaming -----------------------------------------------------------------------------


async def test_boundary_split_across_deltas() -> None:
    assert await _collect(["Hello", ".", " World", "."]) == ["Hello.", "World."]


async def test_decimal_split_across_deltas() -> None:
    assert await _collect(["It costs $3", ".", "50 today", ". Bye."]) == [
        "It costs $3.50 today.",
        "Bye.",
    ]


async def test_abbreviation_split_across_deltas() -> None:
    assert await _collect(["Ask Dr", ".", " ", "Smith now. Ok"]) == ["Ask Dr. Smith now.", "Ok"]


async def test_conditional_waits_for_next_word() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("We left at 9 a.m. ") == []  # undecided until the next word
    assert splitter.feed("then") == []
    assert splitter.feed(" ate. Done") == ["We left at 9 a.m. then ate."]
    assert splitter.flush() == ["Done"]


@pytest.mark.parametrize(
    "text",
    [
        "Hello there. How are you? I'm fine... Thanks! Dr. Who is 3.5 m tall. Bye.",
        'He said "stop." Then\nNew line. U.S. The end etc. Fin.',
    ],
)
async def test_char_by_char_matches_whole_text(text: str) -> None:
    assert await _collect(_char_stream(text)) == split_text(text)


async def test_whitespace_and_empty_chunks() -> None:
    chunks = ["", "   ", "Hi", "", ".", " ", "\n", "  ", "Bye", ""]
    assert await _collect(chunks) == ["Hi.", "Bye"]


async def test_only_whitespace_stream_yields_nothing() -> None:
    assert await _collect(["  ", "\n", "\t"]) == []


async def test_never_emits_empty_strings() -> None:
    out = await _collect(["...", " ", "!", "\n\n", ". ", "? ", "ok"])
    assert all(s.strip() for s in out)
    assert out[-1] == "ok"


async def test_first_sentence_yielded_before_stream_ends() -> None:
    release = asyncio.Event()
    finished = False

    async def slow_brain() -> AsyncIterator[str]:
        nonlocal finished
        yield "Tokyo is the capital"
        yield " of Japan. "
        await release.wait()  # the brain is still "thinking"
        yield "It is large."
        finished = True

    gen = split_sentences(slow_brain())
    first = await asyncio.wait_for(anext(gen), timeout=1.0)
    assert first == "Tokyo is the capital of Japan."
    assert not finished
    release.set()
    assert [s async for s in gen] == ["It is large."]
    assert finished


# --- long runs without punctuation ---------------------------------------------------------


def test_long_first_sentence_forced_at_clause() -> None:
    clause = "when the rain started falling on the quiet little town, "
    text = clause + "the people went inside and waited " * 5 + "for hours."
    out = split_text(text, first_max_chars=80, max_chars=120, min_chars=10)
    assert out[0] == clause.strip()
    assert " ".join(out) == " ".join(text.split())
    assert all(len(s) <= 120 for s in out)


def test_long_sentence_forced_at_space_when_no_clause() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    out = split_text(text, first_max_chars=50, max_chars=80, min_chars=10)
    assert len(out) > 1
    assert len(out[0]) <= 50
    assert all(len(s) <= 80 for s in out)
    assert " ".join(out) == text  # nothing lost, no word cut in half


async def test_long_stream_starts_early() -> None:
    words = [f"token{i} " for i in range(100)]
    splitter = SentenceSplitter(first_max_chars=60, max_chars=100, min_chars=10)
    emitted_at = None
    for i, w in enumerate(words):
        if splitter.feed(w) and emitted_at is None:
            emitted_at = i
    assert emitted_at is not None and emitted_at < 15


def test_long_token_without_spaces_is_not_cut_mid_token() -> None:
    url = "https://example.com/" + "a" * 200
    out = split_text(f"See {url} for details", first_max_chars=50, max_chars=80, min_chars=10)
    assert any(url in s for s in out)


def test_short_sentences_kept_as_is() -> None:
    assert split_text("Ok. Yes. No.") == ["Ok.", "Yes.", "No."]


def test_invalid_limits_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceSplitter(min_chars=50, max_chars=40)
    with pytest.raises(ValueError):
        SentenceSplitter(min_chars=0)


async def test_long_sentence_streamed_matches_whole_text() -> None:
    text = (
        "when the rain started falling on the quiet little town, "
        + "the people went inside and waited " * 5
        + "for hours. Then it stopped."
    )
    kw = {"first_max_chars": 80, "max_chars": 120, "min_chars": 10}
    words = [w + " " for w in text.split(" ")]
    assert await _collect(words, **kw) == split_text(text, **kw)
