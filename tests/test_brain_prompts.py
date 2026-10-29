"""Voice prompt, sanitize_for_speech, and "new topic" detection (spec §8.2, §11)."""

from __future__ import annotations

import re

import pytest

from spark.brain.prompts import (
    MARKDOWN_CHARS,
    NO_TOOLS_NOTE,
    TOOL_RULES,
    VOICE_SYSTEM_PROMPT,
    SpeechFilter,
    is_new_topic,
    sanitize_for_speech,
    split_new_topic,
    voice_system_prompt,
)
from spark.util.sentences import SentenceSplitter

PHRASES = ["new topic", "start over", "forget that", "reset conversation"]

URL_RE = re.compile(r"https?://|www\.|\.com/|\.io/", re.IGNORECASE)
EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️‍]")


def assert_speakable(text: str) -> None:
    leaked = sorted(set(text) & MARKDOWN_CHARS)
    assert not leaked, f"markdown characters {leaked} in {text!r}"
    assert not URL_RE.search(text), f"URL in {text!r}"
    assert not EMOJI_RE.search(text), f"emoji in {text!r}"
    assert "@" not in text, f"email/handle in {text!r}"


# --- system prompt ---------------------------------------------------------------------------


def test_voice_prompt_covers_every_8_2_rule():
    p = VOICE_SYSTEM_PROMPT.lower()
    assert "spark" in p
    assert "speaking, not writing" in p
    assert "one to three short spoken sentences" in p
    for banned in ("markdown", "bullet", "urls", "code", "emoji"):
        assert banned in p, banned
    assert "spell out symbols" in p
    assert "summarize the result" in p and "ids" in p and "long lists" in p
    assert "want the details?" in p
    assert "clarifying question" in p
    assert "should i go ahead?" in p
    # Phase 2: tools exist, so the Phase 1 "no tools" note is gone
    assert NO_TOOLS_NOTE not in VOICE_SYSTEM_PROMPT
    assert "no tools" not in p


def test_voice_prompt_covers_tool_result_rules():
    p = VOICE_SYSTEM_PROMPT.lower()
    assert TOOL_RULES in VOICE_SYSTEM_PROMPT
    assert "one to three short spoken sentences" in p
    assert "count first" in p and "top one or two items" in p
    for never in ("ids", "commit hashes", "urls", "email addresses", "tokens", "timestamps"):
        assert never in p, never
    assert "offer the rest" in p or "want the others?" in p
    assert "can't do that yet" in p


def test_voice_prompt_without_tools_keeps_the_no_tools_note():
    p = voice_system_prompt(tools_available=False)
    assert NO_TOOLS_NOTE in p and TOOL_RULES not in p


def test_voice_prompt_is_itself_plain_text():
    # The prompt models the style it asks for: no markdown emphasis, headers, or links.
    assert "**" not in VOICE_SYSTEM_PROMPT and "#" not in VOICE_SYSTEM_PROMPT
    assert not URL_RE.search(VOICE_SYSTEM_PROMPT)


def test_voice_prompt_with_tools_drops_the_no_tools_note():
    p = voice_system_prompt(tools_available=True, extra="Extra rule.")
    assert NO_TOOLS_NOTE not in p and TOOL_RULES in p
    assert p.endswith("Extra rule.")


# --- sanitize_for_speech ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**Tokyo** is the capital of *Japan*.", "Tokyo is the capital of Japan."),
        ("__Bold__ and _italic_ and ~~gone~~ text.", "Bold and italic and gone text."),
        ("## The answer", "The answer"),
        ("> A quoted line.", "A quoted line."),
        ("Run `uv sync` first.", "Run uv sync first."),
        ("Use `snake_case_name` here.", "Use snake case name here."),
        ("**partial bold", "partial bold"),
        ("trailing emphasis*", "trailing emphasis"),
        ("```python", ""),
        ("```", ""),
        ("---", ""),
        ("|---|---|", ""),
        ("| Name | Value |", "Name Value"),
    ],
)
def test_strips_markdown(raw, expected):
    assert sanitize_for_speech(raw) == expected


def test_strips_fenced_code_block_in_one_piece():
    raw = "Here's how:\n```python\nprint('hi')\n```\nThat's it."
    assert sanitize_for_speech(raw) == "Here's how: That's it."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("- first item", "first item"),
        ("* second item", "second item"),
        ("• third item", "third item"),
        ("1. Buy milk.", "Buy milk."),
        ("2) Call Bob.", "Call Bob."),
        ("- [x] done task", "done task"),
        ("You have:\n- one\n- two", "You have: one. two"),
    ],
)
def test_strips_list_markers(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("See [the docs](https://example.com/a_b) now.", "See the docs now."),
        ("Details at https://example.com/x?y=1.", "Details at."),
        ("Go to www.example.org for more.", "Go to for more."),
        ("It lives at github.com/anthropics/sdk today.", "It lives at today."),
        ("Link: <https://example.com>", "Link:"),
        ("![a chart](https://img.example.com/c.png)", "a chart"),
        ("Email jane.doe@example.com for access.", "Email for access."),
        ("Source (https://example.com/page).", "Source."),
    ],
)
def test_removes_urls_and_emails(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Commit a1b2c3d4e5f6 is merged.", "Commit is merged."),
        ("Event 123e4567-e89b-12d3-a456-426614174000 was created.", "Event was created."),
        ("Message msg_01XyZabc123DEF456 sent.", "Message sent."),
        ("Address 0xdeadbeef42 used.", "Address used."),
        ("Short sha a1b2c3d merged.", "Short sha merged."),
        ("ULID 01HXYZABCDEF0123456789ABCD done.", "ULID done."),
        ("Node PR_kwDOK3x9Ys5nQ2Jb and I_kwDOAbc12 here.", "Node and here."),
        ("Legacy MDExOlB1bGxSZXF1ZXN0MTIz id.", "Legacy id."),
        ("Token ghp_16C7e42F292c6912E7710c838347Ae178B4a set.", "Token set."),
        ("Key sk-ant-api03-abcDEF123456 set.", "Key set."),
        ("Slack xoxb-1234-abcdef set.", "Slack set."),
        ("Jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123 end.", "Jwt end."),
    ],
)
def test_removes_ids(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("See acme/api#412 now.", "See number 412 now."),
        ("PR #42 is open.", "PR number 42 is open."),
        ("Updated 2026-09-24T10:15:00Z by the bot.", "Updated 2026-09-24 by the bot."),
        ("At 2026-09-24T10:15:00.123+02:00 exactly.", "At 2026-09-24 exactly."),
    ],
)
def test_keeps_spoken_references(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Run ```echo hi``` then relax.", "Run then relax."),
        ("Try ```uv sync --frozen``` or ```uv lock```.", "Try or."),
        ("Use ~~~ls -la~~~ to list.", "Use to list."),
    ],
)
def test_drops_inline_triple_backtick_code(raw, expected):
    assert sanitize_for_speech(raw) == expected


def test_speech_filter_drops_inline_triple_backtick_code():
    assert SpeechFilter().filter(["Run ```echo hi``` then relax.", "Done."]) == [
        "Run then relax.",
        "Done.",
    ]


# Raw-ish tool results as a model might paste them into a reply (spec §10 Phase 2: nothing raw,
# no IDs or URLs, is read aloud). Every leak marker below must be gone after sanitizing.
RAW_TOOL_TEXT = [
    'Found 3 open PRs: #412 "Fix login" (https://github.com/acme/api/pull/412, node '
    "PR_kwDOK3x9Ys5nQ2Jb, head sha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b) by "
    "octocat@example.com",
    '{"id": "01J8ZK4Q6V2W3X4Y5Z6A7B8C9D", "html_url": "https://github.com/acme/api/issues/7", '
    '"node_id": "I_kwDOK3x9Ys6Ab1Cd", "title": "Crash on start"}',
    "Event 123e4567-e89b-12d3-a456-426614174000 at 2026-09-25T09:00:00-07:00 "
    "(meet.google.com/abc-defg-hij), organizer jane.doe@corp.example.com",
    "Thread 18f3a2b4c5d6e7f8 from sam@example.org: see www.example.org/x?id=42",
    "Latest commit 3f2a9c1 by @octocat, token ghp_16C7e42F292c6912E7710c838347Ae178B4a",
]
LEAK_RE = re.compile(
    r"https?://|www\.|\.com/|\.org/|@|PR_kw|I_kw|ghp_|"
    r"[0-9a-f]{8}-[0-9a-f]{4}|(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}|"
    r"(?=\w*\d\w*\d)(?=\w*[A-Za-z]\w*[A-Za-z])\w{16,}",
    re.IGNORECASE,
)


@pytest.mark.parametrize("raw", RAW_TOOL_TEXT)
def test_raw_tool_results_leave_nothing_raw(raw):
    out = sanitize_for_speech(raw)
    assert not LEAK_RE.search(out), f"raw data leaked: {out!r}"
    assert_speakable(out)
    assert sanitize_for_speech(out) == out


def test_raw_tool_result_keeps_the_speakable_parts():
    out = sanitize_for_speech(RAW_TOOL_TEXT[0])
    assert out.startswith("Found 3 open PRs: number 412")
    assert "Fix login" in out
    assert "Crash on start" in sanitize_for_speech(RAW_TOOL_TEXT[1])


@pytest.mark.parametrize(
    "keep",
    [
        "The year 2025 was busy.",
        "Call 5551234567 tomorrow.",  # digits only: not an ID
        "It is 3.5 degrees.",
        "Ticket PROJ-1234 is open.",
        "The deadbeef bakery.",  # letters only: not an ID
        "It is -5 degrees.",
        "It's about 3:30, e.g. mid afternoon.",
    ],
)
def test_keeps_ordinary_text(keep):
    assert sanitize_for_speech(keep) == keep


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Great job \U0001f389!", "Great job!"),
        ("✅ Done ✨", "Done"),
        ("I ❤️ it", "I it"),
        ("Thumbs \U0001f44d\U0001f3fd up", "Thumbs up"),
        ("\U0001f1ef\U0001f1f5 Japan", "Japan"),
    ],
)
def test_removes_emoji(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Salt & pepper", "Salt and pepper"),
        ("Battery at 50%.", "Battery at 50 percent."),
        ("2 + 2 = 4", "2 plus 2 equals 4"),
        ("Answer yes/no.", "Answer yes or no."),
        ("Issue #5 is open.", "Issue number 5 is open."),
        ("It takes ~10 minutes.", "It takes about 10 minutes."),
        ("It costs $5.99.", "It costs 5.99 dollars."),
        ("Raised $3 million.", "Raised 3 million dollars."),
        ("It's 72°F outside.", "It's 72 degrees F outside."),
        ("Go from A -> B.", "Go from A to B."),
        ("I like C++.", "I like C plus plus."),
        ("3 * 4 is 12.", "3 times 4 is 12."),
        ("If 3 > 2 then", "If 3 greater than 2 then"),
        ("Dates like 9/24 stay.", "Dates like 9/24 stay."),
    ],
)
def test_spells_out_symbols(raw, expected):
    assert sanitize_for_speech(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "**", "- ", "```", "\U0001f389", "[](https://x.io)"])
def test_markup_only_becomes_empty(raw):
    assert sanitize_for_speech(raw) == ""


IDEMPOTENCY_CORPUS = [
    "**Tokyo** is the capital of *Japan*.",
    "See [docs](https://x.io/a) & call 555-1234, 50% off!",
    "- item one\n- item two\n1. step",
    "Commit a1b2c3d4e5f6 by jane@x.com \U0001f389",
    "C++ and 2 + 2 = 4, yes/no, #5, ~10, $5, 72°F",
    "Partial [link text",
    "`code` and __bold__ and _it_",
    "Plain sentence with nothing to do.",
    "Header:\n## Title\n> quote",
]


@pytest.mark.parametrize("raw", IDEMPOTENCY_CORPUS)
def test_idempotent(raw):
    once = sanitize_for_speech(raw)
    assert sanitize_for_speech(once) == once
    assert_speakable(once)


def test_safe_on_every_prefix_of_a_messy_reply():
    raw = "**Sure!** Here's [the link](https://ex.com/a_b) \U0001f389 and `x_y` & 5%."
    for i in range(len(raw) + 1):
        out = sanitize_for_speech(raw[:i])
        assert_speakable(out)
        assert sanitize_for_speech(out) == out


# --- SpeechFilter (fence-aware, per sentence) -------------------------------------------------


def test_speech_filter_drops_lines_inside_fences():
    f = SpeechFilter()
    sentences = ["Here is the code:", "```python", "print('hi')", "x = 1", "```", "Done."]
    assert f.filter(sentences) == ["Here is the code:", "Done."]


def test_speech_filter_reset_clears_fence_state():
    f = SpeechFilter()
    assert f.feed("```") == ""
    assert f.feed("inside") == ""
    f.reset()
    assert f.feed("outside.") == "outside."


# --- recorded-transcript test (spec §11) ------------------------------------------------------

# Realistic raw model outputs (as streamed deltas) that leak markdown, URLs, IDs, and emoji.
RECORDED_TRANSCRIPTS: list[list[str]] = [
    ["The capital of Japan is **", "Tokyo", "**. It's also the largest city."],
    [
        "Here are your meetings:\n",
        "- **9:00** Standup\n",
        "- **11:30** Design review ",
        "(https://meet.example.com/abc-defg-hij)\n",
        "- 2pm: 1:1 with Sam\n\nWant the details?",
    ],
    [
        "## Summary\n\n",
        "PR #42 (`fix/audio_ring`) was merged by ",
        "@alice in commit 9f8e7d6c5b4a. ",
        "See [the PR](https://github.com/org/repo/pull/42) \U0001f680",
    ],
    [
        "Sure! \U0001f60a To install it, run:\n",
        "```bash\n",
        "uv sync && uv run python -m spark --text\n",
        "```\n",
        "That's it ✅",
    ],
    [
        "1. Open the app\n2. Click *Settings*\n3. Toggle **Dark mode** on",
        "\n\n| Option | Value |\n|---|---|\n| Theme | Dark |",
    ],
    [
        "Your ticket 123e4567-e89b-12d3-a456-426614174000 is ~80% done & ",
        "due 9/30. Email ops@example.com with questions.",
    ],
    ["It's 72°F and sunny ☀️ — perfect for a walk -> the park!"],
]


@pytest.mark.parametrize("deltas", RECORDED_TRANSCRIPTS)
def test_recorded_transcript_is_speakable_after_split_and_sanitize(deltas):
    splitter = SentenceSplitter()
    speech = SpeechFilter()
    spoken: list[str] = []
    for delta in deltas:
        spoken.extend(speech.filter(splitter.feed(delta)))
    spoken.extend(speech.filter(splitter.flush()))
    assert spoken, "a reply must leave something to say"
    for sentence in spoken:
        assert sentence.strip()
        assert_speakable(sentence)
    joined = " ".join(spoken)
    assert "uv sync" not in joined  # fenced code is not read aloud
    assert not re.search(r"[0-9a-f]{8,}", joined)  # no long hex IDs


def test_recorded_transcript_keeps_the_content():
    splitter = SentenceSplitter()
    sentences = splitter.feed("".join(RECORDED_TRANSCRIPTS[0])) + splitter.flush()
    spoken = SpeechFilter().filter(sentences)
    assert spoken == ["The capital of Japan is Tokyo.", "It's also the largest city."]


# --- new topic ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "new topic",
        "New topic.",
        "Spark, new topic",
        "spark new topic!",
        "Hey Spark, new topic please",
        "new topic please",
        "NEW TOPIC",
        "Okay Spark, let's start over.",
        "forget that",
        "Spark, reset conversation.",
        "Let's do a new topic",
    ],
)
def test_is_new_topic(text):
    assert is_new_topic(text, PHRASES)
    assert split_new_topic(text, PHRASES) == (True, "")


@pytest.mark.parametrize(
    "text",
    [
        "What's a good new topic for my blog?",
        "new topics are interesting",
        "I want to start overseas",
        "Tell me about Spark design",
        "",
    ],
)
def test_is_not_new_topic(text):
    assert not is_new_topic(text, PHRASES)


def test_new_topic_with_a_follow_on_question_keeps_the_rest():
    assert split_new_topic("Spark, new topic. What's the capital of Japan?", PHRASES) == (
        True,
        "What's the capital of Japan?",
    )
    assert split_new_topic("new topic, how tall is Everest", PHRASES) == (
        True,
        "how tall is Everest",
    )


def test_new_topic_with_no_phrases_never_matches():
    assert split_new_topic("new topic", []) == (False, "new topic")


def test_double_ampersand_is_one_and():
    assert sanitize_for_speech("this && that") == "this and that"


NOVA = ("nova", "novah", "noah")


def test_new_topic_accepts_a_custom_name_and_its_stt_spellings():
    assert split_new_topic("Novah, new topic.", PHRASES, NOVA) == (True, "")
    assert split_new_topic("Hey Nova, start over. What time is it?", PHRASES, NOVA) == (
        True,
        "What time is it?",
    )
    assert split_new_topic("Noah, new topic", PHRASES, NOVA) == (True, "")
    assert not is_new_topic("Tell me about Novah", PHRASES, NOVA)
    assert not is_new_topic("Novah, new topic", PHRASES)  # default names: spark only


def test_system_prompt_uses_the_configured_name():
    prompt = voice_system_prompt(name="Nova")
    assert prompt.startswith("You are Nova, a voice assistant")
    assert not re.search(r"\bSpark\b", prompt)
    assert voice_system_prompt() == VOICE_SYSTEM_PROMPT
    assert VOICE_SYSTEM_PROMPT.startswith("You are Spark,")
