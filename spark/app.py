"""Spark's state machine and main async loop (spec §3, §6).

IDLE --wake word--> LISTENING --end of speech--> THINKING --first sentence--> SPEAKING
  ^                                                                             |
  +---- awake window expires (no speech) <------ FOLLOW_UP <-- done speaking ---+

Awake and asleep (Phase 5, P5.2): FOLLOW_UP is "awake". After a reply Spark listens with no wake
word until ``awake_until`` (``settings.awake_s``, default 180 s, from the end of the reply). Each
handled request restarts the window. When it runs out, the sleep chime plays, the state goes to
IDLE ("asleep": wake word only) and ``on_sleep`` is called.

"Is this for me?" (P5.4, :mod:`spark.attention`): speech heard while awake (FOLLOW_UP) is checked
before it reaches the brain. Ignored speech is logged, doesn't restart the awake window, and
gets no thinking tick (the tick for a follow-up plays once the check passes).

A new conversation per wake (P5.6): when Spark falls asleep (or a turn fails) the brain is reset
in the background, so the next wake finds a fresh, already primed session whose system prompt
carries the briefing (:mod:`spark.briefing`: the last conversation's exchanges if it ended under
``attention.carry_over_s`` ago). The user never sees the break.

Activity log (P5.5, :mod:`spark.activity`): every request, reply, command, ignored utterance,
confirmation question and answer, announcement and failure is recorded (the brain adds routes,
tool calls and results). A conversation opens at the first request and closes when Spark goes to
sleep, on "new conversation", or when the app closes.

Voice shortcuts (P5.7a, :mod:`spark.shortcuts`): when the utterance is one of the user's trigger
phrases ("good morning, how is today looking"), the brain gets the shortcut's detailed prompt
instead of the words (plus the words, for context), on the shortcut's route. A shortcut counts as
addressed while awake. ``shortcuts.json`` is re-read when it changes.

Mute (P5.7k): while ``muted`` the mic is ignored entirely. Blocks are dropped before anything
sees them (not even the pre-roll ring keeps them): no wake word, no follow-ups, and a pending
confirmation gets no answer (so the tool doesn't run). Work already under way goes on: a reply
being spoken finishes, timers fire. The awake window's clock keeps running; unmuting with time
left listens again without the wake word, otherwise Spark is asleep. "Mute" also works by voice;
unmuting is the window's mic button (Spark can't hear while muted).

Local commands (P5.3, :mod:`spark.commands`): an utterance that is only "go to sleep", "never
mind" or "new conversation" (give or take the name and filler) is handled here without the
brain: "Okay." then sleep, sleep silently, or "Okay, fresh start." with a brain reset in the
background while Spark stays awake.

THINKING/SPEAKING --side-effecting tool--> CONFIRMING --answer heard--> THINKING (P3.1)

The mic loop never blocks: each finished utterance is handled in its own task (STT, brain,
TTS), while the loop keeps feeding frames (for barge-in and the follow-up recorder).

Timers (P2.4): one :class:`TimerService` belongs to the app, so timers outlive brain resets. When
one fires, "Your pasta timer is done." is announced (earcon plus speech). If Spark is busy
(LISTENING, THINKING, SPEAKING) the announcement waits until the app is back in IDLE or
FOLLOW_UP. In FOLLOW_UP the follow-up recorder is paused while Spark speaks and restarted after.

Spoken confirmation (P3.1): the brain's ``PreToolUse`` gate asks the :class:`VoiceConfirmer`, which
calls back into the app:

1. ``say_for_confirmation(question)`` puts the question into the running turn's speech queue. The
   current TTS stream is closed at that point, so everything Spark already said finishes playing,
   and then the question is spoken on its own. Only after playback ends does ``say`` return, so
   the mic never hears Spark's own question.
2. ``listen_for_answer(timeout)`` switches to ``State.CONFIRMING``. Mic frames go to the recorder
   for one utterance (not to wake-word or follow-up handling), and the transcript is returned
   ('' on silence). The state then goes back to THINKING and the turn continues.

``--text`` mode uses a :class:`TextConfirmer` that reads y/n through the same stdin
:class:`LineReader` as the REPL.

Barge-in (P3.3, ``app.barge_in``, default off): the wake word heard while Spark is SPEAKING or
THINKING, or while a timer announcement plays, interrupts it. :meth:`SparkApp.barge_in`:

1. calls ``player.stop()`` first, synchronously, before any ``await``, so the next output block is
   silent (a :class:`~spark.audio.player.PlaybackBuffer` block is 20 ms);
2. bumps the turn *generation*, so the interrupted turn can never flip the state again (to
   FOLLOW_UP or IDLE) even if it unwinds late;
3. stops the TTS (no later sentence is synthesized or played), drains the turn's pending
   sentences, fails any unspoken confirmation question, cancels the turn and announcement tasks;
4. plays the wake earcon, starts recording with pre-roll and goes to LISTENING at once;
5. in a background task, calls ``brain.interrupt()`` (the SDK brain may wait up to its
   ``interrupt_timeout_s`` for the turn to drain, which must not stall the mic loop) and waits for
   the cancelled tasks. The next turn waits for that cleanup before it calls the brain.

The time from the wake frame to ``player.stop()`` returning is ``stop_latency_ms``, and to the
interrupted ``speak()`` returning is ``speak_return_ms``. Both are on :attr:`SparkApp.last_barge_in`
and on the interrupted turn's latency record, which is logged with ``interrupted=true``.

Decisions:

* THINKING is interruptible too (the brain is aborted before it speaks). It is the same code path,
  and the user saying "Hey Spark" again means they want to start over.
* CONFIRMING: with barge-in on, the wake word is *not* an answer. It cancels the turn, so the
  pending tool gets no answer and the gate denies it (the safe outcome), and Spark listens for a
  new request. With barge-in off, the wake detector is not run while CONFIRMING (as before), and
  a leading "Hey Spark" is stripped from the answer.
* A timer announcement playing in IDLE was already interruptible by the wake word (P2.4), because
  IDLE always listens for it. It now uses this same path and is measured the same way. In
  FOLLOW_UP the announcement is only interruptible with ``app.barge_in`` on.

Echo: while Spark speaks, the mic hears Spark's own voice and the wake-word detector gets those
frames. There is no acoustic echo cancellation here. The spec recommends a hardware speakerphone
(with built-in AEC), and ``app.barge_in`` should only be turned on with one. Per backend:

* Porcupine: robust to non-keyword speech, so false self-triggers are rare, but a reply that says
  "Hey Spark" (or close) through a loud laptop speaker can trigger it. Lower
  ``wakeword.options.porcupine.sensitivity`` if it does.
* openWakeWord: more prone to firing on TTS voices; with barge-in on, raise
  ``wakeword.options.openwakeword.threshold`` or use a speakerphone.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from spark import registry
from spark.activity import ActivityLog, NullActivityLog
from spark.attention import Addressee, Verdict
from spark.audio.recorder import Recorder
from spark.audio.stream import Rechunker, RingBuffer
from spark.brain.confirm import (
    ConfirmationAborted,
    Confirmer,
    LineReader,
    TextConfirmer,
    VoiceConfirmer,
)
from spark.commands import (
    CANCEL,
    MODE_AUTO,
    MODE_COMMANDS,
    MODE_DEEP,
    MODE_FAST,
    MUTE,
    NEW_CONVERSATION,
    SLEEP,
    Command,
    match_command,
    split_model_request,
)
from spark.interfaces import SAMPLE_RATE
from spark.persona import PERSONAS, Voices, persona_for
from spark.prefs import MODES, Prefs, apply_prefs, model_info, pref_updates, set_route_models
from spark.shortcuts import ShortcutMatch, ShortcutStore, expand, match_shortcut
from spark.tools.timers import Timer, TimerService
from spark.util.clock import Clock, SystemClock
from spark.util.latency import LatencyLog, TurnTimer, format_breakdown
from spark.util.logging import TranscriptLog
from spark.util.sentences import split_sentences

if TYPE_CHECKING:
    from spark.config import Settings
    from spark.interfaces import STT, TTS, VAD, AudioInput, AudioOutput, Brain, WakeWordDetector

log = logging.getLogger("spark.app")

__all__ = ["BargeIn", "SparkApp", "State", "failure_line", "strip_wake_phrase"]

#: Earcon played before a timer announcement (the wake chime: short, bright, already shipped).
TIMER_EARCON = "wake"
#: How long the background barge-in cleanup waits for the cancelled turn to unwind.
UNWIND_TIMEOUT_S = 2.0


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


@dataclass
class BargeIn:
    """Timing of one barge-in, measured with ``time.perf_counter`` (wall time, never a fake clock).

    ``stop_latency_ms``: the wake frame was processed -> ``player.stop()`` returned (the next
    output block is silent). ``speak_return_ms``: -> the interrupted ``speak()`` returned.
    ``unwind_ms``: -> the cancelled turn / announcement task had finished.
    """

    source: str  # the state that was interrupted: speaking, thinking, confirming, idle, follow_up
    generation: int  # the turn generation *after* the barge-in
    detected_at: float
    stop_latency_ms: float
    speak_return_ms: float | None = None
    unwind_ms: float | None = None
    brain_interrupted: bool = False

    def note_speak_returned(self) -> None:
        if self.speak_return_ms is None:
            self.speak_return_ms = _ms(time.perf_counter() - self.detected_at)

    def as_meta(self) -> dict[str, Any]:
        return {
            "barge_in_from": self.source,
            "stop_latency_ms": self.stop_latency_ms,
            "speak_return_ms": self.speak_return_ms,
        }


def failure_line(exc: BaseException, voices: Voices | None = None) -> str:
    """One spoken sentence for a failed turn (P5.1), so the user hears more than the error
    beep. Never includes the error text itself (it can hold paths, IDs or secrets).
    ``voices``: the persona's wording (default: the standard one)."""
    from spark.interfaces import BackendUnavailable

    voices = voices or Voices(PERSONAS["standard"])
    spoken = getattr(exc, "spoken", None)
    if isinstance(spoken, str) and spoken.strip():
        return spoken.strip()  # the brain's own line (e.g. "OpenAI refused the API key...")
    text = str(exc).lower()
    if "error_max_turns" in text or "maximum number of turns" in text:
        return voices.line("too_many_steps")
    if isinstance(exc, TimeoutError) or "did not finish within" in text or "timed out" in text:
        return voices.line("too_long")
    if any(k in text for k in ("login", "authenticat", "unauthorized", "401")):
        return "Claude didn't accept the sign-in. Check Claude Code's sign-in or your API key."
    if isinstance(exc, BackendUnavailable):
        return voices.line("broken")
    return voices.line("failed")


def timer_announcement(timer: Timer, voices: Voices | None = None) -> str:
    """ "Your pasta timer is done." / "Your timer 2 is done." (in the persona's words)"""
    return (voices or Voices(PERSONAS["standard"])).line("timer_done", timer=timer.spoken_name)


def wake_prefix_pattern(names: Sequence[str]) -> re.Pattern[str]:
    """Matches a leading "Hey <name>," for any of ``names``."""
    alts = "|".join(re.escape(n) for n in names if n)
    return re.compile(rf"^\s*(?:(?:hey|hi|okay|ok)[\s,]+)?(?:{alts})\b[\s,.!?:;-]*", re.I)


_WAKE_PREFIX = wake_prefix_pattern(("spark", "sparks", "sparky"))


def strip_wake_phrase(text: str, names: Sequence[str] | None = None) -> str:
    """Remove a leading "Hey Spark," (or "Hey <name>," for ``names``) that the pre-roll let
    into the transcript."""
    pattern = _WAKE_PREFIX if names is None else wake_prefix_pattern(names)
    return pattern.sub("", text, count=1).strip()


def strip_spoken_prefix(text: str, spoken: str) -> str:
    """Remove ``spoken`` (Spark's own wake reply, "Yes, sir?") from the start of ``text``,
    ignoring case and punctuation: the mic hears the reply, and Whisper may write it down."""
    words = re.findall(r"[\w']+", spoken.lower())
    if not words:
        return text.strip()
    pattern = r"^\W*" + r"\W+".join(re.escape(w) for w in words) + r"\b\W*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"
    CONFIRMING = "confirming"  # waiting for a spoken yes/no before a side-effecting tool


class SparkApp:
    """Wires the components together. Build with :meth:`from_settings` or inject fakes."""

    def __init__(
        self,
        settings: Settings,
        *,
        mic: AudioInput | None,
        player: AudioOutput | None,
        wakeword: WakeWordDetector | None,
        vad: VAD | None,
        stt: STT | None,
        tts: TTS,
        brain: Brain,
        clock: Clock | None = None,
        transcripts: TranscriptLog | None = None,
        latency_log: LatencyLog | None = None,
        on_state: Callable[[State, State], None] | None = None,
        on_turn: Callable[[TurnTimer], None] | None = None,
        timers: TimerService | None = None,
        on_announce: Callable[[str], None] | None = None,
        confirmer: Confirmer | None = None,
        line_reader: LineReader | None = None,
        on_sleep: Callable[[str], None] | None = None,
        addressee: Addressee | None = None,
        on_ignored: Callable[[str, Verdict], None] | None = None,
        activity: ActivityLog | None = None,
        shortcuts: ShortcutStore | None = None,
        prefs: Prefs | None = None,
        on_model: Callable[[dict[str, Any]], None] | None = None,
        on_mute: Callable[[bool], None] | None = None,
    ) -> None:
        self.settings = settings
        self.voices = persona_for(settings)  # how Spark words its own lines (spark/persona.py)
        # The spoken wake reply ("Yes, sir?"), synthesized once at warmup: (pcm, rate) or None.
        self._wake_clip: tuple[np.ndarray, int] | None = None
        self.mic = mic
        self.player = player
        self.wakeword = wakeword
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.brain = brain
        self.clock: Clock = clock or SystemClock()
        self.transcripts = transcripts or TranscriptLog.from_settings(settings)
        self.latency_log = latency_log or LatencyLog.from_settings(settings)
        self.on_state = on_state
        self.on_turn = on_turn
        self.state = State.IDLE
        self.state_history: list[State] = [State.IDLE]
        self.turn_task: asyncio.Task[None] | None = None
        self.last_timer: TurnTimer | None = None
        self.last_reply: str = ""
        self.errors: list[BaseException] = []
        self._timer: TurnTimer | None = None
        self._stopping = False
        self.recorder = Recorder(settings, vad, clock=self.clock) if vad is not None else None
        self._ring = RingBuffer(int(SAMPLE_RATE * settings.audio.ring_seconds))
        self._wake_chunker = Rechunker(wakeword.frame_length) if wakeword is not None else None
        self._started_at = self.clock.monotonic()
        # Timer announcements: queued while busy, spoken in IDLE / FOLLOW_UP.
        self.timers = timers
        self.on_announce = on_announce
        self.announcements: list[str] = []  # everything announced, for tests and status
        # (text, earcon to play first or None)
        self._pending_announcements: list[tuple[str, str | None]] = []
        self._announce_task: asyncio.Task[None] | None = None
        self._announcing = False
        self._closed = False
        self._speak_lock = asyncio.Lock()  # one speak() at a time: turn replies vs. timers
        if timers is not None:
            timers.on_fire = self._on_timer_fired
        # Spoken confirmation (P3.1).
        self.confirmer = confirmer
        self.line_reader = line_reader  # --text: shared by the REPL and the TextConfirmer
        self.confirm_questions: list[str] = []  # every question asked, for tests and status
        self._speech_queue: asyncio.Queue[Any] | None = None  # the running turn's speech
        self._answer: asyncio.Future[str] | None = None  # recorder outcome while CONFIRMING
        # Barge-in (P3.3). A turn that sees a different generation from the one it started with
        # was interrupted, and must not touch the state any more.
        self._generation = 0
        self.last_barge_in: BargeIn | None = None
        self.barge_ins: list[BargeIn] = []
        self._interrupt_tasks: set[asyncio.Task[None]] = set()
        # Awake window (P5.2): clock time FOLLOW_UP ends, or None while asleep.
        self.awake_until: float | None = None
        self.on_sleep = on_sleep  # on_sleep(reason) once the app falls asleep
        self.commands: list[Command] = []  # local commands handled (P5.3), for tests and status
        # "Is this for me?" (P5.4)
        self.addressee = addressee or Addressee(settings)
        self.on_ignored = on_ignored
        self.ignored: list[tuple[str, Verdict]] = []
        self._reply_done_at: float | None = None
        self.activity: ActivityLog = activity or NullActivityLog()
        self.shortcuts = shortcuts  # None: no voice shortcuts
        self.prefs = prefs  # where a model choice is saved (P5.7c); None: not saved
        self.on_model = on_model  # on_model(model_info()) after the mode or models change
        self.muted = False  # P5.7k: the mic is ignored while True
        self.on_mute = on_mute  # on_mute(muted) whenever it changes (window, voice)
        self.shortcuts_fired: list[ShortcutMatch] = []
        # The brain has answered since its last reset: the next wake needs a fresh session.
        self._brain_used = False
        self.brain_resets = 0  # background resets started (P5.6), for tests and status
        self._background: set[asyncio.Task[Any]] = set()

    # --- construction ------------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        audio: bool = True,
        speaker: bool | None = None,
        clock: Clock | None = None,
    ) -> SparkApp:
        """Build every configured backend through the registry.

        ``audio=False`` (text mode) builds no mic, wake word, VAD or STT. ``speaker`` (default:
        same as ``audio``) decides whether a speaker is opened for the TTS backend.
        """
        build = registry.build
        clock = clock or SystemClock()
        prefs = Prefs.from_settings(settings)
        apply_prefs(settings, prefs)  # the model mode and models picked in the app
        speaker = audio if speaker is None else speaker
        player = build("audio_out", settings) if speaker else None
        tts = build("tts", settings, player=player)
        # The app owns the timers so they survive brain resets; the brain only exposes them as
        # tools. spark_status asks the app, which doesn't exist yet, hence the late binding.
        timers = TimerService(clock=clock)
        activity = ActivityLog.from_settings(settings)
        activity.purge()
        shortcuts = ShortcutStore.from_settings(settings) if settings.shortcuts.enabled else None
        memory = None
        if settings.memory.enabled:
            from spark.memory import MemoryStore

            memory = MemoryStore.from_settings(settings)
            memory.conversation = lambda: activity.conversation  # a note's source
        app_ref: list[SparkApp] = []

        def briefing() -> str:
            from spark.briefing import build_briefing

            brain_now = app_ref[0].brain if app_ref else None
            connectors = getattr(brain_now, "external_connectors", None) or []
            return build_briefing(settings, activity, memory=memory, connectors=connectors)

        def status_provider() -> dict[str, Any]:
            return app_ref[0].status_snapshot() if app_ref else {}

        confirmer, line_reader = _build_confirmer(settings, app_ref, audio=audio)
        brain = build(
            "brain",
            settings,
            timers=timers,
            status_provider=status_provider,
            confirmer=confirmer,
            activity=activity,
            context_provider=briefing,
            memory=memory,
        )
        if not audio:
            app = cls(
                settings,
                mic=None,
                player=player,
                wakeword=None,
                vad=None,
                stt=None,
                tts=tts,
                brain=brain,
                clock=clock,
                timers=timers,
                confirmer=confirmer,
                line_reader=line_reader,
                activity=activity,
                shortcuts=shortcuts,
                prefs=prefs,
            )
        else:
            app = cls(
                settings,
                mic=build("audio_in", settings),
                player=player,
                wakeword=build("wakeword", settings),
                vad=build("vad", settings),
                stt=build("stt", settings),
                tts=tts,
                brain=brain,
                clock=clock,
                timers=timers,
                confirmer=confirmer,
                activity=activity,
                shortcuts=shortcuts,
                prefs=prefs,
            )
        app_ref.append(app)
        return app

    async def warmup(self) -> None:
        """Pre-load models, open the speaker and connect the brain concurrently (spec §9)."""
        parts: list[Any] = [self.tts.warmup(), self.brain.warmup()]
        if self.stt is not None:
            parts.append(self.stt.warmup())
            parts.append(self.addressee.warmup())
        if self.player is not None:
            parts.append(self.player.warmup())
        await asyncio.gather(*parts)
        await self._prepare_wake_reply()

    async def _prepare_wake_reply(self) -> None:
        """Synthesize the persona's wake reply once, so the wake word gets an instant answer."""
        text = self.voices.wake_reply
        render = getattr(self.tts, "render", None)
        if not text or self.player is None or render is None:
            return
        try:
            pcm, rate = await asyncio.to_thread(render, text)
        except Exception as exc:
            log.warning("wake reply %r not prepared, using the chime: %s", text, exc)
            return
        if pcm.size:
            self._wake_clip = (pcm, rate)

    async def close(self) -> None:
        self._closed = True
        await self.cancel_turn()
        await self._cancel_announcement()
        await self.wait_interrupted()
        for task in list(self._background):
            task.cancel()
        if self._background:
            await asyncio.wait(self._background, timeout=UNWIND_TIMEOUT_S)
        if self.timers is not None:
            with contextlib.suppress(Exception):
                await self.timers.close()
        for closer in (self.brain.close, self.tts.close, self.addressee.aclose):
            with contextlib.suppress(Exception):
                await closer()
        if self.stt is not None:
            with contextlib.suppress(Exception):
                await self.stt.close()
        if self.wakeword is not None:
            with contextlib.suppress(Exception):
                self.wakeword.close()
        if self.player is not None:
            with contextlib.suppress(Exception):
                await self.player.close()
        self.activity.end_conversation("app closed")
        self.transcripts.close()

    # --- state -------------------------------------------------------------------------------

    def set_state(self, new: State) -> None:
        old = self.state
        if new is old:
            return
        self.state = new
        self.state_history.append(new)
        log.debug("state %s -> %s", old.value, new.value)
        if self.on_state is not None:
            self.on_state(old, new)
        if new in _READY_STATES:
            self._maybe_announce()

    # --- status and timers -------------------------------------------------------------------

    def status_snapshot(self) -> dict[str, Any]:
        """What ``spark_status`` reports: state, backends, model, uptime, connectors.

        Names only, never keys or tokens (``describe_status`` also drops anything secret-like).
        """
        s = self.settings
        backends: dict[str, str] = {}
        if self.wakeword is not None:
            backends["wakeword"] = s.wakeword.backend
        if self.stt is not None:
            backends["stt"] = s.stt.backend
        backends["tts"] = s.tts.backend
        backends["brain"] = s.brain.backend
        snapshot: dict[str, Any] = {
            "state": self.state.value,
            "backends": backends,
            "uptime_s": round(max(0.0, self.clock.monotonic() - self._started_at), 1),
            # external connectors only; the in-process spark server is Spark itself
            "connectors": [c for c in getattr(self.brain, "connectors", []) if c != "spark"],
        }
        model = getattr(self.brain, "model", None)
        if isinstance(model, str) and model:
            snapshot["model"] = model
        return snapshot

    def _on_timer_fired(self, timer: Timer) -> None:
        text = timer_announcement(timer, self.voices)
        log.info("timer done: %s", timer.spoken_name)
        self._pending_announcements.append((text, TIMER_EARCON))
        self._maybe_announce()

    def _maybe_announce(self) -> None:
        if self._closed or not self._pending_announcements or self.state not in _READY_STATES:
            return
        if self._announce_task is not None and not self._announce_task.done():
            return
        self._announce_task = asyncio.create_task(self._announce(), name="spark-announce")

    async def _announce(self) -> None:
        while self._pending_announcements and self.state in _READY_STATES and not self._closed:
            text, earcon = self._pending_announcements.pop(0)
            in_follow_up = self.state is State.FOLLOW_UP
            gen = self._generation
            self._announcing = True
            try:
                if earcon:
                    self._earcon(earcon)
                if self.on_announce is not None:
                    self.on_announce(text)
                log.info("spark: %s", text)
                self.transcripts.assistant(text)
                self.activity.record("announce", text=text)
                self.announcements.append(text)
                async with self._speak_lock:
                    try:
                        await self.tts.speak(_single(text))
                    finally:
                        if (barge := self._barge_for(gen)) is not None:
                            barge.note_speak_returned()
            except asyncio.CancelledError:
                if self._barge_for(gen) is None:
                    # (After a barge-in, barge_in() already stopped the TTS. Stopping it again
                    # here would also stop the player and cut off the wake earcon.)
                    with contextlib.suppress(Exception):
                        await self.tts.stop()
                raise
            except Exception as exc:
                log.warning("timer announcement failed: %s", exc)
            finally:
                self._announcing = False
            if in_follow_up and self.state is State.FOLLOW_UP and self.recorder is not None:
                # Spark's own voice was not listened to; give the user a fresh follow-up window.
                self._start_recording(follow_up=True)

    async def _cancel_announcement(self) -> None:
        task = self._announce_task
        self._announce_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _earcon(self, name: str) -> None:
        if self.player is not None and self.settings.app.earcons:
            self.player.play_earcon(name)

    # --- voice loop --------------------------------------------------------------------------

    async def run(self) -> None:
        """Consume the mic until :meth:`stop` (or the mic stream ends)."""
        if self.mic is None:
            raise RuntimeError("run() needs audio components; use text_turn() in text mode")
        if self._stopping:  # stop() arrived before this loop started
            self._stopping = False
            return
        await self.mic.start()
        try:
            async for block in self.mic.frames():
                await self.feed(block)
                if self._stopping:
                    break
        finally:
            await self.mic.stop()
            self._stopping = False  # the desktop app can run() again after stop()

    async def stop(self) -> None:
        self._stopping = True
        if self.mic is not None:
            await self.mic.stop()

    async def feed(self, block: np.ndarray) -> None:
        """Advance the state machine with one mic block (any size)."""
        if self.muted:
            # Nothing from the mic is used or kept. Only the awake window's clock runs on.
            remaining = self.awake_remaining()
            expired = remaining is not None and remaining <= 0
            if self.state is State.FOLLOW_UP and expired and not self._announcing:
                self.go_to_sleep("awake window over")
            await asyncio.sleep(0)
            return
        self._ring.write(block)
        barge = self.settings.app.barge_in
        match self.state:
            case State.IDLE:
                if (heard := self._wake_heard_at(block)) is not None:
                    if self._announce_active():
                        # The user spoke over a timer announcement: stop it and listen.
                        await self.barge_in(detected_at=heard)
                    else:
                        self._on_wake()
            case State.FOLLOW_UP if self._announcing:
                # Spark's own voice is not a follow-up; only the wake word interrupts it.
                if barge and (heard := self._wake_heard_at(block)) is not None:
                    await self.barge_in(detected_at=heard)
            case State.LISTENING | State.FOLLOW_UP:
                self._listen(block)
            case State.CONFIRMING:
                # The wake word is never an answer: it cancels, so the tool is denied.
                if barge and (heard := self._wake_heard_at(block)) is not None:
                    await self.barge_in(detected_at=heard)
                else:
                    self._listen_for_answer_block(block)
            case State.SPEAKING | State.THINKING:
                if barge and (heard := self._wake_heard_at(block)) is not None:
                    await self.barge_in(detected_at=heard)
        await asyncio.sleep(0)

    def _wake_heard_at(self, block: np.ndarray) -> float | None:
        """``time.perf_counter()`` right after the wake frame was processed, or None."""
        return time.perf_counter() if self._wake_detected(block) else None

    def _announce_active(self) -> bool:
        return self._announce_task is not None and not self._announce_task.done()

    def _wake_detected(self, block: np.ndarray) -> bool:
        assert self.wakeword is not None and self._wake_chunker is not None
        fired = False
        for frame in self._wake_chunker.feed(block):
            if self.wakeword.process(frame):
                fired = True
        if fired:
            self.wakeword.reset()
        return fired

    def awake_remaining(self) -> float | None:
        """Seconds left in the awake window, or None while asleep."""
        if self.awake_until is None:
            return None
        return max(0.0, self.awake_until - self.clock.monotonic())

    def _wake_up(self) -> None:
        """Open (or restart) the awake window from now."""
        self.awake_until = self.clock.monotonic() + self.settings.awake_s

    def go_to_sleep(self, reason: str) -> None:
        """Leave the awake window: sleep chime, IDLE (wake word only), ``on_sleep(reason)``."""
        was_awake = self.awake_until is not None
        self.awake_until = None
        self.activity.end_conversation(reason)
        if self.recorder is not None and self.state is State.FOLLOW_UP:
            self.recorder.start(None, no_speech_timeout_s=0.0)  # drop any half-heard speech
        self.set_state(State.IDLE)
        if not was_awake:
            return
        log.info("asleep (%s)", reason)
        self._prepare_next_conversation()
        if self.settings.attention.sleep_earcon:
            self._earcon(self.settings.attention.sleep_earcon)
        if self.on_sleep is not None:
            try:
                self.on_sleep(reason)
            except Exception as exc:
                log.warning("on_sleep failed: %s", exc)

    def _start_recording(self, *, follow_up: bool, ignore_s: float = 0.0) -> None:
        assert self.recorder is not None
        if follow_up:
            if self.awake_until is None:
                self._wake_up()
            remaining = self.awake_remaining() or 0.0
            self.recorder.start(None, no_speech_timeout_s=remaining)
        else:
            pre_roll = self._ring.read_latest(
                int(SAMPLE_RATE * self.settings.recorder.pre_roll_ms / 1000)
            )
            self.recorder.start(pre_roll, ignore_s=ignore_s)

    def _on_wake(self) -> None:
        self.awake_until = None  # LISTENING; the window opens again after the reply
        self._timer = TurnTimer(self.clock, mode="voice", **self._backend_meta())
        self._timer.mark("wake")
        log.info("wake word")
        ignore_s = self._wake_cue()
        self._start_recording(follow_up=False, ignore_s=ignore_s)
        self.set_state(State.LISTENING)

    def _wake_cue(self) -> float:
        """Answer the wake word: the persona's spoken reply when it is ready, else the chime.
        Returns how long the recorder should not count as the user's speech (the reply and a
        little output latency), since there is no echo cancellation."""
        clip = self._wake_clip
        play_clip = getattr(self.player, "play_clip", None)
        if clip is None or play_clip is None:
            self._earcon("wake")
            return 0.0
        pcm, rate = clip
        play_clip(pcm, rate)
        return pcm.size / rate + WAKE_REPLY_TAIL_S

    def _listen(self, block: np.ndarray) -> None:
        rec = self.recorder
        assert rec is not None
        rec.add(block)
        if not rec.done:
            return
        if rec.reason == "no_speech":
            self._timer = None
            if self.state is State.FOLLOW_UP:
                self.go_to_sleep("awake window over")
            else:
                log.info("no speech; back to idle")
                self.set_state(State.IDLE)
            return
        # end_of_speech, or max_length with speech heard: transcribe what we have
        timer = self._timer or TurnTimer(self.clock, mode="follow_up", **self._backend_meta())
        self._timer = None
        timer.mark("speech_end")
        timer.annotate(utterance_s=round(rec.duration_s, 3), stop_reason=rec.reason)
        pcm = rec.pcm()
        if self.settings.debug_audio.enabled:
            self._save_debug_audio(rec)
        self.set_state(State.THINKING)
        self.turn_task = asyncio.create_task(self._voice_turn(pcm, timer))

    def _save_debug_audio(self, rec: Recorder) -> None:
        try:
            path = debug_audio_path(self.settings)
            path.parent.mkdir(parents=True, exist_ok=True)
            rec.save_wav(path)
            log.info("saved utterance to %s", path)
        except OSError as exc:
            log.warning("could not save debug audio: %s", exc)

    async def _voice_turn(self, pcm: np.ndarray, timer: TurnTimer) -> None:
        gen = self._generation
        follow_up = timer.meta.get("mode") != "voice"
        try:
            if self.settings.app.thinking_earcon and not follow_up:
                self._earcon("thinking")
            assert self.stt is not None
            raw = await self.stt.transcribe(pcm)
            timer.mark("stt_done")
            if gen != self._generation:
                self._finish_turn(timer, [], gen=gen)  # barged in while transcribing
                return
            # the pre-roll can carry "Hey Spark" into a wake turn; follow-ups have no wake word
            text = (
                strip_spoken_prefix(
                    strip_wake_phrase(raw, self.settings.app.wake_names),
                    self.voices.wake_reply if self._wake_clip is not None else "",
                )
                if timer.meta.get("mode") == "voice"
                else raw.strip()
            )
            if not text:
                log.info("heard nothing intelligible; back to idle")
                self.set_state(State.IDLE)
                return
            if (command := match_command(text, self.settings.app.wake_names)) is not None:
                await self._voice_command(command, text, gen)
                return
            shortcut = self._match_shortcut(text)
            if follow_up and shortcut is None:  # a shortcut is always meant for Spark
                verdict = await self._check_addressee(text)
                if gen != self._generation:
                    return
                if not verdict.addressed:
                    self._ignore(text, verdict)
                    return
            if follow_up and self.settings.app.thinking_earcon:
                self._earcon("thinking")
            await self._respond(text, timer, shortcut=shortcut)
            if gen != self._generation:
                return  # interrupted: the barge-in owns the state now (no stale FOLLOW_UP)
            if self.state is State.THINKING or self.state is State.SPEAKING:
                self._reply_done_at = self.clock.monotonic()
                self._wake_up()  # a handled request restarts the awake window
                self._start_recording(follow_up=True)
                self.set_state(State.FOLLOW_UP)
        except asyncio.CancelledError:
            if not timer.has("turn_done"):
                self._finish_turn(timer, [], gen=gen)  # cancelled before the reply started
            raise
        except Exception as exc:
            if gen != self._generation:
                log.info("interrupted turn ended with %s", exc)
                return
            self._fail(exc)

    # --- voice shortcuts (P5.7a) ----------------------------------------------------------------

    def _match_shortcut(self, text: str) -> ShortcutMatch | None:
        if self.shortcuts is None:
            return None
        try:
            items = self.shortcuts.all()
        except Exception as exc:
            log.warning("shortcuts unavailable: %s", exc)
            return None
        return match_shortcut(
            text,
            items,
            names=self.settings.app.wake_names,
            max_extra_words=self.settings.shortcuts.max_extra_words,
        )

    # --- mute (P5.7k) ---------------------------------------------------------------------------

    def set_muted(self, muted: bool, *, source: str = "window") -> None:
        """Stop (True) or resume (False) listening. Work under way is not touched."""
        muted = bool(muted)
        if muted == self.muted:
            return
        self.muted = muted
        log.info("%s (%s)", "muted" if muted else "unmuted", source)
        self.activity.record("mute", muted=muted, source=source)
        if muted:
            self._abort_answer()  # a pending "Should I go ahead?" gets no answer: not run
            if self.state is State.LISTENING:
                self._timer = None
                self.set_state(State.IDLE)  # drop the half-heard request
        else:
            if self.wakeword is not None:
                self.wakeword.reset()
                self._wake_chunker = Rechunker(self.wakeword.frame_length)
            if self.state is State.FOLLOW_UP and self.recorder is not None:
                remaining = self.awake_remaining()
                if remaining is not None and remaining > 0 and not self._announcing:
                    self._start_recording(follow_up=True)  # fresh: nothing from while muted
                elif not self._announcing:
                    self.go_to_sleep("awake window over")
        if self.on_mute is not None:
            try:
                self.on_mute(muted)
            except Exception as exc:
                log.debug("on_mute failed: %s", exc)

    # --- "is this for me?" (P5.4) ---------------------------------------------------------------

    async def _check_addressee(self, text: str) -> Verdict:
        since = None
        if self._reply_done_at is not None:
            since = self.clock.monotonic() - self._reply_done_at
        try:
            return await self.addressee.decide(
                text, last_reply=self.last_reply, since_reply_s=since
            )
        except Exception as exc:  # never lose a request to a broken check
            log.warning("addressee check failed (%s); answering", exc)
            return Verdict(True, "check failed")

    def _ignore(self, text: str, verdict: Verdict) -> None:
        """Speech not meant for Spark: log it and keep listening for the rest of the window."""
        log.info("ignored (%s, %s): %s", verdict.reason, verdict.by, text)
        self.ignored.append((text, verdict))
        self.activity.record("ignored", text=text, reason=verdict.reason, by=verdict.by)
        if self.on_ignored is not None:
            try:
                self.on_ignored(text, verdict)
            except Exception as exc:
                log.debug("on_ignored failed: %s", exc)
        remaining = self.awake_remaining()
        if remaining is None or remaining <= 0 or self.recorder is None:
            self.go_to_sleep("awake window over")
            return
        self._start_recording(follow_up=True)
        self.set_state(State.FOLLOW_UP)

    # --- local commands (P5.3) -----------------------------------------------------------------

    def _spawn(self, coro: Any, name: str) -> asyncio.Task[Any]:
        """Run ``coro`` in the background; :meth:`close` cancels what is still running."""
        task = asyncio.create_task(coro, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_log_background_failure)
        return task

    def _note_command(self, command: Command, text: str) -> None:
        self.commands.append(command)
        log.info("you: %s", text)
        log.info("command: %s", command.kind)
        self.transcripts.user(text)
        if command.kind != NEW_CONVERSATION:
            self.activity.ensure_conversation("command")
        self.activity.record("command", command=command.kind, text=text)
        if command.kind == NEW_CONVERSATION:
            self.activity.end_conversation("new conversation")

    def _start_new_conversation(self) -> None:
        """Reset the brain in the background (the next turn waits for it in the brain)."""
        self._brain_used = False
        self.brain_resets += 1
        self._spawn(self.brain.reset(), "spark-new-conversation")

    def _prepare_next_conversation(self) -> None:
        """The conversation is over: get a fresh session ready while the user is quiet."""
        if self._brain_used and not self._closed:
            log.info("preparing the next conversation in the background")
            self._start_new_conversation()

    async def _say(self, text: str, gen: int) -> None:
        """Speak one short line of Spark's own (not the brain's)."""
        if gen != self._generation:
            return
        self.set_state(State.SPEAKING)
        log.info("spark: %s", text)
        self.transcripts.assistant(text)
        if self.activity.conversation is not None:
            self.activity.assistant(text)
        self.last_reply = text
        async with self._speak_lock:
            await self.tts.speak(_single(text))

    async def _voice_command(self, command: Command, text: str, gen: int) -> None:
        self._note_command(command, text)
        if command.kind == SLEEP:
            await self._say(self._command_reply(SLEEP), gen)
            if gen == self._generation:
                self.go_to_sleep("command")
        elif command.kind == CANCEL:
            if gen == self._generation:
                self.go_to_sleep("cancelled")
        elif command.kind == NEW_CONVERSATION:
            self._start_new_conversation()
            await self._say(self._command_reply(NEW_CONVERSATION), gen)
            self._stay_awake(gen)
        elif command.kind in MODE_COMMANDS:
            await self._say(self._mode_command(command.kind), gen)
            self._stay_awake(gen)
        elif command.kind == MUTE:
            await self._say(self._command_reply(MUTE), gen)
            if gen == self._generation:
                self._stay_awake(gen)  # the awake window keeps its time for after unmuting
                self.set_muted(True, source="voice")

    def _command_reply(self, kind: str) -> str:
        """What Spark says for a local command, in the persona's words ('' = nothing)."""
        key = COMMAND_REPLIES.get(kind, "")
        return self.voices.line(key) if key else ""

    def _stay_awake(self, gen: int) -> None:
        if gen == self._generation and self.recorder is not None:
            self._wake_up()
            self._start_recording(follow_up=True)
            self.set_state(State.FOLLOW_UP)

    # --- model mode (P5.7c) --------------------------------------------------------------------

    def model_info(self) -> dict[str, Any]:
        return model_info(self.settings, mode=getattr(self.brain, "mode", None))

    def set_model_mode(self, mode: str) -> None:
        """Auto / always fast / always deep: applied to the brain and saved for next time."""
        if mode not in MODES:
            raise ValueError(f"unknown model mode {mode!r}")
        setter = getattr(self.brain, "set_mode", None)
        if setter is not None:
            setter(mode)
        self.settings.brain.mode = mode
        if self.prefs is not None:
            self.prefs.update(model_mode=mode)
        self.activity.record("model", mode=mode)
        self._notify_model()

    def set_models(
        self, fast: str | None = None, deep: str | None = None, worker: str | None = None
    ) -> None:
        """Change the model behind each route (``worker``: the researcher, '' = same as fast,
        None = unchanged), saved for next time."""
        setter = getattr(self.brain, "set_models", None)
        if setter is not None:
            setter(fast, deep, worker)
        # The brain shares this config; set it for other brains too (the active backend's
        # section: Claude ids, or brain.openai_compat).
        set_route_models(self.settings, fast, deep, worker)
        if self.prefs is not None:
            self.prefs.update(**pref_updates(self.settings, fast or None, deep or None, worker))
        self.activity.record("model", fast_model=fast, deep_model=deep, worker_model=worker)
        self._notify_model()

    def _notify_model(self) -> None:
        if self.on_model is not None:
            try:
                self.on_model(self.model_info())
            except Exception as exc:
                log.debug("on_model failed: %s", exc)

    def _mode_command(self, kind: str) -> str:
        """Apply a spoken mode command; returns what Spark says."""
        info = self.model_info()
        fast, deep = info["fast_label"], info["deep_label"]
        if kind == MODE_DEEP:
            self.set_model_mode("deep")
            return f"Okay, I'll use {deep} for everything until you say back to auto."
        if kind == MODE_FAST:
            self.set_model_mode("fast")
            return f"Okay, I'll use {fast} for everything until you say back to auto."
        if kind == MODE_AUTO:
            self.set_model_mode("auto")
            return f"Okay, back to auto: {fast} for quick questions, {deep} for bigger ones."
        mode = info["mode"]
        research = f" {info['worker_label']} does the research." if self._researcher_on() else ""
        if mode == "auto":
            return f"Auto: {fast} for quick questions, {deep} for bigger ones.{research}"
        return f"I'm using {deep if mode == 'deep' else fast} for everything.{research}"

    def _researcher_on(self) -> bool:
        return bool(getattr(self.brain, "workers_active", False))

    def _fail(self, exc: BaseException) -> None:
        self.errors.append(exc)
        self.activity.error(exc)
        self.activity.end_conversation("failed")
        self._prepare_next_conversation()
        self.awake_until = None  # after a failure the wake word starts over
        log.error("turn failed: %s", exc, exc_info=exc)
        self._earcon("error")
        if self.player is not None and not self._closed:
            # Spoken after the error beep, as an announcement once the app is back in IDLE.
            self._pending_announcements.append((failure_line(exc, self.voices), None))
        self.set_state(State.IDLE)

    # --- barge-in (P3.3) ---------------------------------------------------------------------

    @property
    def stop_latency_ms(self) -> float | None:
        """Wake frame -> ``player.stop()`` returned, for the most recent barge-in."""
        return self.last_barge_in.stop_latency_ms if self.last_barge_in is not None else None

    def _barge_for(self, gen: int) -> BargeIn | None:
        """The barge-in that interrupted work started in generation ``gen``, if any."""
        barge = self.last_barge_in
        if barge is not None and gen != self._generation and barge.generation == gen + 1:
            return barge
        return None

    async def barge_in(self, *, detected_at: float | None = None) -> BargeIn:
        """The wake word interrupted Spark: silence now, drop the turn, listen again.

        ``detected_at`` is the ``time.perf_counter()`` at which the wake frame was processed
        (default: now). Returns the :class:`BargeIn` timing record.
        """
        heard = time.perf_counter() if detected_at is None else detected_at
        # 1. Silence first. This is synchronous: the next output block is already quiet.
        if self.player is not None:
            self.player.stop()
        stopped = time.perf_counter()
        source = self.state
        self._generation += 1
        barge = BargeIn(
            source=source.value,
            generation=self._generation,
            detected_at=heard,
            stop_latency_ms=_ms(stopped - heard),
        )
        self.last_barge_in = barge
        self.barge_ins.append(barge)
        log.info("barge-in from %s: stop %.1f ms", source.value, barge.stop_latency_ms)
        # 2. No further sentence may be synthesized or played. Done before the wake earcon is
        #    queued, because stopping the TTS stops the player too.
        with contextlib.suppress(Exception):
            await self.tts.stop()
        queue = self._speech_queue
        if queue is not None:
            _abort_questions(queue, [])  # drop pending sentences, fail unspoken questions
        self._abort_answer()  # a pending confirmation gets no answer, so the gate denies it
        turn, self.turn_task = self.turn_task, None
        if turn is not None and turn.done():
            turn = None
        if turn is not None:
            turn.cancel()
        announce, self._announce_task = self._announce_task, None
        if announce is not None and announce.done():
            announce = None
        if announce is not None:
            announce.cancel()
        # 3. Listen again right away (wake earcon, pre-roll, LISTENING).
        self._on_wake()
        if self._timer is not None:
            self._timer.annotate(after_barge_in=True)
        # 4. Abort the brain and let the cancelled tasks unwind without blocking the mic loop.
        task = asyncio.create_task(
            self._finish_interrupt(barge, turn, announce), name="spark-barge-in-cleanup"
        )
        self._interrupt_tasks.add(task)
        task.add_done_callback(self._interrupt_tasks.discard)
        return barge

    async def _finish_interrupt(
        self,
        barge: BargeIn,
        turn: asyncio.Task[None] | None,
        announce: asyncio.Task[None] | None,
    ) -> None:
        jobs: list[asyncio.Future[Any]] = [t for t in (turn, announce) if t is not None]
        interrupt = None
        if turn is not None:
            interrupt = asyncio.ensure_future(self._interrupt_brain())
        if jobs:
            _, pending = await asyncio.wait(jobs, timeout=UNWIND_TIMEOUT_S)
            for job in pending:
                log.warning("interrupted task %s did not stop in %.1f s", job, UNWIND_TIMEOUT_S)
            for job in jobs:
                if job.done() and not job.cancelled():
                    job.exception()  # mark retrieved; the turn already logged its own error
            barge.unwind_ms = _ms(time.perf_counter() - barge.detected_at)
        if interrupt is not None:
            barge.brain_interrupted = await interrupt

    async def _interrupt_brain(self) -> bool:
        interrupt = getattr(self.brain, "interrupt", None)
        if interrupt is None:
            return False
        try:
            await interrupt()
        except Exception as exc:
            log.warning("brain interrupt failed: %s", exc)
            return False
        return True

    async def wait_interrupted(self) -> None:
        """Wait until every barge-in cleanup (brain interrupt, task unwinding) has finished."""
        while pending := {t for t in self._interrupt_tasks if not t.done()}:
            await asyncio.wait(pending)

    async def cancel_turn(self) -> None:
        self._abort_answer()
        task = self.turn_task
        self.turn_task = None
        if task is None or task.done():
            return
        self._generation += 1
        task.cancel()
        with contextlib.suppress(Exception):
            await self.tts.stop()
        with contextlib.suppress(Exception):
            await self.brain.interrupt()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # --- shared turn pipeline ----------------------------------------------------------------

    async def _respond(
        self, text: str, timer: TurnTimer, *, shortcut: ShortcutMatch | None = None
    ) -> str:
        """Brain -> sentence splitter -> sanitizer -> TTS. Returns the spoken reply.

        A pump task moves sanitized sentences into a queue that the TTS reads. A confirmation
        question (:meth:`say_for_confirmation`) enters the same queue. It ends the current TTS
        stream, is spoken on its own, and the reply then continues in a new TTS stream.
        """
        from spark.brain.prompts import SpeechFilter

        gen = self._generation
        # A barge-in's cleanup (brain.interrupt) must finish before the brain gets a new turn.
        await self.wait_interrupted()
        log.info("you: %s", text)
        self.transcripts.user(text)
        mode = str(timer.meta.get("mode") or "voice")
        self.activity.ensure_conversation(mode)
        self.activity.user(text, mode)
        timer.annotate(user_chars=len(text))
        request = text
        one_shot = split_model_request(text) if shortcut is None else None
        if one_shot is not None and hasattr(self.brain, "next_route"):
            route, request = one_shot  # "use the deep model to ...": this request only
            self.brain.next_route = route
            self.brain.next_route_reason = f"asked for the {route} model"
            self.brain.next_route_wins = True
        if shortcut is not None:
            request = expand(shortcut, text)
            self.shortcuts_fired.append(shortcut)
            log.info("shortcut: %s (%r)", shortcut.shortcut.name, shortcut.trigger)
            self.activity.record(
                "shortcut",
                shortcut=shortcut.shortcut.name,
                id=shortcut.shortcut.id,
                trigger=shortcut.trigger,
                route=shortcut.shortcut.route,
            )
            timer.annotate(shortcut=shortcut.shortcut.name)
            if shortcut.shortcut.route in ("fast", "deep") and hasattr(self.brain, "next_route"):
                self.brain.next_route = shortcut.shortcut.route
                self.brain.next_route_reason = f"shortcut {shortcut.shortcut.name!r}"
        spoken: list[str] = []
        speech_filter = SpeechFilter()  # per turn: drops fenced code, strips markdown/URLs
        queue: asyncio.Queue[Any] = asyncio.Queue()
        asked_before = len(self.confirm_questions)

        self._brain_used = True

        async def deltas() -> AsyncIterator[str]:
            async for delta in self.brain.respond(request):
                if delta:
                    timer.mark("brain_first_token")
                    yield delta

        async def pump() -> None:
            try:
                async for sentence in split_sentences(deltas()):
                    clean = speech_filter.feed(sentence)
                    if clean:
                        queue.put_nowait(clean)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                queue.put_nowait(_Failed(exc))
                return
            queue.put_nowait(_SPEECH_DONE)

        reports_audio = hasattr(self.tts, "on_first_audio")
        if reports_audio:
            self.tts.on_first_audio = lambda: timer.mark("first_audio")  # type: ignore[attr-defined]

        finished = False
        segment_over = False
        questions: list[_Question] = []

        async def segment() -> AsyncIterator[str]:
            """Sentences for one TTS stream; it ends at the end of the reply or at a question."""
            nonlocal finished, segment_over
            while True:
                item = await queue.get()
                if item is _SPEECH_DONE:
                    finished = segment_over = True
                    return
                if isinstance(item, _Failed):
                    finished = segment_over = True
                    raise item.exc
                if isinstance(item, _Question):
                    questions.append(item)
                    segment_over = True
                    return
                timer.mark("first_sentence")
                if not reports_audio:
                    timer.mark("first_audio")
                    timer.annotate(first_audio_is_handoff=True)
                if gen != self._generation:
                    return  # interrupted: nothing more from this reply is spoken
                if self.state is State.THINKING:
                    self.set_state(State.SPEAKING)
                spoken.append(item)
                yield item

        try:
            async with self._speak_lock:
                self._speech_queue = queue
                pump_task = asyncio.create_task(pump(), name="spark-speech-pump")
                try:
                    while not finished and gen == self._generation:
                        segment_over = False
                        await self.tts.speak(segment())
                        if not segment_over:
                            break  # TTS was stopped (barge-in / cancel): no new stream
                        while questions:
                            await self._speak_question(questions.pop(0))
                finally:
                    if (barge := self._barge_for(gen)) is not None:
                        barge.note_speak_returned()
                    if self._speech_queue is queue:
                        self._speech_queue = None
                    pump_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await pump_task
                    _abort_questions(queue, questions)
        except asyncio.CancelledError:
            asked = len(self.confirm_questions) - asked_before
            self._finish_turn(timer, spoken, gen=gen, asked=asked)
            raise
        asked = len(self.confirm_questions) - asked_before
        return self._finish_turn(timer, spoken, gen=gen, asked=asked)

    def _finish_turn(self, timer: TurnTimer, spoken: list[str], *, gen: int, asked: int = 0) -> str:
        """Log a finished turn's reply and latency record. Returns the reply text.

        A turn whose generation is stale (a barge-in, or ``cancel_turn``) is logged with
        ``interrupted=true`` plus the barge-in timings, and doesn't become :attr:`last_reply`.
        """
        interrupted = gen != self._generation
        reply = " ".join(spoken)
        timer.mark("turn_done")
        timer.annotate(reply_chars=len(reply))
        if asked:
            timer.annotate(confirmations=asked)  # the wait for an answer is inside turn_done
        if interrupted:
            timer.annotate(interrupted=True)
            if (barge := self._barge_for(gen)) is not None:
                barge.note_speak_returned()
                timer.annotate(**barge.as_meta())
        else:
            self.last_reply = reply
        self.last_timer = timer
        if reply:
            log.info("spark: %s%s", reply, " [interrupted]" if interrupted else "")
            self.transcripts.assistant(reply + (" [interrupted]" if interrupted else ""))
            self.activity.assistant(reply, interrupted=interrupted)
        self.latency_log.append(timer)
        log.info("latency %s", format_breakdown(timer.breakdown()))
        if self.on_turn is not None and not interrupted:
            self.on_turn(timer)  # benchmarks count completed turns only
        return reply

    # --- spoken confirmation (P3.1) ----------------------------------------------------------

    async def say_for_confirmation(self, text: str) -> None:
        """Speak a confirmation question and return once it has finished playing.

        During a turn, the question goes through the turn's speech queue, so it follows what
        was already said. Outside a turn, it is spoken directly.
        """
        if self.state is State.LISTENING:
            # A barge-in cancelled the turn that wanted to ask: never speak its question.
            raise ConfirmationAborted("the user interrupted with the wake word")
        text = self.voices.confirm_line(text)
        self.confirm_questions.append(text)
        log.info("spark asks: %s", text)
        self.transcripts.assistant(text)
        self.activity.record("confirm_question", text=text)
        queue = self._speech_queue
        if queue is None:
            async with self._speak_lock:
                await self.tts.speak(_single(text))
            return
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        queue.put_nowait(_Question(text, done))
        await done

    async def _speak_question(self, question: _Question) -> None:
        try:
            await self.tts.speak(_single(question.text))
        except asyncio.CancelledError:
            if not question.done.done():
                question.done.set_exception(ConfirmationAborted("the turn was cancelled"))
            raise
        except Exception as exc:
            if not question.done.done():
                question.done.set_exception(exc)
            return
        if not question.done.done():
            question.done.set_result(None)

    async def confirmation_cue(self) -> None:
        """Play the "your turn" earcon and let it finish, so the recorder doesn't hear it."""
        name = self.settings.confirm.earcon
        if not name or self.player is None or not self.settings.app.earcons:
            return
        if self.state not in _TURN_STATES:
            return  # the asking turn was interrupted
        self._earcon(name)
        for _ in range(100):  # at most ~1 s; earcons are a few hundred ms
            await asyncio.sleep(0.01)
            if not self.player.is_playing:
                return

    async def listen_for_answer(self, timeout_s: float) -> str:
        """Record one utterance in ``State.CONFIRMING`` and return its transcript.

        Returns '' if nobody spoke within ``timeout_s``, if the app has no mic, recorder, or STT,
        or if the turn was cancelled or interrupted. The previous state (THINKING) is restored
        afterwards.
        """
        if self.recorder is None or self.stt is None or self.mic is None or self._closed:
            return ""
        if self.muted:
            log.info("muted: the confirmation gets no answer")
            return ""  # can't hear while muted: no answer, so the tool doesn't run
        if self.state not in _TURN_STATES:
            return ""  # the asking turn was interrupted (barge-in): no answer, so denied
        loop = asyncio.get_running_loop()
        answer: asyncio.Future[str] = loop.create_future()
        self._answer = answer
        self.recorder.start(None, no_speech_timeout_s=timeout_s)
        self.set_state(State.CONFIRMING)
        try:
            outcome = await answer
            if outcome != "speech":
                return ""
            raw = await self.stt.transcribe(self.recorder.pcm())
            text = strip_wake_phrase(raw, self.settings.app.wake_names)
            log.info("you (answer): %s", text)
            self.transcripts.user(text)
            self.activity.user(text, "answer")
            return text
        finally:
            if self._answer is answer:
                self._answer = None
            if self.state is State.CONFIRMING:
                self.set_state(State.THINKING)

    def _listen_for_answer_block(self, block: np.ndarray) -> None:
        answer = self._answer
        rec = self.recorder
        if answer is None or answer.done() or rec is None:
            return
        rec.add(block)
        if rec.done:
            answer.set_result("silence" if rec.reason == "no_speech" else "speech")

    def _abort_answer(self) -> None:
        answer, self._answer = self._answer, None
        if answer is not None and not answer.done():
            answer.set_result("cancelled")

    async def text_turn(self, text: str) -> str:
        """One typed turn (``--text`` mode): brain + TTS, no mic, STT or wake word."""
        text = text.strip()
        if not text:
            return ""
        if (command := match_command(text, self.settings.app.wake_names)) is not None:
            self._note_command(command, text)
            if command.kind == NEW_CONVERSATION:
                await self.brain.reset()
            elif command.kind in (SLEEP, CANCEL):
                self.activity.end_conversation("command")
            reply = self._command_reply(command.kind)
            if command.kind in MODE_COMMANDS:
                reply = self._mode_command(command.kind)
            if reply:
                self.transcripts.assistant(reply)
                self.last_reply = reply
            return reply
        timer = TurnTimer(self.clock, mode="text", **self._backend_meta())
        timer.mark("stt_done")
        self.set_state(State.THINKING)
        try:
            return await self._respond(text, timer, shortcut=self._match_shortcut(text))
        except Exception as exc:
            self._fail(exc)
            raise
        finally:
            if self.state is not State.IDLE:
                self.set_state(State.IDLE)

    def _backend_meta(self) -> dict[str, str]:
        s = self.settings
        return {"stt": s.stt.backend, "tts": s.tts.backend, "brain": s.brain.backend}


_READY_STATES = (State.IDLE, State.FOLLOW_UP)

#: The persona line (spark/persona.py) Spark says for a local command ("" = nothing; the sleep
#: chime is the answer).
COMMAND_REPLIES: dict[str, str] = {
    SLEEP: "sleep",
    CANCEL: "",
    NEW_CONVERSATION: "fresh_start",
    MUTE: "muted",
}

#: Added to the wake reply's length before the recorder counts speech: output latency.
WAKE_REPLY_TAIL_S = 0.15


def _log_background_failure(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        log.warning("background task %s failed: %s", task.get_name(), exc)


_TURN_STATES = (State.THINKING, State.SPEAKING)  # a turn is running and may ask for confirmation

_SPEECH_DONE = object()  # end of the reply in a turn's speech queue


class _Failed:
    """The brain stream failed; re-raised inside TTS so the turn fails as before."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _Question:
    """A confirmation question waiting in the speech queue; ``done`` resolves once it's spoken."""

    def __init__(self, text: str, done: asyncio.Future[None]) -> None:
        self.text = text
        self.done = done


def _abort_questions(queue: asyncio.Queue[Any], pending: list[_Question]) -> None:
    """The turn ended with questions still unspoken: fail them, so the gate says no."""
    items: list[Any] = list(pending)
    while not queue.empty():
        items.append(queue.get_nowait())
    for item in items:
        if isinstance(item, _Question) and not item.done.done():
            item.done.set_exception(ConfirmationAborted("the turn ended before the question"))


def _build_confirmer(
    settings: Settings, app_ref: list[SparkApp], *, audio: bool
) -> tuple[Confirmer, LineReader | None]:
    """VoiceConfirmer in voice mode, TextConfirmer (stdin) in ``--text`` mode.

    The app does not exist yet when the brain is built, so the callbacks look it up late.
    """
    cfg = settings.confirm

    async def say(text: str) -> None:
        await app_ref[0].say_for_confirmation(text)

    if not audio:
        reader = LineReader()
        return (
            TextConfirmer(
                reader=reader,
                say=say,
                timeout_s=cfg.text_timeout_s,
                yes_phrases=cfg.yes_phrases,
                no_phrases=cfg.no_phrases,
            ),
            reader,
        )

    async def listen(timeout_s: float) -> str:
        return await app_ref[0].listen_for_answer(timeout_s)

    async def cue() -> None:
        await app_ref[0].confirmation_cue()

    return (
        VoiceConfirmer(
            listen=listen,
            say=say,
            cue=cue,
            timeout_s=cfg.timeout_s,
            reprompt=cfg.reprompt_unclear,
            yes_phrases=cfg.yes_phrases,
            no_phrases=cfg.no_phrases,
        ),
        None,
    )


async def _single(text: str) -> AsyncIterator[str]:
    yield text


def debug_audio_path(settings: Settings) -> Path:
    return settings.path(settings.debug_audio.dir) / "last_utterance.wav"
