# Personality and voice

Spark can talk in more than one way. A **personality** (`app.persona`) changes how Spark words its
answers and its own fixed lines. The **voice** (`tts`) changes how it sounds. You can set both in
the desktop app's **Personality & Voice** window (the waveform button in the toolbar), or in
`config.yaml`.

## The personalities

| Personality | How it talks |
|---|---|
| `standard` (default) | Friendly, plain and brief. "On it, give me a moment." "Should I go ahead?" |
| `butler` | A composed British butler with a dry, understated wit who calls you "sir". "Right away, sir. One moment." "Shall I proceed, sir?" |

The butler answers the wake word out loud ("Yes, sir?") instead of playing the chime, offers the
obvious next step ("Shall I set a reminder?"), and keeps every voice rule: short spoken answers
that lead with the answer, and it still asks before doing anything that changes something.

What a personality changes:

- **The model's own answers.** A short personality section is added to the voice system prompt
  (`spark/persona.py`). It works the same with Claude and with any OpenAI-compatible model,
  including local Ollama models.
- **Spark's fixed lines**, which the model never writes: the acknowledgement on long requests, the
  replies to "go to sleep" and "new conversation", "Should I go ahead?" and "was that a yes or a
  no?", timer announcements, and the error lines.
- **The wake reply.** With `app.wake_reply` on, Spark says a short line instead of the wake chime.
  It is synthesized once at startup, so it plays as quickly as the chime. While it plays, Spark
  doesn't treat what the mic hears as your request (there is no echo cancellation), so start
  talking after it, or right over its end.

## The voice

Pick a voice that suits the personality. The window offers the Kokoro voices (Apache-2.0); they
share one 340 MB download, which Spark makes on its first run for the default voice, George:

| Voice | Accent |
|---|---|
| George (default), Fable, Lewis, Daniel | British male |
| Emma | British female |
| Heart | American female |
| Michael | American male |

Kokoro sounds more natural than Piper but takes longer to speak each sentence: about 0.9 s on a
laptop CPU against Piper's 0.15 s (Apple Silicon is faster). Spark speaks sentence by sentence, so
this delays only the start of each answer.

Piper is still built in and is faster, and you can set a Piper voice in `config.yaml`
(`tts.backend: piper`), for example `en_US-ljspeech-medium`. Avoid the voices fine-tuned from
`lessac` (all the British male ones), whose dataset license allows research use only. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Please don't clone a real person's voice (an actor's, for example) without their permission.

## In config.yaml

```yaml
app:
  persona: butler        # standard | butler
  address_as: sir        # what the butler calls you; "" for no title
  wake_reply: null       # true/false; null = the personality's default (butler: on)

tts:
  backend: kokoro
  options:
    kokoro:
      voice: bm_george   # bm_ / bf_ voices speak British English automatically (lang: null)
```

Then download the voice once: `uv run spark models download` (the desktop app does this
by itself when you save). The choices made in the window are saved in `prefs.json` and win over
`config.yaml`.

## Your own name and wake word

You can rename the assistant on your own computer: set `app.name` (it replaces "Spark" in the
prompt) and `app.wake_phrase`, and [train a wake word](wakeword.md) for the new name. Keep in mind
that character names from films are usually trademarks, so don't publish an assistant under one.
