# Wake words: use "Hey Spark" or train your own

Hey Spark listens for its wake word on your machine. No audio leaves the laptop until it hears it.
You have three options:

| Option | What you say | Cost | Setup |
|---|---|---|---|
| **Bundled model** (default) | "Hey Spark" | Free, non-commercial use | None |
| **Train your own** with the Colab notebook | Anything, e.g. "Hey Nova" | Free (Google Colab GPU), about 1–2 hours | This guide |
| **Porcupine** (Picovoice) | Anything you train in their console | Free personal tier; commercial licenses available | A Picovoice key and a `.ppn` file |

## The bundled "Hey Spark" model

`assets/wakewords/hey_spark.onnx` is an [openWakeWord](https://github.com/dscripka/openWakeWord)
model, and the default `config.yaml` already uses it:

```yaml
wakeword:
  backend: openwakeword
  options:
    openwakeword:
      model_path: assets/wakewords/hey_spark.onnx
      threshold: 0.6
```

It was trained on synthetic voices only, so it works best when you speak clearly at a normal
distance from the mic. If it misses you, lower `threshold` (say to 0.5). If it wakes on its own,
raise it (0.7). Close-sounding phrases like "hey park" or "hey sport" can score near 0.8 for some
voices.

> **License:** the trained models are **CC BY-NC-SA 4.0** (non-commercial), because the
> background audio used for training carries mixed licenses. The code is MIT. See
> [`assets/wakewords/LICENSE`](../assets/wakewords/LICENSE).

## Train your own wake word

The notebook [`tools/wakeword_trainer.ipynb`](../tools/wakeword_trainer.ipynb) trains an
openWakeWord model for any phrase. It runs in Google Colab on a free T4 GPU and needs no
recordings: it synthesizes thousands of examples of your phrase with Piper TTS and mixes them
with background noise, music and room echo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/contextuai/hey-spark/blob/main/tools/wakeword_trainer.ipynb)

**It costs nothing.** The free Colab tier is enough: no Colab Pro, no payment and no API keys,
just a Google account. A few things about the free tier:

- Free GPU time is limited per day. If Colab says it can't connect to a GPU backend, try again
  later. One training run fits in a day's allowance.
- Colab disconnects sessions that sit idle, and deletes their files when it does. Keep the tab
  open, and download your model as soon as training finishes.
- The run needs about 25 GB of disk (mostly openWakeWord's 17 GB of precomputed features), which
  the free runtime has. Step 1 checks it.

### 1. Pick a good phrase

- Use **two or three syllables after "hey"**: "Hey Nova", "Hey Computer", "Okay Atlas". Very short
  words ("Hey Bo") cause false wakes; long phrases are tiring to say.
- Avoid phrases that sound like common speech ("Hey there", "Hey man") or like the TV.
- Use real words or easy-to-spell names. A made-up word needs a pronunciation (see step 4).

### 2. Open the notebook in Colab

1. Click **Open in Colab** above. (Or go to
   [colab.research.google.com](https://colab.research.google.com), choose **File → Upload
   notebook**, and pick `tools/wakeword_trainer.ipynb`.) Colab may warn that the notebook
   wasn't written by Google: choose **Run anyway**.
2. **Runtime → Change runtime type → T4 GPU → Save.** (Not TPU; it won't work.)

### 3. Run the setup cells (Steps 1–5)

Click ▶ on each cell from the top, in order. Step 1 should print `✅ GPU: Tesla T4` and enough
free disk. Steps 2–4
install packages and download models (about 10 minutes the first time). Step 5 writes the training
script.

### 4. Set your wake word (Step 6)

Type your phrase in `target_word`, for example `hey nova`. Spell it how it sounds, lowercase,
with spaces between words. Leave the sliders at their defaults the first time.

If Step 7 later fails with **"Not in CMUdict"**, your word isn't in the pronunciation dictionary.
Add it in `custom_pronunciation` as `word=ARPABET`, for example `kyla=K AY1 L AH0`. The
[CMU dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) shows the ARPAbet for similar words.

### 5. Listen before you train (Steps 7–8)

Step 7 synthesizes one test clip and Step 8 plays it. **If it doesn't sound like your phrase, fix
the spelling or pronunciation now**: the model learns exactly what you hear.

### 6. Train (Steps 9–10)

Step 9 downloads the training data (20–40 minutes, first time only). Step 10 trains (30–60
minutes on a T4). Keep the browser tab open; Colab disconnects idle sessions.

### 7. Get the model file (Steps 11–12)

Step 11 downloads `<your_phrase>.onnx` (for example `hey_nova.onnx`) to your computer. If training
saved the weights in a separate `<name>.onnx.data` file, **Step 12** bundles them into
`<name>_single.onnx` and downloads that: use it instead. Otherwise Step 12 does nothing.
To keep a copy in Google Drive, tick `save_to_drive` in Step 13 and run it.

### 8. Use it in Hey Spark

1. Copy the `.onnx` file into `assets/wakewords/`. In the installed desktop app, that's
   `%LOCALAPPDATA%\HeySpark\assets\wakewords\` on Windows and
   `~/Library/Application Support/HeySpark/assets/wakewords/` on macOS, and `config.yaml` is
   in the folder above `assets`.
2. Edit `config.yaml`:

   ```yaml
   wakeword:
     backend: openwakeword
     options:
       openwakeword:
         model_path: assets/wakewords/hey_nova.onnx
         threshold: 0.6

   app:
     name: Nova                  # what the assistant calls itself
     name_aliases: [Novah]       # how speech-to-text may spell the name
     wake_phrase: Hey Nova       # shown in the desktop window
   ```

3. Check it, then try it:

   ```powershell
   uv run python -m spark doctor     # the wake-word check should show ✔
   uv run python -m spark            # say "Hey Nova, what time is it?"
   ```

### 9. Tune it

- **Misses you:** lower `threshold` by 0.05 at a time. Speak toward the mic.
- **Wakes on its own:** raise `threshold`. Measure it with `uv run python -m spark bench
  false-wake --minutes 30` while a TV or podcast plays. Aim for fewer than one false wake in 30
  minutes.
- **Still poor:** train again with more examples (`number_of_examples`) or a higher
  `false_activation_penalty`, or choose a more distinctive phrase.

## Porcupine instead

[Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) is a commercial engine with a free
personal tier. It's very accurate, but each `.ppn` file is tied to your account, platform and
Porcupine version, so it can't be shared in this repo.

1. In the [Picovoice Console](https://console.picovoice.ai/), train your phrase for **Windows
   (x86_64)** and download the `.ppn`.
2. Save it as `assets/wakewords/hey_spark_windows.ppn` (or any name, matching `keyword_path`).
3. Put your AccessKey in `.env` as `PICOVOICE_ACCESS_KEY=...`.
4. Set `wakeword.backend: porcupine` in `config.yaml` and run `uv run python -m spark doctor`.
