# Manual QA checklist

Checks the automated test suite can't do or prove by itself: a real mic and speaker, real credentials, and real accounts. Run through it before a release, or after changing audio, connectors, or the desktop app. Work top to bottom; each section builds on the one before.
Tick an item only when its **Pass** condition holds.

## Setup
- [ ] **Secrets file.** Run `copy .env.example .env`. The repo ships only `.env.example` and never contains secrets. For Anthropic auth, either set `ANTHROPIC_API_KEY`, or log in once with `claude` → `/login`. Without it, the live brain tests are skipped.
- [ ] **Network access.** On a managed machine, confirm outbound HTTPS to api.anthropic.com, huggingface.co (model downloads), and api.picovoice.ai is allowed.
- [ ] **Wake word.** Spark wakes on a custom openWakeWord **"Hey Spark"** model: `assets/wakewords/hey_spark.onnx` (one self-contained file), set in `config.yaml` → `wakeword.options.openwakeword.model_path`, and the assistant calls itself Spark. Pass: it wakes on "Hey Spark" from across the room, and does **not** wake on "hey park", "hey sport", "hey Mark" or "it'll spark a debate" (synthetic voices pushed the first two to ~0.8). Tune `threshold` (0.6): lower it if it misses you, raise it if it wakes by itself. Porcupine is an alternative: a Windows (x86_64) `.ppn` plus `PICOVOICE_ACCESS_KEY`, and `wakeword.backend: porcupine`.
- [ ] **Models downloaded.** Run `uv run python -m spark models download`. Pass: whisper `base.en` and the Piper voice are in `.models/`.
- [ ] **Doctor passes on your machine.** Run `uv run python -m spark doctor`. Pass: every required check is ✔ and the exit code is 0. With the default openWakeWord "Hey Spark" model, Porcupine is not required.

## Voice round-trip
- [ ] **Spoken answer.** Run `uv run python -m spark`, then say "Hey Spark, what's the capital of Japan?". Pass: you hear the wake chime, then a spoken answer that mentions Tokyo.
- [ ] **Earcons are audible.** In the same session: Pass means you hear the chime on wake and the tick when thinking starts. To hear the error tone, disconnect the network and ask something.
- [ ] **Follow-up window.** After an answer, ask a follow-up within 8 s without the wake word. Pass: it's answered. After 8 s of silence Spark returns to idle (the log shows `idle`).
- [ ] **`--debug-audio`.** Run `uv run python -m spark --debug-audio`, then ask one question. Pass: `debug_audio/last_utterance.wav` plays back your question. Without the flag, no file is written.
- [ ] **Wake-word fixture tests.** Run `uv run pytest -q tests/test_fixtures_audio.py tests/test_wakeword.py -rs`. Until a custom "Hey Spark" model exists, the Hey Spark fixture tests skip by design. With a custom model: the tests run (no skips naming the key or model) and pass, with positives firing and negatives, noise, and babble not firing.
- [ ] **Real recorded fixtures.** Run `uv run python -m spark record-fixture wake_hey_spark_real` and `uv run python -m spark record-fixture cmd_01` (say command 1 from `tests/audio_fixtures/commands.yaml`), then `uv run pytest -q tests/test_fixtures_audio.py`. Pass: the tests use `recorded/` and pass.
- [ ] **Latency.** Run `uv run python -m spark bench latency --trials 10`. Pass: the median wake-to-first-audio is under 2000 ms. Keep the breakdown table (it also prints end-of-speech → first audio) for comparison.
- [ ] **Latency decision (spec §13).** If `bench latency` misses 2 s, compare `uv run python -m spark --stt deepgram bench latency --trials 10` (needs `DEEPGRAM_API_KEY` and `uv sync --extra deepgram`), or try whisper `tiny.en` (set `stt.options.faster_whisper.model: tiny.en`, then run `models download`). Keep whichever setting meets the target in `config.yaml`.
- [ ] **False wakes.** With TV or background talk playing, run `uv run python -m spark bench false-wake --minutes 30`. Pass: fewer than 1 wake.
- [ ] **Live tests.** Run `uv run pytest -q -m live`. Pass: all selected tests pass (capital of Japan, two-turn context, tool results without IDs or URLs, `${VAR}` expansion, confirm yes/no, deep routing), plus Deepgram and ElevenLabs if their keys and extras (`uv sync --extra deepgram --extra elevenlabs`) are set.

## Connectors
- [ ] **Connector credentials.** Follow `docs/connectors.md` §5 to set `SPARK_GITHUB_PAT`, `SPARK_GITHUB_MCP_BIN`, `COMPOSIO_API_KEY`, and `COMPOSIO_MCP_URL` in `.env` and set `enabled: true` for `github` and `composio` in `config.yaml`, then run `uv run python -m spark doctor`. Pass: `✔ connector: github` and `✔ connector: composio`, each listing its tools.
- [ ] **Calendar, email, and PR answers.** Follow the text-mode phrases in `docs/dev/history/phase2_acceptance.md` §1–3. Pass: the answers are correct, summarized in 1–3 sentences, and contain no IDs or URLs, in both text mode and voice.

## Actions, routing, barge-in
- [ ] **Local-only project dry run (safe, no GitHub).** Run (PowerShell) `$env:SPARK_PROJECTS__REPO_HOST="none"; uv run python -m spark --text`, or in Git Bash `SPARK_PROJECTS__REPO_HOST=none uv run python -m spark --text`. Type "start a new project called zz-local", and answer `y`. Pass: `<projects.root_dir>/zz-local` exists with one commit and no remote. Then ask for "zz-local2" and answer `n`. Pass: nothing is created. Delete `zz-local` afterwards.
- [ ] **Spoken confirmation, "no".** Run `uv run python -m spark` and say "Hey Spark, start a new project called zz-test." Spark should say "I'll create a project called zz-test: … Should I go ahead?". Answer "no". Pass: Spark acknowledges, and nothing is created (no folder under `projects.root_dir`, no GitHub repo). The log shows `tool confirm: mcp__spark__start_project -> no`.
- [ ] **Spoken confirmation, silence.** Repeat the request and say nothing for 8 s. Pass: Spark says it didn't do it, the log shows `-> timeout`, and nothing is created.
- [ ] **Unclear answer by voice.** When Spark asks "Should I go ahead?", say "hmm". Pass: Spark asks "Sorry, was that a yes or a no?". Then say "no". Pass: nothing is created.
- [ ] **New-project flow for real** (creates a real private repo; use a throwaway name). First make sure `gh auth status` is logged in, and set `projects.github_owner` in `config.yaml` if needed. Then say "Hey Spark, start a new project called spark-smoke-test" and answer "yes". Pass: `<projects.root_dir>/spark-smoke-test` has a git repo with one commit, the private GitHub repo `spark-smoke-test` exists, has a tracking issue, and wasn't pushed to. Delete it afterwards with `gh repo delete <owner>/spark-smoke-test`.
- [ ] **Connector actions (after the connector credentials).** "Draft an email to <you> about lunch." Pass: Spark asks first (a side-effecting tool). "Send an email to <you>." Pass: refused without asking (destructive tools are off).
- [ ] **Fast route by voice.** Say "Hey Spark, what time is it?". Pass: the log shows `route: fast` and the Haiku model, and the answer is short.
- [ ] **Deep routing.** Say "Hey Spark, summarize what we've discussed and draft a plan for next week." Pass: the log shows `route: deep` and the deep model, and the answer ends with a one-sentence summary. If the first token takes more than 5 s, you also hear "On it, give me a moment." first.
- [ ] **Barge-in on the real speaker.** Set `app.barge_in: true` (only with a speakerphone that has echo cancellation). Ask something with a long answer, then say "Hey Spark" while it's talking. Pass: speech stops at once (under 300 ms) and Spark listens again. To measure it, run `uv run pytest -m hardware -s tests/test_barge_in.py` (it plays audio out loud).
- [ ] **Wake word during a confirmation** (with `app.barge_in: true`). Say "Hey Spark" while the question is playing. Pass: the log shows `-> cancelled`, nothing is created, and Spark listens for a new request.
- [ ] **Self-wake check with barge-in on.** Let Spark say its own name during a reply ("Hi, I'm Spark"). Pass: it doesn't wake itself. If it does, lower `wakeword.options.porcupine.sensitivity` or keep `app.barge_in: false`.

## Desktop app
- [ ] **Installer.** Install Inno Setup 6 (`winget install JRSoftware.InnoSetup`), then run `powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1`. Run `build\installer\HeySpark-Setup-0.1.0.exe`. Pass: it installs without an admin prompt, and Hey Spark shows up in the Start menu. If SmartScreen warns about an unsigned app, choose "More info" → "Run anyway"; note it if a managed-device policy blocks this.
- [ ] **First run from the Start menu.** To test a truly fresh run, first rename `%LOCALAPPDATA%\HeySpark` to `HeySpark.bak`. Open Hey Spark and press Start. Pass: it shows the download steps, then "Say "Hey Spark"".
- [ ] **Every state has its own look.** Say "Hey Spark, what's the capital of Japan?". Pass: the orange halo grows while you talk, it turns green while listening, purple while thinking and light blue while speaking, and "You"/"Spark" show the question and answer. Ask for something that needs confirmation (e.g. "start a new project called zz-test"). Pass: it turns yellow with "Yes or no?".
- [ ] **Stop releases the mic.** Press Stop. Pass: the Windows mic indicator in the taskbar disappears within about 1 s. Press Start. Pass: back to "Say "Hey Spark"" in under 2 s.
- [ ] **Close quits.** Close the window while it's listening. Pass: within about 5 s, Task Manager shows no `HeySpark.exe` or `claude.exe` left.
- [ ] **One instance.** With Hey Spark open, launch it again from the Start menu. Pass: the existing window comes to the front and no second one opens.
- [ ] **Uninstall keeps your data.** Uninstall via Settings → Apps. Pass: the app is gone, and `%LOCALAPPDATA%\HeySpark` (config, `.env`, models) is still there.

## macOS app (Apple Silicon, macOS 14+)
Use the `installer-macos` artifact from a `Build apps` run, or `bash scripts/build_macos.sh` on a Mac.
- [ ] **Install.** Open the DMG and drag HeySpark into Applications. Open it. Pass: macOS shows the "can't verify" dialog for an unsigned build; *System Settings → Privacy & Security → Open Anyway* opens it. (A notarized build opens directly.)
- [ ] **Microphone prompt.** On first Start, macOS asks for microphone access with the Hey Spark explanation. Allow it. Pass: the mic level moves when you talk. Then turn it off in *Privacy & Security → Microphone* and press Start again. macOS feeds a denied app silence: record whether the mic level stays flat with no hint to the user (a follow-up if so).
- [ ] **Commands from Finder.** With `claude` and `npx` installed via Homebrew or npm, open the app from Finder (not Terminal). Pass: the Brain window finds `claude`, and a stdio MCP server connects. The log (`~/Library/Application Support/HeySpark/logs`) has a "PATH for child processes" line with `/opt/homebrew/bin`.
- [ ] **Voice round trip.** Say "Hey Spark, what's the capital of Japan?". Pass: you hear the answer through the speakers, and barge-in works.
- [ ] **Dark mode.** Switch *Appearance* to Dark and reopen. Pass: no white flash before the page paints.
- [ ] **One instance.** Open the app again from Applications or Spotlight. Pass: the existing window comes to the front.
- [ ] **Claude Code in Terminal.** Say "start a new project called zz-test", accept opening Claude Code. Pass: macOS asks once to let Hey Spark control Terminal; then a Terminal window opens in the project folder running `claude`.
- [ ] **Quit cleanly.** Quit with ⌘Q. Pass: Activity Monitor shows no `HeySpark` or `claude` process within about 5 s, and the orange mic dot in the menu bar goes away.
- [ ] **Icon.** Pass: the orange spark icon shows in the Dock, Finder and the DMG window.

## Brain window (needs real providers)
- [ ] **OpenAI from the window.** Open the brain button in the main window, pick OpenAI, paste a real key, press Test connection. Pass: "Connected to OpenAI and the key works", the model lists fill in, and "gpt-4.1 can use tools". Press Use OpenAI. Pass: the main window's Models link reads "OpenAI · ...", and a spoken question is answered. `.env` has `OPENAI_API_KEY`; `prefs.json` and `logs\spark.log` don't contain the key (`Select-String -Path prefs.json,logs\* -Pattern "sk-"`).
- [ ] **A wrong key.** Replace the key with a wrong one and test. Pass: "OpenAI didn't accept the API key (HTTP 401)". Save anyway while Spark is running. Pass: the error shows and Go back to the previous settings restores the working brain.
- [ ] **Ollama download.** With Ollama running and no models, pick Ollama (local). Pass: three recommended models with sizes and the free space. Click Download on Qwen 2.5 (7B), watch the bar, Cancel, then Download again. Pass: it resumes and finishes; the model pickers switch to it; Test connection says whether it can use tools.
- [ ] **Azure OpenAI.** With a real resource: base URL `https://<resource>.openai.azure.com/openai/v1`, the key, your deployment names as models. Test, then apply. Also try a classic `.../openai/deployments/{model}` address with API version `2024-10-21`.
- [ ] **Claude API key and Claude Code.** In the Brain window pick Claude. Pass: there is no sign-in or account line, only an Anthropic API key field. Enter a wrong key and press Test connection. Pass: "Anthropic refused the API key". Enter a real key, test, then Save & apply. Pass: `ANTHROPIC_API_KEY` is in `.env` and Spark answers. With Claude Code not installed (or `brain.cli_path` pointing nowhere), the card says "Claude Code isn't installed" with an *Install Claude Code* link, and pressing Start shows the install fix in one line.

## Security spot checks (§12)
- [ ] **No secrets in doctor output.** With real keys in `.env`, run `uv run python -m spark doctor > doctor.txt 2>&1`, then `Select-String -Path doctor.txt -Pattern "sk-ant-|ghp_|github_pat_"`. Also search for your Composio key's first 8 characters. Pass: no matches. Delete `doctor.txt` afterwards.
- [ ] **No secrets in logs after a real session.** After a voice session with real keys, run `Select-String -Path logs\* -Pattern "sk-ant-|ghp_|github_pat_"`, and search for your Composio key's first 8 characters. Pass: no matches.
- [ ] **Audio only after the wake word (cloud STT).** With `stt.backend: deepgram`, open Resource Monitor → Network, filter on the Spark `python.exe`, and talk for 1 minute *without* the wake word. Pass: no traffic to `api.deepgram.com`. Then say "Hey Spark, …". Pass: one request.
