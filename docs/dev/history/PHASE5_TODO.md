# Phase 5: all-day co-worker (todo)

Branch: `feat/phase5-coworker` (from `feat/desktop-app`). One commit per task; tick the box in the
same commit. The design is in `docs/design/SPEC.md` §10 "Phase 5".

## Decisions (2026-10-02, the user accepted the recommendations)

| # | Decision |
|---|---|
| D1 | While awake, answer only what is clearly a request or a continuation of the conversation. When unsure, stay silent. |
| D2 | Each wake starts a new Claude conversation (the user never sees the break). The previous conversation's last exchanges carry over if it ended under 30 min ago. The memory briefing always carries over. |
| D3 | Finished background tasks: a chime plus a window card right away. Spoken when Spark is awake and free, or at the next wake. |
| D4 | Background tasks may edit files and run tests inside `projects.root_dir` without asking. Commit, push, PRs, email and connector writes need a spoken yes, or are refused. |
| D5 | Memory is Markdown files in the user data folder, readable and editable by the user. |
| D6 | Activity history is kept on disk with no time limit (updated 2026-10-02). Only the user deletes it, through "forget" or the History window. It feeds future conversations and memory. `activity.retention_days: 0` (the default) means keep forever. |
| D7 | Tool results are kept in full in the activity log (updated 2026-10-02), with secrets redacted. `activity.tool_results: full` is the default; `meta` and `excerpt` are available per install. |
| D8 | The activity log is hash-chained (tamper-evident). |

## 5a: Awareness and conversations

- [x] **P5.1 Connector turn fixes (WAREHOUSE failure, 2026-10-02 12:04).** `max_turns` per route (fast 8, deep 25). Per-connector `route: deep`, so data servers send the turn to the deep model. Say "On it" on fast turns too once a tool call is running and no text has come yet. Speak one plain sentence on a failed turn (step limit, timeout, connector or sign-in problem) instead of the error beep alone.
- [x] **P5.2 Awake window.** After a reply Spark stays awake for `attention.awake_s` (default 180 s) with no wake word, and goes to sleep (soft chime) after that much silence. Every request restarts the countdown. The app exposes the awake deadline for the UI.
- [x] **P5.3 Local voice commands, handled without Claude.** "Go to sleep" / "that's all" / "goodbye" put Spark to sleep (chime). "New conversation" / "new topic" start a new conversation. "Stop" / "never mind" cancel. Matched by rules in `spark/commands.py`, in well under 300 ms.
- [x] **P5.4 "Is this for me?" check while awake.** Rules first: the name, a question, a request, or a continuation means answer; filler, chatter and third-person talk mean ignore, and ignored speech does not restart the countdown. An optional `fast_model` classifier has a timeout and falls back to the rules. Every ignored utterance is logged.
- [x] **P5.5 Activity log.** `spark/activity.py`: append-only JSONL per day under `activity.dir`, hash-chained, with secrets redacted. Records conversation start and end, user and assistant text, route, tool calls (name plus redacted input), full tool results (redacted), confirmation questions and answers, local commands, ignored speech and errors. Nothing is purged unless `activity.retention_days` is above 0 (default 0, keep forever). The transcript log keeps working.
- [x] **P5.6 A new conversation per wake.** When Spark goes to sleep, the brain starts the next session in the background (connect plus prime), so the next wake doesn't pay for it. The new session's system prompt carries a briefing: the previous conversation's last exchanges when it ended under `attention.carry_over_s` ago. The Claude session ID is stored on the conversation's activity record.
- [x] **P5.7 Desktop UI for awareness.** "Awake · 2:41" countdown, "Asleep · say Hey Spark", and a cue when Spark ignored something.

## 5a+: Voice shortcuts (requested 2026-10-02)

- [x] **P5.7a Voice shortcuts.** `spark/shortcuts.py`. A shortcut is one or more trigger phrases ("good morning", "how is today looking") plus a detailed prompt. When the user says a trigger, the detailed prompt goes to Claude instead of the words, with an optional route (fast / deep / auto). The words said around the trigger are appended as extra context. Stored in `shortcuts.json` next to `config.yaml` (gitignored, user-owned), with a few examples seeded. Matching is forgiving: case and punctuation are ignored, filler is tolerated, and the trigger must be most of the utterance. The activity log records which shortcut fired.
- [x] **P5.7b Shortcuts window.** A button in the main window opens a Shortcuts window like Connectors: list, add, edit, delete, enable/disable, test a phrase ("which shortcut would this fire?"). Changes apply without a restart.
- [x] **P5.7c Model selection (requested 2026-10-02).** Auto (today's routing), always fast, or always deep, picked in the main window and by voice ("use the deep model", "back to auto"). It persists across restarts. Which model each route uses (`brain.fast_model` / `brain.deep_model`) can be changed from the window too. Shortcuts keep their own route unless the mode is a fixed one.

## 5a++: The Jarvis crew, any provider (requested 2026-10-05)

Design (sci-fi reference points: JARVIS / FRIDAY, the Star Trek computer, Samantha in *Her*). Users talk to **one voice**, the persona and orchestrator: it plans, decides, confirms and speaks. Behind it a small **crew of roles** runs, each on its own provider and model:

| Role | Job | Default |
|---|---|---|
| Voice | Talks to the user, reasons, orchestrates, confirms actions | the deep model (Sonnet 5.5; Opus 5.5 one click away) |
| Researcher (worker) | Tool-heavy lookups in the background (schemas, queries, several calls) and reports compact facts | the fast model (Haiku 4.5) |
| Scribe | Conversation titles, summaries, memory upkeep (P5.8+) | the fast model |
| Sentry (later) | Proactive watchers: meetings, reviews, task results (5c) | the fast model |

Each role is a `{provider, model}` pair. A provider is **Claude** (the `claude` CLI login or an Anthropic API key) or any **OpenAI-compatible endpoint** (base URL plus a key env var; presets for OpenAI, Azure OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio, vLLM). Features that exist only on Claude (Agent SDK sub-agents) fall back gracefully elsewhere.

- [x] **P5.7d Voice and worker split on Claude.** `brain.mode` defaults to **deep**, and the deep model defaults to Sonnet 5.5. A Haiku **researcher** sub-agent (`ClaudeAgentOptions.agents`, the built-in `Agent` tool) holds the read-only connector tools and the connector-notes tools. The voice model delegates multi-step lookups to it and interprets the report. The step budget counts only the voice model's own calls. Worker tool calls are gated and logged like any other (with `agent`). In Auto, once a turn has gone deep, the rest of the conversation stays deep.

Now (2026-10-05, the user's priority: integration, connectors, model choice):

- [x] **P5.7h Researcher model picker.** The Models panel gets a third picker, Researcher (`brain.worker_model`; "same as Fast" by default), saved in `prefs.json`. "Which model are you using?" mentions it.
- [x] **P5.7i Connector health.** After each session connects, read the CLI's MCP status per server. A server that failed to connect gets a line in the next briefing ("WAREHOUSE is unreachable right now: say so, don't call its tools") and a status in the windows. Log it once per change.
- [x] **P5.7j Connectors window additions.** Per connector: a Route switch (Deep / Auto, `route` in `connectors.json`), the live status, and a Notes tab that shows and edits `memory/connectors/<SERVER>.md`.
- [x] **P5.7k Mute (requested 2026-10-05).** A mic button in the window header (and Ctrl+M) stops Spark listening, for when someone walks in. Mic blocks are dropped unseen and unkept. Work under way finishes and timers still fire. A pending confirmation gets no answer, so nothing runs. The awake window's clock keeps going, and unmuting with time left listens again. "Mute" works by voice; unmuting is the button only.

## 5b: Memory and history

- [ ] **P5.8 Conversation summaries.** When a conversation ends, the fast model writes a title and a one-line summary in the background, in a tool-less side session. Stored in the activity log.
- [x] **P5.9a Connector notes (pulled forward after the 2026-10-02 WAREHOUSE test).** `memory/connectors/<SERVER>.md`, one bullet per note with its date and conversation. In-process tools `save_connector_note` and `rewrite_connector_notes` (local writes, pre-approved). A prompt rule: after exploring a system, save where things live. The notes for enabled connectors go into every new session's briefing.
- [ ] **P5.9 Memory store and briefing (the rest).** `spark/memory.py`: `me.md`, `projects/<name>.md`, `connectors/<server>.md`, `journal/YYYY-MM-DD.md` under `memory.dir`. Each entry records the conversation it came from. Every new session's briefing includes `me.md`, the notes for enabled connectors, today's journal, recent summaries and open tasks, within a character budget. In-process tools: `remember`, `recall`, `forget`.
- [ ] **P5.10 Journal and history search.** Every summary is appended to today's journal. A read-only `search_history` tool answers "what did we do this morning" and "what did we decide about X".
- [ ] **P5.11 History window.** Timeline by day, a card per conversation, expandable events (tool calls, confirmations, errors), search, export to Markdown, forget a conversation (log plus memory built from it), and "Continue from here". Plus a "What Spark remembers" tab with edit and delete.
- [ ] **P5.12 Go back by voice.** "Go back to the WAREHOUSE conversation" finds the conversation, and the next session carries that conversation's exchanges over (or resumes it).

## 5c: Delegation

- [ ] **P5.13 Background tasks.** `spark/tasks.py`: a TaskService with a ledger (queued / running / needs-you / done / failed / cancelled) persisted to disk. A `delegate` tool (side-effecting, so confirmed by voice) runs a Claude Code session in a project folder, or a deep connector query, in the background. Git commit and push, `gh` writes, and anything outside the project folder are refused (D4). Tools `list_tasks`, `task_result` and `cancel_task`.
- [ ] **P5.14 Reporting back.** When a task finishes: a chime and a UI event. Spark speaks the result when awake and free; otherwise it goes into the next session's briefing and is mentioned once.
- [ ] **P5.15 Task cards in the window.** Running, done and failed tasks, each with its result text.

## 5d: Working-day polish

- [ ] **P5.16 Auto-pause during calls.** On Windows, detect another app using the mic (CapabilityAccessManager, polled). While that lasts, Spark is paused: no wake word, shown in the UI. Configurable, and off on other platforms.
- [ ] **P5.17 Recovery.** Reopen the mic after a stream error or laptop sleep/resume, and reconnect the brain after a CLI crash. Both are logged and shown in the UI.
- [ ] **P5.18 Phase 5 report.** `docs/reports/phase-5.md` plus the human checklist items.

## Future requirements (parked 2026-10-05 by the user)

- Ship both wake words: the custom **Hey Spark** model and openWakeWord's bundled **Hey Jarvis**, selectable.
- [x] **P5.7e OpenAI-compatible brain (done 2026-10-06).** `brain.backend: openai_compat`: streaming Chat Completions with tool calling over `httpx`, MCP tools through the `mcp` client (both now direct dependencies, approved by the user). Provider-neutral behaviour moved to `spark/brain/core.py` so both brains share the voice prompt, briefing, routing, step budget, ack, activity log, connector health and the one confirmation gate. Presets for OpenAI, Azure, OpenRouter, Groq, Together, Ollama, LM Studio, vLLM (`spark/brain/openai_presets.py`). The researcher is a nested read-only tool loop on the worker model. claude.ai connectors are Claude-only (skipped, shown unreachable). Doctor check, per-backend model prefs, the Models panel lists `/models`. See docs/openai_brain.md. Not live-verified against a real provider yet (no key; Ollama has no models).
- **P5.7f Setup and Settings window (the "boot sequence").** First run walks a new user through: the assistant's name and wake word ("Hey Jarvis" ships with openWakeWord, or a custom model), the voice (Piper voices with a preview), the brain (Claude sign-in check, Anthropic key, or an OpenAI-compatible URL, key and model per role, each with a Test button), the persona, and optionally connectors. The same tabs stay available later as Settings. Choices go to the user's config folder (`prefs.json`), with secrets in `.env`.
  - [x] **Brain tab, done early (2026-10-12)** as its own window (the brain button in the main window header): provider cards (Claude, OpenAI, Azure, OpenRouter, Groq, Together, Ollama, LM Studio, Custom), the Claude account from `claude auth status`, API key to `.env` only (shown back as "ending in ...abcd"), base URL and Azure api-version, Test connection (`GET /models` plus a one-tool probe on the voice model), model pickers filled from `/models`, Ollama install/run checks with recommended models and a cancellable download, Save & apply through `controller.apply_brain()` with in-page "Save anyway?" and one-click revert / switch back to Claude. `spark/desktop/brain_settings.py` (API, built to become a Settings tab), `spark/desktop/ollama.py`, `spark/brain/endpoint_check.py` (shared with the doctor), `ui/brain.html`. Prefs keys `brain_backend`, `openai_provider`, `openai_base_url`, `openai_api_version` override config.yaml (a `--brain` flag beats them). See docs/openai_brain.md. Remaining for P5.7f: the first-run flow and the other tabs (name and wake word, voice, persona).
- **P5.7g Persona.** `persona:` settings feed the voice prompt: name, how to address the user (name, "sir", "ma'am", or nothing), formality, a wit dial (none / dry / playful), proactivity (offer the obvious next step: "Shall I also…"), and verbosity. Presets: **Jarvis** (formal, dry wit, anticipates), **Friday** (casual, quick), **Computer** (terse, factual). Never sycophantic.
