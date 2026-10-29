# Phase 3 report — 2026-09-24

Actions and routing (SPARK_SPEC §10, Phase 3). Built by the Ralph loop on branch `build/phases-1-3`.
Audit: `spec-reviewer` in two rounds. Round 1 had 1 FAIL (the fast route's limited tool set, §8.4), fixed in 71aeb42. Round 2: **CODE-COMPLETE**, 0 FAIL. See the verdict at the end. There was also a dedicated security audit (P3.5), where 9/10 PASS and the 1 FAIL was fixed.

## Acceptance checklist
| Criterion (SPARK_SPEC §10 / §8.3 / §8.4 / §3 / §12) | Status | Evidence / how to verify |
|---|---|---|
| Confirmation hook: side-effecting tools are intercepted by a PreToolUse hook; Spark speaks a one-line summary and listens for yes/no; a timeout counts as no | PASS (code + live, text confirmer) / NEEDS-HUMAN (by voice) | `spark/brain/confirm.py` (`ConfirmationGate`, `VoiceConfirmer`, `TextConfirmer`, `classify_yes_no`) and `State.CONFIRMING` in `spark/app.py`. Verified live: a PreToolUse `{}` falls through to the CLI's permission check (which denies), so "yes" returns an explicit allow. A hook timeout fails closed. Tests: `tests/test_confirm.py` (gate outcomes with exact dicts, voice and text confirmers, app-level voice "no" through the mic path) |
| Every side-effecting action is confirmed, and "no" prevents it | PASS (live, text) / NEEDS-HUMAN (voice) | Live `confirm_no`: asked "I'll create note called groceries. Should I go ahead?", heard "no", the tool **never ran**, and Spark declined (e.g. "Okay, I won't." or "Okay, I won't create it."; wording varies by run) Live `confirm_yes`: the tool ran with the right args. `start_project` with a scripted "no" creates nothing (test). Voice run: HUMAN_CHECKLIST "Spoken confirmation, no/silence" |
| Destructive tools off unless enabled; unknown tools denied (§12) | PASS | The gate denies destructive and unknown tools without asking. Destructive tools go to `disallowed_tools` unless `allow_destructive`. Tests in `test_confirm.py`, `test_mcp_config.py` |
| Fast path: `fast_model` with a limited tool set; deep path: `deep_model` with the full set (§8.4) | PASS | Round-1 audit FAIL, fixed: `routing.fast_tools` (default: every read-only tool) is enforced per call by the PreToolUse gate on fast turns (tools outside it are denied without asking, and the next turn is escalated to deep). Side-effect requests route deep by rule. Live: a fast turn asked to create a note → denied, nothing ran, a short "that's a multi-step task, just ask again" reply (wording varies by run); next turn deep → confirmed → created. Tests in `test_confirm.py`, `test_brain_claude_agent.py`, `test_router.py` |
| Fast/deep routing, with the decision logged | PASS | `spark/brain/router.py` (rules plus an optional fast-model classifier, off by default). `set_model` switches models on the same session (context kept). `route: fast|deep (reason)` is logged at INFO. Live `route_deep`: `claude-sonnet-5` answered after the switch and remembered context from the fast turn |
| Deep tasks > ~5 s get "On it, give me a moment" plus a spoken summary when done | PASS | `ClaudeAgentBrain.respond` yields the acknowledgement after `routing.ack_after_s` (5 s) with no text on a deep turn. It's paused while a confirmation question is pending. The deep answers are prompted to end with a one-sentence summary. Tests use a manual clock (fires at 5.1 s, not at 4.9 s) |
| Barge-in: the wake word while speaking stops TTS and returns to LISTENING | PASS (fakes) / NEEDS-HUMAN (real speaker) | `SparkApp.barge_in`: `player.stop()` runs first and synchronously, a turn-generation counter blocks stale state changes, and `brain.interrupt()` runs in the background. It also works in THINKING. In CONFIRMING it cancels, so the tool is denied. `tests/test_barge_in.py` |
| Barge-in stops speech within 300 ms | PASS (fakes) / NEEDS-HUMAN (real device) | Measured on fake devices (wall clock from the wake frame): `player.stop()` 0.0–0.6 ms, `speak()` returns in 0.4–1.1 ms, and 0 speech samples reach the device after the wake frame (the limit is one 20 ms block). Real speaker: `uv run pytest -m hardware -s tests/test_barge_in.py` (plays audio) |
| New-project flow: local folder plus git repo, GitHub repo, tracking task, optional Claude Code session | PASS (code, fakes) / NEEDS-HUMAN (real `gh`) | `spark/tools/project.py`: plan (no side effects) → mkdir, `git init`, and a first commit → `gh repo create --source` (never `--push`) → `gh issue create` → optionally `claude` in a new console. It's side-effecting, so it's always confirmed. Tests run in a tmp dir with a fake host and tracker, and the argv for `gh`/`claude` is asserted. A real run is in HUMAN_CHECKLIST |
| Security (§12): secrets never logged; no audio stored by default; transcript toggle; audio leaves only after the wake word; per-server allowlists | PASS | P3.5 audit: 9/10 PASS. The FAIL (the `--text` log fallback and console error text weren't redacted) is fixed with tests. `gh`/`git`/`claude` child processes get a scrubbed env. MCP URLs are treated as secrets. Details: `docs/connectors.md` "Security notes (P3.5)" |
| README: setup, run modes, bench, swapping backends (P3.6) | PASS | `README.md`, including the Phase 3 section (confirmation, new projects, routing with the fast-route tool limit, barge-in) |

## Latency (this machine, no mic)
| Step | Median | p90 | Budget | Notes |
|---|---|---|---|---|
| Brain TTFT, fast route (Haiku 4.5, warm, MCP + spark tools loaded) | 1148 ms | 1874 ms | < 600 ms (§9) | `spark bench brain --trials 5`. It was 1037 / 1352 ms before the tool definitions were loaded (Phase 1). **Over budget** |
| Model switch fast→deep (`set_model`) | 0–16 ms after warm-up | — | — | The first switch to each model costs 0.45–0.7 s, so warm-up visits the deep model once (P3.2 live measurement) |
| Barge-in stop (fake devices) | 0.0–0.6 ms to `player.stop()`; 0.4–1.1 ms to `speak()` return | — | < 300 ms (§10) | `tests/test_barge_in.py`. On a real device add one 20 ms output block plus the driver buffer (not measured; hardware test) |
| Barge-in stop, real speaker (this laptop's Echo Cancelling Speakerphone, Piper audio) | wake frame → `player.stop()` 0.5 ms; worst-case audible stop 60.5 ms (includes the 40 ms device buffer) | — | < 300 ms (§10) | `uv run pytest -m hardware -s tests/test_barge_in.py` (automated: the wake is injected, not spoken). Two runs: 60.6 / 60.5 ms. Detection delay for a *spoken* wake word isn't included (HUMAN_CHECKLIST) |
| Confirmation round-trip | not measured | — | — | Depends on the human answering; the voice timeout is 8 s |
| Deep-route TTFT (Sonnet) | not benchmarked | — | — | The live `route_deep` test logged 1.5–2.1 s in two runs. That's an observation, not a benchmark, and isn't used as a median |

STT (707 ms), Piper (334 ms) and end-of-speech (700 ms) are unchanged from Phase 1. The end-to-end wake→first-audio figure still needs the human mic run (`spark bench latency --trials 10`).

## What was built
- `spark/brain/confirm.py`: the Confirmer ABC, `VoiceConfirmer` (speaks, then listens via the app's mic, Recorder and STT; one re-prompt when unclear), `TextConfirmer` (y/n on stdin), a robust yes/no classifier, and `ConfirmationGate` (the PreToolUse hook). Summaries come from a per-tool registry plus a generic fallback and are sanitized (no URLs, IDs, or secrets).
- `spark/brain/router.py`: the rule router with an optional classifier, per-turn `set_model`, the "On it" acknowledgement, and route metadata.
- The app: `State.CONFIRMING` and barge-in (SPEAKING/THINKING/CONFIRMING, timer announcements), with a turn-generation counter, background interrupt, and interrupted-turn latency records (excluded from summaries).
- `spark/tools/project.py`: `start_project` with injectable `RepoHost`/`Tracker`/`SessionLauncher` (the real ones use `gh`/`claude` with argv lists, timeouts, and a scrubbed env).
- The security fixes (P3.5) and the README.

## Deviations from spec (and why)
- **The confirmation question is asked by the hook, not the model** (§8.2 wording kept: "Should I go ahead?"). When the model asked on its own it never called the tool, so the user was asked twice. The prompt now tells it to call the tool and let Spark ask. Two audits judged this to meet §8.2/§8.3.
- **The fast route's limited tool set is enforced by the hook, not by `allowed_tools`.** The CLI fixes its tool list at start, so the model still *sees* every tool on fast turns, but the gate refuses anything outside `routing.fast_tools`. That keeps one session (and context) without an ~10 s reconnect per switch. The tool definitions still cost fast-route TTFT.
- **Timers are pre-approved** (local in-memory state only). This is documented in `docs/connectors.md`.
- **Repos are created with `gh`, not the GitHub MCP server,** which runs `--read-only` (§12 "start read-only").
- **Offline suite time:** 1002 tests in 53–63 s depending on load (whisper is shared per session). Earlier runs under CPU load took 76–84 s, over the ~60 s guideline in CLAUDE.md.

## Open decisions touched (§13)
- **Routing classifier:** rules by default. A fast-model classifier is available (`routing.classifier: fast_model`) but costs a second CLI process and a round-trip.

## Known issues
- The Claude CLI and the stdio MCP servers it spawns inherit Spark's whole environment (the SDK can add variables but not remove them). The mitigation is documented: keep `.env` minimal, and prefer the official binaries and http servers.
- Without an echo-cancelling speakerphone, barge-in could let Spark wake itself when it says "Spark". `app.barge_in` stays off by default.
- A timer announcement interrupted by the wake word is dropped, not re-queued.
- The rules router is keyword-based. For example, "create a timer" is fast (timer is matched first), which is the desired outcome here, but edge cases will exist.

## Verdict
**CODE-COMPLETE.** Round-2 audit: 0 FAIL rows. The round-1 FAIL (the fast path's limited tool set) is fixed and proven offline and live. In the live test, a side effect requested on a fast turn was refused without asking and nothing ran; the retry went deep, was confirmed, and ran. `list_notes` was refused on a fast turn even though it was pre-approved, because the gate's deny wins over `allowed_tools`. The fast TTFT figure above comes from `uv run python -m spark bench brain --trials 5`, run on 2026-09-24 after P3.2.
