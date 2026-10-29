# Ralph loop prompt: build Spark, Phases 1–3

You are building **Spark**, the wake-word voice assistant specified in `docs/design/SPEC.md`, inside a Ralph loop. The same prompt comes back to you on every iteration. Your memory between iterations is the **repo**: `.claude/ralph/PLAN.md`, `docs/`, the code, and `git log`.

## Every iteration
1. **Orient (keep it short):** read `CLAUDE.md`, `.claude/ralph/PLAN.md`, and `git log --oneline -10`. Read `docs/design/SPEC.md` in full on the first iteration, and afterwards whenever a task touches a section you haven't read in this context.
2. **Pick** the first `[ ]` task in PLAN.md whose dependencies are done. If the previous iteration left uncommitted work, finish that first (`git status`).
3. **Execute** it:
   - Delegate to the owner agent listed on the task (`audio-engineer`, `speech-engineer`, `brain-engineer`, `test-engineer`). Give the agent the task text, the relevant spec sections, and the done-when condition. Run `sdk-verifier` first whenever the task says so, or when you're about to depend on an API not yet recorded in `docs/sdk_notes.md`.
   - Independent tasks with disjoint files may run as parallel agents (for example P1.3 with P1.4). Never let two agents edit the same file at once.
   - Use the project skills: `spark-backend-pattern`, `claude-agent-sdk-voice`, `spark-audio-fixtures`, `spark-quality-gate`, `spark-phase-gate`.
4. **Verify:** run the `spark-quality-gate` skill yourself (don't trust an agent's "tests pass" without re-running them). It must be green.
5. **Record:** tick the task `[x]` in PLAN.md, add one line of notes under the task if something is non-obvious, and commit on branch `build/phases-1-3`:
   `git add -A && git commit -m "<type>(<scope>): <task id> <summary>"`, followed by a blank line and the trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
6. Continue to the next task in the same iteration while context allows. Stop the iteration at a clean committed state.

## Hard rules
- **Dependencies:** only the ones in `.claude/ralph/APPROVED_DEPS.md`, added with `uv add`. Anything else → log it in the Blocked section and work around it.
- **Honesty:** never mark a hardware, mic, speaker, or credential criterion as passed. Those go to `docs/HUMAN_CHECKLIST.md` as NEEDS-HUMAN with the exact command to run. Never fabricate latency numbers.
- **No hollow code:** no `NotImplementedError`, `TODO`, or `pass` stubs in shipped paths for Phases 1–3. Optional cloud backends must be real implementations, even when their tests are live-only.
- **Stuck:** if the same task fails 3 attempts, mark it `[~]`, write the diagnosis under "Blocked / needs human", and move on. Blocked tasks with a written diagnosis don't prevent completion, but a phase GATE can't be ticked while any of that phase's tasks show FAIL in the spec-reviewer audit because of missing code.
- **Safety:** don't push, don't create remote repos or real tickets, don't send anything with real credentials, and don't touch files outside this project folder (except `models_dir` downloads and the uv cache). The new-project flow is tested only against a temp dir and fakes.
- **Phase stops:** the spec's "stop after each phase" is replaced by the phase GATE task (report plus human checklist). Keep going to the next phase.
- Model downloads: at most about 1.5 GB total into `./.models` (whisper base.en plus one Piper voice are enough). Don't download `small.en` or Kokoro unless a task needs it and the disk has more than 10 GB free.

## Completion
Output the promise **only when all of these are true**, checked right now in this iteration:
1. Every task in PLAN.md from P0 through F.1 is `[x]`, or `[~]` with a written diagnosis that is genuinely external (a missing credential, hardware, or IT restriction). `[~]` is not allowed for a task that just needs more code.
2. `docs/reports/phase-1.md`, `phase-2.md`, and `phase-3.md` exist, and each shows zero FAIL rows.
3. The `spark-quality-gate` skill is green in this iteration (show its output).
4. `docs/HUMAN_CHECKLIST.md` lists every NEEDS-HUMAN criterion with a runnable command.
5. `git status` is clean on `build/phases-1-3`.

If all five hold, print a final summary (tasks done, blocked items, and where the reports are), then on its own line:

<promise>SPARK PHASES 1-3 CODE COMPLETE</promise>

Otherwise, don't mention the promise. Just keep working.
