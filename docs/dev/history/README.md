# Build history

Hey Spark was built with [Claude Code](https://claude.com/claude-code) running in an autonomous loop. It worked phase by phase against [`docs/design/SPEC.md`](../../design/SPEC.md), planned each phase, implemented and tested it, and wrote an acceptance report before moving on. Hardware and credential checks that the loop could not prove itself were handed to a human; those now live in [`docs/dev/MANUAL_QA.md`](../MANUAL_QA.md).

The files here are the original plans and acceptance reports. They are kept for reference only and are not maintained. File paths inside them refer to the layout at the time they were written (for example `docs/reports/`, `docs/HUMAN_CHECKLIST.md` and `.claude/ralph/`).

| File | What it is |
|---|---|
| `phase-1.md`, `phase-2.md`, `phase-3.md` | Acceptance reports for phases 1 to 3 (spec §14.5) |
| `PHASE1_PLAN.md` | The phase 1 implementation plan |
| `PHASE5_TODO.md` | The phase 5 task list |
| `phase2_acceptance.md` | Text-mode acceptance script for the phase 2 connectors |
| `ralph/PROMPT.md` | The instructions the build loop followed on every iteration |
| `ralph/PLAN.md` | The loop's task list and "blocked / needs human" log |
| `ralph/APPROVED_DEPS.md` | Dependencies pre-approved for the loop |
| `ralph/README.md` | How the loop was started |
