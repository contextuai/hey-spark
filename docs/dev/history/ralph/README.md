# Running the Spark build loop

1. Start a **fresh** Claude Code session in this folder, so the new subagents load. In bypass or auto-accept mode it runs unattended.
2. Optional: fill in the "Before the loop" items in `docs/HUMAN_CHECKLIST.md`.
3. Paste:

```
/ralph-loop:ralph-loop Read .claude/ralph/PROMPT.md and follow it exactly --completion-promise 'SPARK PHASES 1-3 CODE COMPLETE' --max-iterations 60
```

- Watch progress: `/build-status`, or `head -10 .claude/ralph-loop.local.md`
- Stop early: `/ralph-loop:cancel-ralph`
- Resume after a stop: run the same command again. PLAN.md and git keep the state.

Files: `PROMPT.md` (loop instructions), `PLAN.md` (task list plus blocked log), `APPROVED_DEPS.md` (pre-approved dependencies).
The stop hook needs `jq` (installed at `~/.local/bin/jq.exe`) and `perl` (it comes with Git Bash).
