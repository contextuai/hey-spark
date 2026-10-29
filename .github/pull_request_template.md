<!--
PR title: use a Conventional Commit, e.g. "feat(tts): add a speaking-rate option" or "fix: ...".
PRs are squash-merged, so the title becomes the commit message on main.
Release bumps are titled exactly "chore: release vX.Y.Z" (see CONTRIBUTING.md).
-->

## Summary

<!-- What does this change and why? Link the issue: "Closes #123". -->

## Test plan

- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run pytest -q`
- [ ] `echo "what is two plus two" | uv run spark --brain mock --text`
- [ ] New or changed behaviour has tests (with the fakes in `tests/fakes.py`, not hardware)

<!-- Anything you checked by hand (voice mode, desktop app, installer, a live connector)? Say what and how. -->

## Checklist

- [ ] The PR title is a Conventional Commit
- [ ] No secrets in code, config, logs or screenshots
- [ ] Side-effecting tools go through the spoken confirmation gate
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for user-visible changes
- [ ] A new wake-word model states every training dataset and its license
