---
name: spark-quality-gate
description: The definition of "green" for Spark — lint, format, import-hygiene, offline test suite, and --text smoke run — plus how to triage failures. Use before every commit, before opening a pull request, and when the user runs /check.
---

# Spark quality gate

Run these in order and stop at the first failure, then fix it:

```bash
uv sync --all-extras   # never plain `uv sync`: it uninstalls the optional extras (pywebview, openwakeword, ...)
uv run ruff check . --fix && uv run ruff format .
uv run ruff check .                                   # must be clean after autofix
uv run python -c "import spark, spark.app, spark.interfaces, spark.registry, spark.config"   # import hygiene: no heavy deps at import time
uv run pytest -q -m "not live" --maxfail=5
uv run python -m spark --help
echo "what is two plus two" | uv run python -m spark --text --no-tts --brain mock   # smoke, offline
```

When a real brain is available (the `claude` CLI is logged in or `ANTHROPIC_API_KEY` is set), also run:
```bash
uv run pytest -q -m live -k "capital or text_mode"
```
If you skip a live test, say so in the report. Don't claim it passed.

## Green means
- ruff is clean, and imports are clean without the optional extras installed.
- 0 failed tests. Every skip has a reason naming a missing model, key, or hardware.
- The smoke run prints a response and exits 0.

## Triage rules
- Read the full traceback and reproduce with `-x -vv -k <test>`.
- Fix the root cause. Never delete, skip, or loosen a test to get to green unless the test itself is wrong, and then say why in the commit message.
- If the same failure survives 3 attempts, stop and report it with the diagnosis instead of piling on more changes.
