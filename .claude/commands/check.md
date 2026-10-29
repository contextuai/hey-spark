---
description: Run Spark's full quality gate (ruff, import hygiene, offline tests, --text smoke) and fix failures
argument-hint: "[--live]"
---
Run the `spark-quality-gate` skill now. Arguments: $ARGUMENTS
If `--live` was passed, also run the live test step. Fix any failure at its root cause, then report: ruff status, test counts (passed / failed / skipped with reasons), and the smoke output.
