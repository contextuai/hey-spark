# Security policy

## Reporting a vulnerability

**Please don't open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting: go to the repository's
**Security** tab and choose **Report a vulnerability**
(<https://github.com/contextuai/hey-spark/security/advisories/new>). If that isn't possible, use
the contact address on the [ContextuAI website](https://contextuai.com).

Please include:

- what the problem is and what an attacker could do with it;
- steps to reproduce, or a proof of concept;
- the Hey Spark version (`spark/__init__.py` or the installer name), how you run it (from source
  or the installer), and the brain backend;
- relevant logs or `spark doctor` output, **with secrets removed**.

We aim to acknowledge a report within a few days, keep you updated while we work on a fix, and
credit you in the advisory unless you'd rather stay anonymous. Please give us a reasonable time
to release a fix before disclosing the issue publicly.

## Supported versions

Only the **latest release** gets security fixes. Pre-releases (`vX.Y.Z-N`) are superseded by the
next release.

| Version | Supported |
|---|---|
| Latest release | Yes |
| Anything older | No |

## Scope

Hey Spark listens to a microphone and lets a language model call tools, so we especially want to
hear about problems in these areas:

- **Secret handling.** Secrets must come only from the environment or `.env`, never be written to
  `config.yaml`, and never appear in logs, transcripts or the activity log. The redacting logger
  (`spark/util/logging.py`) masks known secret values and common token shapes, and child processes
  Spark starts itself get an environment with secret-looking variables removed. A secret that
  leaks to disk, a log, a crash report, or a process that shouldn't have it is in scope.
- **The spoken confirmation gate.** Side-effecting tools (sending, creating, updating) must run
  only after an explicit spoken "yes" (`spark/brain/confirm.py`, `spark/brain/tool_gate.py`).
  Silence, an unclear answer or a timeout must mean "no", and destructive tools must stay blocked
  unless `allow_destructive: true` is set for that server. Any way to run such a tool without a
  real yes (for example via prompt injection in tool results, a misrouted turn, the researcher
  sub-agent, or a race with barge-in) is in scope.
- **MCP connector allowlists.** Per-server policy in `config.yaml` (`connectors:`):
  `allowed_tools`, `side_effecting`, `destructive`, with everything else denied. A way to call a
  tool outside its list, or to make a read-only route use a write tool, is in scope. So is
  `${VAR}` expansion in `mcp_servers.json` leaking values it shouldn't.
- **Audio privacy.** Audio leaving the wake-word detector before the wake word, or audio being
  stored without `--debug-audio`.
- **The Windows installer and desktop app**, for example unsafe file permissions in the per-user
  install or data folder.

Out of scope: vulnerabilities in third-party services or MCP servers themselves (report those
upstream), attacks that need an attacker who already controls your user account, false wakes of
the wake-word model (that's a quality bug, so open a normal issue), and the fact that the
installer is not code-signed (known).
