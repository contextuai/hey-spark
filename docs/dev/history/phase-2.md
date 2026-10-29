# Phase 2 report — 2026-09-24

Connectors (SPARK_SPEC §10, Phase 2). Built by the Ralph loop on branch `build/phases-1-3`.
Audit: `spec-reviewer` round 1 found 1 FAIL (row "`${VAR}` expansion in the real brain path"). It was fixed with a live end-to-end test, see "Known issues and fixes".

## Acceptance checklist
| Criterion (SPARK_SPEC §10 / §7 / §12) | Status | Evidence / how to verify |
|---|---|---|
| Auth spike documented before wiring more than two connectors | PASS | `docs/connectors.md` (a table per connector: transport, auth, env vars, OAuth, read-only, side-effecting and destructive tools, recommendation). The spike commit precedes the loader and wiring commits |
| `mcp_servers.json` loaded and passed to `ClaudeAgentOptions(mcp_servers=...)` | PASS | `spark/brain/mcp_config.py`, `spark/brain/claude_agent.py`. Tests in `tests/test_mcp_config.py` and `tests/test_brain_claude_agent.py` |
| `${VAR}` expansion; a missing var disables the server with a warning; secrets never committed | PASS | Missing variables are named, but their values never are (`test_missing_var_disables_server_and_never_leaks_values`). Real expansion is proven live: `tests/test_live_mcp_env.py` (the real loader, ClaudeAgentBrain, and CLI, with a stdio server that says "yes" only for an *expanded* token). `test_committed_mcp_json_has_no_secrets` |
| In-process local tools (time, timers, status) | PASS | `spark/tools/*`, the `spark` SDK server. Timers announce "Your pasta timer is done." (queued while busy) |
| Explicit per-server allowlists; read-only pre-approved; side-effecting not auto-run; destructive off unless enabled | PASS | `config.yaml` → `connectors.servers.*`. Server-wide wildcards are rejected, and unknown tools are denied. Phase 2 used a deny-by-default gate (`spark/brain/tool_gate.py`), which Phase 3 replaced with the spoken-confirmation gate |
| Start read-only (§12) | PASS | The GitHub MCP server runs with `--read-only`. Work Outlook is `enabled: false` by default |
| Keep MCP servers warm (§9) | PASS | One long-lived client connected at warm-up. Risk: an idle reset (10 min) reconnects inside the next turn (see Known issues) |
| Tool results summarized; nothing raw (IDs, URLs) read aloud | PASS (code + live fake tool) / NEEDS-HUMAN (real accounts) | `sanitize_for_speech` drops URLs, emails, UUIDs, SHAs, ULIDs, GitHub node IDs, and tokens. Live test `test_live_tool_results_spoken_without_ids_urls`: a raw-JSON tool result was spoken as "You have three open pull requests. The newest one is Add dark mode toggle from Sam, created yesterday. Want the others?" |
| "What's on my calendar tomorrow?", "Any unread email from X?", "How many open PRs on repo Y?" answered correctly | NEEDS-HUMAN | Needs real credentials: `scripts/phase2_acceptance.md` §1–3 (text mode with `-v` first, then voice) |
| Doctor covers connectors | PASS | `spark/doctor.py::check_connectors`: env vars, stdio spawn + `tools/list`, http/sse reachability, timeouts. Tests spawn a real tiny stdio MCP server |

## Latency (this machine, no mic)
There were no new pipeline stages in Phase 2. Tool calls add model round-trips that depend on the connector, and these can't be measured without real accounts: **not measured** (needs credentials; `spark bench brain` covers only the no-tool path: median TTFT 1037 ms, p90 1352 ms, see the Phase 1 report). The live fake-tool test answered in a few seconds, but it isn't a benchmark, so no number is claimed.

## What was built
- `spark/brain/mcp_config.py`: the `mcp_servers.json` loader. It checks every `${VAR}` but leaves placeholders for the CLI to expand, resolves `command` (`which`, relative paths, `npx` → `cmd /c npx` on Windows), and builds `McpSetup` (servers, exact `allowed_tools`, `disallowed_tools`, side-effecting set, disabled reasons) plus a `ToolPolicy` (read/side_effect/destructive/unknown). `env_for_sdk` sets `ENABLE_CLAUDEAI_MCP_SERVERS=false`.
- `mcp_servers.json`/`.example.json`: GitHub via the official `github-mcp-server` exe (stdio, `--read-only`, toolsets), the remote GitHub endpoint (disabled), Composio dedicated MCP for Gmail and Calendar, and Composio Outlook (disabled).
- `spark/tools/`: `current_time`, `set_timer`/`list_timers`/`cancel_timer` (TimerService on an injectable clock), and `spark_status`, served as the in-process `spark` SDK MCP server.
- Brain wiring: servers, allowlist, gate hook, SDK env, tool calls logged by name only. The voice prompt is updated for tool results, and the sanitizer is extended for IDs, tokens, and inline code.
- The doctor's connector checks, and `scripts/phase2_acceptance.md`.

## Deviations from spec (and why)
- **The GitHub server is the official `github-mcp-server` binary, not `npx @modelcontextprotocol/server-github`** (§7 example). The npm package is deprecated and archived (docs/connectors.md).
- **Env var names** are `SPARK_GITHUB_PAT`, `COMPOSIO_*`, not `GITHUB_TOKEN`/`AGGREGATOR_*`. `gh` gives `GITHUB_TOKEN` priority over its own login, which would silently switch `start_project` onto the read-only PAT.
- **`${VAR}` is expanded by the Claude CLI, not in Python.** Secrets never pass through Spark's code or its option objects. This is proven by the live test.
- **Timers are pre-approved** (`set_timer`/`cancel_timer` change only Spark's in-memory state). The confirmation rule is applied to external sends, creates, and deletes.

## Open decisions touched (§13)
- **Aggregator:** Composio (dedicated MCP server mode) was chosen over Zapier MCP (2 tasks per call, self-enabling meta-tools) and Pipedream (1-hour tokens, 5 headers per app). Reasons and sources are in docs/connectors.md.

## Known issues and fixes
- Round-1 audit FAIL (fixed): real-path `${VAR}` expansion was unproven. Added `tests/test_live_mcp_env.py` plus a stricter `token_seen` in `tests/mcp/echo_server.py`.
- An idle reset (after 10 min) reconnects the session, and restarts stdio MCP servers, inside the first question after the idle period, adding connect + prime time to that answer.
- Whether `ENABLE_CLAUDEAI_MCP_SERVERS=false` keeps the user's claude.ai connectors out of the session is unverified live. The gate denies their tools regardless.
- The real GitHub tool names under `--read-only` (e.g. `pull_request_read`) are partly UNVERIFIED until a human runs the doctor with a PAT. The doctor prints the server's tool count.
- `mcp` and `httpx2` (used by the doctor's probes) come in transitively via `claude-agent-sdk` and aren't declared directly (not in APPROVED_DEPS). They're imported lazily.

## Verdict
**CODE-COMPLETE.** The round-2 audit found 0 FAIL rows. Row "`${VAR}` expanded in the real brain path" now passes, and the re-run live test answered 'yes'. The Phase 3 confirmation gate (which replaced the deny-by-default gate) was checked for regressions: none were found. The auditor judged that the reworded §8.2 rule still meets the spec, because the gate itself speaks the action and asks before any side effect.
