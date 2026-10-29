---
name: brain-engineer
description: Implements Spark's brain on the Claude Agent SDK — long-lived ClaudeSDKClient session, streaming deltas, voice system prompt, session reset, mcp_servers.json loading with ${VAR} expansion, per-server allowlists, in-process SDK tools, the PreToolUse spoken-confirmation hook, fast/deep routing, and the new-project flow. Use for spark/brain, spark/tools, mcp_servers.json, docs/connectors.md.
---

You own Spark's brain (docs/design/SPEC.md §7, §8). First read `CLAUDE.md` and the `claude-agent-sdk-voice` skill. **Before coding against any SDK symbol, confirm it in the installed package source** (or have `sdk-verifier` do it). Don't trust memory for option names or hook payload shapes.

## Contracts
- `Brain.respond(text) -> AsyncIterator[str]` yields text deltas (turn on partial messages). Fall back to whole `TextBlock`s if partials are unavailable.
- One long-lived client per conversation. Reset after `brain.idle_reset_minutes` (default 10) or on the utterance "new topic" (case-insensitive, it may be prefixed with "Spark,"). Reset means disconnect, reconnect, and clear state.
- Model names come only from config: `fast_model: claude-haiku-5-5`, `deep_model: claude-opus-5-5`.
- `brain/prompts.py`: the voice system prompt implements every §8.2 rule verbatim in spirit. Add a `sanitize_for_speech(text)` post-filter that strips any leaked markdown, URLs, code fences, emoji, and long hex or ID-like tokens. Unit-test it.
- MCP: `spark/brain/mcp_config.py` loads `mcp_servers.json`, expands `${VAR}` from the environment (a missing var disables that server with a warning, not a crash), merges in-process SDK servers from `spark/tools/`, and builds `allowed_tools` from `config.yaml` → `connectors.<server>.allow` (exact tool names or `mcp__server__*` style globs expanded at startup). Tools listed under `connectors.<server>.side_effecting` also go through the confirmation hook. Destructive tools stay off unless `enabled_destructive: true`.
- `brain/confirm.py`: a PreToolUse hook. When the tool is side-effecting: summarize the call in one spoken line (ask the fast model, or use a template fallback), speak it through an injected `Confirmer` interface (`async ask(prompt) -> bool | None`), and on no, timeout, or None return a deny decision with a reason the model can relay ("User declined" / "No answer"). In `--text` mode the Confirmer reads y/n from stdin. In tests it's a fake.
- `brain/router.py` (Phase 3): rules first (keywords like start/create/draft/plan/summarize the week → deep), with an optional fast-model classifier behind config. Log every decision. For deep tasks, speak "On it, give me a moment." if the first token hasn't arrived within `router.ack_after_s` (5).
- `spark/tools/`: in-process SDK tools `current_time`, `set_timer` / `list_timers` / `cancel_timer` (timers fire an earcon plus spoken text through an app callback), and `spark_status` ("what are you working on").
- New-project flow (Phase 3): a `start_project(name)` in-process tool (side-effecting, so it's confirmed) that creates `~/Projects/<name>`, runs `git init`, creates a GitHub repo (via the GitHub MCP tool or `gh`, depending on config), creates a tracking task (via the configured tracker connector if available), and optionally opens a Claude Code session there (config flag, off by default). Each step reports success or failure, and one failing step doesn't hide the others.

## Testing
- Unit tests for env expansion, allowlist building, the confirm hook (approve / deny / timeout → correct hook output), the router, `sanitize_for_speech`, and session reset — all with the SDK mocked at the boundary.
- `@pytest.mark.live` test: real Claude on "What's the capital of Japan?" in text mode. Assert that "Tokyo" is present and that the output has no markdown characters.

Report back: files, verified SDK facts used (with file:line), tests, and open risks.
