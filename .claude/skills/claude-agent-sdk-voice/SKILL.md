---
name: claude-agent-sdk-voice
description: Patterns for using the Python claude-agent-sdk as a low-latency voice brain — streaming partial text, long-lived ClaudeSDKClient sessions, MCP server config (stdio/http/sse + in-process SDK servers), allowed_tools naming, PreToolUse hooks for spoken confirmation, and model switching. Use when writing or debugging anything in spark/brain or spark/tools.
---

# Claude Agent SDK for a voice brain

> **Rule 0:** The snippets below are a *starting hypothesis*. Before relying on any symbol, grep the installed package (`.venv/Lib/site-packages/claude_agent_sdk/`, especially `types.py` and `client.py`) or dispatch `sdk-verifier`. Record verified facts with file:line in `docs/sdk_notes.md` so nobody has to re-verify them.

## Client and streaming
```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
opts = ClaudeAgentOptions(
    system_prompt=VOICE_PROMPT,
    model=cfg.brain.fast_model,
    include_partial_messages=True,     # token-level deltas for low latency
    mcp_servers=servers,               # dict name -> config (see below)
    allowed_tools=allowed,             # e.g. ["mcp__github__list_pull_requests", "mcp__spark__current_time"]
    hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[confirm_hook])]},
    setting_sources=[],                # hypothesis: don't inherit user/project Claude Code settings; verify the default
    max_turns=8,
)
async with ClaudeSDKClient(options=opts) as client:
    await client.query(text)
    async for msg in client.receive_response():
        # StreamEvent (partial): msg.event["type"] == "content_block_delta" and delta.type == "text_delta" → yield delta.text
        # AssistantMessage: content blocks (TextBlock / ToolUseBlock) — use only if no partials were seen
        # ResultMessage: end of turn (cost/usage/duration → latency log)
```
- Keep ONE client open across turns for context (§8.1). Connect at startup (warm-up) so the first turn doesn't pay CLI spawn cost. Measure spawn time on Windows. It's often the biggest latency item.
- A reset is a disconnect plus a fresh client. Check whether the client has `interrupt()` (for barge-in) and `set_model()` (for routing without reconnecting). If `set_model` doesn't exist, keep two clients (fast and deep).
- Windows: if the SDK can't find the CLI, set the `cli_path` option (verify the name) to `shutil.which("claude")`.

## MCP servers
```python
servers = {
  "github": {"type": "stdio", "command": "npx", "args": [...], "env": {...}},
  "aggregator": {"type": "http", "url": url, "headers": {"Authorization": f"Bearer {key}"}},  # verify keys (spec §7 says so)
  "spark": create_sdk_mcp_server(name="spark", version="1.0.0", tools=[current_time, set_timer, ...]),
}
```
- Tool names exposed to the model are `mcp__<server>__<tool>`. The allowlist must use those names.
- On Windows, `npx` might need `"command": "npx.cmd"` or `cmd /c npx`. Test it, and make the loader handle it (resolve with `shutil.which`).
- In-process tools:
```python
@tool("current_time", "Get the current local date and time", {})
async def current_time(args): return {"content": [{"type": "text", "text": now_str()}]}
```

## PreToolUse confirmation hook
```python
async def confirm_hook(input_data, tool_use_id, context):
    name = input_data["tool_name"]; args = input_data["tool_input"]
    if not policy.is_side_effecting(name): return {}
    ok = await confirmer.ask(summarize(name, args))       # speaks, then listens yes/no with timeout
    if ok: return {}
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "The user declined this action." if ok is False else "No confirmation heard."}}
```
Verify the payload keys and return shape against the installed SDK. The unit test must assert the exact dict the SDK expects.

## Voice output hygiene
The system prompt forbids markdown, but models still leak `**`, lists, and URLs. Always pass the stream through `sanitize_for_speech` at sentence level before TTS.

## Latency checklist
Warm client at startup · fast model by default · `include_partial_messages` · sentence splitter emits the first sentence ASAP · keep stdio MCP servers alive (the long-lived client does this) · log `brain_first_token` in the TurnTimer.
