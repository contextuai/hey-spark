# Phase 2 acceptance: connectors (human-run)

SPARK_SPEC §10, Phase 2: *"What's on my calendar tomorrow?", "Any unread email from X?", and "How many open PRs on repo Y?" each give correct, spoken, summarized answers. Nothing raw (IDs, URLs) is read aloud.*

The build loop can't run these because they need your real accounts. Do them once the credentials exist. Everything runs in **text mode** first (typed input, printed output, no mic), then once by voice.

## 0. Prerequisites
1. Credentials in `.env`, following `docs/connectors.md` §5 ("What a human must do"):
   - GitHub: `SPARK_GITHUB_PAT` (a fine-grained, read-only PAT) and `SPARK_GITHUB_MCP_BIN` (the full path to `github-mcp-server.exe`).
   - Gmail + Google Calendar via Composio: `COMPOSIO_API_KEY` and `COMPOSIO_MCP_URL` (a dedicated MCP server with the read-only tools from `config.yaml` → `connectors.servers.composio.allowed_tools`).
2. Run `uv run python -m spark doctor`.
   **Pass:** `✔ connector: github  stdio, N tools` and `✔ connector: composio  http, N tools`. If a connector shows `!`, follow its `fix:` line.
3. Pick the real values you'll check against: a repo you own (`<owner>/<repo>`, with a known number of open PRs), a sender with unread mail today (`<Sender Name>`), and tomorrow's calendar (open it in your browser to compare).

## 1. Text mode, one question per run
Each command pipes one question in, prints Spark's reply, and exits. `-v` shows the tool calls in the log, so you can see which connector answered.

```powershell
echo "What's on my calendar tomorrow?" | uv run python -m spark --text -v
echo "Any unread email from <Sender Name> today?" | uv run python -m spark --text -v
echo "How many open PRs are there on <owner>/<repo>?" | uv run python -m spark --text -v
```

Expected for each:
| Check | Pass condition |
|---|---|
| Correct | Matches what you see in Google Calendar, Gmail, and GitHub (counts, times, sender, top item) |
| Summarized | 1–3 short spoken-style sentences: a count plus the first one or two items, then an offer like "want the details?" |
| Nothing raw | No URLs, email addresses, message/event IDs, PR node IDs (`PR_kw…`), commit SHAs, or JSON in the printed reply |
| Right tool | The `-v` log shows `mcp__composio__GOOGLECALENDAR_…`, `mcp__composio__GMAIL_…`, or `mcp__github__…` calls, with no permission denials for read-only tools |
| Read-only | No send, create, or update tools were called (the log shows none) |

## 2. Text mode, a follow-up in one session
```powershell
uv run python -m spark --text -v
```
Then type these three lines:
1. `How many open PRs are there on <owner>/<repo>?`
2. `Which one is the oldest?`
3. `quit`

**Pass:** the second answer uses context from the first (no need to repeat the repo), and still reads out no IDs or URLs.

## 3. Voice
```powershell
uv run python -m spark
```
Say each of the three questions after "Hey Spark". **Pass:** the same criteria as §1, spoken aloud, with nothing raw heard.

## 4. Record the result
Copy the three replies (text mode) into `docs/reports/phase-2.md` → "Human acceptance run", and tick the Phase 2 items in `docs/HUMAN_CHECKLIST.md`.

## Troubleshooting
- **`! connector: … missing environment variable(s)`:** the variable isn't in `.env`. Its name is printed, but never its value.
- **The answer says it can't access the calendar or email:** check that the Composio dedicated server includes the tool names in `config.yaml`, and that the Google OAuth consent was completed for the same `user_id`.
- **A tool is denied:** in Phase 2 only the `allowed_tools` are callable. Side-effecting tools are blocked until the Phase 3 spoken-confirmation hook is in place.
- **Work Outlook:** it stays disabled (`composio_outlook.enabled: false`) by default; a work/school tenant may need admin consent for the app and scopes (spec §12).
