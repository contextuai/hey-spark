# Connectors: auth spike (P2.1, SPARK_SPEC §7)

Research date: 2026-09-24. SDK: claude-agent-sdk **0.2.159**, bundled CLI **2.1.281**
(`.venv/Lib/site-packages/claude_agent_sdk/_cli_version.py:3`). SDK citations are
`file:line` under `.venv/Lib/site-packages/claude_agent_sdk/`. "CLI binary" means a
string found by grepping `_bundled/claude.exe` (minified JS, no line numbers).
**UNVERIFIED** marks anything not confirmed in source or in first-party docs.
No credentials were used. No MCP server was started.

---

## 1. Summary and recommendation

| Decision | Recommendation | Why |
|---|---|---|
| GitHub server | **`github/github-mcp-server` local binary (stdio)**, v1.12.x, Windows x86_64 zip. Start it with `--toolsets context,repos,issues,pull_requests` plus an explicit `--tools` list, and authenticate with a **fine-grained PAT** | Official and maintained. The npm `@modelcontextprotocol/server-github` is **deprecated/archived**. Unlike the remote endpoint, the binary has no possible Copilot-licence dependency. It needs no Docker daemon or `npx`. `--tools` / `--read-only` limit the tool surface at the server |
| GitHub fallback | Remote `https://api.githubcopilot.com/mcp/` (http plus `Authorization: Bearer <PAT>`), with the `/readonly` path or `X-MCP-Readonly: true` in Phase 2 | Zero install, but whether it works without a Copilot licence is **contradictory in GitHub's docs** (§6, Q1) |
| Aggregator | **Composio, in "dedicated MCP server" mode** (not the session/Tool Router mode) | Static URL plus a static `x-api-key` header works headless from the SDK. The per-app allowlist is enforced **server-side** (`allowed_tools` when the server is created) *and* on our side. Tool names are stable and descriptive (`GMAIL_FETCH_EMAILS`, `GOOGLECALENDAR_EVENTS_LIST`, `OUTLOOK_LIST_MESSAGES`), so Spark's allowlist and confirmation hook can match on them. Gmail, Calendar and Outlook all offer Composio-managed OAuth (a one-time browser consent) |
| Runner-up | Zapier MCP | Simple long-lived bearer token on one URL. But **every call costs 2 Zapier tasks**, tool names are less predictable (**UNVERIFIED** format), and its meta tools (`discover_zapier_actions`, `enable_zapier_action`) let the agent widen its own tool set unless blocked |
| Not recommended now | Pipedream MCP | Bearer access tokens come from OAuth client credentials and **expire after 1 hour**, so a static header in `mcp_servers.json` goes stale. Spark would need a token minter plus reconnects. It also needs 5 required headers, with **one app per connection** (`x-pd-app-slug`) |
| Work Outlook / M365 | **Do not connect in Phase 2.** Personal Google (Gmail and Calendar) first, read-only | §12: a work tenant needs admin consent. Composio's Outlook toolkit notes that tenant admin consent is required. Try it only after the tenant admin approves Composio's (or our own) Entra app registration and scopes |

**Key SDK facts** (verified, §4):
- The server config keys are exactly `type`, `url`, `headers` (http/sse) and `command`, `args`, `env` (stdio).
- Tool names are `mcp__<server>__<tool>`.
- `allowed_tools` **accepts** `mcp__server`, `mcp__server__*` and prefix globs like `mcp__github__get_*` (CLI binary). Spark should still expand globs itself so it can log and test the list.
- **The SDK puts the whole `mcp_servers` dict on the CLI command line as JSON**, so Spark must **not** pre-expand secrets (§4.4).

---

## 2. Per-connector tables

### 2.1 GitHub

| | github-mcp-server binary (**recommended**) | Remote `api.githubcopilot.com/mcp/` | npm `@modelcontextprotocol/server-github` |
|---|---|---|---|
| Status | Official, active. Latest release **v1.12.2 (2026-09-16)** includes `github-mcp-server_Windows_x86_64.zip` [R3] | Official, hosted by GitHub [R2] | **Deprecated on npm ("Package no longer supported"); source moved to `modelcontextprotocol/servers-archived`**, "no security guarantees" [R4]. Do not use |
| Transport | stdio (`github-mcp-server stdio`) [R5] | http (Streamable HTTP) [R2][R5] | stdio via `npx` |
| Auth | env `GITHUB_PERSONAL_ACCESS_TOKEN` [R1][R5] | `Authorization: Bearer <PAT>` header [R5]. OAuth is the default in IDEs, but the Agent SDK cannot run the OAuth flow (§4.5), so use a PAT | env `GITHUB_PERSONAL_ACCESS_TOKEN` |
| Spark env var | `SPARK_GITHUB_PAT` (see §5 for why not `GITHUB_TOKEN`) | same | n/a |
| Interactive OAuth? | No | No with a PAT. OAuth needs `/mcp` in interactive Claude Code, which Spark can't do headless | No |
| Read-only switch | `--read-only` flag or `GITHUB_READ_ONLY=1` env. `--toolsets a,b` or `GITHUB_TOOLSETS`. `--tools t1,t2` enables specific tools; "read-only mode takes priority: write tools are skipped if `--read-only` is set, even if explicitly requested via `--tools`" [R1] | URL paths `/readonly`, `/x/{toolset}`, `/x/{toolset}/readonly`, and headers `X-MCP-Readonly`, `X-MCP-Toolsets`, `X-MCP-Tools`, `X-MCP-Lockdown` [R2] | none |
| Default toolsets | `context, repos, issues, pull_requests, users` [R1] | same | n/a |
| Licence caveat | none | GitHub docs say "available to all GitHub users regardless of plan type" [R6], but the other-IDE guide lists "GitHub Copilot License: Any Copilot plan" as a prerequisite [R7]. **UNVERIFIED for a non-Copilot account from the Agent SDK** | n/a |
| Windows | Unzip the exe and use an absolute path as `command`. No `npx` or `cmd /c` needed | nothing to spawn | would need `cmd /c npx` (§4.6) |

**Read-only tools (pre-approve)**, with names from the current README [R1]. Note that the
old npm names `get_pull_request`, `get_pull_request_status` and `create_issue` **no longer exist**.
- context: `get_me`
- pull_requests: `list_pull_requests`, `pull_request_read` (methods `get`, `get_diff`, `get_status`, `get_files`, `get_commits`, `get_review_comments`, `get_reviews`, `get_comments`, `get_check_runs`), `search_pull_requests`
- issues: `list_issues`, `issue_read`, `search_issues`
- repos: `get_file_contents`, `list_commits`, `get_commit`, `list_branches`, `search_repositories`, `get_latest_release`, `list_releases`
- actions (not a default toolset, optional for "did CI pass"): `actions_list`, `actions_get`

**Side-effecting tools (confirmation hook):** `create_repository` (needed by the Phase 3
`start_project`), `issue_write` (method `create`/`update`, which covers "create issue"),
`add_issue_comment`, `create_pull_request`, `update_pull_request`, `create_branch`.

**Destructive (off unless `enabled_destructive: true`):** `merge_pull_request`, `delete_file`,
`delete_repository` (this one is in the README tool list; **UNVERIFIED** whether it ships in v1.12.2),
`push_files`, `create_or_update_file`, `update_pull_request_branch`, `fork_repository`,
`actions_run_trigger`.

**Fine-grained PAT permissions** [R8]:

| Need | Permission |
|---|---|
| PR list and details (`list_pull_requests`, `pull_request_read get/files/diff`) | Pull requests: **Read** |
| PR status (`pull_request_read get_status`) | Commit statuses: **Read** |
| Check runs (`get_check_runs`) | Checks: **Read** (**UNVERIFIED**: the endpoint was not in the fetched table) |
| Issues (`list_issues`, `issue_read`) | Issues: **Read** |
| File contents and commits | Contents: **Read** |
| Always | Metadata: Read (mandatory, auto-selected) |
| Phase 3 `create_repository` | **Administration: Read and write** (`POST /user/repos`), on "All repositories" |
| `issue_write create` | Issues: Read and write |
| `merge_pull_request` (destructive, leave off) | Contents: Read and write |

Search endpoints (`search_issues`, `search_pull_requests`) are not in the permissions table.
They only see repos that the token can read (**UNVERIFIED**).

**Recommendation:** use two PATs, or one PAT that is upgraded later. Phase 2 gets a read-only
fine-grained PAT (Metadata, Contents, Issues, Pull requests, Commit statuses, Checks, all *Read*)
on the repos you care about. Phase 3 adds Administration and Issues *write* only when
`start_project` is enabled. Alternatively, have `start_project` use `gh repo create` (the `gh`
2.97 on this laptop uses its own stored auth), so the MCP PAT can stay read-only.

### 2.2 Aggregators compared

| | **Composio** (recommended) | Zapier MCP | Pipedream MCP |
|---|---|---|---|
| Transport | http (Streamable HTTP). The Claude Agent SDK example uses `type: 'http'` [R10] | http (Streamable HTTP) [R12] | http **or** sse, "supports both … dynamically" [R14] |
| URL shape | Dedicated server: `https://backend.composio.dev/v3/mcp/<SERVER_ID>?user_id=<USER_ID>` [R9]. Session mode: `session.mcp.url`, created at runtime by the Composio SDK [R10] | `https://mcp.zapier.com/api/v1/connect` [R12] | `https://remote.mcp.pipedream.net/v3` [R14] |
| Auth | Header `x-api-key: <COMPOSIO_API_KEY>`. It is required when `require_mcp_api_key` is on, which is the default for new orgs [R9]. `user_id` in the URL is an identifier, not a secret | `Authorization: Bearer <connection token>` (preferred) or `?token=` in the URL (avoid: "more likely to end up in a log"). The token is "long-lived, tied to that one server" [R12] | `Authorization: Bearer <access token>` **plus** `x-pd-project-id`, `x-pd-environment`, `x-pd-external-user-id`, `x-pd-app-slug` [R14]. The token comes from client credentials at `POST https://api.pipedream.com/v1/oauth/token` and **expires in 3600 s** [R15] |
| Per-app OAuth (Gmail/Calendar/Outlook) | Once per app: create an auth config (Composio-managed OAuth app available for all three) and connect the account for your `user_id` through a Connect Link in the browser. "Composio automatically refreshes OAuth tokens" [R11][R16][R17][R18] | Once, in the Zapier dashboard (mcp.zapier.com): add the app and pick actions; the account connects in the browser [R13] | Once per app: the server returns a Connect Link (`https://pipedream.com/_static/connect.html?token=ctok_…&app=<slug>`) when the account is missing [R14] |
| Headless from Agent SDK with URL plus header? | **Yes** (dedicated server mode) | **Yes** | **Partly**: needs token refresh every hour, and a reconnect of the client or MCP server |
| Tool naming (as model sees it) | `mcp__composio__GMAIL_FETCH_EMAILS` (`TOOLKIT_ACTION` [R9]) | Something like `mcp__zapier__gmail_find_email` (**UNVERIFIED** exact format), plus meta tools `discover_zapier_actions`, `enable_zapier_action`, `get_configuration_url`, `list-skills` [R13] | Per-app slug tools, e.g. `gmail-send-email` (**UNVERIFIED**) |
| Server-side restriction | `allowed_tools` on `composio.mcp.create(...)` or in the dashboard [R9] | Enable only chosen actions per server [R13]. The agent-side meta tools can enable more, so **disallow them** | `x-pd-app-slug` limits to one app per connection. Per-tool limits **UNVERIFIED** |
| Cost | Plan-based (**UNVERIFIED** free-tier limits) | **2 tasks per successful call**. Listing tools is free [R19] | Plan-based (**UNVERIFIED**) |
| Caveat | Composio docs say "for most use cases, use a regular session instead" [R9]. Session mode exposes **meta tools** (`COMPOSIO_SEARCH_TOOLS`, `COMPOSIO_MANAGE_CONNECTIONS`, `COMPOSIO_MULTI_EXECUTE_TOOL`, …) [R10], which would hide the real action inside a tool input and defeat name-based allowlists and confirmation. **Spark must use dedicated servers** | Task billing adds up for chatty voice use | Pipedream was acquired by Workday (**UNVERIFIED** effect on the product) |

### 2.3 Google Calendar (via Composio `GOOGLECALENDAR` toolkit)

| Field | Value |
|---|---|
| Transport / auth | http, `x-api-key` header, OAuth2 connected account (Composio-managed app available) [R17] |
| Env vars | `COMPOSIO_API_KEY`, `COMPOSIO_MCP_URL` |
| Interactive OAuth | One-time Google consent in the browser, in Composio's dashboard or Connect Link. Afterwards headless |
| Read-only (pre-approve) | `GOOGLECALENDAR_EVENTS_LIST`, `GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS`, `GOOGLECALENDAR_FIND_EVENT`, `GOOGLECALENDAR_EVENTS_GET`, `GOOGLECALENDAR_FIND_FREE_SLOTS`, `GOOGLECALENDAR_FREE_BUSY_QUERY`, `GOOGLECALENDAR_LIST_CALENDARS` |
| Side-effecting (confirm) | `GOOGLECALENDAR_CREATE_EVENT`, `GOOGLECALENDAR_QUICK_ADD`, `GOOGLECALENDAR_PATCH_EVENT` |
| Destructive (off) | `GOOGLECALENDAR_DELETE_EVENT`, `GOOGLECALENDAR_CLEAR_CALENDAR`, `GOOGLECALENDAR_CALENDARS_DELETE`, `GOOGLECALENDAR_BATCH_EVENTS`, `GOOGLECALENDAR_EVENTS_MOVE`, `GOOGLECALENDAR_REMOVE_ATTENDEE`, all `GOOGLECALENDAR_ACL_*` writes |
| Recommendation | Phase 2: read-only only. Add `CREATE_EVENT` behind confirmation in Phase 3 |

### 2.4 Gmail (via Composio `GMAIL` toolkit)

| Field | Value |
|---|---|
| Transport / auth | http, `x-api-key`, OAuth2 connected account (Composio-managed app available) [R16] |
| Env vars | `COMPOSIO_API_KEY`, `COMPOSIO_MCP_URL` (same server as Calendar; one dedicated server can hold several toolkits, **UNVERIFIED** for >1 toolkit in "single toolkit" docs, so it may need one server per toolkit) |
| Interactive OAuth | One-time Google consent. Composio notes that an OAuth client requesting scopes Google hasn't verified causes "App is blocked" [R16], so prefer the Composio-managed app |
| Read-only (pre-approve) | `GMAIL_FETCH_EMAILS`, `GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID`, `GMAIL_FETCH_MESSAGE_BY_THREAD_ID`, `GMAIL_LIST_THREADS`, `GMAIL_LIST_LABELS`, `GMAIL_GET_PROFILE`, `GMAIL_SEARCH_PEOPLE` |
| Side-effecting (confirm) | `GMAIL_CREATE_EMAIL_DRAFT`, `GMAIL_UPDATE_DRAFT`, `GMAIL_ADD_LABEL_TO_EMAIL` |
| Destructive (off) | `GMAIL_SEND_EMAIL`, `GMAIL_SEND_DRAFT`, `GMAIL_REPLY_TO_THREAD`, `GMAIL_FORWARD_MESSAGE` (sends are irreversible, so keep them off until explicitly wanted, then move them to side_effecting), `GMAIL_DELETE_*`, `GMAIL_BATCH_DELETE_MESSAGES`, `GMAIL_MOVE_TO_TRASH`, `GMAIL_MOVE_THREAD_TO_TRASH`, `GMAIL_CREATE_FILTER`, `GMAIL_BATCH_MODIFY_MESSAGES` |
| Recommendation | Phase 2: read plus summarize. Drafting (not sending) is the first write to enable |

### 2.5 Outlook / Microsoft 365 (via Composio `OUTLOOK` toolkit)

| Field | Value |
|---|---|
| Transport / auth | http, `x-api-key`, OAuth2 (Microsoft Graph) connected account [R18] |
| Env vars | `COMPOSIO_API_KEY`, `COMPOSIO_MCP_URL_OUTLOOK` (separate dedicated server, so it can be switched off independently) |
| Interactive OAuth | One-time Microsoft sign-in. **For a work tenant: "Tenant administrators must grant consent via Azure Portal before users can connect"** [R18]. Composio's page also says work/school accounts are required (**UNVERIFIED** whether that applies to all tools or only some) |
| Read-only (pre-approve) | `OUTLOOK_LIST_MESSAGES`, `OUTLOOK_GET_MESSAGE`, `OUTLOOK_LIST_EVENTS`, `OUTLOOK_LIST_CALENDARS`, `OUTLOOK_LIST_MAIL_FOLDERS` |
| Side-effecting (confirm) | `OUTLOOK_CREATE_DRAFT`, `OUTLOOK_CALENDAR_CREATE_EVENT` |
| Destructive (off) | `OUTLOOK_SEND_EMAIL`, `OUTLOOK_SEND_DRAFT`, `OUTLOOK_DELETE_MESSAGE`, `OUTLOOK_MOVE_MESSAGE`, `OUTLOOK_BATCH_UPDATE_MESSAGES` |
| Recommendation | **`enabled: false`** by default (§12). When approved, read-only first. The toolkit has 306 tools [R18], so an explicit allowlist is mandatory |

---

## 3. Proposed files

> **Shipped today (0.1.0):** `mcp_servers.json` and `config.yaml` carry one sample, `github`,
> pointing at GitHub's hosted server (`https://api.githubcopilot.com/mcp/`, token in
> `SPARK_GITHUB_PAT`), switched off. Fill in the token in the Connectors window (*Edit config*)
> or in `.env`. The local exe and the Composio servers below are kept as `_..._example` entries
> in `mcp_servers.example.json`: rename one and add a `connectors.servers` entry to use it. The
> rest of this section is the original P2.1 proposal.

### 3.1 `mcp_servers.json` (placeholders only)

```json
{
  "github": {
    "type": "stdio",
    "command": "${SPARK_GITHUB_MCP_BIN}",
    "args": ["stdio", "--toolsets", "context,repos,issues,pull_requests",
             "--tools", "get_me,list_pull_requests,pull_request_read,search_pull_requests,list_issues,issue_read,search_issues,get_file_contents,list_commits,get_commit,list_branches,create_repository,issue_write"],
    "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${SPARK_GITHUB_PAT}" }
  },
  "github_remote": {
    "type": "http",
    "url": "https://api.githubcopilot.com/mcp/readonly",
    "headers": { "Authorization": "Bearer ${SPARK_GITHUB_PAT}" }
  },
  "composio": {
    "type": "http",
    "url": "${COMPOSIO_MCP_URL}",
    "headers": { "x-api-key": "${COMPOSIO_API_KEY}" }
  },
  "composio_outlook": {
    "type": "http",
    "url": "${COMPOSIO_MCP_URL_OUTLOOK}",
    "headers": { "x-api-key": "${COMPOSIO_API_KEY}" }
  }
}
```

Notes:
- `github` and `github_remote` are alternatives. `config.yaml` enables at most one.
- Phase 2 can add `"--read-only"` to the `github` args. That also strips `create_repository` and `issue_write`, which is fine until Phase 3.
- Whether `--tools` alone already hides the unlisted tools of an enabled toolset is **UNVERIFIED**; the README describes it as "enable specific tools". Spark's allowlist and `disallowed_tools` guard regardless.
- stdio `env`: the CLI merges `env` for the child. Whether the child also inherits the full CLI environment is **UNVERIFIED**, so pass what the server needs explicitly.
- Pipedream and Zapier are not included; see §2.2.

### 3.2 `config.yaml` → `connectors:`

```yaml
connectors:
  enabled_destructive: false        # global kill switch; destructive tools stay out unless true
  github:
    enabled: true                    # needs SPARK_GITHUB_PAT and SPARK_GITHUB_MCP_BIN
    allowed_tools:                   # read-only, pre-approved (bare names; mcp__github__ is added)
      - get_me
      - list_pull_requests
      - pull_request_read
      - search_pull_requests
      - list_issues
      - issue_read
      - search_issues
      - get_file_contents
      - list_commits
      - get_commit
      - list_branches
    side_effecting:                  # allowed only through the spoken confirmation hook
      - create_repository
      - issue_write
      - add_issue_comment
      - create_pull_request
    destructive:                     # never exposed unless enabled_destructive: true
      - merge_pull_request
      - delete_repository
      - delete_file
      - push_files
      - create_or_update_file
      - fork_repository
      - actions_run_trigger
      - update_pull_request_branch
  github_remote:
    enabled: false
    allowed_tools: [list_pull_requests, pull_request_read, search_pull_requests, list_issues, issue_read, search_issues, get_me]
    side_effecting: []
    destructive: []
  composio:
    enabled: true                    # needs COMPOSIO_API_KEY and COMPOSIO_MCP_URL
    allowed_tools:
      - GMAIL_FETCH_EMAILS
      - GMAIL_FETCH_MESSAGE_BY_THREAD_ID
      - GMAIL_LIST_THREADS
      - GMAIL_GET_PROFILE
      - GOOGLECALENDAR_EVENTS_LIST
      - GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS
      - GOOGLECALENDAR_FIND_EVENT
      - GOOGLECALENDAR_FIND_FREE_SLOTS
    side_effecting:
      - GMAIL_CREATE_EMAIL_DRAFT
      - GOOGLECALENDAR_CREATE_EVENT
    destructive:
      - GMAIL_SEND_EMAIL
      - GMAIL_SEND_DRAFT
      - GMAIL_REPLY_TO_THREAD
      - GMAIL_DELETE_*               # glob: expanded against the server's live tool list, else passed as a deny rule
      - GMAIL_MOVE_TO_TRASH
      - GOOGLECALENDAR_DELETE_EVENT
      - GOOGLECALENDAR_CLEAR_CALENDAR
  composio_outlook:
    enabled: false                   # work tenant: needs admin consent (§12)
    allowed_tools: [OUTLOOK_LIST_MESSAGES, OUTLOOK_GET_MESSAGE, OUTLOOK_LIST_EVENTS, OUTLOOK_LIST_CALENDARS]
    side_effecting: []
    destructive: [OUTLOOK_SEND_EMAIL, OUTLOOK_SEND_DRAFT, OUTLOOK_DELETE_MESSAGE, OUTLOOK_MOVE_MESSAGE]
```

How `mcp_config.py` should map this (P2.2):
- `allowed_tools` (SDK option) = `mcp__<server>__<tool>` for every `allowed_tools` entry of enabled servers, plus `mcp__spark__*` read-only in-process tools. **Do not** put `side_effecting` tools in `allowed_tools`: they fall through to the PreToolUse hook, which returns allow or deny explicitly.
- `disallowed_tools` = `mcp__<server>__<tool>` for every `destructive` entry when `enabled_destructive` is false. Deny rules accept wildcards anywhere (CLI binary, §4.3), so `mcp__composio__GMAIL_DELETE_*` is valid there. Also add `mcp__zapier__enable_zapier_action`-style meta tools if Zapier is ever used.
- Hook matcher = a regex alternation of the side-effecting full names (`HookMatcher.matcher`, `types.py:610`).
- Anything on a server that is in none of the three lists: with `permission_mode` unset (default) and no `can_use_tool`, the CLI would "ask". In SDK mode that is a denial (**UNVERIFIED** exact behavior). Spark should make it explicit. Either add `mcp__<server>__*` minus the allowed tools to `disallowed_tools` (not expressible as a set difference), or better, route every `mcp__.*` call through the hook and **deny unknown tools** there. The server-side restrictions (`--tools`, Composio `allowed_tools`) are the first line of defense.

---

## 4. SDK side (verified)

### 4.1 Config shapes
Already verified in `docs/sdk_notes.md` §1 "MCP server configs" (`types.py:627-672`):
- `McpStdioServerConfig`: `type?: "stdio"`, `command`, `args?`, `env?`
- `McpSSEServerConfig`: `type: "sse"`, `url`, `headers?`
- `McpHttpServerConfig`: `type: "http"`, `url`, `headers?`
- `McpSdkServerConfig`: `type: "sdk"`, `name`, `instance`

Import them from `claude_agent_sdk.types`. `ClaudeAgentOptions.mcp_servers: dict | str | Path` is at `types.py:2002`.

### 4.2 Tool naming
- Tools are named `mcp__<server>__<tool>` (sdk_notes; the Claude Code docs say the same [R20]).
- The server key is normalized by the CLI function `wn`: `e.replace(/[^a-zA-Z0-9_-]/g,"_")` (CLI binary). Keep server keys to `[a-z0-9_]` so the names are predictable. **Avoid double underscores in server keys**: the CLI splits the name on `"__"` to get the server and tool (`Ls`: `let n=e.split("__"),[t,r,...o]=n`, CLI binary).

### 4.3 `allowed_tools` / `disallowed_tools` wildcards
- SDK side: `allowed_tools` is passed through as `--allowedTools` and means "auto-approved without prompting", **not** an availability filter (docstring `types.py:1976-1981`). `disallowed_tools` means "removed from the model's context and cannot be used, even if they would otherwise be allowed" (`types.py:2053-2057`). Deny wins.
- The SDK's own parser, `_whole_tool_allowed` (`types.py:1834-1848`), treats `"X"`, `"X()"` and `"X(*)"` as whole-tool entries.
- CLI matching (bundled binary), function `qan(rule, tool)`: `t.serverName===r.serverName&&(t.toolName===void 0||t.toolName==="*"||… k3e(t.toolName,r.toolName))`, where `k3e` is a `*`→`.*` anchored regex. So:
  - `mcp__github` matches **all** tools of that server.
  - `mcp__github__*` matches all tools.
  - `mcp__github__get_*` is a prefix glob. The CLI validation message says: "globs are permitted only in the tool position after a literal `mcp__<server>__` prefix. Deny and ask rules accept wildcards anywhere", with examples `mcp__puppeteer__*` and `mcp__github__get_*`.
  - The agent-definition docs string also says server-level specs (`mcp__server`, `mcp__server__*`, `mcp__*`) remove every tool from the named server (or all MCP tools) in `disallowedTools`.
- **Spark policy:** expand globs itself at startup (against the configured lists, or the live `tools/list` when doctor runs), so the allowlist is explicit, loggable and unit-testable. Never pass `mcp__<server>` or `mcp__<server>__*` in `allowed_tools`, because that would auto-approve side-effecting tools and bypass the hook's "default deny" intent. (A PreToolUse hook still runs for auto-approved tools and can deny, but relying on that is fragile.)

### 4.4 Secrets on the command line (new finding, important)
- `subprocess_cli.py:663-684`: a dict `mcp_servers` is serialized with `json.dumps({"mcpServers": ...})` and passed **as an argv value** to `--mcp-config`. If `mcp_config.py` substitutes `${VAR}` itself, the PAT and Composio key end up in the `claude.exe` command line, where any process of the same user can read them and EDR tools on a company laptop may log them.
- The CLI parses an inline `--mcp-config` with **`expandVars:!0`** (`DZe({configObject:nr,filePath:"command line",expandVars:!0,scope:"dynamic",...})`, CLI binary). The CLI therefore expands `${VAR}` and `${VAR:-default}` itself, in `command`, `args`, `env`, `url` and `headers` [R20], from its own environment (= `os.environ` plus `ClaudeAgentOptions.env`, `subprocess_cli.py:815-820` per sdk_notes).
- **Recommendation for P2.2:**
  - In Python, check that every referenced var is set and non-empty (disable the server with a warning if not, per the contract).
  - Then pass the config **with `${VAR}` placeholders left intact**, so the secrets only travel in the child's environment.
  - A unit test should assert that the options dict contains no secret values.
- Caveat: the CLI reads some credential vars as **empty** in remote `url`/`headers` (`ANTHROPIC_API_KEY`, `NPM_TOKEN`, `HTTPS_PROXY`, `GOOGLE_OAUTH_ACCESS_TOKEN`, etc.; lists in the CLI binary and [R20]). None of the proposed names (`SPARK_GITHUB_PAT`, `COMPOSIO_API_KEY`, `COMPOSIO_MCP_URL*`) is on those lists.
- Alternative: write a temp JSON file and pass a `Path` (the SDK passes `str(path)` at `subprocess_cli.py:688`). That moves the secret to disk instead, which is worse.

### 4.5 OAuth-only remote servers do not work headless
Claude Code docs: "Claude Code can't run the OAuth flow for you. As of v2.1.196, when a configured server needs authentication during a `claude -p` or Agent SDK run … Claude Code tells Claude that the server's tools are unavailable until you authorize it" [R20]. Every Spark connector must therefore use a static header or env credential. (The docs also describe `headersHelper` for dynamic headers [R20]. It is not in the SDK TypedDicts, and it only runs after workspace trust for project/local scope, so it is **UNVERIFIED** for `--mcp-config` dynamic scope. It would be the only clean way to use Pipedream's 1-hour tokens.)

### 4.6 Windows `npx`
- `npx` on this laptop resolves to `C:\Program Files\nodejs\npx.cmd` (plus a bash shim `npx`). Node's `spawn` cannot run a `.cmd` without a shell. The CLI's grep shows no special npx handling. Claude Code issues and docs require the **`cmd /c` wrapper** on native Windows for npx-based stdio servers [R21].
- For P2.2, when `sys.platform == "win32"` and `command` is `npx`/`npx.cmd`/`uvx`, rewrite `{"command": "cmd", "args": ["/c", "npx", ...]}`.
- The known "/c mangled to C:/" bug is in `claude mcp add` argument parsing [R21], not in JSON configs. It should not affect `--mcp-config` (**UNVERIFIED**).
- The recommended config avoids `npx` entirely: the GitHub binary is an exe, and Composio is http.

### 4.7 claude.ai account connectors (verified live 2026-10-05)
- The logged-in CLI can bring in the user's **claude.ai connectors** (Gmail, Google Calendar, Slack, Microsoft 365, and so on): the ones connected on claude.ai (Settings -> Connectors, https://claude.ai/settings/connectors) or with `/mcp` in Claude Code.
- They load **only when `ENABLE_CLAUDEAI_MCP_SERVERS` is truthy AND `strict_mcp_config=False`**. Resolved: `strict_mcp_config=True` alone keeps them out (`get_mcp_status()` listed none). A falsy env var (`"0"`, `"false"`, `"no"`, `"off"`) or the `disableClaudeAiConnectors` setting also disables them.
- `get_mcp_status()` names them `"claude.ai Gmail"`, `"claude.ai Google Calendar"`. The model sees `mcp__claude_ai_Gmail__<tool>` (server key: `"claude.ai "` -> `"claude_ai_"`, spaces -> `_`).
- Resolved: a server-level `disallowed_tools` entry (`mcp__claude_ai_Slack`) hides every tool of that server, and a tool-level entry hides one tool (checked in the init tool list).
- Non-strict mode could also load user/project MCP servers in principle (none on this machine with `setting_sources=[]`), so the PreToolUse gate keeps denying every tool outside Spark's policy.

**How Spark uses them** (`spark/brain/claudeai.py`):
- Configure each as an ordinary `connectors.servers` entry whose key starts with `claude_ai_`. `claude_ai_Gmail` and `claude_ai_Google_Calendar` are in the defaults, **disabled**: they
  work only through a claude.ai login in the Claude CLI, and Anthropic's Agent SDK terms don't
  let a third-party app offer that login. On your own machine, with your own CLI login, you can
  set `enabled: true`. They come with the usual `allowed_tools` / `side_effecting` / `destructive` / `allow_destructive` / `route`. They need no `mcp_servers.json` entry: the CLI injects the server, and the loader only registers the policy.
- `block_claudeai_connectors` (default `false`): `true` forces them all off. Otherwise they load only when at least one `claude_ai_*` server is enabled. Then the voice session runs non-strict with `ENABLE_CLAUDEAI_MCP_SERVERS=true`. The router classifier and the side-question sessions always stay strict, with the variable `false`.
- Hiding the rest: each session's status check saves the claude.ai servers the CLI reported to `claudeai_connectors.json` (next to `config.yaml`, app-managed). The next session (there is one per wake) adds a server-level `mcp__<key>` deny rule for each of them that isn't enabled. Configured-but-off servers get one too, and so does a static list of common ones (Slack, Asana, Atlassian Rovo, Claude Docs, Microsoft 365, Google Drive) for the very first session. Disabled destructive tools are tool-level deny rules, as for any server. Anything that still slips through is "unknown" to the policy, and the gate denies it.
- Status: `needs-auth` is reported as "sign in on claude.ai (Settings -> Connectors) or run /mcp in Claude Code" in the log and the doctor, and once per session to the model.
- The Connectors window lists every claude.ai server from the last status under "claude.ai account", with the same on/off switch and per-tool Allow / Ask / Off. Those choices are stored in `connectors.json` (`"kind": "claude_ai"`) and override the `config.yaml` defaults.

---

## 5. What a human must do

Put all values in `.env` (never commit it). `.env.example` currently has `GITHUB_TOKEN`,
`AGGREGATOR_MCP_URL` and `AGGREGATOR_API_KEY`. P2.2 should rename them to the names below.

**Why not `GITHUB_TOKEN`:** `gh` gives `GITHUB_TOKEN`/`GH_TOKEN` in the environment priority over its
stored login. A read-only PAT exported as `GITHUB_TOKEN` would silently change the identity used by
`gh repo create` in `start_project`. The bundled CLI also treats `GITHUB_TOKEN`/`GH_TOKEN` specially:
they are in its credential-name sets (e.g. `new Set([...,"GITHUB_TOKEN","GH_TOKEN"])` and
`{clis:["gh"],realCredentialEnv:["GH_TOKEN","GITHUB_TOKEN"],placeholders:...}`, CLI binary), used for
env scrubbing and placeholder substitution. Whether that affects `${GITHUB_TOKEN}` expansion in MCP
config is **UNVERIFIED**, so a Spark-specific name avoids the question.

1. **GitHub PAT** → `SPARK_GITHUB_PAT`
   1. Go to github.com → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token.
   2. Resource owner: your user. Expiration: 90 days. Repository access: *Only select repositories* (or All).
   3. Repository permissions: **Issues: Read and write** (so Spark can create issues and comments after you say yes), and Read-only for **Contents, Pull requests, Commit statuses, Checks** (Metadata is added automatically). For a read-only Spark, make Issues Read-only too.
   4. `start_project` creates repos with `gh`, not this PAT, so the PAT never needs **Administration**.
   5. If the account is an Enterprise Managed User, PATs may be disabled by the enterprise admin [R6].
2. **GitHub MCP binary** → `SPARK_GITHUB_MCP_BIN`
   1. Download `github-mcp-server_Windows_x86_64.zip` from https://github.com/github/github-mcp-server/releases (v1.12.2 or later) and verify it against the release checksums file.
   2. Unzip it to e.g. `C:/Users/<you>/.local/bin/github-mcp-server.exe`, and set `SPARK_GITHUB_MCP_BIN` to that absolute path.
   3. App-control policies on managed machines may block unsigned exes. Use the hosted server (the shipped `github` sample) instead.
   4. To run the exe, replace the `github` entry in `mcp_servers.json` with `_github_local_example` from `mcp_servers.example.json`.
3. **Composio** → `COMPOSIO_API_KEY`, `COMPOSIO_MCP_URL`, (later) `COMPOSIO_MCP_URL_OUTLOOK`
   1. Sign up at https://dashboard.composio.dev and copy the project API key into `COMPOSIO_API_KEY`.
   2. Auth configs: add **Gmail** and **Google Calendar** with *Composio-managed OAuth*.
   3. Connect accounts: for user id `spark-<yourname>` (any stable string), complete the Google consent in the browser for both toolkits.
   4. MCP → create a server containing the Gmail and Google Calendar toolkits with **allowed tools** = the `allowed_tools` plus `side_effecting` lists in §3.2 (leave the destructive ones out).
   5. Copy its URL `https://backend.composio.dev/v3/mcp/<id>?user_id=spark-<yourname>` into `COMPOSIO_MCP_URL`.
   6. Keep "require MCP API key" **on**.
4. **Outlook (work)**: for a work/school tenant, get admin consent first. Raise a request naming the app (Composio's Entra app, or our own registration) and delegated Graph scopes `Mail.Read` and `Calendars.Read` only.
5. In `config.yaml`, set `enabled: true` under `connectors.servers.github` (it ships off), and add a `connectors.servers.composio` entry with the lists in §3.2 for Composio. Verify with `uv run python -m spark doctor` (P2.5). Then run the text-mode phrases in `docs/dev/history/phase2_acceptance.md` (P2.6).

---

## 6. Open questions and uncertainties

1. **Remote GitHub MCP and Copilot licence.** [R6] says the server is available to all users regardless of plan. [R7] lists a Copilot licence as a prerequisite, and third-party posts claim the remote endpoint requires Copilot. This is **UNVERIFIED** for a PAT from a non-IDE client. It is the reason the local binary is recommended.
2. Whether `--tools` restricts the listed tools *within* the enabled toolsets, or adds to them, is **UNVERIFIED** (README wording [R1]). A human can check with `github-mcp-server stdio --tools …` and `tools/list` (P2.5 doctor).
3. Whether a Composio dedicated MCP server may hold multiple toolkits (Gmail and Calendar together) is **UNVERIFIED**. The docs page is titled "Single Toolkit MCP", yet the REST API has "Create a new custom MCP server with multiple apps" [R9][R22]. The fallback is one server per toolkit (`COMPOSIO_MCP_URL_GMAIL`, `COMPOSIO_MCP_URL_GCAL`).
4. Composio is steering users toward sessions and the Tool Router [R9][R10]. The dedicated-server mode could be deprecated later, which is a risk to watch.
5. Composio free-tier limits and pricing: **UNVERIFIED**.
6. Zapier exact MCP tool-name format: **UNVERIFIED** (needs a live `tools/list`).
7. ~~`strict_mcp_config=True` versus claude.ai connectors (§4.7)~~ **Resolved 2026-10-05**: strict mode keeps them out. They need the env var truthy and non-strict mode.
8. Behavior for an MCP tool in none of the permission lists, in SDK mode with no `can_use_tool`: **UNVERIFIED**. Deny unknown tools in the hook.
9. Whether stdio children inherit the full CLI env: **UNVERIFIED** (pass needed vars explicitly in `env`).
10. `headersHelper` support for `--mcp-config` dynamic scope: **UNVERIFIED** (only relevant for Pipedream).
11. GitHub "Checks: Read" for `get_check_runs`, and the search endpoints' permission needs: **UNVERIFIED** [R8].

## Sources

- [R1] github/github-mcp-server README: https://github.com/github/github-mcp-server (raw: https://raw.githubusercontent.com/github/github-mcp-server/main/README.md)
- [R2] Remote server docs: https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md
- [R3] Latest release: https://api.github.com/repos/github/github-mcp-server/releases/latest
- [R4] Archived reference servers: https://github.com/modelcontextprotocol/servers-archived/tree/main/src/github
- [R5] Install guide for Claude: https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-claude.md
- [R6] GitHub Docs, set up the GitHub MCP server: https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp-in-your-ide/set-up-the-github-mcp-server
- [R7] Install guide, other Copilot IDEs: https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-other-copilot-ides.md
- [R8] Fine-grained PAT permissions: https://docs.github.com/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens
- [R9] Composio single toolkit MCP: https://docs.composio.dev/docs/single-toolkit-mcp (and https://docs.composio.dev/docs/mcp-overview)
- [R10] Composio + Claude Agent SDK: https://composio.dev/toolkits/composio/framework/claude-agents-sdk ; Tool Router quick start: https://docs.composio.dev/docs/tool-router/quick-start
- [R11] Composio authenticating tools: https://docs.composio.dev/docs/authenticating-tools
- [R12] Zapier MCP connections: https://docs.zapier.com/mcp/overview/how-connections-work
- [R13] Zapier first MCP workflow: https://docs.zapier.com/mcp/get-started/your-first-mcp-workflow
- [R14] Pipedream MCP for developers: https://pipedream.com/docs/connect/mcp/developers
- [R15] Pipedream REST auth (1-hour client-credentials tokens): https://pipedream.com/docs/rest-api/auth
- [R16] Composio Gmail toolkit: https://docs.composio.dev/toolkits/gmail
- [R17] Composio Google Calendar toolkit: https://docs.composio.dev/toolkits/googlecalendar
- [R18] Composio Outlook toolkit: https://docs.composio.dev/toolkits/outlook
- [R19] Zapier MCP usage and billing: https://docs.zapier.com/mcp/overview/usage
- [R20] Claude Code MCP docs: https://code.claude.com/docs/en/mcp
- [R21] Windows `cmd /c` for npx: https://github.com/anthropics/claude-code/issues/9594 , https://github.com/anthropics/claude-code/issues/20061
- [R22] Composio REST, custom MCP server with multiple apps: https://docs.composio.dev/reference/api-reference/mcp/postMcpServersCustom

---

## Implemented in P2.2

The code is in `spark/brain/mcp_config.py` and the config models are in `spark/config.py` (`ConnectorsConfig`).
The tests are in `tests/test_mcp_config.py`.

**`config.yaml` → `connectors:`**

| Key | Default | Meaning |
|---|---|---|
| `mcp_config` | `mcp_servers.json` | Server definitions, relative to the config folder. Keys starting with `_` are comments and are ignored. A `{"mcpServers": {...}}` wrapper is accepted |
| `block_claudeai_connectors` | `false` | `true` forces claude.ai connectors off. Otherwise `env_for_sdk()` sets `ENABLE_CLAUDEAI_MCP_SERVERS=true` (and the voice session goes non-strict) only when a `claude_ai_*` server is enabled (§4.7) |
| `claudeai_state_file` | `claudeai_connectors.json` | The claude.ai servers the CLI reported last. Unconfigured ones are hidden in the next session (app-managed) |
| `servers.<name>.enabled` | per server | `false` skips the server quietly. Defaults: all four off, because each needs your own credentials; turn one on after setting its variables in `.env`. The `claude_ai_*` servers (§4.7) default to on |
| `servers.<name>.allowed_tools` | see yaml | Read-only tools, pre-approved as exact `mcp__<name>__<tool>` names. Globs such as `GMAIL_FETCH_*` expand only against known tool names (the `known_tools=` argument, e.g. a live `tools/list`). A glob with no match is dropped with a warning. `*`, `mcp__<name>` and `mcp__<name>__*` are rejected at config load |
| `servers.<name>.side_effecting` | see yaml | Never pre-approved. They reach the PreToolUse confirmation hook (§8.3). Globs allowed |
| `servers.<name>.destructive` | see yaml | Added to `disallowed_tools` (globs pass through as deny rules, e.g. `mcp__composio__GMAIL_DELETE_*`) unless `allow_destructive: true` |
| `servers.<name>.allow_destructive` | `false` | When `true`, destructive tools are no longer denied, but they are still confirmed, never pre-approved |

A server has to appear in both `config.yaml` and `mcp_servers.json`. A server listed in only one of them is not loaded, and a warning says why.
The per-server `allow_destructive` replaces the single `enabled_destructive` switch proposed in §3.2.

**Loader behavior (`load_mcp_servers(settings) -> McpSetup`):**
- Every `${VAR}` in a server entry must be set and non-empty, or the server is disabled. The reason names the variable, never its value. `${VAR:-default}` is optional.
- Placeholders are passed to the SDK **unexpanded**, because the CLI expands them (§4.4). The one exception is `command`: it is expanded and resolved in Python, since it is a path, not a secret.
  - A bare name goes through `shutil.which`.
  - A relative path resolves against the config folder, with `.exe` tried on Windows.
  - A `.cmd`/`.bat` shim such as `npx` becomes `cmd /c npx ...` on Windows.
  - A missing binary disables the server with a fix hint (e.g. "set SPARK_GITHUB_MCP_BIN in .env ...").
- `McpSetup` has these fields: `servers`, `allowed_tools`, `disallowed_tools`, `side_effecting` (full names), `disabled` (name → reason), `warnings`, `required_env` (for doctor), and `policy`.
- `policy.classify(name)` returns `read`, `side_effect`, `destructive` or `unknown`. Precedence is destructive, then side_effect, then read. Tools on no list, tools of unloaded servers, and non-MCP tools are all `unknown`. The confirmation hook must deny `unknown` by default.
- Other policy helpers: `is_allowed` (pre-approved read), `is_side_effecting`, `is_destructive`, `needs_confirmation`, and `is_blocked`.
- `McpSetup.add_sdk_server(name, server, read_only=[...], side_effecting=[...], destructive=[...])` merges the in-process `spark` server (P2.3/P2.4). It takes exact tool names only.
- `mcp_servers.json` starts the GitHub binary with `--toolsets context,repos,issues,pull_requests`. Phase 2 also passed `--read-only`; 0.1.0 dropped it so `issue_write` and `add_issue_comment` work after a spoken yes (`side_effecting`), while every tool on no list stays denied. Add `--read-only` back to strip all write tools at the server. `mcp_servers.example.json` is the annotated copy.

## Wired in P2.4
- `ClaudeAgentBrain` loads `load_mcp_servers(settings)`, or takes an injected `mcp=`. When the app passes `timers=`, it adds the in-process `spark` server.
- It passes `mcp_servers`, the exact read-only `allowed_tools`, `disallowed_tools`, `env=env_for_sdk(settings)` and a deny-by-default `PreToolUse` gate (`spark/brain/tool_gate.py`). The SDK facts are in docs/sdk_notes.md, P2.4 section.
- In Phase 2, side-effecting, destructive and unknown tools are **denied**, and the model tells the user it can't do that yet.
- With `-v`, the log shows `MCP servers enabled: ...` and `MCP servers disabled: ...`, plus `tool call: <name>` and `tool denied: <name> (<class>)` per turn. Only names are logged.


## Security notes (P3.5)
- **Timers are pre-approved.** `mcp__spark__set_timer`/`cancel_timer` create and cancel only Spark's own in-memory timers: nothing leaves the process, and a wrong timer is harmless and easy to undo by voice. So they sit in `allowed_tools`. Spec §7's confirmation rule is applied to tools with external effects (send, create, update, delete). `mcp__spark__start_project` is side-effecting and always confirmed.
- **Environment inheritance.** The SDK starts the Claude CLI with Spark's environment (it can only *add* variables, via `ClaudeAgentOptions.env`), and the CLI passes its environment on to the stdio MCP servers it spawns. So every stdio server can see all of Spark's `.env` variables. Mitigations: keep `.env` minimal; prefer the official `github-mcp-server` binary over third-party `npx` packages; use http servers (which get only their configured headers) for aggregators. Processes Spark starts itself (`gh`, `git`, the Claude Code window) get a scrubbed environment (`spark.util.logging.child_env`).
- **MCP URLs are treated as secrets** (`*_MCP_URL`), because Composio URLs embed server and user ids. The doctor also quiets httpx's URL logging during probes.
