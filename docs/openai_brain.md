# The OpenAI-compatible brain (P5.7e)

Spark can run its whole voice brain on any endpoint that speaks the OpenAI **Chat Completions**
API with tool calling: OpenAI, Azure OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio and
vLLM. Claude stays the default (`brain.backend: claude_agent`) and is unchanged.

## From the app (no files to edit)

Click the **brain** button in the main window's header (next to Shortcuts and Connectors). The
Brain window opens:

1. **Pick a provider** on the left: Claude, OpenAI, Azure OpenAI, OpenRouter, Groq, Together,
   Ollama (local), LM Studio (local) or Custom. Each one says in a line what it needs, with a
   link to the page where you get a key or download the app.
   - **Claude** needs Claude Code installed on the computer, because the Claude brain runs it.
     It uses the account Claude Code is signed in with (run `claude`, then `/login`); the window
     shows whether it is signed in. An Anthropic API key (`ANTHROPIC_API_KEY`) is optional and
     switches to pay-per-use billing. The app doesn't ship Claude
     Code: if it isn't found (on `PATH`, in `~/.local/bin`, or at `brain.cli_path`), the window
     says so with an *Install Claude Code* link.
   - If you enabled claude.ai connectors, every other provider shows a note that they don't work
     with it. Connectors you added in the Connectors window keep working.
2. **Enter the key** (Claude and the cloud providers). It is saved in the `.env` file next to
   `config.yaml` under the provider's usual name (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...)
   and never shown again: the window
   only says "a key is saved, ending in ...abcd". *Replace key* and *Remove key* are next to it.
   A key set in your Windows environment variables is shown too, but has to be removed there.
3. **Check the address.** Filled in for you; you can change it for Azure, the local servers and
   Custom. Azure also has an *API version* field, only needed for the older
   `.../openai/deployments/{model}` addresses. On Azure the model names are your deployment
   names.
4. **Test connection.** For Claude, Spark checks that Claude Code is installed and that Anthropic
   accepts the key (`GET /v1/models`). For the others, it connects, checks the key, lists the
   models (`GET /models`), then
   asks the voice model a question that needs a tool. You get "this model can use tools", or a
   warning that Spark's connectors and timers won't work well with it. Problems are explained
   in plain words: a refused key (401/403), a wrong address (404), too many requests (429),
   "Ollama isn't running. Start it from the Start menu.", or no answer in time.
5. **Pick the models**: Voice (deep), Fast and Researcher. After a test the lists hold what the
   provider offers; *Other* lets you type a name it doesn't list.
6. **Save & apply** (the button reads *Use OpenAI* and so on when you switch). Spark rebuilds its
   brain without restarting the app; the main window then shows the provider, for example
   "OpenAI · gpt-4.1". If you didn't test since your last change, the window asks
   "Save anyway?". If the new brain doesn't start, you see why and can click *Go back to the
   previous settings*. *Switch back to Claude* is always one click.

**Ollama (local, free).** If Ollama isn't installed, the window links to
https://ollama.com/download. If it is installed but not running, it says to start it from the
Start menu. If it runs without models, it offers three that are good with tools, with their
download sizes and the free space where Ollama keeps models (it warns when there is less than
twice the size). Nothing downloads until you click **Download**; you can cancel, and Ollama
resumes where it stopped next time.

Where it is kept: the provider, address and model names go to `prefs.json` (`brain_backend`,
`openai_provider`, `openai_base_url`, `openai_api_version`, `openai_fast_model`, ...), which
overrides `config.yaml`; the key goes to `.env`. `config.yaml` is never changed and stays the
default when nothing was chosen in the window. A `--brain` flag on the command line beats the
window's choice, and `spark doctor` checks the brain the window chose.

## By hand in config.yaml

1. Pick a provider in `config.yaml`:

   ```yaml
   brain:
     backend: openai_compat
     openai_compat:
       provider: openai          # or azure | openrouter | groq | together | ollama | lmstudio | hermes | vllm | custom
       fast_model: null          # null = the preset's suggestion
       deep_model: null
   ```

2. Put the key in `.env` next to `config.yaml` (never in `config.yaml`), using the variable the
   preset names, for example `OPENAI_API_KEY=sk-...`. Local servers need no key.
3. Check it: `uv run python -m spark doctor`. It resolves the preset, checks that the key variable
   is set (it never prints the value), calls `GET /models`, and checks that your models are listed.
4. Start Spark as usual (`uv run python -m spark --text` to try it typed).

To go back, set `backend: claude_agent`. The model choices you make in the window are saved per
backend in `prefs.json` (`openai_fast_model`, ...), so neither backend's ids leak into the other.

### Examples

OpenAI:
```yaml
    openai_compat: {provider: openai, fast_model: gpt-4.1-mini, deep_model: gpt-4.1}
```
`.env`: `OPENAI_API_KEY=sk-...`

Ollama (local, free; needs a model with tool calling, about 5 GB for the 7B one):
```bash
ollama pull qwen2.5:7b
```
```yaml
    openai_compat: {provider: ollama, fast_model: qwen2.5:7b, deep_model: qwen2.5:7b}
```

Azure OpenAI (the model names are your **deployment** names):
```yaml
    openai_compat:
      provider: azure
      base_url: https://<resource>.openai.azure.com/openai/v1
      fast_model: my-gpt-4-1-mini-deployment
      deep_model: my-gpt-4-1-deployment
```
`.env`: `AZURE_OPENAI_API_KEY=...`. For a classic per-deployment endpoint use
`base_url: https://<resource>.openai.azure.com/openai/deployments/{model}` plus
`api_version: 2024-10-21`; `{model}` is replaced by the deployment name.

## Presets

Model names are **suggestions**. Providers rename and retire models often, and on local servers
the right name is whatever you pulled or loaded. Every field you set overrides the preset.

| provider | base URL | key variable | auth | suggested fast / deep |
|---|---|---|---|---|
| `openai` | https://api.openai.com/v1 | `OPENAI_API_KEY` | Bearer | gpt-4.1-mini / gpt-4.1 |
| `azure` | (required) `https://<resource>.openai.azure.com/openai/v1` | `AZURE_OPENAI_API_KEY` | `api-key` header, optional `?api-version=` | your deployments |
| `openrouter` | https://openrouter.ai/api/v1 | `OPENROUTER_API_KEY` | Bearer | openai/gpt-4.1-mini / openai/gpt-4.1 |
| `groq` | https://api.groq.com/openai/v1 | `GROQ_API_KEY` | Bearer | llama-3.1-8b-instant / llama-3.3-70b-versatile |
| `together` | https://api.together.xyz/v1 | `TOGETHER_API_KEY` | Bearer | Meta-Llama-3.1-8B-Instruct-Turbo / Llama-3.3-70B-Instruct-Turbo |
| `ollama` | http://localhost:11434/v1 | none | none | qwen2.5:7b / qwen2.5:14b |
| `lmstudio` | http://localhost:1234/v1 | none | none | the loaded model (set it) |
| `hermes` | http://127.0.0.1:8642/v1 | `HERMES_API_KEY` (Hermes's `API_SERVER_KEY`) | Bearer | hermes-agent / hermes-agent |
| `vllm` | http://localhost:8000/v1 | `VLLM_API_KEY` (only with `--api-key`) | Bearer | the served model (set it) |
| `custom` | (required) your server's address | `OPENAI_COMPAT_API_KEY` (optional) | Bearer if a key is saved | the served model (set it) |

Other settings under `brain.openai_compat`: `api_key_env` (use another variable), `extra_headers`
(values may use `${VAR}` from `.env`), `extra_body` (extra request fields, for example
`{reasoning_effort: minimal}` for an OpenAI reasoning model), `timeout_s`, `max_tokens`,
`temperature`, `parallel_tool_calls`, `max_tool_result_chars`, `worker_model` (the researcher).

## What is the same as on Claude

The same code runs both backends (`spark/brain/core.py`): the voice prompt and briefing, connector
notes, fast/deep routing and the Auto / Fast / Deep mode, voice model switching ("use the deep
model", "which model are you using?"), the step budget, the "On it, give me a moment."
acknowledgement, idle reset and "new topic", barge-in, the activity log, connector health in the
Connectors window and the briefing, and the **confirmation gate**: read-only tools run,
side-effecting tools need a spoken yes, destructive and unknown tools are refused. Only tools
the policy allows are offered to the model at all.

## What differs

- **MCP servers are run by Spark itself** (the `mcp` Python client) instead of the Claude CLI,
  with `${VAR}` values expanded in Spark's process. Each server keeps one session across turns
  and is retried after a failure.
- **No claude.ai account connectors** (claude.ai Gmail, Google Calendar, ...). They come from the
  Claude CLI's login. Enabled ones are skipped with a warning, shown as unavailable in the
  Connectors window, and the model is told they are unreachable. The doctor flags them.
- **The researcher is a nested tool loop**, not an Agent SDK sub-agent. The voice model calls a
  `research` function; the researcher runs on `worker_model` with only the read-only connector
  tools and the connector-notes tools, for at most `worker_max_turns` rounds, and reports back.
- **Tool-calling quality depends on the model.** Small local models often call tools with wrong
  arguments or ignore them. Recommended: OpenAI gpt-4.1 / gpt-4.1-mini, or another frontier
  model through OpenRouter; locally Qwen 2.5 (7B or larger) or Llama 3.1 8B+. A model without
  tool support fails with "The selected model can't use tools".
- Errors are spoken in plain words: a refused key ("check the key in the .env file"), an unknown
  model, rate limits (retried twice first), and "I can't reach Ollama on this computer. Is it
  running?" for a local server that is down.
- The optional router classifier (`routing.classifier: fast_model`) and the addressee check use
  the same provider's fast model with tool-less requests.

## Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research) is an agent with its
own model, tools, memory and skills. Its API server speaks the OpenAI API, so Spark can be its
voice:

1. Turn on Hermes's API server. In Hermes's `.env` (`~/.hermes/.env`, or
   `%LOCALAPPDATA%\hermes\.env` on Windows) set `API_SERVER_ENABLED=true` and
   `API_SERVER_KEY=` a long random value (at least 16 characters). See
   [Hermes's API server guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server).
2. Start it with `hermes gateway`. It listens on http://127.0.0.1:8642.

   **Hermes on a server:** also set `API_SERVER_HOST=0.0.0.0` there, and in Spark's Brain window
   change the address to the server's, for example `https://hermes.example.com/v1`. Put it
   behind https (a reverse proxy) or keep it on a private network such as Tailscale: the key
   is sent with every request.
3. In Spark's **Brain** window pick **Hermes Agent** (under "Recommended"), paste the same
   `API_SERVER_KEY`, press **Test connection**, then **Save**. Spark keeps it in `.env` as
   `HERMES_API_KEY`. The model is `hermes-agent` (or your Hermes profile's name).

Or in `config.yaml`:

```yaml
brain:
  backend: openai_compat
  openai_compat:
    provider: hermes
```

What to expect:

- **Hermes runs its own tools.** It ignores the tools Spark offers, so Spark's connectors,
  timers and spoken confirmation don't apply to its turns; Hermes's own approval settings do.
  The connection test only checks the address, the key and the model list, because a test
  question would start a whole agent turn.
- **Answers can take longer** when Hermes uses its tools. Hermes keeps the stream alive while
  it works, so Spark waits.
- Spark still sends its voice prompt, so answers stay short and speakable.

## Speed on a local model

A question you type into `ollama run` is a few dozen tokens. Every Spark request also carries the
voice prompt and the tool list, about 3,000 tokens, and a model running on a CPU or a small GPU
needs 10–30 s just to read that. Measured on a laptop CPU with a 7.5B model: 0.6 s to read a
bare question, 15 s to read Spark's prompt and tools.

Ollama, LM Studio and llama.cpp keep what they read (a prompt cache) and reuse it for the next
request with the same start, so only the first question of a conversation pays the full cost.
Spark takes care of that:

- **Priming.** For a local server, Spark reads the prompt and tools into the model in the
  background when it starts and after each new conversation (`brain.prime_on_connect`, on by
  default). In the measurement above, the first question then took 4 s instead of 30–40 s.
- **Keeping the model loaded.** Ollama unloads a model after 5 idle minutes. Spark asks it to keep
  the model for 30 minutes. Change it with `extra_body: {keep_alive: 2h}` (or `-1` for always).

What else helps:

- **Use one model for Fast and Deep.** Two models mean two caches, and on a machine without much
  memory, switching between them reloads each one.
- **Prefer models that don't "think" first.** Reasoning models (DeepSeek-R1, Qwen3 in thinking
  mode, ...) write a hidden chain of thought before the answer. In a terminal you watch it stream;
  in Spark you hear nothing until it ends. Pick a non-reasoning model, or try
  `extra_body: {reasoning_effort: none}`.
- **A smaller model, or a GPU.** On Apple Silicon, Ollama uses the GPU automatically; a 7–8B model
  is much faster than a 27–32B one.
