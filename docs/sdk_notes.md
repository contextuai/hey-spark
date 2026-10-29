# SDK / library ground truth (verified 2026-09-23)

Verified against **installed source** in an isolated scratch venv (Python 3.11, `uv pip install`).
Citations are `file:line` relative to `site-packages/`. Anything not read in source is marked
**UNVERIFIED**.

| Package | Version verified |
|---|---|
| claude-agent-sdk | **0.2.159** (bundles Claude Code CLI **2.1.281**, `claude_agent_sdk/_cli_version.py:3`) |
| faster-whisper | **1.2.1** (ctranslate2 4.8.2, onnxruntime 1.30.0, av 18.1.0) |
| pvporcupine | **4.0.3** (native lib reports `4.0.0`) |
| openwakeword | **0.6.0** |
| piper-tts | **1.8.0** |

---

## 1. claude-agent-sdk 0.2.159

### ClaudeSDKClient (`claude_agent_sdk/client.py`)
- `ClaudeSDKClient(options: ClaudeAgentOptions | None = None, transport: Transport | None = None)` at `client.py:67`.
- `async connect(prompt: str | AsyncIterable[dict] | None = None)` at `client.py:84`. With `None` it opens an empty stream and stays connected (streaming mode).
- `async query(prompt: str | AsyncIterable[dict], session_id: str = "default")` at `client.py:262`. It only **writes** the user message. It does not wait.
- `async receive_messages() -> AsyncIterator[Message]` at `client.py:250`. It runs forever across turns.
- `async receive_response() -> AsyncIterator[Message]` at `client.py:552`. It yields up to and **including** the `ResultMessage`, then stops (`client.py:588-591`).
- `async interrupt()` **exists** at `client.py:298`. It sends the control request `{"subtype": "interrupt"}` (`_internal/query.py:709`).
- `async set_model(model: str | None = None)` **exists** at `client.py:331`. It sends the `set_model` control request (`_internal/query.py:722`). `None` means the default model.
- `async set_permission_mode(mode)` is at `client.py:304`. Other methods: `get_mcp_status`, `reconnect_mcp_server`, `toggle_mcp_server`, `get_context_usage`, `get_server_info`, `stop_task`, `rewind_files`.
- `async disconnect()` is at `client.py:593`. `__aenter__` calls `connect()` (`client.py:609`), and `__aexit__` calls `disconnect()` (`client.py:614`).
- Caveat in the docstring (`client.py:58-64`): the client holds an anyio task group from `connect()` until `disconnect()`, so **all use must happen in the same async context/task group where it was connected**.
- There is **no `reset()` / clear method**. `ConversationResetMessage` (`types.py:1438`) is emitted after `/clear`. Sending `"/clear"` via `query()` to trigger it is **UNVERIFIED**.

### ClaudeAgentOptions (`claude_agent_sdk/types.py:1962`, dataclass)
| Field | Type / default | Line | Maps to CLI |
|---|---|---|---|
| `tools` | `list[str] \| ToolsPreset \| None = None` | 1965 | `[]` gives `--tools ""` (**disables all built-in tools**), `subprocess_cli.py:589-597` |
| `allowed_tools` | `list[str] = []` | 1976 | `--allowedTools a,b`. These are auto-approved tools, **not** an availability filter |
| `system_prompt` | `str \| SystemPromptPreset \| SystemPromptCustom \| SystemPromptFile \| None = None` | 1988 | `None` gives `--system-prompt ""` (empty, **not** the Claude Code prompt) `subprocess_cli.py:572-573` |
| `mcp_servers` | `dict[str, McpServerConfig] \| str \| Path = {}` | 2002 | `--mcp-config '{"mcpServers": {...}}'` (the SDK strips `instance` from sdk servers) |
| `strict_mcp_config` | `bool = False` | 2009 | `--strict-mcp-config` (ignore other MCP configs) |
| `permission_mode` | `Literal["default","acceptEdits","plan","bypassPermissions","dontAsk","auto"] \| None` | 2016 (Literal at 25) | `--permission-mode` |
| `max_turns` | `int \| None` | 2040 | `--max-turns` |
| `max_budget_usd` | `float \| None` | 2046 | |
| `disallowed_tools` | `list[str] = []` | 2053 | `--disallowedTools` (removed from context) |
| `model` | `str \| None` | 2060 | `--model` |
| `fallback_model` | `str \| None` | 2066 | |
| `cwd` | `str \| Path \| None` | 2086 | defaults to the process cwd |
| `cli_path` | `str \| Path \| None` | 2089 | **exact name `cli_path`** |
| `env` | `dict[str,str] = {}` | 2109 | merged **over** `os.environ` (minus `CLAUDECODE`), `subprocess_cli.py:815-820` |
| `stderr` | `Callable[[str], None] \| None` | 2129 | CLI stderr lines callback |
| `can_use_tool` | `CanUseTool \| None` | 2135 | forces `permission_prompt_tool_name="stdio"` (`types.py:1917-1940`) |
| `hooks` | `dict[HookEvent, list[HookMatcher]] \| None` | 2153 | sent in the initialize control request, not on argv |
| `include_partial_messages` | `bool = False` | 2169 | `--include-partial-messages` |
| `setting_sources` | `list[Literal["user","project","local"]] \| None = None` | 2273 | **Default None loads ALL sources** (user, project, local, and CLAUDE.md). `[]` gives `--setting-sources=` (isolation), `subprocess_cli.py:725-726` |
| `thinking` / `effort` | ThinkingConfig / `Literal["low","medium","high","xhigh","max"]` | 2336 / 2349 | |

**No built-in Claude Code tools (Bash, Read, and so on):** use `tools=[]`. The docstring at `types.py:1968` says the empty list disables all built-in tools. An offline smoke test confirmed the argv contained `'--tools', ''`. MCP tools (`mcp__server__tool`) are separate and stay available. Belt and braces: also set `setting_sources=[]` and `strict_mcp_config=True`.

### CLI discovery (`_internal/transport/subprocess_cli.py`)
- `_find_cli()` at `:251` checks, in order: **the bundled `_bundled/claude.exe` first** (`:254`, `_find_bundled_cli` at `:337`), then `shutil.which("claude")` (on Windows only if it is a native exe), then `shutil.which("claude.exe")`, then `~/.local/bin/claude.exe`.
- The Windows wheel **bundles `claude.exe` (~240 MB, CLI 2.1.281)**. So `cli_path` is normally unnecessary. The bundled CLI wins over PATH.
- The frozen app leaves `_bundled/` out (`packaging/hey_spark.spec`, re-checked 2026-10-23 against 0.2.159 and 0.2.174). There `_find_bundled_cli()` returns None and `_find_cli()` falls through to PATH and `~/.local/bin/claude.exe`, so the user's installed Claude Code is used. A missing CLI raises `CLINotFoundError` at `connect()`, which the brain turns into an "install Claude Code" fix. `spark.doctor.find_claude_cli` mirrors this order for the doctor and the Brain window.
- `.cmd`/`.bat` shims (the npm install) are **refused** with `CLIConnectionError` (`_reject_windows_batch_cli`, `:411`).
- On this machine, `claude --version` gives `2.1.281 (Claude Code)` at `~/.local/bin/claude.exe` (native exe, so it would be accepted).

### Message types (`types.py`)
- `Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent | RateLimitEvent | ConversationResetMessage` (`:1498`).
- `StreamEvent(uuid: str, session_id: str, event: dict[str, Any], parent_tool_use_id: str | None = None)` (`:1381`). `event` is the **raw Anthropic API stream event** (parsed at `_internal/message_parser.py:343`). These events are emitted only when `include_partial_messages=True`.
  - Text delta shape, from the Anthropic Messages streaming API: `{"type": "content_block_delta", "index": n, "delta": {"type": "text_delta", "text": "..."}}`.
  - The strings `content_block_delta` and `text_delta` are present in the bundled CLI. The exact dict shape comes from the API spec, not from SDK source.
- `AssistantMessage(content: list[ContentBlock], model: str, parent_tool_use_id=None, error=None, usage=None, message_id=None, stop_reason=None, session_id=None, uuid=None)` (`:1140`).
- `TextBlock(text)` `:956`. `ThinkingBlock(thinking, signature)` `:963`. `ToolUseBlock(id, name, input: dict)` `:971`. `ToolResultBlock(tool_use_id, content=None, is_error=None)` `:980`.
- `SystemMessage(subtype: str, data: dict)` `:1155`. Subclasses include TaskStarted/Progress/Notification and HookEventMessage.
- `ResultMessage` `:1340` has these fields:
  - Required: `subtype: str`, `duration_ms: int`, `duration_api_ms: int`, `is_error: bool`, `num_turns: int`, `session_id: str`.
  - Optional, all defaulting to `None`: `stop_reason`, `total_cost_usd: float|None`, `usage: dict|None`, `result: str|None`, `structured_output`, `model_usage`, `permission_denials`, `deferred_tool_use`, `errors`, `api_error_status: int|None`, `uuid`, `terminal_reason: str|None`, `origin`.
  - `terminal_reason` is `"aborted_streaming"` or `"aborted_tools"` after `interrupt()` (`:1363-1371`).

### MCP server configs (`types.py:627-672`)
The TypedDicts below are **not exported at top level**. Import them from `claude_agent_sdk.types`.
```
McpStdioServerConfig  :627  type: NotRequired["stdio"]; command: str; args: NotRequired[list[str]]; env: NotRequired[dict[str,str]]
McpSSEServerConfig    :636  type: "sse";  url: str; headers: NotRequired[dict[str,str]]
McpHttpServerConfig   :644  type: "http"; url: str; headers: NotRequired[dict[str,str]]
McpSdkServerConfig    :652  type: "sdk";  name: str; instance: mcp.server.Server
```
- `tool(name: str, description: str, input_schema: type | dict, annotations: ToolAnnotations | None = None)` is a decorator that produces `SdkMcpTool` (`__init__.py:251`).
  - The handler must be `async def h(args: dict) -> dict`. It returns `{"content": [{"type": "text", "text": "..."}]}`, plus `"is_error": True` for errors.
  - Content types passed through are `text` and `image` (`data`, `mimeType`). `resource_link` and text `resource` are flattened to text. Anything else is dropped (`__init__.py:451-488`).
  - An exception in the handler is reported as an error result.
- `create_sdk_mcp_server(name: str, version: str = "1.0.0", tools: list[SdkMcpTool] | None = None) -> McpSdkServerConfig` (`__init__.py:491`). The smoke test returned keys `type`, `name`, `instance`.
- The model sees tool names as **`mcp__<serverKey>__<toolName>`**. Use this form in `allowed_tools` and hook matchers. The CLI name template was found in the bundled CLI binary. The docstring example `allowed_tools=["add"]` at `__init__.py:537` is misleading.

### Hooks and permissions (`types.py`)
- `HookMatcher(matcher: str | None = None, hooks: list[HookCallback] = [], timeout: float | None = None)` (`:610`). The timeout is in seconds and **defaults to 60** (`:622`). The matcher is a tool name or pattern such as `"Write|Edit"` or `"mcp__spark__.*"`.
- `HookCallback = Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]` (`:598`). It is invoked as `callback(input, tool_use_id, {"signal": None})` (`_internal/query.py:564-568`).
- `PreToolUseHookInput` (`:333`) has keys `session_id`, `transcript_path`, `cwd`, `permission_mode?`, `hook_event_name="PreToolUse"`, `tool_name`, `tool_input: dict`, `tool_use_id`, plus optional `agent_id` and `agent_type`.
- **Deny shape** (`PreToolUseHookSpecificOutput` `:437`, `SyncHookJSONOutput` `:541`). `permissionDecision` is one of allow, deny, ask, or defer:
  ```python
  {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                          "permissionDecision": "deny",
                          "permissionDecisionReason": "User declined."}}
  ```
  - Optional top-level keys are `continue_` and `async_`, `suppressOutput`, `stopReason`, `decision: "block"`, `systemMessage`, and `reason`.
  - `continue_` and `async_` are renamed to `continue` and `async` at `_internal/query.py:83-99`.
  - `hookSpecificOutput` may also carry `updatedInput` and `additionalContext`.
- Matchers on the same event run **concurrently** (`types.py:2155-2163`).
- `can_use_tool: CanUseTool = Callable[[str, dict, ToolPermissionContext], Awaitable[PermissionResultAllow | PermissionResultDeny]]` (`:278`).
  - `PermissionResultAllow(behavior="allow", updated_input=None, updated_permissions=None)` is at `:259`.
  - `PermissionResultDeny(behavior="deny", message="", interrupt=False)` is at `:268`.
  - `ToolPermissionContext` (`:223`) has `signal`, `suggestions`, `tool_use_id`, `agent_id`, `blocked_path`, `decision_reason`, `title`, `display_name`, `description`.
  - It is **only called when the CLI rules evaluate to "ask"**. It is skipped for `allowed_tools` entries, for `bypassPermissions`, and when a PreToolUse hook returned allow (`types.py:2135-2151`).
  - It cannot be combined with `permission_prompt_tool_name` (ValueError, `types.py:1934`).

### Errors (`claude_agent_sdk/_errors.py`)
All of these are exported at top level except `MessageParseError`.
- `ClaudeSDKError(Exception)` `:6`.
- `CLIConnectionError(ClaudeSDKError)` `:10`.
- `CLINotFoundError(CLIConnectionError)` `:14`.
- `ProcessError(ClaudeSDKError)` `:25`, with `.exit_code` and `.stderr`.
- `ResultError(ProcessError)` `:56`, raised when the CLI exits non-zero after an is_error result. It carries `.subtype`, `.errors`, `.result`, `.api_error_status`, `.terminal_reason`, `.session_id`, and `.data`.
- `CLIJSONDecodeError` `:120`.
- `MessageParseError` `:129`.

### Implications for Spark
- Brain options should be:
  - `tools=[]`, `setting_sources=[]`, `strict_mcp_config=True`
  - `system_prompt=<Spark persona str>`, `include_partial_messages=True`
  - `mcp_servers={...}`, `allowed_tools=[read-only mcp__ names]`
  - `hooks={"PreToolUse": [HookMatcher(matcher=<side-effecting pattern>, hooks=[confirm], timeout=<above 60 if spoken confirmation can take longer>)]}`
  - a small `max_turns`

  Without `setting_sources=[]` the brain would inherit the user settings, hooks, and CLAUDE.md files.
- Leave `cli_path` unset by default; the bundled exe is used. Expose `cli_path` in config only as an override. Catch `CLINotFoundError` or `CLIConnectionError` in `spark doctor`.
- Stream TTS from `StreamEvent.event` where `event["type"] == "content_block_delta"` and `event["delta"]["type"] == "text_delta"`. Ignore events whose `parent_tool_use_id` is not None (subagent). Fall back to `AssistantMessage`/`TextBlock` when partials are off.
- Barge-in: call `await client.interrupt()`, then keep draining `receive_response()` until the `ResultMessage` (terminal_reason `aborted_*`) before the next `query()`.
- Model routing per query: call `await client.set_model("...")` before `query()`. No reconnect is needed. Verified live in P3.2 (see "Routing (P3.2)").
- To start a new conversation, `disconnect()`, then create a new client and `connect()`. A direct reset API does not exist.
- Keep the whole client lifecycle (connect, query, receive, disconnect) inside **one long-lived asyncio task**, and feed it through an `asyncio.Queue`.
- In the confirmation hook, return deny with a `permissionDecisionReason` the model can speak. Default to deny on timeout or exception.
- `env` is merged over `os.environ`, so secrets already in the env are inherited by the CLI anyway.

---

## 2. faster-whisper 1.2.1

### Silero VAD (`faster_whisper/vad.py`, `faster_whisper/utils.py`)
- Model file: `os.path.join(faster_whisper.utils.get_assets_path(), "silero_vad_v6.onnx")`. `get_assets_path()` is at `utils.py:39` and returns `<pkg>/assets`. The same path is used by `get_vad_model()` at `vad.py:288-292` (lru_cached).
- `SileroVADModel(path)` is at `vad.py:295`. It creates an `onnxruntime.InferenceSession` with 1 inter-op thread, 1 intra-op thread, and the CPU provider, exposed as `.session` (`vad.py:304-314`).
- `SileroVADModel.__call__(audio: np.ndarray, num_samples=512, context_size_samples=64)` (`vad.py:316`) is a **batch/offline API**. **It resets h and c to zeros on every call** (`vad.py:324-325`), so calling it per 512-sample frame loses LSTM state. For streaming, call `.session.run` directly and carry the state.
- ONNX I/O (inspected at runtime):
  - Inputs: `input` float32 `[seq_len, 576]` (64 context + 512 new samples), `h` float32 `[1,1,128]`, and `c` float32 `[1,1,128]`.
  - Outputs: `speech_probs` `[seq_len]`, `hn` `[1,1,128]`, and `cn` `[1,1,128]`.
  - There is **no sr input**; the model is 16 kHz only.
- Input must be float32 in [-1, 1]. Convert with `int16 / 32768.0`.
- **Verified snippet** (run in the scratch venv). Streaming matched the batch `SileroVADModel` output to within 0.02.
  ```python
  import os, numpy as np
  from faster_whisper.utils import get_assets_path
  from faster_whisper.vad import SileroVADModel
  sess = SileroVADModel(os.path.join(get_assets_path(), "silero_vad_v6.onnx")).session

  class StreamingSilero:
      def __init__(self, s):
          self.s = s
          self.reset()

      def reset(self):
          self.h = np.zeros((1, 1, 128), np.float32)
          self.c = np.zeros((1, 1, 128), np.float32)
          self.ctx = np.zeros((1, 64), np.float32)

      def __call__(self, frame512_f32) -> float:
          x = np.asarray(frame512_f32, np.float32).reshape(1, 512)
          inp = np.concatenate([self.ctx, x], 1)
          out, self.h, self.c = self.s.run(None, {"input": inp, "h": self.h, "c": self.c})
          self.ctx = x[:, -64:]
          return float(np.asarray(out).reshape(-1)[0])
  ```
  Results over 1 s (31 frames):

  | Signal | Mean probability | Max probability |
  |---|---|---|
  | silence | 0.003 | 0.024 |
  | white noise (0.1 rms) | 0.030 | 0.065 |
  | synthetic voiced buzz | 0.234 | 0.777 |
- `get_speech_timestamps(audio, vad_options=None, sampling_rate=16000, **kwargs)` (`vad.py:45`) is the offline alternative, with a window of 512 (`vad.py:70`).

### WhisperModel (`faster_whisper/transcribe.py`)
- `WhisperModel(model_size_or_path, device="auto", device_index=0, compute_type="default", cpu_threads=0, num_workers=1, download_root=None, local_files_only=False, files=None, revision=None, use_auth_token=None, **model_kwargs)` is at `transcribe.py:621`. `cpu_threads=0` means the CTranslate2 default of 4 (docstring `:652`). `download_root` is passed as the HF cache_dir (`:684`).
- `transcribe(audio: str | BinaryIO | np.ndarray, language=None, task="transcribe", beam_size=5, best_of=5, ..., condition_on_previous_text=True, initial_prompt=None, ..., without_timestamps=False, word_timestamps=False, vad_filter=False, vad_parameters=None, chunk_length=None, hotwords=None, ...)` is at `transcribe.py:747`.
- It returns a **tuple (segments, info)** (`:791`, `:1022`). `segments` is a **lazy generator** (`generate_segments` at `:1103` yields), and decoding happens while you iterate. When `language=None`, language detection runs eagerly inside `transcribe()`.
- A `np.ndarray` input is used as-is (`:875`). It must be **float32 mono 16 kHz in [-1, 1]**; other inputs go through `decode_audio` (PyAV).
- `Segment` fields: id, seek, start, end, text, tokens, avg_logprob, compression_ratio, no_speech_prob, words, temperature.
- `TranscriptionInfo` fields: language, language_probability, duration, duration_after_vad, all_language_probs, transcription_options, vad_options.

### Implications for Spark
- VAD backend: wrap the snippet above. Feed 512-sample 16 kHz frames (int16 to float32 /32768). Call `reset()` at the start of each utterance. The model is about 2 MB and ships with the wheel, so it needs no download and no torch.
- STT:
  - Construct `WhisperModel(size, device="cpu", compute_type="int8", download_root=str(models_dir / "whisper"), cpu_threads=N)`.
  - Call `segs, info = model.transcribe(audio_f32, language="en", beam_size=1..5, vad_filter=False, without_timestamps=True, condition_on_previous_text=False)`.
  - Join the text with `"".join(s.text for s in segs).strip()`.
  - Materialize the generator **inside** `asyncio.to_thread`.
- The Windows console is cp1252. Printing faster-whisper defaults, or Piper phonemes, raised UnicodeEncodeError in the scratch runs. Set `PYTHONIOENCODING=utf-8` or `sys.stdout.reconfigure(encoding="utf-8")` in the entry point, and do not print raw phonemes.

---

## 3. pvporcupine 4.0.3
- `pvporcupine.create(access_key: str, library_path=None, model_path=None, device=None, keyword_paths=None, keywords=None, sensitivities=None) -> Porcupine` is at `pvporcupine/_factory.py:33`.
  - **New in 4.x:** a `device` argument that defaults to "best" (`:70-71`). Options are "cpu", "cpu:N", and "gpu[:i]".
  - Either `keywords` or `keyword_paths` is required, otherwise it raises ValueError (`:73-75`). `sensitivities` defaults to 0.5 each, and a length mismatch raises ValueError (`:84-88`).
- Built-in `pvporcupine.KEYWORDS` (`:28-30`) are alexa, americano, blueberry, bumblebee, computer, grapefruit, grasshopper, hey barista, hey google, hey siri, jarvis, ok google, pico clock, picovoice, porcupine, and terminator. A custom "Hey Spark" needs a `.ppn` from Picovoice Console, or `train_wake_word_from_phrase(access_key, output_path, language, phrase, platform=None)` (`:116`, new, needs network and key).
- `Porcupine.process(pcm: Sequence[int]) -> int` is at `_porcupine.py:238`. It returns the keyword index or -1, and raises ValueError if `len(pcm) != frame_length` (`:249-250`). It builds `(c_short * n)(*pcm)`. A numpy int16 array works (verified).
- `.frame_length` is at `:268` and `.sample_rate` at `:274`. Read via the C lib without a key: **512** and **16000**. `.version` is at `:262`. `delete()` is at `:233` (there is no `__del__`, so call it yourself).
- Exceptions (`_porcupine.py:19-83`) are `PorcupineError` (with `.message` and `.message_stack`) and its subclasses: `PorcupineMemoryError`, `PorcupineIOError`, `PorcupineInvalidArgumentError`, `PorcupineStopIterationError`, `PorcupineKeyError`, `PorcupineInvalidStateError`, `PorcupineRuntimeError`, `PorcupineActivationError`, `PorcupineActivationLimitError`, `PorcupineActivationThrottledError`, and `PorcupineActivationRefusedError`.
  - An empty access_key raises plain ValueError (`:151-152`).
  - A malformed key raised **PorcupineInvalidArgumentError** ("Failed to parse AccessKey"), verified. It is not raised as PorcupineActivationError.
- The Windows DLL ships at `pvporcupine/lib/windows/amd64/libpv_porcupine.dll`. Keyword files are at `resources/keyword_files/windows/<name>_windows.ppn`.

### Implications for Spark
- Frame size is 512 int16 samples at 16 kHz. Pass `device="cpu"` explicitly; there is no GPU, and it avoids probing the bundled CUDA DLL.
- `spark doctor` should map a missing key to ValueError, PorcupineInvalidArgumentError to a bad key, PorcupineActivation* to account or network issues, and PorcupineIOError to a bad path.
- Use the built-in "jarvis" or "computer" keyword as a placeholder until a "Hey Spark" `.ppn` exists. Creating it is a human task because it needs a Picovoice Console key.

---

## 4. openwakeword 0.6.0
- **The wheel ships NO model files.** There is no `resources/` dir after install. You must call `openwakeword.utils.download_models(model_names: list[str] = [], target_directory=<pkg>/resources/models)` (`utils.py:625`).
  - It always downloads melspectrogram and embedding (tflite + onnx) plus `silero_vad.onnx`, then the named models (tflite + onnx), from GitHub release v0.5.1 (`__init__.py:8-51`).
  - Result for `["hey_jarvis"]`: embedding_model.{onnx,tflite}, melspectrogram.{onnx,tflite}, hey_jarvis_v0.1.{onnx,tflite}, silero_vad.onnx.
- `Model(wakeword_models: list[str] = [], class_mapping_dicts=[], enable_speex_noise_suppression=False, vad_threshold=0, custom_verifier_models={}, custom_verifier_threshold=0.1, inference_framework="tflite", **kwargs)` is at `model.py:38`.
  - **The default is "tflite"**, and tflite-runtime is only a dependency on Linux, so on Windows you **must pass `inference_framework="onnx"`**.
  - `**kwargs` go to `AudioFeatures(melspec_model_path="", embedding_model_path="", sr=16000, ncpu=1, inference_framework="onnx", device="cpu")` (`model.py:213`, `utils.py:38`). This lets you point the feature models at models_dir (the defaults are package-relative, `utils.py:70-73`).
  - Entries in `wakeword_models` can be file paths or pretrained names (`model.py:90-100`). For a file path the key becomes the basename without extension, for example "hey_jarvis_v0.1". A pretrained name is resolved in the package dir and the key is the given name.
- `predict(x: np.ndarray, patience={}, threshold={}, debounce_time=0.0, timing=False) -> dict[str, float]` is at `model.py:232`.
  - `x` **must be int16**, otherwise it raises ValueError (`utils.py:194-197`). Ideally pass 1280 samples (80 ms); other lengths are buffered.
  - It returns `{model_key: np.float32 score in [0, 1]}`. Scores are **forced to 0.0 for the first 5 frames** (`model.py:330-333`).
  - `patience` and `debounce_time` require `threshold` (`:340-345`). `reset()` is at `model.py:226`.
- Verified: downloaded to scratch, `Model(wakeword_models=[<path>/hey_jarvis_v0.1.onnx], inference_framework="onnx", melspec_model_path=..., embedding_model_path=...)` gave keys `['hey_jarvis_v0.1']` and took about 6 ms per 1280-sample frame on this CPU.

### Implications for Spark
- The oww backend should call `download_models(["hey_jarvis"], target_directory=str(models_dir / "openwakeword"))` if files are missing. Then pass explicit onnx paths for the wakeword model and both feature models, and use `inference_framework="onnx"`. Threshold on `max(scores.values())`. Call `reset()` after each detection to avoid re-triggers.
- The original dependency list (`docs/dev/history/ralph/APPROVED_DEPS.md`) calls hey_jarvis "bundled". It is **not** bundled in 0.6.0 and needs a network download of about 10 MB.

---

## 5. piper-tts 1.8.0
- `from piper import PiperVoice, SynthesisConfig, AudioChunk` (`piper/__init__.py`).
- `PiperVoice.load(model_path, config_path=None, use_cuda=False, espeak_data_dir=ESPEAK_DATA_DIR, download_dir=None, include_alignments=False)` is at `voice.py:123`. `config_path` defaults to `model_path + ".json"` (`:144-146`). The espeak-ng data is bundled.
- `synthesize(text, syn_config: SynthesisConfig | None = None, include_alignments=False) -> Iterable[AudioChunk]` is at `voice.py:333`. It yields **one chunk per sentence** (`:433`). There is **no synthesize_stream_raw** in 1.8.0.
- `synthesize_wav(text, wav_file: wave.Wave_write, syn_config=None, set_wav_format=True, include_alignments=False)` is at `voice.py:444`.
- `AudioChunk` is at `voice.py:41`. Fields: `sample_rate`, `sample_width` (=2), `sample_channels` (=1), `audio_float_array` in [-1, 1], `phonemes`, and `phoneme_ids`. Properties: `.audio_int16_array` (`:78`) and `.audio_int16_bytes` (`:92`).
- `SynthesisConfig(speaker_id=None, length_scale=None, noise_scale=None, noise_w_scale=None, normalize_audio=True, volume=1.0)` is at `config.py:132`.
- Sample rate comes from `voice.config.sample_rate`, which is **22050** for en_US-lessac-medium (verified).
- Download: `piper.download_voices.download_voice(voice: str, download_dir: Path, force_redownload=False)` (`download_voices.py:76`). The CLI form is `python -m piper.download_voices en_US-lessac-medium --download-dir DIR`.
  - The URL template (`:11`) is `https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{quality}/{lang_code}-{voice_name}-{quality}.onnx`, plus the matching `.onnx.json`.
  - For lessac-medium that is `en/en_US/lessac/medium/en_US-lessac-medium.onnx`.
- **Voice downloaded for the project:** `.models/piper/en_US-lessac-medium.onnx` (63,201,294 bytes) and `.models/piper/en_US-lessac-medium.onnx.json` (4,885 bytes).
- Verified synthesis: 3 sentences gave 3 chunks, each 22050 Hz, int16, mono. The model loads in about 3 to 4.6 s. The first run took 0.99 s to synthesize 4.41 s of audio (RTF 0.22), and the first chunk was ready after about 635 ms, including warm-up.

### Implications for Spark
- The TTS backend should call `PiperVoice.load(path)` once, off the event loop. For each sentence, run `for ch in voice.synthesize(sentence, SynthesisConfig(length_scale=...))` inside `asyncio.to_thread`. Feed `ch.audio_int16_array` at `ch.sample_rate` (22050) to the output stream; open the speaker at 22050 or resample. Spark's 16 kHz internal format applies to *input* only.
- Warm up with one short synth at startup to hide the first-call latency.
- `download_voice` streams straight into the final filename, and `_needs_download` (`download_voices.py:123`) only checks for a missing or empty file. An interrupted download leaves a truncated `.onnx` that will not be re-fetched. Download to a temp name and rename, or check the size.

---

## UNVERIFIED / risks
- The shape of `StreamEvent.event` comes from the Anthropic Messages streaming spec and the CLI strings, not from a live run. Verify it with a `-m live` test.
- ~~Whether `strict_mcp_config=True` also suppresses claude.ai-account connectors~~ **Verified live 2026-10-05**: it does (see "claude.ai account connectors" at the end).
- ~~Whether a PreToolUse allow decision fully bypasses the CLI permission check for tools not in `allowed_tools`~~ **Verified live in P3.1**: it does, and `{}` does NOT (see the P3.1 section).
- Resetting a conversation with "/clear" through `query()` is not verified. Use disconnect and reconnect.
- The Porcupine real detection path needs a valid AccessKey, so it goes to the manual QA checklist (`docs/dev/MANUAL_QA.md`).

---

## Verified live (2026-09-24, P1.10)
Run on this laptop with the logged-in CLI (no `ANTHROPIC_API_KEY` in env). SDK 0.2.159 with its bundled CLI 2.1.281. Model `claude-haiku-4-5-20251001`. Commands: `uv run --no-sync pytest -q -m live -k capital -s` and `-k two_turn`, plus a 4-turn scratch script through `ClaudeAgentBrain`.
- Options `tools=[]`, `setting_sources=[]`, `strict_mcp_config=True`, `include_partial_messages=True`, `system_prompt=<str>`, `max_turns=8`, and a `stderr` callback are all accepted. The client connects and answers with them.
- `StreamEvent.event` text deltas arrive exactly as `{"type": "content_block_delta", ..., "delta": {"type": "text_delta", "text": ...}}` (`partial=True` in the live test). This closes the "UNVERIFIED" item on the event shape above.
- The answer is "Tokyo is the capital of Japan.", with no markdown under the voice prompt. A second turn on the same client keeps context ("teal" recalled).
- The client must be connected in the task that uses it. Note that `asyncio.wait_for(client.connect(), t)` on Python 3.11 runs `connect()` in a **new task**, so use `async with asyncio.timeout(t): await client.connect()` instead (found through the offline task-ownership fake).
- **Measured latency** (TTFT = `query()` sent → first text delta):
  - Connect/warm-up (CLI spawn plus initialize) took 9.2 s and 15.6 s in two runs.
  - The first turn after connect had a TTFT of **5.5 to 5.7 s** (3 runs: 5469, 5719 ms, plus the context test).
  - Later turns on the warm client had a TTFT of **1.00 to 1.13 s** (1110, 1125, 1000 ms). Total time was about 1.0 to 1.2 s for one-sentence answers.
  - `ResultMessage.duration_api_ms` is cumulative for the session (11014 → 12075 → 13186 → 14172 ms), not per turn. The first turn adds about 5 s of API time beyond its own answer, cause unknown (not investigated).
  - `usage.cache_read_input_tokens` was 0 on every turn; prompt caching did not hit for this short prompt.

---

## MCP permissions and config: extra facts (P2.1, 2026-09-24)
Verified against SDK 0.2.159 source and the bundled CLI 2.1.281 binary (`_bundled/claude.exe`, grepped; minified names quoted). Details are in `docs/connectors.md` §4.
- **Secrets on argv:** a dict `mcp_servers` goes out as `--mcp-config <json.dumps(...)>` on the command line (`_internal/transport/subprocess_cli.py:663-684`). A str/Path is passed through as-is (`:688`). Do not pre-expand secrets.
- **The CLI expands `${VAR}` in inline `--mcp-config`:** `DZe({configObject, filePath:"command line", expandVars:!0, scope:"dynamic"})`. So leave placeholders in and let the CLI read from its env (`os.environ` plus `options.env`). Some credential vars read as empty in remote url/headers: ANTHROPIC_*, NPM_TOKEN, *_PROXY, GOOGLE_OAUTH_ACCESS_TOKEN, and others.
- **Allow-rule matching** (`qan`): same server and (the rule has no tool part, `mcp__srv`, or the tool is `*`, or a glob match via `*`→`.*`). So `mcp__srv`, `mcp__srv__*` and `mcp__srv__get_*` are all valid allow rules. Allow globs must sit after a literal `mcp__<server>__`. Deny and ask rules accept wildcards anywhere.
- `allowed_tools` means auto-approve, not availability (`types.py:1976`). `disallowed_tools` removes tools from context and wins (`types.py:2053`). `_whole_tool_allowed` is at `types.py:1834`.
- **Server-key normalization** `wn`: `replace(/[^a-zA-Z0-9_-]/g,"_")`. Rule parsing splits on `"__"`, so never use `__` inside a server key.
- **claude.ai account connectors** are injected into SDK sessions by a logged-in CLI unless `ENABLE_CLAUDEAI_MCP_SERVERS` is falsy (`0/false/no/off`) or the `disableClaudeAiConnectors` setting is on. `strict_mcp_config=True` blocks them too (verified live 2026-10-05, see the last section).
- **OAuth-only remote servers:** from CLI 2.1.196 on, SDK runs report their tools as unavailable. They cannot authorize headless (code.claude.com/docs/en/mcp).
- **Windows:** the CLI has no special `npx` handling, so wrap it as `cmd /c npx ...`.

---

## Brain and MCP wiring (P2.4, 2026-09-24)
Verified against the installed SDK 0.2.159 (`.venv/Lib/site-packages/claude_agent_sdk/`) and live with the bundled CLI 2.1.281 on this laptop.

**Options passed by `ClaudeAgentBrain._option_kwargs`**
- `mcp_servers`: `McpSetup.servers`, which holds the external servers plus the in-process `spark` server (`types.py:2002`).
- `allowed_tools`: `McpSetup.allowed_tools`. These are exact `mcp__<server>__<tool>` names, read-only only (`types.py:1976`). The brain also drops any glob as a last guard.
- `disallowed_tools`: the destructive tools (`types.py:2053`).
- `env`: `env_for_sdk(settings)` (`types.py:2109`).
- `hooks`: `{"PreToolUse": [HookMatcher(matcher=None, hooks=[gate])]}` (`types.py:2153`, `HookMatcher` at `:610-623`).
- Also passed: `tools=[]`, `setting_sources=[]`, and `strict_mcp_config=True`.

**How the hooks are delivered**
- `ClaudeSDKClient.connect()` converts the hooks with `_hooks_to_internal_format` (`client.py:203`, `types.py:1943`) and sends them in the initialize control request.
- A matcher of `None` means every tool.
- The callback is awaited as `callback(input, tool_use_id, {"signal": None})` (`_internal/query.py:565`). Its dict is passed through `_convert_hook_output_for_cli` (`:571`).

**Phase 2 permission decision: a deny-by-default PreToolUse hook (`spark/brain/tool_gate.py`), not `disallowed_tools`**
- `disallowed_tools` removes a tool from the model's context (`types.py:2053`), and it can't name *unknown* tools.
- `mcp__srv__*` would also deny the read-only tools, because deny rules win.
- The hook sees every tool call, including in-process SDK tools. It works like this:
  - It returns `{}` for pre-approved read-only tools, so the CLI rules apply and `allowed_tools` auto-approves them.
  - For side-effecting, destructive and unknown tools, it returns the verified deny shape (`PreToolUseHookSpecificOutput`, `types.py:437`). The `permissionDecisionReason` is written so the model can relay it.
- `can_use_tool` was rejected for this. It only fires when the CLI rules evaluate to "ask", it is skipped after a hook allow (`types.py:2135-2151`), and it forces `permission_prompt_tool_name="stdio"` (`:1931`).
- P3.1's `confirm.py` replaces the gate through the `tool_gate=` constructor argument (same callback signature).

**Verified live** (`uv run --no-sync pytest -q -m live -k side_effecting -o log_cli=true`):
- The model called `mcp__gh__create_pr`, which is side-effecting and not in `allowed_tools`. The hook logged `tool denied: mcp__gh__create_pr (side_effect)`.
- The handler never ran. Spark said "I can't create pull requests yet through voice confirmation."
- This closes the UNVERIFIED item "a PreToolUse decision is honored for tools not in `allowed_tools`" for the **deny** direction. P3.1 verified the allow direction (below).

**Also verified live**
- The in-process `spark` server (`create_sdk_mcp_server`) works through the long-lived client. The `--text` smoke logged `tool call: mcp__spark__current_time` and spoke the time.
- A fake read-only `lookup_prs` tool returning raw GitHub JSON produced the answer "You have three open pull requests. The newest one is Add dark mode toggle from Sam, created yesterday. Want the others?" (`test_live_tool_results_spoken_without_ids_urls`).
  - The model's own output already had no IDs or URLs under the Phase 2 prompt. `sanitize_for_speech` is the safety net.

**Logging:** `tool call: <name>` is logged from each `ToolUseBlock` of a top-level `AssistantMessage`, and `tool denied: <name> (<class>)` from the gate. Only names are logged, never arguments or results.

---

## Spoken confirmation hook (P3.1, 2026-09-24)
Verified against the installed SDK 0.2.159 (`.venv/Lib/site-packages/claude_agent_sdk/`) and live with the bundled CLI 2.1.281. Scratch scripts ran a real `ClaudeSDKClient` with an in-process `create_note` tool that was **not** in `allowed_tools`, plus `tools=[]`, `setting_sources=[]` and `strict_mcp_config=True`.

**Hook `{}` vs. explicit allow (the open question from P2.4)**

| Hook returns | What the CLI did | Tool ran? |
|---|---|---|
| `{}` | Fell through to its own permission check, which refused the call. The tool result was `is_error=True`: "Claude requested permissions to use mcp__t__create_note, but you haven't granted it yet." It was listed in `ResultMessage.permission_denials`. | **no** |
| `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "permissionDecisionReason": "..."}}` | Ran the tool, with an empty `permission_denials`. | **yes** |

- For a confirmed side-effecting tool, `spark/brain/confirm.py` therefore returns the **explicit allow** (`confirm.allow()`). Read-only tools still get `{}`, and `allowed_tools` auto-approves them, which keeps the allowlist as a second fence.
- Consequence: a side-effecting tool can run only with the hook's explicit allow. With `{}`, a crash, or a timeout it cannot run, because it is never in `allowed_tools`.
- There is no interactive prompt in SDK mode without `can_use_tool`, so no hang either. The call is simply refused.

**Hook timeout**
- `HookMatcher(timeout=...)` defaults to 60 s (`types.py:622`). The SDK forwards it per matcher in the initialize request (`types.py:1955-1956`, `_internal/query.py:280-281`).
- Verified live with `timeout=3` and a hook that sleeps 10 s:
  - The CLI sent `control_cancel_request`, and the SDK cancelled the hook task (`_internal/query.py:365-371`). The hook saw `CancelledError`, and no response is written for it (`:606-610`).
  - The tool result was `is_error=True`: "PreToolUse hook did not respond before its timeout (host client may be unreachable). The tool call was not executed". **The tool did not run.** So an overrun fails closed.
- Spark passes `HookMatcher(matcher=None, hooks=[gate], timeout=confirm.hook_timeout_s)` (default **120 s**). The gate bounds its own wait at `confirm.ask_budget_s` (= hook timeout minus 10 s), so Spark normally answers "No confirmation was heard" itself before the CLI gives up.
- `ConfirmConfig` rejects a hook timeout shorter than the longest answer timeout plus 30 s.

**Dispatch facts relied on**
- Each control request (hook callback, SDK MCP call) runs in its **own task**: `spawn_task(self._handle_control_request(request))` (`_internal/query.py:325`). So a hook that waits on the app's mic does not block the message reader.
- The hook request arrives on the control channel, **not in order** with the `StreamEvent`/`AssistantMessage` stream that the brain worker consumes. So the text before a tool call may not have reached the app yet when the hook fires.
  - Spark's fix is the gate's `before_ask`, which is `ClaudeAgentBrain._await_tool_announced(tool_use_id)`. It waits (at most 2 s) until the worker has seen the `content_block_start` of type `tool_use` with that id, or the `ToolUseBlock`.
  - Meanwhile the worker pushes a `"\n"` delta, so the app's sentence splitter releases the unfinished sentence.
  - Live result: "Sure..." preamble text (when the model writes one) is spoken first, then the question.
- `PreToolUseHookInput` carries `tool_use_id` (`types.py:333`), and the callback also receives it as the second argument (`_internal/query.py:565`). They match the `ToolUseBlock.id` / `content_block.id` from the stream.

**Verified live (P3.1 tests)** `uv run --no-sync pytest -q -m live -k "confirm_no or confirm_yes" -s tests/test_confirm.py`
- **confirm_no**: the model called `mcp__notes__create_note`. The gate asked "I'll create note called groceries. Go ahead?", and the scripted `TextConfirmer` answered "no". The log shows `tool confirm: mcp__notes__create_note -> no`, the handler never ran, and Spark said "Okay, I won't."
- **confirm_yes**: the same flow with "yes". `tool confirm: ... -> yes`, the handler **ran** with `{"title": "groceries"}`, and Spark said "Done, I created a note called groceries."
- **Prompt interaction:** with the unmodified §8.2 rule ("ask 'Should I go ahead?' before any side effect"), Haiku 4.5 asked in its own words and **never called the tool**. The user would then be asked twice.
  - The rule now says the user must hear what will happen and "Should I go ahead?", and that for tool actions Spark does this step automatically when the tool is called.
  - The tool rules add: call the tool directly, and if the answer is no, acknowledge briefly and don't retry.
  - After that change, both live tests passed, and the Phase 1/2 live tests still pass (capital, two-turn, tool results without IDs, side-effecting denied with no confirmer).
- **`--text` smoke:** "Start a new project called zz-confirm-smoke." followed by "no" on stdin. Spark asked the P3.4 summary ("I'll create a project called zz-confirm-smoke: a local folder and git repo, a private GitHub repo and a tracking issue. Should I go ahead?"), read "no" through the shared `LineReader`, and answered "Okay, I won't." Nothing was created under `~/Projects`.

## Routing (P3.2, 2026-09-24)
Verified against the installed SDK 0.2.159 (`.venv/Lib/site-packages/claude_agent_sdk/`) and live with the bundled CLI 2.1.281.

**`set_model` (source)**
- `async def set_model(self, model: str | None = None) -> None` is at `client.py:331`. It is awaitable, takes one positional `model` (`None` means the default model), and raises `CLIConnectionError` if not connected (`client.py:351-352`).
- It calls `Query.set_model` (`_internal/query.py:722`), which sends the control request `{"subtype": "set_model", "model": model}` through `_send_control_request` (`_internal/query.py:623`). The default timeout is 60 s, after which it raises `Exception("Control request timeout: set_model")` (`_internal/query.py:665-668`). A CLI error response is re-raised (`_internal/query.py:660-661`).
- Like every client call, it must run in the task that connected the client (the brain's session worker does this).

**`set_model` (measured live, `probe_set_model.py` scratch plus `test_live_route_deep_switches_model_and_answers`)**
- `claude-sonnet-5` **is accepted** by CLI 2.1.281. After `set_model("claude-sonnet-5")`, `AssistantMessage.model` (`types.py:1144`) reports `claude-sonnet-5`.
- An unknown name raises `Exception("Model 'claude-no-such-model-9' not found")`, and the session **stays on the previous model** (the next reply was served by it).
- Cost: the first `set_model` to a given name on a warm session took 703 ms (sonnet) and 453 ms (haiku). Repeats took about 0 to 16 ms. Right after `connect()` (before any turn) one call took 4.4 s.
  - So the brain pre-visits both models during warm-up priming (`_prewarm_models`: deep, then back to fast) and never pays this on a user turn. A rejected `deep_model` is logged at startup.
- Context carries across the switch. In the live test, the fast turn said "My favourite colour is teal" and the deep turn, served by `claude-sonnet-5`, mentioned teal. Deep TTFT was 2.08 s.

**Per-route tool sets: not possible in the CLI options without reconnecting, so the limit lives in the gate**
- `allowed_tools`, `disallowed_tools`, `mcp_servers` and `model` are **spawn-time CLI flags** (`--allowedTools` `_internal/transport/subprocess_cli.py:604`, `--disallowedTools` :613, `--model` :619, `--mcp-config` :682-688).
- `ClaudeSDKClient` has no runtime method to change the tool set (methods at `client.py:84-614`: `set_permission_mode`, `set_model`, `rewind_files`, `reconnect_mcp_server`, `toggle_mcp_server`, `stop_task`, `get_mcp_status`, `get_context_usage`, `get_server_info`).
- The only per-server switch, `toggle_mcp_server(name, enabled)` (`client.py:409`, control request `mcp_toggle` at `_internal/query.py:759`), "disconnects it and removes its tools". Re-enabling reconnects the server, which respawns a stdio process: exactly the per-turn reconnect we must avoid (spec §9, keep MCP servers warm).
- Decision: one client is spawned with the full tool set, and the per-turn tool limit (spec §8.4 "fast path ... with a limited tool set") is enforced in the `PreToolUse` gate: on a fast-routed turn (`ClaudeAgentBrain.current_route == "fast"`) `ConfirmationGate` denies, without asking, any tool outside `routing.fast_tools` (default: the read-only tools), logs `tool denied: <name> (fast route)`, and the next turn is routed deep. Deep turns keep the full set, with side effects confirmed as before.

**Design as built (`spark/brain/router.py`, `spark/brain/claude_agent.py`)**
- `Router.route(text)` is the rules only (instant). `Router.decide(text)` also runs the optional classifier and logs `route: fast|deep (<reason>)` at INFO. Decisions land in `TurnStats` as `brain_route`, `brain_route_reason`, `brain_model_switched`, `brain_served_model` and `brain_acked` (via `last_turn.as_meta()`).
- `routing.classifier: fast_model` runs a separate tool-less `ClaudeSDKClient` (`max_turns=1`, its own worker task, reconnected every 20 calls). It is bounded by `routing.classifier_timeout_s` and falls back to the rules on timeout, error or an unclear answer. It is off by default and creates nothing when off.
- If a deep turn has no reply text after `routing.ack_after_s` (5 s, injected clock), `respond()` yields `"On it, give me a moment. "` once. `ConfirmationGate(on_asking=...)` pauses that timer while a confirmation question is pending. The new-topic ack (`FRESH_START_ACK`) returns before any routing.

## Sub-agents for the voice/worker split (P5.7d, verified live 2026-10-05, CLI 2.1.x)

Probed with `ClaudeAgentOptions(tools=["Agent"], agents={"researcher": AgentDefinition(model="haiku", tools=["mcp__data__lookup"], maxTurns=4)})`:

- The sub-agent tool is **`Agent`**. `tools=["Task"]` is accepted as an alias, but the tool the model calls is still named `Agent`.
- The main thread's `Agent` call reaches `PreToolUse` with `agent_id=None`. The **worker's own tool calls reach `PreToolUse` too**, with `agent_id` (for example `ace099ea476185fc3`) and `agent_type` (`"researcher"`). So the confirmation gate and the activity log cover workers. The step budget counts only `agent_id is None` calls; the worker has its own `maxTurns`.
- Worker `AssistantMessage`s arrive in `receive_response()` with `parent_tool_use_id` set to the `Agent` call's id and `model` set to the worker's model (`claude-haiku-4-5-20251001`). The brain already never speaks those.
- `ResultMessage.model_usage` has one entry per model: `claude-haiku-4-5-20251001` (the worker) and the main model.
- `model="claude-sonnet-5-5"` works on the subscription login, but the bundled CLI logs `[claude-code:unrecognized_model]` to stderr. That's harmless.

## claude.ai account connectors (verified live 2026-10-05, SDK 0.2.159, bundled CLI)

Probed with scratch scripts (`probe_claudeai.py`, `probe_hide.py`) on the user's logged-in CLI, with `tools=[]` and `setting_sources=[]`.

- The CLI injects the logged-in user's claude.ai connectors **only when `ENABLE_CLAUDEAI_MCP_SERVERS` is truthy AND `strict_mcp_config=False`**. With `strict_mcp_config=True`, `get_mcp_status()` returned `[]`.
- With `strict_mcp_config=False` and `setting_sources=[]`, the only extra servers were the claude.ai ones (this user's `~/.claude.json` has no `mcpServers` and there is no `.mcp.json`). In principle non-strict could also load user/project MCP servers, so the PreToolUse gate keeps denying every tool outside Spark's policy.
- `client.get_mcp_status()` lists each one as `{"name": "claude.ai Gmail", "status": "connected" | "needs-auth" | ..., "source", "scope", "serverInfo", "config", "tools": [{"name": ...}, ...]}`.
- Tool names the model sees: `mcp__claude_ai_Gmail__<tool>`, `mcp__claude_ai_Google_Calendar__<tool>`. Server key = display name with `"claude.ai "` -> `"claude_ai_"` and spaces -> `"_"` (the CLI's general key normalization replaces any char outside `[a-zA-Z0-9_-]` with `_`).
- `disallowed_tools` with a **server-level** entry (`mcp__claude_ai_Slack`) removes all of that server's tools from the model's tool list. A single-tool entry (`mcp__claude_ai_Gmail__send_message`) removes just that tool. Checked in the init `SystemMessage` tool list.
- A `needs-auth` server only offers `authenticate` / `complete_authentication`. Spark's policy doesn't list them, so the gate denies them. The user signs in on claude.ai (Settings -> Connectors) or with `/mcp` in Claude Code.
- This account had: Claude Docs, Slack, Microsoft 365 (needs-auth), Atlassian Rovo, Google Drive (needs-auth), Asana, Gmail (30 tools), Google Calendar (9 tools).
  - Gmail: create_draft, update_draft, delete_draft, send_message, reply, forward, list_drafts, get_draft, get_thread, get_message, search_threads, label_thread, unlabel_thread, apply_sensitive_thread_label, trash_thread, untrash_thread, mark_thread_spam, unmark_thread_spam, list_labels, label_message, update_message_labels, unlabel_message, apply_sensitive_message_label, trash_message, untrash_message, mark_message_spam, unmark_message_spam, create_label, update_label, delete_label.
  - Calendar: list_events, get_event, list_calendars, suggest_time, create_event, update_event, delete_event, respond_to_event, search_events.

**How Spark uses this** (`spark/brain/claudeai.py`, `mcp_config.py`, `claude_agent.py`): `claude_ai_*` keys in `connectors.servers` are policies without an `mcp_servers.json` entry. When at least one is enabled (and `block_claudeai_connectors` is false), the voice session runs with `strict_mcp_config=False` and `ENABLE_CLAUDEAI_MCP_SERVERS=true`. The router classifier and side-question sessions stay strict with the variable false. A server-level deny rule goes to every claude.ai server that isn't enabled: the ones seen in the last `get_mcp_status()` (saved to `claudeai_connectors.json`), configured-but-off ones, and a static list of common ones (Slack, Asana, Atlassian Rovo, Claude Docs, Microsoft 365, Google Drive) for the first session.
