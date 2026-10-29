"""Provider presets for the OpenAI-compatible brain (``brain.backend: openai_compat``, P5.7e).

A preset fills in what differs between OpenAI-compatible endpoints: the base URL, the env var
that holds the key (its NAME; the value only ever comes from the environment or ``.env``), how
the key is sent, and suggested model names. Every explicit field under ``brain.openai_compat``
overrides the preset.

The model names in :data:`PRESETS` are **suggestions**, not checks: providers rename and retire
models often, and for local servers (Ollama, LM Studio, vLLM) the right name is whatever you
pulled or loaded. ``python -m spark doctor`` lists what the endpoint offers.

Pure Python, no network, no ``httpx``: config, prefs and the desktop window import it freely.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from spark.brain.mcp_config import expand_vars, referenced_vars

if TYPE_CHECKING:
    from spark.config import OpenAICompatConfig, Settings

__all__ = [
    "PRESETS",
    "Preset",
    "Provider",
    "provider_names",
    "resolve_provider",
    "resolved_models",
]

Auth = Literal["bearer", "api-key", "none"]


@dataclass(frozen=True)
class Preset:
    name: str
    label: str  # how Spark says it ("I can't reach Ollama")
    base_url: str | None
    key_env: str | None  # env var NAME holding the key
    key_required: bool
    auth: Auth = "bearer"
    fast_model: str | None = None  # suggestion only
    deep_model: str | None = None  # suggestion only
    local: bool = False  # runs on this computer: "is it running?" on connection errors
    #: Ask for token usage in the stream (``stream_options.include_usage``). Off where a server
    #: may reject the field.
    stream_usage: bool = False
    #: OpenAI's current models want ``max_completion_tokens``; most others take ``max_tokens``.
    max_tokens_field: str = "max_tokens"
    headers: Mapping[str, str] = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    "openai": Preset(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        True,
        # Suggestions: non-reasoning models answer fastest by voice. A reasoning model
        # (gpt-5-mini, ...) works too; set extra_body: {reasoning_effort: minimal} for speed.
        fast_model="gpt-4.1-mini",
        deep_model="gpt-4.1",
        stream_usage=True,
        max_tokens_field="max_completion_tokens",
    ),
    "azure": Preset(
        "azure",
        "Azure OpenAI",
        None,  # https://<resource>.openai.azure.com/openai/v1 (model = deployment name)
        "AZURE_OPENAI_API_KEY",
        True,
        auth="api-key",
        stream_usage=True,
        max_tokens_field="max_completion_tokens",
    ),
    "openrouter": Preset(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        True,
        fast_model="openai/gpt-4.1-mini",  # suggestion
        deep_model="openai/gpt-4.1",  # suggestion
        stream_usage=True,
        headers={"X-Title": "Spark"},
    ),
    "groq": Preset(
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        True,
        fast_model="llama-3.1-8b-instant",  # suggestion
        deep_model="llama-3.3-70b-versatile",  # suggestion
    ),
    "together": Preset(
        "together",
        "Together",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        True,
        fast_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # suggestion
        deep_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # suggestion
    ),
    "ollama": Preset(
        "ollama",
        "Ollama",
        "http://localhost:11434/v1",
        None,
        False,
        auth="none",
        # Suggestions with good tool calling: `ollama pull qwen2.5:7b` (about 5 GB).
        fast_model="qwen2.5:7b",
        deep_model="qwen2.5:14b",
        local=True,
    ),
    "lmstudio": Preset(
        "lmstudio",
        "LM Studio",
        "http://localhost:1234/v1",
        None,
        False,
        auth="none",
        local=True,  # models: whatever is loaded in LM Studio (set fast_model / deep_model)
    ),
    # Hermes Agent (Nous Research): its API server speaks the OpenAI API and runs the request
    # through the agent, with Hermes's own tools, memory and skills. It needs a key: the
    # API_SERVER_KEY from Hermes's .env.
    "hermes": Preset(
        "hermes",
        "Hermes Agent",
        "http://127.0.0.1:8642/v1",
        "HERMES_API_KEY",
        True,
        fast_model="hermes-agent",  # the name its /v1/models lists for the default profile
        deep_model="hermes-agent",
        local=True,
    ),
    "vllm": Preset(
        "vllm",
        "vLLM",
        "http://localhost:8000/v1",
        "VLLM_API_KEY",  # only if the server was started with --api-key
        False,
        local=True,  # models: the --model the server was started with
    ),
    # Any other OpenAI-compatible server (the Brain window's "Custom"): the address is required;
    # a key is sent only when one is saved.
    "custom": Preset(
        "custom",
        "Custom server",
        None,
        "OPENAI_COMPAT_API_KEY",
        False,
    ),
}


def provider_names() -> list[str]:
    return list(PRESETS)


@dataclass
class Provider:
    """A preset with the user's overrides applied. Holds the key's env var NAME, never a key."""

    name: str
    label: str
    base_url: str
    key_env: str | None
    key_required: bool
    auth: Auth
    local: bool
    stream_usage: bool
    max_tokens_field: str
    api_version: str | None
    headers: dict[str, str]  # extra headers; values may hold ${VAR} placeholders
    fast_model: str | None
    deep_model: str | None
    worker_model: str | None
    problems: list[str] = field(default_factory=list)

    def api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if environ is None else environ
        if not self.key_env:
            return None
        value = (env.get(self.key_env) or "").strip()
        return value or None

    def request_headers(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Headers for every request: the auth header plus ``extra_headers`` (``${VAR}``
        expanded here, in this process only). Never log the result."""
        env = os.environ if environ is None else environ
        out = {k: expand_vars(v, env) for k, v in self.headers.items()}
        key = self.api_key(env)
        if key and self.auth == "api-key":
            out["api-key"] = key
        elif key and self.auth in ("bearer", "none"):
            out["Authorization"] = f"Bearer {key}"
        return out

    def missing_env(self, environ: Mapping[str, str] | None = None) -> list[str]:
        """Env var names that must be set but aren't (the key, ${VAR}s in extra headers)."""
        env = os.environ if environ is None else environ
        missing = []
        if self.key_required and self.key_env and not self.api_key(env):
            missing.append(self.key_env)
        required, _ = referenced_vars(self.headers)
        missing += sorted(v for v in required if not env.get(v))
        return missing

    def url(self, path: str, model: str | None = None) -> str:
        """``{base_url}/{path}``. A ``{model}`` placeholder in the base URL (classic Azure
        ``.../openai/deployments/{model}``) takes the model (deployment) name."""
        base = self.base_url.rstrip("/")
        if "{model}" in base:
            base = base.replace("{model}", model or "")
        return f"{base}/{path.lstrip('/')}"

    def params(self) -> dict[str, str]:
        return {"api-version": self.api_version} if self.api_version else {}


def resolve_provider(cfg: OpenAICompatConfig) -> Provider:
    """Apply ``brain.openai_compat`` over its preset. Problems (no base URL, no model name) are
    listed in ``problems`` rather than raised, so the doctor and the window can show them."""
    preset = PRESETS.get(cfg.provider)
    problems: list[str] = []
    if preset is None:
        problems.append(f"unknown provider preset {cfg.provider!r} (choices: {', '.join(PRESETS)})")
        preset = Preset(cfg.provider, cfg.provider, None, None, False)
    base_url = (cfg.base_url or preset.base_url or "").strip()
    if not base_url:
        problems.append(
            f"brain.openai_compat.base_url is required for {preset.label}"
            + (
                " (e.g. https://<resource>.openai.azure.com/openai/v1)"
                if preset.name == "azure"
                else ""
            )
        )
    elif not re.match(r"^https?://", base_url, re.I):
        problems.append("brain.openai_compat.base_url must start with http:// or https://")
    key_env = cfg.api_key_env if cfg.api_key_env is not None else preset.key_env
    key_env = (key_env or "").strip() or None
    # An explicit api_key_env means "this server wants a key"; a preset decides otherwise.
    key_required = preset.key_required or (cfg.api_key_env is not None and bool(key_env))
    headers: dict[str, str] = {**preset.headers, **cfg.extra_headers}
    fast = (cfg.fast_model or preset.fast_model or "").strip() or None
    deep = (cfg.deep_model or preset.deep_model or "").strip() or None
    if fast is None and deep is not None:
        fast = deep
    if deep is None and fast is not None:
        deep = fast
    if fast is None:
        problems.append(
            f"no model set for {preset.label}: set brain.openai_compat.fast_model and deep_model"
        )
    return Provider(
        name=preset.name,
        label=preset.label,
        base_url=base_url,
        key_env=key_env,
        key_required=key_required,
        auth=preset.auth if key_env else "none",
        local=preset.local or _is_local(base_url),
        stream_usage=preset.stream_usage,
        max_tokens_field=preset.max_tokens_field,
        api_version=(cfg.api_version or "").strip() or None,
        headers=headers,
        fast_model=fast,
        deep_model=deep,
        worker_model=(cfg.worker_model or "").strip() or None,
        problems=problems,
    )


def _is_local(url: str) -> bool:
    return bool(re.match(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:|/|$)", url, re.I))


def resolved_models(settings: Settings) -> tuple[str, str, str | None]:
    """``(fast, deep, worker or None)`` for the active backend, presets applied. Claude ids for
    ``claude_agent`` (and the mock), the ``brain.openai_compat`` names for ``openai_compat``."""
    brain = settings.brain
    if brain.backend != "openai_compat":
        return brain.fast_model, brain.deep_model, brain.worker_model
    prov = resolve_provider(brain.openai_compat)
    return prov.fast_model or "", prov.deep_model or "", prov.worker_model


def describe(provider: Provider) -> dict[str, Any]:
    """Safe-to-show summary (names only, never a key or header value)."""
    return {
        "provider": provider.name,
        "label": provider.label,
        "base_url": provider.base_url,
        "key_env": provider.key_env,
        "fast_model": provider.fast_model,
        "deep_model": provider.deep_model,
        "worker_model": provider.worker_model,
    }
