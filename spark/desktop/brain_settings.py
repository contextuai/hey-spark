"""The Brain window's API (P5.7f, done early): choose what thinks for Spark, enter its key, test
it, pick its models, and apply it without restarting. Claude runs through the user's installed
Claude Code, with the sign-in Claude Code already has (a Claude subscription) or an Anthropic API
key (``ANTHROPIC_API_KEY``). The window never signs anyone in itself: it only reads ``claude auth
status``.

Built to become one tab of the later Settings window: everything it needs comes in through the
constructor and it keeps no window state of its own.

Where things go:

* The provider, address and model names go to ``prefs.json`` (:mod:`spark.prefs`), which
  overrides ``config.yaml``. ``config.yaml`` is never written; it stays the default for anyone
  who doesn't use the window.
* An API key goes to ``.env`` next to ``config.yaml`` under the preset's variable name
  (:func:`spark.brain.connector_store.set_env_secret`) and into this process's environment. It is
  never written to ``prefs.json``, never logged, and never sent back to the page: the page only
  learns "a key is saved, ending in ...abcd".
* Applying asks the controller to rebuild the brain (``on_apply``), which answers None or what
  went wrong. The brain keys from before the save are kept in ``prefs.json`` (``brain_previous``)
  so a failed change can be reverted in one click.

pywebview calls these methods on its own worker threads, so each is plain blocking code that
returns a JSON-able dict; network checks run their own event loop with hard timeouts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from spark.brain import connector_store as store
from spark.brain.openai_presets import PRESETS, Provider, resolve_provider
from spark.config import OpenAICompatConfig, Settings
from spark.desktop import ollama
from spark.prefs import BRAIN_KEYS, MODEL_CHOICES, Prefs, apply_prefs, model_label, reset_brain

log = logging.getLogger(__name__)

__all__ = ["LINKS", "PROVIDERS", "BrainSettingsApi"]

CLAUDE = "claude"
#: An optional Anthropic API key for the Claude brain, saved to .env under this name. Without one,
#: Claude Code uses the account it is signed in with (``claude`` then ``/login``).
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
AUTH_STATUS_TIMEOUT_S = 15.0
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
CLAUDE_CODE_URL = "https://code.claude.com/docs/en/setup"
HERMES_API_SERVER_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server"
TEST_TIMEOUT_S = 20.0  # per request in a test (the brain's own timeout_s is for long answers)
APPLY_TIMEOUT_S = 300.0  # a first start may download the speech models

#: What the window offers, in order. ``kind``: claude | agent | cloud | local | custom.
PROVIDERS: tuple[dict[str, Any], ...] = (
    {
        "id": CLAUDE,
        "label": "Claude",
        "kind": "claude",
        "description": "Anthropic's Claude, through Claude Code on this computer: it uses the "
        "Claude account Claude Code is signed in with, or an Anthropic API key if you add one.",
        "link": "https://platform.claude.com/settings/keys",
        "link_label": "Get a key at platform.claude.com",
    },
    {
        "id": "hermes",
        "label": "Hermes Agent",
        "kind": "agent",
        "description": "Nous Research's Hermes Agent, on this computer or on your server, "
        "with its own model, tools, memory and skills. Spark becomes its voice. Turn on "
        "Hermes's API server, then enter its address and its API_SERVER_KEY below.",
        "link": HERMES_API_SERVER_URL,
        "link_label": "How to turn on Hermes's API server",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "kind": "cloud",
        "description": "Needs an API key from platform.openai.com. You pay OpenAI per use.",
        "link": "https://platform.openai.com/api-keys",
        "link_label": "Get a key at platform.openai.com",
    },
    {
        "id": "azure",
        "label": "Azure OpenAI",
        "kind": "cloud",
        "description": "Your organization's Azure OpenAI resource. Needs its address, a key "
        "and the names of your deployments.",
        "link": "https://portal.azure.com/",
        "link_label": "Find the address and key in the Azure portal",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "kind": "cloud",
        "description": "One key for models from many companies. Key from openrouter.ai.",
        "link": "https://openrouter.ai/keys",
        "link_label": "Get a key at openrouter.ai",
    },
    {
        "id": "groq",
        "label": "Groq",
        "kind": "cloud",
        "description": "Very fast open models, with a free tier. Key from console.groq.com.",
        "link": "https://console.groq.com/keys",
        "link_label": "Get a key at console.groq.com",
    },
    {
        "id": "together",
        "label": "Together",
        "kind": "cloud",
        "description": "Open models in the cloud. Key from together.ai.",
        "link": "https://api.together.ai/settings/api-keys",
        "link_label": "Get a key at together.ai",
    },
    {
        "id": "ollama",
        "label": "Ollama (local)",
        "kind": "local",
        "description": "Runs on this computer, free, needs a download (about 2 to 5 GB).",
        "link": ollama.DOWNLOAD_URL,
        "link_label": "Download Ollama",
    },
    {
        "id": "lmstudio",
        "label": "LM Studio (local)",
        "kind": "local",
        "description": "Runs on this computer, free. Load a model in LM Studio and start its "
        "server.",
        "link": "https://lmstudio.ai/download",
        "link_label": "Download LM Studio",
    },
    {
        "id": "custom",
        "label": "Custom",
        "kind": "custom",
        "description": "Any other server that speaks the OpenAI API (vLLM, a company "
        "gateway, ...). You need its address.",
    },
)

#: The only pages :meth:`BrainSettingsApi.open_link` opens.
LINKS = frozenset({*(p["link"] for p in PROVIDERS if p.get("link")), CLAUDE_CODE_URL})

_EDITABLE_URL = {"azure", "ollama", "lmstudio", "hermes", "custom", "vllm"}
_MODEL_NAME = re.compile(r"^[A-Za-z0-9._:/@-]{1,120}$")


def _info(provider_id: str) -> dict[str, Any]:
    for p in PROVIDERS:
        if p["id"] == provider_id:
            return p
    preset = PRESETS.get(provider_id)  # vllm: configured in config.yaml, not offered here
    label = preset.label if preset else provider_id
    return {"id": provider_id, "label": label, "kind": "custom", "description": ""}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _last4(value: str) -> str:
    return value[-4:] if len(value) >= 12 else ""


# --- Claude Code --------------------------------------------------------------------------------


def _claude_cli(settings: Settings) -> str | None:
    """The installed ``claude`` CLI the brain will run (the frozen app doesn't ship one)."""
    from spark.doctor import find_claude_cli

    return find_claude_cli(settings)


_PLANS = {"max": "Max plan", "pro": "Pro plan", "team": "Team plan", "enterprise": "Enterprise"}


def parse_auth_status(text: str) -> dict[str, Any]:
    """``claude auth status --json`` -> whether Claude Code is signed in, and how. Only the
    sign-in flag, method and plan are kept; ids, emails, paths and tokens never leave here."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"signed_in": False, "method": None, "plan": None}
    method = data.get("authMethod") if isinstance(data.get("authMethod"), str) else None
    plan = data.get("subscriptionType")
    plan_label = _PLANS.get(str(plan).lower(), str(plan).capitalize()) if plan else None
    return {"signed_in": bool(data.get("loggedIn")), "method": method, "plan": plan_label}


def run_auth_status(cli: str) -> dict[str, Any]:
    """Run ``claude auth status --json`` without Spark's keys, so it reports Claude Code's own
    sign-in (an ``ANTHROPIC_API_KEY`` in the environment would mask it)."""
    from spark.util.logging import child_env

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        out = subprocess.run(
            [cli, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=AUTH_STATUS_TIMEOUT_S,
            check=False,
            env=child_env(),
            creationflags=flags,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("brain settings: claude auth status failed: %s", type(exc).__name__)
        return {"signed_in": False, "method": None, "plan": None}
    return parse_auth_status(out.stdout or "")


def _sign_in_text(auth: Mapping[str, Any]) -> str:
    plan = f" ({auth['plan']})" if auth.get("plan") else ""
    if (auth.get("method") or "").lower() in ("api_key", "apikey"):
        return "Claude Code is signed in with an Anthropic API key."
    return f"Claude Code is signed in with your Claude account{plan}."


NOT_SIGNED_IN = (
    "Claude Code isn't signed in. Open a terminal, run claude and type /login, or add an "
    "Anthropic API key below."
)


# --- the API -----------------------------------------------------------------------------------


class BrainSettingsApi:
    """Methods the Brain page calls as ``pywebview.api.<name>()``.

    Args:
        settings: the app's live settings (changed in place on apply).
        baseline: ``settings.brain`` as ``config.yaml`` had it, before prefs (the controller's
            ``config_brain``). Default: a copy of ``settings.brain`` now.
        on_apply: rebuilds the brain; returns None or the error in words.
        environ: where keys are read and set (default ``os.environ``; tests pass a dict).
        transport: ``httpx`` transport for tests (provider checks).
        ollama_transport: ``httpx`` transport for tests (Ollama's own API).
        find_cli: replaces looking for the installed ``claude`` CLI (tests).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        baseline: Any | None = None,
        on_apply: Callable[[], str | None] | None = None,
        environ: MutableMapping[str, str] | None = None,
        transport: Any | None = None,
        ollama_transport: Any | None = None,
        find_cli: Callable[[Settings], str | None] | None = None,
        auth_status: Callable[[str], dict[str, Any]] | None = None,
        open_url: Callable[[str], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._baseline = baseline if baseline is not None else settings.brain.model_copy(deep=True)
        self._on_apply = on_apply
        self._environ: MutableMapping[str, str] = os.environ if environ is None else environ
        self._transport = transport
        self._ollama_transport = ollama_transport
        self._puller = ollama.Puller(transport=ollama_transport)
        self._find_cli = find_cli or _claude_cli
        self._auth_status = auth_status or run_auth_status
        self._open_url = open_url

    # --- helpers -------------------------------------------------------------------------------

    def _prefs(self) -> Prefs:
        return Prefs.from_settings(self._settings)

    def _env_file(self) -> Any:
        return store.env_path(self._settings)

    def _active_id(self) -> str:
        if self._settings.brain.backend == "openai_compat":
            return self._settings.brain.openai_compat.provider
        return CLAUDE

    def _form_config(self, form: Mapping[str, Any]) -> tuple[OpenAICompatConfig, Provider]:
        """The form's provider as a config section and a resolved provider. The settings that
        aren't in the window (timeouts, extra_body, ...) come from the live config; the key
        variable and extra headers/body only when the provider is the one config.yaml names."""
        name = _clean(form.get("provider"))
        cfg = self._settings.brain.openai_compat.model_copy(deep=True)
        base = self._baseline.openai_compat
        cfg.provider = name if name in PRESETS else "custom"
        if name != base.provider:
            cfg.api_key_env = None
            cfg.extra_headers = {}
            cfg.extra_body = {}
        else:
            cfg.api_key_env = base.api_key_env
            cfg.extra_headers = dict(base.extra_headers)
            cfg.extra_body = dict(base.extra_body)
        url = _clean(form.get("base_url"))
        cfg.base_url = url if url and name in _EDITABLE_URL else None
        cfg.api_version = (_clean(form.get("api_version")) or None) if name == "azure" else None
        for role in ("fast", "deep"):
            setattr(cfg, f"{role}_model", _clean(form.get(f"{role}_model")) or None)
        cfg.worker_model = _clean(form.get("worker_model")) or None
        return cfg, resolve_provider(cfg)

    def _key_status(self, key_env: str | None) -> dict[str, Any]:
        if not key_env:
            return {"env": None, "saved": False, "where": None, "last4": ""}
        in_file = store.read_env_file(self._env_file()).get(key_env, "").strip()
        if in_file:
            return {"env": key_env, "saved": True, "where": ".env", "last4": _last4(in_file)}
        in_env = (self._environ.get(key_env) or "").strip()
        if in_env:
            return {"env": key_env, "saved": True, "where": "environment", "last4": _last4(in_env)}
        return {"env": key_env, "saved": False, "where": None, "last4": ""}

    def _provider_view(self, info: Mapping[str, Any], prefs: Prefs) -> dict[str, Any]:
        pid = info["id"]
        view = {k: v for k, v in info.items()}
        if pid == CLAUDE:
            view.update(
                {
                    "needs_key": False,
                    "takes_key": True,
                    "install_link": CLAUDE_CODE_URL,
                    "key": self._key_status(ANTHROPIC_KEY_ENV),
                }
            )
            return view
        preset = PRESETS.get(pid)
        oc = self._settings.brain.openai_compat
        saved = (prefs.get("openai_saved") or {}).get(pid)
        if self._active_id() == pid:
            values = {
                "base_url": oc.base_url or "",
                "api_version": oc.api_version or "",
                "fast_model": oc.fast_model or "",
                "deep_model": oc.deep_model or "",
                "worker_model": oc.worker_model or "",
            }
        elif isinstance(saved, dict):
            values = {
                k: _clean(saved.get(k))
                for k in ("base_url", "api_version", "fast_model", "deep_model", "worker_model")
            }
        elif self._baseline.openai_compat.provider == pid:
            b = self._baseline.openai_compat
            values = {
                "base_url": b.base_url or "",
                "api_version": b.api_version or "",
                "fast_model": b.fast_model or "",
                "deep_model": b.deep_model or "",
                "worker_model": b.worker_model or "",
            }
        else:
            values = dict.fromkeys(
                ("base_url", "api_version", "fast_model", "deep_model", "worker_model"), ""
            )
        _, prov = self._form_config({"provider": pid, **values})
        view.update(
            {
                "default_base_url": preset.base_url if preset else None,
                "url_editable": pid in _EDITABLE_URL,
                "needs_key": bool(preset and preset.key_required),
                "takes_key": bool(prov.key_env),
                "suggested": {
                    "fast_model": preset.fast_model if preset else None,
                    "deep_model": preset.deep_model if preset else None,
                },
                "values": values,
                "key": self._key_status(prov.key_env),
            }
        )
        return view

    # --- reading -------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Everything the page shows. Holds no key, only whether one is saved."""
        prefs = self._prefs()
        active = self._active_id()
        infos = list(PROVIDERS)
        if active not in {p["id"] for p in infos}:  # vllm from config.yaml
            infos.append(_info(active))
        brain = self._settings.brain
        from spark.brain import claudeai
        from spark.brain.mcp_config import claudeai_enabled

        claude_choices = [[mid, label] for mid, label in MODEL_CHOICES]
        known = {mid for mid, _ in claude_choices}
        for mid in (brain.fast_model, brain.deep_model, brain.worker_model):
            if mid and mid not in known:
                claude_choices.append([mid, model_label(mid)])
                known.add(mid)
        try:
            account_connectors = [
                claudeai.display_name(n) for n in claudeai_enabled(self._settings)
            ]
        except Exception:
            account_connectors = []
        return {
            "active": active,
            "active_label": _info(active)["label"],
            "providers": [self._provider_view(p, prefs) for p in infos],
            "claude": {
                "choices": claude_choices,
                "fast_model": brain.fast_model,
                "deep_model": brain.deep_model,
                "worker_model": brain.worker_model or "",
            },
            "account_connectors": account_connectors,
            "can_revert": isinstance(prefs.get("brain_previous"), dict),
            "env_file": str(self._env_file()),
        }

    def claude_status(self) -> dict[str, Any]:
        """Whether Claude Code (the CLI the Claude brain runs) is installed and signed in."""
        try:
            cli = self._find_cli(self._settings)
        except Exception as exc:
            log.warning("brain settings: looking for the claude CLI failed: %s", exc)
            cli = None
        if not cli:
            return {
                "ok": False,
                "installed": False,
                "signed_in": False,
                "message": "Claude Code isn't installed. Install it, then restart Hey Spark.",
            }
        try:
            auth = self._auth_status(cli)
        except Exception as exc:
            log.warning("brain settings: claude auth status failed: %s", exc)
            auth = {"signed_in": False}
        signed_in = bool(auth.get("signed_in"))
        return {
            "ok": True,
            "installed": True,
            "signed_in": signed_in,
            "plan": auth.get("plan") if signed_in else None,
            "message": "Claude Code is installed.",
            "sign_in": _sign_in_text(auth) if signed_in else NOT_SIGNED_IN,
        }

    # --- testing -------------------------------------------------------------------------------

    def test(self, form: Mapping[str, Any]) -> dict[str, Any]:
        """Reach the endpoint, check the key, list the models, then see whether the voice model
        calls tools. ``form`` may hold a key typed but not saved yet; it is used for this test
        only and never returned."""
        form = dict(form or {})
        key = _clean(form.pop("key", ""))
        if _clean(form.get("provider")) == CLAUDE:
            return self._test_claude(key)
        cfg, prov = self._form_config(form)
        cfg.timeout_s = min(cfg.timeout_s, TEST_TIMEOUT_S)
        if prov.problems:
            return _result(False, [("error", _problem_text(prov))])
        environ = dict(self._environ)
        if key and prov.key_env:
            environ[prov.key_env] = key
        missing = prov.missing_env(environ)
        if missing:
            return _result(False, [("error", f"Enter your {prov.label} API key first.")])
        try:
            result = asyncio.run(self._atest(prov, cfg, environ))
        except Exception as exc:  # never leave the page waiting
            log.exception("brain settings: test failed unexpectedly")
            result = _result(False, [("error", f"The test failed: {type(exc).__name__}.")])
        from spark.util.logging import redact

        secrets = [key] if key else []
        for step in result["steps"]:
            step["text"] = redact(step["text"], extra_secrets=secrets, environ={})
        log.info(
            "brain settings: test %s at %s -> %s (tools: %s)",
            prov.name,
            prov.base_url,
            "ok" if result["ok"] else "failed",
            result["tools"],
        )
        return result

    def _test_claude(self, key: str) -> dict[str, Any]:
        """Claude Code installed, then either Anthropic accepts the API key (``GET /v1/models``)
        or, without a key, Claude Code is signed in."""
        status = self.claude_status()
        steps = [("ok" if status["installed"] else "error", status["message"])]
        key = key or (self._environ.get(ANTHROPIC_KEY_ENV) or "").strip()
        if not key:
            if not status["installed"]:
                return _result(False, steps)
            if status["signed_in"]:
                steps.append(("ok", status["sign_in"] + " Spark will use it."))
            else:
                steps.append(("error", NOT_SIGNED_IN))
            ok = all(level == "ok" for level, _ in steps)
            log.info("brain settings: test claude (sign-in) -> %s", "ok" if ok else "failed")
            return _result(ok, steps)
        import httpx

        try:
            with httpx.Client(transport=self._transport, timeout=TEST_TIMEOUT_S) as client:
                resp = client.get(
                    ANTHROPIC_MODELS_URL,
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                )
        except httpx.HTTPError as exc:
            steps.append(("error", f"Couldn't reach Anthropic ({type(exc).__name__})."))
            return _result(False, steps)
        if resp.status_code in (401, 403):
            steps.append(("error", "Anthropic refused the API key. Check it and try again."))
        elif resp.status_code >= 400:
            steps.append(("error", f"Anthropic answered with an error ({resp.status_code})."))
        else:
            steps.append(("ok", "Anthropic accepted the API key."))
        ok = all(level == "ok" for level, _ in steps)
        log.info("brain settings: test claude -> %s", "ok" if ok else "failed")
        return _result(ok, steps)

    async def _atest(
        self, prov: Provider, cfg: OpenAICompatConfig, environ: Mapping[str, str]
    ) -> dict[str, Any]:
        from spark.brain.endpoint_check import explain, fetch_models, probe_tools
        from spark.brain.openai_chat import ChatClient, ProviderError

        chat = ChatClient(prov, cfg, transport=self._transport, environ=environ)
        steps: list[tuple[str, str]] = []
        try:
            try:
                models = await fetch_models(chat)
            except ProviderError as exc:
                return _result(False, [("error", explain(exc, prov))])
            keyed = " and the key works" if prov.api_key(environ) else ""
            if models:
                steps.append(
                    (
                        "ok",
                        f"Connected to {prov.label}{keyed}. It offers "
                        f"{len(models)} model{'s' if len(models) != 1 else ''}.",
                    )
                )
            else:
                steps.append(
                    (
                        "ok",
                        f"Connected to {prov.label}{keyed}. It doesn't list its "
                        "models, so type the model names.",
                    )
                )
            if prov.name == "ollama" and not models:
                steps.append(("error", "Ollama has no models yet. Download one below."))
                return _result(False, steps, models)
            model = prov.deep_model or ""
            if not model:
                steps.append(("error", "Choose the voice model."))
                return _result(False, steps, models)
            if models and model not in models:
                steps.append(
                    (
                        "error",
                        f"{prov.label} doesn't list a model called “{model}”. "
                        "Pick one from the list.",
                    )
                )
                return _result(False, steps, models)
            if prov.name == "hermes":
                # Hermes runs its own tools; the probe would start a whole agent turn.
                steps.append(("ok", "Hermes answers with its own tools, memory and skills."))
                return _result(True, steps, models, tools="agent")
            try:
                called = await probe_tools(chat, model)
            except ProviderError as exc:
                if exc.kind == "bad_request" and "tool" in str(exc).lower():
                    steps.append(
                        (
                            "warn",
                            f"“{model}” can't use tools. Spark's connectors and "
                            "timers won't work with it. Pick another model.",
                        )
                    )
                    return _result(True, steps, models, tools="no")
                steps.append(("error", explain(exc, prov, model=model)))
                return _result(False, steps, models)
            if called:
                steps.append(("ok", f"“{model}” can use tools."))
                return _result(True, steps, models, tools="yes")
            steps.append(
                (
                    "warn",
                    f"“{model}” didn't call tools. Spark's connectors and timers "
                    "won't work well with it.",
                )
            )
            return _result(True, steps, models, tools="no")
        finally:
            await chat.aclose()

    # --- saving --------------------------------------------------------------------------------

    def save(self, form: Mapping[str, Any]) -> dict[str, Any]:
        """Save the provider (prefs), its key (.env), and rebuild the brain."""
        form = dict(form or {})
        key = _clean(form.pop("key", ""))
        provider = _clean(form.get("provider"))
        prefs = self._prefs()
        if provider == CLAUDE:
            brain = self._settings.brain
            fast = _clean(form.get("fast_model")) or brain.fast_model
            deep = _clean(form.get("deep_model")) or brain.deep_model
            worker = (
                _clean(form.get("worker_model"))
                if "worker_model" in form
                else (brain.worker_model or "")
            )
            bad = [m for m in (fast, deep, worker) if m and not _MODEL_NAME.match(m)]
            if bad:
                return {"ok": False, "error": f"“{bad[0]}” isn't a model name."}
            if key:
                try:
                    store.set_env_secret(
                        self._env_file(), ANTHROPIC_KEY_ENV, key, environ=self._environ
                    )
                except OSError as exc:
                    return {
                        "ok": False,
                        "error": f"Couldn't save the key to .env ({exc.strerror}).",
                    }
                log.info("brain settings: saved the Anthropic key to .env as %s", ANTHROPIC_KEY_ENV)
            if not (self._environ.get(ANTHROPIC_KEY_ENV) or "").strip():
                status = self.claude_status()
                if not status["installed"]:
                    return {"ok": False, "error": status["message"]}
                if not status["signed_in"]:
                    return {"ok": False, "error": NOT_SIGNED_IN}
            updates: dict[str, Any] = {
                "brain_backend": "claude_agent",
                "fast_model": fast,
                "deep_model": deep,
                "worker_model": worker,
            }
            label = "Claude"
        else:
            if provider not in PRESETS:
                return {"ok": False, "error": "Choose a provider."}
            cfg, prov = self._form_config(form)
            if prov.problems:
                return {"ok": False, "error": _problem_text(prov)}
            models = (prov.fast_model, prov.deep_model, prov.worker_model)
            bad = [m for m in models if m and not _MODEL_NAME.match(m)]
            if bad:
                return {"ok": False, "error": f"“{bad[0]}” isn't a model name."}
            if key:
                if not prov.key_env:
                    return {"ok": False, "error": f"{prov.label} doesn't take a key here."}
                try:
                    store.set_env_secret(self._env_file(), prov.key_env, key, environ=self._environ)
                except OSError as exc:
                    return {
                        "ok": False,
                        "error": f"Couldn't save the key to .env ({exc.strerror}).",
                    }
                log.info("brain settings: saved the %s key to .env as %s", prov.label, prov.key_env)
            if prov.key_required and not prov.api_key(self._environ):
                return {"ok": False, "error": f"Enter your {prov.label} API key first."}
            preset = PRESETS[provider]
            base_url = cfg.base_url or ""
            if base_url.rstrip("/") == (preset.base_url or "").rstrip("/"):
                base_url = ""  # the preset's: follows the preset if it ever moves
            values = {
                "base_url": base_url,
                "api_version": cfg.api_version or "",
                "fast_model": prov.fast_model or "",
                "deep_model": prov.deep_model or "",
                "worker_model": prov.worker_model or "",
            }
            saved = prefs.get("openai_saved")
            saved = dict(saved) if isinstance(saved, dict) else {}
            saved[provider] = values
            updates = {
                "brain_backend": "openai_compat",
                "openai_provider": provider,
                "openai_base_url": values["base_url"],
                "openai_api_version": values["api_version"],
                "openai_fast_model": values["fast_model"],
                "openai_deep_model": values["deep_model"],
                "openai_worker_model": values["worker_model"],
                "openai_saved": saved,
            }
            label = prov.label
        if not prefs.get("brain_failed"):  # only a setup that worked is worth going back to
            previous = {k: prefs.get(k) for k in BRAIN_KEYS if prefs.get(k) is not None}
            updates["brain_previous"] = previous
        prefs.update(**updates)
        log.info("brain settings: saved %s; applying", label)
        return self._apply(label)

    def use_claude(self) -> dict[str, Any]:
        """Switch back to Claude with its current models, in one click."""
        return self.save({"provider": CLAUDE})

    def revert(self) -> dict[str, Any]:
        """Go back to the brain settings from before the last save."""
        prefs = self._prefs()
        previous = prefs.get("brain_previous")
        if not isinstance(previous, dict):
            return {"ok": False, "error": "There's nothing to go back to."}
        prefs.delete(*BRAIN_KEYS, "brain_previous")
        prefs.update(**{k: v for k, v in previous.items() if k in BRAIN_KEYS})
        log.info("brain settings: reverted to the previous brain")
        backend = previous.get("brain_backend") or self._baseline.backend
        if backend == "openai_compat":
            provider = previous.get("openai_provider") or self._baseline.openai_compat.provider
            label = _info(provider)["label"]
        else:
            label = "Claude"
        return self._apply(label)

    def _apply(self, label: str) -> dict[str, Any]:
        reset_brain(self._settings, self._baseline)
        apply_prefs(self._settings, self._prefs())
        error = None
        if self._on_apply is not None:
            try:
                error = self._on_apply()
            except Exception as exc:
                log.exception("brain settings: applying failed")
                error = f"Applying failed: {exc}"
        prefs = self._prefs()
        if error:
            prefs.update(brain_failed=True)
            log.warning("brain settings: the new brain didn't start: %s", error)
            return {"ok": False, "error": error, "state": self.state()}
        prefs.delete("brain_failed")
        return {"ok": True, "message": f"Spark now thinks with {label}.", "state": self.state()}

    def remove_key(self, provider: str) -> dict[str, Any]:
        """Delete the provider's key from ``.env`` (and this process)."""
        if _clean(provider) == CLAUDE:
            key_env: str | None = ANTHROPIC_KEY_ENV
        else:
            key_env = self._form_config({"provider": provider})[1].key_env
        status = self._key_status(key_env)
        if not key_env or not status["saved"]:
            return {"ok": True, "key": status}
        if status["where"] == "environment":
            return {
                "ok": False,
                "error": f"This key comes from your environment variables "
                f"({key_env}). Remove it there.",
                "key": status,
            }
        try:
            store.remove_env_vars(self._env_file(), [key_env], self._environ)
        except OSError as exc:
            return {"ok": False, "error": f"Couldn't change .env ({exc.strerror}).", "key": status}
        log.info("brain settings: removed %s from .env", key_env)
        return {"ok": True, "key": self._key_status(key_env)}

    def open_link(self, url: str) -> dict[str, Any]:
        """Open a provider's key or download page in the browser (only those pages)."""
        if url not in LINKS:
            return {"ok": False, "error": "That link can't be opened from here."}
        if self._open_url is not None:
            self._open_url(url)
        else:
            import webbrowser

            webbrowser.open(url)
        return {"ok": True}

    # --- Ollama --------------------------------------------------------------------------------

    def ollama_status(self, base_url: str = "") -> dict[str, Any]:
        url = _clean(base_url) or PRESETS["ollama"].base_url
        return ollama.status(url, transport=self._ollama_transport, environ=self._environ)

    def ollama_pull(self, model: str, base_url: str = "") -> dict[str, Any]:
        """Start downloading ``model`` (only ever on the user's click). Poll
        :meth:`ollama_progress`."""
        model = _clean(model)
        if not _MODEL_NAME.match(model):
            return {"ok": False, "error": "That isn't a model name."}
        url = _clean(base_url) or PRESETS["ollama"].base_url
        if not self._puller.start(url, model):
            return {"ok": False, "error": "A download is already running."}
        return {"ok": True, "progress": self._puller.progress()}

    def ollama_progress(self) -> dict[str, Any]:
        return self._puller.progress()

    def ollama_cancel(self) -> dict[str, Any]:
        self._puller.cancel()
        return {"ok": True}


def _result(
    ok: bool,
    steps: list[tuple[str, str]],
    models: list[str] | None = None,
    *,
    tools: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "steps": [{"level": level, "text": text} for level, text in steps],
        "models": sorted(set(models or []), key=str.lower),
        "tools": tools,
    }


def _problem_text(prov: Provider) -> str:
    """resolve_provider's problems in window words."""
    text = "; ".join(prov.problems)
    if "base_url is required" in text:
        return f"Enter the address of {prov.label}" + (
            " (like https://<resource>.openai.azure.com/openai/v1)."
            if prov.name == "azure"
            else "."
        )
    if "must start with http" in text:
        return "The address must start with http:// or https://."
    if "no model set" in text:
        return "Choose the models."
    return text
