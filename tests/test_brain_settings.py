"""The Brain window's API (P5.7f): provider choice, keys in .env only, connection test and tool
probe, saving with reload and revert, prefs precedence, Ollama helpers, Claude API key setup.
No network: every HTTP call goes to an ``httpx.MockTransport``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from spark.brain import connector_store as store
from spark.config import Settings, load_settings
from spark.desktop import brain_settings as bs
from spark.desktop import ollama
from spark.desktop.brain_settings import BrainSettingsApi
from spark.prefs import Prefs, apply_prefs, reset_brain
from tests.openai_fakes import finish_chunk, reply, sse, tool_chunk

KEY = "sk-test-0123456789abcdefWXYZ"
ANTHROPIC = {"ANTHROPIC_API_KEY": "sk-ant-test-0123456789abcd"}


# --- helpers -------------------------------------------------------------------------------------


class Endpoint:
    """A fake OpenAI-compatible server: GET /models and POST /chat/completions."""

    def __init__(
        self,
        *,
        models=("gpt-4.1", "gpt-4.1-mini"),
        calls_tool=True,
        status=None,
        chat_status=None,
        raise_exc=None,
    ):
        self.models = list(models)
        self.calls_tool = calls_tool
        self.status = status  # for /models
        self.chat_status = chat_status
        self.raise_exc = raise_exc
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        if request.url.path.endswith("/models"):
            if self.status:
                return httpx.Response(self.status, json={"error": {"message": "nope"}})
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if self.chat_status:
            return httpx.Response(self.chat_status, json={"error": {"message": "no"}})
        if self.calls_tool:
            body = sse(
                tool_chunk(0, id="c1", name="get_current_time", args="{}"),
                finish_chunk("tool_calls"),
            )
        else:
            body = reply("It is noon.")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def make_api(tmp_path: Path, endpoint=None, **kw):
    settings = Settings(config_dir=tmp_path)
    applied: list[int] = []
    errors = kw.pop("errors", [])

    def on_apply():
        applied.append(1)
        return errors.pop(0) if errors else None

    environ = kw.pop("environ", {})
    kw.setdefault("find_cli", lambda s: "C:/Users/x/.local/bin/claude.exe")
    kw.setdefault("auth_status", lambda cli: {"signed_in": False, "method": None, "plan": None})
    api = BrainSettingsApi(
        settings,
        on_apply=on_apply,
        environ=environ,
        transport=httpx.MockTransport(endpoint or Endpoint()),
        **kw,
    )
    return api, settings, applied, environ


OPENAI_FORM = {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "fast_model": "gpt-4.1-mini",
    "deep_model": "gpt-4.1",
    "worker_model": "",
}


# --- surface -------------------------------------------------------------------------------------


def test_api_exposes_only_its_methods(tmp_path):
    api, *_ = make_api(tmp_path)
    public = {n for n in dir(api) if not n.startswith("_")}
    assert public == {
        "state",
        "claude_status",
        "test",
        "save",
        "use_claude",
        "revert",
        "remove_key",
        "open_link",  # only provider key / download pages
        "ollama_status",
        "ollama_pull",
        "ollama_progress",
        "ollama_cancel",
    }


def test_offers_every_provider_with_claude_first(tmp_path):
    api, *_ = make_api(tmp_path)
    state = api.state()
    ids = [p["id"] for p in state["providers"]]
    assert ids == [
        "claude",
        "hermes",
        "openai",
        "azure",
        "openrouter",
        "groq",
        "together",
        "ollama",
        "lmstudio",
        "custom",
    ]
    assert state["active"] == "claude"
    ollama_view = next(p for p in state["providers"] if p["id"] == "ollama")
    assert ollama_view["url_editable"] and not ollama_view["needs_key"]
    openai_view = next(p for p in state["providers"] if p["id"] == "openai")
    assert not openai_view["url_editable"] and openai_view["needs_key"]
    assert openai_view["suggested"]["deep_model"]
    assert openai_view["key"] == {
        "env": "OPENAI_API_KEY",
        "saved": False,
        "where": None,
        "last4": "",
    }


def test_open_link_only_opens_known_provider_pages(tmp_path):
    opened: list[str] = []
    api, *_ = make_api(tmp_path, open_url=opened.append)
    assert api.open_link("https://platform.openai.com/api-keys")["ok"]
    assert api.open_link("https://ollama.com/download")["ok"]
    for bad in (
        "https://evil.example/api-keys",
        "file:///C:/Windows/system32/calc.exe",
        "https://platform.openai.com/api-keys?x=1",
        "javascript:alert(1)",
    ):
        assert not api.open_link(bad)["ok"]
    assert opened == ["https://platform.openai.com/api-keys", "https://ollama.com/download"]


# --- test connection -----------------------------------------------------------------------------


def test_connection_test_lists_models_and_sees_a_tool_call(tmp_path):
    endpoint = Endpoint()
    api, *_ = make_api(tmp_path, endpoint)
    result = api.test({**OPENAI_FORM, "key": KEY})
    assert result["ok"] and result["tools"] == "yes"
    assert result["models"] == ["gpt-4.1", "gpt-4.1-mini"]
    assert [s["level"] for s in result["steps"]] == ["ok", "ok"]
    assert "key works" in result["steps"][0]["text"]
    assert KEY not in json.dumps(result)
    # The typed key went in the header only; the probe offered one tool with tool_choice auto.
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in endpoint.requests)
    body = json.loads(endpoint.requests[1].content)
    assert body["model"] == "gpt-4.1" and body["tool_choice"] == "auto"
    assert [t["function"]["name"] for t in body["tools"]] == ["get_current_time"]
    assert body["max_completion_tokens"] <= 256


def test_connection_test_warns_when_the_model_does_not_call_tools(tmp_path):
    api, *_ = make_api(tmp_path, Endpoint(calls_tool=False))
    result = api.test({**OPENAI_FORM, "key": KEY})
    assert result["ok"] and result["tools"] == "no"
    assert result["steps"][-1]["level"] == "warn"
    assert "connectors and timers won't work well" in result["steps"][-1]["text"]


@pytest.mark.parametrize(
    ("endpoint", "provider", "expected"),
    [
        (Endpoint(status=401), "openai", "didn't accept the API key (HTTP 401)"),
        (Endpoint(status=403), "openai", "refused the request (HTTP 403)"),
        (Endpoint(status=404), "openai", "Nothing answered at that address (HTTP 404)"),
        (Endpoint(status=429), "openai", "too many requests (HTTP 429)"),
        (Endpoint(chat_status=404), "openai", "doesn't have a model called “gpt-4.1”"),
        (
            Endpoint(raise_exc=httpx.ConnectError("refused")),
            "ollama",
            "Ollama isn't running. Start it from the Start menu",
        ),
        (
            Endpoint(raise_exc=httpx.ConnectError("refused")),
            "lmstudio",
            "LM Studio's server isn't running",
        ),
        (Endpoint(raise_exc=httpx.ConnectError("refused")), "openai", "Can't reach OpenAI"),
        (Endpoint(raise_exc=httpx.ReadTimeout("slow")), "openai", "didn't answer in time"),
    ],
)
def test_connection_test_explains_failures_in_plain_words(tmp_path, endpoint, provider, expected):
    api, *_ = make_api(tmp_path, endpoint)
    form = {**OPENAI_FORM, "provider": provider, "base_url": ""}
    if provider != "openai":
        form.update(fast_model="qwen2.5:7b", deep_model="qwen2.5:7b")
    result = api.test({**form, "key": KEY if provider == "openai" else ""})
    assert not result["ok"]
    assert expected in result["steps"][-1]["text"], result
    assert KEY not in json.dumps(result)


HERMES_FORM = {"provider": "hermes", "base_url": "", "fast_model": "", "deep_model": ""}


def test_hermes_is_offered_as_an_agent_with_its_api_server_defaults(tmp_path):
    api, *_ = make_api(tmp_path)
    view = next(p for p in api.state()["providers"] if p["id"] == "hermes")
    assert view["kind"] == "agent" and view["label"] == "Hermes Agent"
    assert view["default_base_url"] == "http://127.0.0.1:8642/v1"
    assert view["url_editable"] and view["needs_key"]
    assert view["suggested"] == {"fast_model": "hermes-agent", "deep_model": "hermes-agent"}
    assert view["link"] in bs.LINKS


def test_hermes_connection_test_skips_the_tool_probe(tmp_path):
    endpoint = Endpoint(models=("hermes-agent",))
    api, *_ = make_api(tmp_path, endpoint)
    result = api.test({**HERMES_FORM, "key": KEY})
    assert result["ok"] and result["tools"] == "agent", result
    assert result["models"] == ["hermes-agent"]
    assert "its own tools" in result["steps"][-1]["text"]
    # Only GET /models: no chat request, which would start a whole Hermes agent turn.
    assert [r.url.path for r in endpoint.requests] == ["/v1/models"]
    assert str(endpoint.requests[0].url).startswith("http://127.0.0.1:8642/v1")
    assert endpoint.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_hermes_needs_its_key_and_explains_a_stopped_server(tmp_path):
    api, *_ = make_api(tmp_path, Endpoint(raise_exc=httpx.ConnectError("refused")))
    result = api.test(HERMES_FORM)
    assert not result["ok"] and "Enter your Hermes Agent API key" in result["steps"][0]["text"]
    result = api.test({**HERMES_FORM, "key": KEY})
    assert not result["ok"]
    assert (
        "Can't reach Hermes Agent's API server at http://127.0.0.1:8642/v1"
        in (result["steps"][-1]["text"])
    )
    assert "hermes gateway" in result["steps"][-1]["text"]


def test_hermes_on_a_server_uses_the_address_typed(tmp_path):
    endpoint = Endpoint(models=("hermes-agent",))
    api, settings, *_ = make_api(tmp_path, endpoint)
    url = "https://hermes.example.com/v1"
    result = api.test({**HERMES_FORM, "base_url": url, "key": KEY})
    assert result["ok"], result
    assert str(endpoint.requests[0].url) == f"{url}/models"
    # Saved: the key goes to .env as HERMES_API_KEY; the server's address is kept.
    result = api.save({**HERMES_FORM, "base_url": url, "key": KEY})
    assert result["ok"], result
    assert store.read_env_file(tmp_path / ".env")["HERMES_API_KEY"] == KEY
    oc = settings.brain.openai_compat
    assert (oc.provider, oc.base_url, oc.deep_model) == ("hermes", url, "hermes-agent")
    assert result["state"]["active"] == "hermes"


def test_connection_test_without_a_key_asks_for_one_and_sends_nothing(tmp_path):
    endpoint = Endpoint()
    api, *_ = make_api(tmp_path, endpoint)
    result = api.test(OPENAI_FORM)
    assert not result["ok"] and "Enter your OpenAI API key" in result["steps"][0]["text"]
    assert endpoint.requests == []


def test_connection_test_flags_a_model_the_endpoint_does_not_list(tmp_path):
    api, *_ = make_api(tmp_path, Endpoint(models=["llama3"]))
    result = api.test({**OPENAI_FORM, "key": KEY})
    assert (
        not result["ok"] and "doesn't list a model called “gpt-4.1”" in result["steps"][-1]["text"]
    )
    assert result["models"] == ["llama3"]


def test_ollama_without_models_says_to_download_one(tmp_path):
    api, *_ = make_api(tmp_path, Endpoint(models=[]))
    result = api.test(
        {
            "provider": "ollama",
            "base_url": "http://localhost:11434/v1",
            "deep_model": "qwen2.5:7b",
            "fast_model": "qwen2.5:7b",
        }
    )
    assert not result["ok"] and "no models yet" in result["steps"][-1]["text"]


def test_custom_needs_an_address(tmp_path):
    api, *_ = make_api(tmp_path)
    result = api.test({"provider": "custom", "base_url": "", "deep_model": "m"})
    assert not result["ok"] and "Enter the address" in result["steps"][0]["text"]
    assert not api.save({"provider": "custom", "base_url": "ftp://x", "deep_model": "m"})["ok"]


# --- saving: key in .env only, prefs, reload, revert ---------------------------------------------


def test_save_puts_the_key_in_env_only_and_applies(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    api, settings, applied, environ = make_api(tmp_path)
    result = api.save({**OPENAI_FORM, "key": KEY})
    assert result["ok"], result
    assert applied == [1]
    # .env (in tmp, next to config.yaml) has it under the preset's variable; so does the process.
    env_file = tmp_path / ".env"
    assert store.read_env_file(env_file)["OPENAI_API_KEY"] == KEY
    assert environ["OPENAI_API_KEY"] == KEY
    # ...and nothing else does.
    prefs_text = Path(settings.path(settings.app.prefs_file)).read_text(encoding="utf-8")
    assert KEY not in prefs_text
    assert KEY not in json.dumps(result)
    assert KEY not in json.dumps(api.state())
    assert KEY not in caplog.text
    view = next(p for p in result["state"]["providers"] if p["id"] == "openai")
    assert view["key"] == {"env": "OPENAI_API_KEY", "saved": True, "where": ".env", "last4": "WXYZ"}
    # The live settings now use the choice; prefs hold it for the next start.
    assert settings.brain.backend == "openai_compat"
    oc = settings.brain.openai_compat
    assert (oc.provider, oc.deep_model, oc.fast_model) == ("openai", "gpt-4.1", "gpt-4.1-mini")
    prefs = json.loads(prefs_text)
    assert prefs["brain_backend"] == "openai_compat" and prefs["openai_provider"] == "openai"
    assert prefs["openai_base_url"] == ""  # the preset's
    assert result["state"]["active"] == "openai"


def test_replacing_a_key_rewrites_its_line_and_keeps_the_rest(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# mine\nOTHER=1\nOPENAI_API_KEY=sk-old-key-000000000000\n", "utf-8")
    api, *_ = make_api(tmp_path)
    assert api.save({**OPENAI_FORM, "key": KEY})["ok"]
    values = store.read_env_file(env_file)
    assert values == {"OTHER": "1", "OPENAI_API_KEY": KEY}
    assert env_file.read_text("utf-8").startswith("# mine\nOTHER=1\n")
    assert "sk-old" not in env_file.read_text("utf-8")


def test_remove_key(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\nOPENAI_API_KEY={KEY}\n", "utf-8")
    api, _s, _a, environ = make_api(tmp_path, environ={"OPENAI_API_KEY": KEY})
    result = api.remove_key("openai")
    assert result["ok"] and result["key"]["saved"] is False
    assert store.read_env_file(env_file) == {"OTHER": "1"}
    assert "OPENAI_API_KEY" not in environ


def test_a_key_from_the_windows_environment_is_shown_but_not_removed(tmp_path):
    api, *_ = make_api(tmp_path, environ={"GROQ_API_KEY": "gsk_environment_key_1234"})
    view = next(p for p in api.state()["providers"] if p["id"] == "groq")
    assert view["key"]["where"] == "environment" and view["key"]["last4"] == "1234"
    result = api.remove_key("groq")
    assert not result["ok"] and "environment variables" in result["error"]


def test_save_refuses_a_cloud_provider_without_a_key(tmp_path):
    api, _s, applied, _e = make_api(tmp_path)
    result = api.save(OPENAI_FORM)
    assert not result["ok"] and "API key" in result["error"] and applied == []


def test_failed_apply_reports_and_revert_goes_back(tmp_path):
    api, settings, applied, _e = make_api(tmp_path, errors=["OpenAI refused the API key."])
    result = api.save({**OPENAI_FORM, "key": KEY})
    assert not result["ok"] and result["error"] == "OpenAI refused the API key."
    assert result["state"]["can_revert"]
    assert settings.brain.backend == "openai_compat"
    back = api.revert()
    assert back["ok"] and back["message"] == "Spark now thinks with Claude."
    assert applied == [1, 1]
    assert settings.brain.backend == "claude_agent"  # config.yaml's (no prefs left)
    prefs = Prefs.from_settings(settings)
    assert prefs.get("brain_backend") is None and prefs.get("brain_previous") is None
    assert not api.revert()["ok"]  # nothing further back


def test_a_failed_setup_is_not_what_revert_returns_to(tmp_path):
    api, settings, _a, _e = make_api(tmp_path, errors=["broken", "broken again"])
    assert not api.save({**OPENAI_FORM, "key": KEY})["ok"]
    assert not api.save({"provider": "groq", "deep_model": "llama", "key": "gsk_" + "x" * 20})["ok"]
    assert api.revert()["ok"]
    assert settings.brain.backend == "claude_agent"


def test_switch_back_to_claude_is_one_call(tmp_path):
    api, settings, applied, _e = make_api(tmp_path, environ=dict(ANTHROPIC))
    assert api.save({**OPENAI_FORM, "key": KEY})["ok"]
    result = api.use_claude()
    assert result["ok"] and settings.brain.backend == "claude_agent"
    assert result["state"]["active"] == "claude" and applied == [1, 1]
    # The OpenAI choice is remembered for next time the user picks it.
    view = next(p for p in result["state"]["providers"] if p["id"] == "openai")
    assert view["values"]["deep_model"] == "gpt-4.1"


def test_claude_save_keeps_claude_models_apart(tmp_path):
    api, settings, *_ = make_api(tmp_path, environ=dict(ANTHROPIC))
    assert api.save(
        {
            "provider": "claude",
            "fast_model": "claude-haiku-4-5-20251001",
            "deep_model": "claude-sonnet-5",
            "worker_model": "",
        }
    )["ok"]
    assert settings.brain.deep_model == "claude-sonnet-5" and settings.brain.worker_model is None
    assert not api.save({"provider": "claude", "deep_model": "bad name; rm -rf"})["ok"]


def test_local_provider_saves_its_address(tmp_path):
    api, settings, *_ = make_api(tmp_path)
    form = {
        "provider": "lmstudio",
        "base_url": "http://localhost:4321/v1",
        "deep_model": "qwen",
        "fast_model": "",
        "worker_model": "",
    }
    assert api.save(form)["ok"]
    oc = settings.brain.openai_compat
    assert (oc.provider, oc.base_url, oc.fast_model) == (
        "lmstudio",
        "http://localhost:4321/v1",
        "qwen",
    )


def test_azure_keeps_its_api_version(tmp_path):
    api, settings, *_ = make_api(tmp_path)
    form = {
        "provider": "azure",
        "base_url": "https://r.openai.azure.com/openai/deployments/{model}",
        "api_version": "2024-10-21",
        "deep_model": "my-gpt",
        "key": "az-key-0000000000000",
    }
    assert api.save(form)["ok"]
    oc = settings.brain.openai_compat
    assert oc.api_version == "2024-10-21" and oc.provider == "azure"
    assert store.read_env_file(tmp_path / ".env")["AZURE_OPENAI_API_KEY"].startswith("az-key")


def test_repo_env_file_is_never_written(tmp_path):
    """Settings() points at the repo; conftest sends .env writes to tmp instead."""
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    before = repo_env.read_bytes() if repo_env.exists() else None
    api = BrainSettingsApi(Settings(), environ={}, on_apply=lambda: None)
    assert api.save({**OPENAI_FORM, "key": KEY})["ok"]
    after = repo_env.read_bytes() if repo_env.exists() else None
    assert before == after
    assert store.read_env_file(tmp_path / ".env")["OPENAI_API_KEY"] == KEY


# --- prefs precedence --------------------------------------------------------------------------


CONFIG = """
brain:
  backend: claude_agent
  fast_model: claude-haiku-4-5-20251001
  openai_compat:
    provider: ollama
    base_url: http://localhost:9999/v1
    api_key_env: ""
    fast_model: qwen2.5:7b
    deep_model: qwen2.5:14b
"""


def _load(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG, "utf-8")
    return load_settings(tmp_path / "config.yaml", load_env_file=False)


def test_without_prefs_config_yaml_decides(tmp_path):
    settings = _load(tmp_path)
    apply_prefs(settings, Prefs.from_settings(settings))
    oc = settings.brain.openai_compat
    assert settings.brain.backend == "claude_agent"
    assert (oc.provider, oc.base_url, oc.deep_model) == (
        "ollama",
        "http://localhost:9999/v1",
        "qwen2.5:14b",
    )


def test_prefs_override_backend_provider_url_and_models(tmp_path):
    settings = _load(tmp_path)
    Prefs.from_settings(settings).update(
        brain_backend="openai_compat",
        openai_provider="openai",
        openai_base_url="",
        openai_api_version="",
        openai_fast_model="gpt-4.1-mini",
        openai_deep_model="gpt-4.1",
        openai_worker_model="",
    )
    apply_prefs(settings, Prefs.from_settings(settings))
    oc = settings.brain.openai_compat
    assert settings.brain.backend == "openai_compat"
    assert (oc.provider, oc.base_url, oc.fast_model, oc.deep_model, oc.worker_model) == (
        "openai",
        None,
        "gpt-4.1-mini",
        "gpt-4.1",
        None,
    )
    # config.yaml's key variable belonged to Ollama ("" = no key); OpenAI gets its preset's.
    assert oc.api_key_env is None
    from spark.brain.openai_presets import resolve_provider

    assert resolve_provider(oc).key_env == "OPENAI_API_KEY"


def test_bad_prefs_values_are_ignored(tmp_path):
    settings = _load(tmp_path)
    Prefs.from_settings(settings).update(brain_backend="nonsense", openai_provider="evil")
    apply_prefs(settings, Prefs.from_settings(settings))
    assert settings.brain.backend == "claude_agent"
    assert settings.brain.openai_compat.provider == "ollama"


def test_reset_brain_lets_removed_prefs_fall_back_to_config(tmp_path):
    settings = _load(tmp_path)
    baseline = settings.brain.model_copy(deep=True)
    prefs = Prefs.from_settings(settings)
    prefs.update(brain_backend="openai_compat", openai_provider="groq", openai_base_url="")
    apply_prefs(settings, prefs)
    assert settings.brain.openai_compat.provider == "groq"
    prefs.delete("brain_backend", "openai_provider", "openai_base_url")
    reset_brain(settings, baseline)
    apply_prefs(settings, Prefs.from_settings(settings))
    oc = settings.brain.openai_compat
    assert settings.brain.backend == "claude_agent"
    assert (oc.provider, oc.base_url, oc.api_key_env) == ("ollama", "http://localhost:9999/v1", "")


def test_prefs_update_keeps_keys_saved_by_another_instance(tmp_path):
    a = Prefs(tmp_path / "p.json")
    b = Prefs(tmp_path / "p.json")
    a.update(brain_backend="openai_compat")
    b.update(model_mode="fast")  # b was read before a's save
    assert Prefs(tmp_path / "p.json").data == {
        "brain_backend": "openai_compat",
        "model_mode": "fast",
    }


# --- Claude: the installed Claude Code, signed in or with an Anthropic API key ----------------

SIGNED_IN = {"signed_in": True, "method": "claude.ai", "plan": "Max plan"}


def test_claude_key_is_optional(tmp_path):
    api, *_ = make_api(tmp_path)
    view = next(p for p in api.state()["providers"] if p["id"] == "claude")
    assert not view["needs_key"] and view["takes_key"]
    assert view["key"] == {"env": "ANTHROPIC_API_KEY", "saved": False, "where": None, "last4": ""}
    assert "signed in" in view["description"] and "API key" in view["description"]


def test_parse_auth_status_keeps_only_the_sign_in():
    text = json.dumps(
        {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "email": "someone@example.com",
            "orgId": "0000",
            "subscriptionType": "max",
        }
    )
    assert bs.parse_auth_status(text) == SIGNED_IN
    assert not bs.parse_auth_status("not json")["signed_in"]
    assert not bs.parse_auth_status('{"loggedIn": false}')["signed_in"]


def test_claude_with_the_claude_code_sign_in_and_no_key(tmp_path):
    api, settings, applied, environ = make_api(tmp_path, auth_status=lambda cli: SIGNED_IN)
    status = api.claude_status()
    assert status["signed_in"] and status["plan"] == "Max plan"
    assert "signed in with your Claude account (Max plan)" in status["sign_in"]
    test = api.test({"provider": "claude"})
    assert test["ok"] and "Spark will use it" in test["steps"][-1]["text"]
    result = api.save({"provider": "claude"})
    assert result["ok"] and settings.brain.backend == "claude_agent" and applied == [1]
    assert "ANTHROPIC_API_KEY" not in environ


def test_claude_save_without_sign_in_or_key_says_how_to_fix(tmp_path):
    api, _settings, applied, _environ = make_api(tmp_path)
    result = api.save({"provider": "claude"})
    assert not result["ok"] and "/login" in result["error"] and applied == []
    test = api.test({"provider": "claude"})
    assert not test["ok"] and "/login" in test["steps"][-1]["text"]


def test_claude_save_stores_an_optional_key_in_env(tmp_path):
    api, settings, _applied, environ = make_api(tmp_path)
    secret = "sk-ant-api03-0123456789abcdefWXYZ"
    result = api.save({"provider": "claude", "key": secret})
    assert result["ok"] and settings.brain.backend == "claude_agent"
    assert environ["ANTHROPIC_API_KEY"] == secret
    assert store.read_env_file(store.env_path(settings))["ANTHROPIC_API_KEY"] == secret
    view = next(p for p in result["state"]["providers"] if p["id"] == "claude")
    assert view["key"]["saved"] and view["key"]["last4"] == "WXYZ"
    assert secret not in repr(result)
    assert api.remove_key("claude")["ok"]
    assert "ANTHROPIC_API_KEY" not in store.read_env_file(store.env_path(settings))


def test_claude_test_checks_claude_code_and_the_key(tmp_path):
    seen: list[httpx.Request] = []

    def anthropic(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        ok = request.headers.get("x-api-key") == "sk-ant-good-0123456789"
        return httpx.Response(200 if ok else 401, json={"data": []})

    api, *_ = make_api(tmp_path, endpoint=anthropic)
    good = api.test({"provider": "claude", "key": "sk-ant-good-0123456789"})
    assert good["ok"], good
    assert seen[-1].url == httpx.URL("https://api.anthropic.com/v1/models")
    assert "anthropic-version" in seen[-1].headers
    bad = api.test({"provider": "claude", "key": "sk-ant-bad-0123456789"})
    assert not bad["ok"] and "refused" in bad["steps"][-1]["text"]
    none = api.test({"provider": "claude"})
    assert not none["ok"] and "Anthropic API key" in none["steps"][-1]["text"]  # not signed in


def test_claude_status_reports_a_missing_claude_code(tmp_path):
    api, *_ = make_api(tmp_path, find_cli=lambda s: None, environ=dict(ANTHROPIC))
    status = api.claude_status()
    assert not status["installed"] and "install" in status["message"].lower()
    result = api.test({"provider": "claude"})
    assert not result["ok"] and result["steps"][0]["level"] == "error"
    view = next(p for p in api.state()["providers"] if p["id"] == "claude")
    assert view["install_link"] in bs.LINKS
    opened: list[str] = []
    api2, *_ = make_api(tmp_path, open_url=opened.append)
    assert api2.open_link(view["install_link"])["ok"] and opened == [view["install_link"]]


# --- Ollama ------------------------------------------------------------------------------------


def _tags(models):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": models})

    return httpx.MockTransport(handler)


def test_ollama_status_running_with_models(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, "free_bytes", lambda p: 100 * ollama.GB)
    out = ollama.status(
        "http://localhost:11434/v1",
        transport=_tags([{"name": "qwen2.5:7b", "size": 4_700_000_000}]),
        environ={"OLLAMA_MODELS": str(tmp_path)},
    )
    assert out["running"] and out["installed"]
    assert out["models"] == [{"name": "qwen2.5:7b", "size_gb": 4.7}]
    assert out["disk"] == {"path": str(tmp_path), "free_gb": 100.0}
    rec = {m["name"]: m for m in out["recommended"]}
    assert rec["qwen2.5:7b"]["installed"] and not rec["llama3.1:8b"]["installed"]
    assert not any(m["low_space"] for m in out["recommended"])


def test_ollama_status_warns_when_space_is_low(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama, "free_bytes", lambda p: 6 * ollama.GB)
    out = ollama.status(None, transport=_tags([]), environ={"OLLAMA_MODELS": str(tmp_path)})
    low = {m["name"]: m["low_space"] for m in out["recommended"]}
    assert low == {"qwen2.5:7b": True, "llama3.1:8b": True, "qwen2.5:3b": False}


def test_ollama_status_not_running_or_installed(monkeypatch):
    def refused(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama, "find_ollama", lambda environ=None: None)
    out = ollama.status(None, transport=httpx.MockTransport(refused), environ={})
    assert not out["running"] and not out["installed"]
    assert out["download_url"] == "https://ollama.com/download"


def test_api_root():
    assert ollama.api_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert ollama.api_root("http://box:11434/v1/") == "http://box:11434"
    assert ollama.api_root(None) == "http://localhost:11434"


PULL_LINES = [
    {"status": "pulling manifest"},
    {"status": "pulling abc", "digest": "sha256:abc", "total": 1000, "completed": 0},
    {"status": "pulling abc", "digest": "sha256:abc", "total": 1000, "completed": 600},
    {"status": "pulling def", "digest": "sha256:def", "total": 1000, "completed": 1000},
    {"status": "pulling abc", "digest": "sha256:abc", "total": 1000, "completed": 1000},
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "success"},
]


def test_pull_progress_parsing():
    p = ollama.PullProgress("qwen2.5:7b")
    for item in PULL_LINES[:3]:
        p.feed_line(json.dumps(item))
    snap = p.snapshot()
    assert (
        snap["state"] == "running" and snap["percent"] == 60.0 and snap["status"] == "Downloading…"
    )
    for item in PULL_LINES[3:6]:
        p.feed_line(json.dumps(item))
    assert p.snapshot()["percent"] == 100.0 and p.status == "Verifying sha256 digest"
    p.feed_line("")
    p.feed_line("garbage")
    for item in PULL_LINES[6:]:
        p.feed_line(json.dumps(item))
    assert p.snapshot()["state"] == "done"
    err = ollama.PullProgress("nope")
    err.feed({"error": "pull model manifest: file does not exist"})
    assert err.state == "error" and "file does not exist" in err.error


def _pull_transport(lines, seen, before_line=None):
    def handler(request):
        seen.append(json.loads(request.content))

        def stream():
            for i, item in enumerate(lines):
                if before_line is not None:
                    before_line(i)
                yield (json.dumps(item) + "\n").encode()

        return httpx.Response(200, content=stream())

    return httpx.MockTransport(handler)


def test_pull_downloads_only_when_asked_and_reports_done(tmp_path):
    seen: list[dict] = []
    api, *_ = make_api(tmp_path, ollama_transport=_pull_transport(PULL_LINES, seen))
    assert api.ollama_progress() == {"state": "idle"}
    assert seen == []  # nothing until the user clicks
    assert api.ollama_pull("qwen2.5:7b", "http://localhost:11434/v1")["ok"]
    api._puller._thread.join(5)
    assert seen == [{"model": "qwen2.5:7b", "stream": True}]
    progress = api.ollama_progress()
    assert progress["state"] == "done" and progress["percent"] == 100.0


def test_pull_can_be_cancelled(tmp_path):
    seen: list[dict] = []
    puller = ollama.Puller()

    def cancel_midway(i):
        if i == 3:
            puller.cancel()

    puller._transport = _pull_transport(PULL_LINES, seen, cancel_midway)
    assert puller.start("http://localhost:11434/v1", "qwen2.5:7b", wait=True)
    progress = puller.progress()
    assert progress["state"] == "cancelled" and progress["percent"] < 100
    # A new download may start after a cancel.
    puller._transport = _pull_transport(PULL_LINES, seen)
    assert puller.start(None, "qwen2.5:7b", wait=True)
    assert puller.progress()["state"] == "done"


def test_pull_rejects_odd_names_and_reports_server_errors(tmp_path):
    def handler(request):
        return httpx.Response(500, content=b'{"error": "disk full"}')

    api, *_ = make_api(tmp_path, ollama_transport=httpx.MockTransport(handler))
    assert not api.ollama_pull("qwen; rm -rf /")["ok"]
    assert api.ollama_pull("qwen2.5:7b")["ok"]
    api._puller._thread.join(5)
    progress = api.ollama_progress()
    assert progress["state"] == "error" and "disk full" in progress["error"]


def test_pull_stream_that_ends_early_is_an_error():
    puller = ollama.Puller(transport=_pull_transport(PULL_LINES[:3], []))
    puller.start(None, "qwen2.5:7b", wait=True)
    assert puller.progress()["state"] == "error"


def test_a_backend_given_on_the_command_line_beats_prefs(tmp_path):
    from spark.__main__ import build_parser, load

    (tmp_path / "config.yaml").write_text(CONFIG, "utf-8")
    args = build_parser().parse_args(["--config", str(tmp_path / "config.yaml"), "--brain", "mock"])
    settings = load(args)
    Prefs.from_settings(settings).update(brain_backend="openai_compat")
    apply_prefs(settings, Prefs.from_settings(settings))
    assert settings.brain.backend == "mock"
