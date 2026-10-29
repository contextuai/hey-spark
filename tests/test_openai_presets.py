"""Provider presets, the brain.openai_compat config, and backend-aware model prefs (P5.7e)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spark.brain.openai_presets import PRESETS, resolve_provider, resolved_models
from spark.config import OpenAICompatConfig, Settings
from spark.prefs import (
    Prefs,
    apply_prefs,
    model_choices,
    model_info,
    pref_updates,
    route_models_cfg,
    set_route_models,
)


def oc(**kw) -> OpenAICompatConfig:
    return OpenAICompatConfig(**kw)


@pytest.mark.parametrize(
    ("name", "url", "key"),
    [
        ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        ("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        ("ollama", "http://localhost:11434/v1", None),
        ("lmstudio", "http://localhost:1234/v1", None),
        ("hermes", "http://127.0.0.1:8642/v1", "HERMES_API_KEY"),
        ("vllm", "http://localhost:8000/v1", "VLLM_API_KEY"),
    ],
)
def test_presets_cover_base_url_and_key_env(name, url, key):
    prov = resolve_provider(oc(provider=name, fast_model="m"))
    assert prov.base_url == url
    assert prov.key_env == key
    assert not prov.problems


def test_local_presets_need_no_key_and_say_is_it_running():
    for name in ("ollama", "lmstudio", "vllm"):
        prov = resolve_provider(oc(provider=name, fast_model="m"))
        assert prov.local
        assert prov.missing_env({}) == []
    assert not resolve_provider(oc(provider="openai")).local


def test_explicit_fields_override_the_preset():
    prov = resolve_provider(
        oc(
            provider="openai",
            base_url="https://proxy.example/v1/",
            api_key_env="MY_KEY",
            fast_model="small",
            deep_model="big",
            worker_model="tiny",
        )
    )
    assert prov.url("chat/completions") == "https://proxy.example/v1/chat/completions"
    assert prov.key_env == "MY_KEY"
    assert (prov.fast_model, prov.deep_model, prov.worker_model) == ("small", "big", "tiny")


def test_suggested_models_fill_in_and_one_name_serves_both_routes():
    prov = resolve_provider(oc(provider="ollama"))
    assert prov.fast_model == PRESETS["ollama"].fast_model
    assert prov.deep_model == PRESETS["ollama"].deep_model
    one = resolve_provider(oc(provider="lmstudio", deep_model="loaded-model"))
    assert one.fast_model == one.deep_model == "loaded-model"
    none = resolve_provider(oc(provider="lmstudio"))
    assert any("fast_model" in p for p in none.problems)


def test_bearer_auth_and_missing_key():
    prov = resolve_provider(oc(provider="openai"))
    assert prov.missing_env({}) == ["OPENAI_API_KEY"]
    headers = prov.request_headers({"OPENAI_API_KEY": "sk-test"})
    assert headers["Authorization"] == "Bearer sk-test"
    assert prov.request_headers({}) == {}  # no key, no header (and nothing to log)


def test_azure_uses_the_api_key_header_api_version_and_deployment_url():
    prov = resolve_provider(
        oc(
            provider="azure",
            base_url="https://res.openai.azure.com/openai/deployments/{model}",
            api_version="2024-10-21",
            fast_model="my-deploy",
        )
    )
    headers = prov.request_headers({"AZURE_OPENAI_API_KEY": "az"})
    assert headers == {"api-key": "az"}
    assert "Authorization" not in headers
    assert prov.params() == {"api-version": "2024-10-21"}
    assert prov.url("chat/completions", "my-deploy") == (
        "https://res.openai.azure.com/openai/deployments/my-deploy/chat/completions"
    )
    assert any("base_url" in p for p in resolve_provider(oc(provider="azure")).problems)


def test_extra_headers_expand_vars_and_count_as_required():
    prov = resolve_provider(oc(provider="ollama", extra_headers={"X-Proxy": "${PROXY_TOKEN}"}))
    assert prov.missing_env({}) == ["PROXY_TOKEN"]
    assert prov.request_headers({"PROXY_TOKEN": "t"})["X-Proxy"] == "t"


def test_unknown_provider_is_rejected_by_config():
    with pytest.raises(ValidationError):
        OpenAICompatConfig(provider="bogus")


def test_resolved_models_follow_the_backend():
    claude = Settings(brain={"fast_model": "claude-a", "deep_model": "claude-b"})
    assert resolved_models(claude) == ("claude-a", "claude-b", None)
    other = Settings(
        brain={
            "backend": "openai_compat",
            "fast_model": "claude-a",
            "openai_compat": {"provider": "groq", "deep_model": "big"},
        }
    )
    fast, deep, worker = resolved_models(other)
    assert (fast, deep, worker) == (PRESETS["groq"].fast_model, "big", None)


def test_model_info_and_choices_on_openai_compat_report_its_names():
    s = Settings(
        brain={
            "backend": "openai_compat",
            "openai_compat": {"provider": "ollama", "fast_model": "llama3.1:8b"},
        }
    )
    info = model_info(s)
    assert info["fast_model"] == "llama3.1:8b"
    assert info["fast_label"] == "llama3.1:8b"
    assert info["provider"] == "Ollama"
    ids = [c[0] for c in model_choices(s, ["served-a", 3])]
    assert ids[0] == "llama3.1:8b"
    assert "served-a" in ids
    assert not any(i.startswith("claude-") for i in ids)
    claude_ids = [c[0] for c in model_choices(Settings())]
    assert "claude-opus-5-5" in claude_ids


def test_model_prefs_are_kept_per_backend(tmp_path):
    s = Settings(brain={"backend": "openai_compat"})
    assert route_models_cfg(s) is s.brain.openai_compat
    set_route_models(s, "f", "d", "")
    assert (s.brain.openai_compat.fast_model, s.brain.openai_compat.deep_model) == ("f", "d")
    assert s.brain.fast_model.startswith("claude-")  # the Claude ids are untouched
    prefs = Prefs(tmp_path / "prefs.json")
    prefs.update(**pref_updates(s, "f", "d", ""), fast_model="claude-x")
    assert prefs.get("openai_fast_model") == "f"
    fresh = Settings(brain={"backend": "openai_compat"})
    apply_prefs(fresh, Prefs(tmp_path / "prefs.json"))
    assert fresh.brain.openai_compat.fast_model == "f"
    assert fresh.brain.openai_compat.worker_model is None
    assert fresh.brain.fast_model == "claude-x"


# --- doctor -------------------------------------------------------------------------------------


async def test_doctor_openai_compat_checks():
    import httpx

    from spark import doctor
    from tests.openai_fakes import Script

    assert await doctor.check_openai_compat(Settings()) == []  # Claude: nothing to check
    s = Settings(brain={"backend": "openai_compat", "openai_compat": {"provider": "openai"}})
    missing = await doctor.check_openai_compat(s, environ={})
    assert not missing[-1].ok and "OPENAI_API_KEY" in missing[-1].detail
    env = {"OPENAI_API_KEY": "sk-secret-value"}
    script = Script(models=["gpt-4.1-mini"])
    checks = await doctor.check_openai_compat(s, transport=script.transport(), environ=env)
    by_name = {c.name: c for c in checks}
    assert by_name["openai_compat (openai): endpoint"].ok
    assert by_name["openai_compat (openai): fast model"].ok
    assert not by_name["openai_compat (openai): deep model"].ok  # gpt-4.1 not listed
    assert all("sk-secret-value" not in c.detail + c.fix for c in checks)
    down = httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    o = Settings(brain={"backend": "openai_compat", "openai_compat": {"provider": "ollama"}})
    [_, endpoint] = await doctor.check_openai_compat(o, transport=down, environ={})
    assert not endpoint.ok and "ollama serve" in endpoint.fix
    empty = await doctor.check_openai_compat(o, transport=Script().transport(), environ={})
    assert "ollama pull" in empty[-1].fix


def test_claude_cli_is_optional_on_openai_compat(monkeypatch):
    from spark import doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor, "_bundled_cli", lambda: None)
    s = Settings(brain={"backend": "openai_compat"})
    assert doctor.check_claude_cli(s).required is False
    assert doctor.check_claude_cli(Settings()).required is True
