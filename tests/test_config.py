from pathlib import Path

import pytest
import yaml

from spark.config import Settings, apply_legacy_env, load_settings

ROOT = Path(__file__).resolve().parents[1]


def _clear_spark_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith(("SPARK_", "CHIP_")):
            monkeypatch.delenv(key)


def test_repo_config_yaml_matches_code_defaults(monkeypatch):
    _clear_spark_env(monkeypatch)
    loaded = load_settings(ROOT / "config.yaml", load_env_file=False).model_dump()
    defaults = Settings(config_dir=ROOT).model_dump()
    # Intended differences: config.yaml uses the user's custom "Hey Spark" wake word and name,
    # while the code defaults keep the spec's "Spark" and the bundled placeholder model.
    oww = loaded["wakeword"]["options"]["openwakeword"]
    default_oww = defaults["wakeword"]["options"]["openwakeword"]
    assert oww.pop("model_path") == "assets/wakewords/hey_spark.onnx"
    assert default_oww.pop("model_path") is None
    # stricter than the default 0.5: "hey park"/"hey sport" score up to ~0.8 on this model
    assert oww.pop("threshold") == 0.6
    assert default_oww.pop("threshold") == 0.5
    assert loaded["app"].pop("name") == "Spark"
    assert defaults["app"].pop("name") == "Spark"
    assert loaded["app"].pop("name_aliases") == ["Sparks", "Sparky"]
    assert defaults["app"].pop("name_aliases") == []
    assert loaded["app"].pop("wake_phrase") == "Hey Spark"
    assert defaults["app"].pop("wake_phrase") == ""
    assert loaded == defaults


def test_wake_names_are_the_name_and_its_aliases():
    s = Settings(app={"name": "Nova", "name_aliases": ["Novah", " novah ", ""]})
    assert s.app.wake_names == ("nova", "novah")
    assert Settings().app.wake_names == ("spark",)


def test_wake_prompt_is_the_wake_phrase_or_the_name():
    app = Settings(app={"name": "Spark", "wake_phrase": " Hey Spark "}).app
    assert app.wake_prompt == "Hey Spark"
    assert Settings(app={"name": "Spark"}).app.wake_prompt == "Spark"


def test_spec_defaults(monkeypatch):
    _clear_spark_env(monkeypatch)
    s = load_settings(ROOT / "config.yaml", load_env_file=False)
    # Deliberate MVP deviation from spec §4 (Porcupine): no Picovoice key yet, so openWakeWord
    # with the user-trained "Hey Spark" model.
    assert s.wakeword.backend == "openwakeword"
    assert s.vad.backend == "silero"
    assert s.stt.backend == "faster_whisper"
    assert s.stt.options.faster_whisper.model == "base.en"
    assert s.stt.options.faster_whisper.compute_type == "int8"
    assert s.tts.backend == "kokoro" and s.tts.options.kokoro.voice == "bm_george"
    assert s.brain.backend == "claude_agent"
    assert s.brain.fast_model == "claude-haiku-5-5"
    assert s.brain.deep_model == "claude-opus-5-5" and s.brain.mode == "auto"
    assert s.tts.options.piper.voice == "en_US-ljspeech-medium"  # public-domain dataset
    assert s.brain.workers is True and s.brain.worker_model is None
    assert s.brain.idle_reset_s == 600
    assert s.recorder.end_silence_ms == 700
    assert s.app.follow_up_s is None  # deprecated; attention.awake_s rules
    assert s.attention.awake_s == 180.0 and s.awake_s == 180.0
    assert s.app.barge_in is False
    assert s.debug_audio.enabled is False


def test_yaml_partial_file_merges_with_defaults(tmp_path, monkeypatch):
    _clear_spark_env(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"stt": {"backend": "deepgram"}, "app": {"follow_up_s": 3}}))
    s = load_settings(cfg, load_env_file=False)
    assert s.stt.backend == "deepgram"
    assert s.stt.options.faster_whisper.model == "base.en"
    assert s.app.follow_up_s == 3
    assert s.awake_s == 3  # the old key still overrides the awake window
    assert s.tts.backend == "kokoro"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    _clear_spark_env(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"stt": {"backend": "deepgram"}}))
    monkeypatch.setenv("SPARK_STT__BACKEND", "faster_whisper")
    monkeypatch.setenv("SPARK_TTS__OPTIONS__PIPER__LENGTH_SCALE", "1.2")
    s = load_settings(cfg, load_env_file=False)
    assert s.stt.backend == "faster_whisper"
    assert s.tts.options.piper.length_scale == 1.2


def test_dotenv_loaded_without_overriding_env(tmp_path, monkeypatch):
    _clear_spark_env(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}")
    (tmp_path / ".env").write_text("SPARK_TTS__BACKEND=kokoro\nSPARK_TEST_SECRET_X=fromfile\n")
    monkeypatch.setenv("SPARK_TEST_SECRET_X", "fromenv")
    # Register the key with monkeypatch while it is absent, so teardown restores "absent" and
    # undoes the value load_dotenv writes. (A delenv *after* load_settings would record
    # "kokoro" as the original and leak it into every later test.)
    monkeypatch.setenv("SPARK_TTS__BACKEND", "placeholder")
    monkeypatch.delenv("SPARK_TTS__BACKEND")
    s = load_settings(cfg)
    import os

    assert s.tts.backend == "kokoro"
    assert os.environ["SPARK_TEST_SECRET_X"] == "fromenv"


def test_overrides_beat_env(tmp_path, monkeypatch):
    _clear_spark_env(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}")
    monkeypatch.setenv("SPARK_BRAIN__BACKEND", "claude_agent")
    s = load_settings(cfg, load_env_file=False, brain={"backend": "mock"})
    assert s.brain.backend == "mock"
    assert s.brain.fast_model == "claude-haiku-5-5"


def test_unknown_key_rejected(tmp_path, monkeypatch):
    _clear_spark_env(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"stt": {"bakend": "x"}}))
    with pytest.raises(ValueError):
        load_settings(cfg, load_env_file=False)


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml", load_env_file=False)


def test_relative_paths_resolve_against_config_dir(tmp_path, monkeypatch):
    _clear_spark_env(monkeypatch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"models_dir": "m"}))
    s = load_settings(cfg, load_env_file=False)
    assert s.models_path == (tmp_path / "m").resolve()
    assert s.path("assets/sounds") == (tmp_path / "assets/sounds").resolve()


def test_legacy_chip_env_is_copied_to_spark_unless_set(monkeypatch, tmp_path, caplog):
    _clear_spark_env(monkeypatch)
    env = {"CHIP_STT__BACKEND": "deepgram", "CHIP_GITHUB_PAT": "x", "SPARK_GITHUB_PAT": "y"}
    assert apply_legacy_env(env) == ["CHIP_STT__BACKEND"]
    assert env["SPARK_STT__BACKEND"] == "deepgram" and env["SPARK_GITHUB_PAT"] == "y"

    monkeypatch.setenv("CHIP_TTS__BACKEND", "kokoro")
    monkeypatch.setenv("SPARK_TTS__BACKEND", "")  # so monkeypatch removes the copy afterwards
    monkeypatch.delenv("SPARK_TTS__BACKEND")
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    assert load_settings(tmp_path / "config.yaml", load_env_file=False).tts.backend == "kokoro"
    assert "CHIP_TTS__BACKEND is deprecated" in caplog.text and "kokoro" not in caplog.text
