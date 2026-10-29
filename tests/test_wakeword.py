"""Wake-word backends: Porcupine plumbing and error mapping, openWakeWord thresholding.

Both libraries are replaced with fake modules in ``sys.modules`` so the tests run offline and
without keys. Real-engine tests skip unless the key/model files are present.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from spark import registry
from spark.config import Settings
from spark.interfaces import BackendUnavailable, WakeWordDetector
from spark.wakeword.openwakeword import OpenWakeWord
from spark.wakeword.porcupine import PorcupineWakeWord

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-TEST-porcupine-key-1234567890"


def _settings(tmp_path: Path, **wakeword) -> Settings:
    return Settings(config_dir=tmp_path, models_dir=str(tmp_path / "models"), wakeword=wakeword)


# --- fake pvporcupine ------------------------------------------------------------------------


class _PvError(Exception):
    def __init__(self, message: str = "", message_stack=None) -> None:
        super().__init__(message)
        self.message = message
        self.message_stack = list(message_stack or [])

    def __str__(self) -> str:
        return "\n".join([self.message, *self.message_stack])


_PV_ERRORS: dict[str, type[Exception]] = {"PorcupineError": _PvError}
for _name in (
    "PorcupineIOError",
    "PorcupineInvalidArgumentError",
    "PorcupineKeyError",
    "PorcupineActivationError",
    "PorcupineActivationLimitError",
    "PorcupineActivationThrottledError",
    "PorcupineActivationRefusedError",
):
    _PV_ERRORS[_name] = type(_name, (_PvError,), {})


def _fake_pvporcupine(fire_on: set[int] | None = None, raise_exc: Exception | None = None):
    mod = types.ModuleType("pvporcupine")
    for name, cls in _PV_ERRORS.items():
        setattr(mod, name, cls)
    mod.created = []

    class Handle:
        frame_length = 512
        sample_rate = 16000

        def __init__(self, kwargs) -> None:
            self.kwargs = kwargs
            self.frames = 0
            self.deleted = 0

        def process(self, pcm) -> int:
            assert len(pcm) == 512
            self.frames += 1
            return 0 if self.frames - 1 in (fire_on or set()) else -1

        def delete(self) -> None:
            self.deleted += 1

    def create(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        h = Handle(kwargs)
        mod.created.append(h)
        return h

    mod.create = create
    return mod


@pytest.fixture
def ppn(tmp_path) -> Path:
    path = tmp_path / "assets" / "wakewords" / "hey_spark_windows.ppn"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ppn")
    return path


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", SECRET)


def test_porcupine_missing_key(monkeypatch, tmp_path, ppn):
    monkeypatch.delenv("PICOVOICE_ACCESS_KEY", raising=False)
    with pytest.raises(BackendUnavailable) as info:
        PorcupineWakeWord(_settings(tmp_path))
    assert "PICOVOICE_ACCESS_KEY" in info.value.reason
    assert info.value.fix == (
        "Get a key at https://console.picovoice.ai and set PICOVOICE_ACCESS_KEY in .env"
    )


def test_porcupine_custom_key_env_named_in_fix(monkeypatch, tmp_path, ppn):
    monkeypatch.delenv("MY_PV_KEY", raising=False)
    s = _settings(tmp_path, options={"porcupine": {"access_key_env": "MY_PV_KEY"}})
    with pytest.raises(BackendUnavailable) as info:
        PorcupineWakeWord(s)
    assert "MY_PV_KEY" in info.value.fix


def test_porcupine_missing_ppn(tmp_path, with_key):
    with pytest.raises(BackendUnavailable) as info:
        PorcupineWakeWord(_settings(tmp_path))
    fix = info.value.fix
    assert "Picovoice Console" in fix and "Windows" in fix and "Hey Spark" in fix
    assert str(tmp_path / "assets" / "wakewords" / "hey_spark_windows.ppn") in fix
    assert SECRET not in str(info.value)


def test_porcupine_not_installed(monkeypatch, tmp_path, ppn, with_key):
    monkeypatch.setitem(sys.modules, "pvporcupine", None)
    with pytest.raises(BackendUnavailable, match="pvporcupine is not installed"):
        PorcupineWakeWord(_settings(tmp_path))


def test_porcupine_plumbing(monkeypatch, tmp_path, ppn, with_key):
    pv = _fake_pvporcupine(fire_on={2})
    monkeypatch.setitem(sys.modules, "pvporcupine", pv)
    s = _settings(tmp_path, backend="porcupine", options={"porcupine": {"sensitivity": 0.7}})
    det = registry.build("wakeword", s)
    assert isinstance(det, PorcupineWakeWord) and isinstance(det, WakeWordDetector)
    assert det.frame_length == 512
    (handle,) = pv.created
    assert handle.kwargs == {
        "access_key": SECRET,
        "keyword_paths": [str(ppn)],
        "sensitivities": [0.7],
        "device": "cpu",
    }
    frame = np.zeros(512, np.int16)
    assert [det.process(frame) for _ in range(4)] == [False, False, True, False]
    with pytest.raises(ValueError):
        det.process(np.zeros(1280, np.int16))
    with pytest.raises(ValueError):
        det.process(np.zeros(512, np.float32))
    det.reset()
    det.close()
    det.close()  # idempotent
    assert handle.deleted == 1
    with pytest.raises(RuntimeError):
        det.process(frame)


@pytest.mark.parametrize(
    ("exc_name", "message", "stack", "reason_part", "fix_part"),
    [
        (
            "PorcupineInvalidArgumentError",
            "Initialization failed",
            [f"Failed to parse AccessKey `{SECRET}`."],
            "invalid AccessKey",
            "console.picovoice.ai",
        ),
        (
            "PorcupineInvalidArgumentError",
            "Initialization failed",
            ["Keyword file belongs to a different platform"],
            "another platform",
            "Picovoice Console",
        ),
        ("PorcupineActivationError", f"activation failed for {SECRET}", [], "network", "key"),
        ("PorcupineActivationLimitError", "limit", [], "limit", "limits"),
        ("PorcupineActivationThrottledError", "throttled", [], "throttled", "retry"),
        ("PorcupineActivationRefusedError", "refused", [], "refused", "key"),
        ("PorcupineIOError", "io", [], "could not read keyword file", "Picovoice Console"),
    ],
)
def test_porcupine_error_mapping_never_leaks_key(
    monkeypatch, tmp_path, ppn, with_key, caplog, exc_name, message, stack, reason_part, fix_part
):
    exc = _PV_ERRORS[exc_name](message, stack)
    pv = _fake_pvporcupine(raise_exc=exc)
    monkeypatch.setitem(sys.modules, "pvporcupine", pv)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(BackendUnavailable) as info:
        PorcupineWakeWord(_settings(tmp_path))
    err = info.value
    assert reason_part in err.reason and fix_part in err.fix
    assert SECRET not in str(err) and SECRET not in err.fix
    assert err.__cause__ is None and err.__suppress_context__
    assert SECRET not in caplog.text


def test_porcupine_plain_value_error_mapped(monkeypatch, tmp_path, ppn, with_key):
    pv = _fake_pvporcupine(raise_exc=ValueError("sensitivities length mismatch"))
    monkeypatch.setitem(sys.modules, "pvporcupine", pv)
    with pytest.raises(BackendUnavailable, match="invalid Porcupine arguments"):
        PorcupineWakeWord(_settings(tmp_path))


def test_porcupine_real_library_rejects_bad_key_without_leaking(monkeypatch, tmp_path, ppn):
    """Uses the installed pvporcupine: a malformed key fails offline with a redacted message."""
    if importlib.util.find_spec("pvporcupine") is None:
        pytest.skip("pvporcupine not installed")
    import pvporcupine

    kw = Path(pvporcupine.KEYWORD_PATHS["jarvis"])
    s = _settings(tmp_path, options={"porcupine": {"keyword_path": str(kw)}})
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", SECRET)
    with pytest.raises(BackendUnavailable) as info:
        PorcupineWakeWord(s)
    assert "AccessKey" in info.value.reason
    assert SECRET not in str(info.value)


_REAL_PPN = ROOT / "assets" / "wakewords" / "hey_spark_windows.ppn"


@pytest.mark.skipif(
    not (os.environ.get("PICOVOICE_ACCESS_KEY") and _REAL_PPN.is_file()),
    reason="needs PICOVOICE_ACCESS_KEY and assets/wakewords/hey_spark_windows.ppn",
)
def test_porcupine_real_engine_on_silence():
    det = PorcupineWakeWord(Settings(config_dir=ROOT))
    try:
        assert det.frame_length == 512
        assert not any(det.process(np.zeros(512, np.int16)) for _ in range(60))
    finally:
        det.close()


# --- fake openwakeword -----------------------------------------------------------------------


def _install_fake_oww(monkeypatch, scores: list[float] | None = None, download_ok: bool = True):
    pkg = types.ModuleType("openwakeword")
    pkg.__path__ = []
    utils = types.ModuleType("openwakeword.utils")
    model_mod = types.ModuleType("openwakeword.model")
    state = types.SimpleNamespace(downloads=[], models=[])

    def download_models(model_names, target_directory):
        state.downloads.append((list(model_names), target_directory))
        if not download_ok:
            raise OSError("network unreachable")
        d = Path(target_directory)
        for name in ("melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"):
            (d / name).write_bytes(b"f")
        for name in model_names:
            (d / f"{name}_v0.1.onnx").write_bytes(b"m")

    class Model:
        def __init__(self, wakeword_models, inference_framework, **kwargs):
            self.wakeword_models = wakeword_models
            self.inference_framework = inference_framework
            self.kwargs = kwargs
            self.queue = list(scores or [])
            self.resets = 0
            self.frames = []
            state.models.append(self)

        def predict(self, x):
            assert x.dtype == np.int16 and x.size == 1280
            self.frames.append(x)
            key = Path(self.wakeword_models[0]).stem
            return {key: np.float32(self.queue.pop(0) if self.queue else 0.0)}

        def reset(self):
            self.resets += 1

    utils.download_models = download_models
    model_mod.Model = Model
    pkg.utils = utils
    pkg.model = model_mod
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_mod)
    return state


def test_oww_not_installed(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)
    with pytest.raises(BackendUnavailable) as info:
        OpenWakeWord(_settings(tmp_path, backend="openwakeword"))
    assert info.value.fix == "uv sync --extra oww"


def test_oww_downloads_placeholder_once_and_uses_onnx(monkeypatch, tmp_path):
    state = _install_fake_oww(monkeypatch)
    s = _settings(tmp_path, backend="openwakeword")
    det = registry.build("wakeword", s)
    assert isinstance(det, OpenWakeWord) and det.frame_length == 1280
    oww_dir = tmp_path / "models" / "openwakeword"
    assert state.downloads == [(["hey_jarvis"], str(oww_dir))]
    (model,) = state.models
    assert model.inference_framework == "onnx"
    assert model.wakeword_models == [str(oww_dir / "hey_jarvis_v0.1.onnx")]
    assert model.kwargs == {
        "melspec_model_path": str(oww_dir / "melspectrogram.onnx"),
        "embedding_model_path": str(oww_dir / "embedding_model.onnx"),
    }
    OpenWakeWord(s)  # files now present: no second download
    assert len(state.downloads) == 1


def test_oww_threshold_and_reset_after_detection(monkeypatch, tmp_path):
    state = _install_fake_oww(monkeypatch, scores=[0.1, 0.49, 0.5, 0.9, 0.2])
    det = OpenWakeWord(_settings(tmp_path, options={"openwakeword": {"threshold": 0.5}}))
    frame = np.zeros(1280, np.int16)
    fired = [det.process(frame) for _ in range(5)]
    assert fired == [False, False, True, True, False]
    assert state.models[0].resets == 2  # reset after each detection
    assert det.last_score == pytest.approx(0.2)
    det.reset()
    assert state.models[0].resets == 3 and det.last_score == 0.0


def test_oww_rejects_wrong_frames(monkeypatch, tmp_path):
    _install_fake_oww(monkeypatch)
    det = OpenWakeWord(_settings(tmp_path))
    with pytest.raises(ValueError, match="1280"):
        det.process(np.zeros(512, np.int16))
    with pytest.raises(ValueError, match="int16"):
        det.process(np.zeros(1280, np.float32))
    det.close()
    with pytest.raises(RuntimeError):
        det.process(np.zeros(1280, np.int16))


def test_oww_custom_model_path(monkeypatch, tmp_path):
    state = _install_fake_oww(monkeypatch)
    custom = tmp_path / "assets" / "wakewords" / "hey_spark.onnx"
    s = _settings(
        tmp_path, options={"openwakeword": {"model_path": "assets/wakewords/hey_spark.onnx"}}
    )
    with pytest.raises(BackendUnavailable) as info:
        OpenWakeWord(s)
    assert str(custom) in info.value.reason and "model_path" in info.value.fix
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"m")
    det = OpenWakeWord(s)
    assert state.downloads[-1][0] == []  # only feature models fetched
    assert state.models[-1].wakeword_models == [str(custom)]
    assert det.model_file == custom


def test_oww_download_failure_is_backend_unavailable(monkeypatch, tmp_path):
    _install_fake_oww(monkeypatch, download_ok=False)
    with pytest.raises(BackendUnavailable) as info:
        OpenWakeWord(_settings(tmp_path))
    assert "download failed" in info.value.reason and "github.com" in info.value.fix


def test_wakeword_modules_import_lazily():
    code = (
        "import sys, spark.wakeword.porcupine, spark.wakeword.openwakeword; "
        "assert 'pvporcupine' not in sys.modules and 'openwakeword' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


_REAL_OWW_DIR = ROOT / ".models" / "openwakeword"


@pytest.mark.skipif(
    importlib.util.find_spec("openwakeword") is None
    or not (_REAL_OWW_DIR / "melspectrogram.onnx").is_file()
    or not list(_REAL_OWW_DIR.glob("hey_jarvis*.onnx")),
    reason="openwakeword not installed or its models are not in .models/openwakeword",
)
def test_oww_real_model_on_silence():
    det = OpenWakeWord(Settings(config_dir=ROOT, wakeword={"backend": "openwakeword"}))
    try:
        frame = np.zeros(1280, np.int16)
        assert not any(det.process(frame) for _ in range(30))
        assert 0.0 <= det.last_score < det.threshold
    finally:
        det.close()
