"""Silero VAD backend: streaming state, frame validation, error mapping, and the real model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from spark import registry
from spark.config import Settings
from spark.interfaces import VAD, BackendUnavailable
from spark.vad import silero
from spark.vad.silero import SileroVAD

ROOT = Path(__file__).resolve().parents[1]
HAS_FW = importlib.util.find_spec("faster_whisper") is not None
needs_model = pytest.mark.skipif(not HAS_FW, reason="faster-whisper (bundled Silero ONNX) missing")


def _tone(n: int = 512, amp: float = 8000.0, hz: float = 220.0) -> np.ndarray:
    t = np.arange(n) / 16_000
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


class _FakeSession:
    """Stands in for an ORT session: records inputs, returns a fixed prob and bumped state."""

    def __init__(self, prob: float = 0.9) -> None:
        self.prob = prob
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, _outputs, feeds):
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return np.array([[self.prob]], np.float32), feeds["h"] + 1.0, feeds["c"] + 1.0


@pytest.fixture
def fake_vad(monkeypatch, tmp_path) -> tuple[SileroVAD, _FakeSession]:
    """A SileroVAD whose ONNX session is a fake (no model needed)."""
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"x")
    session = _FakeSession()

    class _Ort:
        class SessionOptions:
            pass

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            assert providers == ["CPUExecutionProvider"]
            assert sess_options.intra_op_num_threads == 1
            assert sess_options.inter_op_num_threads == 1
            return session

    monkeypatch.setitem(sys.modules, "onnxruntime", _Ort)
    return SileroVAD(Settings(config_dir=ROOT), model_path=model), session


def test_state_and_context_carry_between_frames(fake_vad):
    vad, session = fake_vad
    a, b = _tone(), _tone(hz=440)
    vad.probability(a)
    vad.probability(b)
    first, second = session.calls
    assert first["input"].shape == (1, 576)
    assert not first["h"].any() and not first["input"][0, :64].any()
    assert (second["h"] == 1.0).all() and (second["c"] == 1.0).all()
    np.testing.assert_allclose(second["input"][0, :64], a[-64:] / 32768.0, rtol=1e-6)
    np.testing.assert_allclose(second["input"][0, 64:], b / 32768.0, rtol=1e-6)


def test_reset_clears_state_and_context(fake_vad):
    vad, session = fake_vad
    vad.probability(_tone())
    vad.reset()
    vad.probability(_tone())
    last = session.calls[-1]
    assert not last["h"].any() and not last["c"].any() and not last["input"][0, :64].any()
    assert vad.last_probability == pytest.approx(0.9)


def test_threshold_from_settings(fake_vad, monkeypatch, tmp_path):
    vad, session = fake_vad
    assert vad.threshold == 0.5 and vad.is_speech(_tone())
    session.prob = 0.2
    assert not vad.is_speech(_tone())
    model = vad.model_path
    strict = SileroVAD(
        Settings(config_dir=ROOT, vad={"options": {"silero": {"threshold": 0.95}}}),
        model_path=model,
    )
    assert strict.threshold == 0.95


@pytest.mark.parametrize("n", [0, 511, 513, 1280])
def test_wrong_frame_size_raises(fake_vad, n):
    vad, _ = fake_vad
    with pytest.raises(ValueError, match="512"):
        vad.is_speech(np.zeros(n, np.int16))


def test_two_dimensional_frame_rejected(fake_vad):
    vad, _ = fake_vad
    with pytest.raises(ValueError):
        vad.probability(np.zeros((1, 512), np.int16))


def test_missing_model_file_is_backend_unavailable(tmp_path):
    with pytest.raises(BackendUnavailable) as info:
        SileroVAD(Settings(config_dir=ROOT), model_path=tmp_path / "nope.onnx")
    assert "not found" in info.value.reason and info.value.fix


def test_missing_faster_whisper_is_backend_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", None)
    with pytest.raises(BackendUnavailable) as info:
        SileroVAD(Settings(config_dir=ROOT))
    assert "faster-whisper" in info.value.reason and "uv sync" in info.value.fix


def test_missing_onnxruntime_is_backend_unavailable(monkeypatch, tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(BackendUnavailable, match="onnxruntime"):
        SileroVAD(Settings(config_dir=ROOT), model_path=model)


def test_module_import_is_lazy():
    import subprocess

    code = (
        "import sys, spark.vad.silero; "
        "assert 'onnxruntime' not in sys.modules and 'faster_whisper' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# --- real model ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_vad() -> SileroVAD:
    if not HAS_FW:
        pytest.skip("faster-whisper not installed")
    return registry.build("vad", Settings(config_dir=ROOT))


@needs_model
def test_registry_builds_silero(real_vad):
    assert isinstance(real_vad, VAD) and real_vad.frame_length == 512
    assert real_vad.model_path.name == silero.MODEL_FILE


@needs_model
def test_real_silence_and_noise_have_low_probability(real_vad):
    real_vad.reset()
    probs = [real_vad.probability(np.zeros(512, np.int16)) for _ in range(31)]
    assert max(probs) < 0.1
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(512 * 31) * 0.05 * 32768).astype(np.int16)
    real_vad.reset()
    probs = [real_vad.probability(noise[i : i + 512]) for i in range(0, noise.size, 512)]
    assert max(probs) < real_vad.threshold
    assert not real_vad.is_speech(np.zeros(512, np.int16))


@needs_model
def test_real_reset_restores_initial_output(real_vad):
    rng = np.random.default_rng(1)
    frames = [(rng.standard_normal(512) * 6000).astype(np.int16) for _ in range(5)]
    real_vad.reset()
    first = real_vad.probability(frames[0])
    for f in frames[1:]:
        real_vad.probability(f)
    real_vad.reset()
    assert real_vad.probability(frames[0]) == pytest.approx(first, abs=1e-6)


@needs_model
def test_real_float_and_int16_inputs_agree(real_vad):
    frame = _tone(amp=5000)
    real_vad.reset()
    p_int = real_vad.probability(frame)
    real_vad.reset()
    p_float = real_vad.probability(frame.astype(np.float32) / 32768.0)
    assert p_int == pytest.approx(p_float, abs=1e-5)
