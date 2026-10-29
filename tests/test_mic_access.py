"""macOS microphone permission (spark.audio.mic_access) and the mic error hints."""

from __future__ import annotations

import sys
import threading

import pytest

from spark.audio import mic_access, stream
from spark.audio.stream import MicUnavailable, mic_fix


def _as_frozen_mac(monkeypatch):
    monkeypatch.setattr(mic_access.sys, "platform", "darwin")
    monkeypatch.setattr(mic_access.sys, "frozen", True, raising=False)


def test_mic_fix_is_per_platform():
    assert "Windows Settings" in mic_fix("win32")
    mac = mic_fix("darwin")
    assert "Privacy & Security > Microphone" in mac and "Windows" not in mac
    assert "external mic" in mac
    assert "Windows" not in mic_fix("linux")


def test_ensure_does_nothing_outside_the_mac_app(monkeypatch):
    monkeypatch.setattr(mic_access, "status", lambda: pytest.fail("must not ask"))
    monkeypatch.setattr(mic_access.sys, "platform", "win32")
    assert mic_access.ensure() == "unknown"
    monkeypatch.setattr(mic_access.sys, "platform", "darwin")
    monkeypatch.delattr(mic_access.sys, "frozen", raising=False)
    assert mic_access.ensure() == "unknown"  # from a terminal, the terminal app prompts


@pytest.mark.parametrize(
    ("answer", "after", "expected"),
    [(True, "authorized", "authorized"), (False, "denied", "denied")],
)
def test_ensure_asks_once_when_undetermined(monkeypatch, answer, after, expected):
    _as_frozen_mac(monkeypatch)
    statuses = iter(["not_determined", after])
    asked = []
    monkeypatch.setattr(mic_access, "status", lambda: next(statuses))
    monkeypatch.setattr(mic_access, "request", lambda: asked.append(1) or answer)
    assert mic_access.ensure() == expected
    assert asked == [1]


def test_ensure_does_not_ask_again_once_answered(monkeypatch):
    _as_frozen_mac(monkeypatch)
    monkeypatch.setattr(mic_access, "status", lambda: "denied")
    monkeypatch.setattr(mic_access, "request", lambda: pytest.fail("must not ask"))
    assert mic_access.ensure() == "denied"


class _FakeCaptureDevice:
    def __init__(self, grant: bool | None) -> None:
        self.grant = grant

    def requestAccessForMediaType_completionHandler_(self, media, handler):
        assert media == mic_access.AV_MEDIA_TYPE_AUDIO
        if self.grant is not None:  # the system answers on another thread
            threading.Thread(target=handler, args=(self.grant,)).start()


@pytest.mark.parametrize("grant", [True, False])
def test_request_waits_for_the_answer(monkeypatch, grant):
    monkeypatch.setattr(mic_access, "_av_capture_device", lambda: _FakeCaptureDevice(grant))
    assert mic_access.request(timeout=5) is grant


def test_request_gives_up_without_an_answer(monkeypatch):
    monkeypatch.setattr(mic_access, "_av_capture_device", lambda: _FakeCaptureDevice(None))
    assert mic_access.request(timeout=0.05) is False


def test_request_survives_missing_pyobjc(monkeypatch):
    def boom():
        raise ImportError("No module named 'objc'")

    monkeypatch.setattr(mic_access, "_av_capture_device", boom)
    assert mic_access.request(timeout=0.05) is False


def test_status_is_unknown_off_mac(monkeypatch):
    monkeypatch.setattr(mic_access.sys, "platform", "win32")
    assert mic_access.status() == "unknown"


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS only")
def test_status_reads_avfoundation_on_mac():
    pytest.importorskip("objc")
    assert mic_access.status() in set(mic_access.STATUSES.values())


class _FakeSD:
    """sounddevice stand-in whose default input appears only after PortAudio restarts."""

    def __init__(self) -> None:
        self.restarted = False
        self.default = type("D", (), {"device": [-1, 3]})()

    def query_devices(self, index):
        if index == -1:
            raise RuntimeError("Error querying device -1")
        return {
            "name": "MacBook Pro Microphone",
            "max_input_channels": 1,
            "default_samplerate": 48000.0,
            "hostapi": 0,
        }

    def query_hostapis(self, index=None):
        return {"name": "Core Audio"}

    def _terminate(self):
        self.restarted = True

    def _initialize(self):
        self.default.device = [1, 3]


def test_device_factory_reports_mac_permission_denied(monkeypatch):
    monkeypatch.setattr(mic_access, "ensure", lambda: "denied")
    with pytest.raises(MicUnavailable) as exc:
        stream._device_factory(_FakeSD(), None)
    assert "microphone" in str(exc.value) and exc.value.fix == mic_access.SETTINGS_FIX


def test_device_factory_relists_devices_on_mac(monkeypatch):
    monkeypatch.setattr(mic_access, "ensure", lambda: "authorized")
    monkeypatch.setattr(stream.sys, "platform", "darwin")
    sd = _FakeSD()
    _, rate, channels = stream._device_factory(sd, None)
    assert sd.restarted and (rate, channels) == (48000, 1)


def test_device_factory_without_any_mic_on_mac(monkeypatch):
    monkeypatch.setattr(mic_access, "ensure", lambda: "authorized")
    monkeypatch.setattr(stream.sys, "platform", "darwin")
    sd = _FakeSD()
    sd._initialize = lambda: None  # still no input device after the restart
    with pytest.raises(MicUnavailable) as exc:
        stream._device_factory(sd, None)
    assert "device -1" in str(exc.value) and "external mic" in exc.value.fix
