import importlib.util

import pytest

from spark import registry
from spark.config import Settings
from spark.interfaces import STT, TTS


def _dep_available(dotted: str) -> bool:
    module = dotted.partition(":")[0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize(
    ("kind", "name", "target"),
    [(k, n, t) for k, entries in registry.REGISTRY.items() for n, t in entries.items()],
)
def test_every_entry_is_well_formed(kind, name, target):
    assert kind in registry.BASES
    if isinstance(target, str):
        module, sep, attr = target.partition(":")
        assert sep == ":" and module.startswith("spark.") and attr.isidentifier()
        if not _dep_available(target):
            pytest.skip(f"{module} not written yet or its dependency is not installed")
    try:
        cls = registry.resolve(kind, name)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {kind}/{name}: {exc}")
    assert issubclass(cls, registry.BASES[kind])


def test_unknown_backend_lists_choices():
    with pytest.raises(registry.UnknownBackend) as info:
        registry.resolve("stt", "nope")
    assert "faster_whisper" in str(info.value)


class _DummySTT(STT):
    def __init__(self, settings, **deps):
        self.settings = settings
        self.deps = deps

    async def transcribe(self, pcm):
        return ""


def test_build_uses_config_selection_and_passes_deps():
    registry.register("stt", "dummy", _DummySTT)
    try:
        s = Settings(stt={"backend": "dummy"})
        obj = registry.build("stt", s, player="p")
        assert isinstance(obj, _DummySTT)
        assert obj.deps == {"player": "p"}
    finally:
        registry.unregister("stt", "dummy")


def test_build_rejects_wrong_base():
    registry.register("tts", "wrong", _DummySTT)
    try:
        with pytest.raises(TypeError):
            registry.resolve("tts", "wrong")
    finally:
        registry.unregister("tts", "wrong")
    assert TTS  # imported for clarity of the base check


def test_selected_name_for_audio():
    s = Settings()
    assert registry.selected_name("audio_in", s) == "sounddevice"
    assert registry.selected_name("audio_out", s) == "sounddevice"
