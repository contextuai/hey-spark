---
name: spark-backend-pattern
description: How to add or modify a swappable backend in Spark (wake word, VAD, STT, TTS, brain, audio I/O) — ABC in interfaces.py, registry entry, lazy imports, config schema, doctor check, and tests. Use whenever creating or changing any file under spark/wakeword, spark/vad, spark/stt, spark/tts, spark/brain/claude_agent.py, or spark/registry.py.
---

# Adding a Spark backend

Backends must be swappable **by editing `config.yaml` alone** (Phase 1 acceptance). Follow all six steps.

## 1. Interface (`spark/interfaces.py`)
The ABCs already exist (or create them exactly like this):
```python
class WakeWordDetector(ABC):
    frame_length: int                      # samples @16k this detector consumes
    @abstractmethod
    def process(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...

class VAD(ABC):
    frame_length: int = 512
    @abstractmethod
    def is_speech(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...

class STT(ABC):
    async def warmup(self) -> None: ...
    @abstractmethod
    async def transcribe(self, pcm: np.ndarray) -> str: ...

class TTS(ABC):
    async def warmup(self) -> None: ...
    @abstractmethod
    async def speak(self, text_chunks: AsyncIterator[str]) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...

class Brain(ABC):
    @abstractmethod
    def respond(self, text: str) -> AsyncIterator[str]: ...
    async def reset(self) -> None: ...
    async def close(self) -> None: ...

class BackendUnavailable(RuntimeError):
    def __init__(self, backend: str, reason: str, fix: str): ...
```
Don't change a signature without updating every backend, fake, and test in the same commit.

## 2. Module
`spark/<kind>/<name>.py`: one class. Import heavy libraries **inside `__init__`**, and convert an ImportError or a missing file/key into `BackendUnavailable(backend, reason, fix)`, where `fix` is a copy-pasteable command.

## 3. Registry (`spark/registry.py`)
```python
REGISTRY = {
  "stt": {"faster_whisper": "spark.stt.faster_whisper:FasterWhisperSTT",
          "deepgram": "spark.stt.deepgram:DeepgramSTT"},
  ...
}
def build(kind: str, cfg: Settings, **deps) -> Any   # importlib by dotted path
```
Registration uses a string path, so nothing is imported until selected. The app calls `build(...)` only.

## 4. Config (`spark/config.py` + `config.yaml`)
Add a pydantic sub-model `<Name>Config` under `stt.options.<name>` (etc.), with defaults that match the spec. Selection key: `stt.backend: faster_whisper`. Env overrides use the prefix `SPARK_` with the `__` delimiter. Document every new key with a comment in `config.yaml`.

## 5. Doctor (`spark/doctor.py`)
Add a check: try constructing the backend (or just verify the file, key, and import) and print ✔ or ✖ with `fix`. The doctor never crashes; it collects every result.

## 6. Tests
- Registry test: every registry entry resolves to a class that subclasses the right ABC (import only if the dependency is installed, otherwise assert the dotted path is well-formed).
- A backend-specific test that skips with a reason if the model or dependency is missing.
- A swap test: build the app from two configs that differ only in `stt.backend`/`tts.backend` (using fake backends registered for the test) and assert the right classes were built. That proves "swap without code changes".
