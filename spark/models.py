"""Model files in ``settings.models_path``: where they live, whether they're present, download.

Layout (the backends read from exactly these paths)::

    <models_dir>/whisper/<model>/        CTranslate2 faster-whisper model (model.bin, ...)
    <models_dir>/piper/<voice>.onnx      Piper voice (+ <voice>.onnx.json)
    <models_dir>/kokoro/<model_file>     Kokoro ONNX model (+ voices file)

Downloads land in a ``.partial`` temp name first and are renamed into place only when complete,
so an interrupted download is never mistaken for a finished one. Re-running is a no-op for
anything already present. ``huggingface_hub`` is imported only when something is downloaded.

CLI: ``uv run python -m spark.models download [--config config.yaml] [--no-whisper] ...``
"""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from spark.config import Settings

__all__ = [
    "DOWNLOAD_FIX",
    "KOKORO_URL_BASE",
    "MIN_FREE_BYTES",
    "InsufficientDiskSpace",
    "download_models",
    "free_bytes",
    "kokoro_paths",
    "model_status",
    "piper_hf_path",
    "piper_voice_paths",
    "whisper_model_dir",
    "whisper_present",
    "whisper_repo_id",
]

DOWNLOAD_FIX = "uv run python -m spark models download"
MIN_FREE_BYTES = 2 * 1024**3
PIPER_REPO = "rhasspy/piper-voices"
KOKORO_URL_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
WHISPER_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json"]
WHISPER_PATTERNS = [*WHISPER_FILES, "vocabulary.*"]
WHISPER_REQUIRED = ("model.bin", "config.json", "tokenizer.json")
_PARTIAL = ".partial"


class InsufficientDiskSpace(RuntimeError):
    """Less than :data:`MIN_FREE_BYTES` free where the models would be written."""


class _Console(Protocol):
    def print(self, *args: Any, **kwargs: Any) -> None: ...


class _PrintConsole:
    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args, flush=True)


# --- paths -----------------------------------------------------------------------------------


def whisper_repo_id(model: str) -> str:
    """Hugging Face repo for a faster-whisper size name (``base.en``) or a full repo id."""
    return model if "/" in model else f"Systran/faster-whisper-{model}"


def whisper_model_dir(settings: Settings) -> Path:
    """Directory holding the configured faster-whisper model."""
    model = settings.stt.options.faster_whisper.model
    return settings.models_path / "whisper" / model.replace("/", "--")


def whisper_present(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in WHISPER_REQUIRED) and any(
        model_dir.glob("vocabulary.*")
    )


def piper_hf_path(voice: str) -> str:
    """Path of ``<voice>.onnx`` inside ``rhasspy/piper-voices``.

    ``en_US-ljspeech-medium`` -> ``en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx``. Speaker
    names may contain ``_`` or ``-`` (``en_GB-northern_english_male-medium``).
    """
    parts = voice.split("-")
    if len(parts) < 3 or "_" not in parts[0]:
        raise ValueError(
            f"piper voice {voice!r} is not <lang_CODE>-<name>-<quality> "
            "(e.g. en_US-ljspeech-medium)"
        )
    lang_code, quality, name = parts[0], parts[-1], "-".join(parts[1:-1])
    family = lang_code.split("_")[0]
    return f"{family}/{lang_code}/{name}/{quality}/{voice}.onnx"


def piper_voice_paths(settings: Settings) -> tuple[Path, Path]:
    """``(<voice>.onnx, <voice>.onnx.json)`` under ``models_path/piper``."""
    voice = settings.tts.options.piper.voice
    onnx = settings.models_path / "piper" / f"{voice}.onnx"
    return onnx, onnx.with_name(onnx.name + ".json")


def kokoro_paths(settings: Settings) -> tuple[Path, Path]:
    """``(model, voices)`` files under ``models_path/kokoro``."""
    cfg = settings.tts.options.kokoro
    root = settings.models_path / "kokoro"
    return root / cfg.model_file, root / cfg.voices_file


def free_bytes(path: Path) -> int:
    """Free disk space on the volume holding ``path`` (or its nearest existing parent)."""
    probe = Path(path).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def _mb(n: int) -> str:
    return f"{n / 1024**2:,.1f} MB"


def _gb(n: int) -> str:
    return f"{n / 1024**3:,.1f} GB"


# --- status ----------------------------------------------------------------------------------


def model_status(settings: Settings) -> dict[str, dict[str, Any]]:
    """What's present in ``models_path`` (for ``spark doctor``). Never downloads or raises.

    ``{name: {"present", "path", "required", "model", "size_bytes"}}``; a model is required
    when the configured backend uses it (whisper for ``faster_whisper``, piper, kokoro).
    """
    wdir = whisper_model_dir(settings)
    p_onnx, p_json = piper_voice_paths(settings)
    k_model, k_voices = kokoro_paths(settings)
    return {
        "whisper": {
            "model": settings.stt.options.faster_whisper.model,
            "path": str(wdir),
            "present": whisper_present(wdir),
            "required": settings.stt.backend == "faster_whisper",
            "size_bytes": _size(wdir),
        },
        "piper": {
            "model": settings.tts.options.piper.voice,
            "path": str(p_onnx),
            "present": p_onnx.is_file() and p_json.is_file(),
            "required": settings.tts.backend == "piper",
            "size_bytes": _size(p_onnx) + _size(p_json),
        },
        "kokoro": {
            "model": settings.tts.options.kokoro.model_file,
            "path": str(k_model),
            "present": k_model.is_file() and k_voices.is_file(),
            "required": settings.tts.backend == "kokoro",
            "size_bytes": _size(k_model) + _size(k_voices),
        },
    }


# --- downloaders (module-level so tests can monkeypatch them) --------------------------------


def _snapshot_download(repo_id: str, local_dir: Path, allow_patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns)


def _hf_file_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download one file from a HF model repo into ``local_dir``; returns its local path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, local_dir=str(local_dir)))


def _ssl_context() -> ssl.SSLContext:
    """Verify against certifi's CA bundle. The macOS app's Python has no system CA store, so
    the default context fails there with CERTIFICATE_VERIFY_FAILED."""
    try:
        import certifi
    except ImportError:  # certifi comes with httpx and requests; fall back to the system store
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _url_download(url: str, dest: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as resp,
        dest.open("wb") as out,
    ):
        shutil.copyfileobj(resp, out, length=1024 * 1024)


# --- download --------------------------------------------------------------------------------


def _rmtree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _download_whisper(settings: Settings, say: Callable[[str], None]) -> Path:
    final = whisper_model_dir(settings)
    if whisper_present(final):
        say(f"whisper {final.name}: present ({_mb(_size(final))}) at {final}")
        return final
    repo = whisper_repo_id(settings.stt.options.faster_whisper.model)
    tmp = final.with_name(final.name + _PARTIAL)
    _rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    say(f"whisper {final.name}: downloading {repo} ...")
    _snapshot_download(repo, tmp, WHISPER_PATTERNS)
    _rmtree(tmp / ".cache")  # huggingface_hub bookkeeping, not needed once complete
    if not whisper_present(tmp):
        raise RuntimeError(f"download of {repo} is incomplete (missing files in {tmp})")
    _rmtree(final)
    os.replace(tmp, final)
    say(f"whisper {final.name}: done ({_mb(_size(final))}) at {final}")
    return final


def _download_piper(settings: Settings, say: Callable[[str], None]) -> Path:
    onnx, cfg = piper_voice_paths(settings)
    voice = settings.tts.options.piper.voice
    if onnx.is_file() and cfg.is_file():
        say(f"piper {voice}: present ({_mb(_size(onnx) + _size(cfg))}) at {onnx.parent}")
        return onnx
    hf_path = piper_hf_path(voice)
    tmp = onnx.parent / f".{voice}{_PARTIAL}"
    _rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    say(f"piper {voice}: downloading {PIPER_REPO}/{hf_path} ...")
    got_json = _hf_file_download(PIPER_REPO, hf_path + ".json", tmp)
    got_onnx = _hf_file_download(PIPER_REPO, hf_path, tmp)
    # Config first, model last: the .onnx appearing is what marks the voice as complete.
    os.replace(got_json, cfg)
    os.replace(got_onnx, onnx)
    _rmtree(tmp)
    say(f"piper {voice}: done ({_mb(_size(onnx) + _size(cfg))}) at {onnx.parent}")
    return onnx


def _download_kokoro(settings: Settings, say: Callable[[str], None]) -> Path:
    model, voices = kokoro_paths(settings)
    model.parent.mkdir(parents=True, exist_ok=True)
    for dest in (voices, model):
        if dest.is_file():
            say(f"kokoro {dest.name}: present ({_mb(_size(dest))})")
            continue
        tmp = dest.with_name(dest.name + _PARTIAL)
        _rmtree(tmp)
        url = f"{KOKORO_URL_BASE}/{dest.name}"
        say(f"kokoro {dest.name}: downloading {url} ...")
        _url_download(url, tmp)
        os.replace(tmp, dest)
        say(f"kokoro {dest.name}: done ({_mb(_size(dest))})")
    return model


def download_models(
    settings: Settings,
    *,
    whisper: bool = True,
    piper: bool = True,
    kokoro: bool = False,
    console: _Console | None = None,
) -> dict[str, Path]:
    """Fetch the configured models into ``settings.models_path``. Idempotent.

    Raises :class:`InsufficientDiskSpace` (before downloading anything) when something is
    missing and less than 2 GB is free.
    """
    out = console or _PrintConsole()

    def say(msg: str) -> None:
        out.print(msg)

    root = settings.models_path
    status = model_status(settings)
    wanted = {"whisper": whisper, "piper": piper, "kokoro": kokoro}
    missing = [k for k, on in wanted.items() if on and not status[k]["present"]]
    free = free_bytes(root)
    say(f"models dir: {root} (free disk: {_gb(free)})")
    if missing and free < MIN_FREE_BYTES:
        raise InsufficientDiskSpace(
            f"only {_gb(free)} free at {root}; need at least {_gb(MIN_FREE_BYTES)} to download "
            f"{', '.join(missing)}. Free some space or set models_dir to another drive."
        )
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    if whisper:
        result["whisper"] = _download_whisper(settings, say)
    if piper:
        result["piper"] = _download_piper(settings, say)
    if kokoro:
        result["kokoro"] = _download_kokoro(settings, say)
    say(f"free disk now: {_gb(free_bytes(root))}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spark.models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download the configured STT/TTS models")
    dl.add_argument("--config", default=None, help="path to config.yaml")
    dl.add_argument("--no-whisper", action="store_true", help="skip the faster-whisper model")
    dl.add_argument("--no-piper", action="store_true", help="skip the Piper voice")
    dl.add_argument("--kokoro", action="store_true", help="also fetch the Kokoro ONNX model")
    st = sub.add_parser("status", help="show which models are present")
    st.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    from spark.config import load_settings

    settings = load_settings(args.config)
    if args.cmd == "status":
        free = _gb(free_bytes(settings.models_path))
        print(f"models dir: {settings.models_path} (free disk: {free})")
        for key, info in model_status(settings).items():
            state = "present" if info["present"] else "MISSING"
            need = "required" if info["required"] else "optional"
            print(f"{key}: {state} ({need}) {info['path']}")
        return 0
    try:
        kokoro = args.kokoro or settings.tts.backend == "kokoro"
        download_models(
            settings, whisper=not args.no_whisper, piper=not args.no_piper, kokoro=kokoro
        )
    except InsufficientDiskSpace as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
