"""Ollama helpers for the Brain window (P5.7f): is it installed and running, which models it has,
free disk space, and a model download the user starts and can cancel.

Ollama's own API (not the OpenAI-compatible ``/v1`` one) lives at the server root:

* ``GET /api/tags`` -> ``{"models": [{"name": "qwen2.5:7b", "size": 4683087332, ...}]}``
* ``POST /api/pull`` with ``{"model": name, "stream": true}`` -> newline-delimited JSON:
  ``{"status": "pulling manifest"}``, then per layer ``{"status": "pulling <digest>", "digest",
  "total", "completed"}``, ``{"status": "verifying sha256 digest"}``, ``{"status": "writing
  manifest"}`` and ``{"status": "success"}``; a failure is ``{"error": "..."}``. Closing the
  connection stops the pull; Ollama keeps the parts and resumes the next time.

Nothing here downloads on its own: :class:`Puller` only starts when the page calls it, which it
does only on the user's click. ``httpx`` is imported inside the functions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "DOWNLOAD_URL",
    "RECOMMENDED",
    "PullProgress",
    "Puller",
    "api_root",
    "find_ollama",
    "free_bytes",
    "models_dir",
    "status",
]

DOWNLOAD_URL = "https://ollama.com/download"
GB = 1_000_000_000

#: Tool-capable models a basic user can start with; sizes are Ollama's download sizes (approx).
RECOMMENDED: tuple[dict[str, Any], ...] = (
    {
        "name": "qwen2.5:7b",
        "label": "Qwen 2.5 (7B)",
        "size_gb": 4.7,
        "note": "Recommended. Good with tools; needs about 8 GB of memory.",
    },
    {
        "name": "llama3.1:8b",
        "label": "Llama 3.1 (8B)",
        "size_gb": 4.9,
        "note": "Also good with tools; needs about 8 GB of memory.",
    },
    {
        "name": "qwen2.5:3b",
        "label": "Qwen 2.5 (3B)",
        "size_gb": 1.9,
        "note": "Smaller and quicker on a slow computer, but less reliable with tools.",
    },
)


def api_root(base_url: str | None) -> str:
    """``http://localhost:11434/v1`` -> ``http://localhost:11434`` (Ollama's own API)."""
    root = (base_url or "http://localhost:11434").strip().rstrip("/")
    if root.lower().endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def find_ollama(environ: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if environ is None else environ
    found = shutil.which("ollama")
    if found:
        return Path(found)
    local = env.get("LOCALAPPDATA")
    if local:
        exe = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if exe.is_file():
            return exe
    return None


def models_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Where Ollama keeps models: ``OLLAMA_MODELS`` or ``~/.ollama/models``."""
    env = os.environ if environ is None else environ
    custom = (env.get("OLLAMA_MODELS") or "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".ollama" / "models"


def free_bytes(path: Path) -> int | None:
    """Free space on the drive holding ``path`` (or its nearest existing parent)."""
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            return None
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def status(
    base_url: str | None,
    *,
    transport: Any | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    """What the Ollama card shows. Never raises."""
    import httpx

    root = api_root(base_url)
    running = False
    models: list[dict[str, Any]] = []
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            response = client.get(f"{root}/api/tags")
        if response.status_code < 400:
            running = True
            data = response.json()
            for item in data.get("models", []) if isinstance(data, dict) else []:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    size = item.get("size")
                    gb = round(size / GB, 1) if isinstance(size, int | float) else None
                    models.append({"name": item["name"], "size_gb": gb})
    except (httpx.HTTPError, ValueError) as exc:
        log.debug("ollama: %s/api/tags: %s", root, exc)
    folder = models_dir(environ)
    free = free_bytes(folder)
    free_gb = round(free / GB, 1) if free is not None else None
    return {
        "installed": running or find_ollama(environ) is not None,
        "running": running,
        "models": models,
        "download_url": DOWNLOAD_URL,
        "disk": {"path": str(folder), "free_gb": free_gb},
        "recommended": [
            {
                **m,
                "installed": any(x["name"] == m["name"] for x in models),
                # Room for the download plus as much again (unpacking, the next model, Windows).
                "low_space": free_gb is not None and free_gb < 2 * m["size_gb"],
            }
            for m in RECOMMENDED
        ],
    }


class PullProgress:
    """Folds ``/api/pull`` lines into one progress state."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.state = "running"  # running | done | error | cancelled
        self.status = "Starting…"
        self.error: str | None = None
        self._layers: dict[str, tuple[int, int]] = {}  # digest -> (completed, total)

    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            item = json.loads(line)
        except ValueError:
            return
        if isinstance(item, dict):
            self.feed(item)

    def feed(self, item: Mapping[str, Any]) -> None:
        if item.get("error"):
            self.state = "error"
            self.error = f"Ollama couldn't download {self.model}: {item['error']}"
            return
        status = str(item.get("status") or "")
        digest = item.get("digest")
        total = item.get("total")
        if isinstance(digest, str) and isinstance(total, int) and total > 0:
            completed = item.get("completed")
            done = completed if isinstance(completed, int) else 0
            self._layers[digest] = (min(done, total), total)
        if status == "success":
            self.state = "done"
            self.status = "Downloaded"
        elif status.startswith("pulling") and digest:
            self.status = "Downloading…"
        elif status:
            self.status = status[:1].upper() + status[1:]

    @property
    def completed(self) -> int:
        return sum(c for c, _ in self._layers.values())

    @property
    def total(self) -> int:
        return sum(t for _, t in self._layers.values())

    def snapshot(self) -> dict[str, Any]:
        total = self.total
        percent = 100.0 if self.state == "done" else (100 * self.completed / total if total else 0)
        return {
            "model": self.model,
            "state": self.state,
            "status": self.status,
            "completed_gb": round(self.completed / GB, 2),
            "total_gb": round(total / GB, 2),
            "percent": round(percent, 1),
            "error": self.error,
        }


class Puller:
    """One model download at a time, on a background thread. The page polls :meth:`progress`."""

    def __init__(self, *, transport: Any | None = None) -> None:
        self._transport = transport
        self._lock = threading.Lock()
        self._progress: PullProgress | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._response: Any = None

    @property
    def busy(self) -> bool:
        return self._progress is not None and self._progress.state == "running"

    def start(self, base_url: str | None, model: str, *, wait: bool = False) -> bool:
        """Start downloading ``model``. False if a download is already running."""
        with self._lock:
            if self.busy:
                return False
            self._cancel.clear()
            self._progress = PullProgress(model)
            self._thread = threading.Thread(
                target=self._run, args=(api_root(base_url), model), name="ollama-pull", daemon=True
            )
            self._thread.start()
        if wait:  # tests
            self._thread.join(30)
        return True

    def progress(self) -> dict[str, Any]:
        if self._progress is None:
            return {"state": "idle"}
        return self._progress.snapshot()

    def cancel(self) -> None:
        self._cancel.set()
        response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                log.debug("ollama: closing the pull stream failed", exc_info=True)

    def _run(self, root: str, model: str) -> None:
        import httpx

        progress = self._progress
        assert progress is not None
        log.info("ollama: downloading %s", model)
        try:
            timeout = httpx.Timeout(30.0, read=600.0)  # verifying a big file can be quiet
            with (
                httpx.Client(transport=self._transport, timeout=timeout) as client,
                client.stream(
                    "POST", f"{root}/api/pull", json={"model": model, "stream": True}
                ) as response,
            ):
                self._response = response
                if response.status_code >= 400:
                    progress.feed_line(response.read().decode("utf-8", "replace"))
                    if progress.state != "error":
                        progress.state = "error"
                        progress.error = f"Ollama answered HTTP {response.status_code}."
                    return
                for line in response.iter_lines():
                    if self._cancel.is_set():
                        break
                    progress.feed_line(line)
                    if progress.state != "running":
                        break
        except httpx.HTTPError as exc:
            if not self._cancel.is_set():
                progress.state = "error"
                progress.error = (
                    "Lost the connection to Ollama during the download."
                    if not isinstance(exc, httpx.ConnectError)
                    else "Ollama isn't running. Start it from the Start menu."
                )
        finally:
            self._response = None
            if self._cancel.is_set() and progress.state == "running":
                progress.state = "cancelled"
                progress.status = "Cancelled"
            elif progress.state == "running":  # the stream ended without "success"
                progress.state = "error"
                progress.error = progress.error or "The download stopped before it finished."
            log.info("ollama: download of %s: %s", model, progress.state)
