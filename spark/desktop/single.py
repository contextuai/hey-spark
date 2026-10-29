"""One running window per user (spec §10 Phase 4, "Single instance").

Windows uses a named mutex, which the OS frees if the process dies. Elsewhere an exclusive
``flock`` on a file in the temp folder does the same job. A second launch brings the running
window to the front instead of opening a second microphone.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    def __init__(
        self, name: str, *, window_title: str | None = None, lock_dir: Path | None = None
    ) -> None:
        self.name = name  # the lock/mutex id (the installer's AppMutex matches it)
        self.window_title = window_title or name
        self._lock_dir = lock_dir or Path(tempfile.gettempdir())
        self._handle: Any = None

    def acquire(self) -> bool:
        if sys.platform == "win32":
            return self._acquire_mutex()
        return self._acquire_flock()

    def _acquire_mutex(self) -> bool:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        handle = kernel32.CreateMutexW(None, False, f"Local\\{self.name}-desktop-single-instance")
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def _acquire_flock(self) -> bool:
        import fcntl  # type: ignore[import-not-found, unused-ignore]

        path = self._lock_dir / f"{self.name.lower()}-desktop.lock"
        fh = open(path, "w")  # noqa: SIM115 - held for the process lifetime
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined, unused-ignore]
        except OSError:
            fh.close()
            return False
        fh.write(str(os.getpid()))
        fh.flush()
        self._handle = fh
        return True

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        if sys.platform == "win32":
            import ctypes

            ctypes.WinDLL("kernel32").CloseHandle(handle)
        else:
            with contextlib.suppress(OSError):
                handle.close()

    def focus_existing(self) -> bool:
        """Bring the running window (titled ``window_title``) to the front. Best effort."""
        if sys.platform == "darwin":
            return self._focus_macos()
        if sys.platform != "win32":
            return False
        import ctypes

        user32 = ctypes.WinDLL("user32")
        hwnd = user32.FindWindowW(None, self.window_title)
        if not hwnd:
            return False
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        return bool(user32.SetForegroundWindow(hwnd))

    def _focus_macos(self) -> bool:
        # Finder already does this for a second double-click; this covers a launch from a shell.
        import subprocess

        from spark.desktop.paths import BUNDLE_ID, is_frozen

        if not is_frozen():
            return False
        try:
            out = subprocess.run(
                ["osascript", "-e", f'tell application id "{BUNDLE_ID}" to activate'],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return out.returncode == 0
