"""Small file helpers.

:func:`write_text_atomic` writes through a temp file and ``os.replace`` so a crash never leaves
half a file. On Windows that rename fails now and then with "Access is denied" (WinError 5)
while antivirus or the search indexer briefly holds the target, which was seen in the tests on
some Windows machines. So it retries for a moment and, as a last resort, writes the file directly.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

__all__ = ["write_text_atomic"]

RETRIES = 8
RETRY_DELAY_S = 0.03


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8, LF line endings) to ``path``, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt + 1 < RETRIES:
                time.sleep(RETRY_DELAY_S * (attempt + 1))
    # Still held: write in place (not atomic, but the content is not lost).
    path.write_text(text, encoding="utf-8", newline="\n")
    with contextlib.suppress(OSError):
        tmp.unlink()
