"""THIRD_PARTY_NOTICES.md must exist, name every runtime dependency, and ship with the app."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_deps() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return [re.match(r"[A-Za-z0-9._-]+", spec).group(0) for spec in project["dependencies"]]


def test_notices_file_exists() -> None:
    assert NOTICES.is_file()


def test_notices_mention_every_runtime_dependency() -> None:
    # first cell of every markdown table row, so "mcp" can't match by accident inside prose
    listed = {
        _normalize(line.split("|")[1].strip().strip("`"))
        for line in NOTICES.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    }
    deps = _runtime_deps()
    assert deps, "no [project].dependencies found"
    missing = [d for d in deps if _normalize(d) not in listed]
    assert not missing, (
        f"THIRD_PARTY_NOTICES.md is stale, missing {missing}. "
        "Re-run: uv run python scripts/third_party_notices.py"
    )


def test_licenses_are_shipped_with_the_frozen_app() -> None:
    spec = (ROOT / "packaging" / "hey_spark.spec").read_text(encoding="utf-8")
    assert '"LICENSE"' in spec
    assert '"THIRD_PARTY_NOTICES.md"' in spec
    assert '"wakewords" / "LICENSE"' in spec
    assert (ROOT / "assets" / "wakewords" / "LICENSE").is_file()


def test_gpl_text_is_shipped_with_the_frozen_app() -> None:
    gpl = ROOT / "LICENSES" / "GPL-3.0.txt"
    text = gpl.read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in text and "Version 3, 29 June 2007" in text
    assert "END OF TERMS AND CONDITIONS" in text  # the full text, not a stub
    spec = (ROOT / "packaging" / "hey_spark.spec").read_text(encoding="utf-8")
    assert '"LICENSES" / "GPL-3.0.txt"' in spec


def test_lgpl_text_is_shipped_with_the_frozen_app() -> None:
    lgpl = ROOT / "LICENSES" / "LGPL-2.1.txt"
    text = lgpl.read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text and "Version 2.1, February 1999" in text
    assert "END OF TERMS AND CONDITIONS" in text
    spec = (ROOT / "packaging" / "hey_spark.spec").read_text(encoding="utf-8")
    assert '"LICENSES" / "LGPL-2.1.txt"' in spec


def test_notices_point_to_gpl_sources() -> None:
    notices = NOTICES.read_text(encoding="utf-8")
    assert "Source code for GPL/LGPL components" in notices
    for needle in (
        "LICENSES/GPL-3.0.txt",
        "https://github.com/OHF-Voice/piper1-gpl/tree/v1.8.0",
        "https://github.com/espeak-ng/espeak-ng",
        "https://github.com/PyAV-Org/PyAV",
        "https://github.com/contextuai/hey-spark",
    ):
        assert needle in notices, needle


def test_frozen_app_does_not_bundle_the_claude_cli() -> None:
    spec = (ROOT / "packaging" / "hey_spark.spec").read_text(encoding="utf-8")
    assert 'collect_data_files("claude_agent_sdk", excludes=["_bundled"])' in spec
    assert 'collect_data_files("claude_agent_sdk")' not in spec
