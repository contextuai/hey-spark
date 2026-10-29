"""Print a markdown table of the third-party Python packages Hey Spark ships.

    uv run python scripts/third_party_notices.py

Walks the runtime dependency closure of the installed ``hey-spark`` distribution (its base
dependencies plus the ``desktop``, ``oww`` and ``kokoro`` extras, the ones the frozen desktop app
uses) through ``importlib.metadata``. Environment markers are evaluated for the current
interpreter, so run it on Windows to match the Windows build. Dev/build-only tools are excluded.
Output is sorted, so the same environment always gives the same table. Paste it into
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT_PROJECT = "hey-spark"
EXTRAS = ("desktop", "oww", "kokoro")
EXCLUDE = {"pytest", "pytest-asyncio", "ruff", "pyinstaller", "pyinstaller-hooks-contrib"}

COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|CDDL|Mozilla|General Public)", re.IGNORECASE)
NON_COMMERCIAL = re.compile(r"\b(NC|non-?commercial|proprietary)\b", re.IGNORECASE)


def _wanted(req: Requirement, extras: tuple[str, ...]) -> bool:
    if req.marker is None:
        return True
    env = default_environment()
    return any(req.marker.evaluate({**env, "extra": e}) for e in (*extras, ""))


def closure() -> list[md.Distribution]:
    """Installed distributions reachable from ``hey-spark[desktop,oww]``, minus excluded tools."""
    root = md.distribution(ROOT_PROJECT)
    seen: dict[str, md.Distribution] = {}
    queue: list[tuple[md.Distribution, tuple[str, ...]]] = [(root, EXTRAS)]
    while queue:
        dist, extras = queue.pop()
        for raw in dist.requires or []:
            req = Requirement(raw)
            name = canonicalize_name(req.name)
            if name in EXCLUDE or not _wanted(req, extras):
                continue
            try:
                dep = md.distribution(req.name)
            except md.PackageNotFoundError:
                print(f"warning: {req.name} is required but not installed", file=sys.stderr)
                continue
            key = f"{name}[{','.join(sorted(req.extras))}]"
            if key in seen:
                continue
            seen[key] = dep
            queue.append((dep, tuple(sorted(req.extras))))
    by_name = {canonicalize_name(d.metadata["Name"]): d for d in seen.values()}
    return [by_name[n] for n in sorted(by_name)]


def license_of(dist: md.Distribution) -> str:
    """License expression, else the License classifiers, else a short License field."""
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::") and c.strip() != "License :: OSI Approved"
    ]
    if classifiers:
        return "; ".join(sorted(set(classifiers)))
    field = (meta.get("License") or "").strip()
    if field and "\n" not in field and len(field) <= 80:
        return field
    if field:
        return field.splitlines()[0].strip()[:80] + " (see LICENSE)"
    for name in sorted(meta.get_all("License-File") or []):
        text = dist.read_text(f"licenses/{name}") or dist.read_text(name) or ""
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if first:
            return f"{first[:60]} (from {name})"
    return "UNKNOWN"


def home_page(dist: md.Distribution) -> str:
    meta = dist.metadata
    if meta.get("Home-page"):
        return meta["Home-page"].strip()
    urls = {}
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for label in ("homepage", "home", "source", "repository", "source code", "github"):
        if label in urls:
            return urls[label]
    return next(iter(urls.values()), "")


def flag(lic: str) -> str:
    if lic == "UNKNOWN":
        return "unknown"
    if COPYLEFT.search(lic):
        return "copyleft"
    if NON_COMMERCIAL.search(lic):
        return "non-commercial"
    return ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # same bytes on every OS
    rows = [(d.metadata["Name"], d.version, license_of(d), home_page(d)) for d in closure()]
    print("| Package | Version | License | Home page |")
    print("|---|---|---|---|")
    for name, version, lic, url in rows:
        cell = lic.replace("|", "/")
        print(f"| {name} | {version} | {cell} | {url} |")
    flagged = [(n, lic, flag(lic)) for n, _, lic, _ in rows if flag(lic)]
    if flagged:
        print()
        print("Flagged by the generator (review by hand):")
        print()
        for name, lic, why in flagged:
            print(f"- {name}: {lic} ({why})")


if __name__ == "__main__":
    main()
