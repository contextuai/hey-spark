"""Build every Hey Spark app icon from ``packaging/icon-source.png``.

    uv run --with pillow python packaging/make_icon.py

``icon-source.png`` is the master artwork: a 1024 px charcoal squircle with a glowing orange core
inside waveform rings (generated with Google Gemini, then cropped to the tile and masked so
everything outside it is transparent). Pillow is only needed here, not by the app, hence
``--with pillow``.

Above 48 px, the icons are the master scaled down. Smaller than that the thin rings turn to
mush, so 16-48 px get a hand-drawn simplification of the same design (tile, orb, one bold ring)
that stays crisp in a taskbar or title bar.

Writes:

- ``spark/desktop/ui/icon.png`` (256 px) and ``spark/desktop/ui/icon.ico``: the window icon
  (WinForms only takes .ico) and the README logo
- ``packaging/windows/hey_spark.ico``: the exe and installer
- ``packaging/macos/hey_spark.icns``: the .app, inside the macOS icon-grid margin
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packaging" / "icon-source.png"

FULL = 0.94  # Windows and the window icon: the tile nearly fills the canvas
MAC = 0.805  # macOS icon grid: an 824 px tile on a 1024 px canvas
SMALL_MAX = 48  # at or below this, draw the simplified icon

SQUIRCLE_N = 5.4  # matches the master's corners
TILE = np.array([21, 24, 31], dtype=np.float32)  # #15181F
ORB_EDGE = np.array([255, 71, 0], dtype=np.float32)  # #FF4700, ContextuAI orange
ORB_MID = np.array([255, 138, 61], dtype=np.float32)  # #FF8A3D
ORB_CORE = np.array([255, 238, 226], dtype=np.float32)
RING = np.array([255, 106, 42], dtype=np.float32)


def _master() -> Image.Image:
    return Image.open(SOURCE).convert("RGBA")


def from_master(master: Image.Image, size: int, fill: float) -> Image.Image:
    """The master scaled so its tile spans ``fill`` of a transparent ``size`` canvas."""
    inner = round(size * fill)
    tile = master.resize((inner, inner), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    at = (size - inner) // 2
    canvas.alpha_composite(tile, (at, at))
    return canvas


def small(size: int, fill: float) -> Image.Image:
    """Hand-drawn simplification for tiny sizes: tile, glowing orb, one ring."""
    ss = 8  # heavy supersampling: these are only a few pixels across
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1  # -1..1
    x, y = x / fill, y / fill  # tile spans -1..1
    r = np.hypot(x, y)

    tile = (np.abs(x) ** SQUIRCLE_N + np.abs(y) ** SQUIRCLE_N <= 1.0).astype(np.float32)
    rgb = np.broadcast_to(TILE, (n, n, 3)).copy()

    ring = (np.abs(r - 0.70) <= 0.075).astype(np.float32)[..., None]
    rgb = rgb * (1 - ring) + RING * ring

    orb_r = 0.46
    t = np.clip(r / orb_r, 0, 1)[..., None]
    orb_rgb = np.where(t < 0.45, ORB_CORE + (ORB_MID - ORB_CORE) * (t / 0.45), ORB_MID)
    orb_rgb = np.where(t >= 0.45, ORB_MID + (ORB_EDGE - ORB_MID) * ((t - 0.45) / 0.55), orb_rgb)
    orb = (r <= orb_r).astype(np.float32)[..., None]
    rgb = rgb * (1 - orb) + orb_rgb * orb

    img = np.dstack([rgb, tile * 255.0]).reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    return Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8), "RGBA")


def icon(master: Image.Image, size: int, fill: float) -> Image.Image:
    return small(size, fill) if size <= SMALL_MAX else from_master(master, size, fill)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def ico_bytes(images: list[Image.Image]) -> bytes:
    """An ICO holding one PNG per size (Vista+ reads PNG entries)."""
    datas = [png_bytes(im) for im in images]
    out = struct.pack("<HHH", 0, 1, len(datas))
    offset = 6 + 16 * len(datas)
    for im, data in zip(images, datas, strict=True):
        dim = 0 if im.width >= 256 else im.width
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return out + b"".join(datas)


# ICNS element types that hold PNG data, by pixel size (Retina variants reuse the same bitmap).
ICNS_TYPES = {
    32: [b"ic11"],
    64: [b"ic12"],
    128: [b"ic07"],
    256: [b"ic08", b"ic13"],
    512: [b"ic09", b"ic14"],
    1024: [b"ic10"],
}


def icns_bytes(master: Image.Image) -> bytes:
    body = b""
    for size, kinds in ICNS_TYPES.items():
        data = png_bytes(icon(master, size, MAC))
        for kind in kinds:
            body += kind + struct.pack(">I", 8 + len(data)) + data
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def main() -> None:
    master = _master()
    png = ROOT / "spark" / "desktop" / "ui" / "icon.png"
    ico = ROOT / "packaging" / "windows" / "hey_spark.ico"
    icns = ROOT / "packaging" / "macos" / "hey_spark.icns"
    for path in (ico, icns):
        path.parent.mkdir(parents=True, exist_ok=True)

    png.write_bytes(png_bytes(icon(master, 256, FULL)))
    ico_data = ico_bytes([icon(master, s, FULL) for s in (16, 24, 32, 48, 64, 128, 256)])
    ico.write_bytes(ico_data)
    png.with_suffix(".ico").write_bytes(ico_data)
    icns.write_bytes(icns_bytes(master))
    print(f"wrote {png}, {png.with_suffix('.ico')}, {ico} and {icns}")


if __name__ == "__main__":
    main()
