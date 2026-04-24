"""Render a PNG for every ``visual`` key in the curriculum.

Strategy: the scene renderer composites discrete coloured circles on a
white canvas so that:

- For ``counting`` items (visual like ``goats_5``), there are exactly N
  well-separated blobs, so :class:`tutor.visual_count.BlobCounter` has
  ground-truth-known counts.
- For ``addition``/``subtraction`` items (``beads_2_plus_3`` /
  ``beads_8_minus_3``), we render the left-hand-side count as separated
  blobs. Enough for demo illustration; not scored by the blob counter.
- For ``number_sense`` compare items, we render the two numerals as
  big Pillow text.
- For word-problem items we render a neutral placeholder; those are
  presented by stem text, not by the visual.

Run:
    python scripts/render_scenes.py --curriculum data/T3.1_Math_Tutor/curriculum.json --out assets/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 480, 240
BG = "white"
# Colour per object so the image looks child-friendly but the blob
# counter (which thresholds a grayscale copy) still sees N dark blobs.
OBJECT_COLOUR = {
    "apples":  "#c0392b",
    "goats":   "#8e6b3a",
    "cows":    "#2c3e50",
    "drums":   "#b86c0c",
    "beads":   "#7d3c98",
    "mangoes": "#e67e22",
    "books":   "#1e7a3c",
    "pots":    "#5d4037",
}
DEFAULT_COLOUR = "#2c3e50"

_COUNT_RE      = re.compile(r"^([a-zA-Z]+)_(\d+)$")
_PLUS_RE       = re.compile(r"^([a-zA-Z]+)_(\d+)_plus_(\d+)$")
_MINUS_RE      = re.compile(r"^([a-zA-Z]+)_(\d+)_minus_(\d+)$")
_COMPARE_RE    = re.compile(r"^compare_(\d+)_(\d+)$")


def _blob(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, colour: str) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour, outline="black")


def _layout_positions(n: int, w: int = W, h: int = H, r: int = 22) -> list[tuple[int, int]]:
    """Grid positions so blobs never touch (so connected-components sees N)."""
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    xs = [int(w * (i + 1) / (cols + 1)) for i in range(cols)]
    ys = [int(h * (j + 1) / (rows + 1)) for j in range(rows)]
    out: list[tuple[int, int]] = []
    for j in range(rows):
        for i in range(cols):
            if len(out) < n:
                out.append((xs[i], ys[j]))
    return out


def _render_count_scene(n: int, obj: str, path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    colour = OBJECT_COLOUR.get(obj, DEFAULT_COLOUR)
    for (x, y) in _layout_positions(n):
        _blob(draw, x, y, r=22, colour=colour)
    img.save(path)


def _render_compare_scene(a: int, b: int, path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((W * 0.18, H * 0.15), str(a), font=font, fill="#1f3864")
    draw.text((W * 0.60, H * 0.15), str(b), font=font, fill="#1f3864")
    img.save(path)


def _render_placeholder(label: str, path: Path) -> None:
    img = Image.new("RGB", (W, H), "#f2f2f2")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((20, H // 2 - 12), label, font=font, fill="#333")
    img.save(path)


def render_one(visual: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{visual}.png"
    if (m := _COUNT_RE.match(visual)):
        obj, n = m.group(1), int(m.group(2))
        _render_count_scene(n, obj, out_path)
    elif (m := _PLUS_RE.match(visual)):
        obj, a, _b = m.group(1), int(m.group(2)), int(m.group(3))
        _render_count_scene(a, obj, out_path)
    elif (m := _MINUS_RE.match(visual)):
        obj, a, _b = m.group(1), int(m.group(2)), int(m.group(3))
        _render_count_scene(a, obj, out_path)
    elif (m := _COMPARE_RE.match(visual)):
        a, b = int(m.group(1)), int(m.group(2))
        _render_compare_scene(a, b, out_path)
    else:
        _render_placeholder(visual, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="data/T3.1_Math_Tutor/curriculum.json")
    parser.add_argument("--out", default="assets/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.curriculum, "r", encoding="utf-8") as fh:
        items = json.load(fh)

    visuals = sorted({it["visual"] for it in items if it.get("visual")})
    for v in visuals:
        render_one(v, out_dir)
    print(f"Rendered {len(visuals)} scenes into {out_dir}")


if __name__ == "__main__":
    main()
