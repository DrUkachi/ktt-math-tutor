"""BlobCounter must match the ground-truth count baked into the filename.

The scene renderer at scripts/render_scenes.py lays out N non-touching
coloured blobs for filenames of the form ``<object>_<N>.png``. This test
verifies that the runtime counter agrees with the renderer across every
counting-skill asset shipped with the repo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tutor.visual_count import BlobCounter


ASSETS = Path(__file__).parent.parent / "assets"
_COUNT_RE = re.compile(r"^([a-zA-Z]+)_(\d+)\.png$")


def _counting_assets() -> list[tuple[Path, str, int]]:
    out = []
    for p in sorted(ASSETS.glob("*.png")):
        m = _COUNT_RE.match(p.name)
        if not m:
            continue
        out.append((p, m.group(1), int(m.group(2))))
    return out


@pytest.mark.parametrize("path,label,expected", _counting_assets())
def test_blob_count_matches_ground_truth(path: Path, label: str, expected: int) -> None:
    got = BlobCounter().count(path, label)
    assert got == expected, f"{path.name}: expected {expected}, got {got}"


def test_at_least_one_counting_asset_exists() -> None:
    assert len(_counting_assets()) >= 10, (
        "Run `python scripts/render_scenes.py` before running the test suite."
    )
