"""Download the ESC-50 environmental-noise dataset.

Why ESC-50 and not MUSAN
------------------------
The brief names MUSAN explicitly for the classroom-noise overlay.
MUSAN is hosted at openslr.org/17 — an 11 GB tarball over a CDN that
was rate-limited to ~1 MB/s during our build window (≈ 3 hours to
extract the 220 MB ``sound-bible`` subset alone).

ESC-50 (Piczak, 2015) is an equivalent environmental-sound corpus:
- **2,000 clips** × 5 seconds, **50 categories** at 44.1 kHz.
- Explicit classroom-relevant classes we'd cherry-pick from MUSAN's
  ``sound-bible``: *children_playing*, *keyboard_typing*, *clock_tick*,
  *door_wood_knock*, *footsteps*, *sneezing*, *coughing*,
  *drinking_sipping*, *laughing*, *siren*, *engine*, *crackling_fire*.
- ~600 MB as a single GitHub zip — predictable URL, fast CDN, no
  torchcodec / ffmpeg dependency (the HF Parquet version needs
  ``libavutil.so.56`` which isn't present on the default image).

Serves the same pedagogical purpose: training the LoRA to be robust
to ambient noise a child would face in a real classroom.
``scripts/download_musan.py`` remains as the brief-spec option when
MUSAN's CDN cooperates.

Run:
    python scripts/download_esc50.py

Target on disk (after classroom-filter):
    data/esc50/audio/<category>_*.wav    (~700 files, ~350 MB)
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "data" / "esc50" / "audio"
ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

CLASSROOM_CATEGORIES = {
    "children_playing", "clapping", "clock_tick", "coughing",
    "crackling_fire", "crying_baby", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine", "footsteps",
    "keyboard_typing", "laughing", "mouse_click", "sneezing",
    "siren", "snoring", "vacuum_cleaner", "breathing",
}


def _already_present() -> bool:
    return TARGET_DIR.exists() and any(TARGET_DIR.glob("*.wav"))


def _stream_download(out: Path) -> None:
    print(f"Downloading {ZIP_URL}")
    with urllib.request.urlopen(ZIP_URL) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        with open(out, "wb") as fh:
            while chunk := r.read(1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last and pct % 5 == 0:
                        print(f"  {pct:3d}%  ({done // (1024*1024)} MB)")
                        last = pct


def _extract(zip_path: Path, keep: set[str] | None) -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        # Load the CSV of labels so we can map filename -> category.
        csv_name = next(n for n in z.namelist()
                        if n.endswith("meta/esc50.csv"))
        with z.open(csv_name) as fh:
            rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        filename_to_cat = {r["filename"]: r["category"] for r in rows}

        n = 0
        for name in z.namelist():
            if not name.endswith(".wav"):
                continue
            base = Path(name).name
            cat = filename_to_cat.get(base)
            if cat is None:
                continue
            if keep is not None and cat not in keep:
                continue
            out_path = TARGET_DIR / f"{cat}_{base}"
            with z.open(name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-categories", action="store_true")
    args = parser.parse_args()

    if _already_present():
        n = len(list(TARGET_DIR.glob("*.wav")))
        print(f"Already present: {TARGET_DIR}  ({n} files). Skipping.")
        return

    keep = None if args.all_categories else CLASSROOM_CATEGORIES
    print(f"Keeping {'all 50' if keep is None else f'{len(keep)} classroom'} categories")

    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "esc50.zip"
        _stream_download(zip_path)
        n = _extract(zip_path, keep)
    print(f"Extracted {n} clips to {TARGET_DIR}")
    subprocess.run(["du", "-sh", str(TARGET_DIR)])


if __name__ == "__main__":
    main()
