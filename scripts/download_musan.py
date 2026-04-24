"""Fetch the MUSAN ``noise/sound-bible`` subset from OpenSLR.

We only extract the ``sound-bible`` folder (~220 MB, 93 short clips of
classroom-style environmental noise: door knocks, coughs, chatter,
tableware, sirens, etc.). That's the light-weight corner of MUSAN most
appropriate for overlaying onto short child-voice utterances at
~12 dB SNR.

Brief reference:
    https://www.openslr.org/17/

Target on disk:
    data/musan/noise/sound-bible/*.wav   (~220 MB, 93 files)

Idempotent: if the extracted directory already has content, we skip
the download entirely. Safe to re-run.

Run:
    python scripts/download_musan.py
    python scripts/download_musan.py --full          # grab all of noise/ (~1 GB)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


MUSAN_TARBALL_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "data" / "musan"


def _already_present(subset: str) -> bool:
    path = TARGET_DIR / "noise" / subset
    return path.exists() and any(path.glob("**/*.wav"))


def _download(tmp_path: Path) -> None:
    """Stream the 11 GB MUSAN tar.gz with a visible progress line."""
    print(f"Downloading MUSAN tarball from {MUSAN_TARBALL_URL}")
    print("(This is ~11 GB. Streaming to a temp file; only the noise/"
          "sound-bible subset will be kept on disk.)")
    with urllib.request.urlopen(MUSAN_TARBALL_URL) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        with open(tmp_path, "wb") as fh:
            while chunk := resp.read(1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 2 == 0:
                        mb = downloaded // (1024 * 1024)
                        print(f"  {pct:3d}%  ({mb} MB)")
                        last_pct = pct
    print(f"Downloaded {tmp_path.stat().st_size // (1024 * 1024)} MB")


def _extract_subset(tar_path: Path, subset: str) -> int:
    """Extract only ``musan/noise/<subset>/*`` into TARGET_DIR."""
    prefix = f"musan/noise/{subset}/"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    print(f"Extracting '{prefix}' into {TARGET_DIR}")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.name.startswith(prefix):
                continue
            # Re-root paths so we end up with
            # data/musan/noise/<subset>/* rather than musan/musan/...
            member.name = member.name[len("musan/"):]
            tar.extract(member, path=TARGET_DIR)
            if member.isfile():
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Extract all of noise/ (~1 GB) not just sound-bible.")
    parser.add_argument("--keep-tarball", action="store_true")
    args = parser.parse_args()

    subset = "" if args.full else "sound-bible"

    if not args.full and _already_present(subset):
        print(f"Already present at {TARGET_DIR / 'noise' / subset}. Skipping.")
        return

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "musan.tar.gz"
        _download(tar_path)
        if args.full:
            n_sound_bible = _extract_subset(tar_path, "sound-bible")
            n_free_sound = _extract_subset(tar_path, "free-sound")
            print(f"Extracted {n_sound_bible + n_free_sound} noise files "
                  "(sound-bible + free-sound)")
        else:
            n = _extract_subset(tar_path, "sound-bible")
            print(f"Extracted {n} sound-bible noise files")
        if args.keep_tarball:
            shutil.copy(tar_path, REPO_ROOT / "data" / "musan.tar.gz")
    print(f"Done. du -sh {TARGET_DIR}:")
    subprocess.run(["du", "-sh", str(TARGET_DIR)])


if __name__ == "__main__":
    main()
